"""Client for the CBR / datakaveri data-access platform.

Flow:
    1. Authenticate against Keycloak (username/password) -> OAuth access token.
       Or skip this step by passing an API key (``sk_...``), which the server
       accepts anywhere an access token is accepted.
    2. Call the connection-info endpoint with that token -> DB connection
       metadata (host, port, database, username).
    3. Connect to PostgreSQL via SQLAlchemy, using the access token as the
       database password.
    4. Run SQL with plain table names against your **approved access request**
       — the client resolves them against it automatically (``set_access()``
       picks one when you have several approved requests).

Example:
    >>> from cbr_data_access import DataAccessClient
    >>> with DataAccessClient(username="u@example.org", password="...") as client:
    ...     df = client.query("SELECT count(*) FROM person")
"""

from __future__ import annotations

import logging
import os
import pathlib
import re
import time
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from typing import Any

import jwt
import pandas as pd
import requests
from sqlalchemy import URL, Connection, Engine, create_engine, text
from sqlalchemy.dialects import postgresql
from sqlalchemy.exc import SQLAlchemyError

from . import config
from ._sql import qualify_statement
from .access_requests import AccessRequest, RequestList, RequestTable
from .exceptions import (
    AuthenticationError,
    ConnectionInfoError,
    DataAccessError,
    DBConnectionError,
    QueryError,
    RequestsError,
)

logger = logging.getLogger(__name__)

_EXP_SKEW_SECONDS = 30  # treat the token as expired this many seconds early

# Quotes SQL identifiers (and doubles embedded quotes) for PostgreSQL. Schema and
# table names are identifiers, not values, so they can't be bind parameters —
# they must be safely quoted when interpolated into a query.
_IDENT = postgresql.dialect().identifier_preparer

# Rows pulled per batch when streaming a table to disk. Bounds peak memory: only
# this many rows are held at once regardless of how large the table is.
_CHUNK_SIZE = 50_000


def _safe_filename(name: str) -> str:
    """Turn a table name into a filesystem-safe filename stem.

    Table names are normally plain identifiers (``[a-z0-9_]``), but be defensive
    against path separators or other surprises in server-provided names.
    """
    return re.sub(r"[^A-Za-z0-9._-]", "_", os.path.basename(name)) or "table"


def _select_tables(request: AccessRequest, tables: Iterable[str] | None) -> list[RequestTable]:
    """Resolve requested table names against a request (all tables if ``None``).

    Raises:
        RequestsError: If a requested name is not granted by the request.
    """
    if tables is None:
        return list(request.tables)
    selected: list[RequestTable] = []
    for name in tables:
        table = request.get_table(name)
        if table is None:
            raise RequestsError(f"Table {name!r} is not granted by request {request.id}")
        selected.append(table)
    return selected


def _write_csv_stream(
    rows: Iterator[pd.DataFrame], path: pathlib.Path, col_names: list[str]
) -> None:
    """Stream chunks to a CSV: header on the first chunk, data appended after.

    If the table has no rows (no chunks), a header-only file is still written
    from the known column names.
    """
    wrote = False
    for chunk in rows:
        chunk.to_csv(path, mode="a", header=not wrote, index=False, encoding="utf-8")
        wrote = True
    if not wrote:
        pd.DataFrame(columns=col_names).to_csv(path, index=False, encoding="utf-8")


def _write_parquet_stream(
    rows: Iterator[pd.DataFrame], path: pathlib.Path, col_names: list[str]
) -> None:
    """Stream chunks to a Parquet file, one row group per chunk.

    The first chunk fixes the schema; later chunks are cast to it so per-chunk
    dtype inference (e.g. int vs float-with-nulls) cannot drift. An empty table
    yields an empty Parquet file carrying just the known columns. pyarrow is
    imported lazily so CSV-only use never pays its import cost.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    writer: pq.ParquetWriter | None = None
    try:
        for chunk in rows:
            table = pa.Table.from_pandas(chunk, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(path, table.schema)
                writer.write_table(table)
            else:
                writer.write_table(table.cast(writer.schema))
        if writer is None:
            pd.DataFrame(columns=col_names).to_parquet(path, index=False)
    finally:
        if writer is not None:
            writer.close()


def decode_token(token: str) -> dict[str, Any]:
    """Decode a JWT's claims WITHOUT verifying its signature.

    Useful for introspection and for reading ``exp``. This does not validate
    the token's signature, issuer, or audience.
    """
    return jwt.decode(token, options={"verify_signature": False})


class DataAccessClient:
    """Encapsulates auth, access-request retrieval, and database access.

    Authentication is lazy: constructing the client performs no network I/O.
    The access token and SQLAlchemy engine are cached on the instance and
    reused; an expired token triggers automatic re-authentication.

    Usage:
        :meth:`query` is the primary entry point and handles everything for you
        (auth, connecting, disconnecting, and resolving plain table names
        against your approved access request) — you do not need to call
        :meth:`connect`. With several approved requests, pick one first via
        :meth:`set_access`. :meth:`list_requests` lists your access requests
        and what each one grants. :meth:`row_counts` reports how many rows each
        table has, and :meth:`download_request` streams a request's tables to
        Parquet (or CSV) files. :meth:`connect` is an optional context manager
        for power users who want a raw SQLAlchemy ``Connection``.

    Args:
        username: Keycloak username. Omit when passing ``api_key``.
        password: Keycloak password. Omit when passing ``api_key``.
        api_key: An opaque API key (``sk_...``) issued by the OMOP auth server,
            used instead of a username/password. The key is presented as the
            bearer token on HTTP endpoints and as the password in the Postgres
            proxy handshake; the server resolves it to the same identity a
            Keycloak token would give. Keys do not expire mid-run, so no
            re-authentication happens.
        keycloak_url: Token endpoint (defaults to the CBR Keycloak realm).
        base_url: Base URL of the data-access service; the connection-info and
            requests endpoints hang off it.
        client_id: OAuth client id.
        request_timeout: Per-request HTTP timeout, in seconds (default 300, i.e.
            5 minutes).

    Raises:
        AuthenticationError: If neither ``api_key`` nor a username/password pair
            is given.

    Example:
        >>> with DataAccessClient(username="u@example.org", password="...") as client:
        ...     print(client.list_requests().to_frame())
        ...     df = client.query("SELECT * FROM person LIMIT 10")

        >>> with DataAccessClient(api_key="sk_...") as client:
        ...     df = client.query("SELECT * FROM person LIMIT 10")
    """

    def __init__(
        self,
        username: str | None = None,
        password: str | None = None,
        *,
        api_key: str | None = None,
        keycloak_url: str = config.DEFAULT_KEYCLOAK_URL,
        base_url: str = config.DEFAULT_BASE_URL,
        client_id: str = config.DEFAULT_CLIENT_ID,
        request_timeout: float = 300.0,
    ) -> None:
        if not api_key and not (username and password):
            raise AuthenticationError("either api_key or username and password are required")

        self._username = username
        self._password = password
        self._api_key = api_key
        self._keycloak_url = keycloak_url
        self._base_url = base_url.rstrip("/")
        self._connection_info_url = f"{self._base_url}{config.CONNECTION_INFO_PATH}"
        self._requests_url = f"{self._base_url}{config.REQUESTS_PATH}"
        self._client_id = client_id
        self._timeout = request_timeout

        # An API key is presented exactly where a Keycloak access token would be
        # (bearer header, proxy password), so it seeds the token slot directly.
        # It carries no `exp`, so it never triggers re-authentication.
        self._access_token: str | None = api_key
        self._token_exp: float | None = None
        self._connection_info: dict[str, Any] | None = None
        self._engine: Engine | None = None
        self._engine_token: str | None = None  # token the cached engine was built with
        self._default_request: AccessRequest | None = None  # pinned by set_access()

    # ------------------------------------------------------------------ #
    # context manager
    # ------------------------------------------------------------------ #

    def __enter__(self) -> DataAccessClient:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def close(self) -> None:
        """Dispose the cached SQLAlchemy engine and its connection pool."""
        if self._engine is not None:
            self._engine.dispose()
            self._engine = None
            self._engine_token = None

    # ------------------------------------------------------------------ #
    # authentication
    # ------------------------------------------------------------------ #

    def login(self) -> str:
        """Authenticate against Keycloak and cache the access token.

        In API-key mode there is nothing to log in to: the key is already the
        credential, so this returns it unchanged.

        Returns:
            The access token, or the API key when the client was built with one.

        Raises:
            AuthenticationError: On transport failure, a non-200 response, or a
                response without an ``access_token``.
        """
        if self._api_key:
            return self._api_key

        data = {
            "grant_type": "password",
            "client_id": self._client_id,
            "username": self._username,
            "password": self._password,
        }
        headers = {"Content-Type": "application/x-www-form-urlencoded"}
        try:
            response = requests.post(
                self._keycloak_url, data=data, headers=headers, timeout=self._timeout
            )
        except requests.RequestException as exc:
            raise AuthenticationError(f"Keycloak request failed: {exc}") from exc

        if response.status_code != 200:
            raise AuthenticationError(response.text, status_code=response.status_code)

        token = response.json().get("access_token")
        if not token:
            raise AuthenticationError("No access_token in Keycloak response")

        self._access_token = token
        self._token_exp = self._read_exp(token)
        logger.debug("Authenticated as %s; token exp=%s", self._username, self._token_exp)
        return token

    @staticmethod
    def _read_exp(token: str) -> float | None:
        """Read the ``exp`` claim, returning None if absent or unparseable."""
        try:
            exp = decode_token(token).get("exp")
            return float(exp) if exp is not None else None
        except Exception:
            return None

    def _token_expired(self) -> bool:
        if self._token_exp is None:
            return False
        return time.time() >= (self._token_exp - _EXP_SKEW_SECONDS)

    def _ensure_token(self) -> str:
        """Return a valid token, (re)authenticating if missing or expired."""
        if self._access_token is None or self._token_expired():
            self.login()
        assert self._access_token is not None  # narrowed for type-checkers
        return self._access_token

    @property
    def token_claims(self) -> dict[str, Any]:
        """Decoded claims of the current token (authenticates if needed).

        Raises:
            AuthenticationError: In API-key mode. An API key is opaque, so it
                carries no claims to decode — ask the server who you are with
                :attr:`identity` or :meth:`list_requests` instead.
        """
        if self._api_key:
            raise AuthenticationError("an API key is opaque and carries no claims; use identity")
        return decode_token(self._ensure_token())

    @property
    def identity(self) -> str | None:
        """The username the server resolves this credential to.

        Uses the locally known username when the client was built with one;
        otherwise asks the server (API keys are opaque, so the identity behind
        one is only knowable server-side).
        """
        if self._username:
            return self._username
        return self._ensure_connection_info().get("username")

    # ------------------------------------------------------------------ #
    # connection info
    # ------------------------------------------------------------------ #

    def _fetch_connection_info(self) -> dict[str, Any]:
        token = self._ensure_token()
        headers = {"Authorization": f"Bearer {token}"}
        try:
            response = requests.get(
                self._connection_info_url, headers=headers, timeout=self._timeout
            )
        except requests.RequestException as exc:
            raise ConnectionInfoError(f"Connection-info request failed: {exc}") from exc

        if response.status_code != 200:
            raise ConnectionInfoError(response.text, status_code=response.status_code)

        info = response.json()
        if not info:
            raise ConnectionInfoError("Empty connection-info response")
        return info

    def _ensure_connection_info(self) -> dict[str, Any]:
        if self._connection_info is None:
            self._connection_info = self._fetch_connection_info()
        return self._connection_info

    # ------------------------------------------------------------------ #
    # access requests / catalog
    # ------------------------------------------------------------------ #

    def list_requests(self) -> RequestList:
        """List the caller's access requests and what each one makes queryable.

        Calls the requests endpoint with the current access token and returns
        the catalog as structured objects. The result is a sequence of
        :class:`~cbr_data_access.access_requests.AccessRequest` (iterate or
        index it) that also carries ``username``. Approved requests expose
        their queryable tables (each with its ready-to-use
        ``req_<request_id>.<table>`` relation); pending/rejected requests are
        listed for visibility but expose none.

        Returns:
            A :class:`~cbr_data_access.access_requests.RequestList`.

        Raises:
            AuthenticationError: If authentication fails.
            RequestsError: On transport failure or a non-200 response.

        Example:
            >>> reqs = client.list_requests()
            >>> [r.id for r in reqs.approved]
            ['dc43d8d9-6f6c-4152-b5f1-0f9668ba01e4']
            >>> reqs.to_frame()  # one row per (request, table), handy for browsing
        """
        token = self._ensure_token()
        headers = {"Authorization": f"Bearer {token}"}
        try:
            response = requests.get(self._requests_url, headers=headers, timeout=self._timeout)
        except requests.RequestException as exc:
            raise RequestsError(f"Requests catalog request failed: {exc}") from exc

        if response.status_code != 200:
            raise RequestsError(response.text, status_code=response.status_code)

        return RequestList.from_dict(response.json())

    def _resolve_request(self, catalog: RequestList, request_id: str | None) -> AccessRequest:
        """Resolve ``request_id`` (or the sole approved request) in the catalog.

        Raises:
            RequestsError: If the id is unknown, the request is not approved, or
                ``request_id`` is None while zero / more than one request is
                approved.
        """
        if request_id is not None:
            request = catalog.get(request_id)
            if request is None:
                raise RequestsError(f"Unknown access request: {request_id!r}")
            if not request.is_approved:
                raise RequestsError(
                    f"Access request {request.id} is not approved (status: {request.status!r})"
                )
            return request

        approved = catalog.approved
        if len(approved) == 1:
            return approved[0]
        if not approved:
            raise RequestsError("You have no approved access requests")
        ids = ", ".join(r.id for r in approved)
        raise RequestsError(
            f"You have {len(approved)} approved access requests — pass request_id "
            f"to pick one of: {ids}"
        )

    def set_access(self, request_id: str | None = None) -> AccessRequest:
        """Pin an approved access request as this client's default.

        You only need this when you have **several** approved requests — with
        exactly one, every method resolves it automatically. After pinning,
        bare table names in :meth:`query` and :meth:`query_stream` resolve
        against the pinned request's schema, and :meth:`download_request`,
        :meth:`download_table`, and :meth:`row_counts` default to it:

            >>> client.set_access("dc43d8d9-6f6c-4152-b5f1-0f9668ba01e4")
            >>> client.query("SELECT * FROM person LIMIT 5")

        Args:
            request_id: The access request to pin — the UUID or the ``req_...``
                schema form. ``None`` (the default) pins your sole approved
                request and raises if you have zero or several.

        Returns:
            The pinned :class:`~cbr_data_access.access_requests.AccessRequest`.

        Raises:
            RequestsError: If listing fails, the id is unknown, the request is
                not approved, or ``request_id`` is None while zero / more than
                one request is approved.
        """
        request = self._resolve_request(self.list_requests(), request_id)
        self._default_request = request
        return request

    def clear_access(self) -> None:
        """Forget the pinned request; the next query re-resolves automatically."""
        self._default_request = None

    @property
    def default_request(self) -> AccessRequest | None:
        """The pinned access request, or ``None`` if none is pinned yet."""
        return self._default_request

    def _ensure_access(self) -> AccessRequest:
        """The pinned request, auto-pinning the sole approved one if needed.

        Raises:
            RequestsError: If listing fails, or zero / more than one request is
                approved (call :meth:`set_access` to pick one).
        """
        if self._default_request is None:
            self._default_request = self._resolve_request(self.list_requests(), None)
        return self._default_request

    def _effective_request_id(self, request_id: str | None) -> str | None:
        """An explicit ``request_id``, else the pinned default's id, else None."""
        if request_id is not None:
            return request_id
        if self._default_request is not None:
            return self._default_request.id
        return None

    # ------------------------------------------------------------------ #
    # downloads
    # ------------------------------------------------------------------ #

    def download_request(
        self,
        dest: str | os.PathLike[str],
        *,
        request_id: str | None = None,
        tables: Iterable[str] | None = None,
        file_format: str = "parquet",
        max_rows: int | None = None,
        create_dir: bool = True,
        overwrite: bool = True,
    ) -> list[pathlib.Path]:
        """Download an approved request's table data to files, one per table.

        For each granted table, reads ``req_<request_id>.<table>`` through the
        proxy and writes it to ``<dest>/<table>.<ext>``. Every row carries the
        ``cohort_id`` column the server prepends, so rows from different cohorts
        stay distinguishable.

        The data is **streamed** to disk in batches (server-side cursor), so peak
        memory stays flat no matter how large the table is — it never loads a
        whole table into memory. Writes are atomic: each file is written to a
        temporary ``.part`` and renamed into place only on success, so a failure
        never leaves a truncated file.

        This stops at the first error: if a table fails to download, the
        exception propagates and remaining tables are not attempted
        (already-written files are kept).

        Args:
            dest: Destination directory. One ``<table>.<ext>`` is written under it.
            request_id: Which access request to download. ``None`` (the default)
                uses your sole approved request and raises if you have zero or
                several. Accepts the UUID or the ``req_...`` schema form.
            tables: Optional subset of table names to download. ``None`` (the
                default) downloads every table the request grants. A requested
                name the request doesn't grant raises :class:`RequestsError`.
            file_format: ``"parquet"`` (the default) or ``"csv"``. Parquet is
                columnar and compressed, so files are much smaller than CSV while
                still holding all the data; use ``"csv"`` if you need plain text.
            max_rows: If set, export only the first this-many rows per table (a
                sample — the row order is arbitrary). ``None`` exports everything.
            create_dir: Create ``dest`` (and parent directories) if it does not
                exist. If ``False`` and ``dest`` is missing, raises
                :class:`DataAccessError`.
            overwrite: Overwrite an existing output file. If ``False`` and the file
                already exists, raises :class:`DataAccessError`.

        Returns:
            The absolute paths of the files written, in catalog order.

        Raises:
            RequestsError: If listing the catalog fails, the request can't be
                resolved, or a requested table name isn't granted by it.
            QueryError: If reading a table's data fails.
            DataAccessError: For invalid arguments (unknown ``file_format`` or a
                non-positive ``max_rows``) or filesystem problems (missing
                destination with ``create_dir=False``, or an existing file with
                ``overwrite=False``).

        Example:
            >>> paths = client.download_request("./exports")
            >>> paths
            [PosixPath('exports/measurement.parquet'), PosixPath('exports/person.parquet')]
            >>> client.download_request("./exports", tables=["person"], file_format="csv")
        """
        if file_format not in ("parquet", "csv"):
            raise DataAccessError(
                f"Unsupported file_format: {file_format!r} (use 'parquet' or 'csv')"
            )
        if max_rows is not None and max_rows <= 0:
            raise DataAccessError("max_rows must be a positive integer or None")

        request = self._resolve_request(
            self.list_requests(), self._effective_request_id(request_id)
        )
        selected = _select_tables(request, tables)

        dest_dir = pathlib.Path(dest)
        if not dest_dir.exists():
            if create_dir:
                dest_dir.mkdir(parents=True, exist_ok=True)
            else:
                raise DataAccessError(f"Destination directory does not exist: {dest_dir}")

        # Resolve and validate output paths before opening a connection.
        worklist: list[tuple[RequestTable, pathlib.Path]] = []
        used_stems: set[str] = set()
        for table in selected:
            stem = _safe_filename(table.name)
            if stem in used_stems:
                raise DataAccessError(
                    f"Filename collision for table {table.name!r}: {stem}.{file_format}"
                )
            used_stems.add(stem)

            final_path = dest_dir / f"{stem}.{file_format}"
            if final_path.exists() and not overwrite:
                raise DataAccessError(f"File exists (pass overwrite=True): {final_path}")
            worklist.append((table, final_path))

        written: list[pathlib.Path] = []
        if not worklist:
            return written

        with self.connect() as conn:
            # Fetch up to _CHUNK_SIZE rows per server round-trip rather than
            # SQLAlchemy's ~1000-row default buffer; far fewer round-trips on a
            # high-latency DB link, with peak memory still ~one chunk.
            streamed = conn.execution_options(
                stream_results=True, max_row_buffer=_CHUNK_SIZE
            )
            for table, final_path in worklist:
                relation = self._quote_relation(request, table)
                sql = f"SELECT * FROM {relation}"  # noqa: S608
                if max_rows is not None:
                    sql += f" LIMIT {int(max_rows)}"

                part_path = final_path.with_name(final_path.name + ".part")
                try:
                    # Cheap zero-row probe so an empty table still yields a file
                    # with the right header/schema (column names are not part of
                    # the request catalog).
                    probe = pd.read_sql_query(f"SELECT * FROM {relation} LIMIT 0", con=conn)  # noqa: S608
                    col_names = list(probe.columns)

                    rows = pd.read_sql_query(sql, con=streamed, chunksize=_CHUNK_SIZE)
                    if file_format == "csv":
                        _write_csv_stream(rows, part_path, col_names)
                    else:
                        _write_parquet_stream(rows, part_path, col_names)
                    os.replace(part_path, final_path)
                except SQLAlchemyError as exc:
                    raise QueryError(f"Query failed: {exc}") from exc
                finally:
                    # No-op on success (already renamed); cleans up a partial file
                    # if writing failed partway through.
                    part_path.unlink(missing_ok=True)
                written.append(final_path.resolve())

        return written

    def download_table(
        self,
        table: str,
        dest: str | os.PathLike[str],
        *,
        request_id: str | None = None,
        file_format: str = "parquet",
        max_rows: int | None = None,
        create_dir: bool = True,
        overwrite: bool = True,
    ) -> pathlib.Path:
        """Download a single table of an approved request, returning its path.

        Convenience wrapper around :meth:`download_request` for the common case
        of exporting just one table. The data is streamed and the file is
        written atomically, exactly as in :meth:`download_request`.

        Args:
            table: Bare table name to download (must be granted by the request).
            dest: Destination directory. The file ``<table>.<ext>`` is written
                under it.
            request_id: Which access request to use (``None`` = your sole
                approved request).
            file_format: ``"parquet"`` (the default) or ``"csv"``.
            max_rows: If set, export only the first this-many rows (a sample —
                arbitrary order). ``None`` exports everything.
            create_dir: Create ``dest`` (and parents) if it does not exist.
            overwrite: Overwrite an existing output file.

        Returns:
            The absolute path of the file written.

        Example:
            >>> client.download_table("person", "./exports")
            PosixPath('/.../exports/person.parquet')
        """
        paths = self.download_request(
            dest,
            request_id=request_id,
            tables=[table],
            file_format=file_format,
            max_rows=max_rows,
            create_dir=create_dir,
            overwrite=overwrite,
        )
        return paths[0]

    def row_counts(
        self,
        *,
        request_id: str | None = None,
        tables: Iterable[str] | None = None,
    ) -> dict[str, int]:
        """Return the number of rows in each of a request's tables.

        Handy for sizing up a download before running :meth:`download_request` —
        you can see which tables are large and decide whether to pass
        ``max_rows`` or a ``tables`` subset. Runs a cheap ``COUNT(*)`` per table
        (no row data is transferred), so it is safe even for very large tables.

        Args:
            request_id: Which access request to count (``None`` = your sole
                approved request).
            tables: Optional subset of table names to count. ``None`` (the
                default) counts every granted table.

        Returns:
            A dict mapping each table's name to its row count, in catalog order.

        Raises:
            RequestsError: If listing fails, the request can't be resolved, or a
                requested table name isn't granted by it.
            QueryError: If a count query fails.

        Example:
            >>> client.row_counts()
            {'measurement': 12345, 'person': 110}
        """
        request = self._resolve_request(
            self.list_requests(), self._effective_request_id(request_id)
        )
        selected = _select_tables(request, tables)
        counts: dict[str, int] = {}
        for table in selected:
            sql = f"SELECT count(*) AS n FROM {self._quote_relation(request, table)}"  # noqa: S608
            df = self.query(sql)
            counts[table.name] = int(df.iloc[0, 0])
        return counts

    @staticmethod
    def _quote_relation(request: AccessRequest, table: RequestTable) -> str:
        """Render a request table as a safely quoted ``schema.table`` SQL reference."""
        return f"{_IDENT.quote(request.schema)}.{_IDENT.quote(table.name)}"

    # ------------------------------------------------------------------ #
    # engine / connection
    # ------------------------------------------------------------------ #

    def _get_engine(self) -> Engine:
        """Return a cached engine, rebuilding it when the token has refreshed.

        The access token doubles as the database password, so a refreshed token
        invalidates the cached engine's URL; the stale engine is disposed and a
        new one is created.
        """
        token = self._ensure_token()
        if self._engine is not None and self._engine_token == token:
            return self._engine

        info = self._ensure_connection_info()
        if self._engine is not None:
            self._engine.dispose()

        url = URL.create(
            drivername="postgresql+psycopg2",
            username=info.get("username"),
            password=token,
            host=info.get("host"),
            port=info.get("port"),
            database=info.get("database"),
        )
        try:
            self._engine = create_engine(url)
        except SQLAlchemyError as exc:
            raise DBConnectionError(f"Failed to create engine: {exc}") from exc
        self._engine_token = token
        return self._engine

    @contextmanager
    def connect(self) -> Iterator[Connection]:
        """Yield a live SQLAlchemy :class:`Connection`, closed on exit.

        For normal queries you do **not** need this — just call :meth:`query`.
        This is for power users who want a raw SQLAlchemy ``Connection`` (e.g. to
        run several statements or use SQLAlchemy Core directly).

        This is a context manager: it only does anything inside a ``with`` block.
        Calling ``client.connect()`` on its own does nothing and connects to
        nothing — it just returns an un-entered context manager. Always use it
        as ``with client.connect() as conn:``.

        Raises:
            DBConnectionError: If a connection cannot be established (raised when
                the ``with`` block is entered).

        Example:
            >>> with client.connect() as conn:
            ...     conn.execute(text("SELECT 1"))
        """
        engine = self._get_engine()
        try:
            conn = engine.connect()
        except SQLAlchemyError as exc:
            raise DBConnectionError(f"Failed to connect: {exc}") from exc
        try:
            yield conn
        finally:
            conn.close()

    # ------------------------------------------------------------------ #
    # query convenience
    # ------------------------------------------------------------------ #

    def query(self, sql: str, params: dict[str, Any] | None = None) -> pd.DataFrame:
        """Run a read query and return the result as a pandas DataFrame.

        This is the normal way to use the client; you do **not** need to call
        :meth:`connect` first. Each call authenticates if needed, opens its own
        connection, and releases it when done.

        Write plain table names — they resolve against your approved access
        request automatically (with several approved requests, pick one first
        via :meth:`set_access`):

            >>> client.query("SELECT * FROM person WHERE person_id = :pid", {"pid": 42})

        Prefer bind parameters over string interpolation for values.

        Raises:
            RequestsError: If your approved request can't be resolved.
            QueryError: If executing the query fails.
        """
        sql = qualify_statement(sql, self._ensure_access().schema)
        statement: Any = text(sql) if params else sql
        try:
            with self.connect() as conn:
                return pd.read_sql_query(statement, con=conn, params=params)
        except SQLAlchemyError as exc:
            raise QueryError(f"Query failed: {exc}") from exc

    def query_stream(
        self,
        sql: str,
        params: dict[str, Any] | None = None,
        *,
        chunksize: int = _CHUNK_SIZE,
    ) -> Iterator[pd.DataFrame]:
        """Run a read query and yield the result as DataFrame batches.

        Like :meth:`query`, but for results too large (or too unbounded) to hold
        in one DataFrame: rows are pulled through a **server-side cursor** (the
        proxy's ``DECLARE``/``FETCH`` machinery) in batches of ``chunksize``, so
        peak memory is bounded by one chunk no matter how many rows the query
        returns.

        The database connection is held open for exactly as long as you iterate:
        it is released when the iterator is exhausted, when you ``break`` out /
        drop the iterator, or when you call ``.close()`` on it.

        Args:
            sql: The SELECT to run, with plain table names (as in :meth:`query`).
            params: Optional bind parameters, as in :meth:`query`.
            chunksize: Rows per yielded DataFrame (default 50,000).

        Yields:
            ``pandas.DataFrame`` batches of up to ``chunksize`` rows, in order.

        Raises:
            DataAccessError: If ``chunksize`` is not a positive integer.
            RequestsError: If your approved request can't be resolved.
            QueryError: If executing the query or fetching a batch fails.

        Example:
            >>> total = 0.0
            >>> for chunk in client.query_stream("SELECT value_as_number FROM measurement"):
            ...     total += chunk["value_as_number"].sum()
        """
        if not isinstance(chunksize, int) or isinstance(chunksize, bool) or chunksize <= 0:
            raise DataAccessError("chunksize must be a positive integer")

        sql = qualify_statement(sql, self._ensure_access().schema)
        statement: Any = text(sql) if params else sql

        def _stream() -> Iterator[pd.DataFrame]:
            try:
                with self.connect() as conn:
                    # Fetch up to ``chunksize`` rows per server round-trip instead
                    # of SQLAlchemy's ~1000-row default buffer. Over a high-latency
                    # link to the DB this cuts the number of round-trips ~50x (the
                    # dominant cost when streaming a large table); peak memory still
                    # stays bounded to roughly one chunk.
                    streamed = conn.execution_options(
                        stream_results=True, max_row_buffer=chunksize
                    )
                    yield from pd.read_sql_query(
                        statement, con=streamed, params=params, chunksize=chunksize
                    )
            except SQLAlchemyError as exc:
                raise QueryError(f"Query failed: {exc}") from exc

        return _stream()
