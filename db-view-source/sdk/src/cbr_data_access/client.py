"""Client for the CBR / datakaveri data-access platform.

Flow:
    1. Authenticate against Keycloak (username/password) -> OAuth access token.
    2. Call the connection-info endpoint with that token -> DB connection
       metadata (host, port, database, username).
    3. Connect to PostgreSQL via SQLAlchemy, using the access token as the
       database password.
    4. Run SQL queries.

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
from .exceptions import (
    AuthenticationError,
    ConnectionInfoError,
    DataAccessError,
    DBConnectionError,
    QueryError,
    ViewsError,
)
from .views import View, ViewColumn, ViewList

logger = logging.getLogger(__name__)

_EXP_SKEW_SECONDS = 30  # treat the token as expired this many seconds early

# Quotes SQL identifiers (and doubles embedded quotes) for PostgreSQL. View and
# column names are identifiers, not values, so they can't be bind parameters —
# they must be safely quoted when interpolated into a query.
_IDENT = postgresql.dialect().identifier_preparer

# Rows pulled per batch when streaming a view to disk. Bounds peak memory: only
# this many rows are held at once regardless of how large the view is.
_CHUNK_SIZE = 50_000


def _safe_filename(name: str) -> str:
    """Turn a view name into a filesystem-safe filename stem.

    View names are normally plain identifiers (``[a-z0-9_]``), but be defensive
    against path separators or other surprises in server-provided names.
    """
    return re.sub(r"[^A-Za-z0-9._-]", "_", os.path.basename(name)) or "view"


def _select_views(catalog: ViewList, views: Iterable[str] | None) -> list[View]:
    """Resolve requested view names against the catalog (all views if ``None``).

    Raises:
        ViewsError: If a requested name is not in the catalog.
    """
    if views is None:
        return list(catalog.views)
    selected: list[View] = []
    for name in views:
        view = catalog.get(name)
        if view is None:
            raise ViewsError(f"Unknown view: {name!r}")
        selected.append(view)
    return selected


def _write_csv_stream(
    rows: Iterator[pd.DataFrame], path: pathlib.Path, col_names: list[str]
) -> None:
    """Stream chunks to a CSV: header on the first chunk, data appended after.

    If the view has no rows (no chunks), a header-only file is still written
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
    dtype inference (e.g. int vs float-with-nulls) cannot drift. An empty view
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
    """Encapsulates auth, connection-info retrieval, and database access.

    Authentication is lazy: constructing the client performs no network I/O.
    The access token and SQLAlchemy engine are cached on the instance and
    reused; an expired token triggers automatic re-authentication.

    Usage:
        :meth:`query` is the primary entry point and handles everything for you
        (auth, connecting, and disconnecting) — you do not need to call
        :meth:`connect`. :meth:`connect` is an optional context manager for power
        users who want a raw SQLAlchemy ``Connection``; use it as
        ``with client.connect() as conn:``. :meth:`list_views` lists the views
        you can read, :meth:`row_counts` reports how many rows each one has, and
        :meth:`download_views` streams them to Parquet (or CSV) files.

    Args:
        username: Keycloak username.
        password: Keycloak password.
        keycloak_url: Token endpoint (defaults to the CBR Keycloak realm).
        base_url: Base URL of the data-access service; the connection-info and
            views endpoints hang off it.
        client_id: OAuth client id.
        request_timeout: Per-request HTTP timeout, in seconds (default 300, i.e.
            5 minutes).

    Raises:
        AuthenticationError: If username or password is empty.

    Example:
        >>> with DataAccessClient(username="u@example.org", password="...") as client:
        ...     df = client.query("SELECT count(*) FROM person")
    """

    def __init__(
        self,
        username: str,
        password: str,
        *,
        keycloak_url: str = config.DEFAULT_KEYCLOAK_URL,
        base_url: str = config.DEFAULT_BASE_URL,
        client_id: str = config.DEFAULT_CLIENT_ID,
        request_timeout: float = 300.0,
    ) -> None:
        if not username or not password:
            raise AuthenticationError("username and password are required")

        self._username = username
        self._password = password
        self._keycloak_url = keycloak_url
        self._base_url = base_url.rstrip("/")
        self._connection_info_url = f"{self._base_url}{config.CONNECTION_INFO_PATH}"
        self._views_url = f"{self._base_url}{config.VIEWS_PATH}"
        self._client_id = client_id
        self._timeout = request_timeout

        self._access_token: str | None = None
        self._token_exp: float | None = None
        self._connection_info: dict[str, Any] | None = None
        self._engine: Engine | None = None
        self._engine_token: str | None = None  # token the cached engine was built with

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

        Returns:
            The access token.

        Raises:
            AuthenticationError: On transport failure, a non-200 response, or a
                response without an ``access_token``.
        """
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
        """Decoded claims of the current token (authenticates if needed)."""
        return decode_token(self._ensure_token())

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
    # views / catalog
    # ------------------------------------------------------------------ #

    def list_views(self) -> ViewList:
        """List the views available under the caller's tenant schema.

        Calls the views endpoint with the current access token and returns the
        catalog as structured objects. The result is a sequence of
        :class:`~cbr_data_access.views.View` (iterate or index it) that also
        carries ``username`` and ``tenant_schema``; each view's columns report
        whether they are ``accessible`` to the caller.

        Returns:
            A :class:`~cbr_data_access.views.ViewList` of the available views.

        Raises:
            AuthenticationError: If authentication fails.
            ViewsError: On transport failure or a non-200 response.

        Example:
            >>> views = client.list_views()
            >>> views.names
            ['person', 'visit_occurrence']
            >>> views.to_frame()  # one row per column, handy for browsing
        """
        token = self._ensure_token()
        headers = {"Authorization": f"Bearer {token}"}
        try:
            response = requests.get(self._views_url, headers=headers, timeout=self._timeout)
        except requests.RequestException as exc:
            raise ViewsError(f"Views request failed: {exc}") from exc

        if response.status_code != 200:
            raise ViewsError(response.text, status_code=response.status_code)

        return ViewList.from_dict(response.json())

    def download_views(
        self,
        dest: str | os.PathLike[str],
        *,
        views: Iterable[str] | None = None,
        file_format: str = "parquet",
        max_rows: int | None = None,
        create_dir: bool = True,
        overwrite: bool = True,
    ) -> list[pathlib.Path]:
        """Download view data to files, one file per view.

        For each view, reads its **accessible** columns (the ones
        :meth:`list_views` reports you may read) and writes them to
        ``<dest>/<view>.<ext>``. A view with no accessible columns is skipped.

        The data is **streamed** to disk in batches (server-side cursor), so peak
        memory stays flat no matter how large the view is — it never loads a whole
        table into memory. Writes are atomic: each file is written to a temporary
        ``.part`` and renamed into place only on success, so a failure never
        leaves a truncated file.

        This stops at the first error: if a view fails to download, the exception
        propagates and remaining views are not attempted (already-written files
        are kept).

        Args:
            dest: Destination directory. One ``<view>.<ext>`` is written under it.
            views: Optional subset of view names to download. ``None`` (the
                default) downloads every view in your catalog. A requested name
                that isn't in the catalog raises :class:`ViewsError`.
            file_format: ``"parquet"`` (the default) or ``"csv"``. Parquet is
                columnar and compressed, so files are much smaller than CSV while
                still holding all the data; use ``"csv"`` if you need plain text.
            max_rows: If set, export only the first this-many rows per view (a
                sample — the row order is arbitrary). ``None`` exports everything.
            create_dir: Create ``dest`` (and parent directories) if it does not
                exist. If ``False`` and ``dest`` is missing, raises
                :class:`DataAccessError`.
            overwrite: Overwrite an existing output file. If ``False`` and the file
                already exists, raises :class:`DataAccessError`.

        Returns:
            The absolute paths of the files written, in catalog order.

        Raises:
            ViewsError: If listing the catalog fails, or a requested view name is
                not in it.
            QueryError: If reading a view's data fails.
            DataAccessError: For invalid arguments (unknown ``file_format`` or a
                non-positive ``max_rows``) or filesystem problems (missing
                destination with ``create_dir=False``, or an existing file with
                ``overwrite=False``).

        Example:
            >>> paths = client.download_views("./exports")
            >>> paths
            [PosixPath('exports/person.parquet'), PosixPath('exports/visit_occurrence.parquet')]
            >>> client.download_views("./exports", file_format="csv", max_rows=1000)
        """
        if file_format not in ("parquet", "csv"):
            raise DataAccessError(
                f"Unsupported file_format: {file_format!r} (use 'parquet' or 'csv')"
            )
        if max_rows is not None and max_rows <= 0:
            raise DataAccessError("max_rows must be a positive integer or None")

        catalog = self.list_views()
        selected = _select_views(catalog, views)

        dest_dir = pathlib.Path(dest)
        if not dest_dir.exists():
            if create_dir:
                dest_dir.mkdir(parents=True, exist_ok=True)
            else:
                raise DataAccessError(f"Destination directory does not exist: {dest_dir}")

        # Resolve and validate output paths before opening a connection.
        worklist: list[tuple[View, list[ViewColumn], pathlib.Path]] = []
        used_stems: set[str] = set()
        for view in selected:
            cols = view.accessible_columns
            if not cols:
                logger.warning("Skipping view %r: no accessible columns", view.name)
                continue

            stem = _safe_filename(view.name)
            if stem in used_stems:
                raise DataAccessError(
                    f"Filename collision for view {view.name!r}: {stem}.{file_format}"
                )
            used_stems.add(stem)

            final_path = dest_dir / f"{stem}.{file_format}"
            if final_path.exists() and not overwrite:
                raise DataAccessError(f"File exists (pass overwrite=True): {final_path}")
            worklist.append((view, cols, final_path))

        written: list[pathlib.Path] = []
        if not worklist:
            return written

        with self.connect() as conn:
            streamed = conn.execution_options(stream_results=True)
            for view, cols, final_path in worklist:
                col_list = ", ".join(_IDENT.quote(c.name) for c in cols)
                sql = f"SELECT {col_list} FROM {_IDENT.quote(view.name)}"  # noqa: S608
                if max_rows is not None:
                    sql += f" LIMIT {int(max_rows)}"

                part_path = final_path.with_name(final_path.name + ".part")
                col_names = [c.name for c in cols]
                try:
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

    def download_view(
        self,
        view: str,
        dest: str | os.PathLike[str],
        *,
        file_format: str = "parquet",
        max_rows: int | None = None,
        create_dir: bool = True,
        overwrite: bool = True,
    ) -> pathlib.Path:
        """Download a single view's data to one file, returning its path.

        Convenience wrapper around :meth:`download_views` for the common case of
        exporting just one view. The data is streamed and the file is written
        atomically, exactly as in :meth:`download_views` — see it for details.

        Args:
            view: Name of the view to download (must be in your catalog).
            dest: Destination directory. The file ``<view>.<ext>`` is written
                under it.
            file_format: ``"parquet"`` (the default) or ``"csv"``.
            max_rows: If set, export only the first this-many rows (a sample —
                arbitrary order). ``None`` exports everything.
            create_dir: Create ``dest`` (and parents) if it does not exist.
            overwrite: Overwrite an existing output file.

        Returns:
            The absolute path of the file written.

        Raises:
            ViewsError: If listing the catalog fails, or ``view`` is not in it.
            QueryError: If reading the view's data fails.
            DataAccessError: For invalid arguments, filesystem problems, or if
                ``view`` has no accessible columns (so there is nothing to write).

        Example:
            >>> client.download_view("person", "./exports")
            PosixPath('/.../exports/person.parquet')
            >>> client.download_view("person", "./exports", file_format="csv", max_rows=1000)
        """
        paths = self.download_views(
            dest,
            views=[view],
            file_format=file_format,
            max_rows=max_rows,
            create_dir=create_dir,
            overwrite=overwrite,
        )
        if not paths:
            raise DataAccessError(f"View {view!r} has no accessible columns to download")
        return paths[0]

    def row_counts(self, views: Iterable[str] | None = None) -> dict[str, int]:
        """Return the number of rows in each view, as a ``{view_name: count}`` dict.

        Handy for sizing up a download before running :meth:`download_views` — you
        can see which views are large and decide whether to pass ``max_rows`` or a
        ``views`` subset. Runs a cheap ``COUNT(*)`` per view (no row data is
        transferred), so it is safe even for very large views.

        Args:
            views: Optional subset of view names to count. ``None`` (the default)
                counts every view in your catalog. A requested name that isn't in
                the catalog raises :class:`ViewsError`.

        Returns:
            A dict mapping each view's name to its row count, in catalog order.

        Raises:
            ViewsError: If listing the catalog fails, or a requested view name is
                not in it.
            QueryError: If a count query fails.

        Example:
            >>> client.row_counts()
            {'person': 12345, 'visit_occurrence': 98765}
        """
        catalog = self.list_views()
        selected = _select_views(catalog, views)
        counts: dict[str, int] = {}
        for view in selected:
            sql = f"SELECT count(*) AS n FROM {_IDENT.quote(view.name)}"  # noqa: S608
            df = self.query(sql)
            counts[view.name] = int(df.iloc[0, 0])
        return counts

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

        Prefer bind parameters over string interpolation:

            >>> client.query(
            ...     "SELECT * FROM person WHERE person_id = :pid", {"pid": 42}
            ... )

        Raises:
            QueryError: If executing the query fails.
        """
        statement: Any = text(sql) if params else sql
        try:
            with self.connect() as conn:
                return pd.read_sql_query(statement, con=conn, params=params)
        except SQLAlchemyError as exc:
            raise QueryError(f"Query failed: {exc}") from exc
