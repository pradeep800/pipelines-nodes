# cbr-data-access

Python SDK for authenticated data access to the CBR / datakaveri platform.

It wraps a three-step flow behind a single client:

1. **Authenticate** against Keycloak with a username/password to obtain a
   short-lived OAuth access token — or skip this step by passing an API key
   (`sk_...`), which the server accepts anywhere an access token is accepted.
2. **Fetch** managed PostgreSQL connection metadata from the connection-info
   service using that credential.
3. **Connect** to PostgreSQL via SQLAlchemy — the access token (or API key) is
   used as the database password — and run read queries (returned as pandas
   DataFrames).

Data access is granted per **access request**: an approved request unlocks a set
of cohorts and concepts, and each of its tables is queryable through the proxy as
a `req_<request_id>.<table>` relation. Behind each relation the server constructs
a `UNION ALL` across the request's cohorts, row-filtered to the granted concept
ids, with a `cohort_id` column telling you which cohort each row came from.

> For a full walkthrough of every feature with copy-paste examples, see the
> [Usage Guide](docs/USAGE.md).

## Requirements

- Python >= 3.10
- Network access to the Keycloak realm and the connection-info service.

## Installation

```bash
pip install cbr-data-access
```

For local development (editable install with dev tooling):

```bash
pip install -e ".[dev]"
```

## Quickstart

```python
from cbr_data_access import DataAccessClient

with DataAccessClient(
    username="you@datakaveri.org",
    password="your-password",
) as client:
    df = client.query("SELECT * FROM person LIMIT 10")
    print(df)
```

Or authenticate with an API key issued by the OMOP auth server instead — the
identity and grants behind it are the same:

```python
with DataAccessClient(api_key="sk_...") as client:
    df = client.query("SELECT * FROM person LIMIT 10")
```

See [`examples/quickstart.py`](examples/quickstart.py) for a fuller example.

Normal use is just `client.query(...)` — it authenticates, connects,
disconnects, and resolves plain table names against your approved access
request for you. You don't call `connect()` yourself. If you have **several**
approved requests, pick one once with `client.set_access(request_id)`; with
exactly one, nothing else is needed.

> Don't commit real credentials to source.

## Discovering your access requests

List your access requests and what each approved one makes queryable:

```python
with DataAccessClient(username="you@datakaveri.org", password="...") as client:
    requests = client.list_requests()

    print(requests.to_frame())     # one row per (request, table) — id, status, relation
    print(requests.ids)            # ['dc43d8d9-6f6c-4152-b5f1-0f9668ba01e4', ...]

    request = requests.approved[0]
    print(request.table_names)     # ['measurement', 'person', ...]
    print(request.concepts_frame())  # the concept grants behind the request
```

Pending/rejected requests are listed too, but expose no tables. If you have
exactly one approved request you can omit `request_id` everywhere; with several,
pass it explicitly (the UUID or the `req_...` form both work).

## Downloading request data

Dump an approved request's tables to files in one call — one file per table,
named after the table, written under a destination directory. It returns the
list of file paths it created:

```python
with DataAccessClient(username="you@datakaveri.org", password="...") as client:
    # See how big each table is before downloading.
    print(client.row_counts())   # {'measurement': 12345, 'person': 110}

    paths = client.download_request("./exports")
    print(paths)  # [PosixPath('exports/measurement.parquet'), ...]

    # Or just one table — returns the single path it wrote:
    path = client.download_table("person", "./exports")
```

The data is **streamed** to disk in batches, so memory stays flat no matter how
large a table is — it never loads a whole table into RAM.

- **Format:** Parquet by default (columnar + compressed, so files are far smaller
  than CSV while keeping all the data). Pass `file_format="csv"` for plain text:
  `client.download_request("./exports", file_format="csv")`.
- **Sampling:** pass `max_rows=1000` to export only the first N rows per table
  instead of the whole table.
- Pass `tables=["person", ...]` to export a subset, and `request_id=...` to pick
  one of several approved requests.
- The destination directory is created if missing (`create_dir=False` to require
  it); existing files are overwritten (`overwrite=False` to refuse). Writes are
  atomic — a failed download never leaves a half-written file — and the call
  stops and raises on the first failing table.

Use `client.row_counts()` (or `client.row_counts(tables=["person"])`) on its own
to get a `{table_name: row_count}` dict — a cheap `COUNT(*)` per table, handy for
deciding whether to pass `max_rows`.

## Configuration

Pass either an API key or a username and password to the constructor. The
endpoint settings have sensible defaults and only need to be set if you target a
different deployment.

| Constructor argument   | Required | Default |
|------------------------|----------|---------|
| `username`             | unless `api_key` | — |
| `password`             | unless `api_key` | — |
| `api_key`              | unless `username`/`password` | — |
| `keycloak_url`         | no       | `https://keycloak.cbr-iisc.ac.in/auth/realms/cbr/protocol/openid-connect/token` |
| `base_url`             | no       | `http://127.0.0.1:6060` |
| `client_id`            | no       | `angular-client` |

`base_url` is the root of the data-access service; the connection-info
(`/postgres/connection-info`) and requests (`/requests`) endpoints hang off it.

## API overview

- **`DataAccessClient(username=None, password=None, *, api_key=None, keycloak_url=..., base_url=..., client_id=..., request_timeout=300.0)`**
  — construct a client (no network I/O until first use). Pass `api_key` instead
  of `username`/`password` to authenticate with an API key. Usable as a context
  manager; exiting disposes the connection pool.
- **`client.login() -> str`** — authenticate and cache the token (called lazily
  when needed; an expired token triggers re-authentication automatically). In
  API-key mode there is nothing to log in to, so it just returns the key.
- **`client.token_claims -> dict`** — decoded claims of the current token.
  Raises `AuthenticationError` in API-key mode, as a key carries no claims.
- **`client.identity -> str | None`** — the username this credential resolves
  to; asks the server when only an API key is known.
- **`client.set_access(request_id=None) -> AccessRequest`** — pick which
  approved request to query (only needed when you have several; with one it is
  resolved automatically). `client.default_request` shows the current pick;
  `client.clear_access()` forgets it.
- **`client.query(sql, params=None) -> pandas.DataFrame`** — the normal way to
  read data; no `connect()` call required, as `query()` connects and disconnects
  for you. Prefer bind parameters: `client.query("... WHERE id = :id", {"id": 1})`.
- **`client.query_stream(sql, params=None, *, chunksize=50_000) -> Iterator[pandas.DataFrame]`**
  — like `query()`, but yields the result in DataFrame batches through a
  server-side cursor, so memory stays bounded for arbitrarily large results. The
  connection is released when the iterator is exhausted or closed.
- **`client.connect()`** — context manager yielding a live SQLAlchemy
  `Connection`, closed on exit. Use it as `with client.connect() as conn:`;
  calling it bare does nothing. For power users who want raw SQLAlchemy access —
  you don't need it for normal queries.
- **`client.list_requests() -> RequestList`** — list your access requests.
  Returns a `RequestList` (a sequence of `AccessRequest`, each carrying its
  queryable `tables` and the `concepts` granted). Call `.to_frame()` on the
  result for a pandas DataFrame you can browse and filter.
- **`client.row_counts(*, request_id=None, tables=None) -> dict[str, int]`** —
  return a `{table_name: row_count}` dict (a cheap `COUNT(*)` per table); handy
  for sizing up a download.
- **`client.download_request(dest, *, request_id=None, tables=None, file_format="parquet", max_rows=None, create_dir=True, overwrite=True) -> list[pathlib.Path]`**
  — stream an approved request's tables to files (one `<table>.<ext>` per table
  under `dest`) and return the paths written. Parquet by default
  (`file_format="csv"` for CSV); `max_rows` caps rows per table. Memory-bounded,
  atomic writes, stops and raises on the first failure.
- **`client.download_table(table, dest, *, request_id=None, file_format="parquet", max_rows=None, create_dir=True, overwrite=True) -> pathlib.Path`**
  — download a single table and return the one path written; a convenience
  wrapper over `download_request` with the same options.
- **`decode_token(token) -> dict`** — decode a JWT's claims without verifying
  its signature.

### Exceptions

All SDK errors derive from `DataAccessError` (which carries an optional
`.status_code`):

- `AuthenticationError` — Keycloak token request failed.
- `ConnectionInfoError` — connection-info endpoint failed or returned nothing.
- `RequestsError` — the requests endpoint failed, a referenced request/table is
  not in your catalog, or a request is not approved.
- `DBConnectionError` — establishing the database connection failed.
- `QueryError` — query execution failed.

## Authentication & security notes

- The access token is **short-lived** and is used as the database password; the
  client re-authenticates automatically when the token nears expiry.
- Token refresh uses a fresh password grant (no `refresh_token` flow yet).
- An API key is used in the same two places (bearer header, database password)
  but does not expire, so it is never refreshed. Treat it as a password: it
  stands in for the user until it is revoked.
- Access is request-scoped server-side: only relations granted by an APPROVED
  access request resolve, queries are read-only, and concept-level row filters
  are applied inside the server-constructed queries — they cannot be bypassed
  from SQL.
- Rotate any credentials or tokens that have ever been committed to source.

## Development

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

ruff check . && ruff format --check .
mypy
pytest
```

## License

Apache-2.0. See [LICENSE](LICENSE).
