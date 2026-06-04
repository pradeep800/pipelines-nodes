# cbr-data-access

Python SDK for authenticated data access to the CBR / datakaveri platform.

It wraps a three-step flow behind a single client:

1. **Authenticate** against Keycloak with a username/password to obtain a
   short-lived OAuth access token.
2. **Fetch** managed PostgreSQL connection metadata from the connection-info
   service using that token.
3. **Connect** to PostgreSQL via SQLAlchemy — the access token is used as the
   database password — and run read queries (returned as pandas DataFrames).

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

See [`examples/quickstart.py`](examples/quickstart.py) for a fuller example.

Normal use is just `client.query(...)` — it authenticates, connects, and
disconnects for you on each call. You don't call `connect()` yourself.

> Don't commit real credentials to source.

## Discovering available views

List the views under your tenant schema and inspect their columns (including
which ones you're allowed to read):

```python
with DataAccessClient(username="you@datakaveri.org", password="...") as client:
    views = client.list_views()

    print(views.names)            # ['person', 'visit_occurrence', ...]
    print(views.to_frame())       # one row per column — view, type, accessible, ...

    person = views.get("person")
    accessible = [c.name for c in person.accessible_columns]
    print(accessible)
```

## Downloading view data

Dump the data from your views to files in one call — one file per view, named
after the view, written under a destination directory. It returns the list of
file paths it created:

```python
with DataAccessClient(username="you@datakaveri.org", password="...") as client:
    # See how big each view is before downloading.
    print(client.row_counts())   # {'person': 12345, 'visit_occurrence': 98765}

    paths = client.download_views("./exports")
    print(paths)  # [PosixPath('exports/person.parquet'), ...]

    # Or just one view — returns the single path it wrote:
    path = client.download_view("person", "./exports")
```

The data is **streamed** to disk in batches, so memory stays flat no matter how
large a view is — it never loads a whole table into RAM.

- **Format:** Parquet by default (columnar + compressed, so files are far smaller
  than CSV while keeping all the data). Pass `file_format="csv"` for plain text:
  `client.download_views("./exports", file_format="csv")`.
- **Sampling:** pass `max_rows=1000` to export only the first N rows per view
  instead of the whole table.
- Only the columns you're allowed to read are exported; a view with no readable
  columns is skipped. Pass `views=["person", ...]` to export a subset.
- The destination directory is created if missing (`create_dir=False` to require
  it); existing files are overwritten (`overwrite=False` to refuse). Writes are
  atomic — a failed download never leaves a half-written file — and the call
  stops and raises on the first failing view.

Use `client.row_counts()` (or `client.row_counts(views=["person"])`) on its own
to get a `{view_name: row_count}` dict — a cheap `COUNT(*)` per view, handy for
deciding whether to pass `max_rows`.

## Configuration

Pass the username and password to the constructor. The endpoint settings have
sensible defaults and only need to be set if you target a different deployment.

| Constructor argument   | Required | Default |
|------------------------|----------|---------|
| `username`             | yes      | — |
| `password`             | yes      | — |
| `keycloak_url`         | no       | `https://keycloak.cbr-iisc.ac.in/auth/realms/cbr/protocol/openid-connect/token` |
| `base_url`             | no       | `http://127.0.0.1:6060` |
| `client_id`            | no       | `omop-auth` |

`base_url` is the root of the data-access service; the connection-info
(`/postgres/connection-info`) and views (`/views`) endpoints hang off it.

## API overview

- **`DataAccessClient(username, password, *, keycloak_url=..., base_url=..., client_id=..., request_timeout=300.0)`**
  — construct a client (no network I/O until first use). Usable as a context
  manager; exiting disposes the connection pool.
- **`client.login() -> str`** — authenticate and cache the token (called lazily
  when needed; an expired token triggers re-authentication automatically).
- **`client.token_claims -> dict`** — decoded claims of the current token.
- **`client.query(sql, params=None) -> pandas.DataFrame`** — the normal way to
  read data; no `connect()` call required, as `query()` connects and disconnects
  for you. Prefer bind parameters: `client.query("... WHERE id = :id", {"id": 1})`.
- **`client.connect()`** — context manager yielding a live SQLAlchemy
  `Connection`, closed on exit. Use it as `with client.connect() as conn:`;
  calling it bare does nothing. For power users who want raw SQLAlchemy access —
  you don't need it for normal queries.
- **`client.list_views() -> ViewList`** — list the views available under your
  tenant schema. Returns a `ViewList` (a sequence of `View`, each carrying
  typed `columns` that flag whether they're `accessible` to you). Call
  `.to_frame()` on the result — or on a single `View` — for a pandas DataFrame
  you can browse and filter.
- **`client.row_counts(views=None) -> dict[str, int]`** — return a
  `{view_name: row_count}` dict (a cheap `COUNT(*)` per view); handy for sizing
  up a download.
- **`client.download_views(dest, *, views=None, file_format="parquet", max_rows=None, create_dir=True, overwrite=True) -> list[pathlib.Path]`**
  — stream view data to files (one `<view>.<ext>` per view under `dest`,
  accessible columns only) and return the paths written. Parquet by default
  (`file_format="csv"` for CSV); `max_rows` caps rows per view. Memory-bounded,
  atomic writes, stops and raises on the first failure.
- **`client.download_view(view, dest, *, file_format="parquet", max_rows=None, create_dir=True, overwrite=True) -> pathlib.Path`**
  — download a single named view and return the one path written; a convenience
  wrapper over `download_views` with the same options.
- **`decode_token(token) -> dict`** — decode a JWT's claims without verifying
  its signature.

### Exceptions

All SDK errors derive from `DataAccessError` (which carries an optional
`.status_code`):

- `AuthenticationError` — Keycloak token request failed.
- `ConnectionInfoError` — connection-info endpoint failed or returned nothing.
- `ViewsError` — the views endpoint failed or returned an unexpected response.
- `DBConnectionError` — establishing the database connection failed.
- `QueryError` — query execution failed.

## Authentication & security notes

- The access token is **short-lived** and is used as the database password; the
  client re-authenticates automatically when the token nears expiry.
- Token refresh uses a fresh password grant (no `refresh_token` flow yet).
- Database access is schema-scoped server-side — DDL and cross-schema reads are
  denied by the platform.
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
