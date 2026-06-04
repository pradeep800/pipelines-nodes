# cbr-data-access — Usage Guide

A task-oriented walkthrough of everything you can do with the SDK. For a quick
overview see the [README](../README.md); this guide has a copy-paste example for
every feature.

The SDK wraps a three-step flow behind one client:

1. **Authenticate** against Keycloak with your username/password → a short-lived
   OAuth access token.
2. **Fetch** managed PostgreSQL connection metadata using that token.
3. **Connect** to PostgreSQL (the token doubles as the DB password) and run
   read-only queries, returned as pandas DataFrames.

All of this is lazy — constructing the client does no network I/O; it happens on
your first call.

## Contents

1. [Install & connect](#1-install--connect)
2. [Run a query → DataFrame](#2-run-a-query--dataframe)
3. [Discover & inspect views](#3-discover--inspect-views)
4. [Row counts](#4-row-counts)
5. [Download data to Parquet / CSV](#5-download-data-to-parquet--csv)
6. [Raw SQLAlchemy connection](#6-raw-sqlalchemy-connection)
7. [Tokens & auth introspection](#7-tokens--auth-introspection)
8. [Handling errors](#8-handling-errors)
9. [Configuration reference](#9-configuration-reference)
10. [Full API cheat-sheet](#10-full-api-cheat-sheet)

---

## 1. Install & connect

```bash
pip install cbr-data-access          # or: pip install -e ".[dev]" for development
```

Create a client with your Keycloak credentials. Nothing hits the network until
your first query/call — authentication and the DB connection are established
lazily and cached.

```python
from cbr_data_access import DataAccessClient

client = DataAccessClient(
    username="you@datakaveri.org",
    password="your-password",
)

df = client.query("SELECT * FROM person LIMIT 5")
print(df)
```

Use it as a context manager so the underlying connection pool is disposed when
you're done:

```python
with DataAccessClient(username="you@datakaveri.org", password="...") as client:
    df = client.query("SELECT count(*) FROM person")
# connection pool disposed here
```

> **Don't commit real credentials to source.** Read them from the environment or
> a secrets manager, e.g. `os.environ["CBR_PASSWORD"]`.

---

## 2. Run a query → DataFrame

`query()` is the normal way to read data. Each call authenticates if needed,
opens its own connection, runs the SQL, returns a `pandas.DataFrame`, and
releases the connection. You do **not** need to call `connect()` first.

```python
df = client.query("SELECT person_id, year_of_birth FROM person LIMIT 100")
```

Prefer **bind parameters** over string formatting (safer, avoids SQL injection):

```python
df = client.query(
    "SELECT * FROM person WHERE person_id = :pid",
    {"pid": 42},
)
```

Access is **read-only and schema-scoped** server-side — DDL and cross-schema
reads are denied by the platform.

---

## 3. Discover & inspect views

`list_views()` returns the catalog of views you can see under your tenant schema,
as structured objects.

```python
views = client.list_views()

views.names                 # ['person', 'visit_occurrence', ...]
views.username              # the authenticated user the catalog was built for
views.tenant_schema         # your tenant schema name
len(views)                  # number of views
views[0]                    # index it like a list
for v in views:             # or iterate it
    print(v.name)
```

Flatten the whole catalog into a DataFrame (one row per column) — handy for
browsing in a notebook:

```python
df = views.to_frame()
# columns: view, category, column, type, description, accessible
df.query("accessible")      # only the columns you're allowed to read
```

Inspect a single view and its columns:

```python
person = views.get("person")        # -> View, or None if not found
person.name, person.description, person.category

for col in person.columns:          # every column
    print(col.name, col.type, col.description, col.accessible)

person.accessible_columns           # only columns you can read
person.to_frame()                   # this view's columns as a DataFrame
```

Each column is a `ViewColumn` with `.name`, `.type` (e.g. `integer`, `string`),
`.description`, and `.accessible` (whether you may read it).

---

## 4. Row counts

`row_counts()` returns `{view_name: row_count}` — a cheap `COUNT(*)` per view (no
row data transferred). Useful to size up a download before running it.

```python
client.row_counts()
# {'person': 12345, 'visit_occurrence': 98765}

client.row_counts(views=["person"])   # just the views you care about
# {'person': 12345}
```

---

## 5. Download data to Parquet / CSV

Export view data to files in one call. The data is **streamed** to disk in
batches (a server-side cursor), so memory stays flat no matter how large a view
is — it never loads a whole table into RAM. Writes are **atomic** (a failed
download never leaves a half-written file), and the call **stops on the first
failing view**.

### All views (or a subset)

`download_views()` writes one file per view under a destination directory and
returns the list of paths it wrote:

```python
paths = client.download_views("./exports")
# [PosixPath('exports/person.parquet'), PosixPath('exports/visit_occurrence.parquet'), ...]

# A subset of views:
client.download_views("./exports", views=["person", "visit_occurrence"])
```

### A single view

`download_view()` is a convenience wrapper for one view; it returns the single
path it wrote:

```python
path = client.download_view("person", "./exports")
# PosixPath('exports/person.parquet')
```

### Options (both functions)

| Option        | Default     | What it does |
|---------------|-------------|--------------|
| `file_format` | `"parquet"` | `"parquet"` (columnar + compressed, much smaller) or `"csv"` (plain text). |
| `max_rows`    | `None`      | Export only the first N rows per view (a sample; order is arbitrary). |
| `create_dir`  | `True`      | Create the destination directory (and parents) if missing. |
| `overwrite`   | `True`      | Overwrite an existing output file; `False` raises if it exists. |

```python
# CSV instead of Parquet:
client.download_views("./exports", file_format="csv")

# Just a 1000-row sample of one view, as CSV:
client.download_view("person", "./exports", file_format="csv", max_rows=1000)
```

Only the columns you're allowed to read are exported; a view with no readable
columns is skipped by `download_views` (and raises for `download_view`, since
there'd be nothing to write).

### Reading the files back

```python
import pandas as pd

df = pd.read_parquet("exports/person.parquet")
df = pd.read_csv("exports/person.csv")
```

---

## 6. Raw SQLAlchemy connection

For power users who want a live SQLAlchemy `Connection` (e.g. to run several
statements, or use SQLAlchemy Core directly), use `connect()` as a context
manager:

```python
from sqlalchemy import text

with client.connect() as conn:
    result = conn.execute(text("SELECT 1"))
    print(result.scalar())
```

> `connect()` is a context manager — it only does anything inside a `with` block.
> Calling `client.connect()` on its own does nothing (it returns an un-entered
> context manager). For normal queries just use `query()`.

---

## 7. Tokens & auth introspection

Authentication is automatic and cached; an expired token triggers re-auth on the
next call. You rarely need these directly, but they're available:

```python
client.login()              # force authentication now; returns the access token (str)

client.token_claims         # decoded claims of the current token (dict), e.g. exp/sub
```

Decode any JWT's claims without verifying its signature (module-level helper):

```python
from cbr_data_access import decode_token

claims = decode_token(some_token)   # dict of claims
```

The token is short-lived and is used as the database password; the client
re-authenticates automatically as it nears expiry. Token refresh uses a fresh
password grant (no `refresh_token` flow yet).

---

## 8. Handling errors

Every error raised by the SDK derives from `DataAccessError`, so you can catch
that single base. Errors originating from an HTTP response also carry a
`.status_code`.

| Exception              | Raised when |
|------------------------|-------------|
| `DataAccessError`      | Base class for all SDK errors (also used for bad arguments / filesystem issues in downloads). |
| `AuthenticationError`  | Keycloak token request failed (bad credentials, non-200, transport error, no token). |
| `ConnectionInfoError`  | The connection-info endpoint failed or returned nothing. |
| `ViewsError`           | The views endpoint failed, returned an unexpected response, or a requested view name isn't in the catalog. |
| `DBConnectionError`    | Establishing the PostgreSQL connection failed. |
| `QueryError`           | Executing a SQL query (or streaming a download) failed. |

```python
from cbr_data_access import DataAccessError

try:
    df = client.query("SELECT * FROM person")
except DataAccessError as exc:
    print(f"Data access failed: {exc}")
    print("HTTP status:", exc.status_code)   # None if not from an HTTP response
```

---

## 9. Configuration reference

Pass these to the constructor. Only `username`/`password` are required; the
endpoint settings default to the CBR deployment and only need changing if you
target a different one.

| Argument          | Required | Default |
|-------------------|----------|---------|
| `username`        | yes      | — |
| `password`        | yes      | — |
| `keycloak_url`    | no       | `https://keycloak.cbr-iisc.ac.in/auth/realms/cbr/protocol/openid-connect/token` |
| `base_url`        | no       | `http://127.0.0.1:6060` |
| `client_id`       | no       | `omop-auth` |
| `request_timeout` | no       | `300.0` (seconds — 5 minutes) |

`base_url` is the root of the data-access service; the connection-info
(`/postgres/connection-info`) and views (`/views`) endpoints hang off it.

```python
client = DataAccessClient(
    username="you@datakaveri.org",
    password="...",
    base_url="https://data-access.example.org",
    request_timeout=120,        # override the per-request HTTP timeout
)
```

---

## 10. Full API cheat-sheet

**`DataAccessClient`** (construct with `username`, `password`, and optional
`keycloak_url` / `base_url` / `client_id` / `request_timeout`):

| Call | Returns | Description |
|------|---------|-------------|
| `query(sql, params=None)` | `DataFrame` | Run a read query (the normal path); supports bind params. |
| `list_views()` | `ViewList` | List the views you can see, with typed columns. |
| `row_counts(views=None)` | `dict[str, int]` | `{view: COUNT(*)}` for all views or a subset. |
| `download_views(dest, *, views=None, file_format="parquet", max_rows=None, create_dir=True, overwrite=True)` | `list[Path]` | Stream every view (or a subset) to one file each; returns paths written. |
| `download_view(view, dest, *, file_format="parquet", max_rows=None, create_dir=True, overwrite=True)` | `Path` | Stream a single view to one file; returns its path. |
| `connect()` | context mgr → `Connection` | Live SQLAlchemy connection (`with`-only); for raw access. |
| `login()` | `str` | Force authentication; returns the access token. |
| `token_claims` | `dict` | Decoded claims of the current token (property). |
| `close()` | `None` | Dispose the connection pool (called automatically on `with` exit). |

**Models** (`from cbr_data_access import View, ViewColumn, ViewList`):

| Type | Attributes / methods |
|------|----------------------|
| `ViewList` | `.username`, `.tenant_schema`, `.views`, `.names`, `.get(name)`, `.to_frame()`, iterable/indexable |
| `View` | `.name`, `.description`, `.category`, `.columns`, `.accessible_columns`, `.to_frame()` |
| `ViewColumn` | `.name`, `.type`, `.description`, `.accessible` |

**Module-level:** `decode_token(token) -> dict`, `__version__`.

**Exceptions:** `DataAccessError` (base, `.status_code`), `AuthenticationError`,
`ConnectionInfoError`, `ViewsError`, `DBConnectionError`, `QueryError`.
