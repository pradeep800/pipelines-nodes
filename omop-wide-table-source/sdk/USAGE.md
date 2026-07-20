# cbr-data-access — Usage Guide

A task-oriented walkthrough of everything you can do with the SDK. For a quick
overview see the [README](../README.md); this guide has a copy-paste example for
every feature.

The SDK wraps a three-step flow behind one client:

1. **Read** the platform-provided bearer token from the notebook sidecar.
2. **Fetch** managed PostgreSQL connection metadata using that token.
3. **Connect** to PostgreSQL with SQLAlchemy/psycopg2 or Apache Arrow ADBC (the
   token doubles as the DB password) and run read-only queries, returned as
   pandas DataFrames, polars DataFrames, or native Arrow objects.

Data access is granted per **access request**. An approved request unlocks a set
of cohorts and concepts; each of its tables is queryable through the proxy as a
`req_<request_id>.<table>` relation. Behind that relation the server constructs
a `UNION ALL` across the request's cohorts, row-filtered to the granted concept
ids, with a `cohort_id` column telling you which cohort each row came from.

All of this is lazy — constructing the client does no network I/O; it happens on
your first call.

## Contents

1. [Install & connect](#1-install--connect)
2. [Run a query → DataFrame](#2-run-a-query--dataframe)
   - [Arrow-native ADBC](#arrow-native-adbc)
3. [Discover your access requests](#3-discover-your-access-requests)
4. [Row counts](#4-row-counts)
5. [Raw SQLAlchemy connection](#5-raw-sqlalchemy-connection)
6. [Tokens & auth introspection](#6-tokens--auth-introspection)
7. [Handling errors](#7-handling-errors)
8. [Configuration reference](#8-configuration-reference)
9. [Full API cheat-sheet](#9-full-api-cheat-sheet)

---

## 1. Install & connect

```bash
pip install cbr-data-access          # or: pip install -e ".[dev]" for development
```

Create a client. Nothing hits the network until your first query/call — token
loading and the DB connection are established lazily and cached.

```python
from cbr_data_access import DataAccessClient

client = DataAccessClient()

df = client.query("SELECT * FROM person LIMIT 5")
print(df)
```

Use it as a context manager so the underlying connection pool is disposed when
you're done:

```python
with DataAccessClient() as client:
    df = client.query("SELECT count(*) FROM person")
# connection pool disposed here
```

Closing is not final: after `client.close()` (or the `with` block) the client
object stays usable — the next call re-authenticates and reconnects lazily,
exactly like a fresh client. If a `with` block doesn't fit (e.g. a long-lived
notebook client), just call `client.close()` when you're done.

> **Don't commit real tokens to source.** In the platform, the token is provided
> at `/var/run/sandbox-connect/platform/token`; for local tests, set
> `CBR_TOKEN_FILE` to another token file.

---

## 2. Run a query → DataFrame

`query()` is the normal way to read data. Each call authenticates if needed,
opens its own connection, runs the SQL, returns a `pandas.DataFrame`, and
releases the connection. You do **not** need to call `connect()` first.

Write plain table names — the client resolves them against your approved
access request automatically:

```python
df = client.query("SELECT person_id, year_of_birth FROM person LIMIT 100")
```

With **several** approved requests, pick one once with
`client.set_access("dc43d8d9-...")` (UUID or `req_...` form); with exactly one,
nothing else is needed. CTE names and table functions are recognized and left
alone, so normal SQL just works.

Prefer **bind parameters** over string formatting for values (safer, avoids SQL
injection):

```python
df = client.query(
    "SELECT * FROM person WHERE person_id = :pid",
    {"pid": 42},
)
```

Joins across a request's tables work normally — join on `person_id` **and**
`cohort_id`, since the same person_id can exist in different cohorts:

```python
df = client.query(
    "SELECT p.person_id, m.value_as_number FROM person p "
    "JOIN measurement m ON m.person_id = p.person_id AND m.cohort_id = p.cohort_id"
)
```

Access is **read-only and request-scoped** server-side — only relations granted
by an APPROVED access request resolve; everything else is rejected before it
touches the database. Concept-level row filters are baked into the
server-constructed queries, so they cannot be bypassed from SQL.

### pandas or polars

`query()` and `query_stream()` return **pandas** DataFrames by default and can
return **polars** instead — set a client-wide default at construction, or
override per call. Anything other than `"pandas"` or `"polars"` raises
`DataAccessError`.

```python
client = DataAccessClient(dataframe="polars")   # client-wide default
df = client.query("SELECT * FROM person")        # -> polars.DataFrame

client = DataAccessClient()                      # default: pandas
pdf = client.query("SELECT * FROM person", dataframe="polars")  # per-call override
```

The polars path converts fetched batches straight to Arrow instead of
materializing Python row objects, which makes it noticeably lighter for large
results. (The catalog helpers like `to_frame()` always return pandas;
`aggregate()` always returns polars.)

### Arrow-native ADBC

ADBC is integrated into the same `DataAccessClient`; it is not a separate SDK.
Select it for the normal `query()` and `query_stream()` methods:

```python
with DataAccessClient(driver="adbc", dataframe="polars") as client:
    frame = client.query("SELECT * FROM person LIMIT 100")
    datasets = client.aggregate(progress=True)
```

`aggregate()` also honors the selected driver. With ADBC, its long-format pull
streams Arrow record batches directly into the temporary Parquet spool instead
of materializing psycopg2 row objects. Its final dataset frames are still
Polars DataFrames.

For native Arrow results, use the ADBC methods directly:

```python
table = client.query_arrow("SELECT * FROM person LIMIT 100")

for batch in client.query_arrow_stream("SELECT * FROM measurement"):
    print(batch.num_rows, batch.schema)

with client.connect_adbc() as connection:
    objects = connection.adbc_get_objects(depth="columns").read_all()
```

The initial proxy support accepts non-parameterized ADBC reads. Queries with
bind parameters must use the default SQLAlchemy driver for now. The SDK safely
percent-encodes the short-lived bearer token into the libpq connection URI and
does not log that URI.

### Streaming large results in batches

`query()` returns the whole result as one DataFrame. For results too large to
materialize in one frame, `query_stream()` yields it in batches instead — rows
are fetched in `chunksize` batches over a plain forward-only cursor
(`fetchmany`; the proxy rejects true server-side cursors, so the driver still
buffers the raw result client-side), so peak DataFrame memory is bounded by one
chunk no matter how many rows the query returns:

```python
total = 0.0
for chunk in client.query_stream(
    "SELECT value_as_number FROM measurement", chunksize=50_000
):
    total += chunk["value_as_number"].sum()
```

The connection is held only while you iterate — it's released when the iterator
is exhausted, when you `break`/drop it, or when you call `.close()` on it. Bind
parameters work exactly as in `query()`.

---

## 3. Discover your access requests

`list_requests()` returns your access requests (any status) and what each
approved one makes queryable, as structured objects.

```python
requests = client.list_requests()

requests.ids                # ['dc43d8d9-6f6c-4152-b5f1-0f9668ba01e4', ...]
requests.username           # the authenticated user the catalog was built for
requests.approved           # only the requests that grant access
len(requests)               # number of requests
requests[0]                 # index it like a list
for r in requests:          # or iterate it
    print(r.id, r.status)
```

Flatten the whole catalog into a **pandas** DataFrame (one row per request
table; the catalog helpers are always pandas, whatever the `dataframe`
setting) — handy
for browsing in a notebook:

```python
df = requests.to_frame()
# columns: request_id, dataset, use_case, status, cohorts, table, relation
df.query("status == 'approved'")

requests.summary()
# one row per request: request_id, dataset, use_case, status, cohorts, tables (count)
# (also what a RequestList displays as in a Jupyter notebook)
```

Inspect a single request, its tables, and the concept grants behind it:

```python
request = requests.get("dc43d8d9-6f6c-4152-b5f1-0f9668ba01e4")  # UUID or req_... form

request.is_approved          # only approved requests expose tables
request.dataset              # e.g. 'AFT'
request.cohorts              # e.g. [101, 102]
request.table_names          # ['measurement', 'person', ...]
request.to_frame()           # tables as a pandas DataFrame
request.concepts_frame()     # concept grants: concept_id, cohort, source_field_name, ...
```

If you have exactly **one** approved request, every client method that takes
`request_id` lets you omit it; with several approved requests, pass it
explicitly.

---

## 4. Row counts

`row_counts()` returns `{table_name: row_count}` — a cheap `COUNT(*)` per table
(no row data transferred). Useful to size up a table before pulling it.

```python
client.row_counts()
# {'measurement': 12345, 'person': 110}

client.row_counts(tables=["person"])          # just the tables you care about
client.row_counts(request_id="dc43d8d9-...")  # a specific request
```

---

## 5. Raw SQLAlchemy connection

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

## 6. Tokens & auth introspection

Authentication is automatic and cached; an expired platform token is read again
from the token file on the next call. You rarely need these directly, but
they're available:

```python
client.login()              # force token loading now; returns the access token (str)

client.token_claims         # decoded claims of the current token (dict), e.g. exp/sub
```

Decode any JWT's claims without verifying its signature (module-level helper):

```python
from cbr_data_access import decode_token

claims = decode_token(some_token)   # dict of claims
```

The token is short-lived and is used as the database password; the client
rereads the platform token file as the cached token nears expiry.

---

## 7. Handling errors

Every error raised by the SDK derives from `DataAccessError`, so you can catch
that single base. Errors originating from an HTTP response also carry a
`.status_code`.

| Exception              | Raised when |
|------------------------|-------------|
| `DataAccessError`      | Base class for all SDK errors (also used for bad arguments). |
| `AuthenticationError`  | Platform token loading failed or a service rejected the token. |
| `ConnectionInfoError`  | The connection-info endpoint failed or returned nothing. |
| `RequestsError`        | The requests endpoint failed, a referenced request/table isn't in your catalog, or the request isn't approved. |
| `DBConnectionError`    | Establishing the PostgreSQL connection failed. |
| `QueryError`           | Executing a SQL query failed. |

```python
from cbr_data_access import DataAccessError

try:
    df = client.query("SELECT * FROM person")
except DataAccessError as exc:
    print(f"Data access failed: {exc}")
    print("HTTP status:", exc.status_code)   # None if not from an HTTP response
```

---

## 8. Configuration reference

Pass these to the constructor. By default, the client reads the platform token
file and uses the CBR deployment endpoints. Override settings only when targeting
a different deployment or local test environment.

| Argument          | Required | Default |
|-------------------|----------|---------|
| `token_file`      | no       | `CBR_TOKEN_FILE` or `/var/run/sandbox-connect/platform/token` |
| `base_url`        | no       | `http://127.0.0.1:6060` |
| `request_timeout` | no       | `300.0` (seconds — 5 minutes) |
| `dataframe`       | no       | `"pandas"` — what `query()`/`query_stream()` return; `"polars"` to switch (either call can override per invocation) |
| `driver`          | no       | `"sqlalchemy"`; use `"adbc"` for Arrow-native PostgreSQL reads |

`base_url` is the root of the data-access service; the connection-info
(`/postgres/connection-info`) and requests (`/requests`) endpoints hang off it.

```python
client = DataAccessClient(
    token_file="/tmp/cbr-token",
    base_url="https://data-access.example.org",
    request_timeout=120,        # override the per-request HTTP timeout
    dataframe="polars",         # make query()/query_stream() return polars
    driver="adbc",              # use Arrow-native PostgreSQL transport
)
```

---

## 9. Full API cheat-sheet

**`DataAccessClient`** (construct with optional `token_file`, `base_url`,
`request_timeout`, `dataframe`, and `driver`):

| Call | Returns | Description |
|------|---------|-------------|
| `set_access(request_id=None)` | `AccessRequest` | Pick which approved request to query (only needed with several; one is auto-resolved). |
| `clear_access()` | `None` | Forget the pick; the next query re-resolves automatically. |
| `default_request` | `AccessRequest \| None` | The current pick (property). |
| `query(sql, params=None, *, dataframe=None)` | `pandas.DataFrame` (default) or `polars.DataFrame` | Run a read query (the normal path); supports bind params. `dataframe="pandas"|"polars"` overrides the client default. |
| `query_stream(sql, params=None, *, chunksize=50_000, dataframe=None)` | `Iterator` of `pandas.DataFrame` (default) or `polars.DataFrame` | Run a read query and yield it in `chunksize`-row batches (forward-only cursor); same `dataframe` override. |
| `list_requests()` | `RequestList` | List your access requests and their queryable relations. |
| `row_counts(*, request_id=None, tables=None)` | `dict[str, int]` | `{table: COUNT(*)}` for all granted tables or a subset. |
| `aggregate(cohort=None, *, gender_male=8507, gender_female=8532, max_workers=1, progress=False)` | `dict[str, polars.DataFrame]` | Wide-format OMOP aggregate, one frame per dataset. Honors the client's SQLAlchemy/ADBC driver and always returns polars. |
| `connect()` | context mgr → `Connection` | Live SQLAlchemy connection (`with`-only); for raw access. |
| `connect_adbc()` | context mgr → ADBC `Connection` | Native ADBC DBAPI connection (`with`-only). |
| `query_arrow(sql)` | `pyarrow.Table` | Run a non-parameterized read through ADBC and return Arrow directly. |
| `query_arrow_stream(sql)` | `Iterator[pyarrow.RecordBatch]` | Stream native Arrow batches through ADBC. |
| `login()` | `str` | Force token loading; returns the access token. |
| `token_claims` | `dict` | Decoded claims of the current token (property). |
| `close()` | `None` | Dispose the connection pool (called automatically on `with` exit); not final — the client reconnects lazily on the next call. |

**Models** (`from cbr_data_access import AccessRequest, RequestConcept, RequestList, RequestTable`):

| Type | Attributes / methods |
|------|----------------------|
| `RequestList` | `.username`, `.requests`, `.ids`, `.approved`, `.get(request_id)`, `.to_frame()`, `.summary()`, iterable/indexable |
| `AccessRequest` | `.id`, `.schema`, `.dataset`, `.use_case`, `.status`, `.is_approved`, `.cohorts`, `.created_at`, `.tables`, `.table_names`, `.concepts`, `.get_table(name)`, `.to_frame()`, `.concepts_frame()` |
| `RequestTable` | `.name`, `.relation` |
| `RequestConcept` | `.concept_id`, `.cohort`, `.subgroup`, `.source_field_name`, `.source_field_description` |

The catalog frame methods — `to_frame()`, `summary()`, `concepts_frame()` —
always return **pandas** DataFrames, regardless of the client's `dataframe`
setting.

**Module-level:** `decode_token(token) -> dict`, `__version__`.

**Exceptions:** `DataAccessError` (base, `.status_code`), `AuthenticationError`,
`ConnectionInfoError`, `RequestsError`, `DBConnectionError`, `QueryError`.
