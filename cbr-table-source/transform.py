#!/usr/bin/env python3
"""
DB Table Source - Custom Node Container

Uses the cbr-data-access SDK to authenticate with Keycloak, download a
table from the caller's approved access request on the CBR data-access
server, and upload the result as Parquet to MinIO so downstream pipeline
nodes can consume it as an artifact.

Authentication reuses the caller's own Keycloak session instead of asking
for a username/password in the node config: Argo injects KEYCLOAK_ACCESS_TOKEN
/ KEYCLOAK_REFRESH_TOKEN / KEYCLOAK_TOKEN_URL into every node pod (see the
"Auth Check" node), so this node seeds the SDK client with that access token
directly and refreshes it via the refresh-token grant if it expires mid-run.
"""

import json
import os
import re
import sys
import tempfile
import time

import boto3
import requests
from boto3.s3.transfer import TransferConfig
from botocore.client import Config
from cbr_data_access import DataAccessClient, decode_token
from cbr_data_access.exceptions import AuthenticationError, DataAccessError

# Same Keycloak env vars the "Auth Check" node verifies are reachable in the pod.
KEYCLOAK_CLIENT_ID = os.environ.get("KEYCLOAK_CLIENT_ID", "angular-client")

# Bump this on every code change so a run's logs prove which build is live.
NODE_VERSION = "2026-06-30.1-omop-contract-keys"


def log(msg):
    print(f"[DB-TABLE-SOURCE] {msg}", flush=True)


def log_error(msg):
    print(f"[DB-TABLE-SOURCE ERROR] {msg}", file=sys.stderr, flush=True)


def parse_context():
    raw = os.environ.get("NODE_CONTEXT", "")
    if not raw:
        raise ValueError("NODE_CONTEXT is required")
    return json.loads(raw)


def split_s3(s3_path):
    # s3://bucket/some/key  →  ("bucket", "some/key")
    rest = s3_path[len("s3://"):] if s3_path.startswith("s3://") else s3_path
    bucket, _, key = rest.partition("/")
    return bucket, key


def parse_row_limit(raw):
    """Validate the optional row cap. Blank/None/0 means no limit (full table)."""
    if raw is None or (isinstance(raw, str) and not raw.strip()):
        return None
    try:
        n = int(str(raw).strip())
    except (TypeError, ValueError):
        raise ValueError(f"Invalid row_limit: {raw!r} (must be a positive integer)")
    if n < 0:
        raise ValueError(f"row_limit must be >= 0, got {n}")
    return n or None  # 0 => no limit


def resolve_request_id(config):
    """Extract the access request id chosen in the editor.

    The system "OMOP Table Source" contract stores the editor selection under
    `access_request` as a composite `{ "request_id": ..., "use_case": ... }`
    (see the access-request field type). The Argo step needs the request id,
    not the use case. Falls back to the legacy bare `view_id` key.
    """
    access_request = config.get("access_request")
    if isinstance(access_request, dict):
        request_id = (access_request.get("request_id") or "").strip()
        if request_id:
            return request_id
    # Legacy / manual config: a bare access-request UUID.
    return (config.get("view_id") or "").strip() or None


def _refresh_via_argo_token(client, token_url, timeout):
    """Bound onto the client as `login`: refresh KEYCLOAK_REFRESH_TOKEN instead
    of doing a password grant, since this node has no username/password."""
    refresh_token = os.environ.get("KEYCLOAK_REFRESH_TOKEN")
    if not refresh_token:
        raise AuthenticationError("Access token expired and KEYCLOAK_REFRESH_TOKEN is not set")

    data = {
        "grant_type": "refresh_token",
        "client_id": KEYCLOAK_CLIENT_ID,
        "refresh_token": refresh_token,
    }
    headers = {"Content-Type": "application/x-www-form-urlencoded"}
    try:
        response = requests.post(token_url, data=data, headers=headers, timeout=timeout)
    except requests.RequestException as exc:
        raise AuthenticationError(f"Keycloak refresh request failed: {exc}") from exc
    if response.status_code != 200:
        raise AuthenticationError(response.text, status_code=response.status_code)

    token = response.json().get("access_token")
    if not token:
        raise AuthenticationError("No access_token in Keycloak refresh response")

    client._access_token = token
    client._token_exp = client._read_exp(token)
    return token


def build_authenticated_client(**sdk_kwargs):
    """Build a DataAccessClient from the Keycloak tokens Argo injects into the
    pod, instead of a username/password stored in the pipeline config."""
    access_token = os.environ.get("KEYCLOAK_ACCESS_TOKEN")
    token_url = os.environ.get("KEYCLOAK_TOKEN_URL")
    if not access_token:
        raise AuthenticationError("KEYCLOAK_ACCESS_TOKEN is not set in this pod")

    claims = decode_token(access_token)
    display_user = claims.get("preferred_username") or claims.get("email") or claims.get("sub")
    log(f"Using injected Keycloak session for: {display_user}")

    # username/password are unused (auth is seeded below) but the constructor
    # requires non-empty strings.
    client = DataAccessClient(username=str(display_user), password="argo-token-auth", **sdk_kwargs)
    client._access_token = access_token
    client._token_exp = client._read_exp(access_token)

    if token_url:
        client.login = lambda: _refresh_via_argo_token(client, token_url, client._timeout)

    return client


def main():
    log(f"version {NODE_VERSION}")
    ctx = parse_context()
    config   = ctx["config"]
    node     = ctx["node"]
    output   = ctx["output"]

    # The system "OMOP Table Source" contract sends `omop_table`; accept the
    # legacy `table_name` key too so manually-wired configs keep working.
    table_name = (config.get("omop_table") or config.get("table_name") or "").strip()
    if not table_name:
        raise ValueError("omop_table is required (select an OMOP table in the editor)")
    view_id    = resolve_request_id(config)
    row_limit  = parse_row_limit(config.get("row_limit"))
    out_files  = output["files"]

    log(f"Node: {node['name']} | Table: {table_name} | "
        f"View ID: {view_id or 'auto'} | Limit: {row_limit or 'none'}")

    # Optional endpoint overrides (pass as pod env vars if needed)
    sdk_kwargs = {}
    if os.environ.get("CBR_KEYCLOAK_URL"):
        sdk_kwargs["keycloak_url"] = os.environ["CBR_KEYCLOAK_URL"]
    if os.environ.get("CBR_BASE_URL"):
        sdk_kwargs["base_url"] = os.environ["CBR_BASE_URL"]

    # Some S3-compatible endpoints (older MinIO/Ceph) reject the request
    # checksums newer botocore adds by default. Only send them when the
    # operation actually requires it. The env form is ignored by botocore
    # versions that predate the option, so it is safe across boto3 versions.
    os.environ.setdefault("AWS_REQUEST_CHECKSUM_CALCULATION", "when_required")
    os.environ.setdefault("AWS_RESPONSE_CHECKSUM_VALIDATION", "when_required")

    # This node only writes its result, so it uses the artifact bucket (the
    # read+write output bucket). The platform injects ARTIFACT_S3_* — the old
    # single-bucket S3_* vars are no longer set.
    endpoint = os.environ["ARTIFACT_S3_ENDPOINT"]
    use_ssl = os.environ.get("ARTIFACT_S3_USE_SSL", "false").lower() == "true"
    if "://" not in endpoint:
        endpoint = ("https://" if use_ssl else "http://") + endpoint
    s3 = boto3.client(
        "s3",
        endpoint_url=endpoint,
        aws_access_key_id=os.environ["ARTIFACT_S3_ACCESS_KEY"],
        aws_secret_access_key=os.environ["ARTIFACT_S3_SECRET_KEY"],
        aws_session_token=os.environ.get("ARTIFACT_S3_SESSION_TOKEN") or None,
        region_name=os.environ.get("ARTIFACT_S3_REGION", "us-east-1"),
        config=Config(signature_version="s3v4"),
    )

    try:
        with build_authenticated_client(**sdk_kwargs) as client:
            log("Authenticated using injected Keycloak session")

            if view_id:
                log(f"Setting access request: {view_id}")
                client.set_access(view_id)

            # query() interpolates table_name into SQL via auto-qualification.
            # node.json's contract is a bare table name, so validate it as a
            # plain identifier first.
            if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", table_name):
                raise ValueError(f"Invalid table_name: {table_name!r}")

            sql = f"SELECT * FROM {table_name}"  # auto-qualified by the SDK
            if row_limit:
                sql += f" LIMIT {row_limit}"     # row_limit is a validated int
            # Write to the exact path the platform assigned (always artifact bucket).
            bucket, first_key = split_s3(out_files[0]["path"])

            # Parquet codec. snappy is ~5-10x cheaper to encode than zstd and
            # the output stays small. Override with PARQUET_COMPRESSION
            # (e.g. "zstd", "none").
            compression = os.environ.get("PARQUET_COMPRESSION", "snappy").strip().lower()
            if compression in ("", "none"):
                compression = None
            log(f"Parquet compression: {compression or 'none'}")

            with tempfile.TemporaryDirectory() as tmp:
                local_path = os.path.join(tmp, f"{out_files[0]['name']}.parquet")

                # Phase 1: fetch the (row-capped) result set as a single
                # DataFrame, then encode Parquet once. The row limit keeps this
                # memory-bounded, and read/encode are no longer interleaved, so
                # the encode never back-pressures the DB read.
                t0 = time.monotonic()
                log(f"Querying: {sql}")
                df = client.query(sql)
                log(f"DB read complete: {len(df):,} rows in "
                    f"{time.monotonic() - t0:.1f}s; encoding Parquet")

                t_enc = time.monotonic()
                df.to_parquet(local_path, index=False, compression=compression)
                size_mb = os.path.getsize(local_path) / 1e6
                log(f"Parquet written: {len(df):,} rows, {size_mb:.1f} MB on disk in "
                    f"{time.monotonic() - t_enc:.1f}s "
                    f"(DB+encode total {time.monotonic() - t0:.1f}s); uploading next")

                # Phase 2: upload the seekable file.
                # This S3 endpoint doesn't support multipart uploads, so force a
                # single PUT by raising the threshold above any realistic size.
                # upload_file streams the body straight from disk, so even a large
                # single PUT never buffers the whole table in memory.
                t1 = time.monotonic()
                transfer = TransferConfig(
                    multipart_threshold=5 * 1024 * 1024 * 1024,  # 5 GiB: never split
                    use_threads=False,
                )
                log(f"Uploading {size_mb:.1f} MB -> s3://{bucket}/{first_key} ...")
                s3.upload_file(local_path, bucket, first_key, Config=transfer)
                dt = time.monotonic() - t1
                log(f"Upload done in {dt:.1f}s ({size_mb / max(dt, 1e-3):.1f} MB/s)")

                # Additional outputs share the same bytes: single-request
                # server-side copy (copy_object, not the managed multipart copy).
                for f in out_files[1:]:
                    _, extra_key = split_s3(f["path"])
                    log(f"Copying to s3://{bucket}/{extra_key} ...")
                    s3.copy_object(
                        Bucket=bucket,
                        CopySource={"Bucket": bucket, "Key": first_key},
                        Key=extra_key,
                    )

        log(f"Completed OK (version {NODE_VERSION})")
        print(json.dumps({
            "success": True,
            "version": NODE_VERSION,
            "nodeName": node["name"],
            "tableName": table_name,
            "outputs": [
                {"name": f["name"], "path": f["path"]}
                for f in out_files
            ],
        }))

    except DataAccessError as e:
        log_error(str(e))
        print(json.dumps({"success": False, "nodeName": node["name"], "error": str(e)}))
        sys.exit(1)
    except Exception as e:
        log_error(str(e))
        print(json.dumps({"success": False, "nodeName": node["name"], "error": str(e)}))
        sys.exit(1)


if __name__ == "__main__":
    main()
