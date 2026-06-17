#!/usr/bin/env python3
"""
DB Table Source - Custom Node Container

Uses the cbr-data-access SDK to authenticate with Keycloak, download a
table from the caller's approved access request on the CBR data-access
server, and upload the result as Parquet to MinIO so downstream pipeline
nodes can consume it as an artifact.
"""

import json
import os
import re
import sys
import tempfile
import time

import boto3
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from boto3.s3.transfer import TransferConfig
from botocore.client import Config
from cbr_data_access import DataAccessClient
from cbr_data_access.exceptions import DataAccessError


def log(msg):
    print(f"[DB-TABLE-SOURCE] {msg}", flush=True)


def log_error(msg):
    print(f"[DB-TABLE-SOURCE ERROR] {msg}", file=sys.stderr, flush=True)


def parse_context():
    raw = os.environ.get("NODE_CONTEXT", "")
    if not raw:
        raise ValueError("NODE_CONTEXT is required")
    return json.loads(raw)


def s3_key_from_path(s3_path):
    # s3://bucket/some/key  →  some/key
    return s3_path.replace("s3://", "").split("/", 1)[1]


def main():
    ctx = parse_context()
    config   = ctx["config"]
    node     = ctx["node"]
    output   = ctx["output"]

    username   = config["username"]
    password   = config["password"]
    table_name = config["table_name"].strip()
    view_id    = config.get("view_id", "").strip() or None
    base_path  = output["basePath"]
    out_files  = output["files"]

    log(f"Node: {node['name']} | Table: {table_name} | User: {username} | View ID: {view_id or 'auto'}")

    # Optional endpoint overrides (pass as pod env vars if needed)
    sdk_kwargs = {}
    if os.environ.get("CBR_KEYCLOAK_URL"):
        sdk_kwargs["keycloak_url"] = os.environ["CBR_KEYCLOAK_URL"]
    if os.environ.get("CBR_BASE_URL"):
        sdk_kwargs["base_url"] = os.environ["CBR_BASE_URL"]

    s3 = boto3.client(
        "s3",
        endpoint_url="{}://{}".format(
            "https" if os.environ.get("S3_USE_SSL", "false").lower() == "true" else "http",
            os.environ.get("S3_ENDPOINT", "minio:9000"),
        ),
        aws_access_key_id=os.environ.get("S3_ACCESS_KEY", ""),
        aws_secret_access_key=os.environ.get("S3_SECRET_KEY", ""),
        aws_session_token=os.environ.get("S3_SESSION_TOKEN") or None,
        region_name=os.environ.get("S3_REGION", "us-east-1"),
        config=Config(signature_version="s3v4"),
    )
    bucket = os.environ.get("S3_BUCKET", "data-pipeline")

    try:
        with DataAccessClient(username=username, password=password, **sdk_kwargs) as client:
            log("Authenticating with Keycloak...")
            client.login()
            log("Authenticated")

            if view_id:
                log(f"Setting access request: {view_id}")
                client.set_access(view_id)

            # query_stream interpolates table_name into SQL. node.json's contract
            # is a bare table name, so validate it as a plain identifier first.
            if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", table_name):
                raise ValueError(f"Invalid table_name: {table_name!r}")

            sql = f"SELECT * FROM {table_name}"  # auto-qualified by query_stream
            first_key = s3_key_from_path(f"{base_path}/{out_files[0]['name']}.parquet")

            # Cheap LIMIT 0 probe so an empty table still yields a valid
            # header-only Parquet (mirrors the SDK's own download probe).
            col_names = list(client.query(sql + " LIMIT 0").columns)

            with tempfile.TemporaryDirectory() as tmp:
                local_path = os.path.join(tmp, f"{out_files[0]['name']}.parquet")

                # Phase 1: stream the table to a local Parquet at full DB speed.
                # Writing to disk (not a pipe) means the server-side cursor is
                # never backpressured by the upload. Timing logs make the read
                # cost visible (it was hidden before).
                t0 = time.monotonic()
                rows_total, next_log, writer, first = 0, 250_000, None, True
                for chunk in client.query_stream(sql):
                    if first:
                        log(f"First DB chunk after {time.monotonic() - t0:.1f}s")
                        first = False
                    tbl = pa.Table.from_pandas(chunk, preserve_index=False)
                    if writer is None:
                        writer = pq.ParquetWriter(local_path, tbl.schema)
                        writer.write_table(tbl)
                    else:
                        # Pin schema so later chunks can't drift dtypes.
                        writer.write_table(tbl.cast(writer.schema))
                    rows_total += len(chunk)
                    if rows_total >= next_log:
                        log(f"  read {rows_total:,} rows, "
                            f"{os.path.getsize(local_path) / 1e6:.1f} MB, "
                            f"{time.monotonic() - t0:.1f}s")
                        next_log += 250_000
                if writer is None:  # empty table -> header-only Parquet
                    pd.DataFrame(columns=col_names).to_parquet(local_path, index=False)
                else:
                    writer.close()
                size_mb = os.path.getsize(local_path) / 1e6
                log(f"DB read+encode done: {rows_total:,} rows, {size_mb:.1f} MB "
                    f"in {time.monotonic() - t0:.1f}s")

                # Phase 2: upload the seekable file with parallel multipart.
                t1 = time.monotonic()
                transfer = TransferConfig(
                    multipart_threshold=8 * 1024 * 1024,
                    multipart_chunksize=8 * 1024 * 1024,
                    max_concurrency=8,
                    use_threads=True,
                )
                log(f"Uploading {size_mb:.1f} MB -> s3://{bucket}/{first_key} ...")
                s3.upload_file(local_path, bucket, first_key, Config=transfer)
                dt = time.monotonic() - t1
                log(f"Upload done in {dt:.1f}s ({size_mb / max(dt, 1e-3):.1f} MB/s)")

                # Additional outputs share the same bytes: server-side copy,
                # never re-stream the table.
                for f in out_files[1:]:
                    extra_key = s3_key_from_path(f"{base_path}/{f['name']}.parquet")
                    log(f"Copying to s3://{bucket}/{extra_key} ...")
                    s3.copy({"Bucket": bucket, "Key": first_key}, bucket, extra_key)

        print(json.dumps({
            "success": True,
            "nodeName": node["name"],
            "tableName": table_name,
            "outputs": [
                {"name": f["name"], "path": f"{base_path}/{f['name']}.parquet"}
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
