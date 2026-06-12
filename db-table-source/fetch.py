#!/usr/bin/env python3
"""
DB Table Source - Custom Node Container

Uses the cbr-data-access SDK to authenticate with Keycloak, run a query
against the caller's approved access request on the CBR data-access server,
and upload the result as Parquet to MinIO so downstream pipeline nodes can
consume it as an artifact.
"""

import json
import os
import sys
import tempfile

import boto3
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
    table_name = (config.get("table_name") or "").strip()
    sql_query  = (config.get("sql_query") or "").strip()
    request_id = (config.get("request_id") or "").strip() or None
    base_path  = output["basePath"]
    out_files  = output["files"]

    if not table_name and not sql_query:
        raise ValueError("Either 'table_name' or 'sql_query' must be provided")

    log(f"Node: {node['name']} | Table: {table_name or '(custom SQL)'} | User: {username}")

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

            if request_id:
                log(f"Pinning access request: {request_id}")
                client.set_access(request_id)

            with tempfile.TemporaryDirectory() as tmp:
                if sql_query:
                    log("Running custom SQL query...")
                    df = client.query(sql_query)
                    local_path = os.path.join(tmp, "output.parquet")
                    df.to_parquet(local_path, index=False)
                    log(f"Query returned {len(df)} rows -> {local_path}")
                else:
                    log(f"Downloading table '{table_name}'...")
                    local_path = client.download_table(table_name, tmp, file_format="parquet")
                    log(f"Downloaded -> {local_path}")

                for f in out_files:
                    s3_key = s3_key_from_path(f"{base_path}/{f['name']}.parquet")
                    log(f"Uploading to s3://{bucket}/{s3_key} ...")
                    s3.upload_file(str(local_path), bucket, s3_key)
                    log("Upload complete")

        print(json.dumps({
            "success": True,
            "nodeName": node["name"],
            "tableName": table_name or None,
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
