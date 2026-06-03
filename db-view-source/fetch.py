#!/usr/bin/env python3
"""
DB View Source - Custom Node Container

Connects to a PostgreSQL database, reads a specified view, and exports
the result to S3 as Parquet for downstream pipeline nodes.

Environment Variables:
  NODE_CONTEXT       - JSON object with structure:
                       {
                         "node": { "name": "fetch_users_view", "slug": "db-view-source" },
                         "config": {
                           "host": "localhost",
                           "port": 5432,
                           "database": "mydb",
                           "username": "postgres",
                           "password": "secret",
                           "view_name": "public.active_users"
                         },
                         "output": {
                           "basePath": "s3://bucket/artifacts/...",
                           "files": [{ "name": "output", "format": "parquet" }]
                         }
                       }

  S3_ENDPOINT        - MinIO endpoint
  S3_ACCESS_KEY      - S3 access key
  S3_SECRET_KEY      - S3 secret key
  S3_SESSION_TOKEN   - S3 session token (STS)
  S3_BUCKET          - MinIO bucket name
  S3_USE_SSL         - Use SSL for S3
  S3_REGION          - S3 region
"""

import os
import sys
import json
import duckdb


def log(message: str):
    print(f"[DB-VIEW-SOURCE] {message}", flush=True)


def log_error(message: str):
    print(f"[DB-VIEW-SOURCE ERROR] {message}", file=sys.stderr, flush=True)


def parse_node_context() -> dict:
    ctx_str = os.environ.get("NODE_CONTEXT", "")
    if not ctx_str:
        raise ValueError("NODE_CONTEXT environment variable is required")
    try:
        return json.loads(ctx_str)
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse NODE_CONTEXT: {e}")


def get_s3_config() -> dict:
    return {
        "endpoint": os.environ.get("S3_ENDPOINT", "minio:9000"),
        "access_key": os.environ.get("S3_ACCESS_KEY", ""),
        "secret_key": os.environ.get("S3_SECRET_KEY", ""),
        "session_token": os.environ.get("S3_SESSION_TOKEN", ""),
        "bucket": os.environ.get("S3_BUCKET", "data-pipeline"),
        "use_ssl": os.environ.get("S3_USE_SSL", "false").lower() == "true",
        "region": os.environ.get("S3_REGION", "us-east-1"),
    }


def validate_context(ctx: dict):
    config = ctx.get("config", {})
    required = ["host", "database", "username", "password", "view_name"]
    for field in required:
        if not config.get(field):
            raise ValueError(f"config.{field} is required in NODE_CONTEXT")
    if not ctx.get("output", {}).get("basePath"):
        raise ValueError("output.basePath is required in NODE_CONTEXT")
    if not ctx.get("output", {}).get("files"):
        raise ValueError("output.files is required in NODE_CONTEXT")


def main():
    ctx = parse_node_context()
    s3_config = get_s3_config()

    node = ctx["node"]
    config = ctx["config"]
    output = ctx["output"]

    node_name = node["name"]
    host = config["host"]
    port = int(config.get("port") or 5432)
    database = config["database"]
    username = config["username"]
    password = config["password"]
    view_name = config["view_name"]
    base_path = output["basePath"]
    output_files = output["files"]

    log("=== DB View Source Node ===")
    log(f"Node: {node_name}")
    log(f"Connecting to: {host}:{port}/{database} as {username}")
    log(f"View: {view_name}")

    validate_context(ctx)

    conn = duckdb.connect(":memory:")

    try:
        log("Loading extensions...")
        conn.load_extension("httpfs")
        conn.load_extension("postgres")

        log("Configuring S3...")
        session_token_clause = ""
        if s3_config["session_token"]:
            session_token_clause = f",\n                SESSION_TOKEN '{s3_config['session_token']}'"

        conn.execute(f"""
            CREATE SECRET minio_secret (
                TYPE S3,
                KEY_ID '{s3_config["access_key"]}',
                SECRET '{s3_config["secret_key"]}',
                ENDPOINT '{s3_config["endpoint"]}',
                URL_STYLE 'path',
                USE_SSL {str(s3_config["use_ssl"]).lower()},
                REGION '{s3_config["region"]}'{session_token_clause}
            )
        """)
        log("S3 configured")

        log(f"Attaching PostgreSQL database...")
        pg_conn_str = f"host={host} port={port} dbname={database} user={username} password={password}"
        conn.execute(f"ATTACH '{pg_conn_str}' AS pg_db (TYPE POSTGRES, READ_ONLY)")
        log("PostgreSQL attached")

        log(f"Reading view: {view_name}")
        conn.execute(f"CREATE TABLE source_data AS SELECT * FROM pg_db.{view_name}")
        result = conn.execute("SELECT COUNT(*) FROM source_data").fetchone()
        row_count = result[0] if result else 0
        log(f"Fetched {row_count} rows from view")

        for output_file in output_files:
            output_name = output_file["name"]
            output_format = output_file.get("format", "parquet")
            output_path = f"{base_path}/{output_name}.parquet"

            log(f"Exporting to {output_path}...")

            if output_format == "csv":
                conn.execute(f"COPY source_data TO '{output_path}' (FORMAT CSV, HEADER true)")
            elif output_format == "json":
                conn.execute(f"COPY source_data TO '{output_path}' (FORMAT JSON)")
            else:
                conn.execute(f"COPY source_data TO '{output_path}' (FORMAT PARQUET, COMPRESSION 'snappy')")

            log(f"Exported {row_count} rows")

        log("=== DB View Source Complete ===")
        print(json.dumps({
            "success": True,
            "nodeName": node_name,
            "rowCount": row_count,
            "outputs": [
                {
                    "name": f["name"],
                    "path": f"{base_path}/{f['name']}.parquet",
                    "format": f.get("format", "parquet"),
                }
                for f in output_files
            ],
        }))

    except Exception as e:
        log_error(f"Failed: {e}")
        print(json.dumps({"success": False, "nodeName": node_name, "error": str(e)}))
        sys.exit(1)
    finally:
        conn.close()


if __name__ == "__main__":
    main()
