#!/usr/bin/env python3
"""
DB View Source - Custom Node Container

Authenticates against Keycloak using provided credentials, then connects to the
Keycloak PostgreSQL database, reads a specified view, and exports the result to
S3 as Parquet for downstream pipeline nodes.

Environment Variables:
  NODE_CONTEXT       - JSON object with structure:
                       {
                         "node": { "name": "fetch_users_view", "slug": "db-view-source" },
                         "config": {
                           "username": "admin",
                           "password": "admin",
                           "view_name": "public.user_entity"
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

  KEYCLOAK_URL       - Keycloak service URL (default: http://keycloak.data-pipeline.svc.cluster.local:8080)
  KEYCLOAK_REALM     - Keycloak realm (default: data-pipeline-builder)
  KC_DB_HOST         - Keycloak PostgreSQL host (default: postgres.data-pipeline.svc.cluster.local)
  KC_DB_PORT         - Keycloak PostgreSQL port (default: 5432)
  KC_DB_NAME         - Keycloak database name (default: keycloak)
  KC_DB_USER         - Keycloak database user (default: keycloak)
  KC_DB_PASSWORD     - Keycloak database password (default: keycloak)
"""

import os
import sys
import json
import duckdb
import requests


KEYCLOAK_URL = os.environ.get("KEYCLOAK_URL", "http://keycloak.data-pipeline.svc.cluster.local:8080")
KEYCLOAK_REALM = os.environ.get("KEYCLOAK_REALM", "data-pipeline-builder")
KC_DB_HOST = os.environ.get("KC_DB_HOST", "postgres.data-pipeline.svc.cluster.local")
KC_DB_PORT = int(os.environ.get("KC_DB_PORT", "5432"))
KC_DB_NAME = os.environ.get("KC_DB_NAME", "keycloak")
KC_DB_USER = os.environ.get("KC_DB_USER", "keycloak")
KC_DB_PASSWORD = os.environ.get("KC_DB_PASSWORD", "keycloak")


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
    required = ["username", "password", "view_name"]
    for field in required:
        if not config.get(field):
            raise ValueError(f"config.{field} is required in NODE_CONTEXT")
    if not ctx.get("output", {}).get("basePath"):
        raise ValueError("output.basePath is required in NODE_CONTEXT")
    if not ctx.get("output", {}).get("files"):
        raise ValueError("output.files is required in NODE_CONTEXT")


def authenticate_keycloak(username: str, password: str) -> str:
    """Authenticate against Keycloak and return an access token."""
    token_url = f"{KEYCLOAK_URL}/realms/{KEYCLOAK_REALM}/protocol/openid-connect/token"
    response = requests.post(
        token_url,
        data={
            "grant_type": "password",
            "client_id": "admin-cli",
            "username": username,
            "password": password,
        },
        timeout=15,
    )
    if response.status_code != 200:
        raise ValueError(
            f"Keycloak authentication failed for user '{username}': "
            f"{response.status_code} {response.json().get('error_description', response.text)}"
        )
    return response.json()["access_token"]


def main():
    ctx = parse_node_context()
    s3_config = get_s3_config()

    node = ctx["node"]
    config = ctx["config"]
    output = ctx["output"]

    node_name = node["name"]
    username = config["username"]
    password = config["password"]
    view_name = config["view_name"]
    base_path = output["basePath"]
    output_files = output["files"]

    log("=== DB View Source Node ===")
    log(f"Node: {node_name}")

    validate_context(ctx)

    log(f"Authenticating with Keycloak as '{username}'...")
    authenticate_keycloak(username, password)
    log("Keycloak authentication successful")

    log(f"Connecting to Keycloak database: {KC_DB_HOST}:{KC_DB_PORT}/{KC_DB_NAME}")
    log(f"View: {view_name}")

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

        log("Attaching Keycloak PostgreSQL database...")
        pg_conn_str = (
            f"host={KC_DB_HOST} port={KC_DB_PORT} "
            f"dbname={KC_DB_NAME} user={KC_DB_USER} password={KC_DB_PASSWORD}"
        )
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
