#!/usr/bin/env python3
"""
SQL Transformer - Parquet Only

Executes SQL transformations using DuckDB on Parquet files.
Reads Parquet from MinIO, runs SQL, writes Parquet back to MinIO.

Environment Variables:
  NODE_CONTEXT       - JSON object with node configuration
  S3_ENDPOINT        - MinIO endpoint
  S3_ACCESS_KEY      - S3 access key
  S3_SECRET_KEY      - S3 secret key
  S3_SESSION_TOKEN   - S3 session token (optional)
  S3_BUCKET          - MinIO bucket name
  S3_USE_SSL         - Use SSL for S3
  S3_REGION          - S3 region
"""

import os
import sys
import json
import duckdb


def log(message: str):
    print(f"[TRANSFORM] {message}", flush=True)


def log_error(message: str):
    print(f"[TRANSFORM ERROR] {message}", file=sys.stderr, flush=True)


def sanitize_table_name(name: str) -> str:
    """Sanitize a string for use as a SQL table name."""
    return "".join(c if c.isalnum() or c == "_" else "_" for c in name)


def parse_node_context() -> dict:
    """Parse NODE_CONTEXT from environment."""
    ctx_str = os.environ.get("NODE_CONTEXT", "")
    if not ctx_str:
        raise ValueError("NODE_CONTEXT environment variable is required")
    
    try:
        return json.loads(ctx_str)
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse NODE_CONTEXT: {e}")


def get_s3_config() -> dict:
    """Get S3 configuration from environment variables."""
    return {
        "endpoint": os.environ.get("S3_ENDPOINT", "minio:9000"),
        "access_key": os.environ.get("S3_ACCESS_KEY", ""),
        "secret_key": os.environ.get("S3_SECRET_KEY", ""),
        "session_token": os.environ.get("S3_SESSION_TOKEN", ""),
        "bucket": os.environ.get("S3_BUCKET", "data-pipeline"),
        "use_ssl": os.environ.get("S3_USE_SSL", "false").lower() == "true",
        "region": os.environ.get("S3_REGION", "us-east-1"),
    }


def main():
    # Parse context
    ctx = parse_node_context()
    s3_config = get_s3_config()
    
    node = ctx["node"]
    inputs = ctx["inputs"]
    output = ctx["output"]
    config = ctx["config"]
    
    node_name = node["name"]
    sql_query = config["sql"]
    base_path = output["basePath"]
    output_files = output["files"]
    
    log("=== SQL Transformer ===")
    log(f"Node: {node_name}")
    log(f"Inputs: {len(inputs)}")
    
    # Create in-memory DuckDB connection
    conn = duckdb.connect(":memory:")
    
    try:
        # Load httpfs extension
        log("Loading httpfs extension...")
        conn.load_extension("httpfs")
        
        # Configure S3
        log("Configuring S3...")
        session_token_clause = ""
        if s3_config["session_token"]:
            session_token_clause = f",\n                SESSION_TOKEN '{s3_config['session_token']}'"
        
        create_secret_sql = f"""
            CREATE SECRET minio_secret (
                TYPE S3,
                KEY_ID '{s3_config["access_key"]}',
                SECRET '{s3_config["secret_key"]}',
                ENDPOINT '{s3_config["endpoint"]}',
                URL_STYLE 'path',
                USE_SSL {str(s3_config["use_ssl"]).lower()},
                REGION '{s3_config["region"]}'{session_token_clause}
            )
        """
        conn.execute(create_secret_sql)
        
        # Load all inputs as tables (Parquet only)
        log("Loading inputs...")
        for inp in inputs:
            table_name = sanitize_table_name(inp["nodeName"])
            path = inp["output"]["path"]
            
            log(f"  Loading '{table_name}' from {path}")
            conn.execute(f"CREATE TABLE {table_name} AS SELECT * FROM read_parquet('{path}')")
            
            result = conn.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()
            log(f"  Loaded {result[0] if result else 0} rows")
        
        # Create input_data alias if single input
        if len(inputs) == 1:
            table_name = sanitize_table_name(inputs[0]["nodeName"])
            conn.execute(f"CREATE VIEW input_data AS SELECT * FROM {table_name}")
        
        # Execute SQL transformation
        log("Executing SQL...")
        result_table = sanitize_table_name(node_name)
        conn.execute(f"CREATE TABLE {result_table} AS {sql_query}")
        
        result = conn.execute(f"SELECT COUNT(*) FROM {result_table}").fetchone()
        row_count = result[0] if result else 0
        log(f"Result: {row_count} rows")
        
        # Export results (Parquet only)
        outputs = []
        for output_file in output_files:
            output_name = output_file["name"]
            output_path = f"{base_path}/{output_name}.parquet"
            
            log(f"Exporting to {output_path}...")
            conn.execute(f"COPY {result_table} TO '{output_path}' (FORMAT PARQUET, COMPRESSION 'snappy')")
            
            outputs.append({
                "name": output_name,
                "path": output_path,
                "format": "parquet",
            })
        
        log("=== Complete ===")
        
        print(json.dumps({
            "success": True,
            "nodeName": node_name,
            "rowCount": row_count,
            "outputs": outputs,
        }))
        
    except Exception as e:
        log_error(f"Failed: {e}")
        print(json.dumps({
            "success": False,
            "nodeName": node_name,
            "error": str(e),
        }))
        sys.exit(1)
    finally:
        conn.close()


if __name__ == "__main__":
    main()
