#!/usr/bin/env python3
"""
SQL Transformer - Custom Node Container

Executes SQL transformations using DuckDB and exports results to MinIO.
Follows the NodeContext contract - receives a single NODE_CONTEXT JSON.

Environment Variables:
  NODE_CONTEXT       - JSON object with structure:
                       {
                         "node": { "name": "my_transform", "slug": "sql-transformer" },
                         "inputs": [
                           { "nodeSlug": "csv-source", "nodeName": "orders", 
                             "output": { "name": "output", "path": "s3://...", "format": "csv" } }
                         ],
                         "output": {
                           "basePath": "s3://bucket/user/artifacts/flow/exec/nodeName",
                           "files": [{ "name": "result", "format": "parquet" }]
                         },
                         "config": { "sql": "SELECT * FROM ...", ... }
                       }

Resource Environment Variables (always injected for all nodes):
  S3_ENDPOINT        - MinIO endpoint (e.g., minio:9000)
  S3_ACCESS_KEY      - S3 access key (STS temporary credential)
  S3_SECRET_KEY      - S3 secret key (STS temporary credential)
  S3_SESSION_TOKEN   - S3 session token (STS temporary credential)
  S3_BUCKET          - MinIO bucket name (default: data-pipeline)
  S3_USE_SSL         - Use SSL for S3 (default: false)
  S3_REGION          - S3 region (default: us-east-1)
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


def get_extension(format: str) -> str:
    """Get file extension for a format."""
    return {"parquet": "parquet", "csv": "csv", "json": "json"}.get(format.lower(), "parquet")


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


def validate_context(ctx: dict):
    """Validate required fields in NodeContext."""
    if not ctx.get("node", {}).get("name"):
        raise ValueError("node.name is required in NODE_CONTEXT")
    
    config = ctx.get("config", {})
    if not config.get("sql"):
        raise ValueError("config.sql is required in NODE_CONTEXT")
    
    if not ctx.get("inputs"):
        raise ValueError("At least one input is required in NODE_CONTEXT")
    
    if not ctx.get("output", {}).get("basePath"):
        raise ValueError("output.basePath is required in NODE_CONTEXT")
    
    if not ctx.get("output", {}).get("files"):
        raise ValueError("output.files is required in NODE_CONTEXT")


def main():
    # Parse context
    ctx = parse_node_context()
    s3_config = get_s3_config()
    
    # Extract context fields
    node = ctx["node"]
    inputs = ctx["inputs"]
    output = ctx["output"]
    config = ctx["config"]
    
    node_name = node["name"]
    node_slug = node["slug"]
    sql_query = config["sql"]
    base_path = output["basePath"]
    output_files = output["files"]
    
    log("=== SQL Transformer Node ===")
    log(f"Node: {node_name} ({node_slug})")
    log(f"Inputs: {len(inputs)}")
    log(f"Outputs: {len(output_files)}")
    
    # Log inputs
    for inp in inputs:
        log(f"  Input: {inp['nodeName']} ({inp['nodeSlug']}) -> {inp['output']['format']}")
    
    validate_context(ctx)
    
    # Create in-memory DuckDB connection
    conn = duckdb.connect(":memory:")
    
    try:
        # Load httpfs extension
        log("Loading httpfs extension...")
        conn.load_extension("httpfs")
        
        # Configure S3
        log("Configuring S3 connection...")
        session_token_clause = ""
        if s3_config["session_token"]:
            session_token_clause = f",\n                SESSION_TOKEN '{s3_config['session_token']}'"
            log("Using STS temporary credentials")
        
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
        log("S3 secret configured successfully")
        
        # Load all inputs as tables
        log("Loading inputs...")
        for inp in inputs:
            # Use nodeName as table name (sanitized)
            table_name = sanitize_table_name(inp["nodeName"])
            path = inp["output"]["path"]
            format = inp["output"].get("format", "parquet").lower()
            
            log(f"  Loading '{table_name}' from {inp['nodeName']} (format: {format})")
            
            # Determine read function based on format
            if format == "csv":
                read_func = "read_csv_auto"
            elif format == "json":
                read_func = "read_json_auto"
            else:
                read_func = "read_parquet"
            
            load_sql = f"CREATE TABLE {table_name} AS SELECT * FROM {read_func}('{path}')"
            conn.execute(load_sql)
            
            result = conn.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()
            count = result[0] if result else 0
            log(f"  Loaded {count} rows into '{table_name}'")
        
        # Create input_data alias if single input
        if len(inputs) == 1:
            table_name = sanitize_table_name(inputs[0]["nodeName"])
            log(f"Creating 'input_data' alias for '{table_name}'")
            conn.execute(f"CREATE VIEW input_data AS SELECT * FROM {table_name}")
        
        # Execute SQL transformation
        log("Executing SQL transformation...")
        result_table = sanitize_table_name(node_name)
        create_table_sql = f"CREATE TABLE {result_table} AS {sql_query}"
        conn.execute(create_table_sql)
        
        result = conn.execute(f"SELECT COUNT(*) FROM {result_table}").fetchone()
        row_count = result[0] if result else 0
        log(f"Transformation produced {row_count} rows")
        
        # Export results for each output file
        for output_file in output_files:
            output_name = output_file["name"]
            output_format = output_file["format"]
            output_path = f"{base_path}/{output_name}.{get_extension(output_format)}"
            
            log(f"Exporting '{output_name}' to {output_path}...")
            
            # Get format options
            if output_format == "parquet":
                format_options = "FORMAT PARQUET, COMPRESSION 'snappy'"
            elif output_format == "csv":
                format_options = "FORMAT CSV, HEADER true"
            elif output_format == "json":
                format_options = "FORMAT JSON"
            else:
                format_options = "FORMAT PARQUET"
            
            export_sql = f"COPY {result_table} TO '{output_path}' ({format_options})"
            conn.execute(export_sql)
            log(f"  Exported {row_count} rows to {output_path}")
        
        log("=== Transformer Node Complete ===")
        
        # Output result as JSON
        result_json = {
            "success": True,
            "nodeName": node_name,
            "rowCount": row_count,
            "outputs": [
                {
                    "name": f["name"],
                    "path": f"{base_path}/{f['name']}.{get_extension(f['format'])}",
                    "format": f["format"],
                }
                for f in output_files
            ],
        }
        print(json.dumps(result_json))
        
    except Exception as e:
        log_error(f"Failed: {e}")
        result_json = {
            "success": False,
            "nodeName": node_name,
            "error": str(e),
        }
        print(json.dumps(result_json))
        sys.exit(1)
    finally:
        conn.close()


if __name__ == "__main__":
    main()
