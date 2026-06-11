#!/usr/bin/env python3
"""
Patient Measurement Clustering - Custom Node Container

Pivots an OMOP-format measurement table (person_id, measurement_source_value,
value_as_number) so each patient becomes one row, runs KMeans clustering on
the resulting feature matrix, and writes back the original table with an
added 'cluster' column. Uses DuckDB for S3 I/O, following the same pattern
as the SQL Transformer node.
"""

import os
import sys
import json
import duckdb
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer


def log(msg):
    print(f"[CLUSTERING] {msg}", flush=True)


def log_error(msg):
    print(f"[CLUSTERING ERROR] {msg}", file=sys.stderr, flush=True)


def get_s3_config():
    return {
        "endpoint":      os.environ.get("S3_ENDPOINT", "minio:9000"),
        "access_key":    os.environ.get("S3_ACCESS_KEY", ""),
        "secret_key":    os.environ.get("S3_SECRET_KEY", ""),
        "session_token": os.environ.get("S3_SESSION_TOKEN", ""),
        "use_ssl":       os.environ.get("S3_USE_SSL", "false").lower() == "true",
        "region":        os.environ.get("S3_REGION", "us-east-1"),
    }


def setup_duckdb_s3(conn, s3):
    conn.load_extension("httpfs")
    token_clause = f",\n        SESSION_TOKEN '{s3['session_token']}'" if s3["session_token"] else ""
    conn.execute(f"""
        CREATE SECRET minio_secret (
            TYPE S3,
            KEY_ID '{s3["access_key"]}',
            SECRET '{s3["secret_key"]}',
            ENDPOINT '{s3["endpoint"]}',
            URL_STYLE 'path',
            USE_SSL {str(s3["use_ssl"]).lower()},
            REGION '{s3["region"]}'{token_clause}
        )
    """)


def run_clustering(df: pd.DataFrame, n_clusters: int) -> pd.DataFrame:
    """Pivot OMOP measurement table -> KMeans -> cluster labels per person_id."""
    pivot = df.pivot_table(
        index="person_id",
        columns="measurement_source_value",
        values="value_as_number",
        aggfunc="mean"
    ).reset_index()

    person_ids = pivot["person_id"]
    X = pivot.drop(columns=["person_id"])

    X_imputed = SimpleImputer(strategy="mean").fit_transform(X)
    X_scaled  = StandardScaler().fit_transform(X_imputed)
    labels    = KMeans(n_clusters=n_clusters, random_state=42, n_init=10).fit_predict(X_scaled)

    for i in range(n_clusters):
        log(f"  Cluster {i}: {(labels==i).sum()} patients")

    return pd.DataFrame({"person_id": person_ids.values, "cluster": labels})


def main():
    ctx        = json.loads(os.environ["NODE_CONTEXT"])
    node       = ctx["node"]
    inputs     = ctx["inputs"]
    output     = ctx["output"]
    config     = ctx["config"]

    n_clusters = int(config.get("n_clusters", 3))
    base_path  = output["basePath"]
    input_path = inputs[0]["output"]["path"]
    input_format = inputs[0]["output"].get("format", "parquet").lower()
    output_path = f"{base_path}/result.parquet"

    log(f"=== Patient Clustering: {node['name']} ===")
    log(f"Input  : {input_path} (format: {input_format})")
    log(f"Output : {output_path}")
    log(f"n_clusters: {n_clusters}")

    s3 = get_s3_config()
    conn = duckdb.connect(":memory:")
    setup_duckdb_s3(conn, s3)

    if input_format == "csv":
        read_func = "read_csv_auto"
    elif input_format == "json":
        read_func = "read_json_auto"
    else:
        read_func = "read_parquet"

    log("Reading input from S3...")
    df = conn.execute(f"SELECT * FROM {read_func}('{input_path}')").df()
    log(f"Loaded {len(df)} rows, columns: {list(df.columns)}")

    required = {"person_id", "measurement_source_value", "value_as_number"}
    missing  = required - set(df.columns)
    if missing:
        log_error(f"Missing required columns: {missing}")
        sys.exit(1)

    log("Running KMeans clustering...")
    cluster_df = run_clustering(df, n_clusters)

    result_df = df.merge(cluster_df, on="person_id", how="left")
    log(f"Output has {len(result_df)} rows")

    log("Writing output parquet to S3...")
    conn.register("result_table", result_df)
    conn.execute(f"""
        COPY result_table TO '{output_path}'
        (FORMAT PARQUET, COMPRESSION 'snappy')
    """)

    log("=== Done ===")
    print(json.dumps({
        "success":      True,
        "nodeName":     node["name"],
        "totalRows":    len(result_df),
        "nClusters":    n_clusters,
        "clusterCounts": {
            str(i): int((cluster_df["cluster"] == i).sum())
            for i in range(n_clusters)
        },
    }))


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        log_error(f"Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
