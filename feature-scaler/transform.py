#!/usr/bin/env python3
"""
Feature Scaler Node
Scales numeric columns using StandardScaler or MinMaxScaler from scikit-learn.

Reads NODE_CONTEXT from environment (same contract as all other nodes).
Reads input parquet from MinIO via boto3, writes scaled parquet back to MinIO.
"""

import os
import sys
import json
from io import BytesIO

import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import boto3
from botocore.config import Config


def log(msg):
    print(f"[SCALER] {msg}", flush=True)


def log_error(msg):
    print(f"[SCALER ERROR] {msg}", file=sys.stderr, flush=True)


def get_s3_client():
    session_token = os.environ.get("S3_SESSION_TOKEN") or None
    use_ssl = os.environ.get("S3_USE_SSL", "false").lower() == "true"
    endpoint = os.environ["S3_ENDPOINT"]
    scheme = "https" if use_ssl else "http"

    return boto3.client(
        "s3",
        endpoint_url=f"{scheme}://{endpoint}",
        aws_access_key_id=os.environ["S3_ACCESS_KEY"],
        aws_secret_access_key=os.environ["S3_SECRET_KEY"],
        aws_session_token=session_token,
        config=Config(signature_version="s3v4"),
    )


def s3_split(s3_path):
    """Split 's3://bucket/path/to/file' into (bucket, key)."""
    path = s3_path.replace("s3://", "")
    bucket, key = path.split("/", 1)
    return bucket, key


def read_parquet(s3, s3_path):
    bucket, key = s3_split(s3_path)
    log(f"Reading {s3_path}")
    obj = s3.get_object(Bucket=bucket, Key=key)
    return pd.read_parquet(BytesIO(obj["Body"].read()))


def write_parquet(s3, df, s3_path):
    bucket, key = s3_split(s3_path)
    log(f"Writing {len(df)} rows to {s3_path}")
    buf = BytesIO()
    df.to_parquet(buf, index=False, engine="pyarrow")
    buf.seek(0)
    s3.put_object(Bucket=bucket, Key=key, Body=buf.getvalue())


def main():
    # ── Parse context ───────────────────────────────────────────────
    ctx = json.loads(os.environ["NODE_CONTEXT"])
    config  = ctx["config"]
    inputs  = ctx["inputs"]
    output  = ctx["output"]
    node    = ctx["node"]

    method     = config.get("method", "standard")
    cols_raw   = config.get("columns", "").strip()
    base_path  = output["basePath"]

    log(f"=== Feature Scaler: {node['name']} ===")
    log(f"Method: {method}")
    log(f"Columns config: '{cols_raw or '(all numeric)'}'")

    if not inputs:
        log_error("No inputs provided")
        sys.exit(1)

    s3 = get_s3_client()

    # ── Read input ──────────────────────────────────────────────────
    df = read_parquet(s3, inputs[0]["output"]["path"])
    log(f"Loaded {len(df)} rows, {len(df.columns)} columns: {list(df.columns)}")

    # ── Resolve columns to scale ────────────────────────────────────
    if cols_raw:
        cols = [c.strip() for c in cols_raw.split(",") if c.strip()]
        missing = [c for c in cols if c not in df.columns]
        if missing:
            log_error(f"Columns not found in dataset: {missing}")
            sys.exit(1)
        # Only keep numeric ones from the user's list
        non_numeric = [c for c in cols if not pd.api.types.is_numeric_dtype(df[c])]
        if non_numeric:
            log_error(f"Cannot scale non-numeric columns: {non_numeric}")
            sys.exit(1)
    else:
        # Auto-detect all numeric columns
        cols = df.select_dtypes(include="number").columns.tolist()
        if not cols:
            log_error("No numeric columns found in dataset")
            sys.exit(1)
        log(f"Auto-selected numeric columns: {cols}")

    # ── Scale ───────────────────────────────────────────────────────
    scaler = StandardScaler() if method == "standard" else MinMaxScaler()
    df[cols] = scaler.fit_transform(df[cols])
    log(f"Scaled {len(cols)} columns: {cols}")

    # ── Write output ────────────────────────────────────────────────
    out_path = f"{base_path}/result.parquet"
    write_parquet(s3, df, out_path)

    print(json.dumps({
        "success": True,
        "nodeName": node["name"],
        "rowCount": len(df),
        "scaledColumns": cols,
        "method": method,
    }))
    log("=== Done ===")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        log_error(f"Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
