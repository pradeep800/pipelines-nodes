#!/usr/bin/env python3
"""
3 D Merger - batch worker for Argo (no Streamlit server).

Reads NODE_CONTEXT, downloads upstream STL paths from S3, merges meshes with vedo,
uploads merged STL to the node's artifact output path, then exits.

Environment: NODE_CONTEXT, S3_* (same contract as sql-transformer).
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile

import boto3
from botocore.exceptions import BotoCoreError, ClientError
from vedo import load, merge, write


def log(msg: str) -> None:
    print(f"[3 D MERGER] {msg}", flush=True)


def log_err(msg: str) -> None:
    print(f"[3 D MERGER ERROR] {msg}", file=sys.stderr, flush=True)


def parse_node_context() -> dict:
    raw = os.environ.get("NODE_CONTEXT", "")
    if not raw:
        raise ValueError("NODE_CONTEXT is required")
    try:
        return json.loads(raw)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid NODE_CONTEXT JSON: {e}") from e


def get_s3_client():
    endpoint = os.environ.get("S3_ENDPOINT", "minio:9000")
    use_ssl = os.environ.get("S3_USE_SSL", "false").lower() == "true"
    if not endpoint.startswith("http://") and not endpoint.startswith("https://"):
        scheme = "https" if use_ssl else "http"
        endpoint = f"{scheme}://{endpoint}"
    return boto3.client(
        "s3",
        endpoint_url=endpoint,
        aws_access_key_id=os.environ.get("S3_ACCESS_KEY", ""),
        aws_secret_access_key=os.environ.get("S3_SECRET_KEY", ""),
        aws_session_token=os.environ.get("S3_SESSION_TOKEN") or None,
        region_name=os.environ.get("S3_REGION", "us-east-1"),
    )


def parse_s3_uri(uri: str) -> tuple[str, str]:
    if not uri.startswith("s3://"):
        raise ValueError(f"Expected s3:// URI, got: {uri[:120]}")
    rest = uri[5:]
    slash = rest.find("/")
    if slash == -1:
        return rest, ""
    return rest[:slash], rest[slash + 1 :]


def extension_for_format(fmt: str) -> str:
    f = (fmt or "parquet").lower()
    if f == "file":
        return "stl"
    return {"parquet": "parquet", "csv": "csv", "json": "json"}.get(f, "stl")


def main() -> None:
    ctx = parse_node_context()
    node = ctx.get("node") or {}
    inputs = ctx.get("inputs") or []
    output = ctx.get("output") or {}
    base_path = output.get("basePath")
    out_files = output.get("files") or []

    node_name = node.get("name", "node")
    node_slug = node.get("slug", "3-d-merger")

    log(f"=== 3 D merger ({node_slug}) ===")
    log(f"Node name: {node_name}")

    if not base_path:
        raise ValueError("output.basePath is required in NODE_CONTEXT")
    if not out_files:
        raise ValueError("output.files is required in NODE_CONTEXT")
    if not inputs:
        raise ValueError(
            "No upstream inputs in NODE_CONTEXT. Connect a Data Source that selects an STL file "
            "to this node."
        )

    client = get_s3_client()
    bucket = os.environ.get("S3_BUCKET", "data-pipeline")

    stl_local_paths: list[str] = []
    root = tempfile.mkdtemp(prefix="3_d_merger_")
    try:
        for i, inp in enumerate(inputs):
            out = inp.get("output") or {}
            path = out.get("path")
            if not path:
                raise ValueError(f"Missing input path for {inp.get('nodeName', i)}")
            b, key = parse_s3_uri(path)
            if b != bucket:
                log(
                    f"Note: input bucket {b!r} != S3_BUCKET {bucket!r}; "
                    "downloading from URI bucket."
                )
            local = os.path.join(root, f"in_{i}.stl")
            log(f"Downloading {path} -> {local}")
            client.download_file(b, key, local)
            stl_local_paths.append(local)

        log(f"Merging {len(stl_local_paths)} mesh(es)...")
        meshes = [load(p) for p in stl_local_paths]
        merged = merge(meshes)

        out_def = out_files[0]
        out_name = out_def.get("name") or "output"
        fmt = out_def.get("format") or "file"
        ext = extension_for_format(str(fmt))
        merged_name = f"{out_name}.{ext}"

        base = base_path.rstrip("/")
        if base.lower().endswith(f".{ext}"):
            dest_uri = base
        else:
            dest_uri = f"{base}/{merged_name}"
        db, dkey = parse_s3_uri(dest_uri)

        merged_local = os.path.join(root, merged_name)
        write(merged, merged_local)

        log(f"Uploading merged mesh to {dest_uri}")
        client.upload_file(merged_local, db, dkey)

        result = {
            "success": True,
            "nodeName": node_name,
            "inputs": len(stl_local_paths),
            "outputPath": dest_uri,
            "format": fmt,
        }
        print(json.dumps(result), flush=True)
        log("Done.")
    except (BotoCoreError, ClientError) as e:
        log_err(f"S3 operation failed: {e}")
        print(json.dumps({"success": False, "error": "S3 operation failed"}), flush=True)
        sys.exit(1)
    finally:
        shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        log_err(str(e))
        print(json.dumps({"success": False, "error": str(e)}), flush=True)
        sys.exit(1)
