#!/usr/bin/env python3
"""
OMOP Wide Table Source - Custom Node Container

Exports a wide-format ("one row per subject-visit") view of the caller's
granted OMOP cohort data and uploads it as Parquet for downstream nodes.

The heavy lifting is the vendored SDK's aggregate(): it pulls the long-format
rows one cohort at a time, spools them to temporary Parquet parts, then dedupes
and pivots each local_concept.dataset off that spool with polars' streaming
engine. Peak memory is one dataset's wide frame, not the whole pull -- which is
what keeps a large cohort from OOMing in a pod with a ~2Gi limit.

aggregate() returns one wide frame per dataset, but a node declares its outputs
statically. Rather than make the user pick one dataset per node, this stacks
every granted dataset into a single result.parquet and adds a `dataset` column
naming each row's source:

    dataset | Barcode | Visit | aft_score | mmse_total
    AFT     |  B001   |  V1   |    1.0    |    null
    MMSE    |  B001   |  V1   |   null    |     28

Datasets don't share field columns, so the sheet is sparse by construction: a
row is null in the columns owned by the other datasets. That is the trade for
one file and no config -- downstream narrows it with a plain
`WHERE dataset = 'AFT'`.

Authentication reuses the caller's own identity instead of storing credentials
in the node config: Argo injects API_KEY into every node pod, and the SDK hands
it to the server, which resolves it to the user who ran the pipeline. The key
does not expire mid-run and is revoked when the execution finishes.

This node has no S3 inputs (the data comes from the data-access server via the
SDK), so it only needs the artifact bucket's credentials:
  ARTIFACT_S3_*  - artifact bucket (workflow outputs), read+write
Writes to the exact paths the platform assigns in output.files[*].path.
"""

import json
import os
import sys
import tempfile
import time

import boto3
import polars as pl
from boto3.s3.transfer import TransferConfig
from botocore.client import Config
from cbr_data_access import DataAccessClient
from cbr_data_access.aggregate import COHORT_NAMES, aggregate, index_columns
from cbr_data_access.exceptions import AuthenticationError, DataAccessError

# Bump this on every code change so a run's logs prove which build is live.
NODE_VERSION = "2026-07-15.1-wide-aggregate"


def log(msg):
    print(f"[OMOP-WIDE-TABLE-SOURCE] {msg}", flush=True)


def log_error(msg):
    print(f"[OMOP-WIDE-TABLE-SOURCE ERROR] {msg}", file=sys.stderr, flush=True)


def parse_context():
    raw = os.environ.get("NODE_CONTEXT", "")
    if not raw:
        raise ValueError("NODE_CONTEXT is required")
    return json.loads(raw)


def s3_key_from_path(s3_path):
    # s3://bucket/some/key  →  some/key
    return s3_path.replace("s3://", "").split("/", 1)[1]


def parse_request_id(raw):
    """The access-request picker stores {request_id, use_case}; older configs
    (and hand-edited ones) may carry the bare id string."""
    if isinstance(raw, dict):
        return (raw.get("request_id") or "").strip() or None
    return (str(raw).strip() or None) if raw else None


def parse_cohort(raw):
    """Blank means every granted cohort (aggregate's own default)."""
    if raw is None or not str(raw).strip():
        return None
    cohort = str(raw).strip()
    # aggregate() accepts an id or a known name; ids arrive as strings from the
    # select, and it only resolves names from COHORT_NAMES, so an unknown name
    # would fail deep in the pull. Reject it here with the valid set instead.
    if not cohort.isdigit() and cohort not in COHORT_NAMES.values():
        raise ValueError(
            f"Unknown cohort {cohort!r}; known names: {sorted(COHORT_NAMES.values())}"
        )
    return cohort


def build_authenticated_client(**sdk_kwargs):
    """Build a DataAccessClient from the API key Argo injects into the pod,
    instead of credentials stored in the pipeline config."""
    api_key = os.environ.get("API_KEY")
    if not api_key:
        raise AuthenticationError("API_KEY is not set in this pod")

    return DataAccessClient(api_key=api_key, **sdk_kwargs)


DATASET_COLUMN = "dataset"


def stack_datasets(frames, index_cols):
    """Stack every dataset's wide frame into one sheet, tagged by dataset.

    Each frame carries the same identity columns but its own field columns, so
    the stack is a diagonal concat: the result holds the union of all field
    columns, and a row is null in the columns belonging to the other datasets.
    The dataset column is what makes that readable -- it names the frame each
    row came from, so downstream can filter (WHERE dataset = 'AFT') rather than
    guess from which columns happen to be populated.

    Two datasets sharing a field_name land in one column here rather than
    colliding: same field, different rows, told apart by the dataset column.
    "diagonal_relaxed" is what allows that when they disagree on dtype -- it
    widens to a common supertype instead of raising.
    """
    tagged = [
        frame.with_columns(pl.lit(name).alias(DATASET_COLUMN))
        for name, frame in sorted(frames.items())
    ]
    stacked = pl.concat(tagged, how="diagonal_relaxed")
    return order_columns(stacked, index_cols)


def order_columns(frame, index_cols):
    """dataset first, then identity columns in aggregate's order, then fields."""
    lead = [DATASET_COLUMN] + [c for c in index_cols if c in frame.columns]
    ordered = frame.select(lead + sorted(c for c in frame.columns if c not in lead))
    # aggregate sorts each frame's rows on its own; re-sort across the stack so
    # a dataset's rows stay contiguous and the file is byte-stable run to run.
    sort_by = [c for c in (DATASET_COLUMN, "Cohort", "Barcode", "Visit") if c in ordered.columns]
    return ordered.sort(sort_by) if sort_by else ordered


def main():
    log(f"version {NODE_VERSION}")
    ctx = parse_context()
    config = ctx["config"]
    node = ctx["node"]
    output = ctx["output"]

    cohort = parse_cohort(config.get("cohort"))
    request_id = parse_request_id(config.get("access_request"))
    out_files = output["files"]

    log(f"Node: {node['name']} | Cohort: {cohort or 'all granted'} | "
        f"Request: {request_id or 'auto'}")

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

    s3 = boto3.client(
        "s3",
        endpoint_url="{}://{}".format(
            "https" if os.environ.get("ARTIFACT_S3_USE_SSL", "false").lower() == "true" else "http",
            os.environ["ARTIFACT_S3_ENDPOINT"],
        ),
        aws_access_key_id=os.environ["ARTIFACT_S3_ACCESS_KEY"],
        aws_secret_access_key=os.environ["ARTIFACT_S3_SECRET_KEY"],
        aws_session_token=os.environ.get("ARTIFACT_S3_SESSION_TOKEN") or None,
        region_name=os.environ.get("ARTIFACT_S3_REGION", "us-east-1"),
        config=Config(signature_version="s3v4"),
    )
    bucket = os.environ["ARTIFACT_S3_BUCKET"]

    try:
        with build_authenticated_client(**sdk_kwargs) as client:
            # Resolved server-side: the key is opaque, so this both names the
            # user and proves the key is live before we query.
            log(f"Authenticated with injected API key as: {client.identity}")

            if request_id:
                log(f"Setting access request: {request_id}")
                client.set_access(request_id)

            first_key = s3_key_from_path(out_files[0]["path"])

            # Parquet codec. snappy is ~5-10x cheaper to encode than zstd and
            # the output stays small. Override with PARQUET_COMPRESSION
            # (e.g. "zstd", "none").
            compression = os.environ.get("PARQUET_COMPRESSION", "snappy").strip().lower()
            if compression in ("", "none"):
                compression = None
            log(f"Parquet compression: {compression or 'none'}")

            with tempfile.TemporaryDirectory() as tmp:
                local_path = os.path.join(tmp, f"{out_files[0]['name']}.parquet")

                # Phase 1: pull + pivot. aggregate() streams each cohort through
                # an on-disk spool, so this stays memory-bounded; progress=log
                # surfaces its phase timing, which is the only output during a
                # pull that can run for minutes.
                t0 = time.monotonic()
                frames = aggregate(client, cohort, progress=log)
                log(f"Aggregate complete in {time.monotonic() - t0:.1f}s; "
                    f"datasets: {sorted(frames)}")

                frame = stack_datasets(frames, index_columns(cohort))
                log(f"Stacked {len(frames)} dataset(s) into one sheet: "
                    f"{frame.height:,} rows x {frame.width:,} cols")

                t_enc = time.monotonic()
                frame.write_parquet(local_path, compression=compression or "uncompressed")
                size_mb = os.path.getsize(local_path) / 1e6
                log(f"Parquet written: {frame.height:,} rows x {frame.width:,} cols, "
                    f"{size_mb:.1f} MB on disk in {time.monotonic() - t_enc:.1f}s "
                    f"(pull+pivot+encode total {time.monotonic() - t0:.1f}s); uploading next")

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
                    extra_key = s3_key_from_path(f["path"])
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
            "datasets": sorted(frames),
            "cohort": cohort,
            "rows": frame.height,
            "columns": frame.width,
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
