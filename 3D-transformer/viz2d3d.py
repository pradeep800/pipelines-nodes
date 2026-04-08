import streamlit as st
from vedo import load, show, merge, write
import os
import uuid
import zipfile
import io
from multiprocessing import Process, Queue
import time
import boto3
from botocore.exceptions import BotoCoreError, ClientError


def load_show(stl_paths, queue: Queue):
    if isinstance(stl_paths, str):
        stl_paths = [stl_paths]
    colors = ['#ffc800', '#00c8ff', '#c800ff', '#00ff64', '#ff6400', '#ff0064']
    meshes = [load(p).color(colors[i % len(colors)]) for i, p in enumerate(stl_paths)]
    show(meshes, bg='black')
    queue.put('done')


def visualize_stl(stl_paths):
    """Visualize one or more STL files in a separate process. Accepts a path or list of paths."""
    queue = Queue()
    p = Process(target=load_show, args=(stl_paths, queue))
    p.start()
    while True:
        if not queue.empty():
            break
        time.sleep(0.1)
    p.join()


def get_s3_config() -> dict:
    """Get S3/MinIO configuration from environment variables."""
    return {
        "endpoint": os.environ.get("S3_ENDPOINT", "minio:9000"),
        "access_key": os.environ.get("S3_ACCESS_KEY", ""),
        "secret_key": os.environ.get("S3_SECRET_KEY", ""),
        "session_token": os.environ.get("S3_SESSION_TOKEN", ""),
        "bucket": os.environ.get("S3_BUCKET", "data-pipeline"),
        "use_ssl": os.environ.get("S3_USE_SSL", "false").lower() == "true",
        "region": os.environ.get("S3_REGION", "us-east-1"),
    }


def get_minio_client(s3_config: dict):
    """Create a boto3 S3 client using the same env contract as sql-transformer."""
    endpoint = s3_config["endpoint"]
    if not endpoint.startswith("http://") and not endpoint.startswith("https://"):
        scheme = "https" if s3_config["use_ssl"] else "http"
        endpoint = f"{scheme}://{endpoint}"

    return boto3.client(
        "s3",
        endpoint_url=endpoint,
        aws_access_key_id=s3_config["access_key"],
        aws_secret_access_key=s3_config["secret_key"],
        aws_session_token=s3_config["session_token"] or None,
        region_name=s3_config["region"],
    )


def upload_to_minio(client, bucket: str, local_path: str, object_key: str):
    """Upload a local file to MinIO."""
    client.upload_file(local_path, bucket, object_key)


def run_app():
    st.set_page_config(page_title='Vedo Visualization', layout='wide')
    st.title('Vedo Visualization in Streamlit')

    uploaded_files = st.file_uploader(
        "Choose STL file(s)", type="stl", accept_multiple_files=True
    )
    if uploaded_files:
        unique_id = str(uuid.uuid4())
        out_dir = f'./data/{unique_id}'
        os.makedirs(out_dir, exist_ok=True)

        file_paths = []
        for uf in uploaded_files:
            file_path = os.path.join(out_dir, uf.name)
            with open(file_path, 'wb') as f:
                f.write(uf.getbuffer())
            file_paths.append(file_path)

        st.success(f"Uploaded {len(file_paths)} file(s) to: {out_dir}")

        if st.button('Show Mesh'):
            visualize_stl(file_paths)

        # Save merged STL for download
        merged_path = os.path.join(out_dir, 'merged.stl')
        if not os.path.exists(merged_path):
            merged = merge([load(p) for p in file_paths])
            write(merged, merged_path)

        with open(merged_path, 'rb') as f:
            st.download_button(
                'Download Merged STL', f, file_name='merged.stl', mime='model/stl'
            )

        # Also offer a zip of all originals
        zip_buf = io.BytesIO()
        with zipfile.ZipFile(zip_buf, 'w', zipfile.ZIP_DEFLATED) as zf:
            for p in file_paths:
                zf.write(p, arcname=os.path.basename(p))
            zf.write(merged_path, arcname='merged.stl')
        st.download_button(
            'Download All (ZIP)', zip_buf.getvalue(),
            file_name='models.zip', mime='application/zip'
        )

        # Upload files to MinIO using the same S3 env config as sql-transformer.
        s3_config = get_s3_config()
        if s3_config["access_key"] and s3_config["secret_key"]:
            try:
                s3_client = get_minio_client(s3_config)
                remote_prefix = f"3d-transformer/{unique_id}"

                uploaded_keys = []
                for local_file in file_paths + [merged_path]:
                    object_key = f"{remote_prefix}/{os.path.basename(local_file)}"
                    upload_to_minio(s3_client, s3_config["bucket"], local_file, object_key)
                    uploaded_keys.append(object_key)

                st.success(
                    f"Uploaded {len(uploaded_keys)} file(s) to MinIO bucket "
                    f"'{s3_config['bucket']}' under '{remote_prefix}/'"
                )
            except (BotoCoreError, ClientError) as e:
                st.warning(f"MinIO upload failed: {e}")
        else:
            st.info("MinIO upload skipped: set S3_ACCESS_KEY and S3_SECRET_KEY to enable.")


if __name__ == "__main__":
    run_app()
