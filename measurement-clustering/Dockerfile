FROM python:3.11-slim

WORKDIR /app

# Install DuckDB (prebuilt wheel - very fast)
RUN pip install --no-cache-dir duckdb

# Pre-install httpfs extension (required for S3 access)
# This avoids runtime download which fails in air-gapped k8s environments
RUN python -c "import duckdb; conn = duckdb.connect(); conn.install_extension('httpfs'); conn.close()"

# Install clustering dependencies
RUN pip install --no-cache-dir \
    pandas==2.2.3 \
    pyarrow==19.0.1 \
    scikit-learn==1.6.1 \
    numpy==2.2.4

COPY transform.py ./

CMD ["python", "transform.py"]
