FROM python:3.11-slim

WORKDIR /app

RUN pip install --no-cache-dir uv

# Install DuckDB (prebuilt wheel - very fast)
RUN uv pip install --system --no-cache duckdb

# Pre-install httpfs extension (required for S3 access)
# This avoids runtime download which fails in air-gapped k8s environments
RUN python -c "import duckdb; conn = duckdb.connect(); conn.install_extension('httpfs'); conn.close()"

# Copy script
COPY query.py ./

CMD ["python", "query.py"]
