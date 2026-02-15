FROM python:3.11-slim

WORKDIR /app

# Install DuckDB and dependencies
RUN pip install --no-cache-dir duckdb boto3 pyarrow

# Pre-install DuckDB httpfs extension (needed for S3 access)
RUN python -c "import duckdb; duckdb.connect().install_extension('httpfs')"

COPY transform.py .

CMD ["python", "transform.py"]
