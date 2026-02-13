FROM python:3.11-slim

WORKDIR /app

# Install DuckDB and dependencies
RUN pip install --no-cache-dir duckdb boto3 pyarrow

COPY transform.py .

CMD ["python", "transform.py"]
