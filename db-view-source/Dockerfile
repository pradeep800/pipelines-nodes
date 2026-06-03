FROM python:3.11-slim

WORKDIR /app

# Install system dependencies for PostgreSQL client
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Install DuckDB with postgres and httpfs extensions, plus requests for Keycloak auth
RUN pip install --no-cache-dir duckdb requests

# Pre-install extensions to avoid runtime downloads in air-gapped k8s
RUN python -c "import duckdb; conn = duckdb.connect(); conn.install_extension('httpfs'); conn.install_extension('postgres'); conn.close()"

COPY fetch.py ./

CMD ["python", "fetch.py"]
