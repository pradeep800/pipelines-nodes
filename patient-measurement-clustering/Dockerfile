FROM ghcr.io/astral-sh/uv:latest AS uv

FROM python:3.11-slim

WORKDIR /app

COPY --from=uv /uv /usr/local/bin/uv

RUN uv pip install --system --no-cache duckdb

RUN python -c "import duckdb; conn = duckdb.connect(); conn.install_extension('httpfs'); conn.close()"

RUN uv pip install --system --no-cache \
    pandas==2.2.3 \
    pyarrow==19.0.1 \
    scikit-learn==1.6.1 \
    numpy==2.2.4

COPY clustering.py ./

CMD ["python", "clustering.py"]
