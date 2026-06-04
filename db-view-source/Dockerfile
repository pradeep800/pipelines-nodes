FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq-dev \
    git \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    git+https://github.com/datakaveri/data-access-py-sdk.git \
    boto3

COPY fetch.py ./

CMD ["python", "fetch.py"]
