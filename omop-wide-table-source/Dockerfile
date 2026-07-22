FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir uv

# sdk/ is the cbr-data-access-ipynb-sdk submodule; only its py-sdk is used.
COPY sdk/py-sdk/ ./sdk/
# Scrub any build artifacts before installing: setuptools reuses a stale
# build/lib as the wheel staging dir and silently packages old code.
RUN rm -rf ./sdk/build ./sdk/dist ./sdk/src/*.egg-info \
    && uv pip install --system --no-cache ./sdk/ boto3

COPY transform.py ./

CMD ["python", "transform.py"]
