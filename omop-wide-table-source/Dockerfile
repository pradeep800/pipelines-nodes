FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir uv

COPY sdk/ ./sdk/
# The vendored sdk ships a committed, stale build/lib snapshot; setuptools
# reuses it as the wheel staging dir and silently packages the old code.
# Scrub it so the wheel is always built from sdk/src.
RUN rm -rf ./sdk/build ./sdk/dist ./sdk/src/*.egg-info \
    && uv pip install --system --no-cache ./sdk/ boto3

COPY transform.py ./

CMD ["python", "transform.py"]
