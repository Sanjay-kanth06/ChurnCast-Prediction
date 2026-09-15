# ChurnCast API + pipeline image.
#
# Python 3.11 rather than 3.13: Apache Airflow has no 3.13-compatible release at
# the time of writing, and using one interpreter for the API, the pipeline and
# the Airflow container keeps the code path identical across services.
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update \
 && apt-get install -y --no-install-recommends build-essential curl \
 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY src/ ./src/
COPY scripts/ ./scripts/
COPY pipelines/ ./pipelines/

RUN mkdir -p data/raw data/processed reports

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
  CMD curl -fsS http://localhost:8000/health || exit 1

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
