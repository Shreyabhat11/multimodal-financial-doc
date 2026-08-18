# syntax=docker/dockerfile:1
# ============================================================
# API service image.
#
# Deliberately CPU-only / no torch by default (see requirements.txt's trailing
# comment for why this split is possible): the default MODEL_PROVIDER is
# hf-inference, which talks to the Hugging Face Inference API over HTTP and
# needs no local GPU. This keeps the default image small and buildable
# anywhere, including machines with no GPU at all.
#
# To enable LOCAL GPU inference (MODEL_PROVIDER=qwen-vl or llava with
# backend="local"), build with:
#   docker build --build-arg INSTALL_LOCAL_INFERENCE=true -t findoc-api .
# and run the container with GPU access (see docker-compose.gpu.yml).
# ============================================================

FROM python:3.11-slim AS base

# tesseract-ocr: required at runtime by app/extraction/ocr_engine.py (OCR fallback).
# libgl1: required by PyMuPDF's rendering backend on some Debian slim images.
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    libgl1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt requirements-local-inference.txt ./

ARG INSTALL_LOCAL_INFERENCE=false
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt \
    && if [ "$INSTALL_LOCAL_INFERENCE" = "true" ]; then \
         pip install --no-cache-dir -r requirements-local-inference.txt; \
       fi

COPY app/ ./app/
COPY configs/ ./configs/
COPY migrations/ ./migrations/
COPY alembic.ini ./
COPY scripts/docker-entrypoint.sh ./scripts/docker-entrypoint.sh
RUN chmod +x ./scripts/docker-entrypoint.sh

# Non-root user — running as root in a container is unnecessary privilege for a
# process that only needs to bind a port and read/write its own data directory.
RUN useradd --create-home --uid 1000 appuser \
    && mkdir -p /app/data/raw \
    && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

ENTRYPOINT ["./scripts/docker-entrypoint.sh"]
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
