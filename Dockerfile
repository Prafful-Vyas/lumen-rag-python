FROM python:3.12-slim AS base

# Tesseract for the OCR fallback on scanned PDF pages.
RUN apt-get update \
    && apt-get install -y --no-install-recommends tesseract-ocr \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1

COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install ".[ui]"
COPY ui ./ui
COPY eval ./eval
COPY scripts ./scripts

RUN useradd --create-home --uid 1000 lumen && mkdir -p /app/storage && chown lumen /app/storage
USER lumen

EXPOSE 8000 8501
CMD ["uvicorn", "lumen.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
