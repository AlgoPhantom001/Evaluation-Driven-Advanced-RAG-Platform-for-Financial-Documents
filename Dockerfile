# syntax=docker/dockerfile:1.7

FROM python:3.12-slim AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /build

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
COPY src ./src

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN pip install --upgrade pip && pip install .

ENV SECRAG_MODEL_CACHE_DIR=/opt/models
RUN python -c "\
from fastembed import TextEmbedding; \
from fastembed.rerank.cross_encoder import TextCrossEncoder; \
TextEmbedding(model_name='BAAI/bge-small-en-v1.5', cache_dir='/opt/models'); \
TextCrossEncoder(model_name='Xenova/ms-marco-MiniLM-L-6-v2', cache_dir='/opt/models'); \
print('models cached')"

FROM python:3.12-slim AS runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/opt/venv/bin:$PATH" \
    SECRAG_MODEL_CACHE_DIR=/opt/models \
    SECRAG_DATA_DIR=/data \
    SECRAG_LOG_JSON=true \
    SECRAG_PORT=7860 \
    HF_HUB_DISABLE_TELEMETRY=1 \
    SECRAG_ENABLE_SPLADE=false

COPY --from=builder /opt/venv /opt/venv
COPY --from=builder /opt/models /opt/models

WORKDIR /app
COPY src ./src
COPY ui ./ui
COPY evals ./evals
COPY pyproject.toml README.md ./

RUN useradd -m -u 1000 app \
    && mkdir -p /data /opt/models \
    && chown -R app:app /data /app /opt/models
USER app

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:7860/health',timeout=4).status==200 else 1)"

CMD ["uvicorn", "secrag.api.app:app", "--host", "0.0.0.0", "--port", "7860"]
