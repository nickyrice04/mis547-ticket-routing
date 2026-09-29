# The inference image: the ticket router API, CPU only.
#
# Built by GitHub Actions on every push to main, scanned with Trivy, and published to
# GitHub Container Registry. The inference droplet pulls it (deploy/update.sh).
#
# The two pretrained models the router uses are baked in at pinned revisions, so the
# running container never downloads anything from Hugging Face. The fitted router itself
# (about 185 MB) is not in the image. It comes from Spaces at startup, is checked against
# its SHA-256, and can be replaced without rebuilding the image.
#
#   docker build -t ticket-router .

FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/opt/hf

WORKDIR /app

# CPU-only PyTorch, a fraction of the size of the CUDA build, then the pinned runtime.
COPY docker/requirements-api.txt /tmp/requirements-api.txt
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch==2.14.0 \
 && pip install -r /tmp/requirements-api.txt \
 && pip install --upgrade setuptools wheel

# Pretrained models at the exact revisions the router was fitted with.
ARG E5_REVISION=d128750597153bb5987e10b1c3493a34e5a4502a
ARG OPUS_REVISION=1a922f3b32a8e809e17a47d4b32142d8105924e5
RUN python -c "from huggingface_hub import snapshot_download as d; \
d('intfloat/multilingual-e5-base', revision='${E5_REVISION}', allow_patterns=['*.json','*.txt','*.model','*.safetensors','1_Pooling/*','sentencepiece*','tokenizer*']); \
d('Helsinki-NLP/opus-mt-de-en', revision='${OPUS_REVISION}', allow_patterns=['*.json','*.txt','*.spm','*.safetensors','*.bin','vocab*','source*','target*'])"

COPY src/ /app/src/

# Unprivileged user with a numeric UID (same practice as Lab 5), and the model cache it can write.
RUN useradd --create-home --uid 10001 appuser && install -d -o 10001 -g 10001 /models
USER 10001

ENV HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    EMBED_DEVICE=cpu \
    TOKENIZERS_PARALLELISM=false \
    OMP_NUM_THREADS=2 \
    MODEL_CACHE=/models

EXPOSE 8000

# Liveness, polled by Docker. Readiness (/readyz) is what a load balancer would poll.
HEALTHCHECK --interval=30s --timeout=5s --start-period=180s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/healthz', timeout=4)"

CMD ["uvicorn", "serve.app:app", "--app-dir", "/app/src", "--host", "0.0.0.0", "--port", "8000", \
     "--proxy-headers", "--forwarded-allow-ips", "*", "--no-server-header"]
