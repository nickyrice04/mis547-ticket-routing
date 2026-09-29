# Serving image for the ticket routing API.
#
# The model and its preprocessing code ship together in this image, which is the
# point the midterm report makes about the fitted vectorizer. Build it with the
# model directory present:
#
#   docker build -f docker/baseline.Dockerfile --build-arg MODEL_DIR=models/2_distilbert -t ticket-api:distilbert .
#   docker run --rm -p 8000:8000 ticket-api:distilbert

FROM python:3.12-slim

WORKDIR /app

# Dependencies first, so a model swap does not rebuild this layer.
# REQS picks the small scikit-learn set or the larger PyTorch set.
ARG REQS=docker/requirements-baseline.txt
COPY ${REQS} /app/reqs.txt
RUN pip install --no-cache-dir -r /app/reqs.txt

# Shared preprocessing plus the API.
COPY src/common.py src/serve/baseline_api.py ./
COPY splits/meta.json ./splits/meta.json

# The model itself. MODEL_DIR is a path in the build context.
ARG MODEL_DIR=models/1_tfidf_logreg.joblib
COPY ${MODEL_DIR} /app/model

ENV SPLITS_DIR=/app/splits \
    MODEL_PATH=/app/model \
    MODEL_KIND=sklearn \
    MODEL_VERSION=v1 \
    CONFIDENCE_THRESHOLD=0.5 \
    TORCH_THREADS=2 \
    PYTHONUNBUFFERED=1

# Run as a non-root user, the same practice as the Lab 2 Dockerfile.
RUN useradd --create-home --shell /usr/sbin/nologin appuser && chown -R appuser /app
USER appuser

EXPOSE 8000

# The container answers its own health check, which is how the droplet and a
# load balancer find out that the process is alive.
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request;urllib.request.urlopen('http://localhost:8000/health')"

CMD ["uvicorn", "baseline_api:app", "--host", "0.0.0.0", "--port", "8000"]
