#!/bin/bash
# Push a trained model to the droplet, rebuild the API image, and prove it works.
#
#   ./infra/deploy.sh 1_tfidf_logreg sklearn
#   ./infra/deploy.sh 2_distilbert hf
#
# Run it from the repository root.
set -euo pipefail

TIER="${1:?usage: deploy.sh <tier> <sklearn|hf>}"
KIND="${2:?usage: deploy.sh <tier> <sklearn|hf>}"
DROPLET="${DROPLET:-174.138.91.2}"
KEY="${KEY:-$HOME/.ssh/digitalocean_key}"

if [ "$KIND" = "sklearn" ]; then
  MODEL_PATH="models/${TIER}.joblib"
  REQS="requirements-serve.txt"
else
  MODEL_PATH="models/${TIER}"
  REQS="requirements-serve-hf.txt"
fi

echo "==> syncing code and ${MODEL_PATH}"
rsync -az \
  --exclude .venv --exclude data_tickets.parquet --exclude __pycache__ \
  --exclude 'splits/train.json' --exclude 'splits/test.json' --exclude models \
  -e "ssh -i $KEY" ./ "root@${DROPLET}:/srv/ticket-routing/"
rsync -az -e "ssh -i $KEY" "$MODEL_PATH" "root@${DROPLET}:/srv/ticket-routing/models/"

echo "==> building image on the droplet"
ssh -i "$KEY" "root@${DROPLET}" "cd /srv/ticket-routing && \
  docker build -q --build-arg MODEL_DIR=${MODEL_PATH} --build-arg REQS=${REQS} \
    -t ticket-api:${TIER} . >/dev/null && \
  docker rm -f ticket-api >/dev/null 2>&1 || true"

echo "==> starting container"
ssh -i "$KEY" "root@${DROPLET}" "docker run -d --name ticket-api \
  -p 127.0.0.1:8000:8000 --restart unless-stopped \
  -e MODEL_KIND=${KIND} -e MODEL_VERSION=${TIER} -e TORCH_THREADS=2 \
  ticket-api:${TIER} >/dev/null"

echo "==> waiting for health"
ssh -i "$KEY" "root@${DROPLET}" \
  'for i in $(seq 1 60); do
     if curl -sf localhost:8000/health | grep -q "\"status\":\"ok\""; then echo "healthy after ${i}s"; break; fi
     sleep 1
   done
   curl -s localhost:8000/health; echo
   curl -s -X POST localhost:8000/predict -H "content-type: application/json" \
     -d "{\"subject\":\"double charge\",\"body\":\"I was billed twice for my subscription and need a refund.\"}"; echo
   docker stats --no-stream --format "memory: {{.MemUsage}}"'
