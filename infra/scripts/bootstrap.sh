#!/usr/bin/env bash
# One-time setup after `terraform apply`, and again whenever a secret rotates.
#
#   infra/scripts/bootstrap.sh
#
# Reads Terraform's outputs and:
#   1. waits for cloud-init to finish on the droplets
#   2. writes the root-only env files the services read (/etc/ticket-routing/*.env).
#      Secrets travel over SSH and never touch the repository or the droplets' user data
#   3. creates the audit-log tables and the two least-privilege database grants
#   4. uploads the dataset to Spaces
#   5. starts the inference stack (the API waits for the first model version)
#
# API keys are generated once and kept in ~/.config/team3/api-keys.env on this laptop.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
cd "$ROOT/infra/terraform"
SECRETS="$HOME/.config/team3"
KEY="${KEY:-$HOME/.ssh/digitalocean_key}"
USER_NAME="${USER_NAME:-nicky}"
SSH=(ssh -i "$KEY" -o StrictHostKeyChecking=accept-new -o ConnectTimeout=15)

out() { terraform output -raw "$1"; }
urlq() { python3 -c 'import sys, urllib.parse; print(urllib.parse.quote(sys.argv[1], safe=""))' "$1"; }

API_IP=$(out api_ip)
INF=$(out inference_ip)
TRAIN=$(terraform output -raw training_ip 2>/dev/null || true)
[ "$TRAIN" = "null" ] && TRAIN=""
REGION=$(out region)
BUCKET=$(out bucket)
DB="$(out db_private_host):$(out db_port)/$(out db_name)"
SITE="${API_IP//./-}.sslip.io"

mkdir -p -m 700 "$SECRETS"
if [ ! -f "$SECRETS/api-keys.env" ]; then
  (umask 077; printf 'TEAM_KEY=%s\nGRADER_KEY=%s\nREPLAY_KEY=%s\n' \
    "$(openssl rand -hex 24)" "$(openssl rand -hex 24)" "$(openssl rand -hex 24)" > "$SECRETS/api-keys.env")
fi
. "$SECRETS/api-keys.env"

echo "==> waiting for cloud-init on the inference droplet ($INF)"
"${SSH[@]}" "$USER_NAME@$INF" "cloud-init status --wait >/dev/null; test -f /srv/ticket-routing/deploy/compose.yaml" || {
  echo "The droplet's checkout has no deploy/compose.yaml. Push the code to GitHub, then run"
  echo "  ssh $USER_NAME@$INF 'sudo git -C /srv/ticket-routing pull'"
  exit 1
}

echo "==> writing /etc/ticket-routing/compose.env and api.env (root-only)"
"${SSH[@]}" "$USER_NAME@$INF" "sudo install -m 600 /dev/stdin /etc/ticket-routing/compose.env" <<EOF
SITE_ADDRESS=$SITE
ACME_EMAIL=$(out alert_email)
API_IMAGE=ghcr.io/nickyrice04/mis547-ticket-routing:main
EOF
"${SSH[@]}" "$USER_NAME@$INF" "sudo install -m 600 /dev/stdin /etc/ticket-routing/api.env" <<EOF
API_KEYS=team:$TEAM_KEY,grader:$GRADER_KEY,replay:$REPLAY_KEY
CONFIDENCE_THRESHOLD=0.7
RATE_LIMIT_PER_MINUTE=60
DATABASE_URL=postgresql+psycopg://router_api:$(urlq "$(out db_api_password)")@$DB?sslmode=require
SPACES_REGION=$REGION
SPACES_BUCKET=$BUCKET
SPACES_KEY=$(out spaces_inference_key)
SPACES_SECRET=$(out spaces_inference_secret)
MODEL_POLL_SECONDS=120
EOF

echo "==> getting the API image (from the registry if CI has published it, else built on the droplet)"
"${SSH[@]}" "$USER_NAME@$INF" "cd /srv/ticket-routing/deploy && set -a && . /etc/ticket-routing/compose.env && set +a && \
  (sudo -E docker compose pull --quiet api 2>/dev/null || sudo -E docker compose build api)"

echo "==> creating the audit-log schema and grants"
# The admin URL goes over stdin so it never shows up in a process list or a file.
ADMIN_URL="postgresql+psycopg://$(out db_admin_user):$(urlq "$(out db_admin_password)")@$DB?sslmode=require"
"${SSH[@]}" "$USER_NAME@$INF" 'read -r ADMIN_DATABASE_URL; export ADMIN_DATABASE_URL; cd /srv/ticket-routing/deploy && \
  set -a && . /etc/ticket-routing/compose.env && set +a && \
  sudo -E docker compose run --rm --no-deps -e PYTHONPATH=/app/src -e ADMIN_DATABASE_URL api python -m mlops.db migrate' <<<"$ADMIN_URL"

echo "==> uploading the dataset to Spaces"
SPACES_REGION=$REGION SPACES_BUCKET=$BUCKET SPACES_KEY=$(out spaces_training_key) SPACES_SECRET=$(out spaces_training_secret) \
  PYTHONPATH="$ROOT/src" "$ROOT/.venv/bin/python" -c "
from mlops import storage
if not storage.exists('data/data_tickets.parquet'):
    storage.upload('$ROOT/data_tickets.parquet', 'data/data_tickets.parquet'); print('uploaded data_tickets.parquet')
else:
    print('dataset already in Spaces')"

if [ -n "$TRAIN" ]; then
  echo "==> writing /etc/ticket-routing/trainer.env on the training droplet ($TRAIN)"
  "${SSH[@]}" "$USER_NAME@$TRAIN" "cloud-init status --wait >/dev/null; sudo install -m 600 /dev/stdin /etc/ticket-routing/trainer.env" <<EOF
DATABASE_URL=postgresql+psycopg://router_trainer:$(urlq "$(out db_trainer_password)")@$DB?sslmode=require
SPACES_REGION=$REGION
SPACES_BUCKET=$BUCKET
SPACES_KEY=$(out spaces_training_key)
SPACES_SECRET=$(out spaces_training_secret)
EOF
fi

echo "==> starting the inference stack"
"${SSH[@]}" "$USER_NAME@$INF" "sudo systemctl start ticket-router ticket-router-update.timer && sudo systemctl --no-pager status ticket-router | head -5"

echo
echo "Done. Endpoint: https://$SITE"
echo "The API answers 503 until the first model is published. Run the training job next:"
[ -n "$TRAIN" ] && echo "  ssh $USER_NAME@$TRAIN 'sudo /srv/ticket-routing/deploy/train.sh'"
echo "API keys are in $SECRETS/api-keys.env (give GRADER_KEY to the instructor, never commit it)."
