#!/usr/bin/env bash
# Create the audit-log tables and the least-privilege grants (src/mlops/db.py), once per database.
#
#   infra/scripts/migrate-db.sh
#
# This needs the database's admin user (doadmin). The class DigitalOcean token is not
# allowed to read the admin password, so it comes from the control panel instead:
# Databases -> team3-ticket-routing-db -> Overview -> Connection details -> show password.
# The script asks for it without echoing it, sends it to the droplet over SSH on stdin,
# and never writes it anywhere. Set DB_ADMIN_PASSWORD to skip the prompt.
set -uo pipefail
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
cd "$ROOT/infra/terraform"
KEY="${KEY:-$HOME/.ssh/digitalocean_key}"
USER_NAME="${USER_NAME:-nicky}"
out() { terraform output -raw "$1"; }

if [ -z "${DB_ADMIN_PASSWORD:-}" ]; then
  read -r -s -p "doadmin password for team3-ticket-routing-db: " DB_ADMIN_PASSWORD
  echo
fi
# DigitalOcean passwords have no spaces, so strip any a copy and paste picked up.
DB_ADMIN_PASSWORD=$(printf '%s' "$DB_ADMIN_PASSWORD" | tr -d '[:space:]')
PW=$(python3 -c 'import sys, urllib.parse; print(urllib.parse.quote(sys.argv[1], safe=""))' "$DB_ADMIN_PASSWORD")
ADMIN_URL="postgresql+psycopg://doadmin:$PW@$(out db_private_host):$(out db_port)/$(out db_name)?sslmode=require"

ssh -i "$KEY" -o ConnectTimeout=15 "$USER_NAME@$(out inference_ip)" 'sudo bash -c "read -r ADMIN_DATABASE_URL && export ADMIN_DATABASE_URL && \
  cd /srv/ticket-routing/deploy && set -a && . /etc/ticket-routing/compose.env && set +a && \
  docker compose run --rm --no-deps -e PYTHONPATH=/app/src -e ADMIN_DATABASE_URL api python -m mlops.db migrate"' <<<"$ADMIN_URL"
status=$?

# Prove it worked from the API's side: connect as router_api (least privilege) and count rows.
if ssh -i "$KEY" -o ConnectTimeout=15 "$USER_NAME@$(out inference_ip)" 'sudo bash -c "cd /srv/ticket-routing/deploy && \
  set -a && . /etc/ticket-routing/compose.env && set +a && \
  docker compose run --rm --no-deps -e PYTHONPATH=/app/src api python -c \"from mlops import db; print(\\\"verified as router_api:\\\", db.counts())\""'; then
  echo "SUCCESS: the audit log is ready. Confident tickets will now be routed automatically."
else
  echo "FAILED: the tables are not usable yet. If the error above says 'password authentication failed',"
  echo "copy the doadmin password again from the control panel (Databases -> team3-ticket-routing-db -> Connection details)."
  exit 1
fi
