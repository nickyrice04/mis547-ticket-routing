#!/usr/bin/env bash
# Pull-based continuous deployment and a health watchdog, run every five minutes by
# ticket-router-update.timer.
#
# GitHub Actions builds, scans and publishes the API image on every push to main. This
# script, on the droplet, notices a new image and rolls to it. The droplet reaches out to
# the registry, nothing reaches in, so CI needs no SSH key and SSH stays closed to the
# internet. If the new container is not healthy within five minutes, the previous image
# is put back.
set -euo pipefail
cd /srv/ticket-routing/deploy
set -a; . /etc/ticket-routing/compose.env; set +a

# Watchdog. Docker restarts a container that exits, but not one that is running and
# unhealthy (a hung process). This restarts the API if its health check is failing.
api_container=$(docker compose ps -q api 2>/dev/null || true)
if [ -n "$api_container" ] && [ "$(docker inspect -f '{{.State.Health.Status}}' "$api_container" 2>/dev/null)" = unhealthy ]; then
  echo "api container unhealthy, restarting it"
  docker compose restart api
fi

# Configuration changes (this file, Compose, Caddy) arrive with the repository.
git -C /srv/ticket-routing pull --ff-only --quiet || echo "git pull failed, keeping the current checkout"

before=$(docker image inspect --format '{{.Id}}' "$API_IMAGE" 2>/dev/null || echo none)
[ "$before" != none ] && docker tag "$API_IMAGE" ticket-router-api:previous
if ! docker compose pull --quiet api 2>/dev/null; then
  echo "no image to pull yet (CI has not published one), nothing to do"
  exit 0
fi
after=$(docker image inspect --format '{{.Id}}' "$API_IMAGE")
if [ "$before" = "$after" ]; then
  exit 0
fi

echo "new image ${after:7:12}, rolling the api container"
docker compose up -d api
container=$(docker compose ps -q api)
for _ in $(seq 1 60); do
  status=$(docker inspect -f '{{.State.Health.Status}}' "$container" 2>/dev/null || echo unknown)
  if [ "$status" = healthy ]; then
    echo "api healthy on ${after:7:12}"
    exit 0
  fi
  sleep 5
done

echo "new image unhealthy after 5 minutes, rolling back"
docker tag ticket-router-api:previous "$API_IMAGE"
docker compose up -d api
exit 1
