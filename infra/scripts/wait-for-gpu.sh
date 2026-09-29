#!/usr/bin/env bash
# GPU droplets sell out. This keeps trying to create the training droplet until one is free.
#
#   infra/scripts/wait-for-gpu.sh
#
# Tries the cheapest NVIDIA size first (RTX 4000 Ada, $0.76/hr), then the RTX 6000 Ada
# ($1.57/hr), every two minutes for up to an hour. When one is created it writes the size
# that worked into terraform.tfvars (so later applies do not try to resize the droplet)
# and runs a full apply to finish the project assignment. Override with SIZES, TRIES, WAIT.
set -uo pipefail
cd "$(dirname "$0")/../terraform"
source ../scripts/tf-env.sh >/dev/null
SIZES=(${SIZES:-gpu-4000adax1-20gb gpu-6000adax1-48gb})
TRIES=${TRIES:-30}
WAIT=${WAIT:-120}
LOG=$(mktemp)

for i in $(seq 1 "$TRIES"); do
  for size in "${SIZES[@]}"; do
    echo "$(date +%H:%M:%S)  attempt $i  $size"
    if terraform apply -input=false -auto-approve -var training_enabled=true -var "training_size=$size" \
         -target=digitalocean_droplet.training >"$LOG" 2>&1; then
      echo "created the training droplet as $size"
      sed -i.bak -E "s/^training_size *=.*/training_size = \"$size\"/" terraform.tfvars && rm -f terraform.tfvars.bak
      grep -q '^training_size' terraform.tfvars || echo "training_size = \"$size\"" >> terraform.tfvars
      terraform apply -input=false -auto-approve
      echo "training droplet: $(terraform output -raw training_ip)"
      rm -f "$LOG"
      exit 0
    fi
    if ! grep -q "Size is not available" "$LOG"; then
      echo "failed for a reason other than capacity:"; cat "$LOG"; rm -f "$LOG"; exit 1
    fi
  done
  echo "            no capacity, trying again in $WAIT s"
  sleep "$WAIT"
done
echo "no GPU capacity after $TRIES rounds"; rm -f "$LOG"; exit 1
