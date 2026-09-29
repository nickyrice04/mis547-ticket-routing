#!/usr/bin/env bash
# The training job, run on the GPU droplet:   sudo /srv/ticket-routing/deploy/train.sh [--reuse-german]
#
# Builds the Python environment (torch with CUDA from PyPI), then runs the pipeline in
# src/mlops/train_pipeline.py: data, translation on the GPU, embeddings, validation,
# the quality gate, the production fit, and publishing to Spaces and PostgreSQL.
# Credentials come from /etc/ticket-routing/trainer.env, root-only.
set -euo pipefail
cd /srv/ticket-routing
set -a; . /etc/ticket-routing/trainer.env; set +a

if [ ! -x .venv-train/bin/python ]; then
  uv venv --python 3.12 .venv-train
  VIRTUAL_ENV=.venv-train uv pip install -r docker/requirements-train.txt
fi
.venv-train/bin/python -c "import torch; assert torch.cuda.is_available(), 'no CUDA device'; print('GPU:', torch.cuda.get_device_name(0))"

export PYTHONPATH=src TOKENIZERS_PARALLELISM=false HF_HUB_DISABLE_XET=1
mkdir -p logs
.venv-train/bin/python -m mlops.train_pipeline "$@" 2>&1 | tee "logs/train-$(date -u +%Y%m%dT%H%M%SZ).log"
