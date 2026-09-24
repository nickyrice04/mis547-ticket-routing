#!/bin/bash
# Every transformer so far was still improving when training stopped, so the
# obvious test is simply to train longer. 12 epochs instead of 5.
cd "$(dirname "$0")/.."
export PYTHONPATH=src TOKENIZERS_PARALLELISM=false TRANSFORMERS_VERBOSITY=error
while pgrep -f "train_slm.py" >/dev/null; do sleep 60; done
echo "=== 2c: DistilBERT, 12 epochs, no class weights $(date +%H:%M) ==="
CLASS_WEIGHTS=0 EPOCHS=12 LR=5e-5 MAX_LEN=192 .venv/bin/python src/experiments/bigger_models/train_transformer.py \
  distilbert-base-uncased 2c_distilbert_12ep 2>&1 \
  | grep -E --line-buffered "EPOCH|wrote|Traceback|Error|Killed"
echo "=== DONE $(date +%H:%M) ==="
