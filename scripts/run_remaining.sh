#!/bin/bash
# Remaining runs, one at a time (they share the GPU).
cd "$(dirname "$0")/.."
export PYTHONPATH=src TOKENIZERS_PARALLELISM=false TRANSFORMERS_VERBOSITY=error
while pgrep -f "train_transformer.py distilbert-base-uncased 2_distilbert" >/dev/null; do sleep 20; done

echo "=== 2b: DistilBERT, no class weights, lr 5e-5 $(date +%H:%M) ==="
CLASS_WEIGHTS=0 EPOCHS=5 LR=5e-5 MAX_LEN=192 .venv/bin/python src/experiments/bigger_models/train_transformer.py \
  distilbert-base-uncased 2b_distilbert_noweights 2>&1 \
  | grep -E --line-buffered "EPOCH|class weights|wrote|Traceback|Error|Killed"

echo "=== 3: RoBERTa base $(date +%H:%M) ==="
CLASS_WEIGHTS=0 EPOCHS=5 LR=3e-5 MAX_LEN=192 .venv/bin/python src/experiments/bigger_models/train_transformer.py \
  roberta-base 3_roberta_base 2>&1 \
  | grep -E --line-buffered "EPOCH|class weights|wrote|Traceback|Error|Killed"

echo "=== 4: Qwen2.5-0.5B + LoRA $(date +%H:%M) ==="
EPOCHS=2 MAX_LEN=192 BATCH=16 .venv/bin/python src/experiments/bigger_models/train_slm.py Qwen/Qwen2.5-0.5B 4_qwen_0_5b 2 2>&1 \
  | grep -E --line-buffered "base params|wrote|Traceback|Error|Killed"

echo "=== LADDER COMPLETE $(date +%H:%M) ==="
