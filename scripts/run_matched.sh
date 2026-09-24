#!/bin/bash
# DistilBERT needed 12 epochs to converge. RoBERTa got 5 and Qwen got 2, so the
# ladder is not a fair comparison yet. Same budget for everyone: 12 epochs.
cd "$(dirname "$0")/.."
export PYTHONPATH=src TOKENIZERS_PARALLELISM=false TRANSFORMERS_VERBOSITY=error
echo "=== 3b: RoBERTa base, 12 epochs $(date +%H:%M) ==="
CLASS_WEIGHTS=0 EPOCHS=12 LR=3e-5 MAX_LEN=192 .venv/bin/python src/experiments/bigger_models/train_transformer.py \
  roberta-base 3b_roberta_12ep 2>&1 | grep -E --line-buffered "EPOCH|wrote|Traceback|Error|Killed"
echo "=== 4b: Qwen2.5-0.5B + LoRA, 12 epochs $(date +%H:%M) ==="
MAX_LEN=192 BATCH=16 .venv/bin/python src/experiments/bigger_models/train_slm.py Qwen/Qwen2.5-0.5B 4b_qwen_12ep 12 2>&1 \
  | grep -E --line-buffered "base params|wrote|Traceback|Error|Killed"
echo "=== MATCHED LADDER COMPLETE $(date +%H:%M) ==="
