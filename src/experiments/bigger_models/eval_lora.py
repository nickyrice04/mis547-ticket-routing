"""Evaluate a LoRA fine-tuned language model on the same 500 test tickets.

This closes the gap in the comparison. Every earlier language model number was
zero-shot, which measured whether a model can guess our filing conventions
without being told. This measures the fair version: the same model, adapted to
our labels, against the classifiers that were also trained on our labels.

One prompt per ticket, because that is the format it was fine-tuned on. That
costs the batching trick that made zero-shot cheap, which is itself part of the
result.

    python src/experiments/bigger_models/eval_lora.py mlx-community/gemma-3-4b-it-qat-4bit adapters/gemma3_4b gemma3_4b_lora
"""
from __future__ import annotations

import json
import sys
import time

import numpy as np
from sklearn.metrics import accuracy_score, f1_score

from common import RESULTS, load_meta


def main() -> None:
    model_path, adapter_path, tag = sys.argv[1], sys.argv[2], sys.argv[3]
    limit = int(sys.argv[4]) if len(sys.argv) > 4 else 500

    import mlx.core as mx
    from mlx_lm import generate, load
    from mlx_lm.sample_utils import make_sampler

    labels = load_meta()["labels"]
    lower = {n.lower(): i for i, n in enumerate(labels)}
    test_file = sys.argv[5] if len(sys.argv) > 5 else "data/lora/test.jsonl"
    rows = [json.loads(l) for l in open(test_file)][:limit]

    model, tokenizer = load(model_path, adapter_path=adapter_path)
    sampler = make_sampler(temp=0.0)

    preds, gold, unparsed = [], [], 0
    t0 = time.time()
    for i, row in enumerate(rows):
        # MLX wraps prompt/completion pairs in the model's chat template during
        # LoRA training, so inference has to use it too. Feeding the bare prompt
        # makes the fine-tuned model emit end-of-sequence immediately and return
        # an empty string, which looks like 3% accuracy rather than a bug.
        text = tokenizer.apply_chat_template(
            [{"role": "user", "content": row["prompt"]}],
            add_generation_prompt=True, tokenize=False,
        )
        out = generate(model, tokenizer, prompt=text, max_tokens=8,
                       sampler=sampler, verbose=False)
        answer = out.strip().lower()
        hit = -1
        if answer in lower:
            hit = lower[answer]
        else:
            for name, idx in lower.items():
                if answer and (answer.startswith(name) or name.startswith(answer)):
                    hit = idx
                    break
        if hit < 0:
            unparsed += 1
        preds.append(hit)
        gold.append(lower[row["completion"].strip().lower()])
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(rows)}  {(time.time()-t0)/(i+1)*1000:.0f} ms each", flush=True)
    seconds = time.time() - t0

    preds, gold = np.asarray(preds), np.asarray(gold)
    result = {
        "tier": f"6_{tag}",
        "model": f"{model_path} + LoRA adapter (fine-tuned on our labels)",
        "eval_tickets": len(rows),
        "ms_per_ticket": round(seconds / len(rows) * 1000, 1),
        "unparsed": int(unparsed),
        "accuracy": round(float(accuracy_score(gold, preds)), 4),
        "macro_f1": round(float(f1_score(gold, preds, average="macro")), 4),
        "peak_memory_gb": round(mx.get_peak_memory() / 1e9, 2),
    }
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / f"{result['tier']}.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
