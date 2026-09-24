"""Zero-shot classification with a purpose-built classifier, not a chat model.

This is the open-weight answer to what TypeSafe's Jev does. The model is not
generative. It never writes a label, it scores the labels you hand it and
returns a probability for each one, so it cannot invent a queue that does not
exist and it cannot fail to follow an output format.

Two things make this comparison interesting:

  * against Gemma 3 4B zero-shot, which scored 20.6% while being roughly nine
    times larger, it tests whether a model built for the task beats a general
    model improvising it
  * against our own fine-tuned models, it prices what zero-shot costs in
    accuracy, which is the price of never needing labelled data or a retrain

    python src/experiments/bigger_models/zeroshot_nli.py MoritzLaurer/deberta-v3-base-zeroshot-v2.0 deberta_base 500
"""
from __future__ import annotations

import json
import sys
import time

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score

from common import RESULTS, load_meta, load_split, threshold_table

# The label names as written are terse and overlapping. A hypothesis template
# gives the model a sentence to judge, which is how these models were trained.
TEMPLATE = "This customer support ticket should be handled by the {} team."


def main() -> None:
    model_name, tag = sys.argv[1], sys.argv[2]
    n_eval = int(sys.argv[3]) if len(sys.argv) > 3 else 500
    from transformers import pipeline

    labels = load_meta()["labels"]
    x_te, y_te = load_split("test")
    rng = np.random.default_rng(42)
    idx = rng.choice(len(x_te), size=min(n_eval, len(x_te)), replace=False)
    tickets = [x_te[i][:1500] for i in idx]
    gold = np.array([y_te[i] for i in idx])

    device = 0 if torch.backends.mps.is_available() else -1
    clf = pipeline("zero-shot-classification", model=model_name,
                   device="mps" if device == 0 else -1)

    probs = np.zeros((len(tickets), len(labels)))
    t0 = time.time()
    for i, text in enumerate(tickets):
        out = clf(text, candidate_labels=labels, hypothesis_template=TEMPLATE, multi_label=False)
        for lab, score in zip(out["labels"], out["scores"]):
            probs[i, labels.index(lab)] = score
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(tickets)}  {(time.time()-t0)/(i+1)*1000:.0f} ms each", flush=True)
    seconds = time.time() - t0

    pred = probs.argmax(1)
    result = {
        "tier": f"8_{tag}_zeroshot",
        "model": f"{model_name} (zero-shot, never trained on our labels)",
        "eval_tickets": len(tickets),
        "ms_per_ticket": round(seconds / len(tickets) * 1000, 1),
        "accuracy": round(float(accuracy_score(gold, pred)), 4),
        "macro_f1": round(float(f1_score(gold, pred, average="macro")), 4),
        "thresholds": threshold_table(probs, gold),
    }
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / f"{result['tier']}.json").write_text(json.dumps(result, indent=2))
    np.save(f"results/8_{tag}_probs.npy", probs)
    np.save(f"results/8_{tag}_idx.npy", idx)
    print(json.dumps({k: v for k, v in result.items() if k != "thresholds"}, indent=2))


if __name__ == "__main__":
    main()
