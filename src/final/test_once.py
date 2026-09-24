"""The one and only test-set run for the two final routers.

Everything before this file was decided on the validation slice of the
training data. This script fits each router on the full training set and
scores the 4,750 test tickets once. Nothing here is tuned, and nothing is
rerun. If a number disappoints, it stays.

Two routers, run in this order:

    semantic   lookup plus four sentence embedders plus a learned stacker,
               trained on English tickets only
    german     the same idea with the translated German tickets added to the
               retrieval pool behind a leak guard

Each result goes to results/12_<router>.json with the same fields as every
earlier tier, plus the familiarity table (accuracy by similarity of the test
ticket to its nearest training ticket) and calibration.

    PYTHONPATH=src python src/final/test_once.py semantic
    PYTHONPATH=src python src/final/test_once.py german
"""
from __future__ import annotations

import json
import sys
import time

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics.pairwise import linear_kernel

from common import RESULTS, load_meta, load_split, threshold_table

ROUTERS = {
    "semantic": ("final.router_english_only", "12_stack_semantic",
                 "TF-IDF + 4 sentence embedders, nearest-neighbour features, gradient-boosted stacker"),
    "german": ("final.router", "12b_stack_german",
               "same stacker with 16.5k translated German tickets in the retrieval pool, leak guard at 0.60"),
}
BUCKETS = [(0, .2), (.2, .3), (.3, .4), (.4, .5), (.5, .6), (.6, .7), (.7, .8), (.8, 1.01)]


def ece(conf, correct, bins=15):
    """Expected calibration error: how far the model's confidence is from its actual accuracy.

    Tickets are grouped into 15 confidence bins. In each bin the gap between mean confidence
    and mean correctness is weighted by the bin's share of tickets. Zero means "when it says
    90% it is right 90% of the time".
    """
    edges = np.linspace(0, 1, bins + 1)
    e = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            e += m.mean() * abs(conf[m].mean() - correct[m].mean())
    return float(e)


def main() -> None:
    name = sys.argv[1]
    module, tier, desc = ROUTERS[name]
    out = RESULTS / f"{tier}.json"
    if out.exists():
        sys.exit(f"{out} already exists. The test set is scored once. Refusing to rerun.")

    labels = load_meta()["labels"]
    x_tr, y_tr = load_split("train")
    x_te, y_te = load_split("test")
    y_tr, y_te = np.asarray(y_tr), np.asarray(y_te)
    mod = __import__(module)

    t0 = time.time()
    P = np.asarray(mod.fit_predict_proba(x_tr, y_tr, x_te))
    minutes = (time.time() - t0) / 60
    pred = P.argmax(1)
    conf = P.max(1)
    correct = (pred == y_te).astype(float)

    # Familiarity: how close each test ticket is to its nearest training ticket,
    # measured with the plain baseline vectorizer so it matches earlier tables.
    v = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True).fit(x_tr)
    A, B = v.transform(x_tr), v.transform(x_te)
    sim = np.concatenate([linear_kernel(B[s:s + 500], A).max(1) for s in range(0, len(x_te), 500)])
    buckets = []
    for lo, hi in BUCKETS:
        m = (sim >= lo) & (sim < hi)
        buckets.append({"similarity": f"{lo:.1f}-{min(hi, 1):.1f}", "share": round(float(m.mean()), 4),
                        "n": int(m.sum()), "accuracy": round(float(correct[m].mean()), 4) if m.any() else None})

    per_queue = {q: {"n": int((y_te == i).sum()),
                     "recall": round(float(correct[y_te == i].mean()), 4)} for i, q in enumerate(labels)}
    res = {
        "tier": tier, "model": desc, "module": module,
        "accuracy": round(float(accuracy_score(y_te, pred)), 4),
        "macro_f1": round(float(f1_score(y_te, pred, average="macro")), 4),
        "ece": round(ece(conf, correct), 4),
        "unfamiliar_accuracy": round(float(correct[sim < 0.5].mean()), 4),
        "familiar_accuracy": round(float(correct[sim >= 0.5].mean()), 4),
        "train_minutes": round(minutes, 1),
        "by_similarity_to_nearest_training_ticket": buckets,
        "per_queue": per_queue,
        "thresholds": threshold_table(P, y_te),
    }
    np.save(RESULTS / f"{tier}_probs.npy", P)
    out.write_text(json.dumps(res, indent=2))
    print(json.dumps({k: v for k, v in res.items() if k not in ("thresholds", "per_queue")}, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
