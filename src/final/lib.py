"""Small shared helpers for the final router: data loading, the baseline vectorizer, top-k similarity.

Protocol reminder: development runs train on the core of the training set and evaluate on
the validation slice (data/synthetic/split.json). The test set is never opened here.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.neural_network import MLPClassifier

from common import ROOT, load_split

GERMAN = ROOT / "data" / "x_german" / "german_translated.jsonl"
SPLIT = ROOT / "data" / "synthetic" / "split.json"
# Similarity bands used in every "accuracy by familiarity" table in the project.
BUCKETS = [(0.0, 0.2), (0.2, 0.3), (0.3, 0.4), (0.4, 0.5), (0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 1.01)]
N_CLASSES = 10


def load_core_val():
    """The development split: (core texts, core labels, validation texts, validation labels)."""
    x, y = load_split("train")
    s = json.loads(SPLIT.read_text())
    y = np.asarray(y)
    core, val = s["core"], s["validation"]
    return [x[i] for i in core], y[core], [x[i] for i in val], y[val]


def load_german(path: Path = GERMAN):
    """The translated German tickets as (texts, labels, raw rows)."""
    rows = [json.loads(l) for l in open(path)]
    return [r["text"] for r in rows], np.asarray([r["label"] for r in rows]), rows


def baseline_vec(**kw):
    """The project's standard TF-IDF: word unigrams and bigrams, min_df 2, 200k features, sublinear tf."""
    args = dict(ngram_range=(1, 2), min_df=2, max_features=200_000, sublinear_tf=True)
    args.update(kw)
    return TfidfVectorizer(**args)


def topk_sims(Q, P, k=1, block=1000):
    """Cosine top-k of each row of Q against rows of P (both L2-normalised sparse).

    Returns (sims [n, k], indices [n, k]) sorted best first. Works in blocks of rows so the
    dense similarity block never exceeds block x len(P) floats.
    """
    sims = np.zeros((Q.shape[0], k), dtype=np.float32)
    idx = np.zeros((Q.shape[0], k), dtype=np.int64)
    PT = P.T.tocsr()
    for s in range(0, Q.shape[0], block):
        d = (Q[s:s + block] @ PT).toarray()
        if k == 1:
            j = d.argmax(1)[:, None]
        else:
            j = np.argpartition(-d, kth=min(k, d.shape[1] - 1), axis=1)[:, :k]
            order = np.argsort(-np.take_along_axis(d, j, 1), axis=1)
            j = np.take_along_axis(j, order, 1)
        idx[s:s + block] = j
        sims[s:s + block] = np.take_along_axis(d, j, 1)
    return sims, idx


def max_sim(Q, P, block=1000):
    """Similarity of each row of Q to its single nearest row of P."""
    s, _ = topk_sims(Q, P, 1, block)
    return s[:, 0]


def make_mlp(seed=42):
    """The baseline network from src/baselines/train_mlp.py, one hidden layer of 256."""
    return MLPClassifier(hidden_layer_sizes=(256,), max_iter=60, early_stopping=True,
                         n_iter_no_change=5, random_state=seed)


def bucket_table(sim, y, preds: dict[str, np.ndarray]):
    """Accuracy of each named prediction inside each similarity band of BUCKETS."""
    rows = []
    for lo, hi in BUCKETS:
        m = (sim >= lo) & (sim < hi)
        row = {"bucket": f"{lo:.1f}-{min(hi, 1.0):.1f}", "share": round(float(m.mean()), 4), "n": int(m.sum())}
        for name, p in preds.items():
            row[name] = round(float((p[m] == y[m]).mean()), 4) if m.any() else None
        rows.append(row)
    return rows


def fmt_table(rows):
    """Print-friendly version of bucket_table's rows."""
    keys = list(rows[0].keys())
    out = ["  ".join(f"{k:>10s}" for k in keys)]
    for r in rows:
        out.append("  ".join(f"{r[k]:>10}" if not isinstance(r[k], float) else f"{r[k]*100:9.1f}%" for k in keys))
    return "\n".join(out)


def scores(y, p):
    """(accuracy, macro-F1) rounded to four places."""
    return round(float(accuracy_score(y, p)), 4), round(float(f1_score(y, p, average="macro")), 4)
