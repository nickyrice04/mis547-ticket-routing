"""Shared helpers for the "unfamiliar" track. Never touches the test split."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from common import load_meta, load_split

ROOT = Path(__file__).resolve().parents[3]   # retrieval_tracks -> experiments -> src -> repository root
SPLIT = ROOT / "data" / "synthetic" / "split.json"
BUCKETS = [(0.0, 0.2), (0.2, 0.3), (0.3, 0.4), (0.4, 0.5), (0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 1.01)]


def load_core_val():
    x, y = load_split("train")
    s = json.loads(SPLIT.read_text())
    core, val = s["core"], s["validation"]
    xc, yc = [x[i] for i in core], np.array([y[i] for i in core])
    xv, yv = [x[i] for i in val], np.array([y[i] for i in val])
    return xc, yc, xv, yv


def base_vectorizer():
    return TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=200_000, sublinear_tf=True)


def nn_search(A, B, exclude_self=False, block=2000):
    """For every row of sparse L2-normalised A, the best cosine match in B. Returns (sim, idx)."""
    sims = np.zeros(A.shape[0], dtype=np.float32)
    idx = np.zeros(A.shape[0], dtype=np.int64)
    BT = B.T.tocsr()
    for s in range(0, A.shape[0], block):
        S = (A[s:s + block] @ BT).toarray()
        if exclude_self:
            r = np.arange(S.shape[0])
            S[r, r + s] = -1.0
        idx[s:s + block] = S.argmax(1)
        sims[s:s + block] = S.max(1)
    return sims, idx


def bucket_table(sim, correct):
    rows = []
    for lo, hi in BUCKETS:
        m = (sim >= lo) & (sim < hi)
        rows.append({"bucket": f"{lo:.1f}-{min(hi, 1.0):.1f}", "share": float(m.mean()),
                     "n": int(m.sum()), "acc": float(correct[m].mean()) if m.any() else None})
    return rows


def fmt_table(rows):
    return "\n".join(f"{r['bucket']}: n={r['n']:5d} share={r['share']*100:5.1f}% acc={r['acc']*100:5.1f}%" for r in rows)


LABELS = load_meta()["labels"]
