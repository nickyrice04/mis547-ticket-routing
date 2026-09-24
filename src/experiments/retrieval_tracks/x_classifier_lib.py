"""Shared helpers for the "classifier" research track.

Protocol: train on core, score on validation. The test set is never touched here.
Every call to score_on_validation() appends one line to results/x_classifier_val_runs.jsonl
so the number of configurations that ever saw validation labels can be counted honestly.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score

from common import load_meta, load_split

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
VAL_LOG = RESULTS / "x_classifier_val_runs.jsonl"
CV_LOG = RESULTS / "x_classifier_cv_runs.jsonl"

BUCKETS = [(0.0, 0.2), (0.2, 0.3), (0.3, 0.4), (0.4, 0.5), (0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 1.0001)]
BASE_VEC = dict(ngram_range=(1, 2), min_df=2, max_features=200_000, sublinear_tf=True)


def load_core_val():
    x, y = load_split("train")
    split = json.loads((ROOT / "data/synthetic/split.json").read_text())
    core, val = split["core"], split["validation"]
    xc = [x[i] for i in core]
    yc = np.array([y[i] for i in core])
    xv = [x[i] for i in val]
    yv = np.array([y[i] for i in val])
    return xc, yc, xv, yv


def nearest_sim(train_texts, eval_texts, chunk=2000):
    """Cosine similarity (baseline TF-IDF) of each eval ticket to its nearest training ticket, plus that ticket's index."""
    vec = TfidfVectorizer(**BASE_VEC)
    a = vec.fit_transform(train_texts)
    b = vec.transform(eval_texts)
    sims = np.zeros(b.shape[0])
    idx = np.zeros(b.shape[0], dtype=int)
    for s in range(0, b.shape[0], chunk):
        d = (b[s:s + chunk] @ a.T).toarray()
        sims[s:s + chunk] = d.max(axis=1)
        idx[s:s + chunk] = d.argmax(axis=1)
    return sims, idx


def bucket_table(sims, pred, y_true):
    pred = np.asarray(pred)
    y_true = np.asarray(y_true)
    rows = []
    for lo, hi in BUCKETS:
        m = (sims >= lo) & (sims < hi)
        n = int(m.sum())
        rows.append({
            "bucket": f"{lo:.1f}-{min(hi, 1.0):.1f}",
            "share": round(n / len(y_true), 4),
            "n": n,
            "accuracy": round(float((pred[m] == y_true[m]).mean()), 4) if n else None,
        })
    return rows


def fmt_buckets(rows):
    return "  ".join(f"{r['bucket']}:{(r['accuracy'] or 0) * 100:.1f}" for r in rows)


def score(pred, y_true):
    return float(accuracy_score(y_true, pred)), float(f1_score(y_true, pred, average="macro"))


def log_val(name, pred, y_val, sims=None, extra=None):
    acc, f1 = score(pred, y_val)
    rec = {"name": name, "val_acc": round(acc, 4), "val_macro_f1": round(f1, 4), "time": time.strftime("%Y-%m-%d %H:%M:%S")}
    if sims is not None:
        rec["buckets"] = bucket_table(sims, pred, y_val)
    if extra:
        rec.update(extra)
    with VAL_LOG.open("a") as f:
        f.write(json.dumps(rec) + "\n")
    msg = f"[VAL] {name}: acc={acc:.4f} f1={f1:.4f}"
    if sims is not None:
        msg += "  | " + fmt_buckets(rec["buckets"])
    print(msg, flush=True)
    return rec


def log_cv(name, accs, extra=None):
    rec = {"name": name, "cv_mean": round(float(np.mean(accs)), 4), "cv_folds": [round(float(a), 4) for a in accs],
           "time": time.strftime("%Y-%m-%d %H:%M:%S")}
    if extra:
        rec.update(extra)
    with CV_LOG.open("a") as f:
        f.write(json.dumps(rec) + "\n")
    print(f"[CV] {name}: mean={rec['cv_mean']:.4f} folds={rec['cv_folds']}" + (f" {extra}" if extra else ""), flush=True)
    return rec
