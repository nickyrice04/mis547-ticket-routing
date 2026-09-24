"""Shared helpers for the 'semantic' track. Core -> validation only. Never touches test."""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from common import load_meta, load_split  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "data" / "x_semantic"
CACHE.mkdir(parents=True, exist_ok=True)
BUCKETS = [(0.0, 0.2), (0.2, 0.3), (0.3, 0.4), (0.4, 0.5), (0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 1.01)]


def load_core_val():
    x, y = load_split("train")
    sp = json.loads((ROOT / "data/synthetic/split.json").read_text())
    ci, vi = sp["core"], sp["validation"]
    return ([x[i] for i in ci], np.array([y[i] for i in ci]),
            [x[i] for i in vi], np.array([y[i] for i in vi]))


def baseline_tfidf():
    from sklearn.feature_extraction.text import TfidfVectorizer
    return TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=200_000, sublinear_tf=True)


def bucket_table(sim, correct, extra=None):
    """sim: nearest baseline-tfidf similarity per eval ticket. correct: bool array. extra: dict name->bool array."""
    rows = []
    for lo, hi in BUCKETS:
        m = (sim >= lo) & (sim < hi)
        r = {"bucket": f"{lo:.1f}-{min(hi,1.0):.1f}", "share": round(float(m.mean()), 4), "n": int(m.sum()),
             "acc": round(float(correct[m].mean()), 4) if m.sum() else None}
        for k, v in (extra or {}).items():
            r[k] = round(float(v[m].mean()), 4) if m.sum() else None
        rows.append(r)
    return rows


def fmt_table(rows):
    keys = list(rows[0].keys())
    out = ["  ".join(f"{k:>9}" for k in keys)]
    for r in rows:
        out.append("  ".join(f"{(r[k] if r[k] is not None else float('nan')):>9.4f}" if isinstance(r[k], float) or r[k] is None
                             else f"{r[k]:>9}" for k in keys))
    return "\n".join(out)


def embed(model_name, texts, tag, prefix="", batch_size=64, max_seq_length=256, device="mps", force=False):
    """Embed texts with a sentence-transformers model, cached by tag. L2-normalised float32."""
    path = CACHE / f"emb_{tag}.npy"
    if path.exists() and not force:
        return np.load(path)
    import torch
    from sentence_transformers import SentenceTransformer
    torch.set_num_threads(3)
    m = SentenceTransformer(model_name, device=device)
    m.max_seq_length = max_seq_length
    # sort by length for speed
    order = np.argsort([len(t) for t in texts])[::-1]
    sorted_texts = [prefix + texts[i] for i in order]
    e = m.encode(sorted_texts, batch_size=batch_size, normalize_embeddings=True, show_progress_bar=False,
                 convert_to_numpy=True)
    out = np.empty_like(e)
    out[order] = e
    out = out.astype(np.float32)
    np.save(path, out)
    return out
