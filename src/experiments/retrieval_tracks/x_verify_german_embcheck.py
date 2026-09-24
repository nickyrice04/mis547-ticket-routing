"""Reviewer check for track "german": are the cached German e5 embeddings what they claim to be?

Re-embeds a random sample of German tickets (translated text and original German text) with the stock
intfloat/multilingual-e5-base on CPU and compares with the cached rows. A cached row built from anything
other than that ticket's own text (misalignment, or an embedding of some evaluation ticket) would show up
as a cosine well below 1.
"""
from __future__ import annotations

import json
import os

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
import numpy as np
from sentence_transformers import SentenceTransformer

from common import ROOT, clean

G = ROOT / "data" / "x_german"
rows = [json.loads(l) for l in open(G / "german_translated.jsonl")]
raw = {r["row"]: r for r in map(json.loads, open(G / "german_raw.jsonl"))}
Et, Eo = np.load(G / "e5_german_translated.npy"), np.load(G / "e5_german_original.npy")
rng = np.random.default_rng(7)
idx = np.sort(rng.choice(len(rows), 300, replace=False))
enc = SentenceTransformer("intfloat/multilingual-e5-base", device="cpu")
enc.max_seq_length = 256
en = [f"query: {rows[i]['text'][:1000]}" for i in idx]
de = [f"query: {clean(raw[rows[i]['row']]['subject'], raw[rows[i]['row']]['body'])[:1000]}" for i in idx]
Ft = enc.encode(en, batch_size=16, normalize_embeddings=True, show_progress_bar=False)
Fo = enc.encode(de, batch_size=16, normalize_embeddings=True, show_progress_bar=False)
ct = (Ft * Et[idx]).sum(1)
co = (Fo * Eo[idx]).sum(1)
out = {"n": len(idx), "translated_cos_min": float(ct.min()), "translated_cos_mean": float(ct.mean()),
       "original_cos_min": float(co.min()), "original_cos_mean": float(co.mean()),
       "translated_below_0.999": int((ct < 0.999).sum()), "original_below_0.999": int((co < 0.999).sum()),
       "norms_ok": bool(np.allclose(np.linalg.norm(Et, axis=1), 1, atol=1e-3) and np.allclose(np.linalg.norm(Eo, axis=1), 1, atol=1e-3))}
print(json.dumps(out, indent=1))
(ROOT / "results" / "x_verify_german_embcheck.json").write_text(json.dumps(out, indent=1))
