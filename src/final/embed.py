"""Sentence embeddings for the final router, and the one-off embedding of the German pool.

The encoder is intfloat/multilingual-e5-base, a fixed pretrained model that is only run,
never fitted. The router embeds training and evaluation tickets on the fly from their
texts. This script's main() embeds the German pool once, both the English translation and
the original German text, and caches the two arrays under data/x_german/ because they
never change. Nothing here depends on which tickets are core, validation or test.

    PYTHONPATH=src python src/final/embed.py         # writes e5_german_translated.npy (and _original if missing)
    PYTHONPATH=src python src/final/embed.py orig    # force the original-German embeddings too
"""
from __future__ import annotations

import json
import os
import sys
import time

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
import numpy as np

from common import ROOT, clean

OUT = ROOT / "data" / "x_german"
MODEL = "intfloat/multilingual-e5-base"
# Pinned revision, the exact weights the router was fitted with (also baked into the Docker image).
REVISION = "d128750597153bb5987e10b1c3493a34e5a4502a"


def encoder():
    """Load the e5 encoder on the best available device (NVIDIA GPU, Apple GPU, or CPU), capped at 256 tokens per ticket."""
    import torch
    from sentence_transformers import SentenceTransformer
    dev = os.environ.get("EMBED_DEVICE") or (
        "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    enc = SentenceTransformer(MODEL, device=dev, revision=REVISION)
    enc.max_seq_length = 256
    return enc


def embed(enc, texts, batch_size=64):
    """L2-normalised embeddings, one row per text, in the original order.

    e5 expects a "query: " prefix on inputs. Texts are sorted by length before encoding
    so each batch pads as little as possible, then put back in order.
    """
    order = np.argsort([len(t) for t in texts])[::-1]
    out = enc.encode([f"query: {texts[i][:1000]}" for i in order], batch_size=batch_size,
                     normalize_embeddings=True, show_progress_bar=False)
    res = np.zeros_like(out)
    res[order] = out
    return res.astype(np.float32)


def main():
    """Embed the German pool once. Requires final/translate.py to have written data/x_german/."""
    rows = [json.loads(l) for l in open(OUT / "german_translated.jsonl")]
    raw = {r["row"]: r for r in map(json.loads, open(OUT / "german_raw.jsonl"))}
    de = [clean(raw[r["row"]]["subject"], raw[r["row"]]["body"]) for r in rows]   # original German, cleaned
    en = [r["text"] for r in rows]                                                # English translation
    enc = encoder()
    t0 = time.time()
    if "orig" in sys.argv[1:] or not (OUT / "e5_german_original.npy").exists():
        np.save(OUT / "e5_german_original.npy", embed(enc, de))
        print(f"german originals {time.time()-t0:.0f}s", flush=True)
    np.save(OUT / "e5_german_translated.npy", embed(enc, en))
    print(f"german translations {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
