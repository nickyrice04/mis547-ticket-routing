"""Embed core + validation with pretrained embedders (no fitting). Usage: python x_semantic_embed.py name1 name2 ..."""
import sys, time
import numpy as np
from experiments.retrieval_tracks.x_semantic_common import *

MODELS = {
    "e5base": ("intfloat/multilingual-e5-base", "query: "),
    "bge": ("BAAI/bge-base-en-v1.5", ""),
    "mpnet": ("sentence-transformers/all-mpnet-base-v2", ""),
    "gte": ("thenlper/gte-base", ""),
    "minilm": ("sentence-transformers/all-MiniLM-L12-v2", ""),
    "bgelarge": ("BAAI/bge-large-en-v1.5", ""),
    "e5large": ("intfloat/e5-large-v2", "query: "),
    "gtemodern": ("Alibaba-NLP/gte-modernbert-base", ""),
}
ctx, cy, vtx, vy = load_core_val()
for name in sys.argv[1:]:
    mn, prefix = MODELS[name]
    t = time.time()
    ec = embed(mn, ctx, f"{name}_core", prefix=prefix)
    ev = embed(mn, vtx, f"{name}_val", prefix=prefix)
    S = ev @ ec.T
    pred = cy[S.argmax(1)]
    print(f"{name}: {ec.shape} 1-NN acc {np.mean(pred == vy):.4f}  ({time.time()-t:.0f}s)", flush=True)
