"""Print six mid-band validation tickets that lexical 1-NN got wrong, with their five nearest core tickets, to read what went wrong."""
import numpy as np
from experiments.retrieval_tracks.x_semantic_common import *
from common import load_meta
L = load_meta()["labels"]
ctx, cy, vtx, vy = load_core_val()
S = np.load(CACHE / "S_lex_val_core.npy")
sim = S.max(1); nn = S.argmax(1)
rng = np.random.default_rng(0)
idx = np.where((sim >= 0.3) & (sim < 0.5) & (cy[nn] != vy))[0]
for i in rng.choice(idx, 6, replace=False):
    print("=" * 100)
    print(f"VAL [{L[vy[i]]}] sim={sim[i]:.2f}\n  {vtx[i][:600]}")
    top = np.argsort(-S[i])[:5]
    for j in top:
        print(f"  -> core [{L[cy[j]]}] {S[i, j]:.2f}: {ctx[j][:300]}")
