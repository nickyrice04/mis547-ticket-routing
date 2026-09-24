"""Like x_semantic_peek, plus the rank of the first same-queue core ticket and the dense-embedding neighbours, for the hidden-sibling question."""
import numpy as np
from experiments.retrieval_tracks.x_semantic_common import *
from common import load_meta
L = load_meta()["labels"]
ctx, cy, vtx, vy = load_core_val()
S = np.load(CACHE / "S_lex_val_core.npy")
SD = np.load(CACHE / "emb_mpnet_val.npy") @ np.load(CACHE / "emb_mpnet_core.npy").T
sim = S.max(1); nn = S.argmax(1)
rng = np.random.default_rng(1)
idx = np.where((sim >= 0.3) & (sim < 0.5) & (cy[nn] != vy))[0]
# stats: for wrong mid-band tickets, rank of first same-label core ticket in lexical ranking
ranks = []
for i in idx:
    order = np.argsort(-S[i])[:50]
    hit = np.where(cy[order] == vy[i])[0]
    ranks.append(hit[0] if len(hit) else 99)
ranks = np.array(ranks)
print("n wrong mid", len(idx), "first same-label rank pct: <=2:", (ranks <= 2).mean().round(3), "<=5:", (ranks <= 5).mean().round(3), "<=10:", (ranks <= 10).mean().round(3))
for i in rng.choice(idx, 8, replace=False):
    print("=" * 100)
    print(f"VAL [{L[vy[i]]}] sim={sim[i]:.2f}\n  {vtx[i][:500]}")
    top = np.argsort(-S[i])[:6]
    for j in top:
        print(f"  lex-> [{L[cy[j]]}] lex={S[i, j]:.2f} dense={SD[i,j]:.2f}: {ctx[j][:260]}")
    top = np.argsort(-SD[i])[:3]
    for j in top:
        print(f"  dense-> [{L[cy[j]]}] lex={S[i, j]:.2f} dense={SD[i,j]:.2f}: {ctx[j][:260]}")
