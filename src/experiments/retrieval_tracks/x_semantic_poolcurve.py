"""How does accuracy grow with pool size? Inside core only: queries = fixed random 3000 core tickets, pool = random
subsets of the remaining core tickets. Fits the 'sibling coverage' model and extrapolates to the full-train pool size."""
import json
import numpy as np
from experiments.retrieval_tracks.x_semantic_common import *
ctx, cy, _, _ = load_core_val()
n = len(ctx); rng = np.random.default_rng(0)
perm = rng.permutation(n); q = perm[:3000]; rest = perm[3000:]
vec = baseline_tfidf(); X = vec.fit_transform(ctx)
E = [np.load(CACHE / f"emb_{d}_core.npy") for d in ["e5base", "bge", "mpnet", "gte"]]
SL = (X[q] @ X[rest].T).toarray().astype(np.float32)
SF = SL + 5 * np.mean([e[q] @ e[rest].T for e in E], 0)
out = []
for frac in [0.25, 0.4, 0.55, 0.7, 0.85, 1.0]:
    accs_l, accs_f, hi = [], [], []
    for r in range(3):
        cols = rng.choice(len(rest), int(frac * len(rest)), replace=False)
        accs_l.append((cy[rest[cols]][SL[:, cols].argmax(1)] == cy[q]).mean())
        accs_f.append((cy[rest[cols]][SF[:, cols].argmax(1)] == cy[q]).mean())
        hi.append((SL[:, cols].max(1) >= 0.5).mean())
    out.append({"pool": int(frac * len(rest)), "lex_1nn": float(np.mean(accs_l)), "fused_1nn": float(np.mean(accs_f)), "share_sim>=0.5": float(np.mean(hi))})
    print(out[-1], flush=True)
# linear-in-log extrapolation of the last 4 points to pool sizes 16147 (validation protocol) and 18997 (final train->test)
ps = np.array([o["pool"] for o in out]); 
for k in ("lex_1nn", "fused_1nn", "share_sim>=0.5"):
    a = np.array([o[k] for o in out]); c = np.polyfit(np.log(ps[-4:]), a[-4:], 1)
    print(f"{k}: extrapolated @16147 = {np.polyval(c, np.log(16147)):.4f}, @18997 = {np.polyval(c, np.log(18997)):.4f}  (delta {np.polyval(c, np.log(18997)) - np.polyval(c, np.log(16147)):+.4f})")
json.dump(out, open(ROOT / "results/x_semantic_poolcurve.json", "w"), indent=1)
