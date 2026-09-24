"""Use core metadata (type, priority, tags, version) as ground truth for 'same family' and measure how much headroom
sibling retrieval/verification really has. Strictly inside core (LOO)."""
import json
import numpy as np
from experiments.retrieval_tracks.x_semantic_common import *
ctx, cy, _, _ = load_core_val()
n = len(ctx)
meta = json.load(open(CACHE / "core_meta.json"))
key = np.array([hash((m["type"], m["priority"], tuple(m["tags"]))) for m in meta])
tagset = [frozenset(m["tags"]) for m in meta]
vec = baseline_tfidf(); X = vec.fit_transform(ctx)
E = [np.load(CACHE / f"emb_{d}_core.npy") for d in ["e5base", "bge", "mpnet", "gte"]]
K = 30
NB = np.zeros((n, K), np.int64); NL = np.zeros((n, K), np.float32)
for a in range(0, n, 2000):
    b = min(a + 2000, n)
    SL = (X[a:b] @ X.T).toarray().astype(np.float32)
    S = SL + 5 * np.mean([e[a:b] @ e.T for e in E], 0)
    S[np.arange(b - a), np.arange(a, b)] = -9
    idx = np.argpartition(-S, K, 1)[:, :K]; o = np.argsort(-np.take_along_axis(S, idx, 1), 1); idx = np.take_along_axis(idx, o, 1)
    NB[a:b] = idx; NL[a:b] = np.take_along_axis(SL, idx, 1)
sib = key[NB] == key[:, None]
same = cy[NB] == cy[:, None]
print("P(same queue | meta-identical neighbour) =", same[sib].mean().round(4), " n pairs", int(sib.sum()))
print("P(same queue | not meta-identical)      =", same[~sib].mean().round(4))
print("P(meta-identical) by fused rank 1..8:", sib[:, :8].mean(0).round(3))
has = sib.any(1); first = np.where(has, sib.argmax(1), -1)
print("tickets with a meta-identical sibling in top-30:", has.mean().round(4), " of which sibling at rank1:", (first[has] == 0).mean().round(4), " rank<=3:", (first[has] <= 2).mean().round(4))
s1 = np.load(CACHE / "view_lex_loo.npz")["q_s1"]
P = np.load(CACHE / "cv2_pre4_P_all.npy"); ok = P.argmax(1) == cy
top1_lex = np.load(CACHE / "view_lex_loo.npz")["q_top1"]
print("bucket: n, share with sibling in top30, sibling at fused rank1 | stack acc with sibling, stack acc without, lex1nn acc with sibling")
for lo, hi in BUCKETS:
    m = (s1 >= lo) & (s1 < hi)
    print(f"  {lo:.1f}-{min(hi,1):.1f}: n={m.sum():5d} has_sib {has[m].mean():.3f} sib@1 {(first[m & has] == 0).mean() if (m&has).sum() else float('nan'):.3f} | stack|sib {ok[m & has].mean() if (m&has).sum() else float('nan'):.3f}  stack|nosib {ok[m & ~has].mean():.3f}  lex1nn|sib {(top1_lex==cy)[m & has].mean() if (m&has).sum() else float('nan'):.3f}")
print("overall: stack acc | sibling exists:", ok[has].mean().round(4), " | no sibling:", ok[~has].mean().round(4))
print("oracle if sibling label always used when exists, else stack:", (has * 1.0 * same[np.arange(n), np.maximum(first, 0)] + (~has) * ok).mean().round(4))
np.savez_compressed(CACHE / "family_diag.npz", NB=NB, NL=NL, sib=sib, same=same)
