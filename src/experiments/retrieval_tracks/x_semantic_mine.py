"""Mine contrastive training pairs ONLY from core. 80% of core = fine-tune set, 20% = dev slice the embedder never sees.
Positives: same queue + close under the fused (lexical + pretrained dense) similarity. Hard negatives: close but other queue."""
import json, hashlib
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from experiments.retrieval_tracks.x_semantic_common import *

ctx, cy, vtx, vy = load_core_val()
n = len(ctx)
ft, dev = next(StratifiedShuffleSplit(1, test_size=0.2, random_state=7).split(np.zeros(n), cy))
ft, dev = np.sort(ft), np.sort(dev)
json.dump({"ft": ft.tolist(), "dev": dev.tolist()}, open(CACHE / "ft_split.json", "w"))
json.dump(sorted(hashlib.md5(ctx[i].encode()).hexdigest() for i in ft), open(CACHE / "ft_seen_md5.json", "w"))

vec = baseline_tfidf(); X = vec.fit_transform(ctx)
dense = ["e5base", "bge", "mpnet", "gte"]
E = [np.load(CACHE / f"emb_{d}_core.npy") for d in dense]
Xf = X[ft]; yf = cy[ft]
K = 30
nbr = np.zeros((len(ft), K), np.int64); nbs = np.zeros((len(ft), K), np.float32); nbl = np.zeros((len(ft), K), np.float32)
for a in range(0, len(ft), 2000):
    b = min(a + 2000, len(ft))
    SL = (Xf[a:b] @ Xf.T).toarray().astype(np.float32)
    SD = np.mean([e[ft[a:b]] @ e[ft].T for e in E], 0)
    S = SL + 5.0 * SD
    S[np.arange(b - a), np.arange(a, b)] = -9
    idx = np.argpartition(-S, K, axis=1)[:, :K]
    s = np.take_along_axis(S, idx, 1); o = np.argsort(-s, 1)
    idx = np.take_along_axis(idx, o, 1)
    nbr[a:b] = idx; nbs[a:b] = np.take_along_axis(S, idx, 1); nbl[a:b] = np.take_along_axis(SL, idx, 1)
same = yf[nbr] == yf[:, None]
# purity diagnostics: P(same queue) by rank and by lexical sim of the neighbour
print("P(same queue) by fused rank 1..10:", same[:, :10].mean(0).round(3))
for lo, hi in [(0, .2), (.2, .3), (.3, .4), (.4, .5), (.5, .6), (.6, 1.01)]:
    m = (nbl >= lo) & (nbl < hi) & (np.arange(K)[None, :] < 5)
    print(f"  lex {lo}-{hi}: top5 pairs {int(m.sum())}, P(same) {same[m].mean():.3f}")
np.savez_compressed(CACHE / "ft_neighbours.npz", ft=ft, nbr=nbr, nbs=nbs, nbl=nbl, same=same)
