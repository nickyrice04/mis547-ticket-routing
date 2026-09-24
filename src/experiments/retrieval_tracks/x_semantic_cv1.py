"""CV strictly inside core: 1-NN per view, linear score fusion lex + lambda*dense, and weighted k-NN voting."""
import json, sys
import numpy as np
from sklearn.model_selection import StratifiedKFold
from experiments.retrieval_tracks.x_semantic_common import *

ctx, cy, vtx, vy = load_core_val()
DENSE = sys.argv[1:] or ["e5base", "bge", "mpnet", "gte"]
E = {n: np.load(CACHE / f"emb_{n}_core.npy") for n in DENSE}
skf = StratifiedKFold(5, shuffle=True, random_state=0)
LAMS = [0, 0.5, 1, 2, 3, 5, 8, 12, 1e6]
acc = {}
def add(key, ok):
    acc.setdefault(key, []).append(float(ok.mean()))

def knn_vote(S, ytr, k, T):
    idx = np.argpartition(-S, k, axis=1)[:, :k]
    s = np.take_along_axis(S, idx, 1)
    w = np.exp((s - s.max(1, keepdims=True)) / T)
    P = np.zeros((S.shape[0], 10))
    for c in range(10):
        P[:, c] = (w * (ytr[idx] == c)).sum(1)
    return P.argmax(1)

for f, (tr, te) in enumerate(skf.split(ctx, cy)):
    vec = baseline_tfidf(); Xtr = vec.fit_transform([ctx[i] for i in tr]); Xte = vec.transform([ctx[i] for i in te])
    SL = (Xte @ Xtr.T).toarray().astype(np.float32)
    ytr, yte = cy[tr], cy[te]
    add(("lex", "1nn"), ytr[SL.argmax(1)] == yte)
    for k in (5, 10):
        for T in (0.02, 0.05, 0.1):
            add(("lex", f"k{k}_T{T}"), knn_vote(SL, ytr, k, T) == yte)
    Ssum = np.zeros_like(SL)
    for n in DENSE:
        SD = E[n][te] @ E[n][tr].T
        Ssum += SD / len(DENSE)
        for lam in LAMS:
            add((n, f"lam{lam}"), ytr[(SL + lam * SD).argmax(1)] == yte)
    for lam in LAMS:
        add(("avgdense", f"lam{lam}"), ytr[(SL + lam * Ssum).argmax(1)] == yte)
    print("fold", f, "done", flush=True)
out = {f"{a}|{b}": round(float(np.mean(v)), 4) for (a, b), v in acc.items()}
for k, v in out.items():
    print(f"{k:28s} {v:.4f}")
json.dump(out, open(ROOT / "results/x_semantic_cv1.json", "w"), indent=1)
