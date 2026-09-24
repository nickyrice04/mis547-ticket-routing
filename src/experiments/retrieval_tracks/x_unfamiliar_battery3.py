"""Regularisation, family de-duplication weights and simple ensembling on the CV-unfamiliar condition. Core only."""
import json, time, sys
import numpy as np, scipy.sparse as sp
from scipy.sparse.csgraph import connected_components
from sklearn.linear_model import LogisticRegression
from experiments.retrieval_tracks.x_unfamiliar_lib import *
from experiments.retrieval_tracks.x_unfamiliar_cv import folds, fold_sims

xc, yc, xv, yv = load_core_val()
sim, nnidx, fold = fold_sims(xc, yc); unf = sim < 0.4
E = np.load("data/x_unfamiliar/emb_bge_base_core.npy"); E2 = np.load("data/x_unfamiliar/emb_mpnet_core.npy")

def family_ids(X, thr=0.5, block=2000):
    n = X.shape[0]; rows, cols = [], []
    XT = X.T.tocsr()
    for s in range(0, n, block):
        S = (X[s:s + block] @ XT).tocoo()
        m = S.data >= thr
        rows.append(S.row[m] + s); cols.append(S.col[m])
    G = sp.coo_matrix((np.ones(sum(len(r) for r in rows)), (np.concatenate(rows), np.concatenate(cols))), shape=(n, n))
    return connected_components(G, directed=False)[1]

P = {}
def put(name, te, proba):
    P.setdefault(name, np.zeros((len(yc), 10)))[te] = proba

t0 = time.time()
for f, (tr, te) in enumerate(folds(yc)):
    vec = base_vectorizer(); Xtr = vec.fit_transform([xc[i] for i in tr]); Xte = vec.transform([xc[i] for i in te])
    fam = family_ids(Xtr); size = np.bincount(fam)[fam]
    if f == 0: print("families in fold-0 train:", fam.max() + 1, "of", len(tr), "tickets; singletons:", (size == 1).sum(), flush=True)
    for C in (0.1, 0.3, 1.0, 3.0):
        put(f"lr_C{C}", te, LogisticRegression(C=C, max_iter=300).fit(Xtr, yc[tr]).predict_proba(Xte))
    for C in (1.0, 3.0, 10.0):
        put(f"lr_famw_C{C}", te, LogisticRegression(C=C, max_iter=300).fit(Xtr, yc[tr], sample_weight=1.0 / size).predict_proba(Xte))
        put(f"lr_famsqrt_C{C}", te, LogisticRegression(C=C, max_iter=300).fit(Xtr, yc[tr], sample_weight=1.0 / np.sqrt(size)).predict_proba(Xte))
    for C in (0.3, 1.0):
        put(f"bge_lr_C{C}", te, LogisticRegression(C=C, max_iter=500).fit(E[tr], yc[tr]).predict_proba(E[te]))
        put(f"mpnet_lr_C{C}", te, LogisticRegression(C=C, max_iter=500).fit(E2[tr], yc[tr]).predict_proba(E2[te]))
        put(f"bge_lr_famw_C{C}", te, LogisticRegression(C=C, max_iter=500).fit(E[tr], yc[tr], sample_weight=1.0 / size).predict_proba(E[te]))
    print("fold", f, f"{time.time()-t0:.0f}s", flush=True)

def acc(p, m=unf): return float((p.argmax(1)[m] == yc[m]).mean())
res = {k: {"unf": acc(v), "all": acc(v, np.ones(len(yc), bool))} for k, v in P.items()}
combos = {"lr_C1+bge_C1": ["lr_C1.0", "bge_lr_C1.0"], "lr_C1+bge_C1+mpnet_C1": ["lr_C1.0", "bge_lr_C1.0", "mpnet_lr_C1.0"],
          "lr_C0.3+bge_C0.3+mpnet_C0.3": ["lr_C0.3", "bge_lr_C0.3", "mpnet_lr_C0.3"], "lr_famw_C3+bge_famw_C1": ["lr_famw_C3.0", "bge_lr_famw_C1.0"]}
for k, names in combos.items():
    p = np.mean([np.log(P[n] + 1e-9) for n in names], axis=0); res["ens:" + k] = {"unf": acc(p), "all": acc(p, np.ones(len(yc), bool))}
for k, v in res.items(): print(f"{k:34s} unf={v['unf']:.4f} all={v['all']:.4f}")
json.dump(res, open("results/x_unfamiliar_battery3.json", "w"), indent=1)
np.savez_compressed("data/x_unfamiliar/cv_oof_battery3.npz", **P)
