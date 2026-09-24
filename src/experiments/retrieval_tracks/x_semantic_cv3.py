"""Stack comparison from cached views. Model selection by 5-fold CV over LOO rows inside core. No validation labels touched.
   python x_semantic_cv3.py "<label>=view1,view2,...|clf1,clf2" ... """
import sys, json
import numpy as np
from sklearn.model_selection import StratifiedKFold
from experiments.retrieval_tracks.x_semantic_common import *
from experiments.retrieval_tracks.x_semantic_stack import fit_meta, meta_proba, NC

_, cy, _, vy = load_core_val()
n = len(cy)
prior = np.bincount(cy, minlength=NC) / n

def load_clf(name, which):
    if name in ("lr_tfidf", "lr_dense"):   # from the first feature build
        d = np.load(CACHE / f"feats_pre4_{which}.npz", allow_pickle=True)
        return d["X"][:, list(d["names"]).index(f"clf_{name}")].reshape(-1, NC)
    return np.load(CACHE / f"clf_{name}_{which}.npy")

def assemble(views, clfs, which):
    nq = n if which == "loo" else len(vy)
    cols, names, top1s = [], [], []
    for v in views:
        d = np.load(CACHE / f"view_{v}_{which}.npz")
        for k in ("max", "gap", "vote", "cnt5"):
            cols.append(d[f"c_{k}"].reshape(-1)); names.append(f"{v}_{k}")
        for k in ("s1", "s5"):
            cols.append(np.repeat(d[f"q_{k}"], NC)); names.append(f"{v}_{k}")
        top1s.append(d["q_top1"])
    top1 = np.stack(top1s, 1)
    cols.append(np.stack([(top1 == c).sum(1) for c in range(NC)], 1).reshape(-1).astype(np.float32)); names.append("top1_agree")
    for c in clfs:
        p = load_clf(c, which)
        cols.append(p.reshape(-1)); names.append(f"clf_{c}")
        cols.append((p - p.max(1, keepdims=True)).reshape(-1)); names.append(f"clfgap_{c}")
    cols.append(np.tile(prior, nq)); names.append("prior")
    cols.append(np.tile(np.arange(NC), nq).astype(np.float32)); names.append("class_id")
    return np.stack(cols, 1).astype(np.float32), names

def cv(X, names, rows=None, seed=0, **kw):
    q = np.arange(n) if rows is None else rows
    t = (np.tile(np.arange(NC), n) == np.repeat(cy, NC)).astype(int)
    P = np.zeros((n, NC))
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=seed).split(q, cy[q]):
        rtr = (q[tr][:, None] * NC + np.arange(NC)).ravel(); rte = (q[te][:, None] * NC + np.arange(NC)).ravel()
        P[q[te]] = meta_proba(fit_meta(X[rtr], t[rtr], names, **kw), X[rte])
    return P

if __name__ == "__main__":
    s1 = np.load(CACHE / "view_lex_loo.npz")["q_s1"]
    res = {}
    for spec in sys.argv[1:]:
        label, rest = spec.split("=")
        vs, cs = rest.split("|")
        X, names = assemble(vs.split(","), [c for c in cs.split(",") if c], "loo")
        P = cv(X, names); ok = P.argmax(1) == cy
        res[label] = float(ok.mean())
        print(f"{label:30s} F={X.shape[1]:3d} LOO-CV acc {ok.mean():.4f}   mid(0.3-0.5) {ok[(s1>=.3)&(s1<.5)].mean():.4f}  low(<0.3) {ok[s1<.3].mean():.4f}  high {ok[s1>=.5].mean():.4f}", flush=True)
    json.dump(res, open(ROOT / "results/x_semantic_cv3_last.json", "w"), indent=1)
