"""Model selection for the stacker using LOO features inside core (5-fold over query tickets). No validation access."""
import sys, json
import numpy as np
from sklearn.model_selection import StratifiedKFold
from experiments.retrieval_tracks.x_semantic_common import *
from experiments.retrieval_tracks.x_semantic_stack import fit_meta, meta_proba, NC

tag = sys.argv[1]
d = np.load(CACHE / f"feats_{tag}_loo.npz", allow_pickle=True)
X, y, names = d["X"], d["y"], list(d["names"])
n = len(y)
t = (np.tile(np.arange(NC), n) == np.repeat(y, NC)).astype(int)
s1 = d["s1_lex"]
print("LOO lex 1-NN", (d["top1"] == y).mean().round(4), "fused 1-NN", (d["top1_fused"] == y).mean().round(4))
clf = X[:, names.index("clf_lr_tfidf")].reshape(n, NC).argmax(1)
print("simple hybrid (1-NN if s>=0.3 else LR)", (np.where(s1 >= 0.3, d["top1"], clf) == y).mean().round(4))

def run(cols, label, **kw):
    idx = [names.index(c) for c in cols]
    P = np.zeros((n, NC))
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=0).split(np.zeros(n), y):
        rtr = (tr[:, None] * NC + np.arange(NC)).ravel(); rte = (te[:, None] * NC + np.arange(NC)).ravel()
        m = fit_meta(X[rtr][:, idx], t[rtr], cols, **kw)
        P[te] = meta_proba(m, X[rte][:, idx])
    ok = P.argmax(1) == y
    print(f"{label:40s} acc {ok.mean():.4f}")
    return P, ok

allc = names
lex_only = [c for c in names if c.startswith("lex_") or c in ("prior", "class_id")]
lex_clf = lex_only + [c for c in names if "lr_tfidf" in c]
no_dense_views = lex_clf  # lexical kNN + lexical classifier
P0, ok0 = run(lex_only, "meta: lex kNN feats only")
P1, ok1 = run(lex_clf, "meta: lex kNN + tfidf LR")
P2, ok2 = run(allc, "meta: all (lex + 4 dense + fused + clfs)")
print(fmt_table(bucket_table(s1, d["top1"] == y, {"meta_lex": ok1, "meta_all": ok2})))
np.save(CACHE / f"cv2_{tag}_P_all.npy", P2)
