"""Honest evaluation of a fine-tuned embedder: only DEV-slice tickets of core (never seen by the fine-tune) are used
as queries / meta-training rows. Pool = all other core tickets. No validation access.
   python x_semantic_devstack.py ft1_bge [more ft tags]"""
import sys, json
import numpy as np
from sklearn.model_selection import StratifiedKFold
from experiments.retrieval_tracks.x_semantic_common import *
from experiments.retrieval_tracks.x_semantic_cv3 import assemble, cy, NC, n
from experiments.retrieval_tracks.x_semantic_stack import fit_meta, meta_proba

fts = sys.argv[1:]
dev = np.array(json.load(open(CACHE / "ft_split.json"))["dev"]); ftidx = np.array(json.load(open(CACHE / "ft_split.json"))["ft"])
s1 = np.load(CACHE / "view_lex_loo.npz")["q_s1"]
print(f"dev queries {len(dev)}")
for v in ["lex", "bge", "e5base", "fused4"] + fts:
    t1 = np.load(CACHE / f"view_{v}_loo.npz")["q_top1"]
    print(f"  1-NN {v:10s} dev acc {(t1[dev] == cy[dev]).mean():.4f}   (ft-seen rows acc {(t1[ftidx] == cy[ftidx]).mean():.4f})")

P1 = np.load(CACHE / "cv2_pre4_P_all.npy")           # honest 5-fold CV predictions of the pretrained stack for every core ticket
print(f"stage-1 pre4 stack on dev rows: {(P1.argmax(1)[dev] == cy[dev]).mean():.4f}")
t = (np.tile(np.arange(NC), n) == np.repeat(cy, NC)).astype(int)

def cv_dev(X, names, reps=3, **kw):
    accs, Pm = [], np.zeros((n, NC))
    for r in range(reps):
        P = np.zeros((n, NC))
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=r).split(dev, cy[dev]):
            rtr = (dev[tr][:, None] * NC + np.arange(NC)).ravel(); rte = (dev[te][:, None] * NC + np.arange(NC)).ravel()
            P[dev[te]] = meta_proba(fit_meta(X[rtr], t[rtr], names, seed=r, **kw), X[rte])
        accs.append((P.argmax(1)[dev] == cy[dev]).mean()); Pm += P / reps
    return float(np.mean(accs)), float(np.std(accs)), Pm

small = dict(max_iter=200, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=40, l2_regularization=2.0)
base_views = "lex,e5base,bge,mpnet,gte,fused4".split(",")
Xb, nb = assemble(base_views, ["lr_tfidf", "lr_dense"], "loo")
a, s, _ = cv_dev(Xb, nb, **small); print(f"single-stage dev-only meta, pre4 feats:          {a:.4f} +- {s:.4f}")
Xf, nf = assemble(base_views + fts, ["lr_tfidf", "lr_dense"], "loo")
a, s, Pf = cv_dev(Xf, nf, **small); print(f"single-stage dev-only meta, pre4 + {fts}: {a:.4f} +- {s:.4f}")
# two-stage: stage-1 probabilities + ft view features
Xv, nv_ = assemble(fts, [], "loo")
p1 = P1.reshape(-1); p1gap = (P1 - P1.max(1, keepdims=True)).reshape(-1)
X2 = np.column_stack([p1, p1gap, Xv]); n2 = ["p1", "p1gap"] + nv_
a, s, P2 = cv_dev(X2, n2, **small); print(f"two-stage (P1 + ft view feats) on dev:            {a:.4f} +- {s:.4f}")
X0 = np.column_stack([p1, p1gap, Xv[:, [nv_.index('prior'), nv_.index('class_id')]]]); n0 = ["p1", "p1gap", "prior", "class_id"]
a, s, _ = cv_dev(X0, n0, **small); print(f"two-stage control (P1 only, no ft feats):         {a:.4f} +- {s:.4f}")
ok1 = P1.argmax(1) == cy; ok2 = P2.argmax(1) == cy
for lo, hi in [(0, .3), (.3, .4), (.4, .5), (.5, .6), (.6, 1.01)]:
    m = np.zeros(n, bool); m[dev] = True; m &= (s1 >= lo) & (s1 < hi)
    print(f"   lex-sim {lo}-{hi}: n={m.sum():4d} stage1 {ok1[m].mean():.4f}  two-stage {ok2[m].mean():.4f}")
