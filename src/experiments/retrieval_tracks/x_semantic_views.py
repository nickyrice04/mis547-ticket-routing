"""Per-view candidate-class features, cached separately so stacks can be compared without recomputation.
 LOO inside core for meta-training / model selection; validation-vs-core features are computed but never used for fitting.
   python x_semantic_views.py dense e5base bge ...      # one view per embedder tag (emb_<tag>_core/val.npy)
   python x_semantic_views.py lexical                   # word tfidf, char tfidf, subject-only, body-only
   python x_semantic_views.py fused <name> <lam> d1,d2  # lex + lam * mean(dense)
   python x_semantic_views.py clf                       # OOF classifier probabilities
"""
import sys, time
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from experiments.retrieval_tracks.x_semantic_common import *
from experiments.retrieval_tracks.x_semantic_stack import class_feats, NC

ctx, cy, vtx, vy = load_core_val()
n, nv = len(ctx), len(vtx)

def run_view(name, sim_fn):
    """sim_fn(which, a, b) -> dense [b-a, n_core] similarity block; which in {'core','val'}."""
    for which, nq in (("core", n), ("val", nv)):
        acc_c, acc_q = {}, {}
        for a in range(0, nq, 2000):
            b = min(a + 2000, nq)
            S = sim_fn(which, a, b)
            cd, qd = class_feats(S, cy, np.arange(a, b) if which == "core" else None)
            for k, x in cd.items(): acc_c.setdefault(k, []).append(x)
            for k, x in qd.items(): acc_q.setdefault(k, []).append(x)
        out = {f"c_{k}": np.concatenate(x) for k, x in acc_c.items()}
        out.update({f"q_{k}": np.concatenate(x) for k, x in acc_q.items()})
        np.savez_compressed(CACHE / f"view_{name}_{'loo' if which == 'core' else 'val'}.npz", **out)
    d = np.load(CACHE / f"view_{name}_loo.npz")
    print(f"view {name}: LOO 1-NN acc {(d['q_top1'] == cy).mean():.4f}", flush=True)

def sparse_sim(Xc, Xv):
    return lambda which, a, b: ((Xc if which == "core" else Xv)[a:b] @ Xc.T).toarray().astype(np.float32)

def dense_sim(Ec, Ev):
    return lambda which, a, b: (Ec if which == "core" else Ev)[a:b] @ Ec.T

mode = sys.argv[1]
if mode == "dense":
    for d in sys.argv[2:]:
        run_view(d, dense_sim(np.load(CACHE / f"emb_{d}_core.npy"), np.load(CACHE / f"emb_{d}_val.npy")))
elif mode == "lexical":
    v = baseline_tfidf(); run_view("lex", sparse_sim(v.fit_transform(ctx), v.transform(vtx)))
    v = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2, sublinear_tf=True, max_features=300_000)
    run_view("char", sparse_sim(v.fit_transform(ctx), v.transform(vtx)))
    split = lambda t: (t.split(" . ", 1) + [""])[:2]
    for k, nm in ((0, "subj"), (1, "body")):
        v = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
        run_view(nm, sparse_sim(v.fit_transform([split(t)[k] for t in ctx]), v.transform([split(t)[k] for t in vtx])))
elif mode == "fused":
    name, lam, ds = sys.argv[2], float(sys.argv[3]), sys.argv[4].split(",")
    v = baseline_tfidf(); Xc = v.fit_transform(ctx); Xv = v.transform(vtx)
    Ec = [np.load(CACHE / f"emb_{d}_core.npy") for d in ds]; Ev = [np.load(CACHE / f"emb_{d}_val.npy") for d in ds]
    def f(which, a, b):
        S = ((Xc if which == "core" else Xv)[a:b] @ Xc.T).toarray().astype(np.float32)
        E = Ec if which == "core" else Ev
        return S + lam * np.mean([e[a:b] @ ec.T for e, ec in zip(E, Ec)], 0)
    run_view(name, f)
