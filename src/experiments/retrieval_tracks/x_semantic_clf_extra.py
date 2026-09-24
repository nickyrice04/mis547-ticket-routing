"""Extra out-of-fold classifier probabilities inside core (+ full-core fit applied to validation texts, labels unused)."""
import numpy as np, time
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import ComplementNB
from sklearn.linear_model import LogisticRegression
from experiments.retrieval_tracks.x_semantic_common import *
ctx, cy, vtx, vy = load_core_val(); n = len(ctx)
vec = baseline_tfidf(); Xc = vec.fit_transform(ctx); Xv = vec.transform(vtx)
D8c = np.hstack([np.load(CACHE / f"emb_{d}_core.npy") for d in ["e5base", "bge", "mpnet", "gte", "bgelarge", "e5large", "gtemodern", "minilm"]])
D8v = np.hstack([np.load(CACHE / f"emb_{d}_val.npy") for d in ["e5base", "bge", "mpnet", "gte", "bgelarge", "e5large", "gtemodern", "minilm"]])
def softmax(z): z = z - z.max(1, keepdims=True); e = np.exp(z); return e / e.sum(1, keepdims=True)
models = {
    "mlp": (lambda: MLPClassifier(hidden_layer_sizes=(256,), max_iter=60, early_stopping=True, n_iter_no_change=5, random_state=0), "sparse"),
    "mlp_dense8": (lambda: MLPClassifier(hidden_layer_sizes=(512,), max_iter=80, early_stopping=True, n_iter_no_change=5, alpha=1e-3, random_state=0), "dense"),
}
for name, (mk, kind) in models.items():
    t0 = time.time(); A, B = (Xc, Xv) if kind == "sparse" else (D8c, D8v)
    oof = np.zeros((n, 10), np.float32)
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=1).split(ctx, cy):
        oof[te] = mk().fit(A[tr], cy[tr]).predict_proba(A[te])
    np.save(CACHE / f"clf_{name}_loo.npy", oof)
    np.save(CACHE / f"clf_{name}_val.npy", mk().fit(A, cy).predict_proba(B).astype(np.float32))
    print(f"{name}: OOF acc {(oof.argmax(1) == cy).mean():.4f}  {time.time()-t0:.0f}s", flush=True)
