"""Six hyper-parameter variants of the stacker, 5-fold CV over LOO rows inside core. None beat the default by more than noise."""
import numpy as np
from experiments.retrieval_tracks.x_semantic_cv3 import assemble, cv, cy, CACHE
X, names = assemble("lex,e5base,bge,mpnet,gte,fused4".split(","), ["lr_tfidf", "lr_dense"], "loo")
for kw in [dict(), dict(max_iter=600, learning_rate=0.03), dict(max_leaf_nodes=63), dict(max_iter=200, learning_rate=0.1, max_leaf_nodes=15),
           dict(min_samples_leaf=200), dict(max_iter=500, learning_rate=0.03, max_leaf_nodes=15, min_samples_leaf=100, l2_regularization=5.0)]:
    P = cv(X, names, **kw); print(kw, f"acc {(P.argmax(1) == cy).mean():.4f}  logloss {-np.log(P[np.arange(len(cy)), cy] + 1e-9).mean():.4f}", flush=True)
