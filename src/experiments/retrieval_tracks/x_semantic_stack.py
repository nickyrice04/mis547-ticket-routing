"""Learned k-NN fusion ("candidate-class" stacker).

For every (ticket, class c) pair we build class-symmetric features from several similarity views
(lexical TF-IDF + dense embedders): the best similarity to any pool ticket of class c, its gap to the best
other class, rank-weighted votes among the top-k, plus out-of-fold classifier probabilities. A gradient-boosted
binary model scores "is c the right queue". Training rows come from leave-one-out neighbours INSIDE the
training pool (which mimics how an unseen ticket relates to the pool), so nothing is fitted on eval data.
"""
from __future__ import annotations
import numpy as np

NC = 10


def class_feats(S, y_pool, self_idx=None, k=10):
    """S: [n, pool] similarities. Returns dict name -> [n, NC] and query-level dict name -> [n]."""
    n = S.shape[0]
    if self_idx is not None:
        S[np.arange(n), self_idx] = -1.0
    m = np.full((n, NC), -1.0, np.float32)
    for c in range(NC):
        cols = np.where(y_pool == c)[0]
        if len(cols):
            m[:, c] = S[:, cols].max(1)
    srt = np.sort(m, axis=1)
    best, second = srt[:, -1], srt[:, -2]
    other_best = np.where(m == best[:, None], second[:, None], best[:, None])
    gap = m - other_best
    idx = np.argpartition(-S, k, axis=1)[:, :k]
    s = np.take_along_axis(S, idx, 1)
    o = np.argsort(-s, axis=1)
    idx = np.take_along_axis(idx, o, 1); s = np.take_along_axis(s, o, 1)
    lab = y_pool[idx]
    w = 1.0 / np.arange(1, k + 1)
    vote = np.zeros((n, NC), np.float32); cnt5 = np.zeros((n, NC), np.float32)
    for c in range(NC):
        vote[:, c] = ((lab == c) * w).sum(1) / w.sum()
        cnt5[:, c] = (lab[:, :5] == c).mean(1)
    return {"max": m, "gap": gap, "vote": vote, "cnt5": cnt5}, {"s1": s[:, 0], "s5": s[:, 4], "top1": lab[:, 0]}


def assemble(per_view, clf_probs, prior, n):
    """per_view: dict view -> (class dict, query dict). clf_probs: dict name -> [n, NC]. Returns X [n*NC, F], names."""
    cols, names = [], []
    for v, (cd, qd) in per_view.items():
        for k_, a in cd.items():
            cols.append(a.reshape(-1)); names.append(f"{v}_{k_}")
        for k_ in ("s1", "s5"):
            cols.append(np.repeat(qd[k_], NC)); names.append(f"{v}_{k_}")
    views = list(per_view)
    top1 = np.stack([per_view[v][1]["top1"] for v in views], 1)          # [n, V]
    agree = np.stack([(top1 == c).sum(1) for c in range(NC)], 1)          # how many views' 1-NN vote for c
    cols.append(agree.reshape(-1).astype(np.float32)); names.append("top1_agree")
    for k_, p in clf_probs.items():
        cols.append(p.reshape(-1)); names.append(f"clf_{k_}")
        cols.append((p - p.max(1, keepdims=True)).reshape(-1)); names.append(f"clfgap_{k_}")
    cols.append(np.tile(prior, n)); names.append("prior")
    cols.append(np.tile(np.arange(NC), n).astype(np.float32)); names.append("class_id")
    return np.stack(cols, 1).astype(np.float32), names


def fit_meta(X, t, names, seed=0, **kw):
    from sklearn.ensemble import HistGradientBoostingClassifier
    params = dict(max_iter=300, learning_rate=0.06, max_leaf_nodes=31, min_samples_leaf=40, l2_regularization=1.0,
                  random_state=seed)
    params.update(kw)
    cat = [names.index("class_id")]
    return HistGradientBoostingClassifier(categorical_features=cat, **params).fit(X, t)


def meta_proba(model, X):
    s = model.predict_proba(X)[:, 1].reshape(-1, NC)
    return s / s.sum(1, keepdims=True)
