"""Gate v2 (experimental): a learned "same family?" pair model over candidate neighbours from three views,
then a gradient-boosted gate over per-class family evidence + the unfamiliar specialist."""
from __future__ import annotations
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold
import experiments.retrieval_tracks.x_unfamiliar_gate1_helpers as F

N = F.N_CLASSES; K = 10

def _rowdot(Xq, Xp, qi, pi, block=200_000):
    out = np.zeros(len(qi), dtype=np.float32)
    for s in range(0, len(qi), block):
        out[s:s + block] = np.asarray(Xq[qi[s:s + block]].multiply(Xp[pi[s:s + block]]).sum(1)).ravel()
    return out

def candidates(Xq, Xp, Eq, Ep, lens_q, lens_p, exclude_self=False):
    """Union of the top-K pool tickets per view, with every view's similarity for every candidate."""
    k = K + 1 if exclude_self else K
    tops = [F._topk_sparse(Xq, Xp, k)[1]] + [F._topk_dense(Eq[t], Ep[t], k)[1] for t in Ep]
    n = Xq.shape[0]; qi, pi = [], []
    for i in range(n):
        c = np.unique(np.concatenate([t[i] for t in tops]))
        if exclude_self: c = c[c != i]
        qi.append(np.full(len(c), i)); pi.append(c)
    qi, pi = np.concatenate(qi), np.concatenate(pi)
    st = _rowdot(Xq, Xp, qi, pi)
    dense = [np.einsum("ij,ij->i", Eq[t][qi], Ep[t][pi]).astype(np.float32) for t in Ep]
    lr = np.minimum(lens_q[qi], lens_p[pi]) / np.maximum(lens_q[qi], lens_p[pi])
    feats = [st] + dense + [lr.astype(np.float32)]
    # within-query ranks / margins for each similarity
    for s in [st] + dense:
        order = np.lexsort((-s, qi)); rank = np.empty(len(s), dtype=np.float32)
        start = np.searchsorted(qi[order], qi[order]); rank[order] = np.arange(len(s)) - start
        best = np.zeros(n, dtype=np.float32); np.maximum.at(best, qi, s)
        feats += [rank, best[qi] - s]
    return qi, pi, np.stack(feats, 1)

def family_evidence(pair_model, qi, pi, pf, pool_y, n):
    p = pair_model.predict_proba(pf)[:, 1]
    mx = np.zeros((n, N), dtype=np.float32); np.maximum.at(mx, (qi, pool_y[pi]), p)
    lo = np.zeros((n, N), dtype=np.float64); np.add.at(lo, (qi, pool_y[pi]), np.log1p(-np.clip(p, 0, 0.999)))
    return mx, (1 - np.exp(lo)).astype(np.float32)

def base_features(pool_texts, pool_y, pool_E, q_texts, q_E):
    vec = F.base_vectorizer(); Xp = vec.fit_transform(pool_texts).tocsr(); Xq = vec.transform(q_texts).tocsr()
    lp = np.array([len(t) for t in pool_texts], dtype=np.float32); lq = np.array([len(t) for t in q_texts], dtype=np.float32)
    # pair model: leave-one-out candidates inside the pool, target = candidate shares the ticket's queue
    qi, pi, pf = candidates(Xp, Xp, pool_E, pool_E, lp, lp, exclude_self=True)
    pm = HistGradientBoostingClassifier(max_iter=150, learning_rate=0.1, max_leaf_nodes=31, random_state=0).fit(pf, (pool_y[qi] == pool_y[pi]).astype(int))
    qi, pi, pf = candidates(Xq, Xp, q_E, pool_E, lq, lp)
    mx, nor = family_evidence(pm, qi, pi, pf, pool_y, len(q_texts))
    spec = F.specialist_proba(Xp, pool_y, Xq, pool_E, q_E)
    ts, ti = F._topk_sparse(Xq, Xp, 3)
    dtop = [F._topk_dense(q_E[t], pool_E[t], 3)[0] for t in pool_E]
    srt = np.sort(mx, 1)
    cols = [mx, nor, srt[:, -1:], srt[:, -1:] - srt[:, -2:-1], ts] + dtop + [spec, (spec.argmax(1) == mx.argmax(1)).astype(np.float32)[:, None]]
    info = {"tfidf_sim": ts[:, 0], "tfidf_nn_label": pool_y[ti[:, 0]], "specialist": spec, "family_max": mx}
    return np.hstack(cols).astype(np.float32), info

def fit_predict_proba(train_texts, train_labels, eval_texts, n_inner=5, seed=0, return_info=False, use_dense=True):
    y = np.asarray(train_labels, dtype=np.int64); train_texts, eval_texts = list(train_texts), list(eval_texts)
    tags = list(F.ENCODERS)
    E_tr = {t: F.embed(t, train_texts) for t in tags}; E_ev = {t: F.embed(t, eval_texts) for t in tags}
    oof = None
    for tr, te in StratifiedKFold(n_inner, shuffle=True, random_state=seed).split(np.zeros(len(y)), y):
        f, _ = base_features([train_texts[i] for i in tr], y[tr], {t: E_tr[t][tr] for t in tags}, [train_texts[i] for i in te], {t: E_tr[t][te] for t in tags})
        if oof is None: oof = np.zeros((len(y), f.shape[1]), dtype=np.float32)
        oof[te] = f
    gate = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.06, max_leaf_nodes=15, l2_regularization=1.0, early_stopping=True,
                                          validation_fraction=0.15, n_iter_no_change=15, random_state=seed).fit(oof, y)
    f_ev, info = base_features(train_texts, y, E_tr, eval_texts, E_ev)
    proba = np.zeros((len(eval_texts), N)); proba[:, gate.classes_] = gate.predict_proba(f_ev)
    return (proba, info) if return_info else proba
