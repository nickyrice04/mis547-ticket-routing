"""Gate v3 (experimental): pair model + class-symmetric binary gate, trained only on queries the family encoder has not seen."""
from __future__ import annotations
import json, os, hashlib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold
import experiments.retrieval_tracks.x_unfamiliar_gate1_helpers as F
from experiments.retrieval_tracks.x_unfamiliar_gate2 import candidates as _cand_base, _rowdot
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import scipy.sparse as sp

EXTRA = os.environ.get("PAIR_EXTRA", "1") == "1"

class PairViews:
    """Extra pool-fitted lexical views for the pair model: rare-token overlap and subject-line similarity."""
    def __init__(self, pool_texts):
        self.cv = CountVectorizer(binary=True, min_df=2, max_df=30, token_pattern=r"(?u)\b[a-z][a-z0-9\-]{2,}\b").fit(pool_texts)
        self.sv = TfidfVectorizer(sublinear_tf=True, min_df=1).fit([t.partition(" . ")[0] or "empty" for t in pool_texts])
    def mats(self, texts):
        return self.cv.transform(texts).tocsr().astype(np.float32), self.sv.transform([t.partition(" . ")[0] for t in texts]).tocsr(), \
               np.array([t.count(". ") + t.count("? ") for t in texts], dtype=np.float32)

def candidates(Xq, Xp, Eq, Ep, lq, lp, q_texts=None, p_texts=None, pv=None):
    qi, pi, pf = _cand_base(Xq, Xp, Eq, Ep, lq, lp)
    if not EXTRA: return qi, pi, pf
    Rq, Sq, nq = pv.mats(q_texts); Rp, Sp, npool = pv.mats(p_texts)
    shared = _rowdot(Rq, Rp, qi, pi); rq = np.asarray(Rq.sum(1)).ravel()[qi]; rp = np.asarray(Rp.sum(1)).ravel()[pi]
    extra = np.stack([shared, shared / np.maximum(np.minimum(rq, rp), 1), rq, _rowdot(Sq, Sp, qi, pi), np.abs(nq[qi] - npool[pi])], 1)
    return qi, pi, np.hstack([pf, extra.astype(np.float32)])

N = F.N_CLASSES
FAMENC_DIR = F.ROOT / "models" / "x_unfamiliar_famenc"

def embed_view(tag, texts):
    if tag != "famenc": return F.embed(tag, texts)
    import pickle
    path = F.CACHE_DIR / "embcache_famenc.pkl"; cache = pickle.load(open(path, "rb")) if path.exists() else {}
    keys = [F._sha(t) for t in texts]; missing = sorted({k: t for k, t in zip(keys, texts) if k not in cache}.items(), key=lambda kv: -len(kv[1]))
    if missing:
        import torch
        from sentence_transformers import SentenceTransformer, models
        torch.set_num_threads(3); dev = "mps" if torch.backends.mps.is_available() else "cpu"
        w = models.Transformer(str(FAMENC_DIR), max_seq_length=160); m = SentenceTransformer(modules=[w, models.Pooling(w.get_word_embedding_dimension(), "mean")], device=dev)
        V = m.encode([t for _, t in missing], batch_size=64, normalize_embeddings=True, show_progress_bar=False, convert_to_numpy=True).astype(np.float32)
        for (k, _), v in zip(missing, V): cache[k] = v
        pickle.dump(cache, open(path, "wb"), protocol=4)
    return np.stack([cache[k] for k in keys])

def evidence(p, qi, lab, n):
    mx = np.zeros((n, N), dtype=np.float32); np.maximum.at(mx, (qi, lab), p)
    lo = np.zeros((n, N), dtype=np.float64); np.add.at(lo, (qi, lab), np.log1p(-np.clip(p, 0, 0.999)))
    return mx, (1 - np.exp(lo)).astype(np.float32)

def class_rows(mx, nor, spec, prior, s1):
    n = len(mx); smx = np.sort(mx, 1); ssp = np.sort(spec, 1)
    rmx = (mx[:, :, None] < mx[:, None, :]).sum(2); rsp = (spec[:, :, None] < spec[:, None, :]).sum(2)
    per_q = np.stack([smx[:, -1], smx[:, -2], ssp[:, -1], ssp[:, -2], s1], 1)
    rows = np.concatenate([np.stack([mx, nor, spec, np.broadcast_to(prior, mx.shape), rmx, rsp], 2),
                           np.broadcast_to(per_q[:, None, :], (n, N, per_q.shape[1]))], 2)
    return rows.reshape(n * N, -1).astype(np.float32)

def fit_predict_proba(train_texts, train_labels, eval_texts, views=("mpnet", "bge_base"), n_inner=5, seed=0, return_info=False, use_dense=True, restrict_to_unseen=None):
    y = np.asarray(train_labels, dtype=np.int64); train_texts, eval_texts = list(train_texts), list(eval_texts)
    spec_views = [v for v in views if v != "famenc"]
    E_tr = {t: embed_view(t, train_texts) for t in views}; E_ev = {t: embed_view(t, eval_texts) for t in views}
    if restrict_to_unseen is None: restrict_to_unseen = "famenc" in views
    seen = set(json.load(open(FAMENC_DIR / "x_unfamiliar_seen.json"))["seen_sha"]) if restrict_to_unseen else set()
    unseen = np.array([F._sha(t) not in seen for t in train_texts])
    if os.environ.get("SIM_UNSEEN_FRAC"):   # experiment only: how much does the gate lose with fewer training rows?
        unseen = np.random.RandomState(1).rand(len(y)) < float(os.environ["SIM_UNSEEN_FRAC"])
    ltr = np.array([len(t) for t in train_texts], dtype=np.float32); lev = np.array([len(t) for t in eval_texts], dtype=np.float32)
    prior = np.bincount(y, minlength=N) / len(y)
    Q, PI, PF, spec_oof, s1 = [], [], [], np.zeros((len(y), N)), np.zeros(len(y), dtype=np.float32)
    for tr, te in StratifiedKFold(n_inner, shuffle=True, random_state=seed).split(np.zeros(len(y)), y):
        te = te[unseen[te]]
        vec = F.base_vectorizer(); Xp = vec.fit_transform([train_texts[i] for i in tr]).tocsr(); Xq = vec.transform([train_texts[i] for i in te]).tocsr()
        ptx = [train_texts[i] for i in tr]; qtx = [train_texts[i] for i in te]
        qi, pi, pf = candidates(Xq, Xp, {t: E_tr[t][te] for t in views}, {t: E_tr[t][tr] for t in views}, ltr[te], ltr[tr], qtx, ptx, PairViews(ptx) if EXTRA else None)
        Q.append(te[qi]); PI.append(tr[pi]); PF.append(pf); s1[te] = np.asarray(Xq.multiply(0).sum(1)).ravel()
        best = np.zeros(len(te), dtype=np.float32); np.maximum.at(best, qi, pf[:, 0]); s1[te] = best
        spec_oof[te] = F.specialist_proba(Xp, y[tr], Xq, {t: E_tr[t][tr] for t in spec_views}, {t: E_tr[t][te] for t in spec_views})
    Q, PI, PF = np.concatenate(Q), np.concatenate(PI), np.vstack(PF); T = (y[Q] == y[PI]).astype(int)
    mk = lambda: HistGradientBoostingClassifier(max_iter=150, learning_rate=0.1, max_leaf_nodes=31, random_state=seed)
    half = np.array([int(hashlib.sha1(train_texts[i].encode()).hexdigest(), 16) % 2 for i in range(len(y))])
    p = np.zeros(len(Q))
    for h in (0, 1):
        m = half[Q] == h; p[m] = mk().fit(PF[~m], T[~m]).predict_proba(PF[m])[:, 1]
    pair_model = mk().fit(PF, T)
    U = np.where(unseen)[0]; pos_of = -np.ones(len(y), dtype=np.int64); pos_of[U] = np.arange(len(U))
    mx, nor = evidence(p, pos_of[Q], y[PI], len(U))
    rows = class_rows(mx, nor, spec_oof[U], prior, s1[U]); target = (np.arange(N)[None, :] == y[U][:, None]).reshape(-1).astype(int)
    gate = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=15, l2_regularization=1.0, early_stopping=True,
                                          validation_fraction=0.15, n_iter_no_change=20, random_state=seed).fit(rows, target)
    # evaluation tickets against the whole training pool
    vec = F.base_vectorizer(); Xp = vec.fit_transform(train_texts).tocsr(); Xq = vec.transform(eval_texts).tocsr()
    qi, pi, pf = candidates(Xq, Xp, E_ev, E_tr, lev, ltr, eval_texts, train_texts, PairViews(train_texts) if EXTRA else None)
    mx, nor = evidence(pair_model.predict_proba(pf)[:, 1], qi, y[pi], len(eval_texts))
    s1e = np.zeros(len(eval_texts), dtype=np.float32); np.maximum.at(s1e, qi, pf[:, 0])
    nn_lab = np.zeros(len(eval_texts), dtype=np.int64); o = np.lexsort((pf[:, 0], qi)); last = np.r_[np.where(np.diff(qi[o]))[0], len(o) - 1]; nn_lab[qi[o][last]] = y[pi[o][last]]
    spec = F.specialist_proba(Xp, y, Xq, {t: E_tr[t] for t in spec_views}, {t: E_ev[t] for t in spec_views})
    score = gate.predict_proba(class_rows(mx, nor, spec, prior, s1e))[:, 1].reshape(len(eval_texts), N)
    proba = score / score.sum(1, keepdims=True)
    info = {"tfidf_sim": s1e, "tfidf_nn_label": nn_lab, "specialist": spec, "family_max": mx, "n_gate_rows": int(len(U))}
    return (proba, info) if return_info else proba
