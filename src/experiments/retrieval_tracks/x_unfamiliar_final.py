"""Gated ticket router for the "unfamiliar" track.

The dataset is made of small scenario families (a seed ticket plus one or two paraphrases) that share a queue. So:

  familiar tickets   -> family evidence. Candidate neighbours come from three views of the training pool (baseline
                        TF-IDF, frozen all-mpnet-base-v2, frozen bge-base-en-v1.5). A small gradient-boosted PAIR MODEL
                        scores every (ticket, candidate) pair for "shares the queue", from the three similarities, their
                        within-query ranks and margins, rare-token (product name) overlap, subject-line similarity and
                        length / sentence-count differences. Per queue, the evidence is the best and the noisy-OR of its
                        candidates' scores.
  unfamiliar tickets -> the SPECIALIST: strongly regularised logistic regressions on the TF-IDF view and on the two
                        frozen sentence-encoder views, combined by geometric mean. On tickets with no family in the pool
                        every model family we tried saturates at about 40% (majority class 30%), this one included.
  gate               -> a class-symmetric gradient-boosted model over (ticket, queue) rows: family evidence for the
                        queue, specialist probability for the queue, the prior, ranks, and the ticket's best evidence /
                        best similarity. It decides how far to trust the family evidence versus the specialist.

Everything is fitted inside fit_predict_proba from its arguments only; gate and pair model are trained on out-of-fold
features built inside the training set. The only outside ingredients are the two frozen pretrained encoders. Their
embeddings are cached on disk by text hash (a pure function of text and frozen encoder, nothing is fitted on evaluation
tickets).

    PYTHONPATH=src .venv/bin/python src/experiments/retrieval_tracks/x_unfamiliar_final.py        # core -> validation
"""
from __future__ import annotations

import hashlib
import os
import pickle
import time
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = ROOT / "data" / "x_unfamiliar"
ENCODERS = {"mpnet": "sentence-transformers/all-mpnet-base-v2", "bge_base": "BAAI/bge-base-en-v1.5"}
N_CLASSES = 10
K = 10            # neighbours taken from each view
GATE_SIM = 0.4    # only for the simple reference gate and for reporting


# ---------------------------------------------------------------- frozen encoders with an on-disk cache
def _sha(t: str) -> str:
    return hashlib.sha1(t.encode("utf-8")).hexdigest()


def embed(tag: str, texts: list[str]) -> np.ndarray:
    path = CACHE_DIR / f"embcache_{tag}.pkl"
    cache = pickle.load(open(path, "rb")) if path.exists() else {}
    keys = [_sha(t) for t in texts]
    missing = sorted({k: t for k, t in zip(keys, texts) if k not in cache}.items(), key=lambda kv: -len(kv[1]))
    if missing:
        import torch
        from sentence_transformers import SentenceTransformer
        torch.set_num_threads(3)
        dev = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
        m = SentenceTransformer(ENCODERS[tag], device=dev)
        m.max_seq_length = 384
        vecs = m.encode([t for _, t in missing], batch_size=32, normalize_embeddings=True, show_progress_bar=False,
                        convert_to_numpy=True).astype(np.float32)
        for (k, _), v in zip(missing, vecs):
            cache[k] = v
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        pickle.dump(cache, open(path, "wb"), protocol=4)
    return np.stack([cache[k] for k in keys])


# ---------------------------------------------------------------- neighbour search
def base_vectorizer():
    return TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=200_000, sublinear_tf=True)


def _sorted_topk(S, k):
    part = np.argpartition(-S, k - 1, axis=1)[:, :k]
    ps = np.take_along_axis(S, part, 1); o = np.argsort(-ps, axis=1)
    return np.take_along_axis(ps, o, 1), np.take_along_axis(part, o, 1)


def _topk_sparse(Q, P, k, block=1500):
    sims = np.zeros((Q.shape[0], k), dtype=np.float32); idx = np.zeros((Q.shape[0], k), dtype=np.int64)
    PT = P.T.tocsr()
    for s in range(0, Q.shape[0], block):
        sims[s:s + block], idx[s:s + block] = _sorted_topk((Q[s:s + block] @ PT).toarray(), k)
    return sims, idx


def _topk_dense(Q, P, k, block=2000):
    sims = np.zeros((Q.shape[0], k), dtype=np.float32); idx = np.zeros((Q.shape[0], k), dtype=np.int64)
    for s in range(0, Q.shape[0], block):
        sims[s:s + block], idx[s:s + block] = _sorted_topk(Q[s:s + block] @ P.T, k)
    return sims, idx


def _rowdot(Xq, Xp, qi, pi, block=200_000):
    out = np.zeros(len(qi), dtype=np.float32)
    for s in range(0, len(qi), block):
        out[s:s + block] = np.asarray(Xq[qi[s:s + block]].multiply(Xp[pi[s:s + block]]).sum(1)).ravel()
    return out


# ---------------------------------------------------------------- the unfamiliar-ticket specialist
def _full_proba(clf, X):
    p = np.full((X.shape[0], N_CLASSES), 1e-6, dtype=np.float64)
    p[:, clf.classes_] = clf.predict_proba(X)
    return p / p.sum(1, keepdims=True)


def specialist_proba(Xp, yp, Xq, Ep: dict, Eq: dict):
    logs = [np.log(_full_proba(LogisticRegression(C=1.0, max_iter=300).fit(Xp, yp), Xq))]
    for tag in Ep:
        logs.append(np.log(_full_proba(LogisticRegression(C=1.0, max_iter=500).fit(Ep[tag], yp), Eq[tag])))
    p = np.exp(np.mean(logs, axis=0))
    return p / p.sum(1, keepdims=True)


# ---------------------------------------------------------------- candidate pairs and their features
class PairViews:
    """Extra pool-fitted lexical views for the pair model: rare-token overlap and subject-line similarity."""

    def __init__(self, pool_texts):
        self.cv = CountVectorizer(binary=True, min_df=2, max_df=30, token_pattern=r"(?u)\b[a-z][a-z0-9\-]{2,}\b").fit(pool_texts)
        self.sv = TfidfVectorizer(sublinear_tf=True, min_df=1).fit([t.partition(" . ")[0] or "empty" for t in pool_texts])

    def mats(self, texts):
        return (self.cv.transform(texts).tocsr().astype(np.float32),
                self.sv.transform([t.partition(" . ")[0] for t in texts]).tocsr(),
                np.array([t.count(". ") + t.count("? ") for t in texts], dtype=np.float32))


def candidates(Xq, Xp, Eq, Ep, q_texts, p_texts):
    """Union of the top-K pool tickets per view, with pair features for every (query, candidate)."""
    n = Xq.shape[0]
    tops = [_topk_sparse(Xq, Xp, K)[1]] + [_topk_dense(Eq[t], Ep[t], K)[1] for t in Ep]
    qi, pi = [], []
    for i in range(n):
        c = np.unique(np.concatenate([t[i] for t in tops]))
        qi.append(np.full(len(c), i)); pi.append(c)
    qi, pi = np.concatenate(qi), np.concatenate(pi)
    st = _rowdot(Xq, Xp, qi, pi)
    dense = [np.einsum("ij,ij->i", Eq[t][qi], Ep[t][pi]).astype(np.float32) for t in Ep]
    lq = np.array([len(t) for t in q_texts], dtype=np.float32); lp = np.array([len(t) for t in p_texts], dtype=np.float32)
    feats = [st] + dense + [np.minimum(lq[qi], lp[pi]) / np.maximum(lq[qi], lp[pi])]
    for s in [st] + dense:                                   # within-query rank and margin to the best candidate
        order = np.lexsort((-s, qi)); rank = np.empty(len(s), dtype=np.float32)
        rank[order] = np.arange(len(s)) - np.searchsorted(qi[order], qi[order])
        best = np.zeros(n, dtype=np.float32); np.maximum.at(best, qi, s)
        feats += [rank, best[qi] - s]
    pv = PairViews(p_texts); Rq, Sq, nq = pv.mats(q_texts); Rp, Sp, npool = pv.mats(p_texts)
    shared = _rowdot(Rq, Rp, qi, pi); rq = np.asarray(Rq.sum(1)).ravel()[qi]; rp = np.asarray(Rp.sum(1)).ravel()[pi]
    feats += [shared, shared / np.maximum(np.minimum(rq, rp), 1), rq, _rowdot(Sq, Sp, qi, pi), np.abs(nq[qi] - npool[pi])]
    return qi, pi, np.stack(feats, 1).astype(np.float32)


def evidence(p, qi, cand_labels, n):
    mx = np.zeros((n, N_CLASSES), dtype=np.float32); np.maximum.at(mx, (qi, cand_labels), p)
    lo = np.zeros((n, N_CLASSES), dtype=np.float64); np.add.at(lo, (qi, cand_labels), np.log1p(-np.clip(p, 0, 0.999)))
    return mx, (1 - np.exp(lo)).astype(np.float32)


def class_rows(mx, nor, spec, prior, s1):
    n = len(mx); smx = np.sort(mx, 1); ssp = np.sort(spec, 1)
    rmx = (mx[:, :, None] < mx[:, None, :]).sum(2); rsp = (spec[:, :, None] < spec[:, None, :]).sum(2)
    per_q = np.stack([smx[:, -1], smx[:, -2], ssp[:, -1], ssp[:, -2], s1], 1)
    rows = np.concatenate([np.stack([mx, nor, spec, np.broadcast_to(prior, mx.shape), rmx, rsp], 2),
                           np.broadcast_to(per_q[:, None, :], (n, N_CLASSES, per_q.shape[1]))], 2)
    return rows.reshape(n * N_CLASSES, -1).astype(np.float32)


def _query_block(pool_texts, pool_y, pool_E, q_texts, q_E):
    vec = base_vectorizer(); Xp = vec.fit_transform(pool_texts).tocsr(); Xq = vec.transform(q_texts).tocsr()
    qi, pi, pf = candidates(Xq, Xp, q_E, pool_E, q_texts, pool_texts)
    spec = specialist_proba(Xp, pool_y, Xq, pool_E, q_E)
    s1 = np.zeros(len(q_texts), dtype=np.float32); np.maximum.at(s1, qi, pf[:, 0])
    return qi, pi, pf, spec, s1


# ---------------------------------------------------------------- the full gated system
def fit_predict_proba(train_texts, train_labels, eval_texts, n_inner=5, seed=0, return_info=False):
    y = np.asarray(train_labels, dtype=np.int64)
    train_texts, eval_texts = list(train_texts), list(eval_texts)
    E_tr = {t: embed(t, train_texts) for t in ENCODERS}; E_ev = {t: embed(t, eval_texts) for t in ENCODERS}
    prior = np.bincount(y, minlength=N_CLASSES) / len(y)

    # 1. out-of-fold pairs and specialist probabilities inside the training set (pool = the other folds)
    Q, PI, PF = [], [], []
    spec_oof = np.zeros((len(y), N_CLASSES)); s1 = np.zeros(len(y), dtype=np.float32)
    for tr, te in StratifiedKFold(n_inner, shuffle=True, random_state=seed).split(np.zeros(len(y)), y):
        qi, pi, pf, spec_oof[te], s1[te] = _query_block([train_texts[i] for i in tr], y[tr], {t: E_tr[t][tr] for t in ENCODERS},
                                                        [train_texts[i] for i in te], {t: E_tr[t][te] for t in ENCODERS})
        Q.append(te[qi]); PI.append(tr[pi]); PF.append(pf)
    Q, PI, PF = np.concatenate(Q), np.concatenate(PI), np.vstack(PF)
    same = (y[Q] == y[PI]).astype(int)

    # 2. pair model, cross-fitted over two halves of the training tickets so the gate sees honest pair scores
    mk = lambda: HistGradientBoostingClassifier(max_iter=150, learning_rate=0.1, max_leaf_nodes=31, random_state=seed)
    half = np.array([int(_sha(t), 16) % 2 for t in train_texts]); p = np.zeros(len(Q))
    for h in (0, 1):
        m = half[Q] == h
        p[m] = mk().fit(PF[~m], same[~m]).predict_proba(PF[m])[:, 1]
    pair_model = mk().fit(PF, same)

    # 3. the gate, on (ticket, queue) rows
    mx, nor = evidence(p, Q, y[PI], len(y))
    target = (np.arange(N_CLASSES)[None, :] == y[:, None]).reshape(-1).astype(int)
    gate = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=15, l2_regularization=1.0, early_stopping=True,
                                          validation_fraction=0.15, n_iter_no_change=20, random_state=seed)
    gate.fit(class_rows(mx, nor, spec_oof, prior, s1), target)

    # 4. evaluation tickets against the whole training pool
    qi, pi, pf, spec, s1e = _query_block(train_texts, y, E_tr, eval_texts, E_ev)
    mx, nor = evidence(pair_model.predict_proba(pf)[:, 1], qi, y[pi], len(eval_texts))
    score = gate.predict_proba(class_rows(mx, nor, spec, prior, s1e))[:, 1].reshape(len(eval_texts), N_CLASSES)
    proba = score / score.sum(1, keepdims=True)
    if not return_info:
        return proba
    o = np.lexsort((pf[:, 0], qi)); last = np.r_[np.where(np.diff(qi[o]))[0], len(o) - 1]
    nn_lab = np.zeros(len(eval_texts), dtype=np.int64); nn_lab[qi[o][last]] = y[pi[o][last]]
    return proba, {"tfidf_sim": s1e, "tfidf_nn_label": nn_lab, "specialist": spec, "family_max": mx, "n_gate_rows": int(len(y))}


def simple_gate_proba(info, thr=GATE_SIM):
    """Reference system: 1-NN label when the nearest TF-IDF similarity >= thr, else the specialist."""
    p = info["specialist"].copy(); fam = info["tfidf_sim"] >= thr
    p[fam] = 0.0; p[fam, info["tfidf_nn_label"][fam]] = 1.0
    return p


if __name__ == "__main__":
    import json
    from sklearn.metrics import f1_score
    from experiments.retrieval_tracks.x_unfamiliar_lib import load_core_val, bucket_table, fmt_table
    xc, yc, xv, yv = load_core_val()
    t0 = time.time()
    P, info = fit_predict_proba(xc, yc, xv, return_info=True)
    minutes = (time.time() - t0) / 60
    pred = P.argmax(1); sim = info["tfidf_sim"]; unf = sim < GATE_SIM
    print(f"core -> validation: accuracy {np.mean(pred == yv):.4f}  macro-F1 {f1_score(yv, pred, average='macro'):.4f}  [{minutes:.1f} min]")
    print(fmt_table(bucket_table(sim, pred == yv)))
    ref = {"1nn_tfidf": info["tfidf_nn_label"], "specialist_only": info["specialist"].argmax(1), "simple_gate_0.4": simple_gate_proba(info).argmax(1)}
    out = {"protocol": "train on core, score on validation, single run of src/experiments/retrieval_tracks/x_unfamiliar_final.py",
           "final": {"accuracy": float(np.mean(pred == yv)), "macro_f1": float(f1_score(yv, pred, average="macro")),
                     "unfamiliar_lt0.4": float(np.mean(pred[unf] == yv[unf])), "familiar_ge0.4": float(np.mean(pred[~unf] == yv[~unf])),
                     "buckets": bucket_table(sim, pred == yv), "runtime_minutes": minutes}}
    for k, r in ref.items():
        out[k] = {"accuracy": float(np.mean(r == yv)), "macro_f1": float(f1_score(yv, r, average="macro")),
                  "unfamiliar_lt0.4": float(np.mean(r[unf] == yv[unf])), "buckets": bucket_table(sim, r == yv)}
        print(f"reference {k}: accuracy {out[k]['accuracy']:.4f}  unfamiliar {out[k]['unfamiliar_lt0.4']:.4f}")
    np.save(CACHE_DIR / "final_val_probs.npy", P)
    json.dump(out, open(ROOT / "results" / "x_unfamiliar_final_val.json", "w"), indent=1)
