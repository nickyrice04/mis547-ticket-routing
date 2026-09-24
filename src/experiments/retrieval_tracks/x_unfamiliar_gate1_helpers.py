"""Gated ticket router for the "unfamiliar" track.

  familiar tickets   -> lexical / dense nearest-neighbour label (the ticket's scenario family is in the training pool)
  unfamiliar tickets -> a heavily regularised "specialist" (logistic regressions on TF-IDF and on two frozen sentence
                        encoders, geometric mean), which is the best thing found for tickets with no family in the pool
  gate               -> a small gradient-boosted meta-learner over neighbour similarities, neighbour votes and the
                        specialist's probabilities, trained on out-of-fold features built inside the training set only

fit_predict_proba(train_texts, train_labels, eval_texts) uses only its arguments plus two frozen pretrained encoders
(sentence-transformers/all-mpnet-base-v2 and BAAI/bge-base-en-v1.5). Embeddings are cached on disk by text hash; the cache
is a pure function of the text and the frozen encoder, nothing is fitted on evaluation tickets.

    PYTHONPATH=src .venv/bin/python src/experiments/retrieval_tracks/x_unfamiliar_final.py        # core -> validation
"""
from __future__ import annotations

import hashlib
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = ROOT / "data" / "x_unfamiliar"
ENCODERS = {"mpnet": "sentence-transformers/all-mpnet-base-v2", "bge_base": "BAAI/bge-base-en-v1.5"}
N_CLASSES = 10
K = 10            # neighbours kept per view
GATE_SIM = 0.4    # only used by the simple fallback gate and for reporting


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


# ---------------------------------------------------------------- base views
def base_vectorizer():
    return TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=200_000, sublinear_tf=True)


def _topk_sparse(Q, P, k, block=1500):
    sims = np.zeros((Q.shape[0], k), dtype=np.float32); idx = np.zeros((Q.shape[0], k), dtype=np.int64)
    PT = P.T.tocsr()
    for s in range(0, Q.shape[0], block):
        S = (Q[s:s + block] @ PT).toarray()
        part = np.argpartition(-S, k - 1, axis=1)[:, :k]
        ps = np.take_along_axis(S, part, 1); o = np.argsort(-ps, axis=1)
        idx[s:s + block] = np.take_along_axis(part, o, 1); sims[s:s + block] = np.take_along_axis(ps, o, 1)
    return sims, idx


def _topk_dense(Q, P, k, block=2000):
    sims = np.zeros((Q.shape[0], k), dtype=np.float32); idx = np.zeros((Q.shape[0], k), dtype=np.int64)
    for s in range(0, Q.shape[0], block):
        S = Q[s:s + block] @ P.T
        part = np.argpartition(-S, k - 1, axis=1)[:, :k]
        ps = np.take_along_axis(S, part, 1); o = np.argsort(-ps, axis=1)
        idx[s:s + block] = np.take_along_axis(part, o, 1); sims[s:s + block] = np.take_along_axis(ps, o, 1)
    return sims, idx


def _votes(sims, labs, power, floor=0.0):
    w = np.clip(sims - floor, 0, None) ** power
    v = np.zeros((sims.shape[0], N_CLASSES), dtype=np.float32)
    for r in range(sims.shape[1]):
        np.add.at(v, (np.arange(sims.shape[0]), labs[:, r]), w[:, r])
    return v / np.clip(v.sum(1, keepdims=True), 1e-9, None)


def _full_proba(clf, X):
    p = np.full((X.shape[0], N_CLASSES), 1e-6, dtype=np.float64)
    p[:, clf.classes_] = clf.predict_proba(X)
    return p / p.sum(1, keepdims=True)


def specialist_proba(Xp, yp, Xq, Ep: dict, Eq: dict):
    """The unfamiliar-ticket specialist: strongly regularised linear models on three views, geometric mean."""
    logs = [np.log(_full_proba(LogisticRegression(C=1.0, max_iter=300).fit(Xp, yp), Xq))]
    for tag in Ep:
        logs.append(np.log(_full_proba(LogisticRegression(C=1.0, max_iter=500).fit(Ep[tag], yp), Eq[tag])))
    p = np.exp(np.mean(logs, axis=0))
    return p / p.sum(1, keepdims=True)


def base_features(pool_texts, pool_y, pool_E, q_texts, q_E):
    """Everything the gate sees for each query ticket, computed against a labelled pool."""
    vec = base_vectorizer(); Xp = vec.fit_transform(pool_texts); Xq = vec.transform(q_texts)
    k = min(K, len(pool_texts))
    ts, ti = _topk_sparse(Xq, Xp, k); tl = pool_y[ti]
    spec = specialist_proba(Xp, pool_y, Xq, pool_E, q_E)
    cols = [ts[:, :5], _votes(ts[:, :1], tl[:, :1], 1), _votes(ts, tl, 4), (tl[:, :1] == tl[:, 1:4]).astype(np.float32)]
    info = {"tfidf_sim": ts[:, 0], "tfidf_nn_label": tl[:, 0], "specialist": spec}
    nn_labels = [tl[:, 0]]
    for tag in pool_E:
        ds, di = _topk_dense(q_E[tag], pool_E[tag], k); dl = pool_y[di]
        cols += [ds[:, :5], _votes(ds[:, :1], dl[:, :1], 1), _votes(ds, dl, 8, floor=0.5), (dl[:, :1] == dl[:, 1:4]).astype(np.float32)]
        nn_labels.append(dl[:, 0])
    cols.append(np.stack([(a == b) for i, a in enumerate(nn_labels) for b in nn_labels[i + 1:]], 1).astype(np.float32))
    cols += [spec, (spec.argmax(1) == tl[:, 0]).astype(np.float32)[:, None]]
    return np.hstack(cols).astype(np.float32), info


# ---------------------------------------------------------------- the full gated system
def fit_predict_proba(train_texts, train_labels, eval_texts, n_inner=5, seed=0, use_dense=True, return_info=False):
    y = np.asarray(train_labels, dtype=np.int64)
    train_texts, eval_texts = list(train_texts), list(eval_texts)
    tags = list(ENCODERS) if use_dense else []
    E_tr = {t: embed(t, train_texts) for t in tags}; E_ev = {t: embed(t, eval_texts) for t in tags}

    # 1. out-of-fold gate features inside the training set (the pool for each fold is the other folds)
    oof = None
    for tr, te in StratifiedKFold(n_inner, shuffle=True, random_state=seed).split(np.zeros(len(y)), y):
        f, _ = base_features([train_texts[i] for i in tr], y[tr], {t: E_tr[t][tr] for t in tags},
                             [train_texts[i] for i in te], {t: E_tr[t][te] for t in tags})
        if oof is None:
            oof = np.zeros((len(y), f.shape[1]), dtype=np.float32)
        oof[te] = f
    # 2. the gate
    gate = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.06, max_leaf_nodes=15, l2_regularization=1.0,
                                          early_stopping=True, validation_fraction=0.15, n_iter_no_change=15, random_state=seed)
    gate.fit(oof, y)
    # 3. features for the evaluation tickets against the whole training pool
    f_ev, info = base_features(train_texts, y, E_tr, eval_texts, E_ev)
    proba = np.zeros((len(eval_texts), N_CLASSES)); proba[:, gate.classes_] = gate.predict_proba(f_ev)
    return (proba, info) if return_info else proba


def simple_gate_proba(info, thr=GATE_SIM):
    """Reference system: 1-NN label when the nearest TF-IDF similarity >= thr, else the specialist."""
    p = info["specialist"].copy(); fam = info["tfidf_sim"] >= thr
    p[fam] = 0.0; p[fam, info["tfidf_nn_label"][fam]] = 1.0
    return p


if __name__ == "__main__":
    from sklearn.metrics import f1_score
    from experiments.retrieval_tracks.x_unfamiliar_lib import load_core_val, bucket_table, fmt_table
    xc, yc, xv, yv = load_core_val()
    t0 = time.time()
    P, info = fit_predict_proba(xc, yc, xv, return_info=True)
    pred = P.argmax(1)
    print(f"core -> validation: accuracy {np.mean(pred == yv):.4f}  macro-F1 {f1_score(yv, pred, average='macro'):.4f}  [{(time.time()-t0)/60:.1f} min]")
    print(fmt_table(bucket_table(info["tfidf_sim"], pred == yv)))
