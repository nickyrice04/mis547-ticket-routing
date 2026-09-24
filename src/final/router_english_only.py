"""The English-only router: the same retrieval-and-stacker idea as final/router.py without the German pool.

This is the clean modelling result, 83.1% on the test set with exactly the data every
earlier tier had. final/router.py adds the translated German tickets on top and reaches
91.8%. Both are kept because the gap between them is the value of real labelled data.

Method (everything is fitted on the training texts and labels that are passed in, nothing else)
  1. Similarity views between a ticket and the training pool
       lex     baseline TF-IDF (1-2 grams) cosine
       dense   four frozen pretrained sentence embedders (multilingual-e5-base, bge-base-en-v1.5,
               all-mpnet-base-v2, gte-base), cosine
       fused   lex + 5 * mean(dense)        (weight 5 was picked by 5-fold CV inside core)
  2. Class-symmetric "candidate class" features for every (ticket, class c) pair, per view: best similarity to a
     pool ticket of class c, its gap to the best other class, rank-weighted vote of c among the 10 nearest, share of
     c among the 5 nearest, the top-1 / top-5 similarity, how many views' 1-NN vote for c, plus out-of-fold
     probabilities of two plain classifiers (logistic regression on TF-IDF and on the concatenated embeddings).
  3. A gradient-boosted binary model scores "is c the right queue". Its training rows are the TRAINING tickets
     themselves with leave-one-out neighbours (each training ticket is matched against all other training tickets),
     which was verified to reproduce the validation-vs-core similarity distribution almost exactly. Scores are
     normalised over the 10 classes to give probabilities, averaged over three seeds.

No validation or test ticket is used anywhere in fitting. No fine-tuned weights or extra data files are required
(the contrastive fine-tune explored in this track did not improve the stack and is not used). This file was named
x_semantic_final.py when the results were produced. The code is unchanged, only the name.

    PYTHONPATH=src .venv/bin/python src/final/router_english_only.py      # core -> validation
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

ROOT = Path(__file__).resolve().parent.parent.parent
EMB_CACHE = ROOT / "data" / "x_semantic" / "final_cache"   # embeddings keyed by content hash, no labels
NC = 10
# (short name, Hugging Face id, the prefix the model expects on its inputs)
EMBEDDERS = [
    ("e5base", "intfloat/multilingual-e5-base", "query: "),
    ("bge", "BAAI/bge-base-en-v1.5", ""),
    ("mpnet", "sentence-transformers/all-mpnet-base-v2", ""),
    ("gte", "thenlper/gte-base", ""),
]
LAMBDA_FUSE = 5.0     # weight of the mean dense similarity in the fused view, chosen by CV inside core
CHUNK = 2000          # queries per block when computing similarity matrices
META_PARAMS = dict(max_iter=300, learning_rate=0.06, max_leaf_nodes=31, min_samples_leaf=40, l2_regularization=1.0)
META_SEEDS = (0, 1, 2)


# ----------------------------------------------------------------------------------------------- embeddings
def _embed(model_name, prefix, texts, max_seq_length=256, batch_size=64):
    """L2-normalised sentence embeddings from a frozen pretrained model, cached on disk by content hash."""
    h = hashlib.sha1((model_name + "\x00" + "\x00".join(texts)).encode("utf-8")).hexdigest()[:20]
    path = EMB_CACHE / f"{h}.npy"
    if path.exists():
        return np.load(path)
    import torch
    from sentence_transformers import SentenceTransformer
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "3")))
    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    model = SentenceTransformer(model_name, device=device)
    model.max_seq_length = max_seq_length
    order = np.argsort([len(t) for t in texts])[::-1]      # longest first, so batches pad less
    e = model.encode([prefix + texts[i] for i in order], batch_size=batch_size, normalize_embeddings=True,
                     show_progress_bar=False, convert_to_numpy=True)
    out = np.empty_like(e)
    out[order] = e
    out = out.astype(np.float32)
    EMB_CACHE.mkdir(parents=True, exist_ok=True)
    np.save(path, out)
    del model
    return out


# ----------------------------------------------------------------------------------------------- features
def _class_feats(S, y_pool, self_idx=None, k=10):
    """Per-class neighbour features from one similarity matrix S [n_queries, n_pool].

    Returns two dicts. Per (query, class): "max" the best similarity to a pool ticket of that
    class, "gap" its margin over the best other class, "vote" the rank-weighted share of the
    class among the 10 nearest, "cnt5" its share among the 5 nearest. Per query: the top-1
    and top-5 similarities and the label of the nearest ticket.

    With self_idx set (leave-one-out), each query's own pool row is masked out so a training
    ticket can never be its own neighbour.
    """
    n = S.shape[0]
    if self_idx is not None:
        S[np.arange(n), self_idx] = -1.0          # leave-one-out: a ticket may not be its own neighbour
    m = np.full((n, NC), -1.0, np.float32)
    for c in range(NC):
        cols = np.where(y_pool == c)[0]
        if len(cols):
            m[:, c] = S[:, cols].max(1)
    srt = np.sort(m, axis=1)
    best, second = srt[:, -1], srt[:, -2]
    gap = m - np.where(m == best[:, None], second[:, None], best[:, None])
    idx = np.argpartition(-S, k, axis=1)[:, :k]                  # the k nearest, then sorted
    s = np.take_along_axis(S, idx, 1)
    o = np.argsort(-s, axis=1)
    idx = np.take_along_axis(idx, o, 1)
    s = np.take_along_axis(s, o, 1)
    lab = y_pool[idx]
    w = 1.0 / np.arange(1, k + 1)                                # rank weights 1, 1/2, 1/3, ...
    vote = np.zeros((n, NC), np.float32)
    cnt5 = np.zeros((n, NC), np.float32)
    for c in range(NC):
        vote[:, c] = ((lab == c) * w).sum(1) / w.sum()
        cnt5[:, c] = (lab[:, :5] == c).mean(1)
    return {"max": m, "gap": gap, "vote": vote, "cnt5": cnt5}, {"s1": s[:, 0], "s5": s[:, 4], "top1": lab[:, 0]}


def _view_features(Xq, Eq, Xp, Ep, y_pool, loo):
    """Per-view class features of the queries against the pool. Returns ({view: (class dict, query dict)}, order).

    Xq, Xp are TF-IDF matrices, Eq, Ep dicts of embeddings per embedder. The similarity
    matrices are computed in chunks of CHUNK queries so memory stays bounded.
    """
    nq = Xq.shape[0]
    names = ["lex"] + [e[0] for e in EMBEDDERS] + ["fused"]
    acc = {v: ({}, {}) for v in names}
    for a in range(0, nq, CHUNK):
        b = min(a + CHUNK, nq)
        self_idx = np.arange(a, b) if loo else None
        sims = {"lex": (Xq[a:b] @ Xp.T).toarray().astype(np.float32)}
        for name, _, _ in EMBEDDERS:
            sims[name] = Eq[name][a:b] @ Ep[name].T
        sims["fused"] = sims["lex"] + LAMBDA_FUSE * np.mean([sims[e[0]] for e in EMBEDDERS], 0)
        for v in names:
            cd, qd = _class_feats(sims[v], y_pool, self_idx)
            for k_, x in cd.items():
                acc[v][0].setdefault(k_, []).append(x)
            for k_, x in qd.items():
                acc[v][1].setdefault(k_, []).append(x)
    return {v: ({k_: np.concatenate(x) for k_, x in cd.items()}, {k_: np.concatenate(x) for k_, x in qd.items()})
            for v, (cd, qd) in acc.items()}, names


def _assemble(per_view, view_names, clf_probs, prior, n):
    """Flatten everything into the (ticket, class) pair rows the stacker reads. Returns (X [n*10, cols], names).

    Per view: max, gap, vote, cnt5 for the class, and the query's top-1 / top-5 similarity.
    Then how many views' nearest ticket carries the class, the two classifiers' probability
    for the class and its margin over their top class, the class prior, and the class id.
    """
    cols, names = [], []
    for v in view_names:
        cd, qd = per_view[v]
        for k_ in ("max", "gap", "vote", "cnt5"):
            cols.append(cd[k_].reshape(-1)); names.append(f"{v}_{k_}")
        for k_ in ("s1", "s5"):
            cols.append(np.repeat(qd[k_], NC)); names.append(f"{v}_{k_}")
    top1 = np.stack([per_view[v][1]["top1"] for v in view_names], 1)
    agree = np.stack([(top1 == c).sum(1) for c in range(NC)], 1)
    cols.append(agree.reshape(-1).astype(np.float32)); names.append("top1_agree")
    for k_, p in clf_probs.items():
        cols.append(p.reshape(-1)); names.append(f"clf_{k_}")
        cols.append((p - p.max(1, keepdims=True)).reshape(-1)); names.append(f"clfgap_{k_}")
    cols.append(np.tile(prior, n)); names.append("prior")
    cols.append(np.tile(np.arange(NC), n).astype(np.float32)); names.append("class_id")
    return np.stack(cols, 1).astype(np.float32), names


def _full_proba(clf, X):
    """predict_proba padded to NC columns (a fold may miss a rare class)."""
    p = np.zeros((X.shape[0], NC), np.float32)
    p[:, clf.classes_] = clf.predict_proba(X)
    return p


# ----------------------------------------------------------------------------------------------- public API
def fit_predict_proba(train_texts, train_labels, eval_texts, verbose=True, return_debug=False):
    """Fit on the training tickets and return [n_eval, 10] probabilities for the evaluation tickets.

    Steps: TF-IDF and four embeddings for every ticket; out-of-fold logistic-regression
    probabilities for the training rows (10 folds) and full-fit ones for the evaluation
    rows; leave-one-out neighbour features for the training rows and plain neighbour
    features for the evaluation rows; then the stacker, averaged over three seeds.
    """
    import warnings
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold
    warnings.filterwarnings("ignore", category=ConvergenceWarning)

    t0 = time.time()
    log = (lambda *a: print(f"[x_semantic {time.time()-t0:6.0f}s]", *a, flush=True)) if verbose else (lambda *a: None)
    train_texts, eval_texts = list(train_texts), list(eval_texts)
    y = np.asarray(train_labels, dtype=np.int64)
    n, ne = len(train_texts), len(eval_texts)
    assert y.max() < NC
    prior = np.bincount(y, minlength=NC) / n

    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=200_000, sublinear_tf=True)
    Xtr = vec.fit_transform(train_texts)
    Xev = vec.transform(eval_texts)
    Etr, Eev = {}, {}
    for name, model_name, prefix in EMBEDDERS:
        Etr[name] = _embed(model_name, prefix, train_texts)
        Eev[name] = _embed(model_name, prefix, eval_texts)
        log(f"embedded with {name}")
    Dtr = np.hstack([Etr[e[0]] for e in EMBEDDERS])
    Dev = np.hstack([Eev[e[0]] for e in EMBEDDERS])

    # out-of-fold classifier probabilities for the training rows, full fit for the eval rows
    oof = {"lr_tfidf": np.zeros((n, NC), np.float32), "lr_dense": np.zeros((n, NC), np.float32)}
    for tr, te in StratifiedKFold(10, shuffle=True, random_state=1).split(np.zeros(n), y):
        oof["lr_tfidf"][te] = _full_proba(LogisticRegression(C=10, max_iter=200).fit(Xtr[tr], y[tr]), Xtr[te])
        oof["lr_dense"][te] = _full_proba(LogisticRegression(C=10, max_iter=300).fit(Dtr[tr], y[tr]), Dtr[te])
    evp = {"lr_tfidf": _full_proba(LogisticRegression(C=10, max_iter=200).fit(Xtr, y), Xev),
           "lr_dense": _full_proba(LogisticRegression(C=10, max_iter=300).fit(Dtr, y), Dev)}
    log("classifier probabilities done")

    # neighbour features: leave-one-out for the training rows, plain for the eval rows
    pv_tr, view_names = _view_features(Xtr, Etr, Xtr, Etr, y, loo=True)
    pv_ev, _ = _view_features(Xev, Eev, Xtr, Etr, y, loo=False)
    F_tr, names = _assemble(pv_tr, view_names, oof, prior, n)
    F_ev, _ = _assemble(pv_ev, view_names, evp, prior, ne)
    target = (np.tile(np.arange(NC), n) == np.repeat(y, NC)).astype(int)   # 1 on the true-class row
    log(f"features {F_tr.shape} -> meta model")

    P = np.zeros((ne, NC))
    for seed in META_SEEDS:
        meta = HistGradientBoostingClassifier(categorical_features=[names.index("class_id")], random_state=seed,
                                              **META_PARAMS).fit(F_tr, target)
        s = meta.predict_proba(F_ev)[:, 1].reshape(ne, NC)
        P += s / s.sum(1, keepdims=True) / len(META_SEEDS)
    log("done")
    if return_debug:
        return P, {"lex_s1": pv_ev["lex"][1]["s1"], "lex_top1": pv_ev["lex"][1]["top1"],
                   "fused_top1": pv_ev["fused"][1]["top1"], "lr_tfidf": evp["lr_tfidf"]}
    return P


if __name__ == "__main__":
    # The development protocol: core -> validation, with the accuracy-by-familiarity table.
    from sklearn.metrics import f1_score
    from common import load_split

    x, lab = load_split("train")
    sp = json.loads((ROOT / "data/synthetic/split.json").read_text())
    ctx = [x[i] for i in sp["core"]]; cy = np.array([lab[i] for i in sp["core"]])
    vtx = [x[i] for i in sp["validation"]]; vy = np.array([lab[i] for i in sp["validation"]])
    t0 = time.time()
    P, dbg = fit_predict_proba(ctx, cy, vtx, return_debug=True)
    minutes = (time.time() - t0) / 60
    pred = P.argmax(1); ok = pred == vy
    acc, f1 = float(ok.mean()), float(f1_score(vy, pred, average="macro"))
    print(f"core -> validation: accuracy {acc:.4f}  macro-F1 {f1:.4f}  ({minutes:.1f} min)")
    sim = dbg["lex_s1"]
    hyb = np.where(sim >= 0.3, dbg["lex_top1"], dbg["lr_tfidf"].argmax(1))
    rows = []
    print(f"{'bucket':>8} {'share':>7} {'n':>5} {'stack':>7} {'lex1nn':>7} {'fused1nn':>8}")
    for lo, hi in [(0.0, 0.2), (0.2, 0.3), (0.3, 0.4), (0.4, 0.5), (0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 1.01)]:
        m = (sim >= lo) & (sim < hi)
        r = {"bucket": f"{lo:.1f}-{min(hi, 1.0):.1f}", "share": round(float(m.mean()), 4), "n": int(m.sum()),
             "stack_acc": round(float(ok[m].mean()), 4), "lex_1nn_acc": round(float((dbg['lex_top1'][m] == vy[m]).mean()), 4),
             "fused_1nn_acc": round(float((dbg['fused_top1'][m] == vy[m]).mean()), 4)}
        rows.append(r)
        print(f"{r['bucket']:>8} {r['share']:>7.4f} {r['n']:>5d} {r['stack_acc']:>7.4f} {r['lex_1nn_acc']:>7.4f} {r['fused_1nn_acc']:>8.4f}")
    out = {"track": "semantic", "protocol": "train on core, evaluate on validation", "val_accuracy": acc, "val_macro_f1": f1,
           "runtime_minutes": round(minutes, 1), "lex_1nn_acc": float((dbg["lex_top1"] == vy).mean()),
           "fused_1nn_acc": float((dbg["fused_top1"] == vy).mean()), "simple_hybrid_acc": float((hyb == vy).mean()),
           "buckets": rows}
    (ROOT / "results" / "x_semantic_final_run.json").write_text(json.dumps(out, indent=1))
    np.save(ROOT / "results" / "x_semantic_final_core2val_valprobs.npy", P)
