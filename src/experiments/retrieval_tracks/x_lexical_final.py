"""Track 'lexical': retrieval-first ticket routing with sparse text features, stacked with text classifiers.

Method
    1. Pool = the training tickets (+ optionally extra labelled tickets from this track's data files, see EXTRA below).
    2. Base evidence for a ticket, computed from its text only:
         * cosine similarity to every pool ticket under three sparse views (word 1-2gram TF-IDF, char_wb 2-5gram
           TF-IDF, and their mean): per-class best similarity, similarity-weighted votes (k=20 power 4, k=50 power 1),
           per-source best similarity, and scalar context (top similarities, margin to the best rival class, purity)
         * class scores of text classifiers trained on the pool (logistic regression C=1 and C=10, complement NB,
           linear SVM, optionally the baseline MLP network)
    3. A pairwise meta-model (LightGBM, one row per (ticket, queue) pair, parameters shared across queues) answers
       "is this queue right for this ticket?". It is trained with K-fold cross-fitting INSIDE the training set: every
       training ticket's evidence comes from base models / pools that did not contain it.
    4. Scores are normalised across the ten queues and temperature-calibrated on out-of-fold meta predictions.

EXTRA POOL (OFF by default; env X_LEXICAL_EXTRA = "none" | "en" | "en+de", or the `extra` argument)
    data/x_lexical/extra_en_pool.jsonl       English tickets that the parquet mislabels as language == "de"
    data/x_lexical/extra_de_translated.jsonl genuinely German tickets, machine-translated to English (opus-mt-de-en)
  Both come from parquet rows with language == "de", which are in neither the train nor the test split and carry
  their own real queue labels. READ THIS BEFORE SWITCHING IT ON: the German half of the dataset turned out to be
  sentence-by-sentence translations of the English tickets. An English ticket's German twin carries the same queue, so
  adding these rows lets retrieval look the answer up (inside core the out-of-fold accuracy jumps from 72% to the
  high 80s). src/experiments/german_embeddings/add_german.py already treats "a German translation of a test ticket" as leakage, and a lexical
  guard cannot detect translation twins, so numbers obtained with the extra pool must NOT be reported as clean
  generalisation. The default, and every headline number of this track, uses the training tickets only.
  Guards that are applied when it is on: extra rows equal to a training ticket are dropped, and extra rows that are an
  exact or near-exact copy (baseline TF-IDF cosine >= GUARD_SIM) of any ticket being predicted are dropped.

Only the ticket text is used at prediction time. Nothing is fitted on the tickets being predicted.
"""
from __future__ import annotations

import json
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import ComplementNB
from sklearn.svm import LinearSVC

ROOT = Path(__file__).resolve().parents[3]   # retrieval_tracks -> experiments -> src -> repository root
EXTRA_FILES = {"en": ROOT / "data/x_lexical/extra_en_pool.jsonl", "de": ROOT / "data/x_lexical/extra_de_translated.jsonl"}
N_CLASSES = 10
GUARD_SIM = 0.95
N_FOLDS = 5
META_SEEDS = (0, 1, 2)
LGB_PARAMS = dict(objective="binary", num_leaves=15, learning_rate=0.05, n_estimators=300, colsample_bytree=0.5,
                  subsample=0.8, subsample_freq=1, min_child_samples=50, reg_lambda=5.0, n_jobs=3, verbose=-1)
SOURCES = (0, 1, 2)          # 0 = training tickets, 1 = extra English, 2 = extra German translated


# ----------------------------------------------------------------------------------------------- extra pool
def load_extra(mode=None):
    mode = os.environ.get("X_LEXICAL_EXTRA", "none") if mode is None else mode
    texts, labels, src = [], [], []
    if mode and mode != "none":
        for s, key in ((1, "en"), (2, "de")):
            if key in mode.split("+") and EXTRA_FILES[key].exists():
                for line in open(EXTRA_FILES[key]):
                    r = json.loads(line)
                    if r["text"]:
                        texts.append(r["text"]); labels.append(r["label"]); src.append(s)
    return texts, np.array(labels, dtype=int), np.array(src, dtype=int)


def _base_vec():
    return TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=200_000, sublinear_tf=True, dtype=np.float32)


def guard_extra(extra_texts, ref_texts, predict_texts):
    """Boolean keep-mask over extra rows: drop copies of training tickets and (near-)copies of tickets to predict."""
    if not len(extra_texts):
        return np.zeros(0, dtype=bool)
    block = set(ref_texts) | set(predict_texts)
    keep = np.array([t not in block for t in extra_texts])
    vec = _base_vec().fit(list(ref_texts) + list(extra_texts))
    E, P = vec.transform(extra_texts), vec.transform(predict_texts)
    mx = np.zeros(len(extra_texts), dtype=np.float32)
    for s in range(0, P.shape[0], 2000):
        mx = np.maximum(mx, (E @ P[s:s + 2000].T).max(axis=1).toarray().ravel())
    return keep & (mx < GUARD_SIM)


# ----------------------------------------------------------------------------------------------- base evidence
def _votes(idx, val, y_pool, k, power):
    lab = y_pool[idx[:, :k]]
    w = np.maximum(val[:, :k], 0) ** power
    out = np.zeros((idx.shape[0], N_CLASSES))
    np.add.at(out, (np.repeat(np.arange(idx.shape[0]), k), lab.ravel()), w.ravel())
    return out / np.maximum(out.sum(1, keepdims=True), 1e-12)


def _topk(S, k):
    idx = np.argpartition(-S, k - 1, axis=1)[:, :k]
    val = np.take_along_axis(S, idx, 1)
    o = np.argsort(-val, axis=1)
    return np.take_along_axis(idx, o, 1), np.take_along_axis(val, o, 1)


def _full(p, classes, fill=0.0):
    if len(classes) == N_CLASSES:
        return p
    full = np.full((p.shape[0], N_CLASSES), fill, dtype=p.dtype)
    full[:, classes] = p
    return full


def base_evidence(pool_texts, pool_y, pool_src, eval_texts, use_mlp=False):
    """Everything is fitted on the pool only. Returns (per_class blocks dict, context dict)."""
    warnings.filterwarnings("ignore")
    pool_y, pool_src = np.asarray(pool_y), np.asarray(pool_src)
    pc, ctx = {}, {}
    wv = _base_vec()
    Xp = wv.fit_transform(pool_texts); Xe = wv.transform(eval_texts)
    cv = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=2, max_features=500_000, sublinear_tf=True, dtype=np.float32)
    Cp = cv.fit_transform(pool_texts); Ce = cv.transform(eval_texts)
    Sw = (Xe @ Xp.T).toarray(); Sc = (Ce @ Cp.T).toarray()
    del Cp, Ce
    r = np.arange(Sw.shape[0])
    nn_lab = {}
    for name, S in (("w12", Sw), ("c25", Sc), ("wc", (Sw + Sc) / 2)):
        idx, val = _topk(S, 50)
        ms = np.zeros((S.shape[0], N_CLASSES))
        lab = pool_y[idx]
        for c in range(N_CLASSES):
            ms[:, c] = np.where(lab == c, val, 0).max(1)
        pc[f"{name}_maxsim"] = ms
        pc[f"{name}_vote20p4"] = _votes(idx, val, pool_y, 20, 4)
        pc[f"{name}_vote50p1"] = _votes(idx, val, pool_y, 50, 1)
        srt = np.sort(ms, 1)
        nn_lab[name] = lab[:, 0]
        for k, v in (("s1", val[:, 0]), ("s2", val[:, 1]), ("s5", val[:, 4]), ("mean10", val[:, :10].mean(1)),
                     ("rival", srt[:, -2]), ("margin", srt[:, -1] - srt[:, -2]),
                     ("pur5", (lab[:, :5] == lab[:, [0]]).mean(1))):
            ctx[f"{name}_{k}"] = v
        if name == "w12":
            ctx["c25_sim_of_w12_nn"] = Sc[r, idx[:, 0]]
        if name == "c25":
            ctx["w12_sim_of_c25_nn"] = Sw[r, idx[:, 0]]
        if name == "wc":
            ctx["wc_nn_src"] = pool_src[idx[:, 0]].astype(float)
            for s in SOURCES:                                   # per-source, per-class best similarity
                m = np.zeros((S.shape[0], N_CLASSES))
                for c in range(N_CLASSES):
                    cols = np.where((pool_src == s) & (pool_y == c))[0]
                    if len(cols):
                        m[:, c] = S[:, cols].max(1)
                pc[f"wc_maxsim_src{s}"] = m
                ctx[f"wc_s1_src{s}"] = m.max(1)
    ctx["nn_agree"] = (nn_lab["w12"] == nn_lab["c25"]).astype(float)
    ctx["len_words"] = np.array([len(t.split()) for t in eval_texts], dtype=float)
    del Sw, Sc
    for C in (1.0, 10.0):
        m = LogisticRegression(C=C, max_iter=3000).fit(Xp, pool_y)
        pc[f"p_lr{int(C)}"] = _full(m.predict_proba(Xe), m.classes_)
    m = ComplementNB(alpha=0.3).fit(Xp, pool_y)
    pc["p_cnb"] = _full(m.predict_proba(Xe), m.classes_)
    m = LinearSVC(C=0.1).fit(Xp, pool_y)
    pc["d_svc"] = _full(m.decision_function(Xe), m.classes_, fill=-3.0)
    if use_mlp:
        from sklearn.neural_network import MLPClassifier
        m = MLPClassifier(hidden_layer_sizes=(256,), max_iter=60, early_stopping=True, n_iter_no_change=5, random_state=0).fit(Xp, pool_y)
        pc["p_mlp"] = _full(m.predict_proba(Xe), m.classes_)
    return pc, ctx


def pair_matrix(pc, ctx, prior):
    """One row per (ticket, queue): the queue's value in every per-class block, its gap to the best other queue,
    the ticket-level context, the queue id and its prior."""
    names, cols = [], []
    n = len(next(iter(ctx.values())))
    for k in sorted(pc):
        b = np.asarray(pc[k], dtype=np.float64)
        top = np.sort(b, 1)
        best_other = np.where(b >= top[:, [-1]], top[:, [-2]], top[:, [-1]])
        cols += [b.ravel(), (b - best_other).ravel()]; names += [k, k + "_gap"]
    for k in sorted(ctx):
        cols.append(np.repeat(np.asarray(ctx[k], dtype=np.float64), N_CLASSES)); names.append(k)
    cols += [np.tile(np.arange(N_CLASSES), n).astype(float), np.tile(np.asarray(prior), n)]
    names += ["class_id", "class_prior"]
    return np.column_stack(cols).astype(np.float32), names


# ----------------------------------------------------------------------------------------------- meta model
def _fit_meta(X, y_pair, names, seeds=META_SEEDS):
    import lightgbm as lgb
    cat = [names.index("class_id")]
    return [lgb.LGBMClassifier(random_state=s, **LGB_PARAMS).fit(X, y_pair, categorical_feature=cat) for s in seeds]


def _predict_meta(models, X):
    s = sum(m.predict_proba(X)[:, 1] for m in models).reshape(-1, N_CLASSES) / len(models)
    return s / np.maximum(s.sum(1, keepdims=True), 1e-12)


def _apply_temperature(P, T):
    L = np.log(np.clip(P, 1e-9, 1)) / T
    L -= L.max(1, keepdims=True)
    E = np.exp(L)
    return E / E.sum(1, keepdims=True)


def _fit_temperature(P, y):
    grid = np.exp(np.linspace(np.log(0.5), np.log(2.5), 41))
    nll = [-np.mean(np.log(np.clip(_apply_temperature(P, T)[np.arange(len(y)), y], 1e-9, 1))) for T in grid]
    return float(grid[int(np.argmin(nll))])


def crossfit(train_texts, y, extra_texts, extra_y, extra_src, n_folds=N_FOLDS, use_mlp=False, seed=0, verbose=False):
    """Cross-fitted pair rows for every training ticket. Returns X [n*10, F], names, fold id per ticket."""
    xa = np.array(train_texts, dtype=object); ex = np.array(extra_texts, dtype=object)
    prior = np.bincount(y, minlength=N_CLASSES) / len(y)
    X = None; fold_id = np.zeros(len(y), dtype=int)
    t0 = time.time()
    for f, (tr, te) in enumerate(StratifiedKFold(n_folds, shuffle=True, random_state=seed).split(xa, y)):
        keep = guard_extra(list(ex), list(xa[tr]), list(xa[te])) if len(ex) else np.zeros(0, dtype=bool)
        pool_t = list(xa[tr]) + list(ex[keep])
        pool_y = np.concatenate([y[tr], extra_y[keep]]); pool_s = np.concatenate([np.zeros(len(tr), dtype=int), extra_src[keep]])
        pc, ctx = base_evidence(pool_t, pool_y, pool_s, list(xa[te]), use_mlp)
        A, names = pair_matrix(pc, ctx, prior)
        if X is None:
            X = np.zeros((len(y) * N_CLASSES, A.shape[1]), dtype=np.float32)
        rows = (te[:, None] * N_CLASSES + np.arange(N_CLASSES)[None, :]).ravel()
        X[rows] = A; fold_id[te] = f
        if verbose:
            print(f"  cross-fit fold {f}: pool {len(pool_t)} (extra kept {int(keep.sum())}/{len(ex)}), {time.time() - t0:.0f}s", flush=True)
    return X, names, fold_id


def fit_predict_proba(train_texts, train_labels, eval_texts, extra=None, use_mlp=False, n_folds=N_FOLDS, verbose=False, return_details=False):
    y = np.asarray(train_labels, dtype=int)
    train_texts, eval_texts = list(train_texts), list(eval_texts)
    ex_t, ex_y, ex_s = load_extra(extra)
    if len(ex_t):                                   # guard against copies of the tickets being predicted
        keep = guard_extra(ex_t, train_texts, eval_texts)
        if verbose:
            print(f"extra pool: {len(ex_t)} rows, {int((~keep).sum())} dropped by the guards", flush=True)
        ex_t, ex_y, ex_s = [t for t, k in zip(ex_t, keep) if k], ex_y[keep], ex_s[keep]
    prior = np.bincount(y, minlength=N_CLASSES) / len(y)
    X, names, fold_id = crossfit(train_texts, y, ex_t, ex_y, ex_s, n_folds=n_folds, use_mlp=use_mlp, verbose=verbose)
    y_pair = (np.tile(np.arange(N_CLASSES), len(y)) == np.repeat(y, N_CLASSES)).astype(int)
    # out-of-fold meta predictions -> temperature for calibration (and an honest in-train estimate)
    oof = np.zeros((len(y), N_CLASSES)); fp = np.repeat(fold_id, N_CLASSES)
    for f in np.unique(fold_id):
        oof[fold_id == f] = _predict_meta(_fit_meta(X[fp != f], y_pair[fp != f], names, seeds=(0,)), X[fp == f])
    T = _fit_temperature(oof, y)
    if verbose:
        print(f"out-of-fold meta accuracy inside train {np.mean(oof.argmax(1) == y):.4f}, temperature {T:.3f}", flush=True)
    models = _fit_meta(X, y_pair, names)
    pool_t = train_texts + ex_t
    pool_y = np.concatenate([y, ex_y]); pool_s = np.concatenate([np.zeros(len(y), dtype=int), ex_s])
    pc, ctx = base_evidence(pool_t, pool_y, pool_s, eval_texts, use_mlp)
    A, _ = pair_matrix(pc, ctx, prior)
    P = _apply_temperature(_predict_meta(models, A), T)
    if return_details:
        return P, dict(oof=oof, temperature=T, pc=pc, ctx=ctx, n_extra=len(ex_t))
    return P


if __name__ == "__main__":
    from sklearn.metrics import accuracy_score, f1_score
    sys.path.insert(0, str(ROOT / "src"))
    from common import load_split
    x, yy = load_split("train")
    sp = json.loads((ROOT / "data/synthetic/split.json").read_text())
    xc, yc = [x[i] for i in sp["core"]], [yy[i] for i in sp["core"]]
    xv, yv = [x[i] for i in sp["validation"]], np.array([yy[i] for i in sp["validation"]])
    t0 = time.time()
    P = fit_predict_proba(xc, yc, xv, verbose=True)
    pred = P.argmax(1)
    print(f"core -> validation  accuracy {accuracy_score(yv, pred):.4f}  macro-F1 {f1_score(yv, pred, average='macro'):.4f}  "
          f"({(time.time() - t0) / 60:.1f} min, extra={os.environ.get('X_LEXICAL_EXTRA', 'none')})")
