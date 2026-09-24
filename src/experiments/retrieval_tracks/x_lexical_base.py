"""Track 'lexical': base-model outputs for one (train pool, eval set) pair.

Everything here is fitted on the train pool only. The eval texts are only transformed.
Outputs: top-50 neighbours under several sparse similarity views, and class probabilities from
text classifiers (logistic regression, complement NB, the baseline MLP network).
"""
from __future__ import annotations
import time, warnings
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import ComplementNB
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC
from experiments.retrieval_tracks.x_lexical_views import TfidfView, BM25View, topk

N_CLASSES = 10
TOPK = 50


def base_outputs(train_texts, y_train, eval_texts, use_mlp=True, mlp_seeds=(0,), verbose=False):
    warnings.filterwarnings("ignore")
    y_train = np.asarray(y_train)
    out = {}
    t0 = time.time()
    w = TfidfView(ngram_range=(1, 2), min_df=2, max_features=200_000).fit(train_texts)
    Xtr, Xte = w.D, w.transform(eval_texts)
    Sw = (Xte @ Xtr.T).toarray()
    c = TfidfView(analyzer="char_wb", ngram_range=(2, 5), min_df=2, max_features=500_000).fit(train_texts)
    Sc = c.sims(eval_texts)
    for name, S in (("w12", Sw), ("c25", Sc), ("wc", (Sw + Sc) / 2)):
        idx, val = topk(S, TOPK)
        out[f"knn_{name}_idx"], out[f"knn_{name}_val"] = idx.astype(np.int32), val.astype(np.float32)
    # cross-view sims of each view's nearest neighbour (does the other view agree it is close?)
    r = np.arange(Sw.shape[0])
    out["c25_sim_of_w12_nn"] = Sc[r, out["knn_w12_idx"][:, 0]].astype(np.float32)
    out["w12_sim_of_c25_nn"] = Sw[r, out["knn_c25_idx"][:, 0]].astype(np.float32)
    del Sw, Sc
    if verbose: print("  knn", round(time.time() - t0), flush=True)
    for C in (1.0, 10.0):
        m = LogisticRegression(C=C, max_iter=3000).fit(Xtr, y_train)
        out[f"p_lr{int(C)}"] = _full(m.predict_proba(Xte), m.classes_)
    m = ComplementNB(alpha=0.3).fit(Xtr, y_train)
    out["p_cnb"] = _full(m.predict_proba(Xte), m.classes_)
    m = LinearSVC(C=0.1).fit(Xtr, y_train)
    out["d_svc"] = _full(m.decision_function(Xte), m.classes_, fill=-3.0)
    if verbose: print("  linear", round(time.time() - t0), flush=True)
    if use_mlp:
        ps = []
        for s in mlp_seeds:
            m = MLPClassifier(hidden_layer_sizes=(256,), max_iter=60, early_stopping=True,
                              n_iter_no_change=5, random_state=s).fit(Xtr, y_train)
            ps.append(_full(m.predict_proba(Xte), m.classes_))
        out["p_mlp"] = np.mean(ps, 0)
        if verbose: print("  mlp", round(time.time() - t0), flush=True)
    out["len_words"] = np.array([len(t.split()) for t in eval_texts], dtype=np.float32)
    return {k: (v.astype(np.float32) if v.dtype == np.float64 else v) for k, v in out.items()}


def _full(p, classes, fill=0.0):
    if len(classes) == N_CLASSES:
        return p
    full = np.full((p.shape[0], N_CLASSES), fill, dtype=p.dtype)
    full[:, classes] = p
    return full
