"""Feature-set comparison on fold 0 of core: 1-NN, LinearSVC, and kernel ridge (pow2/pow4) per feature set."""
from __future__ import annotations

import time
import warnings

import numpy as np
import scipy.linalg as sla
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import normalize
from sklearn.svm import LinearSVC

from experiments.retrieval_tracks.x_classifier_lib import BASE_VEC, bucket_table, fmt_buckets, load_core_val, log_cv

warnings.filterwarnings("ignore")
xc, yc, _, _ = load_core_val()
xc = np.array(xc, dtype=object)
tr, te = next(iter(StratifiedKFold(n_splits=3, shuffle=True, random_state=0).split(xc, yc)))
bv = TfidfVectorizer(**BASE_VEC)
a0 = bv.fit_transform(xc[tr]); b0 = bv.transform(xc[te])
base_sims = (b0 @ a0.T).max(axis=1).toarray().ravel()
ytr, yte = yc[tr], yc[te]

FEATS = {
    "base": [BASE_VEC],
    "w13": [dict(ngram_range=(1, 3), min_df=2, sublinear_tf=True)],
    "w14": [dict(ngram_range=(1, 4), min_df=2, sublinear_tf=True)],
    "w15": [dict(ngram_range=(1, 5), min_df=2, sublinear_tf=True)],
    "w11": [dict(ngram_range=(1, 1), min_df=2, sublinear_tf=True)],
    "w12_nosub": [dict(ngram_range=(1, 2), min_df=2, max_features=200_000)],
    "w12_stop": [dict(ngram_range=(1, 2), min_df=2, sublinear_tf=True, stop_words="english")],
    "c35": [dict(analyzer="char_wb", ngram_range=(3, 5), min_df=2, sublinear_tf=True)],
    "c36": [dict(analyzer="char", ngram_range=(3, 6), min_df=2, sublinear_tf=True, max_features=1_000_000)],
    "w12+c35": [BASE_VEC, dict(analyzer="char_wb", ngram_range=(3, 5), min_df=2, sublinear_tf=True)],
}


def report(name, pred, t0):
    rows = bucket_table(base_sims, pred, yte)
    log_cv(name, [(pred == yte).mean()], {"buckets": fmt_buckets(rows), "sec": round(time.time() - t0, 1), "note": "fold0 only"})


import sys
only = sys.argv[1].split(",") if len(sys.argv) > 1 else list(FEATS)
for fname in only:
    t0 = time.time()
    mats = []
    for kw in FEATS[fname]:
        v = TfidfVectorizer(**kw)
        mats.append((v.fit_transform(xc[tr]), v.transform(xc[te])))
    a = normalize(sp.hstack([m[0] for m in mats]).tocsr()); b = normalize(sp.hstack([m[1] for m in mats]).tocsr())
    print(fname, a.shape, f"nnz/row={a.nnz / a.shape[0]:.0f}", flush=True)
    Kte = (b @ a.T).toarray().astype(np.float32)
    report(f"feat_{fname}_1nn", ytr[Kte.argmax(1)], t0)
    t0 = time.time()
    report(f"feat_{fname}_svcC3", LinearSVC(C=3, max_iter=3000).fit(a, ytr).predict(b), t0)
    Ktr = (a @ a.T).toarray().astype(np.float32)
    Yoh = -np.ones((len(ytr), 10)); Yoh[np.arange(len(ytr)), ytr] = 1
    for p in (2, 4):
        t0 = time.time()
        A = (Ktr ** p).astype(np.float64); A[np.diag_indices_from(A)] += 0.01
        alpha = sla.solve(A, Yoh, assume_a="pos")
        report(f"feat_{fname}_krr_pow{p}", ((Kte ** p).astype(np.float64) @ alpha).argmax(1), t0)
    del Ktr, Kte
