"""Fold-0 test of explicit pair (degree-2 conjunction) features with linear models."""
from __future__ import annotations

import sys
import time
import warnings

import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import RidgeClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import LinearSVC

from experiments.retrieval_tracks.x_classifier_lib import BASE_VEC, bucket_table, fmt_buckets, load_core_val, log_cv
from experiments.retrieval_tracks.x_classifier_pairs import PairIndexer

warnings.filterwarnings("ignore")
xc, yc, _, _ = load_core_val()
xc = np.array(xc, dtype=object)
tr, te = next(iter(StratifiedKFold(n_splits=3, shuffle=True, random_state=0).split(xc, yc)))
bv = TfidfVectorizer(**BASE_VEC)
a0 = bv.fit_transform(xc[tr]); b0 = bv.transform(xc[te])
base_sims = (b0 @ a0.T).max(axis=1).toarray().ravel()
ytr, yte = yc[tr], yc[te]


def report(name, pred, t0, extra=None):
    rows = bucket_table(base_sims, pred, yte)
    e = {"buckets": fmt_buckets(rows), "sec": round(time.time() - t0, 1), "note": "fold0 only"}
    e.update(extra or {})
    log_cv(name, [(pred == yte).mean()], e)


uv = TfidfVectorizer(ngram_range=(1, 1), min_df=2, sublinear_tf=True)
au = uv.fit_transform(xc[tr]); bu = uv.transform(xc[te])
for min_df in (1, 2):
    t0 = time.time()
    pi = PairIndexer(min_df=min_df)
    pa = pi.fit_transform(au); pb = pi.transform(bu)
    print(f"pairs min_df={min_df}: {pa.shape}, nnz/row={pa.nnz / pa.shape[0]:.0f}, build {time.time() - t0:.0f}s", flush=True)
    for C in (1, 10, 100):
        t0 = time.time()
        m = LinearSVC(C=C, max_iter=3000).fit(pa, ytr)
        report(f"pairs_df{min_df}_svcC{C}", m.predict(pb), t0, {"n_feat": pa.shape[1]})
    for lin_w in (0.5, 1.0):
        A = sp.hstack([pa, lin_w * a0]).tocsr(); B = sp.hstack([pb, lin_w * b0]).tocsr()
        for C in (10,):
            t0 = time.time()
            m = LinearSVC(C=C, max_iter=3000).fit(A, ytr)
            report(f"pairs_df{min_df}+lin{lin_w}_svcC{C}", m.predict(B), t0, {"n_feat": A.shape[1]})
    t0 = time.time()
    m = RidgeClassifier(alpha=0.01, solver="sparse_cg", max_iter=300).fit(pa, ytr)
    report(f"pairs_df{min_df}_ridge0.01", m.predict(pb), t0, {"n_feat": pa.shape[1]})
