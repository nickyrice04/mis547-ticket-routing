"""Kernel-world probe #2 (fold 0 of core, reference only): which similarity + sharpness works best?
Centered targets (class prior removed) so sharp kernels are not swamped by the majority class.
"""
from __future__ import annotations

import sys
import time
import warnings

import numpy as np
import scipy.linalg as sla
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import normalize

from experiments.retrieval_tracks.x_classifier_lib import BASE_VEC, bucket_table, fmt_buckets, load_core_val, log_cv

warnings.filterwarnings("ignore")
xc, yc, _, _ = load_core_val()
xc = np.array(xc, dtype=object)
tr, te = next(iter(StratifiedKFold(n_splits=3, shuffle=True, random_state=0).split(xc, yc)))
ytr, yte = yc[tr], yc[te]
bv = TfidfVectorizer(**BASE_VEC)
a0 = bv.fit_transform(xc[tr]); b0 = bv.transform(xc[te])
K0tr = (a0 @ a0.T).toarray().astype(np.float32); K0te = (b0 @ a0.T).toarray().astype(np.float32)
base_sims = K0te.max(1)
prior = np.bincount(ytr, minlength=10) / len(ytr)
Yc = np.eye(10)[ytr] - prior


def krr(name, Ktr, Kte, lam=0.01):
    t0 = time.time()
    A = Ktr.astype(np.float64); A[np.diag_indices_from(A)] += lam
    alpha = sla.solve(A, Yc, assume_a="pos")
    pred = (Kte.astype(np.float64) @ alpha + prior).argmax(1)
    rows = bucket_table(base_sims, pred, yte)
    log_cv(name, [(pred == yte).mean()], {"buckets": fmt_buckets(rows), "sec": round(time.time() - t0, 1), "note": "fold0 only"})


def nn1(name, Kte):
    pred = ytr[Kte.argmax(1)]
    rows = bucket_table(base_sims, pred, yte)
    log_cv(name, [(pred == yte).mean()], {"buckets": fmt_buckets(rows), "note": "fold0 only"})


which = sys.argv[1] if len(sys.argv) > 1 else "all"
if which in ("all", "tfidf"):
    for p in (3, 4, 6):
        krr(f"k2_tfidf_pow{p}_c", K0tr ** p, K0te ** p)
    for g in (5, 10, 20, 40):
        krr(f"k2_tfidf_exp{g}_c", np.exp(g * (K0tr - 1)), np.exp(g * (K0te - 1)))
    # mixtures of linear + sharp
    for w in (0.1, 0.3):
        krr(f"k2_tfidf_pow4+{w}lin_c", K0tr ** 4 + w * K0tr, K0te ** 4 + w * K0te)

if which in ("all", "svd"):
    for k in (256, 512, 1024):
        svd = TruncatedSVD(n_components=k, algorithm="randomized", n_iter=5, random_state=0)
        za = normalize(svd.fit_transform(a0)); zb = normalize(svd.transform(b0))
        Ktr = (za @ za.T).astype(np.float32); Kte = (zb @ za.T).astype(np.float32)
        nn1(f"k2_svd{k}_1nn", Kte)
        for p in (4, 8, 16):
            krr(f"k2_svd{k}_pow{p}_c", np.clip(Ktr, 0, None) ** p, np.clip(Kte, 0, None) ** p)
        for g in (10, 20, 40):
            krr(f"k2_svd{k}_exp{g}_c", np.exp(g * (Ktr - 1)), np.exp(g * (Kte - 1)))
