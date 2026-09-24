"""Reference only (non-parametric): kernel ridge with sharp kernels on baseline TF-IDF, 3-fold CV in core.

Purpose: learn how sharp a similarity function must be before a learned model matches 1-NN.
"""
from __future__ import annotations

import time
import warnings

import numpy as np
import scipy.linalg as sla
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedKFold

from experiments.retrieval_tracks.x_classifier_lib import BASE_VEC, bucket_table, fmt_buckets, load_core_val, log_cv

warnings.filterwarnings("ignore")
xc, yc, _, _ = load_core_val()
xc = np.array(xc, dtype=object)
skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=0)
folds = []
for tr, te in skf.split(xc, yc):
    vec = TfidfVectorizer(**BASE_VEC)
    a = vec.fit_transform(xc[tr])
    b = vec.transform(xc[te])
    Ktr = (a @ a.T).toarray().astype(np.float32)
    Kte = (b @ a.T).toarray().astype(np.float32)
    folds.append((Ktr, yc[tr], Kte, yc[te], Kte.max(axis=1)))
    break  # one fold is enough for a reference
print("fold ready", flush=True)

kernels = {}
for p in (1, 2, 4, 8):
    kernels[f"pow{p}"] = lambda K, p=p: K ** p
for g in (5, 10, 20, 40):
    kernels[f"exp{g}"] = lambda K, g=g: np.exp(g * (K - 1))

for kname, kf in kernels.items():
    for lam in (1e-1, 1e-2, 1e-3):
        t0 = time.time()
        accs, P, Y, S = [], [], [], []
        for Ktr, ytr, Kte, yte, sims in folds:
            A = kf(Ktr).astype(np.float64)
            A[np.diag_indices_from(A)] += lam
            Yoh = -np.ones((len(ytr), 10)); Yoh[np.arange(len(ytr)), ytr] = 1
            alpha = sla.solve(A, Yoh, assume_a="pos")
            pred = (kf(Kte).astype(np.float64) @ alpha).argmax(1)
            accs.append((pred == yte).mean()); P.append(pred); Y.append(yte); S.append(sims)
        rows = bucket_table(np.concatenate(S), np.concatenate(P), np.concatenate(Y))
        log_cv(f"krr_{kname}_lam{lam}", accs, {"buckets": fmt_buckets(rows), "sec": round(time.time() - t0, 1), "note": "fold0 only"})
