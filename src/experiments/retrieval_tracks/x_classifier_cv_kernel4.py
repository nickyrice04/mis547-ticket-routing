"""Reference (non-parametric) kernel ridge in the 6-fold protocol-like CV: how sharp must the kernel be at pool ~13.5k?"""
import time, warnings, sys
import numpy as np, scipy.linalg as sla
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedKFold
from experiments.retrieval_tracks.x_classifier_lib import BASE_VEC, bucket_table, fmt_buckets, load_core_val, log_cv
warnings.filterwarnings("ignore")
xc, yc, _, _ = load_core_val(); xc = np.array(xc, dtype=object)
splits = list(StratifiedKFold(n_splits=6, shuffle=True, random_state=1).split(xc, yc))[:2]
res = {}
for fi, (tr, te) in enumerate(splits):
    ytr, yte = yc[tr], yc[te]
    bv = TfidfVectorizer(**BASE_VEC); a0 = bv.fit_transform(xc[tr]); b0 = bv.transform(xc[te])
    base_sims = (b0 @ a0.T).max(axis=1).toarray().ravel()
    uv = TfidfVectorizer(ngram_range=(1, 1), min_df=2, sublinear_tf=True, stop_words="english")
    a = uv.fit_transform(xc[tr]); b = uv.transform(xc[te])
    Ktr = (a @ a.T).toarray().astype(np.float32); Kte = (b @ a.T).toarray().astype(np.float32)
    Y = -np.ones((len(ytr), 10)); Y[np.arange(len(ytr)), ytr] = 1
    for p in (2, 3, 4, 6, 8, 12):
        A = (Ktr ** p).astype(np.float64); A[np.diag_indices_from(A)] += 0.01
        alpha = sla.solve(A, Y, assume_a="pos")
        pred = ((Kte ** p).astype(np.float64) @ alpha).argmax(1)
        res.setdefault(p, []).append((pred, yte, base_sims))
        print(fi, p, (pred == yte).mean(), flush=True)
for p, lst in res.items():
    P = np.concatenate([l[0] for l in lst]); Y_ = np.concatenate([l[1] for l in lst]); S = np.concatenate([l[2] for l in lst])
    log_cv(f"k4_unistop_pow{p}", [(l[0] == l[1]).mean() for l in lst], {"buckets": fmt_buckets(bucket_table(S, P, Y_)), "cv": "6fold_seed1"})
