"""Kernel probe #3 (fold 0, reference): degree and term weighting for unigram kernels, uncentered +-1 targets."""
import time, warnings, sys
import numpy as np, scipy.linalg as sla, scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import normalize
from experiments.retrieval_tracks.x_classifier_lib import BASE_VEC, bucket_table, fmt_buckets, load_core_val, log_cv
warnings.filterwarnings("ignore")
xc, yc, _, _ = load_core_val(); xc = np.array(xc, dtype=object)
tr, te = next(iter(StratifiedKFold(n_splits=3, shuffle=True, random_state=0).split(xc, yc)))
ytr, yte = yc[tr], yc[te]
bv = TfidfVectorizer(**BASE_VEC); a0 = bv.fit_transform(xc[tr]); b0 = bv.transform(xc[te])
base_sims = (b0 @ a0.T).max(axis=1).toarray().ravel()
Y = -np.ones((len(ytr), 10)); Y[np.arange(len(ytr)), ytr] = 1

def krr(name, Ktr, Kte, lam=0.01):
    t0 = time.time()
    A = Ktr.astype(np.float64); A[np.diag_indices_from(A)] += lam
    alpha = sla.solve(A, Y, assume_a="pos")
    pred = (Kte.astype(np.float64) @ alpha).argmax(1)
    log_cv(name, [(pred == yte).mean()], {"buckets": fmt_buckets(bucket_table(base_sims, pred, yte)), "sec": round(time.time() - t0, 1), "note": "fold0 only"})

def feats(stop, idf_pow, ngram=(1, 1), min_df=2):
    v = TfidfVectorizer(ngram_range=ngram, min_df=min_df, sublinear_tf=True, stop_words="english" if stop else None)
    a = v.fit_transform(xc[tr]); b = v.transform(xc[te])
    if idf_pow != 1.0:
        w = sp.diags(v.idf_ ** (idf_pow - 1.0))
        a = normalize(a @ w); b = normalize(b @ w)
    return a, b

for stop in (True, False):
    for idf_pow in (1.0, 1.5, 2.0):
        a, b = feats(stop, idf_pow)
        Ktr = (a @ a.T).toarray().astype(np.float32); Kte = (b @ a.T).toarray().astype(np.float32)
        pred = ytr[Kte.argmax(1)]
        log_cv(f"k3_uni_stop{int(stop)}_idf{idf_pow}_1nn", [(pred == yte).mean()], {"buckets": fmt_buckets(bucket_table(base_sims, pred, yte))})
        for p in (2, 3, 4):
            krr(f"k3_uni_stop{int(stop)}_idf{idf_pow}_pow{p}", Ktr ** p, Kte ** p)
