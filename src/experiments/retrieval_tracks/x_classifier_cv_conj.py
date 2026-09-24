"""CV inside core for conjunction-feature linear models.
Usage: x_classifier_cv_conj.py <nfolds> '<json list of configs>'
config: {"name", "stop":0/1, "w":[w1,w2,w3], "min_df":2, "max_terms":40, "model":"ridge"|"svc", "reg":float, "base_lin":0/1}
"""
from __future__ import annotations

import json
import sys
import time
import warnings

import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import RidgeClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import LinearSVC

from experiments.retrieval_tracks.x_classifier_conj import Conjunctions
from experiments.retrieval_tracks.x_classifier_lib import BASE_VEC, bucket_table, fmt_buckets, load_core_val, log_cv

warnings.filterwarnings("ignore")
nfolds = int(sys.argv[1])
configs = json.loads(sys.argv[2])
xc, yc, _, _ = load_core_val()
xc = np.array(xc, dtype=object)
splits = list(StratifiedKFold(n_splits=3, shuffle=True, random_state=0).split(xc, yc))[:nfolds]

cache = {}


def blocks(fi, stop, min_df, max_terms, orders):
    tr, te = splits[fi]
    out = {}
    for o in orders:
        key = (fi, stop, min_df, max_terms if o == 3 else None, o)
        if key not in cache:
            t0 = time.time()
            if o == 1 and stop == -1:   # baseline (1,2)-gram block
                v = TfidfVectorizer(**BASE_VEC)
                cache[key] = (v.fit_transform(xc[tr]), v.transform(xc[te]))
            else:
                ukey = (fi, stop, "uni")
                if ukey not in cache:
                    v = TfidfVectorizer(ngram_range=(1, 1), min_df=2, sublinear_tf=True, stop_words="english" if stop else None)
                    cache[ukey] = (v.fit_transform(xc[tr]), v.transform(xc[te]))
                au, bu = cache[ukey]
                if o == 1:
                    cache[key] = (au, bu)
                else:
                    cj = Conjunctions(order=o, min_df=min_df, max_terms=max_terms if o == 3 else None)
                    cache[key] = (cj.fit_transform(au), cj.transform(bu))
            print(f"  built block order={o} stop={stop} fold={fi}: {cache[key][0].shape} nnz/row={cache[key][0].nnz / cache[key][0].shape[0]:.0f} {time.time() - t0:.0f}s", flush=True)
        out[o] = cache[key]
    return out


sims_cache = {}
for cfg in configs:
    t0 = time.time()
    accs, P, Y, S = [], [], [], []
    w = cfg["w"]
    orders = [o for o, wt in zip((1, 2, 3), w) if wt > 0]
    nfeat = 0
    for fi, (tr, te) in enumerate(splits):
        if fi not in sims_cache:
            v = TfidfVectorizer(**BASE_VEC); a0 = v.fit_transform(xc[tr]); b0 = v.transform(xc[te])
            sims_cache[fi] = (b0 @ a0.T).max(axis=1).toarray().ravel()
        bl = blocks(fi, cfg.get("stop", 1), cfg.get("min_df", 2), cfg.get("max_terms", 40), orders)
        A = sp.hstack([w[o - 1] * bl[o][0] for o in orders]).tocsr()
        B = sp.hstack([w[o - 1] * bl[o][1] for o in orders]).tocsr()
        nfeat = A.shape[1]
        if cfg["model"] == "ridge":
            m = RidgeClassifier(alpha=cfg["reg"], solver="sparse_cg", max_iter=cfg.get("max_iter", 300), tol=1e-4)
        else:
            m = LinearSVC(C=cfg["reg"], max_iter=3000)
        m.fit(A, yc[tr])
        pred = m.predict(B)
        accs.append((pred == yc[te]).mean()); P.append(pred); Y.append(yc[te]); S.append(sims_cache[fi])
    rows = bucket_table(np.concatenate(S), np.concatenate(P), np.concatenate(Y))
    log_cv("conj_" + cfg["name"], accs, {"buckets": fmt_buckets(rows), "sec": round(time.time() - t0, 1), "n_feat": nfeat, "cfg": cfg})
