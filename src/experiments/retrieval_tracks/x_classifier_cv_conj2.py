"""CV inside core that mimics the real protocol's pool size: 6-fold split (pool = 5/6 of core, about 13.5k),
first <nfolds> folds. Conjunction-feature linear models of any order.

Usage: x_classifier_cv_conj2.py <nfolds> '<json configs>'
config: {"name", "stop":1, "w":{"1":0.5,"2":1,"3":1,"4":1}, "mt":{"3":40,"4":16}, "min_df":2, "model":"ridge"|"svc", "reg":0.1}
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
splits = list(StratifiedKFold(n_splits=6, shuffle=True, random_state=1).split(xc, yc))[:nfolds]
cache, sims_cache = {}, {}


def block(fi, stop, order, min_df, mt):
    key = (fi, stop, order, min_df, mt)
    if key not in cache:
        t0 = time.time()
        tr, te = splits[fi]
        ukey = (fi, stop, "uni")
        if ukey not in cache:
            v = TfidfVectorizer(ngram_range=(1, 1), min_df=2, sublinear_tf=True, stop_words="english" if stop else None)
            cache[ukey] = (v.fit_transform(xc[tr]), v.transform(xc[te]))
        au, bu = cache[ukey]
        if order == 1:
            cache[key] = (au, bu)
        else:
            cj = Conjunctions(order=order, min_df=min_df, max_terms=mt)
            cache[key] = (cj.fit_transform(au), cj.transform(bu))
        a = cache[key][0]
        print(f"  block order={order} stop={stop} mt={mt} fold={fi}: {a.shape} nnz/row={a.nnz / a.shape[0]:.0f} {time.time() - t0:.0f}s", flush=True)
    return cache[key]


for cfg in configs:
    t0 = time.time()
    accs, nn_accs, P, Y, S = [], [], [], [], []
    nfeat = 0
    for fi, (tr, te) in enumerate(splits):
        if fi not in sims_cache:
            v = TfidfVectorizer(**BASE_VEC); a0 = v.fit_transform(xc[tr]); b0 = v.transform(xc[te])
            d = (b0 @ a0.T).toarray()
            sims_cache[fi] = (d.max(axis=1), yc[tr][d.argmax(axis=1)]); del d
        sims, nnlab = sims_cache[fi]
        if cfg["model"] == "1nn":
            pred = nnlab
        else:
            bl = [(float(w), block(fi, cfg.get("stop", 1), int(o), cfg.get("min_df", 2), cfg.get("mt", {}).get(o))) for o, w in cfg["w"].items() if float(w) > 0]
            A = sp.hstack([w * b[0] for w, b in bl]).tocsr(); B = sp.hstack([w * b[1] for w, b in bl]).tocsr()
            nfeat = A.shape[1]
            if cfg["model"] == "ridge":
                m = RidgeClassifier(alpha=cfg["reg"], solver="sparse_cg", max_iter=cfg.get("max_iter", 300), tol=1e-4)
            else:
                m = LinearSVC(C=cfg["reg"], max_iter=3000)
            m.fit(A, yc[tr]); pred = m.predict(B)
        accs.append((pred == yc[te]).mean()); P.append(pred); Y.append(yc[te]); S.append(sims)
    rows = bucket_table(np.concatenate(S), np.concatenate(P), np.concatenate(Y))
    log_cv("c2_" + cfg["name"], accs, {"buckets": fmt_buckets(rows), "sec": round(time.time() - t0, 1), "n_feat": nfeat, "cfg": cfg, "cv": "6fold_seed1"})
