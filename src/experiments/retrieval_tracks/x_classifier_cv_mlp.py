"""CV inside core for torch MLP configs. Usage: x_classifier_cv_mlp.py '<json list of configs>' [nfolds]

Each config: {"name":..., "vec": {...tfidf kwargs or "base"}, "mlp": {...train_mlp kwargs}}
"""
from __future__ import annotations

import json
import sys
import time

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedKFold

from experiments.retrieval_tracks.x_classifier_lib import BASE_VEC, bucket_table, fmt_buckets, load_core_val, log_cv
from experiments.retrieval_tracks.x_classifier_torchmlp import predict_proba, train_mlp

configs = json.loads(sys.argv[1])
nfolds = int(sys.argv[2]) if len(sys.argv) > 2 else 1

xc, yc, _, _ = load_core_val()
xc = np.array(xc, dtype=object)
skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=0)
splits = list(skf.split(xc, yc))[:nfolds]

base_cache = {}
for cfg in configs:
    t0 = time.time()
    accs, P, Y, S = [], [], [], []
    for fi, (tr, te) in enumerate(splits):
        if fi not in base_cache:
            bv = TfidfVectorizer(**BASE_VEC)
            a0 = bv.fit_transform(xc[tr]); b0 = bv.transform(xc[te])
            base_cache[fi] = (a0, b0, (b0 @ a0.T).max(axis=1).toarray().ravel())
        a0, b0, sims = base_cache[fi]
        if cfg.get("vec", "base") == "base":
            a, b = a0, b0
        else:
            v = dict(cfg["vec"])
            if "ngram_range" in v:
                v["ngram_range"] = tuple(v["ngram_range"])
            vec = TfidfVectorizer(**v)
            a = vec.fit_transform(xc[tr]); b = vec.transform(xc[te])
        net = train_mlp(a, yc[tr], verbose=True, X_eval=b, y_eval=yc[te], **cfg["mlp"])
        pred = predict_proba(net, b).argmax(1)
        accs.append((pred == yc[te]).mean()); P.append(pred); Y.append(yc[te]); S.append(sims)
        del net
    rows = bucket_table(np.concatenate(S), np.concatenate(P), np.concatenate(Y))
    log_cv("mlp_" + cfg["name"], accs, {"buckets": fmt_buckets(rows), "sec": round(time.time() - t0, 1), "cfg": cfg})
