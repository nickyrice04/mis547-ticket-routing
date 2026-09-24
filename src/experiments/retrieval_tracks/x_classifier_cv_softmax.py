"""Fold-0 probe: softmax (CE) linear model on conjunction features vs ridge/SVC."""
import sys, json, time, warnings
import numpy as np, scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedKFold
from experiments.retrieval_tracks.x_classifier_conj import Conjunctions
from experiments.retrieval_tracks.x_classifier_lib import BASE_VEC, bucket_table, fmt_buckets, load_core_val, log_cv
from experiments.retrieval_tracks.x_classifier_sparselin import train_sparse_softmax, predict_proba_sparse
warnings.filterwarnings("ignore")
configs = json.loads(sys.argv[1])
xc, yc, _, _ = load_core_val(); xc = np.array(xc, dtype=object)
tr, te = next(iter(StratifiedKFold(n_splits=3, shuffle=True, random_state=0).split(xc, yc)))
ytr, yte = yc[tr], yc[te]
bv = TfidfVectorizer(**BASE_VEC); a0 = bv.fit_transform(xc[tr]); b0 = bv.transform(xc[te])
base_sims = (b0 @ a0.T).max(axis=1).toarray().ravel()
uv = TfidfVectorizer(ngram_range=(1, 1), min_df=2, sublinear_tf=True, stop_words="english")
au = uv.fit_transform(xc[tr]); bu = uv.transform(xc[te])
blocks = {1: (au, bu)}
for cfg in configs:
    t0 = time.time()
    w = cfg["w"]; orders = [o for o, wt in zip((1, 2, 3), w) if wt > 0]
    for o in orders:
        if o not in blocks:
            cj = Conjunctions(order=o, min_df=2, max_terms=40 if o == 3 else None)
            blocks[o] = (cj.fit_transform(au), cj.transform(bu))
    A = sp.hstack([w[o - 1] * blocks[o][0] for o in orders]).tocsr(); B = sp.hstack([w[o - 1] * blocks[o][1] for o in orders]).tocsr()
    model = train_sparse_softmax(A, ytr, verbose=True, X_eval=B, y_eval=yte, **cfg["fit"])
    pred = predict_proba_sparse(model, B, scale=cfg["fit"].get("scale", 1.0)).argmax(1)
    log_cv("softmax_" + cfg["name"], [(pred == yte).mean()], {"buckets": fmt_buckets(bucket_table(base_sims, pred, yte)), "sec": round(time.time() - t0, 1), "cfg": cfg, "note": "fold0 only"})
