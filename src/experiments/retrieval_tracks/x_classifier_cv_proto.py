"""6-fold-style CV (first 2 folds) for the prototype MLP at initialisation (k-means prototypes + ridge readout)."""
import sys, json, time, warnings
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedKFold
from experiments.retrieval_tracks.x_classifier_lib import BASE_VEC, bucket_table, fmt_buckets, load_core_val, log_cv
from experiments.retrieval_tracks.x_classifier_proto import classwise_kmeans, hidden, ridge_readout
warnings.filterwarnings("ignore")
configs = json.loads(sys.argv[1]); nfolds = int(sys.argv[2]) if len(sys.argv) > 2 else 2
xc, yc, _, _ = load_core_val(); xc = np.array(xc, dtype=object)
splits = list(StratifiedKFold(n_splits=6, shuffle=True, random_state=1).split(xc, yc))[:nfolds]
fc = {}
for cfg in configs:
    t0 = time.time(); out = []
    for fi, (tr, te) in enumerate(splits):
        if fi not in fc:
            bv = TfidfVectorizer(**BASE_VEC); a0 = bv.fit_transform(xc[tr]); b0 = bv.transform(xc[te])
            uv = TfidfVectorizer(ngram_range=(1, 1), min_df=2, sublinear_tf=True, stop_words="english")
            fc[fi] = ((b0 @ a0.T).max(axis=1).toarray().ravel(), uv.fit_transform(xc[tr]), uv.transform(xc[te]))
        sims, a, b = fc[fi]
        K = cfg["K"]
        if K >= a.shape[0]:
            P = a.toarray().astype(np.float32)
        else:
            P, _ = classwise_kmeans(a, yc[tr], K, n_iter=cfg.get("n_iter", 8), seed=0)
        Htr = hidden(a, P, cfg["power"]); Hte = hidden(b, P, cfg["power"])
        B = ridge_readout(Htr, yc[tr], cfg.get("lam", 0.01))
        pred = (Hte.astype(np.float64) @ B).argmax(1)
        out.append((pred, yc[te], sims)); print(f"  fold {fi} K={P.shape[0]} acc={(pred == yc[te]).mean():.4f} {time.time()-t0:.0f}s", flush=True)
    P_ = np.concatenate([o[0] for o in out]); Y_ = np.concatenate([o[1] for o in out]); S_ = np.concatenate([o[2] for o in out])
    log_cv("proto_" + cfg["name"], [(o[0] == o[1]).mean() for o in out], {"buckets": fmt_buckets(bucket_table(S_, P_, Y_)), "sec": round(time.time() - t0, 1), "cfg": cfg, "cv": "6fold_seed1"})
