"""Run the deliverable's fit_predict_proba inside core (6-fold-style CV, chosen folds) to verify the code path."""
import sys, json, time, warnings
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedKFold
from experiments.retrieval_tracks.x_classifier_lib import BASE_VEC, bucket_table, fmt_buckets, load_core_val, log_cv
from experiments.retrieval_tracks.x_classifier_final import fit_predict_proba
warnings.filterwarnings("ignore")
cfg = json.loads(sys.argv[1]); folds = [int(f) for f in sys.argv[2].split(",")]
name = cfg.pop("name")
xc, yc, _, _ = load_core_val(); xc = np.array(xc, dtype=object)
splits = list(StratifiedKFold(n_splits=6, shuffle=True, random_state=1).split(xc, yc))
out = []
for fi in folds:
    tr, te = splits[fi]
    t0 = time.time()
    P = fit_predict_proba(list(xc[tr]), yc[tr], list(xc[te]), verbose=True, **cfg)
    bv = TfidfVectorizer(**BASE_VEC); a0 = bv.fit_transform(xc[tr]); b0 = bv.transform(xc[te])
    sims = (b0 @ a0.T).max(axis=1).toarray().ravel()
    pred = P.argmax(1); out.append((pred, yc[te], sims))
    ll = -np.mean(np.log(np.clip(P[np.arange(len(te)), yc[te]], 1e-9, None)))
    print(f"fold {fi}: acc={(pred == yc[te]).mean():.4f} logloss={ll:.4f} {time.time()-t0:.0f}s", flush=True)
P_ = np.concatenate([o[0] for o in out]); Y_ = np.concatenate([o[1] for o in out]); S_ = np.concatenate([o[2] for o in out])
log_cv("final_" + name, [(o[0] == o[1]).mean() for o in out], {"buckets": fmt_buckets(bucket_table(S_, P_, Y_)), "cfg": cfg, "cv": "6fold_seed1", "folds": folds})
