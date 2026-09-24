"""In-core test of the number of cross-fitting folds: pseudo-train (80% of core) -> pseudo-validation (20% of core)."""
import json, time
import numpy as np
from sklearn.model_selection import StratifiedKFold
import experiments.retrieval_tracks.x_lexical_final as F
from experiments.retrieval_tracks.x_lexical_data import load_core_val
xc, yc, _, _ = load_core_val(); xa = np.array(xc, dtype=object)
tr, te = next(iter(StratifiedKFold(5, shuffle=True, random_state=123).split(xa, yc)))
res = {}
for k in (5, 10):
    t0 = time.time()
    P = F.fit_predict_proba(list(xa[tr]), yc[tr], list(xa[te]), extra="none", n_folds=k)
    res[k] = dict(acc=float((P.argmax(1) == yc[te]).mean()), logloss=float(-np.mean(np.log(np.clip(P[np.arange(len(te)), yc[te]], 1e-9, 1)))), minutes=(time.time() - t0) / 60)
    np.save(f"data/x_lexical/pseudo_probs_k{k}.npy", P)
    print(k, res[k], flush=True)
json.dump(res, open("results/x_lexical_cv_kfold.json", "w"), indent=1)
