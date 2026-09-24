"""Two principled variants of the final system, scored inside core only (outer folds 0 and 1)."""
import sys, json, numpy as np
import experiments.retrieval_tracks.x_unfamiliar_final as F
from experiments.retrieval_tracks.x_unfamiliar_lib import *
from experiments.retrieval_tracks.x_unfamiliar_cv import folds
xc, yc, _, _ = load_core_val(); out = {}
for f in (0, 1):
    tr, te = folds(yc)[f]; a = [xc[i] for i in tr]; b = [xc[i] for i in te]; y = yc[te]
    p10 = F.fit_predict_proba(a, yc[tr], b, n_inner=10)
    ps = [F.fit_predict_proba(a, yc[tr], b, seed=s) for s in (0, 1, 2)]
    out[f] = {"n_inner10": float((p10.argmax(1) == y).mean()), "seed0": float((ps[0].argmax(1) == y).mean()),
              "seed1": float((ps[1].argmax(1) == y).mean()), "seed2": float((ps[2].argmax(1) == y).mean()),
              "avg3seeds": float((np.mean(ps, 0).argmax(1) == y).mean())}
    print(f, out[f], flush=True)
json.dump(out, open("results/x_unfamiliar_nested_variants.json", "w"), indent=1)
