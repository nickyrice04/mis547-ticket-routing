"""Pool-size curve of the strict final method: random subsets of core -> validation (learning-curve points, not tuning)."""
import json, time
import numpy as np
import experiments.retrieval_tracks.x_lexical_final as F
from experiments.retrieval_tracks.x_lexical_data import load_core_val
xc, yc, xv, yv = load_core_val(); xa = np.array(xc, dtype=object)
rng = np.random.default_rng(0); perm = rng.permutation(len(xc))
res = {}
for frac in (0.5, 0.75):
    sub = np.sort(perm[:int(frac * len(xc))]); t0 = time.time()
    P = F.fit_predict_proba(list(xa[sub]), yc[sub], xv, extra="none")
    res[str(frac)] = dict(pool=int(len(sub)), acc=float((P.argmax(1) == yv).mean()), minutes=(time.time() - t0) / 60)
    print(frac, res[str(frac)], flush=True)
res["1.0"] = dict(pool=len(xc), acc=0.7804, note="from results/x_lexical_final_main.log")
json.dump(res, open("results/x_lexical_curve.json", "w"), indent=1)
