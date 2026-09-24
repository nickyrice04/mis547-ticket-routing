"""In-core evaluation of the final pipeline (no validation data): cross-fit pair rows on core, then meta-level K-fold."""
import json, sys, time
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss
import experiments.retrieval_tracks.x_lexical_final as F
from experiments.retrieval_tracks.x_lexical_data import load_core_val

mode = sys.argv[1] if len(sys.argv) > 1 else "none"
use_mlp = len(sys.argv) > 2 and sys.argv[2] == "mlp"
xc, yc, _, _ = load_core_val()
ex_t, ex_y, ex_s = F.load_extra(mode)
keep = np.array([t not in set(xc) for t in ex_t], dtype=bool) if len(ex_t) else np.zeros(0, dtype=bool)
ex_t, ex_y, ex_s = [t for t, k in zip(ex_t, keep) if k], ex_y[keep], ex_s[keep]
print("mode", mode, "extra rows", len(ex_t), flush=True)
t0 = time.time()
X, names, fold_id = F.crossfit(xc, yc, ex_t, ex_y, ex_s, use_mlp=use_mlp, verbose=True)
tag = mode.replace("+", "_") + ("_mlp" if use_mlp else "")
np.savez_compressed(f"data/x_lexical/pairs_core_{tag}.npz", X=X, names=np.array(names), fold_id=fold_id)
y_pair = (np.tile(np.arange(10), len(yc)) == np.repeat(yc, 10)).astype(int); fp = np.repeat(fold_id, 10)
oof = np.zeros((len(yc), 10))
for f in range(F.N_FOLDS):
    oof[fold_id == f] = F._predict_meta(F._fit_meta(X[fp != f], y_pair[fp != f], names, seeds=(0,)), X[fp == f])
# strict-pool baseline similarity buckets (same folds as the cached w12 neighbours)
s1 = np.zeros(len(yc))
xa = np.array(xc, dtype=object)
for f, (tr, te) in enumerate(StratifiedKFold(5, shuffle=True, random_state=0).split(xa, yc)):
    s1[te] = np.load(f"data/x_lexical/cv_top50_w12_f{f}.npz")["val"][:, 0]
ok = oof.argmax(1) == yc
res = dict(mode=mode, n_extra=len(ex_t), acc=float(ok.mean()), low=float(ok[s1 < 0.4].mean()), high=float(ok[s1 >= 0.4].mean()),
           logloss=float(log_loss(yc, oof, labels=range(10))), minutes=(time.time() - t0) / 60)
bins = [0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.01]
res["buckets"] = [dict(lo=lo, hi=hi, share=float(((s1 >= lo) & (s1 < hi)).mean()), acc=float(ok[(s1 >= lo) & (s1 < hi)].mean())) for lo, hi in zip(bins[:-1], bins[1:])]
print(json.dumps(res, indent=1))
json.dump(res, open(f"results/x_lexical_cvfinal_{tag}.json", "w"), indent=1)
