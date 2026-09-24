"""Cache base-model outputs: K-fold cross-fitted inside core, plus core -> validation."""
import sys, time
import numpy as np
from sklearn.model_selection import StratifiedKFold
from experiments.retrieval_tracks.x_lexical_data import load_core_val
from experiments.retrieval_tracks.x_lexical_base import base_outputs

K = int(sys.argv[1]) if len(sys.argv) > 1 else 5
xc, yc, xv, yv = load_core_val()
xa = np.array(xc, dtype=object)
t0 = time.time()
for f, (tr, te) in enumerate(StratifiedKFold(K, shuffle=True, random_state=0).split(xa, yc)):
    o = base_outputs(list(xa[tr]), yc[tr], list(xa[te]), verbose=True)
    np.savez_compressed(f"data/x_lexical/base_k{K}_f{f}.npz", tr=tr, te=te, **o)
    print("fold", f, round(time.time() - t0), flush=True)
o = base_outputs(xc, yc, xv, verbose=True)
np.savez_compressed(f"data/x_lexical/base_val.npz", **o)
print("val done", round(time.time() - t0), flush=True)
