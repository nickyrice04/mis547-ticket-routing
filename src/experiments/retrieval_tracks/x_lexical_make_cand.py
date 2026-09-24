"""Cache candidate-level features (best pool ticket per class) for each core fold and for validation. Track 'lexical'."""
import pickle, sys, time
import numpy as np
from experiments.retrieval_tracks.x_lexical_data import load_core_val
from experiments.retrieval_tracks.x_lexical_cand import cand_features
K = 5
xc, yc, xv, yv = load_core_val()
mc = pickle.load(open("data/x_lexical/core_meta.pkl", "rb"))
xa = np.array(xc, dtype=object); ma = np.array(mc, dtype=object)
t0 = time.time()
for f in range(K):
    o = np.load(f"data/x_lexical/base_k{K}_f{f}.npz"); tr, te = o["tr"], o["te"]
    c = cand_features(list(xa[tr]), yc[tr], list(xa[te]), o["knn_wc_idx"], o["knn_wc_val"], list(ma[tr]))
    np.savez_compressed(f"data/x_lexical/cand_k{K}_f{f}.npz", **c); print(f, round(time.time() - t0), flush=True)
o = np.load("data/x_lexical/base_val.npz")
c = cand_features(xc, yc, xv, o["knn_wc_idx"], o["knn_wc_val"], mc)
np.savez_compressed("data/x_lexical/cand_val.npz", **c); print("val", round(time.time() - t0), flush=True)
