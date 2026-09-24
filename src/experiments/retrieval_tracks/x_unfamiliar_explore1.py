"""Step 1: reproduce the known numbers and look at label mix by similarity bucket.
Validation labels are only summarised here (bucket label mix), nothing is fitted on them."""
import json
import numpy as np
from experiments.retrieval_tracks.x_unfamiliar_lib import *

xc, yc, xv, yv = load_core_val()
vec = base_vectorizer()
Xc = vec.fit_transform(xc)
Xv = vec.transform(xv)
sim_v, idx_v = nn_search(Xv, Xc)
sim_c, idx_c = nn_search(Xc, Xc, exclude_self=True)
np.save("data/x_unfamiliar/sim_val.npy", sim_v); np.save("data/x_unfamiliar/nnidx_val.npy", idx_v)
np.save("data/x_unfamiliar/sim_core_loo.npy", sim_c); np.save("data/x_unfamiliar/nnidx_core_loo.npy", idx_c)

print("VAL 1-NN acc", (yc[idx_v] == yv).mean())
print(fmt_table(bucket_table(sim_v, yc[idx_v] == yv)))
print("CORE leave-one-out 1-NN acc", (yc[idx_c] == yc).mean())
print(fmt_table(bucket_table(sim_c, yc[idx_c] == yc)))

prior = np.bincount(yc, minlength=10) / len(yc)
print("\nlabel mix (core overall | core LOO-unfamiliar <0.4 | core <0.3 | val <0.4)")
mu = sim_c < 0.4
mu3 = sim_c < 0.3
mv = sim_v < 0.4
for k, name in enumerate(LABELS):
    print(f"{name:32s} {prior[k]*100:5.1f}  {np.mean(yc[mu]==k)*100:5.1f}  {np.mean(yc[mu3]==k)*100:5.1f}  {np.mean(yv[mv]==k)*100:5.1f}")
print("n core unfamiliar", mu.sum(), "share", mu.mean(), "| n val unfamiliar", mv.sum(), "share", mv.mean())
