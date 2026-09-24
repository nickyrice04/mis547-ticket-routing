"""Lexical baselines: reproduce 1-NN 74.5%, MLP, hybrid; cache sims."""
import json, time
import numpy as np
from experiments.retrieval_tracks.x_semantic_common import *

ctx, cy, vtx, vy = load_core_val()
print(len(ctx), len(vtx))
lens = np.array([len(t) for t in ctx]); print("char len pct", np.percentile(lens, [10, 50, 90, 99]))
vec = baseline_tfidf()
Xc = vec.fit_transform(ctx); Xv = vec.transform(vtx)
S = (Xv @ Xc.T).toarray().astype(np.float32)
np.save(CACHE / "S_lex_val_core.npy", S)
nn = S.argmax(1); sim = S.max(1)
np.save(CACHE / "sim_lex_val.npy", sim)
pred = cy[nn]
print("1-NN acc", (pred == vy).mean())
print(fmt_table(bucket_table(sim, pred == vy)))
