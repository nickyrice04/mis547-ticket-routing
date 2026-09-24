"""Does the family encoder help the gate? Decided inside core only: the encoder never saw core fold 0, so fold 0 is split
in two halves; one half trains the gate (as "unseen" rows), the other half is scored, then swapped. Validation is not read."""
import json, hashlib, numpy as np
from experiments.retrieval_tracks.x_unfamiliar_lib import *
from experiments.retrieval_tracks.x_unfamiliar_cv import folds
import experiments.retrieval_tracks.x_unfamiliar_gate3 as G

xc, yc, _, _ = load_core_val(); tr, te = folds(yc)[0]
half = np.array([int(hashlib.md5(xc[i].encode()).hexdigest(), 16) % 2 for i in te])
variants = {"A_all_rows_frozen_views": dict(views=("mpnet", "bge_base"), restrict_to_unseen=False),
            "A_unseen_rows_frozen_views": dict(views=("mpnet", "bge_base"), restrict_to_unseen=True),
            "B_unseen_rows_with_famenc": dict(views=("mpnet", "bge_base", "famenc"), restrict_to_unseen=True)}
P = {k: np.zeros((len(te), 10)) for k in variants}; sim = np.zeros(len(te))
for h in (0, 1):
    ev = te[half == h]; pool = np.concatenate([tr, te[half != h]])
    for k, kw in variants.items():
        p, info = G.fit_predict_proba([xc[i] for i in pool], yc[pool], [xc[i] for i in ev], return_info=True, **kw)
        P[k][half == h] = p; sim[half == h] = info["tfidf_sim"]; print(h, k, float((p.argmax(1) == yc[ev]).mean()), info["n_gate_rows"], flush=True)
P["A_all+B_average"] = 0.5 * (P["A_all_rows_frozen_views"] + P["B_unseen_rows_with_famenc"])
y = yc[te]; out = {}
for k, p in P.items():
    c = p.argmax(1) == y; out[k] = {"acc": float(c.mean()), "unfamiliar": float(c[sim < 0.4].mean()), "buckets": bucket_table(sim, c)}
    print(k, out[k]["acc"], out[k]["unfamiliar"]); print(fmt_table(out[k]["buckets"]))
json.dump(out, open("results/x_unfamiliar_famenc_eval.json", "w"), indent=1)
