"""Track "german": confirmation on validation, guard sensitivity, bucket tables, summary file."""
from __future__ import annotations

import json
import time

import numpy as np

from common import RESULTS
from final.features import DIAG as GDIR, German, e5_embed, guard_mask
from final.router import fit_predict_proba
from final.lib import (baseline_vec, bucket_table, fmt_table, load_core_val, max_sim, scores, topk_sims)

xc, yc, xv, yv = load_core_val()
v0 = baseline_vec().fit(xc)
sim0 = max_sim(v0.transform(xv), v0.transform(xc))       # bucket axis: baseline TF-IDF, validation vs core
out = {"protocol": "train on core (16,147), score on validation (2,850); test set never opened"}
preds = {}

# reference points with the English pool only
A0, B0 = v0.transform(xc), v0.transform(xv)
s, i = topk_sims(B0, A0, 1)
preds["nn_core"] = yc[i[:, 0]]
P_net0 = np.load(RESULTS / "x_german_base_mlp_val_probs.npy")
preds["net_core"] = P_net0.argmax(1)
preds["hybrid_core"] = np.where(s[:, 0] >= 0.3, preds["nn_core"], preds["net_core"])

# simple methods with the German pool under the default guard
g = German()
Ev = e5_embed(xv)
vec = baseline_vec().fit(xc + g.texts)
A, B, G = vec.transform(xc), vec.transform(xv), vec.transform(g.texts)
keep, info = guard_mask(g, G, B, Ev)
out["default_guard"] = info
from scipy.sparse import vstack
pool = vstack([A, G[np.where(keep)[0]]]).tocsr()
ypool = np.concatenate([yc, g.labels[keep]])
s, i = topk_sims(B, pool, 1)
preds["nn_core+german"] = ypool[i[:, 0]]
P_net = np.load(GDIR / "feats_val_mlp.npy")
preds["net_core+german"] = P_net.argmax(1)
preds["hybrid_core+german"] = np.where(s[:, 0] >= 0.3, preds["nn_core+german"], preds["net_core+german"])

# near-copies of training tickets (>= 0.85) dropped as well
near_tr = max_sim(G, A)
keep2 = keep & (near_tr < 0.85)
pool2 = vstack([A, G[np.where(keep2)[0]]]).tocsr()
ypool2 = np.concatenate([yc, g.labels[keep2]])
s2, i2 = topk_sims(B, pool2, 1)
out["drop_train_near_copies_0.85"] = {"dropped": int((keep & ~keep2).sum()),
                                      "nn_accuracy": scores(yv, ypool2[i2[:, 0]])[0]}

# the stacker: default guard, English-only ablation, guard sensitivity
runs = [("stacker_default_guard", dict()),
        ("stacker_english_only", dict(use_german=False)),
        ("stacker_no_guard", dict(tfidf_guard=None, e5_guard=None)),
        ("stacker_tfidf0.7", dict(tfidf_guard=0.7, e5_guard=None)),
        ("stacker_tfidf0.6", dict(tfidf_guard=0.6, e5_guard=None)),
        ("stacker_tfidf0.5", dict(tfidf_guard=0.5, e5_guard=None)),
        ("stacker_tfidf0.5_e5_0.98", dict(tfidf_guard=0.5, e5_guard=0.98)),
        ("stacker_tfidf0.4_e5_0.97", dict(tfidf_guard=0.4, e5_guard=0.97))]
sens = []
for name, kw in runs:
    t0 = time.time()
    P, inf = fit_predict_proba(xc, yc, xv, verbose=False, return_info=True, **kw)
    preds[name] = P.argmax(1)
    acc, f1 = scores(yv, preds[name])
    row = {"config": name, **{k: v for k, v in kw.items()}, "german_dropped": inf["guard_eval"].get("dropped_total", 0)
           if kw.get("use_german", True) else None, "accuracy": acc, "macro_f1": f1, "seconds": round(time.time() - t0)}
    sens.append(row)
    print(row, flush=True)
    if name == "stacker_default_guard":
        np.save(RESULTS / "x_german_val_probs.npy", P)
        conf = P.max(1)
        out["confidence_table"] = [{"min_confidence": t, "coverage": round(float((conf >= t).mean()), 4),
                                    "accuracy_on_routed": round(float((preds[name][conf >= t] == yv[conf >= t]).mean()), 4)}
                                   for t in (0.0, 0.5, 0.7, 0.9, 0.95)]
out["stacker_runs"] = sens
out["headline"] = {k: dict(zip(("accuracy", "macro_f1"), scores(yv, p))) for k, p in preds.items()}
cols = ["nn_core", "net_core", "hybrid_core", "nn_core+german", "net_core+german", "hybrid_core+german",
        "stacker_english_only", "stacker_default_guard"]
out["bucket_table"] = bucket_table(sim0, yv, {k: preds[k] for k in cols})
print(json.dumps(out["headline"], indent=1))
print(fmt_table([{k[:10]: v for k, v in r.items()} for r in out["bucket_table"]]))
(RESULTS / "x_german_eval.json").write_text(json.dumps(out, indent=1))
