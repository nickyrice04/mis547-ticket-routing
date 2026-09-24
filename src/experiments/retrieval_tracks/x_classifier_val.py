"""Score named configurations core -> validation. Every call is logged (results/x_classifier_val_runs.jsonl)."""
import json, sys, time, os
import numpy as np
from experiments.retrieval_tracks.x_classifier_lib import load_core_val, nearest_sim, log_val, ROOT
from experiments.retrieval_tracks.x_classifier_final import fit_predict_proba

xc, yc, xv, yv = load_core_val()
cache = ROOT / "data/x_classifier/val_sims.npz"
if cache.exists():
    z = np.load(cache); sims, nn_idx = z["sims"], z["idx"]
else:
    sims, nn_idx = nearest_sim(xc, xv); np.savez(cache, sims=sims, idx=nn_idx)

for cfg in json.loads(sys.argv[1]):
    name = cfg.pop("name")
    t0 = time.time()
    if name == "baseline_1nn":
        pred = yc[nn_idx]
    else:
        P = fit_predict_proba(xc, yc, xv, verbose=True, **cfg)
        np.save(ROOT / f"data/x_classifier/valprobs_{name}.npy", P)
        pred = P.argmax(1)
    log_val(name, pred, yv, sims, {"cfg": cfg, "minutes": round((time.time() - t0) / 60, 2)})
