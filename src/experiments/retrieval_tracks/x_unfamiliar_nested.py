"""Model selection for the gate, entirely inside core: outer folds act as pseudo-validation, the final module's
fit_predict_proba is called on the remaining core tickets. Validation is never read here."""
import sys, json, time, numpy as np
from sklearn.metrics import f1_score
import importlib, os
F = importlib.import_module(os.environ.get("GATE_MODULE", "experiments.retrieval_tracks.x_unfamiliar_final"))
from experiments.retrieval_tracks.x_unfamiliar_lib import *
from experiments.retrieval_tracks.x_unfamiliar_cv import folds

xc, yc, _, _ = load_core_val()
outer = [int(a) for a in sys.argv[1].split(",")]; variant = sys.argv[2] if len(sys.argv) > 2 else "gbm"
res = []
for f in outer:
    tr, te = folds(yc)[f]; t0 = time.time()
    P, info = F.fit_predict_proba([xc[i] for i in tr], yc[tr], [xc[i] for i in te], return_info=True)
    sim = info["tfidf_sim"]; unf = sim < 0.4; y = yc[te]
    import experiments.retrieval_tracks.x_unfamiliar_final as F0; simple = F0.simple_gate_proba(info).argmax(1); nn = info["tfidf_nn_label"]; spec = info["specialist"].argmax(1)
    r = {"fold": f, "variant": variant, "gate_all": float((P.argmax(1) == y).mean()), "gate_unf": float((P.argmax(1)[unf] == y[unf]).mean()),
         "gate_f1": float(f1_score(y, P.argmax(1), average="macro")),
         "simple_all": float((simple == y).mean()), "simple_unf": float((simple[unf] == y[unf]).mean()),
         "nn_all": float((nn == y).mean()), "spec_unf": float((spec[unf] == y[unf]).mean()), "min": (time.time() - t0) / 60,
         "gate_buckets": bucket_table(sim, P.argmax(1) == y), "simple_buckets": bucket_table(sim, simple == y)}
    res.append(r); print(json.dumps({k: v for k, v in r.items() if "buckets" not in k}), flush=True)
    print(fmt_table(r["gate_buckets"]), flush=True)
json.dump(res, open(f"results/x_unfamiliar_nested_{variant}_{sys.argv[1].replace(',', '')}.json", "w"), indent=1)
