"""CV inside core: compare similarity views and kNN voting settings. No validation data touched."""
import json, sys, time
import numpy as np
from sklearn.model_selection import StratifiedKFold
from experiments.retrieval_tracks.x_lexical_data import load_core_val
from experiments.retrieval_tracks.x_lexical_views import VIEWS, topk, vote

xc, yc, _, _ = load_core_val()
xc = np.array(xc, dtype=object)
names = sys.argv[1].split(",") if len(sys.argv) > 1 else list(VIEWS)
skf = StratifiedKFold(5, shuffle=True, random_state=0)
res = {}
t0 = time.time()
for f, (tr, te) in enumerate(skf.split(xc, yc)):
    for name in names:
        v = VIEWS[name]().fit(list(xc[tr]))
        S = v.sims(list(xc[te]))
        idx, val = topk(S, 50)
        np.savez_compressed(f"data/x_lexical/cv_top50_{name}_f{f}.npz", idx=idx.astype(np.int32), val=val.astype(np.float32))
        for k in (1, 3, 5, 10, 20):
            for p in (1, 2, 4, 8, 16):
                if k == 1 and p > 1: continue
                pred = vote(idx, val, yc[tr], 10, k, p).argmax(1)
                res.setdefault((name, k, p), []).append((pred == yc[te]).mean())
        print(f, name, round(time.time() - t0), flush=True)
out = {f"{n}|k{k}|p{p}": float(np.mean(a)) for (n, k, p), a in res.items()}
for name in names:
    best = sorted(((v, k) for k, v in out.items() if k.startswith(name + "|")), reverse=True)[:4]
    print(name, "1nn", round(out[f"{name}|k1|p1"], 4), "best", [(k, round(v, 4)) for v, k in best])
json.dump(out, open(f"results/x_lexical_cv1_{'_'.join(names)[:40]}.json", "w"), indent=1)
