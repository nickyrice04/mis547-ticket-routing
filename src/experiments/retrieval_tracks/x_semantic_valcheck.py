"""Confirm ONE selected stacker config on validation. Usage: x_semantic_valcheck.py <tag> [colfilter]"""
import sys, json
import numpy as np
from sklearn.metrics import f1_score
from experiments.retrieval_tracks.x_semantic_common import *
from experiments.retrieval_tracks.x_semantic_stack import fit_meta, meta_proba, NC
tag = sys.argv[1]
d = np.load(CACHE / f"feats_{tag}_loo.npz", allow_pickle=True); v = np.load(CACHE / f"feats_{tag}_val.npz", allow_pickle=True)
X, y, names = d["X"], d["y"], list(d["names"])
t = (np.tile(np.arange(NC), len(y)) == np.repeat(y, NC)).astype(int)
Ps = []
for seed in range(3):
    m = fit_meta(X, t, names, seed=seed)
    Ps.append(meta_proba(m, v["X"]))
P = np.mean(Ps, 0); vy = v["y"]; ok = P.argmax(1) == vy
print(f"[{tag}] validation acc {ok.mean():.4f} macroF1 {f1_score(vy, P.argmax(1), average='macro'):.4f}  (lex 1-NN {(v['top1']==vy).mean():.4f}, fused 1-NN {(v['top1_fused']==vy).mean():.4f})")
rows = bucket_table(v["s1_lex"], ok, {"lex1nn": v["top1"] == vy})
print(fmt_table(rows))
np.save(CACHE / f"val_P_{tag}.npy", P)
json.dump({"tag": tag, "val_acc": float(ok.mean()), "val_macro_f1": float(f1_score(vy, P.argmax(1), average='macro')), "buckets": rows},
          open(ROOT / f"results/x_semantic_val_{tag}.json", "w"), indent=1)
