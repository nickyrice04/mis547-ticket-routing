"""Independent verification of track "semantic" (reviewer script, does not modify track files).

    PYTHONPATH=src .venv/bin/python src/experiments/retrieval_tracks/x_verify_semantic_repro.py repro     # core -> validation, fresh embedding cache
    PYTHONPATH=src .venv/bin/python src/experiments/retrieval_tracks/x_verify_semantic_repro.py shuffle   # same with permuted training labels

The embedding cache is redirected to a reviewer-owned directory so that the track's cached .npy files are NOT trusted:
every embedding is recomputed from the raw text with the frozen Hugging Face model. Never touches the test split.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

import final.router_english_only as M
from common import load_split

ROOT = Path(__file__).resolve().parents[3]   # retrieval_tracks -> experiments -> src -> repository root
MODE = sys.argv[1]
CACHE = Path(sys.argv[2])
TRACK_CACHE = M.EMB_CACHE
M.EMB_CACHE = CACHE                       # reviewer-owned cache, starts empty

x, lab = load_split("train")
sp = json.loads((ROOT / "data/synthetic/split.json").read_text())
assert not (set(sp["core"]) & set(sp["validation"]))
ctx = [x[i] for i in sp["core"]]; cy = np.array([lab[i] for i in sp["core"]])
vtx = [x[i] for i in sp["validation"]]; vy = np.array([lab[i] for i in sp["validation"]])
print("core", len(ctx), "val", len(vtx), "exact text overlap core/val:", len(set(ctx) & set(vtx)), flush=True)

from sklearn.metrics import f1_score

y_fit = cy
if MODE == "shuffle":
    y_fit = np.random.default_rng(20260921).permutation(cy)
    print("label agreement after permutation:", float((y_fit == cy).mean()), flush=True)

t0 = time.time()
P = M.fit_predict_proba(ctx, y_fit, vtx)
minutes = (time.time() - t0) / 60
pred = P.argmax(1)
acc = float((pred == vy).mean()); f1 = float(f1_score(vy, pred, average="macro"))
maj = float(np.bincount(vy).max() / len(vy))
print(f"[{MODE}] core -> validation accuracy {acc:.4f} macro-F1 {f1:.4f} majority share {maj:.4f} ({minutes:.1f} min)", flush=True)
out = {"mode": MODE, "val_accuracy": acc, "val_macro_f1": f1, "majority_share": maj, "minutes": round(minutes, 1),
       "pred_hist": np.bincount(pred, minlength=10).tolist()}

if MODE == "repro":
    # compare to the track's saved validation probabilities and to the track's cached embeddings
    Pt = np.load(ROOT / "results" / "x_semantic_final_core2val_valprobs.npy")
    out["track_probs_acc"] = float((Pt.argmax(1) == vy).mean())
    out["argmax_agreement_with_track_probs"] = float((Pt.argmax(1) == pred).mean())
    out["max_abs_prob_diff"] = float(np.abs(Pt - P).max())
    diffs = {}
    for f in sorted(CACHE.glob("*.npy")):
        g = TRACK_CACHE / f.name
        if g.exists():
            a, b = np.load(f), np.load(g)
            diffs[f.name] = {"shape": list(a.shape), "max_abs_diff": float(np.abs(a - b).max()),
                             "min_row_cos": float((a * b).sum(1).min())}
        else:
            diffs[f.name] = "no file with this hash in track cache"
    out["embedding_cache_vs_track"] = diffs
    np.save(ROOT / "results" / "x_verify_semantic_repro_valprobs.npy", P)
(ROOT / "results" / f"x_verify_semantic_{MODE}.json").write_text(json.dumps(out, indent=1))
print(json.dumps(out, indent=1), flush=True)
