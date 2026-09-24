"""Reviewer check for track "german": reproduction and label-shuffle controls.

    PYTHONPATH=src python src/experiments/retrieval_tracks/x_verify_german_repro.py repro
    PYTHONPATH=src python src/experiments/retrieval_tracks/x_verify_german_repro.py shuffle_train      # core labels permuted, German labels intact
    PYTHONPATH=src python src/experiments/retrieval_tracks/x_verify_german_repro.py shuffle_all        # core AND German labels permuted
    PYTHONPATH=src python src/experiments/retrieval_tracks/x_verify_german_repro.py shuffle_german     # only German labels permuted

Only load_split("train") is touched, through x_german_lib.load_core_val. The test split is never opened.
"""
from __future__ import annotations

import json
import os
import sys
import time

assert os.environ.get("X_GERMAN_E5_CACHE") != "1", "the e5 cache must be off for a faithful run"

import numpy as np
from sklearn.metrics import accuracy_score, f1_score

import final.features as features
import final.router as router
from common import RESULTS
from final.lib import load_core_val

mode = sys.argv[1]
xc, yc, xv, yv = load_core_val()
print(f"mode={mode} core={len(xc)} val={len(xv)} majority share in val={np.bincount(yv).max()/len(yv):.4f}", flush=True)
rng = np.random.default_rng(12345)
y_fit = np.array(yc)

if mode in ("shuffle_train", "shuffle_all"):
    y_fit = rng.permutation(y_fit)

if mode in ("shuffle_all", "shuffle_german"):
    _Orig = x_german_feats.German

    class ShuffledGerman(_Orig):
        def __init__(self):
            super().__init__()
            self.labels = np.random.default_rng(999).permutation(self.labels)

    x_german_feats.German = ShuffledGerman
    x_german_final.German = ShuffledGerman

t0 = time.time()
probs, info = x_german_final.fit_predict_proba(xc, y_fit, xv, return_info=True)
pred = probs.argmax(1)
acc = float(accuracy_score(yv, pred))
f1 = float(f1_score(yv, pred, average="macro"))
out = {"mode": mode, "accuracy": round(acc, 4), "macro_f1": round(f1, 4), "guard_eval": info["guard_eval"],
       "minutes": round((time.time() - t0) / 60, 2),
       "pred_class_share_max": round(float(np.bincount(pred, minlength=10).max() / len(pred)), 4)}
print(json.dumps(out), flush=True)
np.save(RESULTS / f"x_verify_german_{mode}_valprobs.npy", probs)
(RESULTS / f"x_verify_german_{mode}.json").write_text(json.dumps(out, indent=1))
