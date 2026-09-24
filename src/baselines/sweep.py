"""Hyperparameter sweep, and the timing that justifies running it in the cloud.

Every configuration is independent of every other one. That is the property
that makes this the textbook case for renting machines instead of owning one.
Sixteen configurations on one laptop run one after another. Sixteen
configurations on sixteen droplets all finish in the time of the slowest one.

This script runs the sweep serially and records how long each configuration
took, so the report can state the real numbers rather than an estimate.

One caveat. This sweep adds the parquet's ticket "type" column as a one-hot
feature next to the words, which no other tier has, so its accuracies are a
little higher than results/1b_mlp_256.json and are not comparable to it. The
point of the script is the timing. What the sweep did show is that width and
depth do not matter: every setting landed within a point of the 256 network.
"""
from __future__ import annotations

import itertools
import json
import time

import numpy as np
import pyarrow.parquet as pq
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import OneHotEncoder

from common import RESULTS, clean, load_split

HIDDEN = [(256,), (512,), (1024,), (512, 256)]
ALPHA = [1e-4, 1e-2]


def main() -> None:
    x_tr, y_tr = load_split("train")
    x_te, y_te = load_split("test")

    t = pq.read_table("data_tickets.parquet").to_pydict()
    lut = {}
    for i in range(len(t["queue"])):
        if t["language"][i] == "en":
            lut[clean(t["subject"][i], t["body"][i])] = t["type"][i]
    ty_tr = np.array([lut.get(x, "") for x in x_tr]).reshape(-1, 1)
    ty_te = np.array([lut.get(x, "") for x in x_te]).reshape(-1, 1)

    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=200_000, sublinear_tf=True)
    A_tr, A_te = vec.fit_transform(x_tr), vec.transform(x_te)
    enc = OneHotEncoder(handle_unknown="ignore")
    X_tr = hstack([A_tr, enc.fit_transform(ty_tr)]).tocsr()
    X_te = hstack([A_te, enc.transform(ty_te)]).tocsr()

    runs = []
    wall = time.time()
    for hidden, alpha in itertools.product(HIDDEN, ALPHA):
        t0 = time.time()
        m = MLPClassifier(hidden_layer_sizes=hidden, alpha=alpha, max_iter=60,
                          early_stopping=True, n_iter_no_change=5, random_state=42)
        m.fit(X_tr, y_tr)
        pred = m.predict(X_te)
        secs = time.time() - t0
        runs.append({
            "hidden": list(hidden), "alpha": alpha,
            "accuracy": round(float(accuracy_score(y_te, pred)), 4),
            "macro_f1": round(float(f1_score(y_te, pred, average="macro")), 4),
            "seconds": round(secs, 1),
        })
        print(f"  hidden={str(hidden):12s} alpha={alpha:<7} "
              f"acc {runs[-1]['accuracy']:.4f}  {secs/60:.1f} min", flush=True)
    total = time.time() - wall

    best = max(runs, key=lambda r: r["accuracy"])
    slowest = max(r["seconds"] for r in runs)
    payload = {
        "configurations": len(runs),
        "serial_minutes_one_machine": round(total / 60, 1),
        "slowest_single_config_minutes": round(slowest / 60, 1),
        "parallel_minutes_one_droplet_each": round(slowest / 60, 1),
        "speedup": round(total / slowest, 1),
        "best": best,
        "runs": runs,
    }
    (RESULTS / "sweep.json").write_text(json.dumps(payload, indent=2))
    print(json.dumps({k: v for k, v in payload.items() if k != "runs"}, indent=2))


if __name__ == "__main__":
    main()
