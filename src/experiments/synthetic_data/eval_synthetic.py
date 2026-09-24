"""Does adding the synthetic tickets actually help, measured on the untouched test set?

Four training sets, each scored on the same held-out test tickets:

  A. core only                the 85% of training left after carving out validation
  B. core + synthetic         core plus the generated tickets
  C. core + validation        core plus the 2,850 real validation tickets
  D. core + validation + synthetic

A against B is the question you asked. C is the fairness check that tells you
whether synthetic tickets are worth as much as real ones. D is what you would
actually ship.

Every result is averaged over three random seeds, because a single training run
of this network moves by a point or more on its own, and a one-point gain is
exactly the size of effect we are trying to detect.

    python src/experiments/synthetic_data/eval_synthetic.py
"""
from __future__ import annotations

import json

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics.pairwise import linear_kernel
from sklearn.neural_network import MLPClassifier

from common import RESULTS, load_meta, load_split

SEEDS = (1, 2, 3)


def load_synthetic(path="data/synthetic/synthetic_tickets.jsonl"):
    """The one call needed to switch the synthetic rows on in any training script."""
    rows = [json.loads(l) for l in open(path)]
    return [r["text"] for r in rows], np.asarray([r["label"] for r in rows])


def fit_score(x, y, x_te, y_te, seed):
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=200_000, sublinear_tf=True)
    X = vec.fit_transform(x)
    m = MLPClassifier(hidden_layer_sizes=(256,), max_iter=60, early_stopping=True,
                      n_iter_no_change=5, random_state=seed).fit(X, y)
    return m.predict(vec.transform(x_te))


def main() -> None:
    labels = load_meta()["labels"]
    x_tr, y_tr = load_split("train")
    x_te, y_te = load_split("test")
    y_tr, y_te = np.asarray(y_tr), np.asarray(y_te)
    split = json.loads(open("data/synthetic/split.json").read())
    core, val = split["core"], split["validation"]
    xs, ys = load_synthetic()

    v = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True).fit(x_tr)
    A, B = v.transform(x_tr), v.transform(x_te)
    sim = np.concatenate([linear_kernel(B[s:s+500], A).max(1) for s in range(0, len(y_te), 500)])
    unfam = sim < 0.5
    weak = [labels.index(q) for q in ("General Inquiry", "Returns and Exchanges",
                                       "Human Resources", "Sales and Pre-Sales")]
    weak_mask = np.isin(y_te, weak)

    x_core, y_core = [x_tr[i] for i in core], y_tr[core]
    x_val, y_val = [x_tr[i] for i in val], y_tr[val]
    setups = {
        "A. core only": (x_core, y_core),
        "B. core + synthetic": (x_core + xs, np.concatenate([y_core, ys])),
        "C. core + real validation": (x_core + x_val, np.concatenate([y_core, y_val])),
        "D. core + validation + synthetic": (x_core + x_val + xs, np.concatenate([y_core, y_val, ys])),
    }

    results = {}
    print(f"synthetic tickets: {len(xs)}    real validation tickets: {len(x_val)}\n")
    print(f"{'training set':34s} {'rows':>6} {'accuracy':>10} {'macro-F1':>9} {'unfamiliar':>11} {'weak queues':>12}")
    for name, (x, y) in setups.items():
        runs = []
        for s in SEEDS:
            p = fit_score(x, y, x_te, y_te, s)
            runs.append((accuracy_score(y_te, p), f1_score(y_te, p, average="macro"),
                         accuracy_score(y_te[unfam], p[unfam]),
                         accuracy_score(y_te[weak_mask], p[weak_mask])))
        r = np.array(runs)
        mean, sd = r.mean(0), r.std(0)
        results[name] = {"rows": len(y), "accuracy": round(float(mean[0]), 4),
                         "accuracy_sd": round(float(sd[0]), 4), "macro_f1": round(float(mean[1]), 4),
                         "unfamiliar": round(float(mean[2]), 4), "weak_queues": round(float(mean[3]), 4)}
        print(f"{name:34s} {len(y):6d} {mean[0]*100:6.2f}±{sd[0]*100:.1f} {mean[1]:9.3f} "
              f"{mean[2]*100:10.2f}% {mean[3]*100:11.2f}%", flush=True)

    (RESULTS / "synthetic_eval.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
