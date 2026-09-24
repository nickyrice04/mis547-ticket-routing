"""Compare synthetic data sources head to head on the untouched test set.

Every source is scored the same way on the same baseline, TF-IDF plus a neural
network with one hidden layer of 256, so the only thing that differs between
rows is where the extra training tickets came from.

Rows:
    core only                      the training set with validation carved out
    core + real validation         the reference, what real extra data is worth
    core + <each synthetic file>   what each generator's tickets are worth

Three numbers make it quantitative rather than impressionistic:

    accuracy, mean and spread over three training seeds, because one run of
        this network moves by half a point on its own
    a 95% confidence interval on the gain over core only, from a paired
        bootstrap over the 4,750 test tickets. If the interval crosses zero,
        the gain is not distinguishable from noise
    style distance, how much each set of tickets reads like the real ones,
        measured as the median similarity to the nearest real training ticket.
        This is the number that explained why the Gemma tickets did not help.

    python src/experiments/synthetic_data/compare_synthetic.py synthetic_tickets qwen35_08b_tickets
"""
from __future__ import annotations

import json
import sys

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics.pairwise import linear_kernel
from sklearn.neural_network import MLPClassifier

from common import RESULTS, load_meta, load_split

SEEDS = (1, 2, 3)
BOOT = 2000
WEAK = ("General Inquiry", "Returns and Exchanges", "Human Resources", "Sales and Pre-Sales")


def load_synthetic(name):
    rows = [json.loads(l) for l in open(f"data/synthetic/{name}.jsonl")]
    return [r["text"] for r in rows], np.asarray([r["label"] for r in rows]), rows


def train_predict(x, y, x_te, seed):
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=200_000, sublinear_tf=True)
    m = MLPClassifier(hidden_layer_sizes=(256,), max_iter=60, early_stopping=True,
                      n_iter_no_change=5, random_state=seed).fit(vec.fit_transform(x), y)
    return m.predict(vec.transform(x_te))


def main() -> None:
    args = sys.argv[1:]
    out_name = "synthetic_comparison"
    if "--out" in args:
        k = args.index("--out"); out_name = args[k + 1]; args = args[:k] + args[k + 2:]
    names = args
    labels = load_meta()["labels"]
    x_tr, y_tr = load_split("train")
    x_te, y_te = load_split("test")
    y_tr, y_te = np.asarray(y_tr), np.asarray(y_te)
    split = json.loads(open("data/synthetic/split.json").read())
    core, val = split["core"], split["validation"]
    x_core, y_core = [x_tr[i] for i in core], y_tr[core]
    x_val, y_val = [x_tr[i] for i in val], y_tr[val]

    v = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True).fit(x_tr)
    R = v.transform(x_tr)
    nearest = lambda texts: np.concatenate(
        [linear_kernel(v.transform(texts[s:s+500]), R).max(1) for s in range(0, len(texts), 500)])
    unfam = nearest(x_te) < 0.5
    weak = np.isin(y_te, [labels.index(q) for q in WEAK])

    setups = {"core only": (x_core, y_core, None),
              "core + real validation": (x_core + x_val, np.concatenate([y_core, y_val]),
                                         float(np.median(nearest(x_val))))}
    for n in names:
        xs, ys, _ = load_synthetic(n)
        setups[f"core + {n}"] = (x_core + xs, np.concatenate([y_core, ys]), float(np.median(nearest(xs))))

    preds = {}
    out = {}
    print(f"{'training set':38s} {'rows':>6} {'accuracy':>12} {'macro-F1':>9} {'unfamiliar':>11} "
          f"{'weak queues':>12} {'style':>6}")
    for name, (x, y, style) in setups.items():
        runs = []
        for s in SEEDS:
            p = train_predict(x, y, x_te, s)
            preds.setdefault(name, []).append(p)
            runs.append((accuracy_score(y_te, p), f1_score(y_te, p, average="macro"),
                         accuracy_score(y_te[unfam], p[unfam]), accuracy_score(y_te[weak], p[weak])))
        r = np.array(runs)
        mu, sd = r.mean(0), r.std(0)
        out[name] = {"rows": len(y), "accuracy": round(float(mu[0]), 4), "accuracy_sd": round(float(sd[0]), 4),
                     "macro_f1": round(float(mu[1]), 4), "unfamiliar": round(float(mu[2]), 4),
                     "weak_queues": round(float(mu[3]), 4), "style_median_similarity": style}
        st = f"{style:.3f}" if style is not None else "  -  "
        print(f"{name:38s} {len(y):6d} {mu[0]*100:7.2f}±{sd[0]*100:.2f} {mu[1]:9.3f} {mu[2]*100:10.2f}% "
              f"{mu[3]*100:11.2f}% {st:>6}", flush=True)

    # Paired bootstrap: resample test tickets, and for each resample compare the
    # same ticket's correctness with and without the extra data, averaged over
    # seeds. Pairing removes ticket difficulty from the comparison.
    rng = np.random.default_rng(0)
    base = np.mean([p == y_te for p in preds["core only"]], axis=0)
    print(f"\ngain over core only, with a 95% confidence interval from {BOOT} paired bootstrap resamples:")
    for name in setups:
        if name == "core only":
            continue
        cur = np.mean([p == y_te for p in preds[name]], axis=0)
        diff = cur - base
        boots = [diff[rng.integers(0, len(diff), len(diff))].mean() for _ in range(BOOT)]
        lo, hi = np.percentile(boots, [2.5, 97.5])
        verdict = "real gain" if lo > 0 else ("real loss" if hi < 0 else "not distinguishable from zero")
        out[name]["gain"] = round(float(diff.mean()), 4)
        out[name]["gain_ci95"] = [round(float(lo), 4), round(float(hi), 4)]
        print(f"  {name:38s} {diff.mean()*100:+6.2f} points   [{lo*100:+.2f}, {hi*100:+.2f}]   {verdict}")

    (RESULTS / f"{out_name}.json").write_text(json.dumps(out, indent=2))
    print(f"\nwrote {RESULTS / (out_name + '.json')}")


if __name__ == "__main__":
    main()
