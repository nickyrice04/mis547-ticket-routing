"""Do generated siblings raise accuracy on the validation slice?

Everything here trains on the core tickets and scores on the validation slice.
The test set is not read. Three routers are scored with and without siblings:

    network     TF-IDF plus the one hidden layer network
    lookup      the queue of the nearest training ticket (1-NN, TF-IDF cosine)
    hybrid      lookup when the nearest ticket is similar enough, else the network

The paired bootstrap resamples validation tickets, so the interval says whether
the change from adding siblings is distinguishable from noise.

    python src/experiments/synthetic_data/eval_siblings.py qwen_sibling_core
    python src/experiments/synthetic_data/eval_siblings.py qwen_sibling_core --per 1     # only the first sibling of each ticket
"""
from __future__ import annotations

import json
import sys

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics.pairwise import linear_kernel
from sklearn.neural_network import MLPClassifier

from common import RESULTS, load_split

GATE = 0.30      # fixed before siblings existed, from the plain core run
BOOT = 2000
BUCKETS = [(0, .3), (.3, .4), (.4, .5), (.5, .6), (.6, .7), (.7, 1.01)]


def route(x_pool, y_pool, x_eval, seed=1):
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=200_000, sublinear_tf=True)
    A, B = vec.fit_transform(x_pool), vec.transform(x_eval)
    net = MLPClassifier(hidden_layer_sizes=(256,), max_iter=60, early_stopping=True,
                        n_iter_no_change=5, random_state=seed).fit(A, y_pool).predict(B)
    sim, nn = np.zeros(len(x_eval)), np.zeros(len(x_eval), int)
    for a in range(0, len(x_eval), 500):
        K = linear_kernel(B[a:a + 500], A)
        sim[a:a + 500], nn[a:a + 500] = K.max(1), K.argmax(1)
    look = y_pool[nn]
    return {"network": net, "lookup": look, "hybrid": np.where(sim >= GATE, look, net)}, sim


def main() -> None:
    args = sys.argv[1:]
    per = None
    if "--per" in args:
        k = args.index("--per"); per = int(args[k + 1]); args = args[:k] + args[k + 2:]
    x, y = load_split("train")
    y = np.asarray(y)
    split = json.loads(open("data/synthetic/split.json").read())
    core, val = split["core"], split["validation"]
    xc, yc, xv, yv = [x[i] for i in core], y[core], [x[i] for i in val], y[val]

    base, base_sim = route(xc, yc, xv)
    out = {"core only": {k: round(float(accuracy_score(yv, p)), 4) for k, p in base.items()}}
    print(f"{'pool':34s} {'rows':>6} {'network':>8} {'lookup':>8} {'hybrid':>8} {'hybrid F1':>10}")
    print(f"{'core only':34s} {len(xc):6d} " + " ".join(f"{accuracy_score(yv, base[k])*100:7.2f}%" for k in base)
          + f" {f1_score(yv, base['hybrid'], average='macro'):10.3f}")

    rng = np.random.default_rng(0)
    for name in args:
        rows = [json.loads(l) for l in open(f"data/synthetic/{name}.jsonl")]
        if per is not None:
            rows = [r for r in rows if r["k"] < per]
        xs, ys = [r["text"] for r in rows], np.asarray([r["label"] for r in rows])
        cur, sim = route(xc + xs, np.concatenate([yc, ys]), xv)
        tag = name + (f" (first {per})" if per is not None else "")
        print(f"{'core + ' + tag:34s} {len(xc)+len(xs):6d} "
              + " ".join(f"{accuracy_score(yv, cur[k])*100:7.2f}%" for k in cur)
              + f" {f1_score(yv, cur['hybrid'], average='macro'):10.3f}")
        res = {k: round(float(accuracy_score(yv, p)), 4) for k, p in cur.items()}
        for k in ("lookup", "hybrid"):
            d = (cur[k] == yv).astype(float) - (base[k] == yv).astype(float)
            boots = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(BOOT)]
            lo, hi = np.percentile(boots, [2.5, 97.5])
            res[f"{k}_gain"], res[f"{k}_gain_ci95"] = round(float(d.mean()), 4), [round(float(lo), 4), round(float(hi), 4)]
            print(f"    {k} gain {d.mean()*100:+.2f} points  [{lo*100:+.2f}, {hi*100:+.2f}]")
        print("    by similarity to the nearest CORE ticket (share, lookup before -> after):")
        for lo_, hi_ in BUCKETS:
            m = (base_sim >= lo_) & (base_sim < hi_)
            print(f"      {lo_:.1f}-{min(hi_,1):.1f}  {m.mean()*100:5.1f}%   "
                  f"{(base['lookup'][m]==yv[m]).mean()*100:5.1f}% -> {(cur['lookup'][m]==yv[m]).mean()*100:5.1f}%")
        res["median_similarity_after"] = round(float(np.median(sim)), 3)
        out["core + " + tag] = res
    out["median_similarity_core_only"] = round(float(np.median(base_sim)), 3)
    (RESULTS / "sibling_eval.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
