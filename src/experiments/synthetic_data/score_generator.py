"""How real does each synthetic set look, before any accuracy test?

Two numbers per source, both measured against the core of the training set, so
real validation tickets are a fair reference (they are not inside core):

    style        median TF-IDF similarity to the nearest core ticket
    label fit    how often a network trained on real core tickets agrees with
                 the queue the ticket was written for

    python src/experiments/synthetic_data/score_generator.py qwen_think_10k qwen_sft_t07 qwen_sft_t10
"""
from __future__ import annotations

import json
import sys

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from sklearn.neural_network import MLPClassifier

from common import RESULTS, load_split
from experiments.synthetic_data.compare_synthetic import load_synthetic


def main() -> None:
    x_tr, y_tr = load_split("train")
    y_tr = np.asarray(y_tr)
    split = json.loads(open("data/synthetic/split.json").read())
    core, val = split["core"], split["validation"]
    x_core = [x_tr[i] for i in core]

    sim_vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True).fit(x_core)
    C = sim_vec.transform(x_core)
    near = lambda xs: np.concatenate(
        [linear_kernel(sim_vec.transform(xs[s:s+500]), C).max(1) for s in range(0, len(xs), 500)])

    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=200_000, sublinear_tf=True)
    judge = MLPClassifier(hidden_layer_sizes=(256,), max_iter=60, early_stopping=True,
                          n_iter_no_change=5, random_state=42).fit(vec.fit_transform(x_core), y_tr[core])

    sources = {"real validation": ([x_tr[i] for i in val], y_tr[val])}
    for n in sys.argv[1:]:
        xs, ys, _ = load_synthetic(n)
        sources[n] = (xs, ys)

    out = {}
    print(f"{'source':22s} {'rows':>6} {'style':>6} {'label fit':>10}")
    for name, (xs, ys) in sources.items():
        style = float(np.median(near(xs)))
        fit = float((judge.predict(vec.transform(xs)) == ys).mean())
        out[name] = {"rows": len(xs), "style_median_similarity": round(style, 3), "label_fit": round(fit, 3)}
        print(f"{name:22s} {len(xs):6d} {style:6.3f} {fit*100:9.1f}%")
    (RESULTS / "generator_realism.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
