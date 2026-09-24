"""Stratified subsets of the 10,000 thinking tickets for the dose-response test.

Each subset keeps the same share per queue as the full set, so the only thing
changing between rows is how many synthetic tickets were added.
"""
import collections, json, random
rows = [json.loads(l) for l in open("data/synthetic/qwen_think_10k.jsonl")]
by = collections.defaultdict(list)
for r in rows: by[r["queue"]].append(r)
for n, name in ((1000, "qwen_think_1k"), (2500, "qwen_think_2500"), (5000, "qwen_think_5k")):
    rng = random.Random(42); out = []
    for q, rs in by.items():
        out += rng.sample(rs, min(len(rs), round(n * len(rs) / len(rows))))
    with open(f"data/synthetic/{name}.jsonl", "w") as f:
        for r in out: f.write(json.dumps(r) + "\n")
    print(f"{name}: {len(out)} tickets")

# The judged set: keep only tickets that a network trained on real core data
# assigns to their intended queue. This is the filter step in NVIDIA's
# generate-then-judge pattern, using a judge we already have. The judge trains
# on core only, so the test set plays no part in deciding what is kept.
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neural_network import MLPClassifier
from common import load_split
x_tr, y_tr = load_split("train"); y_tr = np.asarray(y_tr)
core = json.load(open("data/synthetic/split.json"))["core"]
vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=200_000, sublinear_tf=True)
judge = MLPClassifier(hidden_layer_sizes=(256,), max_iter=60, early_stopping=True,
                      n_iter_no_change=5, random_state=42)
judge.fit(vec.fit_transform([x_tr[i] for i in core]), y_tr[core])
pred = judge.predict(vec.transform([r["text"] for r in rows]))
kept = [r for r, p in zip(rows, pred) if p == r["label"]]
with open("data/synthetic/qwen_think_judged.jsonl", "w") as f:
    for r in kept: f.write(json.dumps(r) + "\n")
print(f"qwen_think_judged: {len(kept)} of {len(rows)} tickets pass the judge")
