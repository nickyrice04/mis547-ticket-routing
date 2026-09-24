"""Find the patterns behind the model's mistakes, without touching the test set.

Hard example mining: look at what the model gets wrong, then write more training
examples like those. The one rule that keeps it honest is where the mistakes
come from. If they come from the test set, generated look-alikes of test
tickets would teach the model the exam, and test accuracy would stop meaning
anything. So the mistakes here come from a validation slice carved out of the
training data, and the test set is never read by this script.

Writes:
    data/synthetic/split.json          which training rows are core vs validation
    data/synthetic/error_report.json   confusion pairs, keywords, seed examples
"""
from __future__ import annotations

import collections
import json
import re
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

from common import load_meta, load_split

OUT = Path("data/synthetic")
VAL_FRACTION = 0.15
TOP_PAIRS = 8
SEEDS_PER_PAIR = 12
STOP = set("""a an the and or but if to of in on for with at by from as is are was were be
been being it its this that these those i we you they he she my our your their me us
them please thank thanks hello dear regards best support customer team would could
can will just also any some more very have has had do does did not no so than then
there here what which who when where how all about into out up down over after
before while because such only own same too s t don now""".split())


def words(text):
    return [w for w in re.findall(r"[a-z][a-z']+", text) if w not in STOP and len(w) > 2]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    labels = load_meta()["labels"]
    x_tr, y_tr = load_split("train")
    y_tr = np.asarray(y_tr)

    core_idx, val_idx = train_test_split(
        np.arange(len(x_tr)), test_size=VAL_FRACTION, random_state=42, stratify=y_tr
    )
    (OUT / "split.json").write_text(json.dumps(
        {"core": core_idx.tolist(), "validation": val_idx.tolist()}))

    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=200_000, sublinear_tf=True)
    X_core = vec.fit_transform([x_tr[i] for i in core_idx])
    model = MLPClassifier(hidden_layer_sizes=(256,), max_iter=60, early_stopping=True,
                          n_iter_no_change=5, random_state=42)
    model.fit(X_core, y_tr[core_idx])

    x_val = [x_tr[i] for i in val_idx]
    y_val = y_tr[val_idx]
    probs = model.predict_proba(vec.transform(x_val))
    pred = probs.argmax(1)
    wrong = pred != y_val
    print(f"validation: {len(y_val)} tickets, accuracy {(~wrong).mean()*100:.1f}%, "
          f"{wrong.sum()} mistakes")

    pairs = collections.Counter((int(y_val[i]), int(pred[i])) for i in np.where(wrong)[0])

    # Keywords that show up more in the misrouted tickets of a queue than in
    # the ones the model got right. These are the words pulling tickets the
    # wrong way.
    report = {"validation_accuracy": round(float((~wrong).mean()), 4),
              "validation_size": int(len(y_val)), "mistakes": int(wrong.sum()), "pairs": []}
    print(f"\n{'true queue':>32} -> {'predicted as':<32} {'count':>5}")
    for (true, guess), n in pairs.most_common(TOP_PAIRS):
        err = [i for i in np.where(wrong)[0] if y_val[i] == true and pred[i] == guess]
        ok = [i for i in np.where(~wrong)[0] if y_val[i] == true]
        c_err = collections.Counter(w for i in err for w in set(words(x_val[i])))
        c_ok = collections.Counter(w for i in ok for w in set(words(x_val[i])))
        score = {w: (c_err[w] + 1) / (len(err) + 2) / ((c_ok[w] + 1) / (len(ok) + 2))
                 for w in c_err if c_err[w] >= 3}
        keywords = [w for w, _ in sorted(score.items(), key=lambda kv: -kv[1])[:12]]
        # Seeds are the misrouted tickets, least confident first, so the
        # generator sees the genuinely ambiguous ones.
        err_sorted = sorted(err, key=lambda i: probs[i].max())
        seeds = [x_val[i][:600] for i in err_sorted[:SEEDS_PER_PAIR]]
        report["pairs"].append({
            "true": labels[true], "predicted": labels[guess], "count": n,
            "keywords": keywords, "seeds": seeds,
        })
        print(f"{labels[true]:>32} -> {labels[guess]:<32} {n:5d}   {', '.join(keywords[:6])}")

    per_queue = {labels[q]: round(float((pred[y_val == q] == q).mean()), 3) for q in range(len(labels))}
    report["per_queue_accuracy"] = per_queue
    print("\naccuracy by true queue:")
    for q, a in sorted(per_queue.items(), key=lambda kv: kv[1]):
        print(f"  {q:34s} {a*100:5.1f}%")

    (OUT / "error_report.json").write_text(json.dumps(report, indent=2))
    print(f"\nwrote {OUT/'error_report.json'}")


if __name__ == "__main__":
    main()
