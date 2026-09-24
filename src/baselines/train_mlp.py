"""The missing rung: a plain neural network on TF-IDF features.

The ladder jumped from logistic regression straight to a pretrained
transformer, which confounds two different things. DistilBERT is bigger *and*
it arrives pretrained on a large amount of English. When it wins, we cannot say
which of the two caused it.

A feed-forward network on the same TF-IDF features separates them. It has
hidden layers, so it has more capacity than logistic regression, but it has no
pretraining and it still cannot see word order. That makes it the control:

  * if it lands near logistic regression, the transformer's gain comes from
    pretraining and word order, not from raw capacity
  * if it closes most of the gap, capacity was the missing ingredient and we do
    not need a 67M parameter model to route a ticket

It is the same architecture as the fishing charter lab, one hidden layer with a
handful of nodes, only wider because the input is 60,000 TF-IDF columns instead
of two scaled numbers.

Also trains two classical text baselines for context, because bag-of-words
methods are what this task was historically solved with.
"""
from __future__ import annotations

import resource
import time

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.naive_bayes import ComplementNB
from sklearn.neural_network import MLPClassifier

from common import load_meta, load_split, save_result, threshold_table


def run(name, tier, make_model, x_tr, y_tr, x_te, y_te, vec_kwargs=None):
    """Fit one model on TF-IDF features, score the test set, save the joblib and the result JSON."""
    vec = TfidfVectorizer(**(vec_kwargs or dict(ngram_range=(1, 2), min_df=2,
                                                max_features=200_000, sublinear_tf=True)))
    t0 = time.time()
    a_tr = vec.fit_transform(x_tr)
    model = make_model()
    model.fit(a_tr, y_tr)
    train_seconds = time.time() - t0

    probs = model.predict_proba(vec.transform(x_te))
    pred = probs.argmax(axis=1)

    # The vectorizer ships with the model, same as the baseline.
    import joblib
    from pathlib import Path

    Path("models").mkdir(exist_ok=True)
    joblib.dump((vec, model), f"models/{tier}.joblib")
    params = getattr(model, "coefs_", None)
    n_params = sum(c.size for c in params) / 1e6 if params else a_tr.shape[1] * len(set(y_tr)) / 1e6

    save_result({
        "tier": tier,
        "model": name,
        "params_millions": round(n_params, 2),
        "train_seconds": round(train_seconds, 1),
        "train_peak_rss_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6),
        "accuracy": round(float(accuracy_score(y_te, pred)), 4),
        "macro_f1": round(float(f1_score(y_te, pred, average="macro")), 4),
        "thresholds": threshold_table(probs, y_te),
    })
    print(f"{tier:22s} acc {accuracy_score(y_te, pred)*100:5.1f}%  "
          f"macroF1 {f1_score(y_te, pred, average='macro'):.3f}  "
          f"{train_seconds/60:.1f} min", flush=True)


def main() -> None:
    x_tr, y_tr = load_split("train")
    x_te, y_te = load_split("test")
    load_meta()

    # One hidden layer, same shape as the lab, scaled up for a text-sized input.
    run("TF-IDF + MLP, one hidden layer of 256", "1b_mlp_256",
        lambda: MLPClassifier(hidden_layer_sizes=(256,), max_iter=60, early_stopping=True,
                              n_iter_no_change=5, random_state=42, verbose=False),
        x_tr, y_tr, x_te, y_te)

    # Two hidden layers, to see whether depth adds anything.
    run("TF-IDF + MLP, hidden layers 512 and 256", "1c_mlp_512_256",
        lambda: MLPClassifier(hidden_layer_sizes=(512, 256), max_iter=60, early_stopping=True,
                              n_iter_no_change=5, random_state=42, verbose=False),
        x_tr, y_tr, x_te, y_te)

    # The classic text baseline, for reference. Trains in about a second.
    run("Complement Naive Bayes on word counts", "1d_complement_nb",
        lambda: ComplementNB(), x_tr, y_tr, x_te, y_te)


if __name__ == "__main__":
    main()
