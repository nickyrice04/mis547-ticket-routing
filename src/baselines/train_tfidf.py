"""Tier 1: TF-IDF + logistic regression.

This is the baseline from the midterm report. It is here so the scaling study
measures every tier the same way, on the same split, with the same metrics.
"""
from __future__ import annotations

import resource
import time

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

from common import load_meta, load_split, save_result, threshold_table


def main() -> None:
    meta = load_meta()
    x_tr, y_tr = load_split("train")
    x_te, y_te = load_split("test")

    t0 = time.time()
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=200_000, sublinear_tf=True)
    a_tr = vec.fit_transform(x_tr)
    clf = LogisticRegression(max_iter=2000, C=5, class_weight="balanced")
    clf.fit(a_tr, y_tr)
    train_seconds = time.time() - t0

    probs = clf.predict_proba(vec.transform(x_te))
    pred = probs.argmax(axis=1)

    # The vectorizer ships with the model, exactly as the midterm report argues.
    import joblib
    from pathlib import Path

    Path("models").mkdir(exist_ok=True)
    joblib.dump((vec, clf), "models/1_tfidf_logreg.joblib")

    # Out-of-vocabulary rate: the share of test words this fitted vectorizer has
    # never seen. Professor Zara asked what happens to out-of-band vocabulary,
    # and for a bag-of-words model the answer is that those words are dropped.
    vocab = set(vec.vocabulary_)
    oov_rates = []
    for text in x_te:
        words = text.split()
        if words:
            oov_rates.append(sum(w not in vocab for w in words) / len(words))
    oov = np.asarray(oov_rates)
    correct = pred == np.asarray(y_te)
    high = oov >= np.quantile(oov, 0.9)

    save_result(
        {
            "tier": "1_tfidf_logreg",
            "model": "TF-IDF 1-2gram + LogisticRegression",
            "params_millions": round(a_tr.shape[1] * len(meta["labels"]) / 1e6, 2),
            "train_seconds": round(train_seconds, 1),
            "train_peak_rss_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6),
            "accuracy": round(accuracy_score(y_te, pred), 4),
            "macro_f1": round(f1_score(y_te, pred, average="macro"), 4),
            "vocab_size": len(vocab),
            "oov_word_rate_mean": round(float(oov.mean()), 4),
            "accuracy_low_oov": round(float(correct[~high].mean()), 4),
            "accuracy_high_oov": round(float(correct[high].mean()), 4),
            "thresholds": threshold_table(probs, y_te),
        }
    )


if __name__ == "__main__":
    main()
