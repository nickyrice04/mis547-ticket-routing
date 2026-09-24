"""Round 1: linear / shallow models on the baseline TF-IDF, 3-fold CV inside core only."""
from __future__ import annotations

import sys
import time
import warnings

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, PassiveAggressiveClassifier, RidgeClassifier, SGDClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import ComplementNB
from sklearn.svm import LinearSVC

from experiments.retrieval_tracks.x_classifier_lib import BASE_VEC, bucket_table, fmt_buckets, load_core_val, log_cv

warnings.filterwarnings("ignore")

xc, yc, _, _ = load_core_val()
xc = np.array(xc, dtype=object)
skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=0)
folds = []
for tr, te in skf.split(xc, yc):
    vec = TfidfVectorizer(**BASE_VEC)
    a = vec.fit_transform(xc[tr])
    b = vec.transform(xc[te])
    d = (b @ a.T).toarray()
    sims = d.max(axis=1)
    nn = yc[tr][d.argmax(axis=1)]
    folds.append((a, yc[tr], b, yc[te], sims, nn))
    del d
print("folds ready", flush=True)


def run(name, make):
    accs, preds, ys, ss = [], [], [], []
    t0 = time.time()
    for a, ya, b, yb, sims, nn in folds:
        if make == "1nn":
            p = nn
        else:
            m = make()
            m.fit(a, ya)
            p = m.predict(b)
        accs.append((p == yb).mean())
        preds.append(p); ys.append(yb); ss.append(sims)
    rows = bucket_table(np.concatenate(ss), np.concatenate(preds), np.concatenate(ys))
    log_cv(name, accs, {"buckets": fmt_buckets(rows), "sec": round(time.time() - t0, 1)})


which = sys.argv[1] if len(sys.argv) > 1 else "all"
run("1nn", "1nn")
for C in (0.1, 0.3, 1, 3, 10, 30, 100):
    run(f"linearsvc_C{C}", lambda C=C: LinearSVC(C=C, max_iter=5000))
for C in (1, 10, 100):
    run(f"linearsvc_cs_C{C}", lambda C=C: LinearSVC(C=C, max_iter=5000, multi_class="crammer_singer"))
for a in (1.0, 0.3, 0.1, 0.03, 0.01):
    run(f"ridge_a{a}", lambda a=a: RidgeClassifier(alpha=a, solver="sparse_cg", max_iter=200))
for C in (1.0, 10.0):
    run(f"pa_C{C}", lambda C=C: PassiveAggressiveClassifier(C=C, max_iter=50, tol=None, random_state=0))
for al in (1e-5, 1e-6, 1e-7):
    run(f"sgd_mhuber_a{al}", lambda al=al: SGDClassifier(loss="modified_huber", alpha=al, max_iter=50, tol=None, random_state=0))
for al in (0.01, 0.1, 0.3):
    run(f"cnb_a{al}", lambda al=al: ComplementNB(alpha=al))
for C in (10, 100, 1000):
    run(f"logreg_C{C}", lambda C=C: LogisticRegression(C=C, max_iter=300))
