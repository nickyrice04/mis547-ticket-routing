"""Cheap-model battery on the CV-simulated unfamiliar condition (core only)."""
import json, time, sys
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import ComplementNB, MultinomialNB
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score
from experiments.retrieval_tracks.x_unfamiliar_lib import *
from experiments.retrieval_tracks.x_unfamiliar_cv import folds, fold_sims

xc, yc, xv, yv = load_core_val()
sim, nnidx, fold = fold_sims(xc, yc)
unf = sim < 0.4
print("CV unfamiliar share", unf.mean(), "n", unf.sum())
print("1-NN on CV-unfamiliar:", (yc[nnidx][unf] == yc[unf]).mean(), " majority:", (yc[unf] == np.bincount(yc).argmax()).mean())
print("1-NN overall in CV:", (yc[nnidx] == yc).mean())

def word_vec(): return base_vectorizer()
def char_vec(): return TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=2, max_features=300_000, sublinear_tf=True)
def uni_vec(): return TfidfVectorizer(ngram_range=(1, 1), min_df=2, sublinear_tf=True)

models = {
  "lr_word_C1": (word_vec, lambda: LogisticRegression(C=1, max_iter=300)),
  "lr_word_C10": (word_vec, lambda: LogisticRegression(C=10, max_iter=300)),
  "lr_word_C10_bal": (word_vec, lambda: LogisticRegression(C=10, max_iter=300, class_weight="balanced")),
  "lr_uni_C3": (uni_vec, lambda: LogisticRegression(C=3, max_iter=300)),
  "svc_word_C0.3": (word_vec, lambda: LinearSVC(C=0.3)),
  "cnb_word": (word_vec, lambda: ComplementNB(alpha=0.3)),
  "mnb_word": (word_vec, lambda: MultinomialNB(alpha=0.1)),
  "lr_char_C10": (char_vec, lambda: LogisticRegression(C=10, max_iter=300)),
  "mlp256": (word_vec, lambda: MLPClassifier(hidden_layer_sizes=(256,), max_iter=60, early_stopping=True, n_iter_no_change=5, random_state=0)),
}
only = sys.argv[1:] 
out = {}
for name, (mkvec, mkclf) in models.items():
    if only and name not in only: continue
    t0 = time.time(); pred = np.zeros(len(xc), dtype=np.int64)
    for f, (tr, te) in enumerate(folds(yc)):
        vec = mkvec(); Xtr = vec.fit_transform([xc[i] for i in tr]); Xte = vec.transform([xc[i] for i in te])
        clf = mkclf().fit(Xtr, yc[tr]); pred[te] = clf.predict(Xte)
    a_unf = (pred[unf] == yc[unf]).mean(); a_all = (pred == yc).mean()
    lo = sim < 0.3; mid = (sim >= 0.3) & (sim < 0.4)
    out[name] = {"unf": a_unf, "lt0.3": (pred[lo] == yc[lo]).mean(), "0.3-0.4": (pred[mid] == yc[mid]).mean(), "all": a_all,
                 "f1_unf": f1_score(yc[unf], pred[unf], average="macro")}
    print(f"{name:18s} unf={a_unf:.4f} (<0.3 {out[name]['lt0.3']:.4f}, 0.3-0.4 {out[name]['0.3-0.4']:.4f}) all={a_all:.4f} mF1_unf={out[name]['f1_unf']:.4f} [{time.time()-t0:.0f}s]", flush=True)
json.dump(out, open("results/x_unfamiliar_battery1.json", "w"), indent=1)
