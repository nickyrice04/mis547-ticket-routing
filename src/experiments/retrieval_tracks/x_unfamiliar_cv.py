"""Leave-family-out style CV inside core.

5 stratified folds over core. For each fold the model is trained on the other 4 folds, and scored
only on held-out tickets whose nearest baseline-TF-IDF similarity to the training folds is < 0.4
(the same "unfamiliar" condition validation tickets are in relative to core). Validation is never touched.
"""
import json, sys, time
import numpy as np
from sklearn.model_selection import StratifiedKFold
from experiments.retrieval_tracks.x_unfamiliar_lib import *

def folds(yc, k=5, seed=0):
    return list(StratifiedKFold(k, shuffle=True, random_state=seed).split(np.zeros(len(yc)), yc))

def fold_sims(xc, yc, cache="data/x_unfamiliar/cv_fold_sims.npz"):
    """nearest-sim of each core ticket to the training folds of its own CV split (vectorizer fit per fold)."""
    import os
    if os.path.exists(cache):
        z = np.load(cache); return z["sim"], z["idx"], z["fold"]
    sim = np.zeros(len(xc), dtype=np.float32); idx = np.zeros(len(xc), dtype=np.int64); fold = np.zeros(len(xc), dtype=np.int64)
    for f, (tr, te) in enumerate(folds(yc)):
        vec = base_vectorizer(); Xtr = vec.fit_transform([xc[i] for i in tr]); Xte = vec.transform([xc[i] for i in te])
        s, j = nn_search(Xte, Xtr)
        sim[te] = s; idx[te] = tr[j]; fold[te] = f
    np.savez(cache, sim=sim, idx=idx, fold=fold)
    return sim, idx, fold
