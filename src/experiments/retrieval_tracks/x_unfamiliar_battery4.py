"""Does training the specialist on (or up-weighting) the pool's own family-less tickets help on CV-unfamiliar? Core only."""
import json, numpy as np
from sklearn.linear_model import LogisticRegression
from experiments.retrieval_tracks.x_unfamiliar_lib import *
from experiments.retrieval_tracks.x_unfamiliar_cv import folds, fold_sims
xc, yc, _, _ = load_core_val(); sim, _, _ = fold_sims(xc, yc); unf = sim < 0.4
E = np.load("data/x_unfamiliar/emb_mpnet_core.npy"); P = {}
for tr, te in folds(yc):
    vec = base_vectorizer(); Xtr = vec.fit_transform([xc[i] for i in tr]); Xte = vec.transform([xc[i] for i in te])
    s, _ = nn_search(Xtr, Xtr, exclude_self=True); single = s < 0.4
    for name, w in (("all", np.ones(len(tr))), ("singletons_only", single.astype(float)), ("singletons_x3", 1 + 2 * single)):
        m = w > 0
        a = LogisticRegression(C=1, max_iter=300).fit(Xtr[m], yc[tr][m], sample_weight=w[m]); b = LogisticRegression(C=1, max_iter=500).fit(E[tr][m], yc[tr][m], sample_weight=w[m])
        pa = np.full((len(te), 10), 1e-6); pa[:, a.classes_] = a.predict_proba(Xte); pb = np.full((len(te), 10), 1e-6); pb[:, b.classes_] = b.predict_proba(E[te])
        P.setdefault(name, np.zeros((len(yc), 10)))[te] = np.log(pa) + np.log(pb)
res = {k: float((v.argmax(1)[unf] == yc[unf]).mean()) for k, v in P.items()}
print(res); json.dump(res, open("results/x_unfamiliar_battery4.json", "w"), indent=1)
