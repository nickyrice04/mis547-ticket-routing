"""Dense-embedding models on the CV-simulated unfamiliar condition (core only)."""
import json, time, sys
import numpy as np
from sklearn.linear_model import LogisticRegression
from experiments.retrieval_tracks.x_unfamiliar_lib import *
from experiments.retrieval_tracks.x_unfamiliar_cv import folds, fold_sims

xc, yc, xv, yv = load_core_val()
sim, nnidx, fold = fold_sims(xc, yc); unf = sim < 0.4
out = {}
for tag in sys.argv[1:]:
    E = np.load(f"data/x_unfamiliar/emb_{tag}_core.npy")
    p1 = np.zeros(len(yc), int); dsim = np.zeros(len(yc)); pk = {k: np.zeros(len(yc), int) for k in (5, 20, 50)}
    plr = {C: np.zeros(len(yc), int) for C in (1, 10)}
    for tr, te in folds(yc):
        S = E[te] @ E[tr].T
        j = S.argmax(1); p1[te] = yc[tr][j]; dsim[te] = S.max(1)
        order = np.argsort(-S, axis=1)[:, :50]
        for k in pk:
            votes = np.zeros((len(te), 10))
            for r in range(k):
                w = S[np.arange(len(te)), order[:, r]] ** 4
                np.add.at(votes, (np.arange(len(te)), yc[tr][order[:, r]]), w)
            pk[k][te] = votes.argmax(1)
        for C in plr:
            plr[C][te] = LogisticRegression(C=C, max_iter=300).fit(E[tr], yc[tr]).predict(E[te])
    r = {"dense_1nn_unf": (p1[unf] == yc[unf]).mean(), "dense_1nn_all": (p1 == yc).mean()}
    for k in pk: r[f"dense_knn{k}_unf"] = (pk[k][unf] == yc[unf]).mean()
    for C in plr: r[f"dense_lr_C{C}_unf"] = (plr[C][unf] == yc[unf]).mean(); r[f"dense_lr_C{C}_all"] = (plr[C] == yc).mean()
    # is dense similarity a better family detector inside the lexical-unfamiliar set?
    qs = np.quantile(dsim[unf], [0, .25, .5, .75, .9, 1.0])
    for a, b in zip(qs[:-1], qs[1:]):
        m = unf & (dsim >= a) & (dsim <= b)
        r[f"dense1nn_unf_dsim_{a:.3f}-{b:.3f}"] = [int(m.sum()), float((p1[m] == yc[m]).mean())]
    out[tag] = r
    print(tag, json.dumps(r, indent=1), flush=True)
    np.savez(f"data/x_unfamiliar/cv_dense_{tag}.npz", p1=p1, dsim=dsim)
json.dump(out, open("results/x_unfamiliar_battery2.json", "w"), indent=1)
