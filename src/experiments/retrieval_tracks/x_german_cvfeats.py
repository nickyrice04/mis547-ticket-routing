"""Track "german": out-of-fold retrieval features inside core, and the same features for validation.

Tuning happens on the core out-of-fold rows only. Validation features are written to a separate
file and are only read by the confirmation step.

    python src/experiments/retrieval_tracks/x_german_cvfeats.py knn      # retrieval channels (fast)
    python src/experiments/retrieval_tracks/x_german_cvfeats.py mlp      # MLP probabilities trained on English + kept German (slow)
"""
from __future__ import annotations

import sys
import time

import numpy as np
from sklearn.model_selection import StratifiedKFold

from final.features import DIAG as GDIR, German, channels, e5_embed, guard_mask
from final.lib import baseline_vec, load_core_val, make_mlp

FOLDS = 5


def folds(y):
    return list(StratifiedKFold(FOLDS, shuffle=True, random_state=0).split(np.zeros(len(y)), y))


def main(what):
    xc, yc, xv, yv = load_core_val()
    g = German()
    Ec, Ev = e5_embed(xc), e5_embed(xv)
    t0 = time.time()
    jobs = [(f"fold{k}", tr, ev) for k, (tr, ev) in enumerate(folds(yc))] + [("val", np.arange(len(xc)), None)]
    oof = np.zeros((len(xc), 10, 10), np.float32)
    oof_mlp = np.zeros((len(xc), 10), np.float32)
    for name, tr, ev in jobs:
        x_tr = [xc[i] for i in tr]
        x_ev = xv if ev is None else [xc[i] for i in ev]
        E_tr, E_ev = Ec[tr], (Ev if ev is None else Ec[ev])
        vec = baseline_vec().fit(x_tr + g.texts)
        A, B, G = vec.transform(x_tr), vec.transform(x_ev), vec.transform(g.texts)
        keep, info = guard_mask(g, G, B, E_ev)
        print(name, info, f"{time.time()-t0:.0f}s", flush=True)
        if what == "knn":
            F = channels(A, yc[tr], E_tr, B, E_ev, G, g, keep)
            if ev is None:
                np.save(GDIR / "feats_val_knn.npy", F)
            else:
                oof[ev] = F
        elif what == "e5lr":
            from sklearn.linear_model import LogisticRegression
            X = np.vstack([E_tr, g.e5_trans[keep]])
            Y = np.concatenate([yc[tr], g.labels[keep]])
            mu, sd = X.mean(0), X.std(0) + 1e-6
            lr = LogisticRegression(C=1.0, max_iter=300).fit((X - mu) / sd, Y)
            P = lr.predict_proba((E_ev - mu) / sd)
            if ev is None:
                np.save(GDIR / "feats_val_e5lr.npy", P)
            else:
                oof_mlp[ev] = P
        else:
            from scipy.sparse import vstack
            X = vstack([A, G[np.where(keep)[0]]]).tocsr()
            Y = np.concatenate([yc[tr], g.labels[keep]])
            P = make_mlp(42).fit(X, Y).predict_proba(B)
            if ev is None:
                np.save(GDIR / "feats_val_mlp.npy", P)
            else:
                oof_mlp[ev] = P
                np.save(GDIR / "feats_core_oof_mlp.partial.npy", oof_mlp)
        print(name, "done", f"{time.time()-t0:.0f}s", flush=True)
    if what == "knn":
        np.save(GDIR / "feats_core_oof_knn.npy", oof)
    elif what == "e5lr":
        np.save(GDIR / "feats_core_oof_e5lr.npy", oof_mlp)
    else:
        np.save(GDIR / "feats_core_oof_mlp.npy", oof_mlp)


if __name__ == "__main__":
    main(sys.argv[1])
