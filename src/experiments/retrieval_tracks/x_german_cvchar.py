"""Track "german": optional extra retrieval channels (character n-gram TF-IDF), out-of-fold inside core + validation."""
from __future__ import annotations

import sys
import time

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from experiments.retrieval_tracks.x_german_cvfeats import folds
from final.features import DIAG as GDIR, TOPK, German, e5_embed, guard_mask, per_class
from final.lib import baseline_vec, load_core_val, topk_sims


def main():
    xc, yc, xv, yv = load_core_val()
    g = German()
    Ec, Ev = e5_embed(xc), e5_embed(xv)
    t0 = time.time()
    jobs = [(f"fold{k}", tr, ev) for k, (tr, ev) in enumerate(folds(yc))] + [("val", np.arange(len(xc)), None)]
    oof = np.zeros((len(xc), 10, 4), np.float32)
    for name, tr, ev in jobs:
        x_tr = [xc[i] for i in tr]
        x_ev = xv if ev is None else [xc[i] for i in ev]
        E_ev = Ev if ev is None else Ec[ev]
        vec = baseline_vec().fit(x_tr + g.texts)
        keep, info = guard_mask(g, vec.transform(g.texts), vec.transform(x_ev), E_ev)
        cv = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=3, max_features=300_000,
                             sublinear_tf=True, dtype=np.float32).fit(x_tr + g.texts)
        A, B, G = cv.transform(x_tr), cv.transform(x_ev), cv.transform(g.texts)
        D, gy = G[np.where(keep)[0]], g.labels[keep]
        s, i = topk_sims(B, A, TOPK, block=500); f1 = per_class(s, yc[tr][i])
        s, i = topk_sims(B, D, TOPK, block=500); f2 = per_class(s, gy[i])
        F = np.concatenate([f1, f2], axis=2)
        if ev is None:
            np.save(GDIR / "feats_val_char.npy", F)
        else:
            oof[ev] = F
        print(name, "done", f"{time.time()-t0:.0f}s", flush=True)
    np.save(GDIR / "feats_core_oof_char.npy", oof)


if __name__ == "__main__":
    main()
