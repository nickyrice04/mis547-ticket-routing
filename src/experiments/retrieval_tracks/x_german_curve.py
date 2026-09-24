"""Track "german": how the gain scales with the size of the German pool. Core-CV only, validation untouched."""
import json, time
import numpy as np
from sklearn.model_selection import StratifiedKFold
from experiments.retrieval_tracks.x_german_cvfeats import folds
from final.features import German, channels, e5_embed, guard_mask
from final.lib import baseline_vec, load_core_val
from final.stacker import fit_stacker, predict_stacker

xc, yc, _, _ = load_core_val()
Ec = e5_embed(xc)
full = German()
rng = np.random.default_rng(0)
order = rng.permutation(len(full.texts))
rows = []
for frac in (0.25, 0.5, 0.75, 1.0):
    idx = np.sort(order[: int(len(order) * frac)])
    g = German.__new__(German)
    g.texts = [full.texts[i] for i in idx]; g.labels = full.labels[idx]
    g.e5_trans = full.e5_trans[idx]; g.e5_orig = full.e5_orig[idx]
    F = np.zeros((len(xc), 10, 10), np.float32)
    for tr, ev in folds(yc):
        x_tr, x_ev = [xc[i] for i in tr], [xc[i] for i in ev]
        vec = baseline_vec().fit(x_tr + g.texts)
        A, B, G = vec.transform(x_tr), vec.transform(x_ev), vec.transform(g.texts)
        keep, _ = guard_mask(g, G, B, Ec[ev])
        F[ev] = channels(A, yc[tr], Ec[tr], B, Ec[ev], G, g, keep)
    acc = []
    for tr, te in StratifiedKFold(3, shuffle=True, random_state=1).split(F, yc):
        m = fit_stacker(F[tr], yc[tr])
        acc.append(float((predict_stacker(m, F[te]).argmax(1) == yc[te]).mean()))
    nn = float((np.maximum(F[:, :, 0], F[:, :, 2]).argmax(1) == yc).mean())
    rows.append({"german_fraction": frac, "german_tickets": len(idx), "stacker_core_cv_accuracy": round(np.mean(acc), 4),
                 "merged_tfidf_1nn_core_oof_accuracy": round(nn, 4)})
    print(rows[-1], flush=True)
json.dump(rows, open("results/x_german_pool_size_curve.json", "w"), indent=1)
