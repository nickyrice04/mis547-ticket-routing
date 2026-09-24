"""Ceiling evidence on CV-unfamiliar core tickets: confusion structure, top-k, calibration, learning curve, style features."""
import json, re, numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold
from experiments.retrieval_tracks.x_unfamiliar_lib import *
from experiments.retrieval_tracks.x_unfamiliar_cv import folds, fold_sims

xc, yc, xv, yv = load_core_val()
sim, nnidx, fold = fold_sims(xc, yc); unf = sim < 0.4
P = np.load("data/x_unfamiliar/cv_oof_battery3.npz")
p = np.exp(np.mean([np.log(P[n] + 1e-9) for n in ("lr_C1.0", "bge_lr_C1.0", "mpnet_lr_C1.0")], axis=0)); p /= p.sum(1, keepdims=True)
res = {}
y, pu = yc[unf], p[unf]; pred = pu.argmax(1)
order = np.argsort(-pu, 1)
res["top1"], res["top2"], res["top3"] = [float(np.mean([(y[i] in order[i, :k]) for i in range(len(y))])) for k in (1, 2, 3)]
prior = np.bincount(yc, minlength=10) / len(yc); po = np.argsort(-prior)
res["prior_top1"], res["prior_top2"], res["prior_top3"] = [float(np.isin(y, po[:k]).mean()) for k in (1, 2, 3)]
res["mean_max_prob"] = float(pu.max(1).mean()); res["mean_sum_p2"] = float((pu ** 2).sum(1).mean())
res["nll_model"] = float(-np.log(pu[np.arange(len(y)), y]).mean()); res["nll_prior"] = float(-np.log(prior[y]).mean())
print(json.dumps(res, indent=1))
print("per-class recall / precision / support on CV-unfamiliar (ensemble LR):")
for k, name in enumerate(LABELS):
    rec = (pred[y == k] == k).mean(); prec = (y[pred == k] == k).mean() if (pred == k).any() else float("nan")
    print(f"  {name:32s} recall={rec:.3f} precision={prec:.3f} n={int((y==k).sum())} predicted={int((pred==k).sum())}")
    res[f"recall_{name}"] = float(rec); res[f"precision_{name}"] = float(prec)
# confidence slices: is there a confidently-predictable subset?
conf = pu.max(1)
for lo, hi in ((0, .3), (.3, .4), (.4, .5), (.5, .7), (.7, 1.01)):
    m = (conf >= lo) & (conf < hi); print(f"  conf {lo:.1f}-{hi:.1f}: n={m.sum():5d} acc={(pred[m]==y[m]).mean():.3f}")
    res[f"conf_{lo}-{hi}"] = [int(m.sum()), float((pred[m] == y[m]).mean())]
# accuracy excluding Billing (the one content-determined queue)
nb = y != LABELS.index("Billing and Payments"); res["acc_excl_billing_true"] = float((pred[nb] == y[nb]).mean())
print("acc on unfamiliar excluding true-Billing tickets:", res["acc_excl_billing_true"], " majority there:", float((y[nb] == po[0]).mean()))

# learning curve on STRICT singletons (leave-one-out sim to ALL of core < 0.4), so the eval set is fixed across train sizes
loo = np.load("data/x_unfamiliar/sim_core_loo.npy"); strict = loo < 0.4
E = np.load("data/x_unfamiliar/emb_mpnet_core.npy")
rng = np.random.RandomState(0); lc = {}
for frac in (0.1, 0.2, 0.4, 0.7, 1.0):
    c_t = c_e = n = 0
    for tr, te in folds(yc):
        sub = tr if frac == 1.0 else rng.choice(tr, int(frac * len(tr)), replace=False)
        ev = te[strict[te]]
        vec = base_vectorizer(); Xtr = vec.fit_transform([xc[i] for i in sub]); Xev = vec.transform([xc[i] for i in ev])
        c_t += (LogisticRegression(C=1, max_iter=300).fit(Xtr, yc[sub]).predict(Xev) == yc[ev]).sum()
        c_e += (LogisticRegression(C=1, max_iter=500).fit(E[sub], yc[sub]).predict(E[ev]) == yc[ev]).sum(); n += len(ev)
    lc[frac] = {"n_train": int(frac * len(tr)), "tfidf_lr": c_t / n, "mpnet_lr": c_e / n, "n_eval": n}
    print("learning curve", frac, lc[frac], flush=True)
res["learning_curve_strict_singletons"] = lc

# style-only features (no topical words): can generation artefacts predict the queue?
def style(t):
    subj, _, body = t.partition(" . ")
    w = t.split()
    return [len(t), len(w), len(subj), float(len(subj) == 0), t.count("<br"), t.count("\\n"), t.count(","), t.count("."), t.count("?"),
            sum(ch.isdigit() for ch in t), float(t.startswith("dear")), float("dear customer support" in t), float("hello" in t[:80]),
            float("hi " in t[:40]), float("thank" in t[-200:]), float("regards" in t[-200:]), np.mean([len(x) for x in w]) if w else 0,
            float(len(t) >= 1999), t.count("!"), t.count(":"), t.count("-"), t.count("(")]
S = np.array([style(t) for t in xc]); ps = np.zeros(len(yc), int)
for tr, te in folds(yc):
    ps[te] = HistGradientBoostingClassifier(max_iter=150, learning_rate=0.08, random_state=0).fit(S[tr], yc[tr]).predict(S[te])
res["style_only_gbm_unf"] = float((ps[unf] == yc[unf]).mean()); res["majority_unf"] = float((yc[unf] == po[0]).mean())
print("style-only GBM on unfamiliar:", res["style_only_gbm_unf"], "majority:", res["majority_unf"])
json.dump(res, open("results/x_unfamiliar_ceiling.json", "w"), indent=1)
