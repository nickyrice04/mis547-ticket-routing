"""Stacking experiments INSIDE core only (cross-fitted base outputs -> meta-level K-fold). No validation data."""
import json, sys, time, warnings
import numpy as np
import lightgbm as lgb
from sklearn.metrics import log_loss
from experiments.retrieval_tracks.x_lexical_data import load_core_val
from experiments.retrieval_tracks.x_lexical_meta import flat_features, pair_features, N_CLASSES, PROB_KEYS
warnings.filterwarnings("ignore")
K = 5
xc, yc, _, _ = load_core_val()
folds = [dict(np.load(f"data/x_lexical/base_k{K}_f{f}.npz")) for f in range(K)]
prior = np.bincount(yc, minlength=N_CLASSES) / len(yc)

def report(name, P, y, s1, store):
    pred = P.argmax(1); ok = pred == y
    lo = s1 < 0.4
    store[name] = dict(acc=float(ok.mean()), low=float(ok[lo].mean()), high=float(ok[~lo].mean()),
                       logloss=float(log_loss(y, np.clip(P, 1e-6, 1) / np.clip(P, 1e-6, 1).sum(1, keepdims=True), labels=range(N_CLASSES))))
    print(f"{name:44s} acc {ok.mean():.4f} low {ok[lo].mean():.4f} high {ok[~lo].mean():.4f} ll {store[name]['logloss']:.4f}", flush=True)

# assemble per-ticket arrays in fold order
te_all = np.concatenate([o["te"] for o in folds]); y_all = yc[te_all]
fold_id = np.concatenate([np.full(len(o["te"]), f) for f, o in enumerate(folds)])
s1 = np.concatenate([o["knn_w12_val"][:, 0] for o in folds])
res = {}
# references
def onehot(l): return np.eye(N_CLASSES)[l]
nn_lab = np.concatenate([yc[o["tr"]][o["knn_w12_idx"][:, 0]] for o in folds])
for k in ("p_mlp", "p_lr1", "p_lr10", "p_cnb"):
    P = np.concatenate([o[k] for o in folds]); report(k, P, y_all, s1, res)
    for thr in (0.3, 0.4):
        H = np.where((s1 >= thr)[:, None], onehot(nn_lab), P); report(f"hybrid 1nn(w12)>={thr} else {k}", H, y_all, s1, res)
report("1nn w12", onehot(nn_lab), y_all, s1, res)

which = sys.argv[1] if len(sys.argv) > 1 else "flat,pair"
if "flat" in which:
    X = np.vstack([flat_features(o, yc[o["tr"]])[0] for o in folds])
    for params in (dict(num_leaves=7, learning_rate=0.05, n_estimators=300), dict(num_leaves=15, learning_rate=0.03, n_estimators=400)):
        P = np.zeros((len(y_all), N_CLASSES)); t0 = time.time()
        for f in range(K):
            m = lgb.LGBMClassifier(objective="multiclass", colsample_bytree=0.5, subsample=0.8, subsample_freq=1, min_child_samples=30,
                                   reg_lambda=5.0, n_jobs=3, verbose=-1, random_state=0, **params)
            m.fit(X[fold_id != f], y_all[fold_id != f]); P[fold_id == f] = m.predict_proba(X[fold_id == f])
        report(f"STACK flat lgb {params}", P, y_all, s1, res); print("   time", round(time.time() - t0))
if "pair" in which:
    Xp_f, names = [], None
    for o in folds:
        a, names = pair_features(o, yc[o["tr"]], prior); Xp_f.append(a)
    Xp = np.vstack(Xp_f); yp = (np.tile(np.arange(N_CLASSES), len(y_all)) == np.repeat(y_all, N_CLASSES)).astype(int)
    fp = np.repeat(fold_id, N_CLASSES)
    cat = [names.index("class_id")]
    for params in (dict(num_leaves=15, learning_rate=0.05, n_estimators=300), dict(num_leaves=31, learning_rate=0.03, n_estimators=500)):
        P = np.zeros((len(y_all), N_CLASSES)); t0 = time.time()
        for f in range(K):
            m = lgb.LGBMClassifier(objective="binary", colsample_bytree=0.5, subsample=0.8, subsample_freq=1, min_child_samples=50,
                                   reg_lambda=5.0, n_jobs=3, verbose=-1, random_state=0, **params)
            m.fit(Xp[fp != f], yp[fp != f], categorical_feature=cat)
            s = m.predict_proba(Xp[fp == f])[:, 1].reshape(-1, N_CLASSES)
            P[fold_id == f] = s / s.sum(1, keepdims=True)
        report(f"STACK pair lgb {params}", P, y_all, s1, res); print("   time", round(time.time() - t0))
json.dump(res, open(f"results/x_lexical_stack_cv_{which.replace(',', '_')}.json", "w"), indent=1)
