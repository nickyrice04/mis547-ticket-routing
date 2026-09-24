"""Pairwise-stack ablations INSIDE core only (cross-fitted base outputs -> meta-level K-fold). No validation data."""
import json, sys, time, warnings
import numpy as np
import lightgbm as lgb
from sklearn.metrics import log_loss
from experiments.retrieval_tracks.x_lexical_data import load_core_val
from experiments.retrieval_tracks.x_lexical_meta import pair_features, N_CLASSES, PROB_KEYS, CAND_KEYS, AUX_CTX, VIEWS
warnings.filterwarnings("ignore")
K = 5
xc, yc, _, _ = load_core_val()
folds = []
for f in range(K):
    o = dict(np.load(f"data/x_lexical/base_k{K}_f{f}.npz")); o.update(dict(np.load(f"data/x_lexical/cand_k{K}_f{f}.npz"))); folds.append(o)
prior = np.bincount(yc, minlength=N_CLASSES) / len(yc)
te_all = np.concatenate([o["te"] for o in folds]); y_all = yc[te_all]
fold_id = np.concatenate([np.full(len(o["te"]), f) for f, o in enumerate(folds)])
s1 = np.concatenate([o["knn_w12_val"][:, 0] for o in folds])
yp = (np.tile(np.arange(N_CLASSES), len(y_all)) == np.repeat(y_all, N_CLASSES)).astype(int)
fp = np.repeat(fold_id, N_CLASSES)
res = {}

def run(name, lgb_params=None, seeds=(0,), **fk):
    Xp = []
    for o in folds:
        a, names = pair_features(o, yc[o["tr"]], prior, **fk); Xp.append(a)
    Xp = np.vstack(Xp); cat = [names.index("class_id")]
    params = dict(num_leaves=15, learning_rate=0.05, n_estimators=300, colsample_bytree=0.5, subsample=0.8, subsample_freq=1,
                  min_child_samples=50, reg_lambda=5.0)
    params.update(lgb_params or {})
    P = np.zeros((len(y_all), N_CLASSES)); t0 = time.time()
    for f in range(K):
        s = 0
        for sd in seeds:
            m = lgb.LGBMClassifier(objective="binary", n_jobs=3, verbose=-1, random_state=sd, **params)
            m.fit(Xp[fp != f], yp[fp != f], categorical_feature=cat)
            s = s + m.predict_proba(Xp[fp == f])[:, 1].reshape(-1, N_CLASSES)
        P[fold_id == f] = s / s.sum(1, keepdims=True)
    ok = P.argmax(1) == y_all; lo = s1 < 0.4
    res[name] = dict(acc=float(ok.mean()), low=float(ok[lo].mean()), high=float(ok[~lo].mean()), logloss=float(log_loss(y_all, P, labels=range(N_CLASSES))), n_feat=Xp.shape[1])
    print(f"{name:50s} acc {ok.mean():.4f} low {ok[lo].mean():.4f} high {ok[~lo].mean():.4f} ll {res[name]['logloss']:.4f} nfeat {Xp.shape[1]} t {time.time()-t0:.0f}s", flush=True)
    return P

CAND_NOMETA = tuple(k for k in CAND_KEYS if k not in ("cand_ptype", "cand_ppriority"))
run("base")
run("base + cand(no meta)", cand_keys=CAND_NOMETA)
run("base + cand(all)", cand_keys=CAND_KEYS)
run("base + cand(all) + auxctx", cand_keys=CAND_KEYS, aux_ctx=AUX_CTX)
run("no mlp + cand(all)", prob_keys=("p_lr1", "p_lr10", "p_cnb", "d_svc"), cand_keys=CAND_KEYS)
run("only lr1 + cand(all)", prob_keys=("p_lr1",), cand_keys=CAND_KEYS)
run("no text models + cand(all)", prob_keys=(), cand_keys=CAND_KEYS)
run("view wc only + cand(all)", views=("wc",), cand_keys=CAND_KEYS)
run("base + cand(all), no gaps", cand_keys=CAND_KEYS, gaps=False)
json.dump(res, open("results/x_lexical_stack_cv2.json", "w"), indent=1)
