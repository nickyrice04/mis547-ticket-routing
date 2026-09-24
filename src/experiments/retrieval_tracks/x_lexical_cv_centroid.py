"""In-core test: does a family-centroid similarity block add to the strict stack? (no validation data)"""
import json, time
import numpy as np, scipy.sparse as sp
from scipy.sparse.csgraph import connected_components
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import normalize
import experiments.retrieval_tracks.x_lexical_final as F
from experiments.retrieval_tracks.x_lexical_data import load_core_val

xc, yc, _, _ = load_core_val(); xa = np.array(xc, dtype=object)
z = np.load("data/x_lexical/pairs_core_none.npz"); X, names, fold_id = z["X"], list(z["names"]), z["fold_id"]
y_pair = (np.tile(np.arange(10), len(yc)) == np.repeat(yc, 10)).astype(int); fp = np.repeat(fold_id, 10)

def meta_cv(X, names, tag, params=None, seeds=(0,)):
    import lightgbm as lgb
    oof = np.zeros((len(yc), 10)); cat = [names.index("class_id")]
    p = dict(F.LGB_PARAMS); p.update(params or {})
    for f in range(5):
        s = 0
        for sd in seeds:
            m = lgb.LGBMClassifier(random_state=sd, **p).fit(X[fp != f], y_pair[fp != f], categorical_feature=cat)
            s = s + m.predict_proba(X[fp == f])[:, 1].reshape(-1, 10)
        oof[fold_id == f] = s / s.sum(1, keepdims=True)
    acc = float((oof.argmax(1) == yc).mean()); print(f"{tag:50s} acc {acc:.4f}", flush=True); return acc

res = {"strict": meta_cv(X, names, "strict (cached pairs)")}
for thr in (0.5, 0.65):
    blk = np.zeros((len(yc), 10)); blk_n = np.zeros((len(yc), 10))
    for f, (tr, te) in enumerate(StratifiedKFold(5, shuffle=True, random_state=0).split(xa, yc)):
        vec = F._base_vec(); P = vec.fit_transform(xa[tr]); E = vec.transform(xa[te])
        G = (P @ P.T).tocsr(); G.data[G.data < thr] = 0; G.eliminate_zeros()
        same = sp.csr_matrix(((yc[tr][G.nonzero()[0]] == yc[tr][G.nonzero()[1]]).astype(np.float32), G.nonzero()), shape=G.shape)
        n_comp, comp = connected_components(G.multiply(same), directed=False)
        M = sp.csr_matrix((np.ones(len(tr), dtype=np.float32), (comp, np.arange(len(tr)))), shape=(n_comp, len(tr)))
        C = normalize(M @ P); cy = np.zeros(n_comp, dtype=int); cy[comp] = yc[tr]; size = np.asarray(M.sum(1)).ravel()
        S = (E @ C.T).toarray()
        for c in range(10):
            cols = np.where(cy == c)[0]
            j = S[:, cols].argmax(1); blk[te, c] = S[np.arange(len(te)), cols[j]]; blk_n[te, c] = np.log1p(size[cols[j]])
        print("  fold", f, "families", n_comp, "largest", int(size.max()), flush=True)
    top = np.sort(blk, 1); other = np.where(blk >= top[:, [-1]], top[:, [-2]], top[:, [-1]])
    X2 = np.column_stack([X, blk.ravel(), (blk - other).ravel(), blk_n.ravel()]).astype(np.float32)
    res[f"centroid thr {thr}"] = meta_cv(X2, names + ["cen_maxsim", "cen_maxsim_gap", "cen_logsize"], f"+ family centroid block (link >= {thr})")
for tag, params, seeds in (("leaves31 lr.03 n600", dict(num_leaves=31, learning_rate=0.03, n_estimators=600), (0,)),
                           ("leaves7 lr.05 n400", dict(num_leaves=7, learning_rate=0.05, n_estimators=400), (0,)),
                           ("default, 3 seeds", None, (0, 1, 2)),
                           ("leaves15 lr.02 n800 mcs100", dict(learning_rate=0.02, n_estimators=800, min_child_samples=100), (0,))):
    res[tag] = meta_cv(X, names, tag, params, seeds)
json.dump(res, open("results/x_lexical_cv_centroid_meta.json", "w"), indent=1)
