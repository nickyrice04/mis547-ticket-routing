"""6-fold-style CV: does a dense sentence-embedding view add to the sparse multi-view kernel machine? Also lambda / power grid."""
import time, warnings, sys
import numpy as np, scipy.linalg as sla
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedKFold
from experiments.retrieval_tracks.x_classifier_lib import BASE_VEC, bucket_table, fmt_buckets, load_core_val, log_cv
warnings.filterwarnings("ignore")
xc, yc, _, _ = load_core_val(); xc = np.array(xc, dtype=object)
E = np.load("data/x_classifier/emb_mpnet_core.npy")
splits = list(StratifiedKFold(n_splits=6, shuffle=True, random_state=1).split(xc, yc))[:2]
VIEWS = {"U": dict(ngram_range=(1, 1), min_df=2, sublinear_tf=True, stop_words="english"), "B": BASE_VEC,
         "C": dict(analyzer="char_wb", ngram_range=(3, 5), min_df=2, sublinear_tf=True)}
res = {}
for fi, (tr, te) in enumerate(splits):
    ytr, yte = yc[tr], yc[te]
    K = {}
    for name, kw in VIEWS.items():
        t0 = time.time()
        v = TfidfVectorizer(**kw); a = v.fit_transform(xc[tr]); b = v.transform(xc[te])
        ad = a.toarray() if name == "C" else None
        if ad is not None:
            K[name] = ((ad @ ad.T).astype(np.float32), (b @ a.T).toarray().astype(np.float32))
        else:
            K[name] = ((a @ a.T).toarray().astype(np.float32), (b @ a.T).toarray().astype(np.float32))
        print(name, f"{time.time()-t0:.0f}s", flush=True)
    K["E"] = (E[tr] @ E[tr].T, E[te] @ E[tr].T)
    base_sims = K["B"][1].max(1)
    Y = -np.ones((len(ytr), 10)); Y[np.arange(len(ytr)), ytr] = 1
    avg3 = lambda k: (k["U"] + k["B"] + k["C"]) / 3
    er = lambda k: np.clip((k["E"] - 0.4) / 0.6, 0, None)
    combos = {
        "avg3^6": lambda k: avg3(k) ** 6,
        "avg3^5": lambda k: avg3(k) ** 5,
        "avg3^4*E^8": lambda k: avg3(k) ** 4 * np.clip(k["E"], 0, None) ** 8,
        "avg3^3*E^12": lambda k: avg3(k) ** 3 * np.clip(k["E"], 0, None) ** 12,
        "(0.75avg3+0.25Er)^6": lambda k: (0.75 * avg3(k) + 0.25 * er(k)) ** 6,
        "(0.5avg3+0.5Er)^6": lambda k: (0.5 * avg3(k) + 0.5 * er(k)) ** 6,
    }
    for cname, f in combos.items():
        for lam in ((0.01, 0.1, 0.001) if cname == "avg3^6" else (0.01,)):
            ktr = f({n: v[0] for n, v in K.items()}).astype(np.float64); ktr[np.diag_indices_from(ktr)] += lam
            alpha = sla.solve(ktr, Y, assume_a="pos")
            pred = (f({n: v[1] for n, v in K.items()}).astype(np.float64) @ alpha).argmax(1)
            res.setdefault(f"{cname}_lam{lam}", []).append((pred, yte, base_sims)); print(fi, cname, lam, round((pred == yte).mean(), 4), flush=True)
for cname, lst in res.items():
    P = np.concatenate([l[0] for l in lst]); Y_ = np.concatenate([l[1] for l in lst]); S = np.concatenate([l[2] for l in lst])
    log_cv(f"k6_{cname}", [(l[0] == l[1]).mean() for l in lst], {"buckets": fmt_buckets(bucket_table(S, P, Y_)), "cv": "6fold_seed1"})
