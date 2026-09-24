"""Multi-view sharp kernels for the kernel-ridge 'prototype network' (6-fold-style CV inside core, first 2 folds)."""
import time, warnings, sys, json
import numpy as np, scipy.linalg as sla
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedKFold
from experiments.retrieval_tracks.x_classifier_lib import BASE_VEC, bucket_table, fmt_buckets, load_core_val, log_cv
warnings.filterwarnings("ignore")
xc, yc, _, _ = load_core_val(); xc = np.array(xc, dtype=object)
splits = list(StratifiedKFold(n_splits=6, shuffle=True, random_state=1).split(xc, yc))[:2]
VIEWS = {
    "U": dict(ngram_range=(1, 1), min_df=2, sublinear_tf=True, stop_words="english"),
    "B": BASE_VEC,
    "C": dict(analyzer="char_wb", ngram_range=(3, 5), min_df=2, sublinear_tf=True),
}
def subj(t): return t.split(" . ", 1)[0]
def body(t): return t.split(" . ", 1)[1] if " . " in t else t
which = sys.argv[1].split(",")
res = {}
for fi, (tr, te) in enumerate(splits):
    ytr, yte = yc[tr], yc[te]
    K = {}
    for name, kw in VIEWS.items():
        v = TfidfVectorizer(**kw); a = v.fit_transform(xc[tr]); b = v.transform(xc[te])
        K[name] = ((a @ a.T).toarray().astype(np.float32), (b @ a.T).toarray().astype(np.float32))
    v = TfidfVectorizer(**VIEWS["U"]); a = v.fit_transform([subj(t) for t in xc[tr]]); b = v.transform([subj(t) for t in xc[te]])
    K["S"] = ((a @ a.T).toarray().astype(np.float32), (b @ a.T).toarray().astype(np.float32))
    v = TfidfVectorizer(**VIEWS["U"]); a = v.fit_transform([body(t) for t in xc[tr]]); b = v.transform([body(t) for t in xc[te]])
    K["D"] = ((a @ a.T).toarray().astype(np.float32), (b @ a.T).toarray().astype(np.float32))
    base_sims = K["B"][1].max(1)
    Y = -np.ones((len(ytr), 10)); Y[np.arange(len(ytr)), ytr] = 1
    combos = {
        "U^6": lambda k: k["U"] ** 6,
        "B^6": lambda k: k["B"] ** 6,
        "C^8": lambda k: k["C"] ** 8,
        "C^12": lambda k: k["C"] ** 12,
        "(U*C)^3": lambda k: (k["U"] * k["C"]) ** 3,
        "(U*B)^3": lambda k: (k["U"] * k["B"]) ** 3,
        "(U*B*C)^2": lambda k: (k["U"] * k["B"] * k["C"]) ** 2,
        "((U+B+C)/3)^6": lambda k: ((k["U"] + k["B"] + k["C"]) / 3) ** 6,
        "((U+B+C)/3)^8": lambda k: ((k["U"] + k["B"] + k["C"]) / 3) ** 8,
        "U^6+0.2U^2": lambda k: k["U"] ** 6 + 0.2 * k["U"] ** 2,
        "U^8+0.1U^2": lambda k: k["U"] ** 8 + 0.1 * k["U"] ** 2,
        "((U+D+0.5S)/2.5)^6": lambda k: ((k["U"] + k["D"] + 0.5 * k["S"]) / 2.5) ** 6,
        "(U^3)*(D^3)": lambda k: k["U"] ** 3 * k["D"] ** 3,
    }
    for cname, f in combos.items():
        if which != ["all"] and cname not in which: continue
        for lam in (0.01,):
            ktr = f({n: v[0] for n, v in K.items()}).astype(np.float64); ktr[np.diag_indices_from(ktr)] += lam
            alpha = sla.solve(ktr, Y, assume_a="pos")
            pred = (f({n: v[1] for n, v in K.items()}).astype(np.float64) @ alpha).argmax(1)
            res.setdefault(cname, []).append((pred, yte, base_sims)); print(fi, cname, round((pred == yte).mean(), 4), flush=True)
for cname, lst in res.items():
    P = np.concatenate([l[0] for l in lst]); Y_ = np.concatenate([l[1] for l in lst]); S = np.concatenate([l[2] for l in lst])
    log_cv(f"k5_{cname}", [(l[0] == l[1]).mean() for l in lst], {"buckets": fmt_buckets(bucket_table(S, P, Y_)), "cv": "6fold_seed1"})
