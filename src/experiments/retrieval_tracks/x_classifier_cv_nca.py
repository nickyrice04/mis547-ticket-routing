"""6-fold-style CV (first 2 folds): does a learned diagonal term weighting (NCA) improve 1-NN and kernel ridge?"""
import sys, json, time, warnings
import numpy as np, scipy.sparse as sp, scipy.linalg as sla
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import normalize
from experiments.retrieval_tracks.x_classifier_lib import BASE_VEC, bucket_table, fmt_buckets, load_core_val, log_cv
from experiments.retrieval_tracks.x_classifier_nca import fit_term_weights
warnings.filterwarnings("ignore")
configs = json.loads(sys.argv[1])
xc, yc, _, _ = load_core_val(); xc = np.array(xc, dtype=object)
splits = list(StratifiedKFold(n_splits=6, shuffle=True, random_state=1).split(xc, yc))[:2]
for cfg in configs:
    out1, out2 = [], []
    for fi, (tr, te) in enumerate(splits):
        ytr, yte = yc[tr], yc[te]
        bv = TfidfVectorizer(**BASE_VEC); a0 = bv.fit_transform(xc[tr]); b0 = bv.transform(xc[te])
        sims = (b0 @ a0.T).max(axis=1).toarray().ravel()
        uv = TfidfVectorizer(ngram_range=(1, 1), min_df=2, sublinear_tf=True, stop_words="english" if cfg.get("stop", 1) else None)
        a = uv.fit_transform(xc[tr]); b = uv.transform(xc[te])
        w, gamma = fit_term_weights(a, ytr, verbose=True, **cfg.get("fit", {}))
        aw = normalize(a @ sp.diags(w)); bw = normalize(b @ sp.diags(w))
        Ktr = (aw @ aw.T).toarray().astype(np.float32); Kte = (bw @ aw.T).toarray().astype(np.float32)
        p1 = ytr[Kte.argmax(1)]; out1.append((p1, yte, sims))
        Y = -np.ones((len(ytr), 10)); Y[np.arange(len(ytr)), ytr] = 1
        A = (Ktr ** cfg.get("power", 6)).astype(np.float64); A[np.diag_indices_from(A)] += 0.01
        p2 = ((Kte ** cfg.get("power", 6)).astype(np.float64) @ sla.solve(A, Y, assume_a="pos")).argmax(1); out2.append((p2, yte, sims))
        print(f"fold {fi}: 1nn={(p1 == yte).mean():.4f} krr={(p2 == yte).mean():.4f} gamma={gamma:.1f}", flush=True)
        if fi == 0:
            voc = np.array(uv.get_feature_names_out()); o = np.argsort(w)
            print("  lowest-weight terms:", list(voc[o[:25]])); print("  highest-weight terms:", list(voc[o[-25:]]))
    for tag, out in (("1nn", out1), ("krr", out2)):
        P = np.concatenate([o[0] for o in out]); Y_ = np.concatenate([o[1] for o in out]); S = np.concatenate([o[2] for o in out])
        log_cv(f"nca_{cfg['name']}_{tag}", [(o[0] == o[1]).mean() for o in out], {"buckets": fmt_buckets(bucket_table(S, P, Y_)), "cfg": cfg, "cv": "6fold_seed1"})
