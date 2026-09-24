"""Fold-0 probe: dense sentence embeddings as the similarity (1-NN and sharp-kernel ridge), alone and mixed with TF-IDF."""
import sys, time, warnings
import numpy as np, scipy.linalg as sla
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedKFold
from experiments.retrieval_tracks.x_classifier_lib import BASE_VEC, bucket_table, fmt_buckets, load_core_val, log_cv
warnings.filterwarnings("ignore")
tag = sys.argv[1]
xc, yc, _, _ = load_core_val(); xc = np.array(xc, dtype=object)
E = np.load(f"data/x_classifier/emb_{tag}_core.npy")
tr, te = next(iter(StratifiedKFold(n_splits=3, shuffle=True, random_state=0).split(xc, yc)))
ytr, yte = yc[tr], yc[te]
bv = TfidfVectorizer(**BASE_VEC); a0 = bv.fit_transform(xc[tr]); b0 = bv.transform(xc[te])
Ttr = (a0 @ a0.T).toarray().astype(np.float32); Tte = (b0 @ a0.T).toarray().astype(np.float32)
base_sims = Tte.max(1)
Etr = E[tr] @ E[tr].T; Ete = E[te] @ E[tr].T
Y = -np.ones((len(ytr), 10)); Y[np.arange(len(ytr)), ytr] = 1
def rep(name, pred, t0=None):
    log_cv(name, [(pred == yte).mean()], {"buckets": fmt_buckets(bucket_table(base_sims, pred, yte)), "note": "fold0 only"})
def krr(name, Ktr, Kte, lam=0.01):
    A = Ktr.astype(np.float64); A[np.diag_indices_from(A)] += lam
    alpha = sla.solve(A, Y, assume_a="pos")
    rep(name, (Kte.astype(np.float64) @ alpha).argmax(1))
print("emb sim quantiles of nearest:", np.quantile(Ete.max(1), [0.1, 0.25, 0.5, 0.75, 0.9]))
rep(f"emb_{tag}_1nn", ytr[Ete.argmax(1)])
for w in (0.3, 0.5, 0.7):
    rep(f"emb_{tag}_mix{w}_1nn", ytr[(w * Ete + (1 - w) * Tte).argmax(1)])
for p in (8, 16, 32):
    krr(f"emb_{tag}_pow{p}", np.clip(Etr, 0, None) ** p, np.clip(Ete, 0, None) ** p)
for p, q in ((4, 16), (4, 32), (3, 16)):
    krr(f"emb_{tag}_tfidf{p}+emb{q}", Ttr ** p + np.clip(Etr, 0, None) ** q, Tte ** p + np.clip(Ete, 0, None) ** q)
    krr(f"emb_{tag}_tfidf{p}xemb{q//2}", Ttr ** p * np.clip(Etr, 0, None) ** (q // 2), Tte ** p * np.clip(Ete, 0, None) ** (q // 2))
