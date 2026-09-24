"""Mine family pairs inside the encoder-training part of core (core CV folds 1-4; fold 0 stays unseen by the encoder).

A scenario family shares queue, type and priority (twins at cosine >= 0.8 share all three ~99.7% of the time), and
priority is close to independent of content, so a differing (queue,type,priority) triple proves "not the same family".
  positives      : same connected component of strong links (TF-IDF >= 0.45 and identical triple), or a direct link
                   with identical triple and TF-IDF >= 0.30
  hard negatives : the most similar tickets (TF-IDF or frozen mpnet) whose triple differs
Core only; type/priority are used purely as training signal."""
import json, pickle, numpy as np, scipy.sparse as sp
from scipy.sparse.csgraph import connected_components
from experiments.retrieval_tracks.x_unfamiliar_lib import *
from experiments.retrieval_tracks.x_unfamiliar_cv import folds
from experiments.retrieval_tracks.x_unfamiliar_meta import load_meta_rows
import experiments.retrieval_tracks.x_unfamiliar_final as F

xc, yc, _, _ = load_core_val(); core, _ = load_meta_rows()
tr, te = folds(yc)[0]                       # tr = encoder-training tickets, te = fold 0, never shown to the encoder
triple = np.array([hash((r["queue"], r["type"], r["priority"])) for r in core])[tr]
texts = [xc[i] for i in tr]
X = F.base_vectorizer().fit_transform(texts).tocsr(); E = np.load("data/x_unfamiliar/emb_mpnet_core.npy")[tr]
ts, ti = F._topk_sparse(X, X, 31); ds, di = F._topk_dense(E, E, 31)
n = len(tr); rows, cols = [], []
pos = [dict() for _ in range(n)]; neg = [dict() for _ in range(n)]
for i in range(n):
    for s, j in list(zip(ts[i], ti[i])) + [(None, j) for j in di[i]]:
        if j == i: continue
        if s is None: s = float(X[i].multiply(X[j]).sum())
        if triple[i] == triple[j]:
            if s >= 0.30: pos[i][int(j)] = float(s)
            if s >= 0.45: rows.append(i); cols.append(j)
        else:
            neg[i][int(j)] = float(s)
comp = connected_components(sp.coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n)), directed=False)[1]
members = {}
for i, c in enumerate(comp): members.setdefault(c, []).append(i)
added = 0
for c, m in members.items():
    if 2 <= len(m) <= 60:
        for a in m:
            for b in m:
                if a != b and b not in pos[a]:
                    pos[a][b] = float(X[a].multiply(X[b]).sum()); added += 1
has = sum(1 for p in pos if p)
allpos = np.array([s for p in pos for s in p.values()])
print("encoder-train tickets", n, "| anchors with positives", has, "| positive pairs", len(allpos), "| transitive-only pairs", added)
print("positive-pair TF-IDF sim quantiles", np.quantile(allpos, [0.05, 0.25, 0.5, 0.75, 0.95]).round(3), "| share < 0.4:", float((allpos < 0.4).mean()))
print("anchors with >=1 hard negative", sum(1 for q in neg if q))
pickle.dump({"train_idx": tr, "texts": texts, "pos": pos, "neg": neg, "seen_sha": sorted({F._sha(t) for t in texts})}, open("data/x_unfamiliar/family_pairs.pkl", "wb"))
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("sentence-transformers/all-mpnet-base-v2")
L = np.array([len(tok(t, truncation=False)["input_ids"]) for t in texts[:3000]]); print("token length quantiles", np.quantile(L, [.5, .75, .9, .95, .99]))
