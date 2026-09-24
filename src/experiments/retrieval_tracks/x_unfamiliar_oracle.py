"""Ceiling diagnostic: how predictable is the queue for CV-unfamiliar core tickets when the model is ALSO given
oracle metadata (type, priority, tags, and even the agent's answer text)? Diagnostic only; these inputs are
forbidden at prediction time. Core only."""
import json, numpy as np, scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression
from experiments.retrieval_tracks.x_unfamiliar_lib import *
from experiments.retrieval_tracks.x_unfamiliar_cv import folds, fold_sims
from experiments.retrieval_tracks.x_unfamiliar_meta import load_meta_rows

xc, yc, xv, yv = load_core_val()
core, _ = load_meta_rows()
sim, nnidx, fold = fold_sims(xc, yc); unf = sim < 0.4

def tagdoc(r):
    tags = []
    for t in r["tags"]:
        tags += [s.strip().lower().replace(" ", "_") for s in t.split(",") if s.strip()]
    return " ".join(["type=" + str(r["type"]).lower(), "prio=" + str(r["priority"]).lower()] + ["tag=" + t for t in tags])
meta_docs = [tagdoc(r) for r in core]
ans_docs = [(r["answer"] or "").lower() for r in core]

def run(name, builders, C=1.0):
    pred = np.zeros(len(xc), dtype=int)
    for tr, te in folds(yc):
        mats_tr, mats_te = [], []
        for docs, mk in builders:
            v = mk(); mats_tr.append(v.fit_transform([docs[i] for i in tr])); mats_te.append(v.transform([docs[i] for i in te]))
        clf = LogisticRegression(C=C, max_iter=400).fit(sp.hstack(mats_tr).tocsr(), yc[tr]); pred[te] = clf.predict(sp.hstack(mats_te).tocsr())
    a = (pred[unf] == yc[unf]).mean(); print(f"{name:40s} C={C:<4} unf={a:.4f} all={(pred==yc).mean():.4f}", flush=True)
    return a

cv_tok = lambda: CountVectorizer(token_pattern=r"\S+", binary=True)
tf = lambda: base_vectorizer()
res = {}
for C in (0.3, 1.0):
    res[f"meta_only_C{C}"] = run("metadata only (type,priority,tags)", [(meta_docs, cv_tok)], C)
res["text_only"] = run("text only", [(xc, tf)], 1.0)
res["text+meta"] = run("text + metadata", [(xc, tf), (meta_docs, cv_tok)], 1.0)
res["answer_only"] = run("answer text only", [(ans_docs, tf)], 1.0)
res["text+meta+answer"] = run("text + metadata + answer", [(xc, tf), (meta_docs, cv_tok), (ans_docs, tf)], 1.0)
json.dump(res, open("results/x_unfamiliar_oracle.json", "w"), indent=1)
