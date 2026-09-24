"""Diagnostic inside core only: is queue predictable from metadata / answer for tickets without a close family member?"""
import json
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from scipy.sparse import hstack
from experiments.retrieval_tracks.x_semantic_common import *

ctx, cy, vtx, vy = load_core_val()
meta = json.load(open(CACHE / "core_meta.json"))
docs_meta = [" ".join([f"type_{m['type']}", f"prio_{m['priority']}", f"ver_{m['version']}"] + ["tag_" + t.replace(" ", "_") for t in m["tags"]]) for m in meta]
docs_tags = [" ".join(["tag_" + t.replace(" ", "_") for t in m["tags"]]) for m in meta]
answers = [m["answer"].lower() for m in meta]
skf = StratifiedKFold(5, shuffle=True, random_state=0)
res = {k: np.zeros(len(ctx), bool) for k in ["meta", "tags", "answer", "text_lr", "nn"]}
simall = np.zeros(len(ctx))
for f, (tr, te) in enumerate(skf.split(ctx, cy)):
    vec = baseline_tfidf(); Xtr = vec.fit_transform([ctx[i] for i in tr]); Xte = vec.transform([ctx[i] for i in te])
    S = (Xte @ Xtr.T).toarray(); simall[te] = S.max(1); res["nn"][te] = cy[tr][S.argmax(1)] == cy[te]
    lr = LogisticRegression(C=10, max_iter=300).fit(Xtr, cy[tr]); res["text_lr"][te] = lr.predict(Xte) == cy[te]
    for name, docs in [("meta", docs_meta), ("tags", docs_tags)]:
        cv = CountVectorizer(token_pattern=r"\S+", binary=True); A = cv.fit_transform([docs[i] for i in tr]); B = cv.transform([docs[i] for i in te])
        lr = LogisticRegression(C=1, max_iter=300).fit(A, cy[tr]); res[name][te] = lr.predict(B) == cy[te]
    av = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True); A = av.fit_transform([answers[i] for i in tr]); B = av.transform([answers[i] for i in te])
    lr = LogisticRegression(C=10, max_iter=300).fit(A, cy[tr]); res["answer"][te] = lr.predict(B) == cy[te]
    print("fold", f, {k: round(float(v[te].mean()), 3) for k, v in res.items()}, flush=True)
print(fmt_table(bucket_table(simall, res["nn"], res)))
