"""CV inside core: which text model is best on tickets WITHOUT a close family member (w12 NN sim < 0.4)?"""
import json, time, warnings
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import ComplementNB, MultinomialNB
import scipy.sparse as sp
from experiments.retrieval_tracks.x_lexical_data import load_core_val
warnings.filterwarnings("ignore")
xc, yc, _, _ = load_core_val()
xa = np.array(xc, dtype=object)
skf = StratifiedKFold(5, shuffle=True, random_state=0)
res = {}
def rec(name, pred, yt, lo):
    ok = pred == yt
    res.setdefault(name, []).append((ok.mean(), ok[lo].mean(), ok[~lo].mean()))
t0 = time.time()
for f, (tr, te) in enumerate(skf.split(xa, yc)):
    z = np.load(f"data/x_lexical/cv_top50_w12_f{f}.npz"); lo = z["val"][:, 0] < 0.4
    ytr, yte = yc[tr], yc[te]
    feats = {
        "w12": TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=200_000, sublinear_tf=True),
        "w11": TfidfVectorizer(ngram_range=(1, 1), min_df=2, sublinear_tf=True),
        "w11bin": TfidfVectorizer(ngram_range=(1, 1), min_df=3, binary=True, use_idf=True),
        "c25wb": TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=3, max_features=300_000, sublinear_tf=True),
    }
    P = {}
    for fn, vec in feats.items():
        Xtr = vec.fit_transform(xa[tr]); Xte = vec.transform(xa[te])
        for C in (0.3, 1, 3):
            m = LogisticRegression(C=C, max_iter=2000).fit(Xtr, ytr)
            P[f"{fn}|LR{C}"] = m.predict_proba(Xte); rec(f"{fn}|LR C={C}", P[f"{fn}|LR{C}"].argmax(1), yte, lo)
        if fn in ("w12", "w11"):
            for C in (0.03, 0.1):
                m = LinearSVC(C=C).fit(Xtr, ytr); rec(f"{fn}|SVC C={C}", m.predict(Xte), yte, lo)
            for a in (0.1, 0.3, 1.0):
                m = ComplementNB(alpha=a).fit(Xtr, ytr); rec(f"{fn}|CNB a={a}", m.predict(Xte), yte, lo)
            m = LogisticRegression(C=1, max_iter=2000, class_weight="balanced").fit(Xtr, ytr); rec(f"{fn}|LR C=1 balanced", m.predict(Xte), yte, lo)
        print(f, fn, round(time.time() - t0), flush=True)
    rec("ens w12LR1+c25LR1", (P["w12|LR1"] + P["c25wb|LR1"]).argmax(1), yte, lo)
    rec("ens w12LR1+w11LR1+c25LR1", (P["w12|LR1"] + P["w11|LR1"] + P["c25wb|LR1"]).argmax(1), yte, lo)
out = {k: [float(x) for x in np.mean(v, 0)] for k, v in res.items()}
for k, v in sorted(out.items(), key=lambda kv: -kv[1][1]): print(f"{k:32s} all {v[0]:.4f} low {v[1]:.4f} high {v[2]:.4f}")
json.dump(out, open("results/x_lexical_cv3_singletons.json", "w"), indent=1)
