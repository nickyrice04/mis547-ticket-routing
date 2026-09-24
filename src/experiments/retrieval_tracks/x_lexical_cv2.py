"""CV inside core: combined similarity views (sum of cosines), LSA view, subject/body views."""
import json, time
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize
from experiments.retrieval_tracks.x_lexical_data import load_core_val
from experiments.retrieval_tracks.x_lexical_views import VIEWS, TfidfView, topk, vote

xc, yc, _, _ = load_core_val()
xa = np.array(xc, dtype=object)
def subj(t): return t.split(" . ", 1)[0] if " . " in t else ""
def body(t): return t.split(" . ", 1)[1] if " . " in t else t
skf = StratifiedKFold(5, shuffle=True, random_state=0)
res = {}
t0 = time.time()
for f, (tr, te) in enumerate(skf.split(xa, yc)):
    S = {}
    for name in ("w12", "c25wb", "bm25"):
        S[name] = VIEWS[name]().fit(list(xa[tr])).sims(list(xa[te]))
    v = TfidfView(ngram_range=(1, 2), min_df=2, max_features=200_000).fit(list(xa[tr]))
    svd = TruncatedSVD(400, random_state=0).fit(v.D)
    L_tr = normalize(svd.transform(v.D)); L_te = normalize(svd.transform(v.transform(list(xa[te]))))
    S["lsa400"] = L_te @ L_tr.T
    vb = TfidfView(ngram_range=(1, 2), min_df=2).fit([body(t) for t in xa[tr]])
    S["body"] = vb.sims([body(t) for t in xa[te]])
    vs = TfidfView(ngram_range=(1, 2), min_df=1).fit([subj(t) for t in xa[tr]])
    S["subj"] = vs.sims([subj(t) for t in xa[te]])
    combos = {
        "w12": S["w12"], "lsa400": S["lsa400"], "body": S["body"], "subj": S["subj"],
        "w12+c25": (S["w12"] + S["c25wb"]) / 2,
        "w12+c25+bm25": (S["w12"] + S["c25wb"] + S["bm25"]) / 3,
        "w12+lsa": (S["w12"] + S["lsa400"]) / 2,
        "w12+c25+lsa": (S["w12"] + S["c25wb"] + S["lsa400"]) / 3,
        "w12+body": (S["w12"] + S["body"]) / 2,
        "w12+.3subj": (S["w12"] + 0.3 * S["subj"]) / 1.3,
        "max(w12,body)": np.maximum(S["w12"], S["body"]),
    }
    for cname, M in combos.items():
        idx, val = topk(M, 20)
        for k, p in ((1, 1), (10, 4), (20, 4), (20, 8)):
            pred = vote(idx, val, yc[tr], 10, k, p).argmax(1)
            res.setdefault(f"{cname}|k{k}|p{p}", []).append(float((pred == yc[te]).mean()))
    print(f, round(time.time() - t0), flush=True)
out = {k: float(np.mean(v)) for k, v in res.items()}
for k, v in out.items(): print(f"{k:32s} {v:.4f}")
json.dump(out, open("results/x_lexical_cv2_combos.json", "w"), indent=1)
