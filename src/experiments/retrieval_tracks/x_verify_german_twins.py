"""Reviewer check for track "german": is the gain driven by translation twins of validation tickets?

CPU only. Uses my own reproduction probabilities (results/x_verify_german_repro_valprobs.npy), core, validation,
the German files and the parquet metadata. English parquet rows are only matched against core / validation texts.
"""
from __future__ import annotations

import json

import numpy as np
import pyarrow.parquet as pq

from common import ROOT, clean
from final.lib import baseline_vec, load_core_val, topk_sims

G = ROOT / "data" / "x_german"
rows = [json.loads(l) for l in open(G / "german_translated.jsonl")]
gt = [r["text"] for r in rows]
gl = np.asarray([r["label"] for r in rows])
xc, yc, xv, yv = load_core_val()
P = np.load(ROOT / "results" / "x_verify_german_repro_valprobs.npy")
pred = P.argmax(1)
t = pq.read_table(ROOT / "data_tickets.parquet").to_pydict()
sv, sc = set(xv), set(xc)
first = {}
for i in range(len(t["queue"])):
    if t["language"][i] != "en":
        continue
    k = clean(t["subject"][i], t["body"][i])
    if k and k not in first and (k in sv or k in sc):
        first[k] = i
tags = lambda i: frozenset(t[f"tag_{k}"][i] for k in range(1, 9) if t[f"tag_{k}"][i])
meta = lambda i: (t["queue"][i], t["priority"][i], t["type"][i], tags(i))

# the vectorizer the final module uses, and its TF-IDF guard
vec = baseline_vec().fit(list(xc) + gt)
A, B, Gm = vec.transform(xc), vec.transform(xv), vec.transform(gt)
g2v = topk_sims(Gm, B, 1)[0][:, 0]
keep = g2v < 0.60                       # (the e5 part of the guard is not reproduced here; it only drops more)
kept = np.where(keep)[0]
s_g, i_g = topk_sims(B, Gm[kept], 1)
s_c, i_c = topk_sims(B, A, 1)
nn_core = yc[i_c[:, 0]]
near_g = kept[i_g[:, 0]]
same_meta_g = np.asarray([meta(first[x]) == meta(rows[j]["row"]) for x, j in zip(xv, near_g)])
same_meta_c = np.asarray([meta(first[x]) == meta(first[xc[j]]) for x, j in zip(xv, i_c[:, 0])])
same_q_g = gl[near_g] == yv

gain = (pred == yv) & (nn_core != yv)          # tickets the final method fixes relative to English 1-NN
loss = (pred != yv) & (nn_core == yv)
out = {
    "acc_final": round(float((pred == yv).mean()), 4), "acc_nn_core": round(float((nn_core == yv).mean()), 4),
    "n_gain": int(gain.sum()), "n_loss": int(loss.sum()),
    "nearest_german_identical_metadata_all_val": round(float(same_meta_g.mean()), 4),
    "nearest_german_identical_metadata_on_gain": round(float(same_meta_g[gain].mean()), 4),
    "nearest_core_identical_metadata_all_val": round(float(same_meta_c.mean()), 4),
    "nearest_german_tfidf_sim_on_gain_quantiles": [round(float(q), 3) for q in np.quantile(s_g[gain, 0], [0.1, 0.25, 0.5, 0.75, 0.9])],
    "nearest_core_tfidf_sim_on_gain_quantiles": [round(float(q), 3) for q in np.quantile(s_c[gain, 0], [0.1, 0.25, 0.5, 0.75, 0.9])],
    "gain_with_german_nn_sim>=0.5": int((s_g[gain, 0] >= 0.5).sum()),
    "gain_with_german_nn_sim>=0.4": int((s_g[gain, 0] >= 0.4).sum()),
    "gain_with_german_nn_sim<0.3": int((s_g[gain, 0] < 0.3).sum()),
}
# core-fit 0.6 guard versus the guard the final module applies
vec0 = baseline_vec().fit(xc)
g2v0 = topk_sims(vec0.transform(gt), vec0.transform(xv), 1)[0][:, 0]
out["german>=0.6_to_val_corefit_vectorizer"] = int((g2v0 >= 0.6).sum())
out["of_those_not_dropped_by_final_tfidf_guard"] = int(((g2v0 >= 0.6) & keep).sum())
out["of_those_with_finalfit_sim>=0.5"] = int(((g2v0 >= 0.6) & (g2v >= 0.5)).sum())
print(json.dumps(out, indent=1))
(ROOT / "results" / "x_verify_german_twins.json").write_text(json.dumps(out, indent=1))
