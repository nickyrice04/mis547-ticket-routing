"""Reviewer check for track "german": twin check in e5 space (where a translation twin would be closest).

For every validation ticket: nearest kept German ticket by e5 (translated and original channels) and nearest core
ticket by e5. Reports how often the neighbour has the identical (queue, priority, type, tag set) tuple, by similarity
band. If German rows were direct translations of validation tickets, identical-metadata rates at high similarity would
be far above the rates seen between English siblings.
"""
from __future__ import annotations

import json
import os

assert os.environ.get("X_GERMAN_E5_CACHE") != "1"
import numpy as np
import pyarrow.parquet as pq

from common import ROOT, clean
from final.features import German, dense_topk, e5_embed, guard_mask
from final.lib import baseline_vec, load_core_val

xc, yc, xv, yv = load_core_val()
g = German()
rows = [json.loads(l) for l in open(ROOT / "data" / "x_german" / "german_translated.jsonl")]
Ev, Ec = e5_embed(xv), e5_embed(xc)
vec = baseline_vec().fit(list(xc) + g.texts)
keep, info = guard_mask(g, vec.transform(g.texts), vec.transform(xv), Ev)
kept = np.where(keep)[0]
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

st, it = dense_topk(Ev, g.e5_trans[kept], 1)
so, io = dense_topk(Ev, g.e5_orig[kept], 1)
sc_, ic = dense_topk(Ev, Ec, 1)
res = {"guard": info}
for name, s, idx in (("german_translated", st[:, 0], kept[it[:, 0]]), ("german_original", so[:, 0], kept[io[:, 0]])):
    same = np.asarray([meta(first[x]) == meta(rows[j]["row"]) for x, j in zip(xv, idx)])
    sameq = g.labels[idx] == yv
    res[name] = {"identical_metadata_all": round(float(same.mean()), 4), "same_queue_all": round(float(sameq.mean()), 4),
                 "sim_quantiles": [round(float(q), 4) for q in np.quantile(s, [0.1, 0.5, 0.9, 0.99])]}
    for lo, hi in ((0.98, 1.01), (0.97, 0.98), (0.96, 0.97), (0.95, 0.96), (0.0, 0.95)):
        m = (s >= lo) & (s < hi)
        res[name][f"{lo}-{hi}"] = {"n": int(m.sum()), "identical_metadata": round(float(same[m].mean()), 4) if m.any() else None,
                                  "same_queue": round(float(sameq[m].mean()), 4) if m.any() else None}
same = np.asarray([meta(first[x]) == meta(first[xc[j]]) for x, j in zip(xv, ic[:, 0])])
sameq = yc[ic[:, 0]] == yv
res["core_english"] = {"identical_metadata_all": round(float(same.mean()), 4), "same_queue_all": round(float(sameq.mean()), 4),
                       "sim_quantiles": [round(float(q), 4) for q in np.quantile(sc_[:, 0], [0.1, 0.5, 0.9, 0.99])]}
for lo, hi in ((0.98, 1.01), (0.97, 0.98), (0.96, 0.97), (0.95, 0.96), (0.0, 0.95)):
    m = (sc_[:, 0] >= lo) & (sc_[:, 0] < hi)
    res["core_english"][f"{lo}-{hi}"] = {"n": int(m.sum()), "identical_metadata": round(float(same[m].mean()), 4) if m.any() else None,
                                         "same_queue": round(float(sameq[m].mean()), 4) if m.any() else None}

# where does the gain sit? tickets fixed relative to e5 1-NN on core
nn_core = yc[ic[:, 0]]
gain = (pred == yv) & (nn_core != yv)
best_g = np.maximum(st[:, 0], so[:, 0])
res["gain_vs_e5_nn_core"] = {"n": int(gain.sum()),
                             "best_german_e5_quantiles": [round(float(q), 4) for q in np.quantile(best_g[gain], [0.1, 0.5, 0.9])],
                             "share_with_german_e5>=0.98": round(float((best_g[gain] >= 0.98).mean()), 4),
                             "share_with_german_e5>=0.97": round(float((best_g[gain] >= 0.97).mean()), 4)}
print(json.dumps(res, indent=1))
(ROOT / "results" / "x_verify_german_twins_e5.json").write_text(json.dumps(res, indent=1))
