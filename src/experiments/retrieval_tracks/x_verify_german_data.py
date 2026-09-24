"""Reviewer check for track "german": provenance of the cached German files and overlap with core / validation.

CPU only. Reads the parquet, data/x_german/*, and load_split("train") through load_core_val. The test split is
never opened. English parquet rows are read only to be matched against core / validation texts; rows that match
neither (i.e. test tickets and dropped duplicates) are discarded without being looked at.
"""
from __future__ import annotations

import json
from collections import Counter

import numpy as np
import pyarrow.parquet as pq

from common import ROOT, clean, load_meta
from final.lib import baseline_vec, load_core_val, topk_sims

G = ROOT / "data" / "x_german"
rows = [json.loads(l) for l in open(G / "german_translated.jsonl")]
raw = {r["row"]: r for r in map(json.loads, open(G / "german_raw.jsonl"))}
meta = load_meta()
t = pq.read_table(ROOT / "data_tickets.parquet").to_pydict()
out = {}

# 1. provenance: every German row is a parquet row with language == 'de', same queue, label = label2id[queue]
bad_lang = sum(t["language"][r["row"]] != "de" for r in rows)
bad_queue = sum(t["queue"][r["row"]] != r["queue"] for r in rows)
bad_label = sum(meta["label2id"][r["queue"]] != r["label"] for r in rows)
bad_raw = sum((raw[r["row"]]["subject"] != (t["subject"][r["row"]] or "")) or
              (raw[r["row"]]["body"] != (t["body"][r["row"]] or "")) for r in rows)
out["provenance"] = {"n": len(rows), "not_de": bad_lang, "queue_mismatch": bad_queue, "label_mismatch": bad_label,
                     "raw_text_mismatch": bad_raw, "unique_rows": len({r["row"] for r in rows}),
                     "languages_in_parquet": dict(Counter(t["language"])),
                     "versions_of_german_rows": dict(Counter(str(t["version"][r["row"]]) for r in rows))}

# 2. exact overlaps with core / validation
xc, yc, xv, yv = load_core_val()
gt = [r["text"] for r in rows]
go = [clean(raw[r["row"]]["subject"], raw[r["row"]]["body"]) for r in rows]
sc, sv = set(xc), set(xv)
out["exact_overlap"] = {"translated_in_core": sum(x in sc for x in gt), "translated_in_val": sum(x in sv for x in gt),
                        "original_in_core": sum(x in sc for x in go), "original_in_val": sum(x in sv for x in go)}

# 3. near overlaps: German (translated) vs validation and vs core with the baseline vectorizer fitted on core only
vec = baseline_vec().fit(xc)
A, B, Gm = vec.transform(xc), vec.transform(xv), vec.transform(gt)
g2v, g2v_i = topk_sims(Gm, B, 1)
g2c, _ = topk_sims(Gm, A, 1)
out["german_nearest_val_tfidf_corefit"] = {f">={th}": int((g2v[:, 0] >= th).sum()) for th in (0.9, 0.8, 0.7, 0.6, 0.5, 0.4)}
out["german_nearest_core_tfidf_corefit"] = {f">={th}": int((g2c[:, 0] >= th).sum()) for th in (0.9, 0.8, 0.7, 0.6, 0.5, 0.4)}
# same with the vectorizer the final module actually uses (fit on core + German)
vec2 = baseline_vec().fit(list(xc) + gt)
B2, G2 = vec2.transform(xv), vec2.transform(gt)
g2v2, _ = topk_sims(G2, B2, 1)
out["german_nearest_val_tfidf_finalfit"] = {f">={th}": int((g2v2[:, 0] >= th).sum()) for th in (0.9, 0.8, 0.7, 0.6, 0.5, 0.4)}
# label agreement of German tickets with their nearest validation ticket, by similarity band (twin check)
lab = np.asarray([r["label"] for r in rows])
bands = [(0.6, 1.01), (0.5, 0.6), (0.4, 0.5), (0.3, 0.4), (0.2, 0.3)]
out["german_vs_nearest_val_label_agreement"] = {
    f"{lo}-{hi}": {"n": int(((g2v[:, 0] >= lo) & (g2v[:, 0] < hi)).sum()),
                   "same_queue": round(float((lab == yv[g2v_i[:, 0]])[(g2v[:, 0] >= lo) & (g2v[:, 0] < hi)].mean()), 4)}
    for lo, hi in bands}

# 4. english_source_share distribution
ess = np.asarray([r["english_source_share"] for r in rows])
out["english_source"] = {"share>=0.9": int((ess >= 0.9).sum()), "0.1<share<0.9": int(((ess > 0.1) & (ess < 0.9)).sum()),
                         "share<=0.1": int((ess <= 0.1).sum())}

# 5. metadata twin check. Map validation / core texts back to parquet English rows (first occurrence, as build_splits does)
first = {}
for i in range(len(t["queue"])):
    if t["language"][i] != "en":
        continue
    k = clean(t["subject"][i], t["body"][i])
    if k and k not in first and (k in sv or k in sc):
        first[k] = i
tags = lambda i: tuple(t[f"tag_{k}"][i] for k in range(1, 9) if t[f"tag_{k}"][i])
answers_val = Counter((t["answer"][first[x]] or "").strip() for x in xv if x in first)
ans_g = [(t["answer"][r["row"]] or "").strip() for r in rows]
out["metadata"] = {"val_mapped": sum(x in first for x in xv), "core_mapped": sum(x in first for x in xc),
                   "german_answer_identical_to_a_val_answer": sum(bool(a) and a in answers_val for a in ans_g)}
# nearest German ticket (tf-idf, core-fit) for each validation ticket: identical metadata tuple?
v2g, v2g_i = topk_sims(B, Gm, 1)
v2c, v2c_i = topk_sims(B, A, 1)
def tup_p(i):
    return (t["queue"][i], t["priority"][i], t["type"][i], tags(i))
same_g, same_c, n = [], [], 0
for j, x in enumerate(xv):
    if x not in first:
        continue
    mv = tup_p(first[x])
    same_g.append(mv == tup_p(rows[v2g_i[j, 0]]["row"]))
    xcj = xc[v2c_i[j, 0]]
    same_c.append(xcj in first and mv == tup_p(first[xcj]))
same_g, same_c = np.asarray(same_g), np.asarray(same_c)
out["metadata"]["nearest_german_full_metadata_identical"] = round(float(same_g.mean()), 4)
out["metadata"]["nearest_core_full_metadata_identical"] = round(float(same_c.mean()), 4)

print(json.dumps(out, indent=1))
(ROOT / "results" / "x_verify_german_data.json").write_text(json.dumps(out, indent=1))
