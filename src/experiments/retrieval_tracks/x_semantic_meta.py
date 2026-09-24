"""Map parquet metadata onto CORE tickets only (training-side signal). Diagnostic: how predictable is queue from metadata
for tickets with no close family member, evaluated by CV strictly inside core."""
import json
import numpy as np
import pyarrow.parquet as pq
from common import clean
from experiments.retrieval_tracks.x_semantic_common import *

ctx, cy, vtx, vy = load_core_val()
want = {t: i for i, t in enumerate(ctx)}
t = pq.read_table(ROOT / "data_tickets.parquet").to_pydict()
meta = [None] * len(ctx)
for i in range(len(t["queue"])):
    if t["language"][i] != "en":
        continue
    text = clean(t["subject"][i], t["body"][i])
    j = want.get(text)
    if j is None or meta[j] is not None:
        continue
    meta[j] = {"type": t["type"][i], "priority": t["priority"][i], "version": t["version"][i],
               "tags": [t[f"tag_{k}"][i] for k in range(1, 9) if t[f"tag_{k}"][i] and t[f"tag_{k}"][i] != "None"],
               "answer": t["answer"][i] or "", "queue": t["queue"][i]}
print("mapped", sum(m is not None for m in meta), "of", len(ctx))
json.dump(meta, open(CACHE / "core_meta.json", "w"))

from collections import Counter
print("types", Counter(m["type"] for m in meta))
print("versions", Counter(m["version"] for m in meta).most_common(10))
print("n distinct tags", len(Counter(tg for m in meta for tg in m["tags"])))
print("top tags", Counter(tg for m in meta for tg in m["tags"]).most_common(25))
