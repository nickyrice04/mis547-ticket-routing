"""Map parquet metadata (type, priority, tags, answer, subject, body) onto core order. Validation rows
get mapped as well but their metadata is only ever used for descriptive ceiling analysis, never as model input."""
import json, pickle, os
import numpy as np
import pyarrow.parquet as pq
from common import clean, load_split
from experiments.retrieval_tracks.x_unfamiliar_lib import SPLIT

CACHE = "data/x_unfamiliar/meta_core_val.pkl"

def load_meta_rows():
    if os.path.exists(CACHE):
        return pickle.load(open(CACHE, "rb"))
    x, y = load_split("train")
    s = json.loads(SPLIT.read_text())
    pos_core = {x[i]: k for k, i in enumerate(s["core"])}
    pos_val = {x[i]: k for k, i in enumerate(s["validation"])}
    t = pq.read_table("data_tickets.parquet").to_pydict()
    core = [None] * len(s["core"]); val = [None] * len(s["validation"]); seen = set()
    for i in range(len(t["queue"])):
        if t["language"][i] != "en": continue
        text = clean(t["subject"][i], t["body"][i])
        if text in seen or not text: continue
        seen.add(text)
        row = {"subject": t["subject"][i] or "", "body": t["body"][i] or "", "answer": t["answer"][i] or "", "queue": t["queue"][i],
               "type": t["type"][i], "priority": t["priority"][i], "row": i,
               "tags": [t[f"tag_{k}"][i] for k in range(1, 9) if t[f"tag_{k}"][i]]}
        if text in pos_core: core[pos_core[text]] = row
        elif text in pos_val: val[pos_val[text]] = row
    pickle.dump((core, val), open(CACHE, "wb"))
    return core, val

if __name__ == "__main__":
    core, val = load_meta_rows()
    print(sum(r is None for r in core), sum(r is None for r in val))
    print(core[0])
