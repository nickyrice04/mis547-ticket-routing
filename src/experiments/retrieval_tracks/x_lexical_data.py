"""Track 'lexical': data helpers. Core/validation texts plus raw parquet metadata for CORE rows only
(metadata is training-side signal; never used as a prediction-time input)."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from common import clean, load_meta, load_split

ROOT = Path(__file__).resolve().parents[3]   # retrieval_tracks -> experiments -> src -> repository root
SPLIT = ROOT / "data/synthetic/split.json"


def load_core_val():
    x, y = load_split("train")
    s = json.loads(SPLIT.read_text())
    xc = [x[i] for i in s["core"]]; yc = np.array([y[i] for i in s["core"]])
    xv = [x[i] for i in s["validation"]]; yv = np.array([y[i] for i in s["validation"]])
    return xc, yc, xv, yv


def parquet_meta_by_text():
    """Map cleaned text -> dict(type, priority, tags, answer, subject, body) for all English rows.
    Callers must only look up TRAINING texts in it."""
    import pyarrow.parquet as pq
    t = pq.read_table(ROOT / "data_tickets.parquet").to_pydict()
    out = {}
    for i in range(len(t["queue"])):
        if t["language"][i] != "en":
            continue
        text = clean(t["subject"][i], t["body"][i])
        if not text or text in out:
            continue
        out[text] = {"type": t["type"][i], "priority": t["priority"][i],
                     "tags": [t[f"tag_{k}"][i] for k in range(1, 9) if t[f"tag_{k}"][i]],
                     "answer": t["answer"][i] or "", "subject": t["subject"][i] or "", "body": t["body"][i] or "",
                     "queue": t["queue"][i]}
    return out
