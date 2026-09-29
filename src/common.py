"""Shared data loading and preprocessing.

The same cleaning code runs at training time and at inference time. That is the
point the midterm report makes about training/serving skew, so every model tier
in this study imports clean() from here instead of writing its own version.
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data_tickets.parquet"
# In the container the code sits at /app, not in a checkout, so the split
# directory is set by environment variable there.
SPLITS = Path(os.environ.get("SPLITS_DIR", ROOT / "splits"))
RESULTS = ROOT / "results"

SEED = 42
TEST_SIZE = 0.2
MAX_CHARS = 2000

_URL = re.compile(r"https?://\S+")
_EMAIL = re.compile(r"\S+@\S+")
_TICKET_ID = re.compile(r"\b[A-Z]{2,}-?\d{3,}\b")
_WS = re.compile(r"\s+")


def clean(subject: str | None, body: str | None) -> str:
    """Join subject and body, then strip the parts a model should not key on."""
    text = f"{subject or ''} . {body or ''}".lower()
    text = _URL.sub(" ", text)
    text = _EMAIL.sub(" ", text)
    # Known quirk, kept on purpose: the text is lowercased above, and this pattern only
    # matches uppercase ids, so ticket ids are NOT actually removed. It affects 39 of the
    # 18,997 training tickets. Fixing it would change every model's input and invalidate
    # the recorded results, so it stays as trained. tests/test_common.py pins it.
    text = _TICKET_ID.sub(" ", text)
    text = _WS.sub(" ", text).strip()
    return text[:MAX_CHARS]


def build_splits() -> dict:
    """Read the raw parquet, keep English rows, and write a fixed train/test split."""
    import pyarrow.parquet as pq
    from sklearn.model_selection import train_test_split

    table = pq.read_table(DATA).to_pydict()
    rows = [
        (clean(table["subject"][i], table["body"][i]), table["queue"][i], table["priority"][i])
        for i in range(len(table["queue"]))
        if table["language"][i] == "en"
    ]
    rows = [r for r in rows if r[0]]

    # Drop exact duplicate tickets before splitting.
    #
    # This dataset repeats the same ticket text many times. With a plain random
    # split, 26% of the test tickets had an identical twin in the training set,
    # and the baseline scored 94% on those against 62% on genuinely unseen
    # tickets. That gap is memorization, not routing skill, and it flattered the
    # 70.5% figure reported at the midterm. Keeping one copy of each unique
    # ticket makes the test set mean what we say it means.
    seen = set()
    deduped = []
    for r in rows:
        if r[0] not in seen:
            seen.add(r[0])
            deduped.append(r)
    print(f"deduplicated {len(rows)} rows down to {len(deduped)} unique tickets")
    rows = deduped

    texts = [r[0] for r in rows]
    queues = [r[1] for r in rows]
    priorities = [r[2] for r in rows]

    idx_tr, idx_te = train_test_split(
        range(len(texts)), test_size=TEST_SIZE, random_state=SEED, stratify=queues
    )
    labels = sorted(set(queues))
    label2id = {name: i for i, name in enumerate(labels)}

    SPLITS.mkdir(exist_ok=True)
    for name, idx in (("train", idx_tr), ("test", idx_te)):
        payload = [
            {"text": texts[i], "queue": queues[i], "label": label2id[queues[i]], "priority": priorities[i]}
            for i in idx
        ]
        (SPLITS / f"{name}.json").write_text(json.dumps(payload))

    meta = {"labels": labels, "label2id": label2id, "n_train": len(idx_tr), "n_test": len(idx_te)}
    (SPLITS / "meta.json").write_text(json.dumps(meta, indent=2))
    return meta


def load_split(name: str):
    """Texts and queue ids of one split, train or test, as written by build_splits()."""
    rows = json.loads((SPLITS / f"{name}.json").read_text())
    return [r["text"] for r in rows], [r["label"] for r in rows]


def load_meta() -> dict:
    """The label list and split sizes written by build_splits()."""
    return json.loads((SPLITS / "meta.json").read_text())


def save_result(payload: dict) -> None:
    """Write one model tier's result JSON into results/."""
    RESULTS.mkdir(exist_ok=True)
    path = RESULTS / f"{payload['tier']}.json"
    path.write_text(json.dumps(payload, indent=2))
    print(f"wrote {path}")


def threshold_table(probs, y_true, thresholds=(0.0, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95)):
    """Coverage and accuracy of auto-routed tickets at each confidence cut.

    Professor Zara asked how the model decides confidence and whether we set a
    threshold. This is the evidence for that answer: at each cut we report the
    share of tickets the model would route on its own and how often it is right.
    """
    import numpy as np

    probs = np.asarray(probs)
    y_true = np.asarray(y_true)
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    out = []
    for t in thresholds:
        keep = conf >= t
        n = int(keep.sum())
        out.append(
            {
                "threshold": t,
                "coverage": round(n / len(y_true), 4),
                "accuracy_on_routed": round(float((pred[keep] == y_true[keep]).mean()), 4) if n else None,
                "sent_to_human": round(1 - n / len(y_true), 4),
            }
        )
    return out


if __name__ == "__main__":
    print(json.dumps(build_splits(), indent=2))
