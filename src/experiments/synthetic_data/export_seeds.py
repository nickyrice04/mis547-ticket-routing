"""Write one seed file per target so every generator sees identical inputs.

The comparison is only fair if Gemma, DistilBERT and Claude all start from the
same seed tickets for the same targets. These are the exact pools the Gemma run
used: the twelve least confident mistakes for each confusion pair, and every
validation ticket of the queue for the weak queues.

Every seed comes from the validation slice of the training data. Nothing here
reads the test set, so a generator that only sees these files cannot see the
test tickets either.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

from common import load_meta, load_split
from experiments.synthetic_data.generate_synthetic import PER_PROMPT, WEAK_QUEUES, prompt_for

OUT = Path("data/synthetic/seeds")


def slug(text):
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


def targets():
    labels = load_meta()["labels"]
    report = json.loads(Path("data/synthetic/error_report.json").read_text())
    split = json.loads(Path("data/synthetic/split.json").read_text())
    x_tr, y_tr = load_split("train")
    y_tr = np.asarray(y_tr)
    out = [(p["true"], p["predicted"], p["seeds"]) for p in report["pairs"]]
    for q in WEAK_QUEUES:
        out.append((q, None, [x_tr[i][:600] for i in split["validation"] if labels[y_tr[i]] == q]))
    return out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for target, confused, seeds in targets():
        name = f"{slug(target)}__vs__{slug(confused)}" if confused else f"{slug(target)}__weak"
        instruction = prompt_for(target, seeds[:3], confused).replace(
            f"Write {PER_PROMPT} NEW tickets", "Write 150 NEW tickets")
        (OUT / f"{name}.json").write_text(json.dumps({
            "name": name, "target": target, "confused_with": confused,
            "seeds": seeds, "instruction": instruction}, indent=2))
        print(f"  {name:62s} {len(seeds):4d} seeds")


if __name__ == "__main__":
    main()
