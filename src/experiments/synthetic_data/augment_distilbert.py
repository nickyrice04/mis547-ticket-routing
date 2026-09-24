"""The DistilBERT arm: contextual augmentation of real seed tickets.

DistilBERT is an encoder. It cannot write a ticket from nothing, because it has
no way to produce text left to right. What it was trained to do is fill in
blanks: hide a word and it predicts what belongs there from the words around it.

So this arm takes each real seed ticket, hides about a quarter of its words, and
lets DistilBERT choose replacements. The result is a variant of a real ticket,
not a new one. That makes it a different kind of experiment from Gemma and
Claude. Those invent tickets and risk drifting away from how this dataset reads.
This one stays close to real tickets and risks adding too little.

Output goes to data/synthetic/raw_distilbert.jsonl, unfiltered. The same
integrity filters as every other arm run afterwards in filter_synthetic.py.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer

MODEL = "distilbert-base-uncased"
PER_TARGET = 150
MASK_RATE = 0.25
TOP_K = 8


def augment(text, tok, model, rng, dev):
    enc = tok(text, truncation=True, max_length=256, return_tensors="pt")
    ids = enc["input_ids"][0].clone()
    # Only mask real word pieces that start a word, never punctuation or specials.
    cand = [i for i in range(1, len(ids) - 1)
            if tok.convert_ids_to_tokens(int(ids[i])).isalpha()
            and not tok.convert_ids_to_tokens(int(ids[i])).startswith("##")]
    if len(cand) < 4:
        return None
    chosen = rng.sample(cand, max(1, int(len(cand) * MASK_RATE)))
    original = {i: int(ids[i]) for i in chosen}
    for i in chosen:
        ids[i] = tok.mask_token_id
    with torch.no_grad():
        logits = model(input_ids=ids.unsqueeze(0).to(dev),
                       attention_mask=enc["attention_mask"].to(dev)).logits[0]
    for i in chosen:
        top = torch.topk(logits[i], TOP_K).indices.tolist()
        options = [t for t in top if t != original[i]
                   and tok.convert_ids_to_tokens(t).isalpha()]
        ids[i] = rng.choice(options) if options else original[i]
    return tok.decode(ids[1:-1], skip_special_tokens=True)


def main() -> None:
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForMaskedLM.from_pretrained(MODEL).to(dev).eval()
    rng = random.Random(42)

    rows = []
    for path in sorted(Path("data/synthetic/seeds").glob("*.json")):
        spec = json.loads(path.read_text())
        made = 0
        tries = 0
        while made < PER_TARGET and tries < PER_TARGET * 5:
            tries += 1
            seed = rng.choice(spec["seeds"])
            out = augment(seed, tok, model, rng, dev)
            if out:
                # DistilBERT returns one lowercase string, so the whole variant is
                # the body. The subject is empty, the same as a ticket with no
                # subject line, which the dataset has too.
                rows.append({"subject": "", "body": out, "queue": spec["target"],
                             "target_confused_with": spec["confused_with"], "seed": seed})
                made += 1
        print(f"  {spec['name']:62s} {made:4d} variants", flush=True)

    with open("data/synthetic/raw_distilbert.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"wrote {len(rows)} raw variants")


if __name__ == "__main__":
    main()
