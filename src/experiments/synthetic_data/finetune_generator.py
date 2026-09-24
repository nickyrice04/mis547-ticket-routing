"""Fine-tune Qwen3.5 0.8B to write tickets that read like the real ones.

Every earlier generator was only prompted to imitate the dataset, and every one
missed its style. Real tickets have a median similarity of 0.545 to their
nearest training ticket, and synthetic ones sat between 0.14 and 0.25. This
script trains the style in instead of asking for it. A LoRA adapter learns to
write a real training ticket when given that ticket's queue, type, priority and
tags.

Generation then reuses the tag sets of real core tickets from the same queue as
prompts. So the synthetic tickets cover the same mix of topics the real ones
do, but the text is new.

Integrity, same rules as every earlier run
    * the generator trains only on the core of the training set. Its validation
      loss uses the validation slice, which is also training data
    * the test set is read once, at the end, only to drop anything too close to it
    * near-copies of training tickets and of each other are dropped. A generator
      fine-tuned on real tickets can memorize them, so this filter matters more
      here than before, and its drop rate is reported
    * the output goes to a separate file, and the real data is never touched

    python src/experiments/synthetic_data/finetune_generator.py data       # write the LoRA training files
    mlx_lm lora -c configs/lora_generator.yaml          # train the adapter
    TEMP=0.7 python src/experiments/synthetic_data/finetune_generator.py generate   # write tickets, resuming if stopped
    TEMP=0.7 python src/experiments/synthetic_data/finetune_generator.py filter     # integrity filters, final file
"""
from __future__ import annotations

import collections
import json
import os
import random
import re
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

from common import clean, load_meta, load_split
from experiments.synthetic_data.generate_synthetic import TEST_SIM_MAX, TRAIN_SIM_MAX

OUT = Path("data/synthetic")
LORA_DATA = Path("data/lora_generator")
ADAPTER = "adapters/qwen35_generator"
MODEL = "mlx-community/Qwen3.5-0.8B-MLX-8bit"
TEMP = float(os.environ.get("TEMP", 1.0))   # sampling temperature, one arm per value
TAG = f"qwen_sft_t{int(TEMP * 10):02d}"
RAW = OUT / f"raw_{TAG}.jsonl"
FINAL = OUT / f"{TAG}.jsonl"
BATCH = 64


def raw_core_rows():
    """The core training tickets with their original subject, body, type, priority and tags."""
    import pyarrow.parquet as pq

    x_tr, y_tr = load_split("train")
    split = json.loads((OUT / "split.json").read_text())
    want = {x_tr[i]: i for i in split["core"]}
    val_want = {x_tr[i]: i for i in split["validation"]}
    t = pq.read_table("data_tickets.parquet").to_pydict()
    core, val, seen = [], [], set()
    for i in range(len(t["queue"])):
        if t["language"][i] != "en":
            continue
        text = clean(t["subject"][i], t["body"][i])
        if text in seen or not text:
            continue
        seen.add(text)
        row = {"subject": t["subject"][i] or "", "body": t["body"][i] or "", "queue": t["queue"][i],
               "type": t["type"][i], "priority": t["priority"][i],
               "tags": [t[f"tag_{k}"][i] for k in range(1, 9) if t[f"tag_{k}"][i]]}
        if text in want:
            core.append(row)
        elif text in val_want:
            val.append(row)
    return core, val


def prompt_of(r):
    return (f'Write a customer support ticket for the "{r["queue"]}" queue.\n'
            f'Type: {r["type"]}. Priority: {r["priority"]}. Tags: {", ".join(r["tags"])}.')


def completion_of(r):
    return f'SUBJECT: {r["subject"]}\nBODY: {r["body"]}'


def parse(text):
    m = re.search(r"SUBJECT:(.*?)\nBODY:(.*)", text, re.S)
    if not m:
        return None
    subject, body = m.group(1).strip(), m.group(2).strip()
    return (subject, body) if 40 <= len(body) <= 3000 else None


def make_data():
    core, val = raw_core_rows()
    print(f"core tickets {len(core)}, validation tickets {len(val)}")
    LORA_DATA.mkdir(parents=True, exist_ok=True)
    rng = random.Random(42)
    rng.shuffle(core)
    for name, rows in (("train", core), ("valid", val[:400])):
        with open(LORA_DATA / f"{name}.jsonl", "w") as f:
            for r in rows:
                f.write(json.dumps({"prompt": prompt_of(r), "completion": completion_of(r)}) + "\n")


def generate_all():
    from mlx_lm import batch_generate, load
    from mlx_lm.sample_utils import make_logits_processors, make_sampler

    alloc = json.loads((OUT / "qwen_think_allocation.json").read_text())  # same mix as the 10k run
    core, _ = raw_core_rows()
    by_q = collections.defaultdict(list)
    for r in core:
        by_q[r["queue"]].append(r)

    have = collections.Counter()
    if RAW.exists():
        for line in open(RAW):
            have[json.loads(line)["queue"]] += 1
    want = {q: int(n * 1.15) for q, n in alloc.items()}   # extra, since filters drop some
    rng = random.Random(7 + sum(have.values()))
    jobs = []
    for q, n in want.items():
        jobs += [q] * max(0, n - have[q])
    rng.shuffle(jobs)
    print(f"{sum(have.values())} on disk, {len(jobs)} to write", flush=True)

    model, tok = load(MODEL, adapter_path=ADAPTER)
    sampler = make_sampler(temp=TEMP, top_p=0.95)
    procs = make_logits_processors(repetition_penalty=1.05, repetition_context_size=64)
    t0 = time.time()
    with open(RAW, "a") as f:
        for s in range(0, len(jobs), BATCH):
            conds = [rng.choice(by_q[q]) for q in jobs[s:s + BATCH]]
            prompts = [tok.apply_chat_template([{"role": "user", "content": prompt_of(c)}],
                                               add_generation_prompt=True, enable_thinking=False)
                       for c in conds]
            res = batch_generate(model, tok, prompts, max_tokens=400, sampler=sampler,
                                 logits_processors=procs, completion_batch_size=BATCH)
            for c, text in zip(conds, res.texts):
                p = parse(text)
                if p:
                    f.write(json.dumps({"subject": p[0], "body": p[1], "queue": c["queue"],
                                        "label": load_meta()["label2id"][c["queue"]],
                                        "cond": {k: c[k] for k in ("type", "priority", "tags")}}) + "\n")
            f.flush()
            done = s + BATCH
            rate = (time.time() - t0) / done
            print(f"  {min(done, len(jobs))}/{len(jobs)}  {(time.time()-t0)/60:5.1f} min  "
                  f"eta {(len(jobs)-done)*rate/60:5.1f} min", flush=True)


def filter_all():
    raw = [json.loads(l) for l in open(RAW)]
    alloc = json.loads((OUT / "qwen_think_allocation.json").read_text())
    x_tr, _ = load_split("train")
    x_te, _ = load_split("test")
    texts = [clean(r["subject"], r["body"]) for r in raw]

    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True).fit(x_tr + texts)
    S, T, R = vec.transform(texts), vec.transform(x_te), vec.transform(x_tr)
    block = lambda M: np.concatenate([linear_kernel(S[i:i+1000], M).max(1) for i in range(0, S.shape[0], 1000)])
    near_test, near_train = block(T), block(R)

    dropped = collections.Counter()
    cand = []
    for i in range(len(raw)):
        if near_test[i] >= TEST_SIM_MAX:
            dropped["near_test"] += 1
        elif near_train[i] >= TRAIN_SIM_MAX:
            dropped["near_train"] += 1
        else:
            cand.append(i)

    Sc = S[cand]
    keep = np.ones(len(cand), bool)
    for a in range(0, len(cand), 1000):
        sims = linear_kernel(Sc[a:a+1000], Sc)
        for r_i in range(sims.shape[0]):
            i = a + r_i
            if keep[i]:
                dup = np.where(sims[r_i] >= TRAIN_SIM_MAX)[0]
                keep[dup[dup > i]] = False
    dropped["near_other_synthetic"] = int((~keep).sum())
    kept = [c for c, k in zip(cand, keep) if k]

    by_q = collections.defaultdict(list)
    for i in kept:
        by_q[raw[i]["queue"]].append(i)
    final = sorted(i for q, idx in by_q.items() for i in idx[:alloc[q]])
    with open(FINAL, "w") as f:
        for i in final:
            r = dict(raw[i])
            r["text"] = texts[i]
            r["nearest_test_similarity"] = round(float(near_test[i]), 3)
            r["source"] = f"{MODEL} + LoRA trained on core tickets, temperature {TEMP}"
            f.write(json.dumps(r) + "\n")
    summary = {"generated": len(raw), "kept": len(final), "dropped": dict(dropped),
               "median_nearest_train_similarity_all": round(float(np.median(near_train)), 3),
               "by_queue": {q: len(v[:alloc[q]]) for q, v in by_q.items()}}
    (OUT / f"{TAG}_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    {"data": make_data, "generate": generate_all, "filter": filter_all}[sys.argv[1]]()
