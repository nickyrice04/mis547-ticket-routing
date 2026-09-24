"""10,000 synthetic tickets from Qwen3.5 0.8B with thinking on, aimed at the misses.

Where the tickets go
    Each queue's share comes from where the network misses on the validation
    slice. 70% of the budget follows the raw number of misses, which is where
    most of the errors are. 30% follows the miss rate, so small queues that are
    missed most of the time still get enough examples to matter.

What each prompt shows the model
    Three real validation tickets from the target queue, two of them taken from
    the ones the network misrouted when there are enough, plus the two queues
    that queue is most often confused with. The model is asked for five new
    tickets that genuinely belong in the target queue.

Thinking
    Qwen3.5 exposes thinking as an on or off switch, with no budget setting. With
    it on, this model loops on "Wait, I need to check the instruction again" and
    never finishes reasoning on its own. So reasoning is capped at 1,500 tokens,
    then closed and the model is made to answer. This is budget forcing. It keeps
    thinking on while guaranteeing an answer.

Integrity, same rules as every earlier run
    * seeds come only from the validation slice of the training data
    * the test set is read once, at the end, only to drop anything too close to it
    * near-copies of real tickets, of seeds, and of each other are dropped
    * everything is written to separate files, and the real data is never touched

Crash safety
    Every parsed ticket is appended to the raw file the moment it exists, queues
    are visited in rotation, and a rerun resumes where the last one stopped.

    python src/experiments/synthetic_data/generate_targeted.py            # generate, resuming if interrupted
    python src/experiments/synthetic_data/generate_targeted.py --filter   # filter only, after generation
"""
from __future__ import annotations

import collections
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

from common import clean, load_meta, load_split
from experiments.synthetic_data.generate_synthetic import TEST_SIM_MAX, TRAIN_SIM_MAX, parse

OUT = Path("data/synthetic")
RAW = OUT / "raw_qwen_think.jsonl"
FINAL = OUT / "qwen_think_10k.jsonl"
MODEL = "mlx-community/Qwen3.5-0.8B-MLX-8bit"
TOTAL = 10_000
PER_PROMPT = 5
THINK_BUDGET = 1500
ANSWER_TOKENS = 900
OVERSHOOT = 1.02   # a little extra so the filters do not leave a queue short


def allocation(labels, y_val, pred):
    wrong = pred != y_val
    miss = np.array([((y_val == q) & wrong).sum() for q in range(len(labels))], float)
    rate = np.array([miss[q] / max((y_val == q).sum(), 1) for q in range(len(labels))])
    share = 0.7 * miss / miss.sum() + 0.3 * rate / rate.sum()
    alloc = np.floor(share * TOTAL).astype(int)
    alloc[np.argmax(share)] += TOTAL - alloc.sum()
    return {labels[q]: int(alloc[q]) for q in range(len(labels))}, miss, rate


def prompt_for(target, seeds, confused):
    shown = "\n\n".join(f"Example {i+1}:\n{s}" for i, s in enumerate(seeds))
    others = " or ".join(f'"{c}"' for c in confused)
    return (
        "You write realistic customer support tickets that customers submit to a company's "
        f'helpdesk.\n\nEvery ticket below was filed in the "{target}" queue. A routing model often '
        f"sends tickets like these to {others} by mistake.\n\n{shown}\n\n"
        f'Write {PER_PROMPT} NEW tickets that genuinely belong in the "{target}" queue, of the kind '
        f"that could be mistaken for {others}. Match the tone, length and phrasing of the examples "
        "closely, but use different products, problems and details. Do not copy them. Each ticket "
        "needs a short subject line and a body of 2 to 5 sentences.\n\n"
        "Format every ticket exactly like this, with nothing else:\n"
        "SUBJECT: <subject>\nBODY: <body>\n---"
    )


def generate_all():
    from mlx_lm import generate, load
    from mlx_lm.sample_utils import make_logits_processors, make_sampler

    labels = load_meta()["labels"]
    x_tr, y_tr = load_split("train")
    y_tr = np.asarray(y_tr)
    split = json.loads((OUT / "split.json").read_text())
    val = np.asarray(split["validation"])
    y_val = y_tr[val]
    pred = np.asarray(json.loads((OUT / "val_predictions.json").read_text())["val_pred"])

    alloc, miss, rate = allocation(labels, y_val, pred)
    print(f"{'queue':32s} {'misses':>7} {'miss rate':>9} {'tickets':>8}")
    for q, n in sorted(alloc.items(), key=lambda kv: -kv[1]):
        i = labels.index(q)
        print(f"{q:32s} {int(miss[i]):7d} {rate[i]*100:8.1f}% {n:8d}")
    (OUT / "qwen_think_allocation.json").write_text(json.dumps(alloc, indent=2))

    plan = {}
    for q in labels:
        i = labels.index(q)
        mine = val[y_val == i]
        missed = [x_tr[j][:600] for j, p in zip(val, pred) if y_tr[j] == i and p != i]
        right = [x_tr[j][:600] for j, p in zip(val, pred) if y_tr[j] == i and p == i]
        conf = collections.Counter(int(p) for j, p in zip(val, pred) if y_tr[j] == i and p != i)
        confused = [labels[c] for c, _ in conf.most_common(2)] or ["another queue"]
        plan[q] = {"missed": missed, "right": right, "confused": confused,
                   "want": math.ceil(alloc[q] * OVERSHOOT)}

    have = collections.Counter()
    if RAW.exists():
        for line in open(RAW):
            have[json.loads(line)["queue"]] += 1
        print(f"\nresuming: {sum(have.values())} tickets already on disk")

    model, tok = load(MODEL)
    sampler = make_sampler(temp=0.6, top_p=0.95, top_k=20)   # Qwen's thinking-mode settings
    procs = make_logits_processors(repetition_penalty=1.1, repetition_context_size=64)
    rng = random.Random(42 + sum(have.values()))

    t0 = time.time()
    prompts = 0
    made_this_run = 0
    with open(RAW, "a") as f:
        while any(have[q] < plan[q]["want"] for q in labels):
            for q in labels:                       # rotate so a partial run stays balanced
                spec = plan[q]
                if have[q] >= spec["want"]:
                    continue
                if len(spec["missed"]) >= 2:
                    seeds = rng.sample(spec["missed"], 2) + rng.sample(spec["right"] or spec["missed"], 1)
                else:
                    seeds = rng.sample(spec["missed"] + spec["right"], 3)
                text = tok.apply_chat_template(
                    [{"role": "user", "content": prompt_for(q, seeds, spec["confused"])}],
                    add_generation_prompt=True, tokenize=False, enable_thinking=True)
                thought = generate(model, tok, prompt=text, max_tokens=THINK_BUDGET,
                                   sampler=sampler, logits_processors=procs, verbose=False)
                if "</think>" in thought:
                    answer = thought.split("</think>", 1)[1]
                    if len(parse(answer)) < PER_PROMPT:
                        answer += generate(model, tok, prompt=text + thought, max_tokens=ANSWER_TOKENS,
                                           sampler=sampler, logits_processors=procs, verbose=False)
                    forced = False
                else:
                    answer = generate(model, tok, prompt=text + thought + "\n</think>\n\n",
                                      max_tokens=ANSWER_TOKENS, sampler=sampler,
                                      logits_processors=procs, verbose=False)
                    forced = True
                for subject, body in parse(answer)[:PER_PROMPT]:
                    f.write(json.dumps({"subject": subject, "body": body, "queue": q,
                                        "label": labels.index(q), "confused_with": spec["confused"],
                                        "seeds": seeds, "thinking_forced_closed": forced}) + "\n")
                    have[q] += 1
                    made_this_run += 1
                f.flush()
                prompts += 1
                if prompts % 50 == 0:
                    done, total = sum(have.values()), sum(p["want"] for p in plan.values())
                    rate_s = (time.time() - t0) / max(made_this_run, 1)
                    print(f"  {done:6d}/{total} tickets  {(time.time()-t0)/60:6.1f} min  "
                          f"eta {(total-done)*rate_s/60:6.1f} min", flush=True)


def filter_all():
    labels = load_meta()["labels"]
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
    for i, r in enumerate(raw):
        if near_test[i] >= TEST_SIM_MAX:
            dropped["near_test"] += 1
        elif near_train[i] >= TRAIN_SIM_MAX:
            dropped["near_train"] += 1
        elif any(texts[i][:120] in clean("", s) for s in r["seeds"]):
            dropped["copies_seed"] += 1
        else:
            cand.append(i)

    # Greedy dedupe among the synthetic tickets themselves, in blocks.
    kept_idx = []
    Sc = S[cand]
    keep_mask = np.ones(len(cand), bool)
    for a in range(0, len(cand), 1000):
        sims = linear_kernel(Sc[a:a+1000], Sc).astype(np.float32)
        for r_i in range(sims.shape[0]):
            i = a + r_i
            if not keep_mask[i]:
                continue
            dup = np.where(sims[r_i] >= TRAIN_SIM_MAX)[0]
            dup = dup[dup > i]
            keep_mask[dup] = False
    for pos, i in enumerate(cand):
        if keep_mask[pos]:
            kept_idx.append(i)
        else:
            dropped["near_other_synthetic"] += 1

    by_q = collections.defaultdict(list)
    for i in kept_idx:
        by_q[raw[i]["queue"]].append(i)
    final = []
    for q, idx in by_q.items():
        final += idx[: alloc[q]]          # trim the overshoot back to the allocation
    final.sort()

    with open(FINAL, "w") as f:
        for i in final:
            r = dict(raw[i])
            r.pop("seeds")
            r["text"] = texts[i]
            r["nearest_test_similarity"] = round(float(near_test[i]), 3)
            r["source"] = MODEL + " thinking on, budget forced at 1500"
            f.write(json.dumps(r) + "\n")
    summary = {"generated": len(raw), "kept": len(final), "dropped": dict(dropped),
               "forced_close_rate": round(float(np.mean([r["thinking_forced_closed"] for r in raw])), 3),
               "by_queue": {q: len([i for i in final if raw[i]["queue"] == q]) for q in labels}}
    (OUT / "qwen_think_10k_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    if "--filter" not in sys.argv:
        generate_all()
    filter_all()
