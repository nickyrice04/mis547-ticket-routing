"""Write synthetic English tickets aimed at the model's weak spots.

Reads data/synthetic/error_report.json, which came from a validation slice of
the training data, never the test set. For each target it shows Gemma real
tickets from that queue and asks for new ones in the same queue, varied in
wording and detail.

Two kinds of target:
  * confusion pairs, for example Product Support tickets that the model sent to
    Technical Support. These teach the boundary between two queues.
  * weak queues, the small ones with low accuracy, where the learning curve says
    extra examples should help most.

Three integrity filters run before anything is saved:
  1. drop anything too close to a TEST ticket, so the model never studies
     near-copies of the exam
  2. drop anything that is a near-copy of an existing training ticket or of
     another synthetic ticket, so the file adds new examples rather than copies
  3. drop anything that just repeats one of the seed examples it was shown

Output is a separate file, data/synthetic/synthetic_tickets.jsonl, so the
original data is never modified and the synthetic rows can be switched on or
off in code.

    python src/experiments/synthetic_data/generate_synthetic.py 20        # prompts per target
"""
from __future__ import annotations

import json
import random
import re
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

from common import clean, load_meta, load_split

OUT = Path("data/synthetic")
MODEL = "mlx-community/gemma-3-4b-it-qat-4bit"
PER_PROMPT = 5
TEST_SIM_MAX = 0.60      # anything closer than this to a test ticket is dropped
TRAIN_SIM_MAX = 0.85     # anything closer than this to an existing ticket is a copy
WEAK_QUEUES = ["General Inquiry", "Returns and Exchanges", "Human Resources", "Sales and Pre-Sales"]


def prompt_for(target, seeds, confused_with=None):
    shown = "\n\n".join(f"Example {i+1}:\n{s}" for i, s in enumerate(seeds))
    if confused_with:
        context = (f'Every ticket below was filed in the "{target}" queue, even though a routing '
                   f'model wrongly sent it to "{confused_with}".')
        focus = (f'They should be the kind of ticket that genuinely belongs in "{target}" but could '
                 f'be mistaken for "{confused_with}".')
    else:
        context = f'Every ticket below was filed in the "{target}" queue.'
        focus = f'They should clearly belong in "{target}".'
    return (
        "You write realistic customer support tickets that customers submit to a company's "
        f"helpdesk.\n\n{context}\n\n{shown}\n\n"
        f"Write {PER_PROMPT} NEW tickets for the \"{target}\" queue. {focus} Use different "
        "products, problems, details and writing styles from the examples. Do not copy them. "
        "Each ticket needs a short subject line and a body of 2 to 5 sentences.\n\n"
        "Format every ticket exactly like this, with nothing else:\n"
        "SUBJECT: <subject>\nBODY: <body>\n---"
    )


def parse(reply):
    out = []
    for block in re.split(r"\n-{3,}\s*\n?", reply):
        s = re.search(r"SUBJECT:\s*(.+)", block)
        b = re.search(r"BODY:\s*(.+)", block, re.S)
        if s and b:
            subject, body = s.group(1).strip(), re.sub(r"\s+", " ", b.group(1)).strip()
            if 40 <= len(body) <= 1500:
                out.append((subject, body))
    return out


def main() -> None:
    prompts_per_target = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    # Model and output name are arguments so a new model never overwrites the
    # tickets an earlier model wrote. The defaults reproduce the Gemma run.
    model_name = sys.argv[2] if len(sys.argv) > 2 else MODEL
    out_name = sys.argv[3] if len(sys.argv) > 3 else "synthetic_tickets"
    from mlx_lm import generate, load
    from mlx_lm.sample_utils import make_sampler

    labels = load_meta()["labels"]
    report = json.loads((OUT / "error_report.json").read_text())
    split = json.loads((OUT / "split.json").read_text())
    x_tr, y_tr = load_split("train")
    x_te, _ = load_split("test")
    y_tr = np.asarray(y_tr)
    val = split["validation"]

    targets = [(p["true"], p["predicted"], p["seeds"]) for p in report["pairs"]]
    for q in WEAK_QUEUES:
        pool = [x_tr[i][:600] for i in val if labels[y_tr[i]] == q]
        targets.append((q, None, pool))

    model, tok = load(model_name)
    sampler = make_sampler(temp=0.85, top_p=0.95)
    # Same seed as every other run, so each model is shown the identical seed
    # tickets in the identical order. The model is the only thing that changes.
    rng = random.Random(42)

    raw = []
    t0 = time.time()
    for n, (target, confused, pool) in enumerate(targets):
        for k in range(prompts_per_target):
            seeds = rng.sample(pool, min(3, len(pool)))
            msg = [{"role": "user", "content": prompt_for(target, seeds, confused)}]
            try:
                text = tok.apply_chat_template(msg, add_generation_prompt=True, tokenize=False,
                                               enable_thinking=False)
            except TypeError:
                text = tok.apply_chat_template(msg, add_generation_prompt=True, tokenize=False)
            reply = generate(model, tok, prompt=text, max_tokens=900, sampler=sampler, verbose=False)
            # Some Qwen models think out loud before answering. Keep only the answer.
            reply = re.sub(r"<think>.*?</think>", "", reply, flags=re.S)
            for subject, body in parse(reply):
                raw.append({"subject": subject, "body": body, "queue": target,
                            "label": labels.index(target), "target_confused_with": confused,
                            "seeds": seeds})
        done = sum(1 for r in raw if r["queue"] == target and r["target_confused_with"] == confused)
        print(f"  [{n+1}/{len(targets)}] {target:32s} vs {str(confused):32s} {done:4d} tickets  "
              f"{(time.time()-t0)/60:.1f} min", flush=True)

    # Integrity filters
    texts = [clean(r["subject"], r["body"]) for r in raw]
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True).fit(x_tr + texts)
    S, T, R = vec.transform(texts), vec.transform(x_te), vec.transform(x_tr)
    near_test = np.concatenate([linear_kernel(S[i:i+500], T).max(1) for i in range(0, S.shape[0], 500)])
    near_train = np.concatenate([linear_kernel(S[i:i+500], R).max(1) for i in range(0, S.shape[0], 500)])

    kept, dropped = [], {"near_test": 0, "near_train": 0, "near_other_synthetic": 0, "copies_seed": 0}
    kept_vecs = []
    for i, r in enumerate(raw):
        if near_test[i] >= TEST_SIM_MAX:
            dropped["near_test"] += 1
            continue
        if near_train[i] >= TRAIN_SIM_MAX:
            dropped["near_train"] += 1
            continue
        if any(texts[i][:120] in clean("", s) for s in r["seeds"]):
            dropped["copies_seed"] += 1
            continue
        if kept_vecs and linear_kernel(S[i], S[kept_vecs]).max() >= TRAIN_SIM_MAX:
            dropped["near_other_synthetic"] += 1
            continue
        kept_vecs.append(i)
        r = dict(r)
        r["text"] = texts[i]
        r["nearest_test_similarity"] = round(float(near_test[i]), 3)
        r["source"] = model_name
        r.pop("seeds")
        kept.append(r)

    with open(OUT / f"{out_name}.jsonl", "w") as f:
        for r in kept:
            f.write(json.dumps(r) + "\n")
    summary = {"generated": len(raw), "kept": len(kept), "dropped": dropped,
               "minutes": round((time.time() - t0) / 60, 1),
               "by_queue": {q: sum(1 for r in kept if r["queue"] == q) for q in labels}}
    summary["model"] = model_name
    (OUT / f"{out_name}_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
