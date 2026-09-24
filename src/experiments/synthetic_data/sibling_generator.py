"""Fine-tune a small LLM to write SIBLINGS of real tickets, then use them as training data.

What we learned first
    This dataset was built in families. One underlying ticket was reworded
    several times, and every rewording kept the same queue. Core tickets with a
    twin at cosine 0.8 or more share the queue 99.7% of the time. A ticket with
    a family member in the training data is routed almost perfectly by looking
    up its nearest neighbour. A ticket with no family member is close to a coin
    flip for every model we tried.

    That is why the first fine-tuned generator did not help. It wrote NEW
    tickets for a queue, which means new families with a label the generator
    picked. This generator does the opposite. It rewrites a REAL ticket the way
    this dataset rewrites tickets, so the label carries over by construction
    and the family gets denser. A test ticket that was too far from its family
    to be matched now has more chances of a close relative.

How the generator learns
    Pairs are mined from the core training tickets only: two tickets in the
    same queue whose TF-IDF cosine is between 0.40 and 0.85. The adapter is
    trained to write B when shown A.

Integrity
    * pairs come only from core, so the validation slice stays a fair exam
    * the test set is never read anywhere in this file
    * siblings that are near-copies of their source (cosine 0.92 or more) are
      dropped, since they add nothing
    * the output goes to a separate file

    python src/experiments/synthetic_data/sibling_generator.py pairs                # mine pairs, write LoRA data
    mlx_lm.lora -c configs/lora_sibling.yaml                     # train the adapter
    python src/experiments/synthetic_data/sibling_generator.py generate core 2      # 2 siblings per core ticket
    python src/experiments/synthetic_data/sibling_generator.py generate train 2     # for the final model, all of train
"""
from __future__ import annotations

import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

from common import load_split

OUT = Path("data/synthetic")
LORA_DATA = Path("data/lora_sibling")
MODEL = os.environ.get("GEN_MODEL", "mlx-community/Qwen3.5-0.8B-MLX-8bit")
ADAPTER = os.environ.get("GEN_ADAPTER", "adapters/qwen35_sibling")
TAG = os.environ.get("GEN_TAG", "qwen_sibling")
TEMP = float(os.environ.get("TEMP", 0.9))
LO, HI = 0.40, 0.85
MAX_CHARS = 1100
MAX_PAIRS = int(os.environ.get("MAX_PAIRS", 7000))
BATCH = 64
INSTRUCTION = ("Rewrite this customer support ticket the way another customer with the same "
               "problem would write it. Keep the same problem, products and details.\n\n")


def core_and_val():
    x, y = load_split("train")
    split = json.loads((OUT / "split.json").read_text())
    return x, np.asarray(y), split["core"], split["validation"]


def mine_pairs():
    x, y, core, val = core_and_val()
    xc, yc = [x[i] for i in core], y[core]
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True).fit(xc)
    C = vec.transform(xc)
    rng = random.Random(42)
    pairs = []
    for a in range(0, len(xc), 1000):
        K = linear_kernel(C[a:a + 1000], C)
        for r in range(K.shape[0]):
            i = a + r
            if len(xc[i]) > MAX_CHARS:
                continue
            ok = np.where((K[r] >= LO) & (K[r] < HI) & (yc == yc[i]))[0]
            ok = [j for j in ok if j != i and len(xc[j]) <= MAX_CHARS]
            if ok:
                j = rng.choice(ok)
                pairs.append((i, int(j), float(K[r, j])))
    rng.shuffle(pairs)
    print(f"core tickets with at least one sibling: {len(pairs)} of {len(xc)}")
    print(f"median pair similarity {np.median([p[2] for p in pairs]):.3f}")
    LORA_DATA.mkdir(parents=True, exist_ok=True)
    use, held = pairs[:MAX_PAIRS], pairs[MAX_PAIRS:MAX_PAIRS + 300]
    for name, rows in (("train", use), ("valid", held)):
        with open(LORA_DATA / f"{name}.jsonl", "w") as f:
            for i, j, _ in rows:
                f.write(json.dumps({"prompt": INSTRUCTION + xc[i], "completion": xc[j]}) + "\n")
    print(f"wrote {len(use)} training pairs and {len(held)} held-out pairs")


def generate(pool_name, per_ticket):
    from mlx_lm import batch_generate, load
    from mlx_lm.sample_utils import make_sampler

    x, y, core, val = core_and_val()
    pool = core if pool_name == "core" else list(range(len(x)))
    raw = OUT / f"raw_{TAG}_{pool_name}.jsonl"
    done = set()
    if raw.exists():
        for line in open(raw):
            r = json.loads(line)
            done.add((r["source"], r["k"]))
    jobs = [(i, k) for k in range(per_ticket) for i in pool
            if (i, k) not in done and len(x[i]) <= 1600]
    # Length-sorted chunks keep batches efficient, shuffled so a partial run is a random sample.
    rng = random.Random(7)
    rng.shuffle(jobs)
    chunks = [sorted(jobs[s:s + BATCH * 20], key=lambda j: len(x[j[0]])) for s in range(0, len(jobs), BATCH * 20)]
    jobs = [j for c in chunks for j in c]
    print(f"{len(done)} on disk, {len(jobs)} to write", flush=True)

    model, tok = load(MODEL, adapter_path=ADAPTER)
    sampler = make_sampler(temp=TEMP, top_p=0.95)
    t0 = time.time()
    with open(raw, "a") as f:
        for s in range(0, len(jobs), BATCH):
            part = jobs[s:s + BATCH]
            prompts = [tok.apply_chat_template([{"role": "user", "content": INSTRUCTION + x[i]}],
                                               add_generation_prompt=True, enable_thinking=False)
                       for i, _ in part]
            longest = max(len(p) for p in prompts)
            res = batch_generate(model, tok, prompts, max_tokens=min(520, longest + 60), sampler=sampler,
                                 completion_batch_size=BATCH)
            for (i, k), text in zip(part, res.texts):
                text = " ".join(text.split()).strip().lower()
                if len(text) >= 40:
                    f.write(json.dumps({"source": i, "k": k, "label": int(y[i]), "text": text}) + "\n")
            f.flush()
            n = s + len(part)
            rate = (time.time() - t0) / n
            if (s // BATCH) % 10 == 0:
                print(f"  {n}/{len(jobs)}  {(time.time()-t0)/60:5.1f} min  eta {(len(jobs)-n)*rate/60:5.1f} min",
                      flush=True)
    finalize(pool_name)


def finalize(pool_name):
    """Drop siblings that copy their source, and duplicates. The test set is not read."""
    x, y, core, val = core_and_val()
    rows = [json.loads(l) for l in open(OUT / f"raw_{TAG}_{pool_name}.jsonl")]
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True).fit([x[i] for i in core])
    S = vec.transform([r["text"] for r in rows])
    A = vec.transform([x[r["source"]] for r in rows])
    to_source = np.asarray(S.multiply(A).sum(1)).ravel()
    seen, kept, dropped = set(), [], {"copies_source": 0, "drifted": 0, "duplicate": 0}
    for r, s in zip(rows, to_source):
        if s >= 0.92:
            dropped["copies_source"] += 1
        elif s < 0.15:
            dropped["drifted"] += 1          # no longer about the same ticket
        elif r["text"] in seen:
            dropped["duplicate"] += 1
        else:
            seen.add(r["text"])
            kept.append({**r, "similarity_to_source": round(float(s), 3)})
    with open(OUT / f"{TAG}_{pool_name}.jsonl", "w") as f:
        for r in kept:
            f.write(json.dumps(r) + "\n")
    summary = {"generated": len(rows), "kept": len(kept), "dropped": dropped,
               "median_similarity_to_source": round(float(np.median([k["similarity_to_source"] for k in kept])), 3),
               "model": MODEL, "adapter": ADAPTER, "temperature": TEMP}
    (OUT / f"{TAG}_{pool_name}_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "pairs":
        mine_pairs()
    elif cmd == "generate":
        generate(sys.argv[2], int(sys.argv[3]))
    elif cmd == "finalize":
        finalize(sys.argv[2])
