"""Zero-shot and few-shot ticket routing with a small language model.

No fine-tuning. The model is handed the list of queues and asked which one each
ticket belongs to. This measures something different from every other tier in
the study: whether a model that was never shown our labels can do the job
straight out of the box.

Efficiency matters here, because a language model is thousands of times slower
per ticket than the classifiers we already have. Two things keep it practical:

  * many tickets per prompt, so the instructions and the queue list are read
    once for a whole batch instead of once per ticket
  * a tight output format, one line per ticket, so the model spends its tokens
    on answers instead of explanation

    python src/experiments/bigger_models/zeroshot_slm.py mlx-community/Qwen3-0.6B-4bit qwen3_0_6b 500 10
    python src/experiments/bigger_models/zeroshot_slm.py mlx-community/gemma-3-4b-it-qat-4bit gemma3_4b 500 10 --fewshot
"""
from __future__ import annotations

import json
import re
import sys
import time

import numpy as np
from sklearn.metrics import accuracy_score, f1_score

from common import load_meta, load_split, RESULTS

TICKET_CHARS = 420   # keeps a batch of 10 inside a comfortable context window
MAX_NEW_PER_TICKET = 8


def build_prompt(labels, tickets, examples=None):
    numbered = "\n".join(f"{i+1}. {name}" for i, name in enumerate(labels))
    head = (
        "You route customer support tickets to the correct team.\n\n"
        f"The teams are:\n{numbered}\n\n"
    )
    if examples:
        shots = "\n".join(f"Ticket: {t[:200]}\nTeam: {labels[y]}" for t, y in examples)
        head += f"Examples:\n{shots}\n\n"
    body = "\n\n".join(
        f"TICKET {i+1}:\n{t[:TICKET_CHARS]}" for i, t in enumerate(tickets)
    )
    # Ask for team numbers rather than names.
    #
    # Two earlier formats failed. Asking for "<ticket number>. <team name>" made
    # the model reply literally with "<ticket 1>. ...". Showing a worked example
    # with real team names made the small model copy those names in order
    # instead of reading the tickets. Numbers give it nothing worth copying, and
    # a model that copies anyway is easy to catch because every answer repeats.
    tail = (
        f"\n\nFor each of the {len(tickets)} tickets above, write the ticket number, "
        f"then the number of the team it should go to.\n"
        f"Write exactly {len(tickets)} lines, one per ticket, in this format:\n"
        f"ticket_1 = team_number_for_ticket_1\n"
        f"Team numbers must be between 1 and {len(labels)}. No other text.\n\nAnswers:\n"
    )
    return head + body + tail


def parse(reply: str, labels, n: int):
    """Map the model's lines back to label ids, -1 where it did not comply."""
    lower = {name.lower(): i for i, name in enumerate(labels)}
    out = [-1] * n
    for line in reply.splitlines():
        # Preferred format: "3 = 7". Also accept "3. 7", "3) 7", "3: 7".
        m = re.match(r"\s*[<\[]?\s*(?:ticket[\s_]*)?(\d+)\s*[>\]]?\s*[=.):\-]\s*(.+)", line, re.I)
        if not m:
            continue
        idx = int(m.group(1)) - 1
        if not (0 <= idx < n):
            continue
        answer = m.group(2).strip().strip(".").lower()

        num = re.match(r"^(\d+)", answer)
        if num and 1 <= int(num.group(1)) <= len(labels):
            out[idx] = int(num.group(1)) - 1
            continue
        if answer in lower:
            out[idx] = lower[answer]
            continue
        # tolerate near-misses like "billing" for "Billing and Payments"
        for name, i in lower.items():
            if answer and (answer in name or name.startswith(answer)):
                out[idx] = i
                break
    return out


def main() -> None:
    model_path, tag = sys.argv[1], sys.argv[2]
    n_eval = int(sys.argv[3]) if len(sys.argv) > 3 else 500
    batch = int(sys.argv[4]) if len(sys.argv) > 4 else 10
    fewshot = "--fewshot" in sys.argv

    from mlx_lm import generate, load
    from mlx_lm.sample_utils import make_sampler

    labels = load_meta()["labels"]
    x_te, y_te = load_split("test")
    x_tr, y_tr = load_split("train")

    rng = np.random.default_rng(42)
    idx = rng.choice(len(x_te), size=min(n_eval, len(x_te)), replace=False)
    tickets = [x_te[i] for i in idx]
    gold = [y_te[i] for i in idx]

    examples = None
    if fewshot:
        # one short example per queue, taken from training data only
        examples = []
        for lab in range(len(labels)):
            for t, y in zip(x_tr, y_tr):
                if y == lab:
                    examples.append((t, y))
                    break

    print(f"loading {model_path}", flush=True)
    t0 = time.time()
    model, tokenizer = load(model_path)
    load_seconds = time.time() - t0
    sampler = make_sampler(temp=0.0)

    preds, unparsed = [], 0
    t0 = time.time()
    for start in range(0, len(tickets), batch):
        chunk = tickets[start : start + batch]
        prompt = build_prompt(labels, chunk, examples)
        messages = [{"role": "user", "content": prompt}]
        try:
            text = tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False, enable_thinking=False
            )
        except TypeError:  # models whose template has no thinking switch
            text = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        reply = generate(
            model, tokenizer, prompt=text,
            max_tokens=MAX_NEW_PER_TICKET * len(chunk) + 32,
            sampler=sampler, verbose=False,
        )
        got = parse(reply, labels, len(chunk))
        unparsed += sum(1 for g in got if g == -1)
        preds.extend(got)
        done = start + len(chunk)
        if done % 100 == 0 or done >= len(tickets):
            secs = time.time() - t0
            print(f"  {done}/{len(tickets)} tickets, {secs/done*1000:.0f} ms each, "
                  f"{unparsed} unparsed", flush=True)
    total_seconds = time.time() - t0

    preds = np.asarray(preds)
    gold = np.asarray(gold)
    ok = preds >= 0
    # Unparsed replies count as wrong. A model that will not follow the output
    # format is not usable in a pipeline, so hiding those would flatter it.
    result = {
        "tier": f"5_{tag}{'_fewshot' if fewshot else '_zeroshot'}",
        "model": f"{model_path} ({'few-shot' if fewshot else 'zero-shot'}, no fine-tuning)",
        "eval_tickets": len(tickets),
        "batch_size": batch,
        "load_seconds": round(load_seconds, 1),
        "total_seconds": round(total_seconds, 1),
        "ms_per_ticket": round(total_seconds / len(tickets) * 1000, 1),
        "unparsed": int(unparsed),
        "accuracy": round(float(accuracy_score(gold, np.where(ok, preds, -1))), 4),
        "accuracy_on_parsed": round(float(accuracy_score(gold[ok], preds[ok])), 4) if ok.any() else 0.0,
        "macro_f1": round(float(f1_score(gold, np.where(ok, preds, -1), average="macro")), 4),
    }
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / f"{result['tier']}.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
