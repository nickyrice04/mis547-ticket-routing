"""Fine-tune Laya on the ten queues, then score accuracy and calibration.

Laya is an encoder with a decision head. Each ticket becomes one "choice"
question whose ten options are the queue names. The head scores a marker token
in front of each option, and the softmax over those ten scores is the routing
distribution. Laya's own package ships the model code but no training loop, so
this script is the training loop, written for Apple's GPU (MPS) in float32. An
earlier DeBERTa run on MPS went to NaN in half precision.

Two phases, so the test set is never used for a decision:

  A. train on the core of the training set, check the held-out validation slice
     after every epoch, and keep the best epoch count. A temperature is fitted
     on the same slice so the probabilities are calibrated.
  B. retrain from the original weights on all of the training data for that
     many epochs, apply the temperature from A, and score the test set once.

Arms
    en      convaiinnovations/laya (ModernBERT-large), English training tickets
    multi   convaiinnovations/laya-multilingual (mmBERT-base), English plus German
            training tickets. German tickets that are near-translations of an
            English test ticket are dropped first.

    PYTHONPATH=src python src/experiments/bigger_models/train_laya.py en
    PYTHONPATH=src python src/experiments/bigger_models/train_laya.py multi
"""
from __future__ import annotations

import copy
import json
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score

from common import RESULTS, clean, load_meta, load_split, threshold_table

ARMS = {
    "en": {"model": "convaiinnovations/laya", "german": False, "tier": "11_laya_en"},
    "multi": {"model": "convaiinnovations/laya-multilingual", "german": True, "tier": "11b_laya_multi_de"},
}
MAX_EPOCHS = int(os.environ.get("EPOCHS", 4))
BATCH = int(os.environ.get("BATCH", 16))
LR_ENC = float(os.environ.get("LR_ENC", 2e-5))
LR_HEAD = float(os.environ.get("LR_HEAD", 1e-4))
LIMIT = int(os.environ.get("LIMIT", 0))          # for a quick smoke test only
GERMAN_LEAK = 0.93                                 # stricter than the 0.95 used before
INSTRUCTIONS = "Which support queue should handle this customer ticket?"


def seed_all(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)


def encode(agent, texts, labels_list, rng=None):
    """Turn tickets into Laya items. With rng, option order is shuffled as augmentation."""
    from laya.common import build_sequence

    q = {"t": "choice", "ins": INSTRUCTIONS, "crit": {l: None for l in labels_list}}
    max_len, head = agent.cfg.get("max_len", 512), agent.cfg.get("head_max_len", 192)
    out = []
    for t in texts:
        order = list(range(len(labels_list)))
        if rng is not None:
            rng.shuffle(order)
        ids, markers = build_sequence(agent.tok, t, q, max_len, head, option_order=order)
        assert len(markers) == len(labels_list)
        out.append((ids, markers, order))
    return out


def batchify(agent, items, idx, dev):
    from laya.common import collate_items

    b = collate_items([[{"ids": items[i][0], "markers": items[i][1], "qtype": 0} for i in idx]],
                      agent.tok.pad_token_id)
    return {k: b[k].to(dev) for k in ("input_ids", "attention_mask", "marker_pos", "marker_mask", "qtype")}


def to_queue_order(logits, orders):
    """Logits come out in shuffled option order. Put them back in queue order."""
    out = torch.empty_like(logits)
    for r, order in enumerate(orders):
        out[r, torch.tensor(order, device=logits.device)] = logits[r]
    return out


@torch.no_grad()
def predict_logits(agent, texts, labels_list, dev):
    model = agent.model.eval()
    items = encode(agent, texts, labels_list)
    order = np.argsort([len(i[0]) for i in items])          # length sort, less padding
    out = np.zeros((len(texts), len(labels_list)), np.float32)
    for s in range(0, len(order), 64):
        idx = order[s:s + 64]
        logits, _ = model(**batchify(agent, items, idx, dev))
        out[idx] = logits.float().cpu().numpy()
    return out


def fit_temperature(logits, y):
    lg, yt = torch.tensor(logits), torch.tensor(y)
    best = min(np.linspace(0.3, 4.0, 75), key=lambda T: F.cross_entropy(lg / T, yt).item())
    return float(best)


def softmax(z):
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def train(arm, x, y, labels_list, epochs, dev, x_val=None, y_val=None, tag=""):
    import laya

    agent = laya.load(arm["model"])
    try:
        agent.model.encoder.config.reference_compile = False
    except Exception:
        pass
    model = agent.model.to(dev).float().train()
    enc_params = [p for n, p in model.named_parameters() if n.startswith("encoder.")]
    head_params = [p for n, p in model.named_parameters() if not n.startswith("encoder.")]
    opt = torch.optim.AdamW([{"params": enc_params, "lr": LR_ENC},
                             {"params": head_params, "lr": LR_HEAD}], weight_decay=0.01)
    steps = epochs * ((len(x) + BATCH - 1) // BATCH)
    warm = int(0.06 * steps)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / max(1, warm)) * max(0.0, (steps - s) / max(1, steps - warm)))

    y = np.asarray(y)
    history, best, best_state = [], -1.0, None
    step, t0 = 0, time.time()
    for ep in range(epochs):
        rng = random.Random(1000 + ep)
        items = encode(agent, x, labels_list, rng)       # fresh option order every epoch
        perm = np.random.default_rng(ep).permutation(len(x))
        # Bucket by length inside chunks, so batches pad less but stay shuffled.
        chunks = [perm[i:i + BATCH * 50] for i in range(0, len(perm), BATCH * 50)]
        batches = []
        for c in chunks:
            c = sorted(c, key=lambda i: len(items[i][0]))
            batches += [c[i:i + BATCH] for i in range(0, len(c), BATCH)]
        random.Random(ep).shuffle(batches)
        model.train()
        run = 0.0
        for bi, idx in enumerate(batches):
            logits, _ = model(**batchify(agent, items, idx, dev))
            logits = to_queue_order(logits, [items[i][2] for i in idx])
            loss = F.cross_entropy(logits, torch.tensor(y[idx], device=dev), label_smoothing=0.05)
            if not torch.isfinite(loss):
                raise RuntimeError(f"loss went to {loss.item()} at epoch {ep} step {bi}")
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step(); step += 1
            run = 0.98 * run + 0.02 * loss.item() if bi else loss.item()
            if step % 100 == 0:
                rate = (time.time() - t0) / step
                print(f"  {tag} epoch {ep+1}/{epochs} step {step}/{steps} loss {run:.3f} "
                      f"{(time.time()-t0)/60:5.1f} min, eta {(steps-step)*rate/60:5.1f} min", flush=True)
        if x_val is not None:
            lv = predict_logits(agent, x_val, labels_list, dev)
            acc = accuracy_score(y_val, lv.argmax(1))
            history.append({"epoch": ep + 1, "val_accuracy": round(float(acc), 4)})
            print(f"  {tag} epoch {ep+1} validation accuracy {acc*100:.2f}%", flush=True)
            if acc > best:
                best, best_state = acc, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    return agent, history, (time.time() - t0) / 60


def german_rows(labels_list, x_te):
    """German tickets in the ten queues, minus near-translations of English test tickets."""
    import pyarrow.parquet as pq
    from sentence_transformers import SentenceTransformer

    lab2id = {l: i for i, l in enumerate(labels_list)}
    t = pq.read_table("data_tickets.parquet").to_pydict()
    seen, xs, ys = set(), [], []
    for i in range(len(t["queue"])):
        if t["language"][i] != "de" or t["queue"][i] not in lab2id:
            continue
        text = clean(t["subject"][i], t["body"][i])
        if text and text not in seen:
            seen.add(text); xs.append(text); ys.append(lab2id[t["queue"][i]])
    enc = SentenceTransformer("intfloat/multilingual-e5-base", device="mps")
    emb = lambda v: enc.encode([f"query: {s[:1000]}" for s in v], batch_size=64,
                               normalize_embeddings=True, show_progress_bar=False)
    E_de, E_te = emb(xs), emb(x_te)
    near = np.concatenate([(E_de[s:s + 2000] @ E_te.T).max(1) for s in range(0, len(E_de), 2000)])
    keep = near < GERMAN_LEAK
    print(f"German tickets {len(xs)}, dropped {int((~keep).sum())} as near-translations of test tickets",
          flush=True)
    del enc
    return [x for x, k in zip(xs, keep) if k], [y for y, k in zip(ys, keep) if k], int((~keep).sum())


def main() -> None:
    arm = ARMS[sys.argv[1]]
    dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    seed_all(42)
    labels_list = load_meta()["labels"]
    x_tr, y_tr = load_split("train")
    x_te, y_te = load_split("test")
    y_tr, y_te = np.asarray(y_tr), np.asarray(y_te)
    split = json.loads(open("data/synthetic/split.json").read())
    core, val = split["core"], split["validation"]

    extra_x, extra_y, dropped = [], [], 0
    if arm["german"]:
        extra_x, extra_y, dropped = german_rows(labels_list, x_te)
    if LIMIT:
        core, val, x_te, y_te = core[:LIMIT], val[:LIMIT // 4], x_te[:LIMIT // 4], y_te[:LIMIT // 4]
        extra_x, extra_y = extra_x[:LIMIT // 2], extra_y[:LIMIT // 2]

    # Phase A: pick the epoch count and the temperature on the validation slice.
    xa = [x_tr[i] for i in core] + extra_x
    ya = np.concatenate([y_tr[core], np.asarray(extra_y, int)])
    xv, yv = [x_tr[i] for i in val], y_tr[val]
    agent, hist, min_a = train(arm, xa, ya, labels_list, MAX_EPOCHS, dev, xv, yv, tag="A")
    best_ep = max(hist, key=lambda h: h["val_accuracy"])["epoch"]
    T = fit_temperature(predict_logits(agent, xv, labels_list, dev), yv)
    print(f"phase A done in {min_a:.0f} min, best epoch {best_ep}, temperature {T:.2f}", flush=True)
    del agent
    torch.mps.empty_cache()

    # Phase B: all training data, fixed epoch count, then the test set once.
    xb = list(x_tr) + extra_x
    yb = np.concatenate([y_tr, np.asarray(extra_y, int)])
    agent, _, min_b = train(arm, xb, yb, labels_list, best_ep, dev, tag="B")
    t0 = time.time()
    logits = predict_logits(agent, x_te, labels_list, dev)
    per_ticket_ms = (time.time() - t0) / len(x_te) * 1000
    P = softmax(logits / T)
    pred = P.argmax(1)

    from laya.common import ece_score
    conf = P.max(1)
    res = {
        "tier": arm["tier"], "model": arm["model"],
        "params_m": round(sum(p.numel() for p in agent.model.parameters()) / 1e6, 1),
        "accuracy": round(float(accuracy_score(y_te, pred)), 4),
        "macro_f1": round(float(f1_score(y_te, pred, average="macro")), 4),
        "ece": round(ece_score(conf, (pred == y_te).astype(float)), 4),
        "ece_before_temperature": round(ece_score(softmax(logits).max(1),
                                                  (pred == y_te).astype(float)), 4),
        "temperature": T, "epochs": best_ep, "phase_a_history": hist,
        "train_rows": len(yb), "german_added": len(extra_x), "german_dropped_as_leaks": dropped,
        "train_minutes": round(min_a + min_b, 1), "ms_per_ticket_batched": round(per_ticket_ms, 2),
        "device": str(dev), "thresholds": threshold_table(P, y_te),
    }
    if not LIMIT:
        np.save(RESULTS / f"{arm['tier']}_probs.npy", P)
        (RESULTS / f"{arm['tier']}.json").write_text(json.dumps(res, indent=2))
        out = f"models/{arm['tier']}"
        os.makedirs(out, exist_ok=True)
        from safetensors.torch import save_file
        save_file({k: v.contiguous().cpu() for k, v in agent.model.state_dict().items()},
                  f"{out}/model.safetensors")
        json.dump({**agent.cfg, "labels": labels_list, "temperature_fitted": T},
                  open(f"{out}/rl_agent_config.json", "w"), indent=2)
    print(json.dumps({k: v for k, v in res.items() if k != "thresholds"}, indent=2))


if __name__ == "__main__":
    main()
