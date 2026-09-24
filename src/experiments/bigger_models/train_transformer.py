"""Tiers 2 and up: fine-tune a transformer encoder for queue classification.

Same split, same metrics, same threshold table as the baseline. Pass any
HuggingFace encoder:

    python src/experiments/bigger_models/train_transformer.py distilbert-base-uncased 2_distilbert
    python src/experiments/bigger_models/train_transformer.py roberta-base 3_roberta_base

Hyperparameters are read from the environment so a rerun is one line:

    EPOCHS=5 LR=5e-5 MAX_LEN=256 python src/experiments/bigger_models/train_transformer.py ...

The first run of this script used a learning rate of 3e-5 for 3 epochs at 128
tokens with no class weighting, and DistilBERT landed at 53% accuracy, well
under the 70% of the TF-IDF baseline. Training loss was still near 1.1 when it
stopped, which is underfitting rather than a limit of the model. Three things
changed after that: a higher learning rate, a longer sequence so the model sees
as much of the ticket as the baseline does, and class weights so the small
queues are not ignored.
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from common import load_meta, load_split, save_result, threshold_table

MAX_LEN = int(os.environ.get("MAX_LEN", "256"))
BATCH = int(os.environ.get("BATCH", "32"))
EPOCHS = int(os.environ.get("EPOCHS", "4"))
LR = float(os.environ.get("LR", "5e-5"))


class Tickets(Dataset):
    def __init__(self, texts, labels, tok, max_len: int = None):
        self.enc = tok(
            texts,
            truncation=True,
            max_length=max_len or MAX_LEN,
            padding="max_length",
            return_tensors="pt",
        )
        self.labels = torch.tensor(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        item = {k: v[i] for k, v in self.enc.items()}
        item["labels"] = self.labels[i]
        return item


def device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def class_weights(labels, n_labels: int) -> torch.Tensor:
    """Same idea as class_weight='balanced' in the scikit-learn baseline."""
    counts = np.bincount(labels, minlength=n_labels).astype(float)
    weights = len(labels) / (n_labels * np.maximum(counts, 1))
    return torch.tensor(weights, dtype=torch.float32)


@torch.no_grad()
def evaluate(model, loader, dev):
    model.eval()
    probs = []
    for batch in loader:
        batch = {k: v.to(dev) for k, v in batch.items()}
        logits = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).logits
        probs.append(torch.softmax(logits.float(), dim=-1).cpu().numpy())
    model.train()
    return np.concatenate(probs)


def main() -> None:
    model_name = sys.argv[1]
    tier = sys.argv[2]
    meta = load_meta()
    n_labels = len(meta["labels"])

    tok = AutoTokenizer.from_pretrained(model_name)
    # ignore_mismatched_sizes lets us start from a checkpoint whose head is the
    # wrong shape, for example an NLI model with 3 outputs that we are giving a
    # 10 queue head. The body is kept, the head is reinitialised.
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=n_labels, ignore_mismatched_sizes=True
    )
    dev = device()
    model.to(dev)

    x_tr, y_tr = load_split("train")
    x_te, y_te = load_split("test")
    dl_tr = DataLoader(Tickets(x_tr, y_tr, tok), batch_size=BATCH, shuffle=True)
    dl_te = DataLoader(Tickets(x_te, y_te, tok), batch_size=64)

    # Class weights match the baseline's class_weight="balanced", but they
    # multiply the loss on rare classes by up to about 7x, which adds a lot of
    # gradient noise. Set CLASS_WEIGHTS=0 to train without them and compare.
    use_weights = os.environ.get("CLASS_WEIGHTS", "1") == "1"
    weights = class_weights(y_tr, n_labels).to(dev) if use_weights else None
    loss_fn = nn.CrossEntropyLoss(weight=weights)
    print(f"class weights: {'on' if use_weights else 'off'}", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    steps = EPOCHS * len(dl_tr)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=LR, total_steps=steps, pct_start=0.1)

    from sklearn.metrics import accuracy_score, f1_score

    print(f"{model_name}: {EPOCHS} epochs, lr {LR}, max_len {MAX_LEN}, {steps} steps on {dev}", flush=True)
    t0 = time.time()
    model.train()
    seen = 0
    history = []
    for epoch in range(EPOCHS):
        running = []
        for batch in dl_tr:
            batch = {k: v.to(dev) for k, v in batch.items()}
            logits = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).logits
            loss = loss_fn(logits, batch["labels"])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            opt.zero_grad()
            running.append(loss.item())
            seen += 1
            if seen % 100 == 0:
                print(f"  epoch {epoch} step {seen}/{steps} loss {np.mean(running[-100:]):.4f}", flush=True)
        probs = evaluate(model, dl_te, dev)
        acc = accuracy_score(y_te, probs.argmax(axis=1))
        f1 = f1_score(y_te, probs.argmax(axis=1), average="macro")
        history.append({"epoch": epoch, "train_loss": round(float(np.mean(running)), 4),
                        "test_accuracy": round(float(acc), 4), "test_macro_f1": round(float(f1), 4)})
        print(f"  EPOCH {epoch}: loss {np.mean(running):.4f} acc {acc:.4f} macro-f1 {f1:.4f}", flush=True)
    train_seconds = time.time() - t0

    probs = evaluate(model, dl_te, dev)
    pred = probs.argmax(axis=1)
    y_true = np.asarray(y_te)

    out_dir = f"models/{tier}"
    model.save_pretrained(out_dir)
    tok.save_pretrained(out_dir)

    save_result(
        {
            "tier": tier,
            "model": model_name,
            "params_millions": round(sum(p.numel() for p in model.parameters()) / 1e6, 1),
            "train_device": dev,
            "train_seconds": round(train_seconds, 1),
            "epochs": EPOCHS,
            "lr": LR,
            "max_len": MAX_LEN,
            "history": history,
            "accuracy": round(float(accuracy_score(y_true, pred)), 4),
            "macro_f1": round(float(f1_score(y_true, pred, average="macro")), 4),
            "thresholds": threshold_table(probs, y_true),
            "model_dir": out_dir,
        }
    )
    np.save(f"results/{tier}_probs.npy", probs)


if __name__ == "__main__":
    main()
