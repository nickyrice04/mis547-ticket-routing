"""Top tier: fine-tune a small language model (SLM) with LoRA.

A decoder-only language model is roughly 10x to 25x larger than DistilBERT. We
adapt it for queue classification with LoRA, which trains a small number of
extra weights instead of all of them. That keeps training on one machine
possible, and it lets us measure what the extra size buys in accuracy and what
it costs in memory and latency.

    python src/experiments/bigger_models/train_slm.py Qwen/Qwen2.5-0.5B 5_qwen_0_5b
    python src/experiments/bigger_models/train_slm.py Qwen/Qwen2.5-1.5B 6_qwen_1_5b
"""
from __future__ import annotations

import sys
import time

import numpy as np
import torch
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from common import load_meta, load_split, save_result, threshold_table
from experiments.bigger_models.train_transformer import BATCH, EPOCHS, MAX_LEN, Tickets, device

import os

# LoRA trains a small set of extra weights, so it wants a larger learning rate
# than full fine-tuning does.
LR = float(os.environ.get("SLM_LR", "2e-4"))


def main() -> None:
    model_name = sys.argv[1]
    tier = sys.argv[2]
    epochs = int(sys.argv[3]) if len(sys.argv) > 3 else EPOCHS
    meta = load_meta()

    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=len(meta["labels"]), torch_dtype=torch.float32
    )
    model.config.pad_token_id = tok.pad_token_id
    base_params = sum(p.numel() for p in model.parameters())

    lora = LoraConfig(
        task_type="SEQ_CLS",
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )
    model = get_peft_model(model, lora)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"base params {base_params/1e6:.1f}M | trainable {trainable/1e6:.2f}M", flush=True)

    dev = device()
    model.to(dev)

    x_tr, y_tr = load_split("train")
    x_te, y_te = load_split("test")
    batch_size = BATCH // 2 if base_params > 8e8 else BATCH
    dl_tr = DataLoader(Tickets(x_tr, y_tr, tok), batch_size=batch_size, shuffle=True)
    dl_te = DataLoader(Tickets(x_te, y_te, tok), batch_size=batch_size)

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=LR)
    steps = epochs * len(dl_tr)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=LR, total_steps=steps, pct_start=0.1)

    t0 = time.time()
    model.train()
    seen = 0
    for epoch in range(epochs):
        for batch in dl_tr:
            batch = {k: v.to(dev) for k, v in batch.items()}
            loss = model(**batch).loss
            loss.backward()
            opt.step()
            sched.step()
            opt.zero_grad()
            seen += 1
            if seen % 50 == 0:
                print(f"epoch {epoch} step {seen}/{steps} loss {loss.item():.4f}", flush=True)
    train_seconds = time.time() - t0

    model.eval()
    all_probs = []
    with torch.no_grad():
        for batch in dl_te:
            batch = {k: v.to(dev) for k, v in batch.items()}
            logits = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).logits
            all_probs.append(torch.softmax(logits.float(), dim=-1).cpu().numpy())
    probs = np.concatenate(all_probs)
    pred = probs.argmax(axis=1)
    y_true = np.asarray(y_te)

    from sklearn.metrics import accuracy_score, f1_score

    out_dir = f"models/{tier}"
    model.save_pretrained(out_dir)
    tok.save_pretrained(out_dir)

    save_result(
        {
            "tier": tier,
            "model": f"{model_name} + LoRA",
            "params_millions": round(base_params / 1e6, 1),
            "trainable_millions": round(trainable / 1e6, 2),
            "train_device": dev,
            "train_seconds": round(train_seconds, 1),
            "epochs": epochs,
            "accuracy": round(float(accuracy_score(y_true, pred)), 4),
            "macro_f1": round(float(f1_score(y_true, pred, average="macro")), 4),
            "thresholds": threshold_table(probs, y_true),
            "model_dir": out_dir,
        }
    )
    np.save(f"results/{tier}_probs.npy", probs)


if __name__ == "__main__":
    main()
