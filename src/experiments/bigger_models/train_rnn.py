"""A real LSTM and GRU over word sequences.

Worth being precise about what this adds, because it is easy to build one of
these that does nothing at all.

A common pattern in notebooks is to feed a TF-IDF vector into an LSTM like this:

    x = x.unsqueeze(1)          # (batch, 5000) becomes (batch, 1, 5000)
    out, _ = self.lstm(x)

That is a sequence of length one. An LSTM over a single timestep has nothing to
remember, so it reduces to a dense layer with extra gates, and it cannot see
word order any more than a bag of words can. It looks like a recurrent model in
the code and behaves like a slightly worse feedforward net in practice.

This version does the real thing. Each ticket becomes a sequence of word ids,
an embedding layer turns those into vectors, and a bidirectional recurrent layer
reads them in order. That is a genuinely different model class from everything
else in our study, because it is the only one besides the transformers that
knows "refund was not processed" differs from "processed a refund".

    python src/experiments/bigger_models/train_rnn.py lstm 7_lstm
    python src/experiments/bigger_models/train_rnn.py gru  7_gru
"""
from __future__ import annotations

import collections
import re
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader, TensorDataset

from common import load_meta, load_split, save_result, threshold_table

VOCAB = 30_000
MAX_LEN = 300
EMBED = 128
HIDDEN = 128
EPOCHS = 8
BATCH = 64
LR = 1e-3


def tokenize(text: str):
    return re.findall(r"[a-z0-9']+", text)


def build_vocab(texts):
    counts = collections.Counter(w for t in texts for w in tokenize(t))
    words = [w for w, _ in counts.most_common(VOCAB - 2)]
    return {w: i + 2 for i, w in enumerate(words)}   # 0 pad, 1 unknown


def encode(texts, vocab):
    out = np.zeros((len(texts), MAX_LEN), dtype=np.int64)
    for i, t in enumerate(texts):
        ids = [vocab.get(w, 1) for w in tokenize(t)][:MAX_LEN]
        out[i, : len(ids)] = ids
    return out


class RNNClassifier(nn.Module):
    def __init__(self, kind: str, n_labels: int):
        super().__init__()
        self.embed = nn.Embedding(VOCAB, EMBED, padding_idx=0)
        cell = nn.LSTM if kind == "lstm" else nn.GRU
        self.rnn = cell(EMBED, HIDDEN, batch_first=True, bidirectional=True, num_layers=1)
        self.dropout = nn.Dropout(0.3)
        # Mean and max over time, concatenated. Pooling across the sequence beats
        # taking only the final state, which forgets the start of a long ticket.
        self.fc = nn.Linear(HIDDEN * 4, n_labels)

    def forward(self, x):
        mask = (x != 0).unsqueeze(-1).float()
        h, _ = self.rnn(self.embed(x))
        h = h * mask
        avg = h.sum(1) / mask.sum(1).clamp(min=1)
        mx = h.masked_fill(mask == 0, -1e9).max(1).values
        return self.fc(self.dropout(torch.cat([avg, mx], dim=-1)))


def main() -> None:
    kind, tier = sys.argv[1], sys.argv[2]
    meta = load_meta()
    n_labels = len(meta["labels"])
    x_tr, y_tr = load_split("train")
    x_te, y_te = load_split("test")

    vocab = build_vocab(x_tr)
    X_tr = torch.from_numpy(encode(x_tr, vocab))
    X_te = torch.from_numpy(encode(x_te, vocab))
    Y_tr = torch.tensor(y_tr)

    dev = "cpu"   # small model, and the GPU is busy with the language model
    torch.manual_seed(42)
    model = RNNClassifier(kind, n_labels).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    loss_fn = nn.CrossEntropyLoss()
    loader = DataLoader(TensorDataset(X_tr, Y_tr), batch_size=BATCH, shuffle=True)

    print(f"{kind}: {sum(p.numel() for p in model.parameters())/1e6:.1f}M parameters", flush=True)
    t0 = time.time()
    history = []
    for epoch in range(EPOCHS):
        model.train()
        losses = []
        for xb, yb in loader:
            loss = loss_fn(model(xb.to(dev)), yb.to(dev))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            opt.zero_grad()
            losses.append(loss.item())
        model.eval()
        with torch.no_grad():
            probs = torch.softmax(model(X_te.to(dev)), dim=-1).cpu().numpy()
        acc = accuracy_score(y_te, probs.argmax(1))
        history.append({"epoch": epoch, "train_loss": round(float(np.mean(losses)), 4),
                        "test_accuracy": round(float(acc), 4)})
        print(f"  epoch {epoch} loss {np.mean(losses):.4f} acc {acc:.4f}", flush=True)
    seconds = time.time() - t0

    pred = probs.argmax(1)
    save_result({
        "tier": tier,
        "model": f"bidirectional {kind.upper()} over word sequences",
        "params_millions": round(sum(p.numel() for p in model.parameters()) / 1e6, 1),
        "train_device": dev,
        "train_seconds": round(seconds, 1),
        "epochs": EPOCHS,
        "history": history,
        "accuracy": round(float(accuracy_score(y_te, pred)), 4),
        "macro_f1": round(float(f1_score(y_te, pred, average="macro")), 4),
        "thresholds": threshold_table(probs, y_te),
    })
    np.save(f"results/{tier}_probs.npy", probs)


if __name__ == "__main__":
    main()
