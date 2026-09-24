"""Multinomial logistic regression (softmax, cross-entropy) on very wide sparse features, trained with torch.
The weight matrix is dense [n_features, n_classes]; batches are handled by gather + index_add so the device
never materialises a dense [batch, n_features] array (n_features can be ~10M).
"""
from __future__ import annotations

import math
import time

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F


def pick_device():
    return torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")


def _batch(X, idx, device, p_in=0.0, rng=None):
    sub = X[idx].tocoo()
    vals = sub.data.astype(np.float32)
    rows, cols = sub.row, sub.col
    if p_in > 0:
        keep = rng.random(len(vals)) >= p_in
        rows, cols, vals = rows[keep], cols[keep], vals[keep] / (1.0 - p_in)
    return (torch.from_numpy(rows.astype(np.int64)).to(device), torch.from_numpy(cols.astype(np.int64)).to(device),
            torch.from_numpy(vals).to(device))


def _forward(W, b, rows, cols, vals, n_rows):
    contrib = W[cols] * vals[:, None]
    out = torch.zeros((n_rows, W.shape[1]), dtype=W.dtype, device=W.device)
    out.index_add_(0, rows, contrib)
    return out + b


def train_sparse_softmax(X, y, n_classes=10, epochs=20, lr=0.05, wd=0.0, ls=0.0, p_in=0.0, bs=256, seed=0,
                         device=None, verbose=False, X_eval=None, y_eval=None, scale=1.0):
    device = device or pick_device()
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    X = sp.csr_matrix(X, dtype=np.float32)
    n, d = X.shape
    W = torch.zeros((d, n_classes), dtype=torch.float32, device=device, requires_grad=True)
    b = torch.zeros(n_classes, dtype=torch.float32, device=device, requires_grad=True)
    opt = torch.optim.Adam([W, b], lr=lr)
    steps = epochs * math.ceil(n / bs)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.1)
    yt = torch.from_numpy(np.asarray(y, dtype=np.int64)).to(device)
    t0 = time.time()
    for ep in range(epochs):
        perm = rng.permutation(n)
        tot = 0.0
        for s in range(0, n, bs):
            idx = np.sort(perm[s:s + bs])
            rows, cols, vals = _batch(X, idx, device, p_in, rng)
            logits = _forward(W, b, rows, cols, vals * scale, len(idx))
            loss = F.cross_entropy(logits, yt[torch.from_numpy(idx).to(device)], label_smoothing=ls)
            if wd > 0:
                # L2 only on the rows touched by this batch (lazy regularisation keeps the step cheap)
                loss = loss + 0.5 * wd * (W[cols.unique()] ** 2).sum() / len(idx)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step(); sched.step()
            tot += float(loss.detach()) * len(idx)
        if verbose and (ep % 5 == 4 or ep == epochs - 1):
            msg = f"  ep {ep + 1}/{epochs} loss {tot / n:.4f} {time.time() - t0:.0f}s"
            if X_eval is not None:
                acc = (predict_proba_sparse((W, b), X_eval, scale=scale).argmax(1) == np.asarray(y_eval)).mean()
                msg += f" eval_acc {acc:.4f}"
            print(msg, flush=True)
    return W.detach(), b.detach()


@torch.no_grad()
def predict_proba_sparse(model, X, bs=512, scale=1.0):
    W, b = model
    X = sp.csr_matrix(X, dtype=np.float32)
    out = []
    for s in range(0, X.shape[0], bs):
        idx = np.arange(s, min(s + bs, X.shape[0]))
        rows, cols, vals = _batch(X, idx, W.device)
        out.append(F.softmax(_forward(W, b, rows, cols, vals * scale, len(idx)), dim=1).cpu().numpy())
    return np.concatenate(out)
