"""Diagonal NCA: learn one positive weight per term so that, inside the training set, a ticket's soft nearest
neighbours (leave-one-out) carry its own queue. Families are paraphrases, so the weights learn which words
survive paraphrasing (product names, technical nouns) and which are noise (politeness, filler verbs).

Only training tickets are used. The result is a vector of term weights -- a fixed-size parametric similarity.
"""
from __future__ import annotations

import time

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F


def fit_term_weights(X, y, steps=150, lr=0.05, gamma0=25.0, anchors=3000, l2=1e-3, seed=0, device=None, verbose=False):
    """X: csr [n, V] un-normalised tf-idf rows (or already normalised; rows are re-normalised after weighting)."""
    device = device or (torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu"))
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    Xd = torch.from_numpy(sp.csr_matrix(X, dtype=np.float32).toarray()).to(device)
    yt = torch.from_numpy(np.asarray(y, dtype=np.int64)).to(device)
    n, V = Xd.shape
    theta = torch.zeros(V, device=device, requires_grad=True)
    log_gamma = torch.tensor(float(np.log(gamma0)), device=device, requires_grad=True)
    opt = torch.optim.Adam([theta, log_gamma], lr=lr)
    t0 = time.time()
    for step in range(steps):
        Z = F.normalize(Xd * torch.exp(theta), dim=1)
        a = torch.from_numpy(rng.choice(n, min(anchors, n), replace=False)).to(device)
        S = Z[a] @ Z.T                                      # [anchors, n]
        S[torch.arange(len(a), device=device), a] = -1e4     # leave-one-out: a ticket may not vote for itself
        logp = F.log_softmax(torch.exp(log_gamma) * S, dim=1)
        same = (yt[a][:, None] == yt[None, :])
        ll = torch.logsumexp(logp.masked_fill(~same, -1e4), dim=1)
        loss = -ll.mean() + l2 * (theta ** 2).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if verbose and (step % 25 == 0 or step == steps - 1):
            print(f"  nca step {step} loss {float(loss):.4f} gamma {float(torch.exp(log_gamma)):.1f} "
                  f"theta[min,max]=[{float(theta.min()):.2f},{float(theta.max()):.2f}] {time.time() - t0:.0f}s", flush=True)
    return np.exp(theta.detach().cpu().numpy()).astype(np.float32), float(torch.exp(log_gamma))
