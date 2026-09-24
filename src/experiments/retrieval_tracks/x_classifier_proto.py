"""Prototype MLP: a two-layer network whose hidden units are "family detectors".

    h_k(x) = relu(p_k . x) ** power          p_k: unit-norm prototype in unigram TF-IDF space
    score  = h(x) @ B + b

It is an ordinary MLP (Linear -> polynomial activation -> Linear) with a fixed number K of hidden units, but
instead of a random start the first layer starts at class-wise spherical k-means centroids of the training
tickets and the second layer at the ridge solution. With power around 6 the unit only fires for tickets of its
own paraphrase family, which is the sharpness a ReLU MLP trained from a random start never finds.
"""
from __future__ import annotations

import numpy as np
import scipy.linalg as sla
import scipy.sparse as sp
from sklearn.preprocessing import normalize


def classwise_kmeans(X, y, K, n_iter=8, seed=0, n_classes=10):
    """Spherical k-means run separately inside each class. Returns dense centroids [K', V] and their class."""
    rng = np.random.default_rng(seed)
    X = sp.csr_matrix(X, dtype=np.float32)
    n = X.shape[0]
    cents, cls = [], []
    for c in range(n_classes):
        idx = np.where(y == c)[0]
        if len(idx) == 0:
            continue
        kc = int(min(len(idx), max(1, round(K * len(idx) / n))))
        Xc = X[idx]
        C = Xc[rng.choice(len(idx), kc, replace=False)].toarray()
        if kc < len(idx):
            for _ in range(n_iter):
                S = Xc @ C.T                                   # dense [n_c, k_c]
                a = np.asarray(S.argmax(1)).ravel()
                M = sp.csr_matrix((np.ones(len(idx), np.float32), (a, np.arange(len(idx)))), shape=(kc, len(idx)))
                newC = (M @ Xc).toarray()
                empty = np.where(np.abs(newC).sum(1) == 0)[0]
                if len(empty):
                    # re-seed empty clusters with the tickets that fit their centroid worst
                    worst = np.argsort(np.asarray(S.max(1)).ravel())[: len(empty)]
                    newC[empty] = Xc[worst].toarray()
                C = normalize(newC)
        cents.append(C.astype(np.float32)); cls.append(np.full(kc, c))
    return np.vstack(cents), np.concatenate(cls)


def hidden(X, P, power, chunk=4096):
    X = sp.csr_matrix(X, dtype=np.float32)
    out = np.empty((X.shape[0], P.shape[0]), dtype=np.float32)
    for s in range(0, X.shape[0], chunk):
        z = np.asarray(X[s:s + chunk] @ P.T)
        np.clip(z, 0, None, out=z)
        out[s:s + chunk] = z ** power
    return out


def ridge_readout(H, y, lam, n_classes=10):
    Y = -np.ones((H.shape[0], n_classes), dtype=np.float64); Y[np.arange(H.shape[0]), y] = 1
    H64 = H.astype(np.float64)
    if H.shape[1] <= H.shape[0]:
        A = H64.T @ H64; A[np.diag_indices_from(A)] += lam
        return sla.solve(A, H64.T @ Y, assume_a="pos")
    A = H64 @ H64.T; A[np.diag_indices_from(A)] += lam
    return H64.T @ sla.solve(A, Y, assume_a="pos")
