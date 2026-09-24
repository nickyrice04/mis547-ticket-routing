"""Explicit degree-2 conjunction features: every unordered pair of distinct terms in a ticket becomes one
sparse feature with value sqrt(2) * x_a * x_b (plus the squares x_a^2), which is the exact feature map of
the polynomial kernel (x . y)^2. A linear model on these features is a parametric stand-in for a sharp
kernel machine.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp

SQRT2 = np.sqrt(2.0)


def pair_matrix(X, n_hash=None, squares=True, max_terms=None):
    """X: csr [n, V] (rows L2-normalised). Returns csr [n, V*V] (or [n, n_hash] if hashing) of pair products.

    Column id for pair (a<b) is a*V+b. With n_hash the id is multiplicatively hashed into n_hash buckets
    with a +-1 sign to keep collisions unbiased.
    """
    X = sp.csr_matrix(X)
    V = X.shape[1]
    indptr, indices, data = X.indptr, X.indices, X.data
    rows, cols, vals = [], [], []
    tri_cache = {}
    for i in range(X.shape[0]):
        idx = indices[indptr[i]:indptr[i + 1]].astype(np.int64)
        v = data[indptr[i]:indptr[i + 1]].astype(np.float64)
        if max_terms is not None and len(idx) > max_terms:
            keep = np.argsort(-v)[:max_terms]
            idx, v = idx[keep], v[keep]
        order = np.argsort(idx)
        idx, v = idx[order], v[order]
        k = len(idx)
        if k not in tri_cache:
            tri_cache[k] = np.triu_indices(k, 1)
        ia, ib = tri_cache[k]
        c = idx[ia] * V + idx[ib]
        w = SQRT2 * v[ia] * v[ib]
        if squares:
            c = np.concatenate([c, idx * V + idx])
            w = np.concatenate([w, v * v])
        rows.append(np.full(len(c), i, dtype=np.int64)); cols.append(c); vals.append(w)
    rows = np.concatenate(rows); cols = np.concatenate(cols); vals = np.concatenate(vals).astype(np.float32)
    if n_hash is None:
        return rows, cols, vals, V * V
    h = (cols * np.int64(0x9E3779B97F4A7C15 & 0x7FFFFFFFFFFFFFFF)) & np.int64(0x7FFFFFFFFFFFFFFF)
    sign = np.where((h >> 40) & 1, 1.0, -1.0).astype(np.float32)
    bucket = (h >> 8) % n_hash
    M = sp.csr_matrix((vals * sign, (rows, bucket)), shape=(X.shape[0], n_hash), dtype=np.float32)
    M.sum_duplicates()
    return M


class PairIndexer:
    """Exact (non-hashed) pair features: keeps only pairs seen in >= min_df training tickets."""

    def __init__(self, min_df=2, squares=True, max_terms=None):
        self.min_df, self.squares, self.max_terms = min_df, squares, max_terms

    def fit_transform(self, X):
        rows, cols, vals, _ = pair_matrix(X, None, self.squares, self.max_terms)
        uniq, inv, counts = np.unique(cols, return_inverse=True, return_counts=True)
        keep_u = counts >= self.min_df
        self.vocab_ = uniq[keep_u]
        remap = -np.ones(len(uniq), dtype=np.int64)
        remap[keep_u] = np.arange(keep_u.sum())
        new = remap[inv]
        m = new >= 0
        return sp.csr_matrix((vals[m], (rows[m], new[m])), shape=(X.shape[0], len(self.vocab_)), dtype=np.float32)

    def transform(self, X):
        rows, cols, vals, _ = pair_matrix(X, None, self.squares, self.max_terms)
        pos = np.searchsorted(self.vocab_, cols)
        pos[pos >= len(self.vocab_)] = 0
        m = self.vocab_[pos] == cols
        return sp.csr_matrix((vals[m], (rows[m], pos[m])), shape=(X.shape[0], len(self.vocab_)), dtype=np.float32)
