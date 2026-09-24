"""Conjunction ("ANOVA kernel") features on sparse TF-IDF rows.

order 1: the terms themselves
order 2: every unordered pair of distinct terms, value x_a * x_b
order 3: every unordered triple of distinct terms, value x_a * x_b * x_c
Each order block is L2-normalised per ticket and scaled by a block weight, so a linear model on the
stacked blocks is a kernel machine with kernel  sum_d w_d^2 * ANOVA_d(x, y)  -- a sharp, family-sensitive
similarity -- but with a fixed-size weight vector instead of a stored training set.

Only conjunctions seen in >= min_df training tickets are kept (exact vocabulary, no hashing).
"""
from __future__ import annotations

from itertools import combinations

import numpy as np
import scipy.sparse as sp
from sklearn.preprocessing import normalize

_COMBO_CACHE: dict = {}


def _combos(k, order):
    key = (k, order)
    if key not in _COMBO_CACHE:
        if order == 2:
            ia, ib = np.triu_indices(k, 1)
            _COMBO_CACHE[key] = (ia.astype(np.int32), ib.astype(np.int32))
        else:
            arr = np.array(list(combinations(range(k), order)), dtype=np.int32).reshape(-1, order)
            _COMBO_CACHE[key] = tuple(arr[:, j] for j in range(order))
    return _COMBO_CACHE[key]


_MULT = np.uint64(0x9E3779B97F4A7C15)


def _raw(X, order, max_terms):
    """Return (rows, conj_ids, vals) of all order-way conjunctions for csr X. conj_ids are 64-bit hashes of the
    sorted term-id tuple (collisions are negligible in a 2^64 space)."""
    X = sp.csr_matrix(X)
    indptr, indices, data = X.indptr, X.indices, X.data
    R, C, W = [], [], []
    with np.errstate(over="ignore"):
        for i in range(X.shape[0]):
            idx = indices[indptr[i]:indptr[i + 1]].astype(np.uint64)
            v = data[indptr[i]:indptr[i + 1]].astype(np.float32)
            if max_terms is not None and len(idx) > max_terms:
                keep = np.argsort(-v, kind="stable")[:max_terms]
                idx, v = idx[keep], v[keep]
            if len(idx) < order:
                continue
            o = np.argsort(idx)
            idx, v = idx[o], v[o]
            cmb = _combos(len(idx), order)
            c = idx[cmb[0]] + np.uint64(1)
            w = v[cmb[0]]
            for j in range(1, order):
                c = (c * _MULT) ^ (idx[cmb[j]] + np.uint64(j + 1) * np.uint64(0x100000001B3))
                w = w * v[cmb[j]]
            R.append(np.full(len(c), i, dtype=np.int32)); C.append(c); W.append(w)
    if not R:
        return np.zeros(0, np.int32), np.zeros(0, np.uint64), np.zeros(0, np.float32)
    return np.concatenate(R), np.concatenate(C), np.concatenate(W)


class Conjunctions:
    def __init__(self, order=2, min_df=2, max_terms=None):
        self.order, self.min_df, self.max_terms = order, min_df, max_terms

    def fit_transform(self, X):
        rows, cols, vals = _raw(X, self.order, self.max_terms)
        uniq, inv, counts = np.unique(cols, return_inverse=True, return_counts=True)
        keep_u = counts >= self.min_df
        self.vocab_ = uniq[keep_u]
        remap = -np.ones(len(uniq), dtype=np.int64)
        remap[keep_u] = np.arange(int(keep_u.sum()))
        new = remap[inv]
        m = new >= 0
        M = sp.csr_matrix((vals[m], (rows[m], new[m])), shape=(X.shape[0], len(self.vocab_)), dtype=np.float32)
        return normalize(M)

    def transform(self, X):
        rows, cols, vals = _raw(X, self.order, self.max_terms)
        if len(self.vocab_) == 0 or len(cols) == 0:
            return sp.csr_matrix((X.shape[0], len(self.vocab_)), dtype=np.float32)
        pos = np.searchsorted(self.vocab_, cols)
        pos[pos >= len(self.vocab_)] = 0
        m = self.vocab_[pos] == cols
        M = sp.csr_matrix((vals[m], (rows[m], pos[m])), shape=(X.shape[0], len(self.vocab_)), dtype=np.float32)
        return normalize(M)
