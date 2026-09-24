"""Track 'lexical': sparse similarity views (word/char TF-IDF, BM25) and kNN voting utilities."""
from __future__ import annotations
import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.preprocessing import normalize


class BM25View:
    """BM25 scores, query side = binary/idf-less term presence, doc side = BM25 weights.
    Scores are normalised by the query's self-score upper bound so they sit roughly in [0,1]."""
    def __init__(self, k1=1.2, b=0.75, ngram_range=(1, 1), min_df=2):
        self.k1, self.b = k1, b
        self.cv = CountVectorizer(ngram_range=ngram_range, min_df=min_df)

    def fit(self, texts):
        C = self.cv.fit_transform(texts).astype(np.float32).tocsr()
        n = C.shape[0]
        df = np.asarray((C > 0).sum(0)).ravel()
        self.idf = np.log(1 + (n - df + 0.5) / (df + 0.5)).astype(np.float32)
        dl = np.asarray(C.sum(1)).ravel(); self.avgdl = dl.mean()
        self.D = self._doc(C, dl)
        return self

    def _doc(self, C, dl):
        C = C.tocoo()
        denom = C.data + self.k1 * (1 - self.b + self.b * dl[C.row] / self.avgdl)
        w = C.data * (self.k1 + 1) / denom * self.idf[C.col]
        return sp.csr_matrix((w, (C.row, C.col)), shape=C.shape, dtype=np.float32)

    def sims(self, texts):
        Q = (self.cv.transform(texts) > 0).astype(np.float32).tocsr()
        S = (Q @ self.D.T).toarray()
        ub = np.asarray(Q @ (self.idf * (self.k1 + 1))).ravel() if False else np.asarray(Q.multiply(self.idf).sum(1)).ravel() * (self.k1 + 1)
        return S / np.maximum(ub[:, None], 1e-6)


class TfidfView:
    def __init__(self, **kw):
        kw.setdefault("sublinear_tf", True)
        kw.setdefault("dtype", np.float32)
        self.vec = TfidfVectorizer(**kw)

    def fit(self, texts):
        self.D = self.vec.fit_transform(texts).tocsr()
        return self

    def transform(self, texts):
        return self.vec.transform(texts)

    def sims(self, texts):
        return (self.vec.transform(texts) @ self.D.T).toarray()


VIEWS = {
    "w12": lambda: TfidfView(ngram_range=(1, 2), min_df=2, max_features=200_000),
    "w11": lambda: TfidfView(ngram_range=(1, 1), min_df=1),
    "w13": lambda: TfidfView(ngram_range=(1, 3), min_df=2, max_features=400_000),
    "w12_nosub": lambda: TfidfView(ngram_range=(1, 2), min_df=2, max_features=200_000, sublinear_tf=False),
    "w12_df1": lambda: TfidfView(ngram_range=(1, 2), min_df=1),
    "w12_stop": lambda: TfidfView(ngram_range=(1, 2), min_df=2, stop_words="english"),
    "c25wb": lambda: TfidfView(analyzer="char_wb", ngram_range=(2, 5), min_df=2, max_features=500_000),
    "c35": lambda: TfidfView(analyzer="char", ngram_range=(3, 5), min_df=2, max_features=500_000),
    "c36wb": lambda: TfidfView(analyzer="char_wb", ngram_range=(3, 6), min_df=2, max_features=800_000),
    "bm25": lambda: BM25View(),
    "bm25_12": lambda: BM25View(ngram_range=(1, 2)),
}


def topk(S, k):
    idx = np.argpartition(-S, k - 1, axis=1)[:, :k]
    val = np.take_along_axis(S, idx, 1)
    o = np.argsort(-val, axis=1)
    return np.take_along_axis(idx, o, 1), np.take_along_axis(val, o, 1)


def vote(idx, val, y_pool, n_classes, k, power):
    """Similarity-weighted votes over the top-k. Returns [n, C] scores normalised to sum 1."""
    idx, val = idx[:, :k], np.maximum(val[:, :k], 0) ** power
    out = np.zeros((idx.shape[0], n_classes), dtype=np.float64)
    rows = np.repeat(np.arange(idx.shape[0]), k)
    np.add.at(out, (rows, y_pool[idx].ravel()), val.ravel())
    return out / np.maximum(out.sum(1, keepdims=True), 1e-12)
