"""Track "classifier": the best single classifier on sparse features found by this track.

What the track learned
    The tickets come in paraphrase families that share a queue, and outside a family the queue is close to
    random. A linear model or an MLP on TF-IDF (they behave the same here, trained to convergence or not)
    scores each word on its own, so it cannot express "these words TOGETHER identify the family". Matching
    1-nearest-neighbour needs a SHARP similarity: a ticket must count only when most of its content matches.

Two models live here. Both are fitted only on the (train_texts, train_labels) passed in.

    method="krr" (default, most accurate)
        A two-layer network with one hidden unit per training ticket, h_i(x) = s(x, x_i)^p, where s is the mean
        cosine over three sparse TF-IDF views (content unigrams, uni+bigrams, character 3-5 grams), and a
        least-squares readout solved in closed form. That is kernel ridge regression with a sharp polynomial
        kernel. The exponent p is chosen by exact leave-one-out inside the training set (closed form, no
        validation data), and the same leave-one-out scores set the softmax temperature for the probabilities.
        Honest label: this is a memory-based model. It stores the training matrix (a few tens of MB) and its
        serving cost is the same sparse product a kNN index needs.

    method="conj" (strictly parametric, fixed-size weight table)
        Ridge classifier on explicit conjunction features: all pairs and triples of content words in a ticket
        (the feature maps of the degree-2 and degree-3 ANOVA kernels), kept if seen in >= 2 training tickets.

    python src/experiments/retrieval_tracks/x_classifier_final.py            # core -> validation with the default method
    python src/experiments/retrieval_tracks/x_classifier_final.py conj       # same for the parametric model
"""
from __future__ import annotations

import sys
import time
import warnings

import numpy as np
import scipy.linalg as sla
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import RidgeClassifier

N_CLASSES = 10
VIEWS = {
    "U": dict(ngram_range=(1, 1), min_df=2, sublinear_tf=True, stop_words="english"),
    "B": dict(ngram_range=(1, 2), min_df=2, max_features=200_000, sublinear_tf=True),
    "C": dict(analyzer="char_wb", ngram_range=(3, 5), min_df=2, sublinear_tf=True),
}
KRR_CONFIG = dict(views=("U", "B", "C"), powers=(4, 6, 8), lam=0.01)
CONJ_CONFIG = dict(stop=True, weights=(0.5, 1.0, 1.0), min_df=2, max_terms=40, alpha=0.1, max_iter=300, calibrate=True)


def _softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def _fit_temperature(scores, y):
    """Softmax temperature that minimises log-loss of held-out (or leave-one-out) scores of TRAINING tickets."""
    best_t, best_ll = 1.0, np.inf
    for t in np.geomspace(0.5, 1000, 60):
        p = _softmax(t * scores)
        ll = -np.mean(np.log(np.clip(p[np.arange(len(y)), y], 1e-9, None)))
        if ll < best_ll:
            best_t, best_ll = t, ll
    return best_t


# ----------------------------------------------------------------------------------------------- kernel ridge
class SharpKernelRidge:
    """Multi-view polynomial-kernel ridge classifier with closed-form leave-one-out model selection."""

    def __init__(self, **kw):
        self.cfg = dict(KRR_CONFIG); self.cfg.update(kw)

    @staticmethod
    def _gram(a, b=None):
        """Cosine similarities between L2-normalised sparse rows, as a dense float32 array."""
        b = a if b is None else b
        if a.nnz / max(1, a.shape[0]) > 300 and a.shape[1] < 80_000:   # dense BLAS is faster for the char view
            ad = a.toarray()
            bd = ad if b is a else b.toarray()
            return (bd @ ad.T).astype(np.float32)
        return (b @ a.T).toarray().astype(np.float32)

    def fit(self, texts, labels, verbose=False):
        t0 = time.time()
        y = np.asarray(labels)
        n = len(y)
        self.vecs, self.mats = {}, {}
        S = np.zeros((n, n), dtype=np.float32)
        for name in self.cfg["views"]:
            v = TfidfVectorizer(**VIEWS[name])
            a = v.fit_transform(texts).astype(np.float32)
            self.vecs[name], self.mats[name] = v, a
            S += self._gram(a)
            if verbose:
                print(f"  view {name}: {a.shape[1]:,} features ({time.time() - t0:.0f}s)", flush=True)
        S /= len(self.cfg["views"])
        np.clip(S, 0, None, out=S)
        Y = -np.ones((n, N_CLASSES)); Y[np.arange(n), y] = 1
        best = None
        self.loo_ = {}
        for p in self.cfg["powers"]:
            A = S.astype(np.float64) ** p
            A[np.diag_indices_from(A)] += self.cfg["lam"]
            c, low = sla.cho_factor(A, lower=True, overwrite_a=True, check_finite=False)
            G = sla.cho_solve((c, low), np.eye(n), overwrite_b=True, check_finite=False)   # (K + lam I)^-1
            alpha = G @ Y
            loo = Y - alpha / np.diag(G)[:, None]                                          # exact leave-one-out scores
            acc = float((loo.argmax(1) == y).mean())
            self.loo_[p] = acc
            if verbose:
                print(f"  power {p}: leave-one-out accuracy inside training set {acc:.4f} ({time.time() - t0:.0f}s)", flush=True)
            if best is None or acc > best[0]:
                best = (acc, p, alpha, loo)
            del A, G, c
        _, self.power_, self.alpha_, loo = best
        self.temperature_ = _fit_temperature(loo, y)
        self.loo_scores_ = loo
        if verbose:
            print(f"  chosen power={self.power_}, temperature={self.temperature_:.2f}", flush=True)
        return self

    def scores(self, texts):
        S = None
        for name in self.cfg["views"]:
            b = self.vecs[name].transform(texts).astype(np.float32)
            g = self._gram(self.mats[name], b)
            S = g if S is None else S + g
        S /= len(self.cfg["views"])
        np.clip(S, 0, None, out=S)
        return (S.astype(np.float64) ** self.power_) @ self.alpha_

    def predict_proba(self, texts, chunk=2000):
        out = [_softmax(self.temperature_ * self.scores(texts[s:s + chunk])) for s in range(0, len(texts), chunk)]
        return np.vstack(out)


# ------------------------------------------------------------------------------------- parametric conjunctions
class ConjRidge:
    def __init__(self, **kw):
        self.cfg = dict(CONJ_CONFIG); self.cfg.update(kw)

    def _features(self, texts, fit):
        from experiments.retrieval_tracks.x_classifier_conj import Conjunctions
        c = self.cfg
        if fit:
            self.vec = TfidfVectorizer(ngram_range=(1, 1), min_df=2, sublinear_tf=True,
                                       stop_words="english" if c["stop"] else None)
            U = self.vec.fit_transform(texts)
            self.conj = {o: Conjunctions(order=o, min_df=c["min_df"], max_terms=c["max_terms"] if o >= 3 else None)
                         for o, w in zip((2, 3), c["weights"][1:]) if w > 0}
        else:
            U = self.vec.transform(texts)
        blocks = [c["weights"][0] * U] if c["weights"][0] > 0 else []
        for o, cj in self.conj.items():
            blocks.append(c["weights"][o - 1] * (cj.fit_transform(U) if fit else cj.transform(U)))
        return sp.hstack(blocks).tocsr().astype(np.float32)

    def fit(self, texts, labels):
        X = self._features(texts, fit=True)
        self.n_features_ = X.shape[1]
        self.clf = RidgeClassifier(alpha=self.cfg["alpha"], solver="sparse_cg", max_iter=self.cfg["max_iter"], tol=1e-4)
        self.clf.fit(X, np.asarray(labels))
        return self

    def scores(self, texts):
        s = self.clf.decision_function(self._features(texts, fit=False))
        full = np.full((s.shape[0], N_CLASSES), -1.0)       # a class absent from training stays at -1
        full[:, self.clf.classes_] = s
        return full


def _conj_predict_proba(train_texts, y, eval_texts, cfg, verbose):
    t0 = time.time()
    temp = 8.0
    if cfg["calibrate"]:
        perm = np.random.default_rng(0).permutation(len(y))
        hold, keep = perm[: len(y) // 5], perm[len(y) // 5:]
        inner = ConjRidge(**cfg).fit([train_texts[i] for i in keep], y[keep])
        temp = _fit_temperature(inner.scores([train_texts[i] for i in hold]), y[hold])
        del inner
    model = ConjRidge(**cfg).fit(train_texts, y)
    if verbose:
        print(f"  conj ridge: {model.n_features_:,} features, temperature={temp:.2f} ({time.time() - t0:.0f}s)", flush=True)
    return _softmax(temp * model.scores(eval_texts))


# ------------------------------------------------------------------------------------------------ entry point
def fit_predict_proba(train_texts, train_labels, eval_texts, method="krr", verbose=False, **overrides):
    warnings.filterwarnings("ignore")
    train_texts = list(train_texts); y = np.asarray(train_labels); eval_texts = list(eval_texts)
    if method == "conj":
        cfg = dict(CONJ_CONFIG); cfg.update(overrides)
        return _conj_predict_proba(train_texts, y, eval_texts, cfg, verbose)
    model = SharpKernelRidge(**overrides).fit(train_texts, y, verbose=verbose)
    return model.predict_proba(eval_texts)


if __name__ == "__main__":
    from sklearn.metrics import accuracy_score, f1_score

    from experiments.retrieval_tracks.x_classifier_lib import load_core_val

    method = sys.argv[1] if len(sys.argv) > 1 else "krr"
    xc, yc, xv, yv = load_core_val()
    t0 = time.time()
    P = fit_predict_proba(xc, yc, xv, method=method, verbose=True)
    pred = P.argmax(1)
    print(f"[{method}] core -> validation: accuracy={accuracy_score(yv, pred):.4f} "
          f"macro_f1={f1_score(yv, pred, average='macro'):.4f} ({(time.time() - t0) / 60:.1f} min)")
