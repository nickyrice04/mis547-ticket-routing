"""Does adding real German tickets improve accuracy on English tickets?

The learning curve said we are data limited: accuracy was still rising 4 points
over the last fifth of the training set. The cheapest real extra data is the
German half of the dataset, which we filtered out at the start.

A bag of words cannot use it, because German and English share almost no
vocabulary. A multilingual embedding model can, because it maps a German ticket
about a refund close to an English ticket about a refund. So every model in
this script sees embeddings, not words, and the only thing that changes between
runs is how much German is added to the training set.

Two guards, because we have been bitten by leakage once already:

  * the German rows keep only the ten queues the English data uses. The German
    file also contains 42 unrelated category labels from a different scheme.
  * any German ticket whose embedding is a near match for an English test
    ticket is dropped. A German translation of a test ticket would let the
    model see the answer, the same way the duplicates did.
"""
from __future__ import annotations

import json
import time

import numpy as np
import pyarrow.parquet as pq
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics.pairwise import linear_kernel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neural_network import MLPClassifier

from common import RESULTS, clean, load_meta, load_split

MODEL = "intfloat/multilingual-e5-base"
LEAK_THRESHOLD = 0.95


def main() -> None:
    from sentence_transformers import SentenceTransformer

    labels = load_meta()["labels"]
    lab2id = {l: i for i, l in enumerate(labels)}
    x_tr, y_tr = load_split("train")
    x_te, y_te = load_split("test")
    y_tr, y_te = np.asarray(y_tr), np.asarray(y_te)

    t = pq.read_table("data_tickets.parquet").to_pydict()
    seen, de_x, de_y = set(), [], []
    for i in range(len(t["queue"])):
        if t["language"][i] != "de" or t["queue"][i] not in lab2id:
            continue
        text = clean(t["subject"][i], t["body"][i])
        if text and text not in seen:
            seen.add(text)
            de_x.append(text)
            de_y.append(lab2id[t["queue"][i]])
    de_y = np.asarray(de_y)
    print(f"German tickets in the ten shared queues: {len(de_x)}", flush=True)

    enc = SentenceTransformer(MODEL, device="mps")
    embed = lambda xs: enc.encode([f"query: {x[:1000]}" for x in xs], batch_size=64,
                                  normalize_embeddings=True, show_progress_bar=False)
    t0 = time.time()
    E_tr, E_te, E_de = embed(x_tr), embed(x_te), embed(de_x)
    print(f"embedded {len(x_tr)+len(x_te)+len(de_x)} tickets in {time.time()-t0:.0f}s", flush=True)

    # Leak guard: drop German tickets that are near-translations of a test ticket.
    near = np.concatenate([(E_de[s:s+2000] @ E_te.T).max(1) for s in range(0, len(E_de), 2000)])
    keep = near < LEAK_THRESHOLD
    print(f"dropped {int((~keep).sum())} German tickets that nearly match an English test ticket", flush=True)
    E_de, de_y = E_de[keep], de_y[keep]

    # Same "unfamiliar wording" buckets as before, measured on English words.
    v = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True).fit(x_tr)
    A, B = v.transform(x_tr), v.transform(x_te)
    sim = np.concatenate([linear_kernel(B[s:s+500], A).max(1) for s in range(0, len(y_te), 500)])
    unfam = sim < 0.5

    rng = np.random.default_rng(42)
    order = rng.permutation(len(E_de))
    rows = []
    print(f"\n{'German added':>13} {'train size':>11} {'accuracy':>9} {'macro-F1':>9} {'unfamiliar':>11}")
    for frac in (0.0, 0.25, 0.5, 0.75, 1.0):
        k = int(len(E_de) * frac)
        X = np.vstack([E_tr, E_de[order[:k]]])
        Y = np.concatenate([y_tr, de_y[order[:k]]])
        m = MLPClassifier(hidden_layer_sizes=(512,), max_iter=200, early_stopping=True,
                          n_iter_no_change=8, random_state=42).fit(X, Y)
        P = m.predict_proba(E_te)
        p = P.argmax(1)
        row = {"german_added": k, "train_size": len(Y),
               "accuracy": round(float(accuracy_score(y_te, p)), 4),
               "macro_f1": round(float(f1_score(y_te, p, average="macro")), 4),
               "unfamiliar_accuracy": round(float(accuracy_score(y_te[unfam], p[unfam])), 4)}
        rows.append(row)
        print(f"{k:13d} {len(Y):11d} {row['accuracy']*100:8.2f}% {row['macro_f1']:9.3f} "
              f"{row['unfamiliar_accuracy']*100:10.2f}%", flush=True)
        if frac == 1.0:
            np.save("results/10_e5_en_de_probs.npy", P)
        if frac == 0.0:
            np.save("results/10_e5_en_only_probs.npy", P)

    (RESULTS / "german_curve.json").write_text(json.dumps(
        {"embedding_model": MODEL, "german_available": len(de_x),
         "german_dropped_as_leaks": int((~keep).sum()), "rows": rows}, indent=2))


if __name__ == "__main__":
    main()
