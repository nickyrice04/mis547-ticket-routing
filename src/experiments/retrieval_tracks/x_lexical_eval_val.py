"""Core -> validation evaluation of the final module for one EXTRA mode. Writes results/x_lexical_val_<mode>.json."""
import json, os, sys, time
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score, log_loss, classification_report
import experiments.retrieval_tracks.x_lexical_final as F
from experiments.retrieval_tracks.x_lexical_data import load_core_val
from common import load_meta, threshold_table

mode = sys.argv[1]
use_mlp = len(sys.argv) > 2 and sys.argv[2] == "mlp"
tag = mode.replace("+", "_") + ("_mlp" if use_mlp else "")
xc, yc, xv, yv = load_core_val()
labels = load_meta()["labels"]
t0 = time.time()
P, det = F.fit_predict_proba(xc, yc, xv, extra=mode, use_mlp=use_mlp, verbose=True, return_details=True)
minutes = (time.time() - t0) / 60
pred = P.argmax(1)
np.save(f"results/x_lexical_val_probs_{tag}.npy", P)

vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=200_000, sublinear_tf=True)
S = (vec.fit(xc).transform(xv) @ vec.transform(xc).T).toarray()
s1 = S.max(1); nn_pred = yc[S.argmax(1)]
bins = [0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.01]
buckets = []
for lo, hi in zip(bins[:-1], bins[1:]):
    m = (s1 >= lo) & (s1 < hi)
    buckets.append(dict(bucket=f"{lo:.1f}-{min(hi, 1.0):.1f}", share=round(float(m.mean()), 4), n=int(m.sum()),
                        acc_1nn_core_pool=round(float((nn_pred[m] == yv[m]).mean()), 4), acc_method=round(float((pred[m] == yv[m]).mean()), 4)))
conf = P.max(1); ok = pred == yv
ece = 0.0
for lo in np.arange(0, 1, 0.1):
    m = (conf >= lo) & (conf < lo + 0.1 + 1e-9)
    if m.any(): ece += m.mean() * abs(ok[m].mean() - conf[m].mean())
res = dict(mode=mode, use_mlp=use_mlp, n_extra_rows_used=det["n_extra"], accuracy=float(accuracy_score(yv, pred)),
           macro_f1=float(f1_score(yv, pred, average="macro")), logloss=float(log_loss(yv, P, labels=range(10))), ece_10bin=float(ece),
           temperature=det["temperature"], oof_accuracy_inside_core=float((det["oof"].argmax(1) == yc).mean()),
           acc_1nn_core_pool=float((nn_pred == yv).mean()), runtime_minutes=minutes, buckets=buckets,
           threshold_table=threshold_table(P, yv),
           per_class=classification_report(yv, pred, target_names=labels, output_dict=True, zero_division=0))
json.dump(res, open(f"results/x_lexical_val_{tag}.json", "w"), indent=1)
print(json.dumps({k: v for k, v in res.items() if k != "per_class"}, indent=1))
