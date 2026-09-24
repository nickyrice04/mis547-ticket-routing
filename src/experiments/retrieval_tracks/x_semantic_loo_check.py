"""Does leave-one-out inside core mimic the validation->core relation? Compare top-1 sim quantiles and 1-NN accuracy."""
import numpy as np
from sklearn.model_selection import StratifiedKFold
from experiments.retrieval_tracks.x_semantic_common import *
ctx, cy, vtx, vy = load_core_val()
simv = np.load(CACHE / "sim_lex_val.npy")
qs = [10, 25, 50, 75, 90]
print("val->core      q", np.percentile(simv, qs).round(3), "acc 0.7449")
vec = baseline_tfidf(); X = vec.fit_transform(ctx)
top = np.zeros(len(ctx)); ok = np.zeros(len(ctx), bool)
for a in range(0, len(ctx), 2000):
    S = (X[a:a+2000] @ X.T).toarray(); S[np.arange(S.shape[0]), np.arange(a, a + S.shape[0])] = -1
    top[a:a+2000] = S.max(1); ok[a:a+2000] = cy[S.argmax(1)] == cy[a:a+2000]
print("LOO full-fit   q", np.percentile(top, qs).round(3), "acc", ok.mean().round(4))
top2 = np.zeros(len(ctx)); ok2 = np.zeros(len(ctx), bool)
for tr, te in StratifiedKFold(5, shuffle=True, random_state=0).split(ctx, cy):
    v = baseline_tfidf().fit([ctx[i] for i in tr]); Xa = v.transform(ctx)
    S = (Xa[te] @ Xa.T).toarray(); S[np.arange(len(te)), te] = -1
    top2[te] = S.max(1); ok2[te] = cy[S.argmax(1)] == cy[te]
print("LOO fold-fit   q", np.percentile(top2, qs).round(3), "acc", ok2.mean().round(4))
