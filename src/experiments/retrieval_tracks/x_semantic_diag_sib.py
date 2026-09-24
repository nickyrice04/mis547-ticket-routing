"""Are there 'hidden siblings' (same family, low lexical overlap) that dense retrieval could find? LOO inside core."""
import numpy as np
from experiments.retrieval_tracks.x_semantic_common import *
ctx, cy, vtx, vy = load_core_val()
n = len(ctx)
vec = baseline_tfidf(); X = vec.fit_transform(ctx)
dense = ["e5base", "bge", "mpnet", "gte"]
E = [np.load(CACHE / f"emb_{d}_core.npy") for d in dense]
rows = []
for a in range(0, n, 2000):
    b = min(a + 2000, n)
    SL = (X[a:b] @ X.T).toarray().astype(np.float32)
    SD = np.mean([e[a:b] @ e.T for e in E], 0)
    r = np.arange(b - a); SL[r, np.arange(a, b)] = -9; SD[r, np.arange(a, b)] = -9
    jd = SD.argmax(1); jl = SL.argmax(1)
    # dense margin: top1 minus 2nd
    part = np.partition(SD, -2, axis=1); margin = part[:, -1] - part[:, -2]
    rows.append(np.stack([SL[r, jd], SD[r, jd], margin, cy[jd] == cy[a:b], jd == jl, SL[r, jl], cy[jl] == cy[a:b]], 1))
R = np.concatenate(rows)
lexsim_of_dense_nn, dsim, margin, same_d, same_idx, lex_top, same_l = R.T
print("dense-avg 1-NN acc", same_d.mean().round(4), " lex 1-NN acc", same_l.mean().round(4))
print("When dense NN != lex NN, by lexical sim of the dense NN:  n, P(same queue | dense NN), P(same | lex NN)")
m0 = same_idx == 0
for lo, hi in [(-1, .1), (.1, .2), (.2, .3), (.3, .4), (.4, .5), (.5, 1.01)]:
    m = m0 & (lexsim_of_dense_nn >= lo) & (lexsim_of_dense_nn < hi)
    print(f"  lex {lo}-{hi}: n={int(m.sum())}  P_dense={same_d[m].mean():.3f}  P_lex={same_l[m].mean():.3f}")
print("By dense margin (top1-top2) when lex of dense NN < 0.3:")
m1 = lexsim_of_dense_nn < 0.3
for lo, hi in [(0, .005), (.005, .01), (.01, .02), (.02, .04), (.04, 1)]:
    m = m1 & (margin >= lo) & (margin < hi)
    print(f"  margin {lo}-{hi}: n={int(m.sum())} P(same)={same_d[m].mean():.3f}")
print("Tickets whose lex top sim < 0.3: n=", int((lex_top < .3).sum()), "dense NN acc", same_d[lex_top < .3].mean().round(3), "lex NN acc", same_l[lex_top < .3].mean().round(3))
