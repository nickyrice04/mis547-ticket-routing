"""Do the four dense 1-NN views agree with lexical 1-NN, and what would an oracle over all views score? Validation diagnostic."""
import numpy as np
from experiments.retrieval_tracks.x_semantic_common import *
ctx, cy, vtx, vy = load_core_val()
S_lex = np.load(CACHE / "S_lex_val_core.npy"); sim = S_lex.max(1)
lex_ok = cy[S_lex.argmax(1)] == vy
extra = {"lex": lex_ok}
any_ok = lex_ok.copy()
for name in ["e5base", "bge", "mpnet", "gte"]:
    S = np.load(CACHE / f"emb_{name}_val.npy") @ np.load(CACHE / f"emb_{name}_core.npy").T
    ok = cy[S.argmax(1)] == vy
    extra[name] = ok
    any_ok |= ok
    print(name, "agree with lex NN idx:", np.mean(S.argmax(1) == S_lex.argmax(1)).round(3), "sim range", np.percentile(S.max(1), [5, 50, 95]).round(3))
extra["oracle"] = any_ok
print(fmt_table(bucket_table(sim, lex_ok, extra)))
print("oracle overall", any_ok.mean())
