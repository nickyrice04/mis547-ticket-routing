"""How many tickets in each similarity bucket actually have their scenario family in the pool?
Family twins share type and priority (99.7-100% at cosine >= 0.8) while unrelated neighbours agree only at a base rate, so
the agreement rate of (type, priority) between a core ticket and its leave-one-out nearest core neighbour is a mixture:
    agree_b = pi_b * 1 + (1 - pi_b) * base,     base = agreement in the 0.0-0.2 bucket (no families there)
This gives pi_b, the share of tickets whose nearest neighbour is a true family member, and from it an oracle-gate ceiling:
    ceiling = sum_b share_b * (pi_b * 0.995 + (1 - pi_b) * specialist_accuracy_on_family-less_tickets)
Core only (metadata is used descriptively, not as model input)."""
import json, numpy as np
from experiments.retrieval_tracks.x_unfamiliar_lib import *
from experiments.retrieval_tracks.x_unfamiliar_meta import load_meta_rows
xc, yc, _, _ = load_core_val(); core, _ = load_meta_rows()
sim = np.load("data/x_unfamiliar/sim_core_loo.npy"); nn = np.load("data/x_unfamiliar/nnidx_core_loo.npy")
tp = np.array([hash((r["type"], r["priority"])) for r in core]); agree = tp == tp[nn]; sameq = yc == yc[nn]
m0 = sim < 0.2; base = float(agree[m0].mean()); base_q = float(sameq[m0].mean())
val_share = [0.032, 0.187, 0.135, 0.100, 0.121, 0.145, 0.156, 0.124]
rows = []; SPEC = 0.40
for (lo, hi), vs in zip(BUCKETS, val_share):
    m = (sim >= lo) & (sim < hi); a = float(agree[m].mean()); pi = max(0.0, (a - base) / (1 - base))
    rows.append({"bucket": f"{lo:.1f}-{min(hi,1.0):.1f}", "core_share": float(m.mean()), "type_priority_agreement": a, "family_share_pi": pi,
                 "nn_same_queue": float(sameq[m].mean()), "implied_same_queue": pi + (1 - pi) * base_q,
                 "oracle_gate_acc": pi * 0.995 + (1 - pi) * SPEC})
    print(rows[-1])
ceil_core = sum(r["core_share"] * r["oracle_gate_acc"] for r in rows); ceil_val = sum(v * r["oracle_gate_acc"] for v, r in zip(val_share, rows))
fam_total = sum(r["core_share"] * r["family_share_pi"] for r in rows)
fam_unf = sum(r["core_share"] * r["family_share_pi"] for r in rows[:3]) / sum(r["core_share"] for r in rows[:3])
out = {"base_type_priority_agreement": base, "base_same_queue_for_unrelated_neighbours": base_q, "buckets": rows,
       "share_of_tickets_with_family_nn_in_pool": fam_total, "share_of_unfamiliar_lt0.4_with_family_nn": fam_unf,
       "oracle_gate_ceiling_core_loo": ceil_core, "oracle_gate_ceiling_validation_shares": ceil_val, "specialist_acc_assumed": SPEC}
print(json.dumps({k: v for k, v in out.items() if k != "buckets"}, indent=1)); json.dump(out, open("results/x_unfamiliar_family_fraction.json", "w"), indent=1)
