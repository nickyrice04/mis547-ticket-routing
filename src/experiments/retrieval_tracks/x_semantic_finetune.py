"""Contrastive fine-tune of a sentence embedder on pairs mined ONLY from the fine-tune part (80%) of core.
Triplets: (anchor, likely sibling = same queue & lexically close, hard negative = close but other queue).
Loss: multiple-negatives ranking (in-batch + hard negatives), cosine scale 20. Custom loop so the wall-clock budget is enforced.

  python x_semantic_finetune.py <base> <out_tag> <epochs> <max_minutes> [smoke]
"""
import sys, time, json, math
import numpy as np
import torch
from experiments.retrieval_tracks.x_semantic_common import *

BASES = {"bge": ("BAAI/bge-base-en-v1.5", ""), "e5base": ("intfloat/multilingual-e5-base", "query: "),
         "mpnet": ("sentence-transformers/all-mpnet-base-v2", ""), "gte": ("thenlper/gte-base", "")}
base, tag, epochs, max_min = sys.argv[1], sys.argv[2], int(sys.argv[3]), float(sys.argv[4])
smoke = len(sys.argv) > 5
POS_LEX_MIN, BATCH, LR, SCALE, MAXLEN = 0.4, 48, 2e-5, 20.0, 192

ctx, cy, _, _ = load_core_val()
nb = np.load(CACHE / "ft_neighbours.npz")
ft, nbr, nbl, same = nb["ft"], nb["nbr"], nb["nbl"], nb["same"]
texts = [ctx[i] for i in ft]
top = np.arange(nbr.shape[1])[None, :] < 10
pos_mask = same & (nbl >= POS_LEX_MIN) & top
neg_mask = (~same) & top
anchors = np.where(pos_mask.any(1) & neg_mask.any(1))[0]
print(f"ft tickets {len(ft)}, anchors with sibling-positive and hard negative: {len(anchors)}", flush=True)

from sentence_transformers import SentenceTransformer
torch.manual_seed(0); rng = np.random.default_rng(0)
torch.set_num_threads(3)
name, prefix = BASES[base]
model = SentenceTransformer(name, device="mps"); model.max_seq_length = MAXLEN
model.train()
steps_per_epoch = len(anchors) // BATCH
total = steps_per_epoch * epochs
opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min((s + 1) / (0.1 * total), max(0.0, (total - s) / (0.9 * total))))

def enc(batch_texts):
    f = model.tokenize([prefix + t for t in batch_texts])
    f = {k: (v.to("mps") if hasattr(v, "to") else v) for k, v in f.items()}
    return torch.nn.functional.normalize(model(f)["sentence_embedding"], dim=-1)

t0 = time.time(); step = 0; done = False
for ep in range(epochs):
    order = rng.permutation(anchors); losses = []
    for s in range(steps_per_epoch):
        a = order[s * BATCH:(s + 1) * BATCH]
        p = np.array([rng.choice(nbr[i][pos_mask[i]]) for i in a])
        ncand = [nbr[i][neg_mask[i]][:5] for i in a]
        ng = np.array([rng.choice(c) for c in ncand])
        ea, ep_, en = enc([texts[i] for i in a]), enc([texts[i] for i in p]), enc([texts[i] for i in ng])
        logits = ea @ torch.cat([ep_, en]).T * SCALE
        # mask in-batch candidates that share the anchor's queue (other than its own positive): they are not true negatives
        ya = torch.tensor(cy[ft[a]]); yc = torch.tensor(np.concatenate([cy[ft[p]], cy[ft[ng]]]))
        m = (ya[:, None] == yc[None, :]); m[torch.arange(len(a)), torch.arange(len(a))] = False
        logits = logits.masked_fill(m.to("mps"), -1e4)
        loss = torch.nn.functional.cross_entropy(logits, torch.arange(len(a), device="mps"))
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sched.step()
        losses.append(loss.item()); step += 1
        if step % 20 == 0:
            print(f"ep {ep} step {step}/{total} loss {np.mean(losses[-20:]):.4f} {time.time()-t0:.0f}s", flush=True)
        if smoke and step >= 8:
            print(f"smoke: {(time.time()-t0)/step:.2f}s/step -> est {(time.time()-t0)/step*total/60:.1f} min total"); sys.exit()
        if (time.time() - t0) / 60 > max_min:
            done = True; break
    print(f"== epoch {ep} mean loss {np.mean(losses):.4f} elapsed {(time.time()-t0)/60:.1f} min", flush=True)
    if done:
        print("time budget reached"); break
model.eval()
model.save(str(ROOT / "models" / f"x_semantic_{tag}"))
print("saved", f"models/x_semantic_{tag}", f"total {(time.time()-t0)/60:.1f} min")
