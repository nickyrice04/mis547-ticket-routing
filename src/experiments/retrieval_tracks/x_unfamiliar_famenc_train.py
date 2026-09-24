"""GPU run 2 of 2: contrastive fine-tune of all-mpnet-base-v2 into a "same scenario family?" bi-encoder.
Trained on family pairs mined inside core folds 1-4 only (x_unfamiliar_mine_pairs.py). Fold 0 of core is held out and
used here to monitor 1-NN accuracy for tickets the encoder has never seen. Validation/test are never read.
float32 on mps, hard wall-clock limit."""
import os, sys, time, json, math, pickle
os.environ.setdefault("HF_HUB_DISABLE_XET", "1"); os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
import numpy as np, torch, torch.nn.functional as Fn
from transformers import AutoTokenizer, AutoModel
from experiments.retrieval_tracks.x_unfamiliar_lib import *
from experiments.retrieval_tracks.x_unfamiliar_cv import folds, fold_sims

torch.set_num_threads(3); torch.manual_seed(0); rng = np.random.RandomState(0)
NAME = "sentence-transformers/all-mpnet-base-v2"; OUT = "models/x_unfamiliar_famenc"
EPOCHS = int(os.environ.get("EPOCHS", 3)); BS = 32; MAXLEN = 160; SCALE = 20.0; LIMIT = float(os.environ.get("LIMIT_MIN", 40)) * 60
dev = torch.device("mps")
D = pickle.load(open("data/x_unfamiliar/family_pairs.pkl", "rb")); texts, pos, neg = D["texts"], D["pos"], D["neg"]
anchors = np.array([i for i, p in enumerate(pos) if p])
xc, yc, _, _ = load_core_val(); sim, _, _ = fold_sims(xc, yc); tr, te = folds(yc)[0]
assert list(tr) == list(D["train_idx"])

tok = AutoTokenizer.from_pretrained(NAME); enc = AutoModel.from_pretrained(NAME, torch_dtype=torch.float32).to(dev)
def embed(batch):
    b = tok(batch, truncation=True, max_length=MAXLEN, padding=True, return_tensors="pt").to(dev)
    h = enc(**b).last_hidden_state; m = b["attention_mask"].unsqueeze(-1).float()
    return Fn.normalize((h * m).sum(1) / m.sum(1), dim=-1)
def embed_all(tx):
    enc.eval(); order = np.argsort([len(t) for t in tx]); out = np.zeros((len(tx), enc.config.hidden_size), dtype=np.float32)
    with torch.no_grad():
        for s in range(0, len(tx), 96):
            ids = order[s:s + 96]; out[ids] = embed([tx[i] for i in ids]).cpu().numpy()
    enc.train(); return out
def heldout_report(tag):
    Ep = embed_all([xc[i] for i in tr]); Eq = embed_all([xc[i] for i in te]); S = Eq @ Ep.T
    p = yc[tr][S.argmax(1)]; y = yc[te]; u = sim[te] < 0.4; mid = (sim[te] >= 0.3) & (sim[te] < 0.5)
    r = {"tag": tag, "heldout_1nn_all": float((p == y).mean()), "heldout_1nn_unfamiliar": float((p[u] == y[u]).mean()),
         "heldout_1nn_sim0.3-0.5": float((p[mid] == y[mid]).mean()), "minutes": (time.time() - t0) / 60}
    print(json.dumps(r), flush=True); return r

steps_total = EPOCHS * math.ceil(len(anchors) / BS); warm = int(0.1 * steps_total)
opt = torch.optim.AdamW(enc.parameters(), lr=2e-5, weight_decay=0.01)
sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / warm) * max(0.0, (steps_total - s) / steps_total))
t0 = time.time(); log = [heldout_report("pretrained")]; step = 0; stop = False
t0 = time.time()   # the limit applies to training time
for ep in range(EPOCHS):
    perm = rng.permutation(anchors); tl = 0.0; nb = 0
    for s in range(0, len(perm) - BS + 1, BS):
        A, P, Ng = [], [], []
        for i in perm[s:s + BS]:
            pj = list(pos[i].keys()); w = np.array([(1.05 - pos[i][j]) ** 2 for j in pj]); j = pj[rng.choice(len(pj), p=w / w.sum())]
            nj = sorted(neg[i], key=lambda q: -neg[i][q])[:6]; k = nj[rng.randint(len(nj))]
            A.append(texts[i]); P.append(texts[j]); Ng.append(texts[k])
        z = embed(A + P + Ng); a, p_, n_ = z[:BS], z[BS:2 * BS], z[2 * BS:]
        logits = SCALE * a @ torch.cat([p_, n_]).T
        loss = Fn.cross_entropy(logits, torch.arange(BS, device=dev))
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(enc.parameters(), 1.0); opt.step(); sched.step(); step += 1
        tl += loss.item(); nb += 1
        if step % 25 == 0: print(f"ep {ep} step {step}/{steps_total} loss {tl/nb:.4f} elapsed {(time.time()-t0)/60:.1f}m", flush=True)
        if time.time() - t0 > LIMIT: stop = True; break
    r = heldout_report(f"epoch{ep+1}" + ("_partial" if stop else "")); r["train_loss"] = tl / max(nb, 1); log.append(r)
    json.dump({"base": NAME, "log": log}, open("results/x_unfamiliar_famenc_train.json", "w"), indent=1)
    if stop: break
os.makedirs(OUT, exist_ok=True); enc.save_pretrained(OUT); tok.save_pretrained(OUT)
json.dump({"seen_sha": D["seen_sha"], "max_len": MAXLEN, "base": NAME}, open(OUT + "/x_unfamiliar_seen.json", "w"))
print("saved", OUT, "train minutes", (time.time() - t0) / 60)
