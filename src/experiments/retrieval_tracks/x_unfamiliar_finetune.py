"""GPU run 1 of 2: multi-task fine-tune of a sentence encoder on 4/5 of core, scored each epoch on the held-out
core fold (overall and on its unfamiliar tickets). Validation and test are not read. float32 on mps, hard time limit."""
import os, sys, time, json, math
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
import numpy as np, torch, torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from experiments.retrieval_tracks.x_unfamiliar_lib import *
from experiments.retrieval_tracks.x_unfamiliar_cv import folds, fold_sims
from experiments.retrieval_tracks.x_unfamiliar_meta import load_meta_rows

torch.set_num_threads(3); torch.manual_seed(0); np.random.seed(0)
NAME = os.environ.get("ENC", "sentence-transformers/all-mpnet-base-v2"); TAG = os.environ.get("TAG", "mpnet_mt")
EPOCHS = int(os.environ.get("EPOCHS", 5)); BS = 32; MAXLEN = 256; LIMIT = float(os.environ.get("LIMIT_MIN", 42)) * 60
AUX = os.environ.get("AUX", "1") == "1"; FOLD = 0
dev = torch.device("mps")

xc, yc, _, _ = load_core_val(); core, _ = load_meta_rows()
sim, nnidx, fold = fold_sims(xc, yc); tr, te = folds(yc)[FOLD]
types = sorted({r["type"] for r in core}); prios = sorted({r["priority"] for r in core})
def tags_of(r): return {s.strip().lower() for t in r["tags"] for s in t.split(",") if s.strip()}
import collections
cnt = collections.Counter(t for i in tr for t in tags_of(core[i])); vocab = [t for t, c in cnt.items() if c >= 50]; tix = {t: k for k, t in enumerate(vocab)}
ytype = np.array([types.index(r["type"]) for r in core]); yprio = np.array([prios.index(r["priority"]) for r in core])
ytags = np.zeros((len(core), len(vocab)), dtype=np.float32)
for i, r in enumerate(core):
    for t in tags_of(r):
        if t in tix: ytags[i, tix[t]] = 1
print("train", len(tr), "heldout", len(te), "heldout unfamiliar", int((sim[te] < 0.4).sum()), "tags", len(vocab), flush=True)

tok = AutoTokenizer.from_pretrained(NAME); enc = AutoModel.from_pretrained(NAME, torch_dtype=torch.float32).to(dev)
H = enc.config.hidden_size
heads = nn.ModuleDict({"queue": nn.Linear(H, 10), "ttype": nn.Linear(H, len(types)), "prio": nn.Linear(H, len(prios)), "tags": nn.Linear(H, len(vocab))}).to(dev)
drop = nn.Dropout(0.1)
def embed(texts):
    b = tok(texts, truncation=True, max_length=MAXLEN, padding=True, return_tensors="pt").to(dev)
    h = enc(**b).last_hidden_state; m = b["attention_mask"].unsqueeze(-1).float()
    return (h * m).sum(1) / m.sum(1)
steps_total = EPOCHS * math.ceil(len(tr) / BS); warm = int(0.06 * steps_total)
opt = torch.optim.AdamW([{"params": enc.parameters(), "lr": 2e-5}, {"params": heads.parameters(), "lr": 1e-3}], weight_decay=0.01)
sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / warm) * max(0.0, (steps_total - s) / steps_total))
ce = nn.CrossEntropyLoss(label_smoothing=0.05); bce = nn.BCEWithLogitsLoss()

def evaluate():
    enc.eval(); out = []
    order = np.argsort([len(xc[i]) for i in te])
    with torch.no_grad():
        for s in range(0, len(te), 64):
            ids = te[order[s:s + 64]]; out.append(torch.softmax(heads["queue"](embed([xc[i] for i in ids])), -1).cpu().numpy())
    p = np.zeros((len(te), 10), dtype=np.float32); p[order] = np.concatenate(out); enc.train(); return p

t0 = time.time(); step = 0; log = []; stop = False
for ep in range(EPOCHS):
    perm = np.random.permutation(tr); tl = 0.0; nb = 0
    for s in range(0, len(perm), BS):
        ids = perm[s:s + BS]; z = drop(embed([xc[i] for i in ids]))
        loss = ce(heads["queue"](z), torch.tensor(yc[ids], device=dev))
        if AUX:
            loss = loss + 0.3 * ce(heads["ttype"](z), torch.tensor(ytype[ids], device=dev)) + 0.1 * ce(heads["prio"](z), torch.tensor(yprio[ids], device=dev)) \
                   + 5.0 * bce(heads["tags"](z), torch.tensor(ytags[ids], device=dev))
        opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(enc.parameters(), 1.0); opt.step(); sched.step(); step += 1
        tl += float(loss); nb += 1
        if step % 50 == 0: print(f"ep {ep} step {step}/{steps_total} loss {tl/nb:.4f} elapsed {(time.time()-t0)/60:.1f}m", flush=True)
        if time.time() - t0 > LIMIT: stop = True; break
    p = evaluate(); pred = p.argmax(1); u = sim[te] < 0.4
    rec = {"epoch": ep + 1, "complete": not stop, "minutes": (time.time() - t0) / 60, "train_loss": tl / max(nb, 1),
           "heldout_all": float((pred == yc[te]).mean()), "heldout_unfamiliar": float((pred[u] == yc[te][u]).mean()), "n_unf": int(u.sum())}
    log.append(rec); print(json.dumps(rec), flush=True)
    np.save(f"data/x_unfamiliar/ft_{TAG}_fold{FOLD}_ep{ep+1}_probs.npy", p)
    json.dump({"encoder": NAME, "aux": AUX, "log": log}, open(f"results/x_unfamiliar_finetune_{TAG}.json", "w"), indent=1)
    if stop: break
print("done", (time.time() - t0) / 60, "min")
