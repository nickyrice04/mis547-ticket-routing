"""Short controlled probe: does the loss fall on CPU but not on MPS?

Trains the same model, same data, same seed, for a small number of steps under
each setting, and prints the loss trajectory. Cheap way to tell a device bug
apart from a learning rate that is simply wrong.
"""
import sys, time
import numpy as np, torch, torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from common import load_meta, load_split
from experiments.bigger_models.train_transformer import Tickets, class_weights

STEPS = 150
dev, lr, weighted = sys.argv[1], float(sys.argv[2]), sys.argv[3] == "weighted"
torch.manual_seed(0)

meta = load_meta(); n = len(meta["labels"])
x, y = load_split("train")
tok = AutoTokenizer.from_pretrained("distilbert-base-uncased")
model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=n).to(dev)
dl = DataLoader(Tickets(x[:4000], y[:4000], tok, max_len=128), batch_size=16, shuffle=True)
loss_fn = nn.CrossEntropyLoss(weight=class_weights(y, n).to(dev)) if weighted else nn.CrossEntropyLoss()
opt = torch.optim.AdamW(model.parameters(), lr=lr)

t0, losses, step = time.time(), [], 0
model.train()
for batch in dl:
    batch = {k: v.to(dev) for k, v in batch.items()}
    logits = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).logits
    loss = loss_fn(logits, batch["labels"])
    loss.backward(); opt.step(); opt.zero_grad()
    losses.append(loss.item()); step += 1
    if step % 25 == 0:
        print(f"  step {step:3d} loss {np.mean(losses[-25:]):.4f}", flush=True)
    if step >= STEPS: break
print(f"RESULT dev={dev} lr={lr} weighted={weighted} first25={np.mean(losses[:25]):.3f} "
      f"last25={np.mean(losses[-25:]):.3f} secs={time.time()-t0:.0f}")
