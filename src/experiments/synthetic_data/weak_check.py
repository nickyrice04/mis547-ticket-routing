"""Is the synthetic-data gain on the weak queues real, or noise? A paired bootstrap.

The 10,000 Qwen tickets moved overall accuracy by less than a point but looked like they
helped the four small queues (General Inquiry, Returns, HR, Sales). This script settles
it. Both training sets, core only and core plus synthetic, are trained with three seeds
and scored on the same test tickets. Then the test tickets are resampled 3,000 times and
the difference in per-ticket correctness is recomputed each time, which gives a 95%
interval on the gain. Pairing on the ticket removes ticket difficulty from the noise.
The same is done for macro-F1 over the whole test set.

    PYTHONPATH=src python src/experiments/synthetic_data/weak_check.py
"""
import json, numpy as np, sys
sys.path.insert(0, "src")
from experiments.synthetic_data.compare_synthetic import train_predict, load_synthetic, WEAK
from sklearn.metrics import f1_score
from common import load_split, load_meta
labels=load_meta()["labels"]; x_tr,y_tr=load_split("train"); x_te,y_te=load_split("test")
y_tr,y_te=np.asarray(y_tr),np.asarray(y_te); sp=json.load(open("data/synthetic/split.json"))
xc=[x_tr[i] for i in sp["core"]]; yc=y_tr[sp["core"]]; xs,ys,_=load_synthetic("qwen_think_10k")
weak=np.isin(y_te,[labels.index(q) for q in WEAK])
# three seeds each, so a single lucky run cannot decide it
A=[train_predict(xc,yc,x_te,s) for s in (1,2,3)]
B=[train_predict(xc+xs,np.concatenate([yc,ys]),x_te,s) for s in (1,2,3)]
rng=np.random.default_rng(0); wi=np.where(weak)[0]
a=np.mean([p==y_te for p in A],0); b=np.mean([p==y_te for p in B],0); d=(b-a)[wi]
boots=[d[rng.integers(0,len(d),len(d))].mean() for _ in range(3000)]
lo,hi=np.percentile(boots,[2.5,97.5])
print(f"weak queues, {len(wi)} test tickets: {a[wi].mean()*100:.2f}% -> {b[wi].mean()*100:.2f}%  gain {d.mean()*100:+.2f}  95% CI [{lo*100:+.2f}, {hi*100:+.2f}]")
fa=[f1_score(y_te,p,average="macro") for p in A]; fb=[f1_score(y_te,p,average="macro") for p in B]
mb=[]
for _ in range(1000):
    i=rng.integers(0,len(y_te),len(y_te))
    mb.append(np.mean([f1_score(y_te[i],p[i],average="macro") for p in B])-np.mean([f1_score(y_te[i],p[i],average="macro") for p in A]))
lo,hi=np.percentile(mb,[2.5,97.5])
print(f"macro-F1: {np.mean(fa):.3f} -> {np.mean(fb):.3f}  gain {np.mean(fb)-np.mean(fa):+.3f}  95% CI [{lo:+.3f}, {hi:+.3f}]")
print("per queue accuracy, core only -> core + synthetic:")
for q in WEAK:
    k=y_te==labels.index(q); print(f"  {q:24s} {k.sum():4d} tickets  {a[k].mean()*100:5.1f}% -> {b[k].mean()*100:5.1f}%")
