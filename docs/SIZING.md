# How much machine does each model need

> Written before the final design. The sizing of the deployed system is in ARCHITECTURE.md. The section below on why the project does not use Kubernetes still stands.

## Kubernetes is not VRAM

These are unrelated things and it is worth separating them before we size
anything.

**Kubernetes** is an orchestrator. It is software that runs containers across a
group of machines, restarts them when they die, and adds or removes copies as
load changes. It solves the availability problem Professor Zara asked about. It
has nothing to do with memory. We are not using it for this project, because one
container on one or two droplets does not need a cluster to manage it, and
DigitalOcean's managed Kubernetes adds cost and a lot of new concepts three
weeks before the deadline. A load balancer in front of two droplets gets us the
same availability for less.

**VRAM** is memory attached to a GPU. A GPU can only work on data that is in its
own memory, so VRAM is the hard limit on what model you can train or run on that
card. It matters only if we use a GPU.

**RAM** is ordinary system memory. On a CPU-only droplet, this is the limit that
decides whether a model loads at all.

For this project we care about RAM, because we serve on CPU droplets. VRAM would
only matter if we rent a GPU droplet for training.

## What each tier needs to serve a ticket

Serving is what runs 24 hours a day, so serving is what sizes the droplet. A
rough rule for a model in 32-bit floats is four bytes per parameter, plus the
runtime itself, plus room for the request.

| Tier | Parameters | Weights alone | Plus PyTorch and the request | Smallest droplet that holds it |
| --- | --- | --- | --- | --- |
| TF-IDF + logistic regression | 0.7M | 8 MB on disk | about 165 MB measured | $6, 1 GB |
| DistilBERT | 67M | 268 MB | roughly 1.2 to 1.5 GB | $24, 4 GB |
| RoBERTa base | 125M | 500 MB | roughly 1.5 to 2 GB | $24, 4 GB |
| Qwen2.5-0.5B | 494M | 2.0 GB | roughly 3 to 4 GB | $48, 8 GB |

The jump that matters is the first one. The baseline fits comfortably in the
$6 droplet the labs use. DistilBERT does not, because PyTorch alone is most of a
gigabyte before any model loads. That is the sentence the final report wants:
the model choice sets the droplet size, and the droplet size is the monthly
bill.

The measured numbers replace these estimates in `results/SUMMARY.md` as each
benchmark finishes. Estimates are for planning, measurements are for the report.

## Training needs more than serving

Training holds the weights, the gradients, and the optimizer state at once.
With AdamW that is roughly four times the weights, before the batch of data.
DistilBERT at a batch of 32 needs somewhere around 4 to 6 GB, which is why we
train on a laptop and serve on a droplet rather than training on the droplet.

This is also the honest limit of our "cloud is necessary" argument. One of our
laptops has 128 GB of memory, so for models this size the laptop is not the
constraint. The constraint is that a laptop is not a service. It sleeps, it
changes networks, and only one person has it. The droplet is always on, has a
fixed address, and all three of us can reach it.

## If we want a GPU

Only worth it for training the largest tier, and only briefly. A GPU droplet
with an RTX 4000 Ada costs $0.76 an hour and has 20 GB of VRAM, which is far
more than we need. We would rent it for under an hour and destroy it.

Two cautions. A GPU droplet keeps billing while powered off, so it has to be
destroyed rather than shut down. And nothing in the course materials mentions
GPU droplets at all, so we should ask Professor Zara before creating one on the
department's account.
