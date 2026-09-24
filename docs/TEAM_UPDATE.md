> Superseded. Written September 20th, before the retrieval finding. The current story is in MODEL_STORY.md and the README.

# Project update, three weeks out

Team 3, MIS 547. Written September 20th, with the final due in the first half of
October.

## The short version

We got a 99% on the midterm update, and Professor Zara's feedback gave us a
clear list of things to answer in the final. I spent this session on three
things. First, I found a problem with our accuracy number that we need to
correct in the final report. Second, I built the experiment that will carry the
final report, which is a comparison of four model sizes on the same task. Third,
I set up a shared droplet on DigitalOcean so all three of us can work on the
same machine.

## The thing we have to fix

Our midterm report says the baseline gets 70.5% accuracy. That number is
inflated and we should not repeat it.

The dataset repeats the same ticket text many times. When we split it randomly,
26% of the test tickets had an identical twin sitting in the training set. The
model had already seen them. On those repeated tickets it scored 94.2%. On
tickets it had genuinely never seen it scored 62.2%. The 70.5% we reported is
the blend of the two, so it mostly measures memorization.

After removing duplicate tickets, the dataset goes from 28,261 tickets to
23,747, and the honest baseline is **61.6% accuracy with a 0.625 macro-F1**.

This is a better story for the final report than the one we had. We found a
leak in our own evaluation, we explain how we found it, and we fix it. That is
what the data preparation section of the report should say.

It also changes one of the answers we gave Professor Zara. She asked what we do
about out-of-band vocabulary. Before the fix, tickets with more unseen words
scored 6 points worse, which looked like a real out-of-vocabulary effect. After
removing the duplicates, that gap disappears completely, 61.6% against 61.4%.
The honest answer is that unseen words are not what limits this model. What
limits it is that the queues genuinely overlap. A ticket about a broken
integration could reasonably be filed under Technical Support, Product Support
or IT Support, and the dataset picks one.

## The experiment for the final report

Nicky's idea, which I think carries the whole report: show what model size buys
you. Four tiers, same tickets, same split, same metrics.

| Tier | Model | Size |
| --- | --- | --- |
| 1 | TF-IDF and logistic regression | 0.7M weights |
| 2 | DistilBERT, fine-tuned | 67M |
| 3 | RoBERTa base, fine-tuned | 125M |
| 4 | Qwen2.5-0.5B with LoRA | 494M |

For each one we record accuracy, macro-F1, memory held while serving, latency
per ticket at 1, 2 and 4 CPU threads, and how much of the queue it can route on
its own at a given confidence.

The argument we expect to make is that accuracy climbs with size, latency climbs
too, and the question is whether the extra wait is acceptable for a support desk
that has a 15 day regulatory clock. If a bigger model takes 200 ms instead of
2 ms, nobody notices, and the accuracy is worth it. That is a cloud computing
argument rather than a machine learning one, because the model has to run
somewhere that is always on, and the size of that somewhere is what we pay for.

## Where the experiment stands tonight

Two tiers are measured and the result so far is the opposite of what we
expected. It is worth writing down plainly.

| Tier | Model | Accuracy | Macro-F1 | Training time |
| --- | --- | --- | --- | --- |
| 1 | TF-IDF and logistic regression | 61.6% | 0.625 | 7 seconds |
| 2 | DistilBERT, fine-tuned | 39.2% | 0.411 | 13 minutes |

DistilBERT is losing badly to a bag of words. Before anyone puts that in the
report, note that its accuracy was still rising on the final epoch, 29%, 30%,
35%, 39%, 39%. A model that is still improving when you stop it is undertrained,
not outclassed. Two runs are going now to sort this out. One removes the class
weighting, which multiplies the loss on rare queues by up to seven and makes
training noisy. The other is RoBERTa at the same settings.

So the honest status is that the comparison is not finished. What we can already
say is that a bigger model is not automatically better, and that getting one to
beat a simple baseline takes real tuning. That is a more interesting finding for
the report than "bigger won," and it is the kind of thing the cost section
should weigh, because those 13 minutes of training are a cost too.

## The DigitalOcean setup

There is now a shared droplet on the class team account.

| | |
| --- | --- |
| Name | `mis547-project-team3` |
| IP | `174.138.91.2` |
| Size | 2 vCPU, 4 GB RAM, $24/month |
| Region | nyc3 |
| Firewall | inbound SSH only |

The baseline model is already deployed there as a container and answering
requests. A ticket sent to the API comes back with a queue, a confidence score,
and whether it would be routed automatically. It answers in 7 milliseconds
including the network round trip and holds 129 MB of memory.

Why 4 GB and not the $6 droplet from the labs: DistilBERT needs about 1.5 GB
resident before it answers anything, so the 1 GB droplet the labs use cannot
host it. This is the first concrete piece of evidence for the report's argument.
The model size decides the droplet size, and the droplet size is the bill.

### How to get on it

Generate a key and send Nicky the public half. The private half never leaves
your laptop.

```bash
ssh-keygen -t ed25519 -C "your_netid@arizona.edu" -f ~/.ssh/digitalocean_key
cat ~/.ssh/digitalocean_key.pub
```

Paste the output of that second command into our chat. It is safe to share. You
will get your own login on the droplet, not a shared one, so we can tell who did
what. Full instructions are in `infra/README.md` in the repository.

### Reaching the API

Nothing is exposed to the internet. The API listens on the droplet's loopback
address only, so you tunnel to it the same way we tunneled to Jupyter in Lab 1.

```bash
ssh -i ~/.ssh/digitalocean_key -L 8000:localhost:8000 root@174.138.91.2
```

Then in another terminal:

```bash
curl -s -X POST localhost:8000/predict -H 'content-type: application/json' \
  -d '{"subject":"double charge","body":"I was billed twice this month."}'
```

## Answering the rest of Professor Zara's questions

**Confidence and thresholds.** Every model now reports a confidence number,
which is the probability it assigns to its top queue. The API applies a cut and
returns whether the ticket would be routed automatically. On the clean data, the
baseline at a 0.5 cut routes 34% of tickets on its own and is right 86% of the
time on those. The rest go to a person. We can pick the cut once we decide how
accurate auto-routing has to be.

**High availability.** One container on one droplet is not highly available, and
we should say so plainly rather than claim otherwise. The container now answers
a health check and restarts itself when it crashes, which is the first half. The
real answer is two droplets behind a load balancer, which costs $60 a month
instead of $24. The full write-up is in `docs/HIGH_AVAILABILITY.md`.

**The cost of DistilBERT.** She was right and our midterm was wrong on this. A
self-hosted open weights model has no per-token charge. The only extra cost is a
bigger droplet, and the benchmarks will tell us exactly how much bigger.

## What is left, and who could take it

1. **Finish the model ladder.** Running now. Two of the four tiers are done.
2. **Benchmark each model on the droplet**, not the laptop. The laptop has 128
   GB of memory, so it proves nothing about what a $24 droplet can hold.
3. **Decide the confidence threshold** from the tables, and write the rule into
   the report.
4. **Add the second droplet and the load balancer** so the high availability
   answer is real and not a plan.
5. **Rewrite the data section** of the report with the duplicate finding.
6. **Send Professor Zara a Word version** of the report. She offered to copy edit
   it, and her feedback said there were small mistakes throughout. That offer is
   free points and we should take it.

The infrastructure is set up, so pieces 3 through 6 are mostly writing and can
be split however you two prefer.
