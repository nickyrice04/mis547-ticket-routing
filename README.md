# Support ticket routing

Team 3, MIS 547. A model that reads a customer support ticket and routes it to
one of ten queues, built to run on rented cloud compute inside a private
network so no ticket ever leaves the company.

The repository is laid out in the order the project happened. `src/final/` is
the model that ships. `src/baselines/` is where it started. `src/experiments/`
is everything tried in between, kept because the negative results are half
the story.

## Results

All numbers are on the same 4,750 held-out test tickets. Each system was scored
on the test set exactly once, after every decision had been made on a
validation slice of the training data.

| Model | Accuracy | Macro-F1 | Where |
| --- | --- | --- | --- |
| TF-IDF + logistic regression | 61.6% | 0.625 | `baselines/train_tfidf.py` |
| TF-IDF + neural network, one hidden layer of 256 | 68.3% | 0.680 | `baselines/train_mlp.py` |
| DistilBERT fine-tuned, 12 epochs | 67.7% | 0.658 | `experiments/bigger_models/` |
| RoBERTa base fine-tuned | 52.3% | 0.450 | `experiments/bigger_models/` |
| Bidirectional LSTM | 52.4% | 0.445 | `experiments/bigger_models/` |
| Gemma 3 4B, LoRA, as a classifier | 43.2% | 0.354 | `experiments/bigger_models/` |
| Qwen2.5 0.5B, LoRA, as a classifier | 42.8% | 0.307 | `experiments/bigger_models/` |
| DeBERTa zero-shot | 31.4% | 0.229 | `experiments/bigger_models/` |
| Gemma 3 4B zero-shot | 20.6% | 0.174 | `experiments/bigger_models/` |
| Retrieval + embeddings + stacker, English data only | **83.1%** | 0.832 | `final/router_english_only.py` |
| Same, with translated German tickets in the pool | **91.8%** | 0.917 | `final/router.py` |

Laya, an open decision model, reached 57.8% on the validation slice after an
hour of GPU and was stopped before a test run (`experiments/bigger_models/train_laya.py`).

The midterm reported 70.5% for the network. That number was inflated. A
quarter of the test tickets had exact copies in the training set, and the
model scored 94% on those against 62% on the rest. Duplicates are now removed
before the split (`common.py`) and every number above is on the clean split.

## The finding that explains everything

The dataset was generated in families. One scenario was reworded several
times, and every rewording kept its queue. That one fact decides what works.

| Similarity of a test ticket to its nearest training ticket | Share of test | English only | With German |
| --- | --- | --- | --- |
| below 0.3, no relative in training | 18% | 45% | 78 to 82% |
| 0.3 to 0.5 | 21% | 59 to 84% | 80 to 89% |
| above 0.5, a relative exists | 61% | 96 to 100% | 96 to 100% |

If a ticket has a relative in the pool, looking that relative up is nearly
perfect. If it has none, every method sits near 40%, including one that was
handed the dataset's own tags and priority. Accuracy is coverage, not
understanding. That is why bigger models lost (nothing in the text to
generalize toward), why synthetic data did not help (a generator can reword
tickets it has seen, it cannot label a family it has not), and why the
16,500 German tickets the pipeline used to discard were worth nine points once
translated (they are real relatives with real labels).

## The final model

`src/final/router.py`, one function, `fit_predict_proba(train_texts, train_labels, eval_texts)`.

1. Every labelled ticket, English and translated German, sits in one pool.
2. A new ticket is compared against the pool two ways, by word overlap
   (TF-IDF) and by meaning (a multilingual sentence embedding), against both
   the English and the translated pool. Each lookup returns its 20 closest
   tickets and their queues.
3. For each of the ten queues the model records how close the nearest ticket
   of that queue was and how many of the 20 belonged to it.
4. A small gradient-boosted model turns those numbers into a calibrated
   probability per queue. It is trained on the training tickets in five folds,
   so every ticket was scored against a pool that did not contain it, the same
   situation a new ticket is in.
5. Non-English tickets are translated first with an open 74M-parameter model
   (`final/translate.py`). It is run, never retrained.

The heavy part is a lookup and the learned part is small. Fitting takes about
five minutes on a GPU, routing one ticket takes milliseconds, and every model
involved is open and runs on the company's own machines.

Because the confidence is calibrated (expected calibration error 0.012), the
threshold means what it says.

| Confidence cut | Auto-routed | Accuracy on those | Sent to a person |
| --- | --- | --- | --- |
| none | 100% | 91.8% | 0% |
| 0.7 | 87% | 98.3% | 13% |
| 0.9 | 81% | 99.6% | 19% |

Every queue is between 87% and 96% recall, including the small ones. Per-queue
numbers are in `results/12b_stack_german.json`.

## How the test set was protected

The training set is split once more into a core (85%) and a validation slice
(15%, `data/synthetic/split.json`). Every model choice, threshold, guard and
epoch count was decided on the validation slice. The test set was read once per
final system by `final/test_once.py`, which refuses to run twice.

German tickets pass a leak guard before entering the pool. Any translated
ticket whose similarity to an evaluation ticket is at or above 0.60 (word
overlap) or 0.988 (embedding) is dropped, so a German copy of a test ticket can
never hand the model the answer. Both final systems were independently
re-run from scratch by a second process, reproduced to the bit, and checked
with shuffled training labels, which collapsed accuracy to 29%, the majority
share.

## What was tried and why it did not work

`src/experiments/bigger_models/`. Transformers, LLM classifiers, a BiLSTM,
Laya, zero-shot models. All below the bag-of-words network. The text does not
carry the label, the family does.

`src/experiments/synthetic_data/`. Five rounds. Prompted Gemma and Qwen,
Qwen with reasoning aimed at the weak queues, a Qwen fine-tuned to write new
tickets, and a Qwen fine-tuned to reword real tickets (siblings). The last one
matched the real data's style (similarity 0.64, real siblings 0.635) and moved
retrieval by 0.0 points. It did give the small network +3.2 on validation, and
a pool made only of synthetic siblings keeps 95% of retrieval accuracy, which
is a privacy result. A real-versus-synthetic test settled it: 2,850 real
tickets added +3.1 points, 7,700 synthetic ones added +0.7.
`weak_check.py` holds the paired bootstrap used for every such comparison.

`src/experiments/german_embeddings/`. The first attempt at German, through
multilingual embeddings into a network. That network sat at 44%, far below the
68% bag-of-words one, and the German rows lifted it to 54% at a loose leak
threshold, a gain that shrank to inside the noise once translation twins of
evaluation tickets were filtered strictly. The approach was dropped. Retrieval
is what made German count.

`src/experiments/retrieval_tracks/`. Five research tracks that ran in
parallel once the family structure was understood. Lexical stacking, dense
embeddings, parametric classifiers, a specialist for tickets with no relative,
and the German translation track. The semantic and German tracks produced the
two final routers. The `x_verify_*` scripts are the independent re-runs.

## Layout

```
src/common.py                       cleaning, the deduplicated split, the threshold table
src/final/                          the model that ships
    router.py                         fit_predict_proba, German pool, 91.8%
    router_english_only.py            same idea, English only, 83.1%
    features.py  stacker.py  lib.py   retrieval channels and the gradient-boosted stacker
    embed.py  translate.py            the sentence embedder and the German translator
    test_once.py                      the one-shot test run
src/baselines/                      TF-IDF models, the width sweep, latency and memory benchmarks
src/experiments/                    everything else, in story order
src/serve/serve.py                  the FastAPI service (/health, /predict)
configs/                            LoRA configs for the fine-tuning runs
scripts/                            shell runners for the long training ladders
infra/                              droplet, firewall, cloud-init, deploy script, teammate access
docs/                               MODEL_STORY.md is the narrative, HIGH_AVAILABILITY.md and SIZING.md the cloud design
results/                            one JSON per system, logs, and test probabilities
data/                               see data/README.md, large derived files are not in git
```

## Reproducing the final number

```bash
uv venv --python 3.12 .venv
VIRTUAL_ENV=.venv uv pip install torch transformers sentence-transformers scikit-learn pyarrow numpy sentencepiece sacremoses
export PYTHONPATH=src HF_HUB_DISABLE_XET=1
.venv/bin/python src/common.py                     # the deduplicated split, seed 42
.venv/bin/python src/final/translate.py extract    # the German rows of the parquet, to data/x_german/
.venv/bin/python src/final/translate.py translate  # German to English, about 11 GPU minutes, resumable
.venv/bin/python src/final/translate.py assemble   # rebuild the tickets, write the pool file
.venv/bin/python src/final/embed.py                # embed the German pool once
.venv/bin/python src/final/router.py               # core to validation, 89.9%
.venv/bin/python src/final/test_once.py german     # the test run, refuses if results/12b_stack_german.json exists
```

The baselines are `src/baselines/train_tfidf.py` and `src/baselines/train_mlp.py`.

## Why this is a cloud project

Not because the model is big. The final router is small. It is because of the
shape of the work and where it has to run.

The heavy steps are bursts. Translating the German tickets took 11 minutes on
a GPU, embedding the pool takes a few, refitting the stacker takes five. Then
nothing until the next refresh. That is a poor fit for owning a GPU and a good
fit for renting one for the job. Serving is the opposite, small and always on,
and it has to survive a dead container or droplet, which is what the load
balancer and the second droplet in `docs/HIGH_AVAILABILITY.md` are for. Every
ticket a person corrects goes back into the pool as a real relative with a
real label, so a nightly refresh keeps improving the model without anyone
retraining anything by hand. And since customer tickets cannot go to an
outside translation or model API, all of it has to run inside the company's
own private network, which a rented VPC provides cheaply and a SaaS product
cannot provide at all.

What exists today is one droplet with the container, a firewall, a deploy
script and a health check (`infra/`). What the final system still needs is a
ticket database with a feedback endpoint, object storage for the fitted model,
a refresh job on its own droplet, and the serving pair behind the balancer.
