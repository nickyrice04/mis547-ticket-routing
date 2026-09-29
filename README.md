# Support ticket routing

Team 3, MIS 547. A model that reads a customer support ticket and routes it to
one of ten queues, served as a live HTTPS endpoint on DigitalOcean. Tickets
arrive in English and German, and nothing leaves the company's private network.

| | |
| --- | --- |
| Endpoint | **https://146-190-188-160.sslip.io** (the grader's API key is in the report, never in git) |
| Try it | [examples/tickets.json](examples/tickets.json) and [scripts/try_endpoint.sh](scripts/try_endpoint.sh) |
| Architecture | [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md), diagram in [docs/architecture.png](docs/architecture.png) |
| Security | [docs/SECURITY.md](docs/SECURITY.md), STRIDE threat model and scan evidence |
| Observability | [docs/OBSERVABILITY.md](docs/OBSERVABILITY.md), audit log, metrics, drift detection |
| Costs | [docs/COSTS.md](docs/COSTS.md), monthly budget and comparison with managed platforms |
| High availability | [docs/HIGH_AVAILABILITY.md](docs/HIGH_AVAILABILITY.md), answers to the midterm feedback |
| How the model was found | [docs/MODEL_STORY.md](docs/MODEL_STORY.md) |
| Other people's results | [docs/EXTERNAL_BENCHMARKS.md](docs/EXTERNAL_BENCHMARKS.md), the public Kaggle notebooks on this dataset |
| Evidence | [evidence/](evidence/README.md), scanner reports, the GPU training log, the live replay |
| Works cited | [docs/REFERENCES.md](docs/REFERENCES.md) |
| Infrastructure runbook | [infra/README.md](infra/README.md) |

The repository is laid out in the order the project happened. `src/final/` is
the model that ships. `src/baselines/` is where it started. `src/experiments/`
is everything tried in between, kept because the negative results are half
the story. `src/serve/`, `src/mlops/`, `deploy/`, `infra/` and `.github/` are
how it runs in the cloud.

## For graders

Where each part of the rubric lives.

| Criterion | Where to look |
| --- | --- |
| Inference endpoint | [Calling the endpoint](#calling-the-endpoint) below, sample tickets in [examples/tickets.json](examples/tickets.json). Errors come back as JSON with a code and a message |
| Code repositories | This README, one repository for the whole system. CI and the scanners in [.github/workflows/ci.yml](.github/workflows/ci.yml), Terraform in [infra/terraform/](infra/terraform), the image at `ghcr.io/nickyrice04/mis547-ticket-routing:main` (public, `docker pull` works without a login), scan results in [evidence/](evidence/README.md) |
| Architecture | [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) and [Why this runs in the cloud](#why-this-runs-in-the-cloud) below |
| Dataset and data requirements | [data/README.md](data/README.md), the preprocessing shared by training and serving in [src/common.py](src/common.py), the audit database in [src/mlops/db.py](src/mlops/db.py) |
| Security | [docs/SECURITY.md](docs/SECURITY.md) and [evidence/](evidence/README.md) |
| Observability | [docs/OBSERVABILITY.md](docs/OBSERVABILITY.md), the live drift report at `GET /v1/drift` |
| Cloud spend | [docs/COSTS.md](docs/COSTS.md) |
| Implementation | [docs/MODEL_STORY.md](docs/MODEL_STORY.md), [docs/HIGH_AVAILABILITY.md](docs/HIGH_AVAILABILITY.md), [Team](#team) |
| Citations | [docs/REFERENCES.md](docs/REFERENCES.md) |

In the DigitalOcean account everything is filed under the **MIS547** project and tagged
`team3`. That is the inference droplet, its reserved IP, the PostgreSQL cluster, the Spaces
bucket, the VPC, two firewalls, the alerts and the uptime checks. The GPU training droplet
only exists while a training run is going, so it is normally absent. That is by design,
see below.

## Calling the endpoint

```bash
curl -s -X POST "$ROUTER_URL/v1/route" \
  -H "Content-Type: application/json" -H "X-API-Key: $ROUTER_API_KEY" \
  -d '{"subject": "Charged twice for my subscription", "body": "I was billed twice this month for the same plan and need the duplicate charge refunded to my card."}'
```

```json
{"ticket_id": "119b9691-...", "queue": "Billing and Payments", "confidence": 0.9571, "auto_routed": true,
 "threshold": 0.7, "top_queues": [{"queue": "Billing and Payments", "probability": 0.9571}, ...],
 "familiarity": 0.305, "language": "en", "translated": false, "model_version": "v20260929-033811",
 "audit_logged": true, "latency_ms": 792.5, "request_id": "..."}
```

German works the same way. `{"subject": "Rückerstattung für doppelte Abbuchung", "body": "..."}`
is translated inside the service and routed to Billing and Payments with `"translated": true`.
On the live droplet an English ticket takes a median of 0.8 seconds and a German one about
4.5, most of it the sentence embedding and translation on shared CPUs (see the latency section
of [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)).

Errors come back as JSON with a code and a message, for example a missing key (401), an
empty ticket (422) or too many requests (429). Interactive documentation is at `/docs`.

| Endpoint | Purpose |
| --- | --- |
| `POST /v1/route` | route a ticket, returns the queue, confidence, top three and a ticket id |
| `POST /v1/feedback` | a person confirms or corrects a routing decision, `{"ticket_id", "correct_queue"}` |
| `GET /v1/queues` | the ten queue names |
| `GET /v1/model` | the served model version, how it was trained, its validation metrics |
| `GET /v1/drift` | drift report over recent traffic |
| `GET /healthz`, `GET /readyz` | liveness and readiness, no key needed |

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
The only public notebook that measures the same task on this dataset reports 53.2%
([docs/EXTERNAL_BENCHMARKS.md](docs/EXTERNAL_BENCHMARKS.md)).

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

`src/final/model.py` is the object the API serves (fit, save, load, predict).
`src/final/router.py` is the same recipe as the one function that was evaluated,
`fit_predict_proba(train_texts, train_labels, eval_texts)`, and a test proves the two agree
to the bit.

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

The heavy part is a lookup and the learned part is small. A full retrain takes
8 minutes on a rented GPU, routing an English ticket takes under a second on a small CPU,
and every model involved is open and runs on the company's own machines.

Because the confidence is calibrated (expected calibration error 0.012), the
threshold means what it says.

| Confidence cut | Auto-routed | Accuracy on those | Sent to a person |
| --- | --- | --- | --- |
| none | 100% | 91.8% | 0% |
| 0.7 | 87% | 98.3% | 13% |
| 0.9 | 81% | 99.6% | 19% |

Every queue is between 87% and 96% recall, including the small ones. Per-queue
numbers are in `results/12b_stack_german.json`.

## Why this runs in the cloud

The final router is small. It needs 2 GB of memory to serve and no GPU at all. So
the case for cloud computing is not size. It is the shape of the work, which has two
very different halves, and the rule that customer text stays inside the company.

| Step | Where it runs | Measured |
| --- | --- | --- |
| Translate the German pool (76,327 sentences) | Rented GPU, only during a refresh | 3.5 minutes, against about 11 on a laptop GPU |
| Embed the pools, validate, fit the model | Rented GPU | 4.3 minutes |
| Publish the new model | GPU droplet to Spaces, same region | 185 MB |
| Route a ticket | $24 a month CPU droplet, always on | median 0.8 s English, 4.5 s German |
| Record the decision | Managed PostgreSQL, private network only | about 2 KB per ticket |

**Training is a burst, and a burst is cheaper to rent than to own.** A refresh needs a
GPU for a few minutes, then nothing until the next one. The measured retrain on a rented
NVIDIA RTX 4000 Ada took 8 minutes end to end, about $0.10 of GPU time, or about $0.19 once
droplet boot and setup are counted ([evidence/gpu-training-run.log](evidence/gpu-training-run.log)).
Terraform creates the GPU droplet for a run and destroys it right after, because it bills by
the hour whether it is working or not. Rented for a weekly refresh it costs under $1 a month.
Left running it would cost $555 a month.

**Serving is small and constant, so it gets a small machine that never sleeps.** Tickets
arrive at any hour and are routed one at a time, the moment they are submitted. A GPU here
would sit idle between tickets, so the API runs on 2 shared vCPUs and 4 GB for $24 a month.
If the bank wanted steadier latency, a dedicated-CPU droplet is $18 a month more and a
one-line Terraform change.

**Customer text never leaves the bank's network.** Tickets carry names, account numbers and
card fragments. Every model in the pipeline, the translator, the embedder and the classifier,
is open and runs on machines the team controls inside a private network (a VPC). Nothing is
sent to an outside translation service or language model API. The database only accepts
connections from the team's own droplets, and the bucket holding the models is private. A
SaaS triage product would need the tickets to leave.

**Managed services take the undifferentiated work.** The audit log lives in Managed
PostgreSQL, where backups, patching and engine failover are DigitalOcean's job. The dataset
and every model version live in Spaces with versioning on, so any past model can be restored
by moving one pointer.

**The whole system is code.** Every resource is defined in Terraform. A push to `main` is
tested, scanned and published by GitHub Actions, and the droplet rolls to the new image by
itself, 19 seconds from rollout to healthy in the recorded run
([evidence/cd-rollout-and-latency.txt](evidence/cd-rollout-and-latency.txt)), or rolls back
if the new image is unhealthy. A dead droplet is rebuilt with one command and keeps its address.

**Growing is a change to a number, not a purchase.** A second droplet behind a load
balancer and a standby database node add about $51 to $81 a month
([docs/HIGH_AVAILABILITY.md](docs/HIGH_AVAILABILITY.md), [docs/COSTS.md](docs/COSTS.md)).

**Renting has one real catch, and the design plans for it.** When the GPU was first
requested, both the RTX 4000 Ada and the larger RTX 6000 Ada were sold out in the region.
A retry script ([infra/scripts/wait-for-gpu.sh](infra/scripts/wait-for-gpu.sh)) got an
RTX 4000 Ada a few minutes later. Because the API keeps serving the last approved model, a delayed retrain
never stops routing.

**What it costs.** About $45 a month for the proof of concept with weekly retraining, and
about $97 to $127 with high availability. The same shape on AWS SageMaker is roughly $190 to
$210, and per-seat SaaS triage roughly $500 to $1,500 for 20 agents, with the tickets leaving
the network. Details are in [docs/COSTS.md](docs/COSTS.md).

**Why the model search ran on a laptop.** Most of the model search, about twenty
systems ([results/SUMMARY.md](results/SUMMARY.md)), ran on a team member's laptop GPU to keep
the class account's spending down. A month of exploration like this one is about 40 GPU
hours, around $30 on the rented GPU. A company usually would not have that option, since its
ticket data cannot sit on a personal machine. So the final pipeline runs where production
would, on the rented GPU, and scored 89.8% on validation there against 89.9% on the laptop.

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
is a privacy result. A real-versus-synthetic test settled it. 2,850 real
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

## Continuous integration and deployment

Every push and pull request runs [.github/workflows/ci.yml](.github/workflows/ci.yml).

| Job | What it checks |
| --- | --- |
| test | unit tests of the API contract, the drift rules and preprocessing |
| sast | Semgrep with the Python, security-audit and secrets rule packs on the shipped code |
| secrets | Gitleaks over the whole git history |
| deps | pip-audit of the pinned runtime dependencies |
| terraform | `terraform fmt` and `validate` |
| image | builds the inference image, Trivy scan (fails on fixable CRITICAL), CycloneDX SBOM, and on `main` publishes `ghcr.io/nickyrice04/mis547-ticket-routing:main` |

Deployment is pull-based. The inference droplet checks the registry every five minutes
(`deploy/update.sh`), rolls to a new image, and rolls back if it is not healthy within five
minutes. CI holds no cloud credentials and never connects to a server.
[.github/workflows/drift-check.yml](.github/workflows/drift-check.yml) calls `/v1/drift`
daily and opens an issue when drift is detected.

Local scan results are in [evidence/](evidence/README.md). No secrets in history, 0 Semgrep
findings on the shipped code, 0 vulnerabilities in the image, and the SBOM.

## Layout

```
src/common.py                       cleaning, the deduplicated split, the threshold table
src/final/                          the model that ships
    model.py                          Router, the fit, save, load, predict object the API serves
    router.py                         fit_predict_proba, German pool, the evaluated code path, 91.8%
    router_english_only.py            same idea, English only, 83.1%
    features.py  stacker.py  lib.py   retrieval channels and the gradient-boosted stacker
    embed.py  translate.py            the sentence embedder and the German translator
    test_once.py                      the one-shot test run
src/serve/app.py                    the inference API (FastAPI)
src/mlops/                          training pipeline, Spaces storage, PostgreSQL audit log, drift
src/baselines/                      TF-IDF models, the width sweep, latency and memory benchmarks
src/experiments/                    everything else, in story order
tests/                              API contract, drift rules, preprocessing, and the model parity check
Dockerfile                          the inference image
deploy/                             Compose stack (API, Caddy, Prometheus), updater, training job
infra/terraform/                    every DigitalOcean resource as code
infra/scripts/                      credentials helper, post-apply bootstrap, database migration, GPU retry
.github/workflows/                  CI (tests, Semgrep, Gitleaks, pip-audit, Terraform, Trivy, SBOM) and the daily drift check
docs/                               architecture, security, observability, costs, high availability, model story, works cited
evidence/                           scanner reports, SBOM, GPU training log, live replay, CD rollout, SSH probe log
examples/  scripts/                 sample tickets, endpoint test, traffic replay, model publish and rollback
configs/                            LoRA configs for the fine-tuning runs
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
.venv/bin/python src/final/translate.py translate  # German to English, 3.5 GPU minutes on the RTX 4000 Ada, resumable
.venv/bin/python src/final/translate.py assemble   # rebuild the tickets, write the pool file
.venv/bin/python src/final/embed.py                # embed the German pool once
.venv/bin/python src/final/router.py               # core to validation, 89.9%
.venv/bin/python src/final/test_once.py german     # the test run, refuses if results/12b_stack_german.json exists
```

The baselines are `src/baselines/train_tfidf.py` and `src/baselines/train_mlp.py`. Tests run
with `pytest` (the slow model parity check with `pytest -m slow`).

## Team

Team 3, MIS 547, Professor Zara Ahmad-Post, University of Arizona.

| Member | Focus |
| --- | --- |
| Nicky Rice | Machine learning, model training and evaluation |
| Anshul Shah | Cloud infrastructure and network security |
| Alejandro Emmanuel | The API, containers and the database |
