# Support ticket routing

Team 3, MIS 547. A model that reads a customer support ticket and routes it to
one of ten queues, served as a live HTTPS endpoint on DigitalOcean. Tickets
arrive in English and German, and nothing leaves the company's private network.

| | |
| --- | --- |
| Endpoint | `https://<reserved-ip-with-dashes>.sslip.io` (the exact URL and the grader's API key are in the report) |
| Try it | [examples/tickets.json](examples/tickets.json) and `scripts/try_endpoint.sh` |
| Architecture | [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md), diagram in [docs/architecture.png](docs/architecture.png) |
| Security | [docs/SECURITY.md](docs/SECURITY.md), STRIDE threat model and scan evidence |
| Observability | [docs/OBSERVABILITY.md](docs/OBSERVABILITY.md), audit log, metrics, drift detection |
| Costs | [docs/COSTS.md](docs/COSTS.md), monthly budget and comparison with managed platforms |
| High availability | [docs/HIGH_AVAILABILITY.md](docs/HIGH_AVAILABILITY.md), answers to the midterm feedback |
| How the model was found | [docs/MODEL_STORY.md](docs/MODEL_STORY.md) |
| Infrastructure runbook | [infra/README.md](infra/README.md) |

The repository is laid out in the order the project happened. `src/final/` is
the model that ships. `src/baselines/` is where it started. `src/experiments/`
is everything tried in between, kept because the negative results are half
the story. `src/serve/`, `src/mlops/`, `deploy/`, `infra/` and `.github/` are
how it runs in the cloud.

## Calling the endpoint

```bash
curl -s -X POST "$ROUTER_URL/v1/route" \
  -H "Content-Type: application/json" -H "X-API-Key: $ROUTER_API_KEY" \
  -d '{"subject": "Charged twice for my subscription", "body": "I was billed twice this month for the same plan and need the duplicate charge refunded to my card."}'
```

```json
{"ticket_id": "4a2b6cc8-...", "queue": "Billing and Payments", "confidence": 0.9596, "auto_routed": true,
 "threshold": 0.7, "top_queues": [{"queue": "Billing and Payments", "probability": 0.9596}, ...],
 "familiarity": 0.305, "language": "en", "translated": false, "model_version": "v20260929-014804",
 "audit_logged": true, "latency_ms": 33.8, "request_id": "..."}
```

German works the same way, `{"subject": "Rückerstattung für doppelte Abbuchung", "body": "..."}`
is translated inside the service and routed to Billing and Payments. Errors come back as JSON
with a code and a message, for example a missing key (401), an empty ticket (422) or too many
requests (429). Interactive documentation is at `/docs`.

| Endpoint | Purpose |
| --- | --- |
| `POST /v1/route` | route a ticket, returns the queue, confidence, top three and a ticket id |
| `POST /v1/feedback` | a person confirms or corrects a routing decision, `{"ticket_id", "correct_queue"}` |
| `GET /v1/queues` | the ten queue names |
| `GET /v1/model` | the served model version, how it was trained, its validation metrics |
| `GET /v1/drift` | drift report over recent traffic |
| `GET /healthz`, `GET /readyz` | liveness and readiness |

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
infra/scripts/                      credentials helper and post-apply bootstrap
.github/workflows/                  CI (tests, Semgrep, Gitleaks, pip-audit, Terraform, Trivy, SBOM) and the daily drift check
docs/                               architecture, security, observability, costs, high availability, model story
evidence/                           scanner reports, SBOM, drift demonstration, SSH probe log
examples/  scripts/                 sample tickets, endpoint and traffic replay scripts
configs/                            LoRA configs for the fine-tuning runs
results/                            one JSON per system, logs, and test probabilities
data/                               see data/README.md, large derived files are not in git
```

## Continuous integration and deployment

Every push and pull request runs [.github/workflows/ci.yml](.github/workflows/ci.yml):

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

Local scan results are in [evidence/](evidence): no secrets in history, 0 Semgrep findings
on the shipped code, 0 fixable HIGH or CRITICAL vulnerabilities in the image, and the SBOM.

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

The heavy steps are bursts. Translating the German tickets, embedding the pools and
refitting the stacker need a GPU for a short while, then nothing until the next refresh.
A company still searching for the right model, as this project did, runs many such bursts.
That is a poor fit for owning a GPU and a good fit for renting one by the hour, which is why
the GPU droplet exists only while `training_enabled` is on. Serving is the opposite, small
and always on, so it runs on a $24 CPU droplet. Every ticket a person corrects goes back into
the pool as a real relative with a real label, so each refresh improves the model. And since
customer tickets cannot go to an outside translation or model API, all of it runs inside the
company's own private network, which a rented VPC provides cheaply and a SaaS product cannot.
