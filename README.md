# Support ticket routing

Team 3, MIS 547. Our model reads a customer support ticket and sends it to one of ten queues.
It runs as a live HTTPS endpoint on DigitalOcean and handles English and German tickets. The
full write-up is our final report.

## Try the endpoint

The endpoint is **https://146-190-188-160.sslip.io**. Routing a ticket needs an API key in the
`X-API-Key` header. The grader's key is in section 7.1 of our report. It is never in this
repository, since the repository is public.

```bash
export ROUTER_URL=https://146-190-188-160.sslip.io
export ROUTER_API_KEY=paste-the-key-from-the-report

curl -s -X POST "$ROUTER_URL/v1/route" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: $ROUTER_API_KEY" \
  -d '{"subject": "Charged twice for my subscription", "body": "I was billed twice this month for the same plan and need the duplicate charge refunded to my card."}'
```

This comes back as Billing and Payments at about 96% confidence with `"auto_routed": true`.
Anything under 0.7 confidence goes to a person instead. To send all five sample tickets in
[examples/tickets.json](examples/tickets.json) at once, run the script below. It needs `jq`.

```bash
scripts/try_endpoint.sh
```

Errors come back as JSON with a code and a message, for example a missing key (401), an empty
ticket (422) or more than 60 requests in a minute (429). Interactive docs are at `/docs`.

| Endpoint | What it does |
| --- | --- |
| `POST /v1/route` | Routes a ticket and returns the queue, the confidence and the top three queues |
| `POST /v1/feedback` | Records an agent's correction, `{"ticket_id", "correct_queue"}` |
| `GET /v1/queues`, `/v1/model`, `/v1/drift` | The queue names, the served model and its metrics, the drift report |
| `GET /healthz`, `/readyz` | Health checks, no key needed |

## Where to find each part

| Rubric item | Where |
| --- | --- |
| Architecture | [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) and the diagram [docs/architecture.png](docs/architecture.png) |
| Data and preprocessing | [data/README.md](data/README.md) and `clean()` in [src/common.py](src/common.py), which training and serving share |
| Model | [src/final/](src/final), and how we got there in [docs/MODEL_STORY.md](docs/MODEL_STORY.md) |
| Security | [docs/SECURITY.md](docs/SECURITY.md) and the scanner reports in [evidence/](evidence/README.md) |
| Observability and drift | [docs/OBSERVABILITY.md](docs/OBSERVABILITY.md) |
| Costs | [docs/COSTS.md](docs/COSTS.md) |
| Terraform | [infra/terraform/](infra/terraform), with the runbook in [infra/README.md](infra/README.md) |
| GitHub Actions | [.github/workflows/](.github/workflows), described below |
| Container image | `ghcr.io/nickyrice04/mis547-ticket-routing:main`, public, so `docker pull` works without a login |
| Works cited | [docs/REFERENCES.md](docs/REFERENCES.md) |

In the DigitalOcean account everything is in the **MIS547** project and tagged `team3`. The GPU
training droplet only exists during a training run, so it is normally not there.

## How the model works

1. German tickets are translated to English inside the service.
2. The ticket is compared with about 35,500 labeled past tickets, once by word overlap (TF-IDF)
   and once by meaning (sentence embeddings), and the 20 closest come back.
3. A small gradient-boosted model turns how close those tickets are, and which queues they
   belong to, into a probability for each queue.
4. At 0.7 or higher the ticket is routed on its own. Below that it goes to an agent along with
   the top three queues.

The API serves `src/final/model.py`. A test checks that it makes the same predictions as
`src/final/router.py`, the code we evaluated.

## Results

Every number is on the same 4,750 held-out test tickets, except Gemma 3, which used a random
500 of them.

| Model | Accuracy | Macro-F1 |
| --- | --- | --- |
| TF-IDF and logistic regression, our baseline | 61.6% | 0.625 |
| TF-IDF and a small neural network | 68.3% | 0.680 |
| DistilBERT, fine-tuned | 67.7% | 0.658 |
| Gemma 3 4B, fine-tuned with LoRA | 43.2% | 0.354 |
| Retrieval and stacker, English tickets only | 83.1% | 0.832 |
| Retrieval and stacker, plus translated German tickets (final) | **91.8%** | **0.917** |

At the 0.7 threshold, 87% of tickets are routed automatically at 98.3% accuracy. Bigger models
lost because many tickets are rewordings of the same scenario, so finding a ticket's closest
relatives beats reading it more carefully. Every model we tried is in
[results/SUMMARY.md](results/SUMMARY.md).

## CI/CD

Every push and pull request to `main` runs [ci.yml](.github/workflows/ci.yml).

| Job | Tool | What it checks |
| --- | --- | --- |
| test | pytest | The API, the drift rules and preprocessing |
| sast | Semgrep | The shipped code, with the Python, security-audit and secrets rules |
| secrets | Gitleaks | The whole git history |
| deps | pip-audit | The pinned runtime packages |
| terraform | Terraform | `fmt` and `validate` |
| image | Docker, Trivy | Builds and scans the image, makes an SBOM, and publishes it on `main` |

The droplet checks for a new image every five minutes and rolls back if it is not healthy
([deploy/update.sh](deploy/update.sh)). GitHub never connects to the server.
[drift-check.yml](.github/workflows/drift-check.yml) asks `/v1/drift` for a report every day and
opens an issue if it finds drift.

## Run it yourself

The tests use a fake model, so they take a few seconds. You need Python 3.12.

```bash
python3.12 -m venv .venv && . .venv/bin/activate
pip install numpy==2.5.3 fastapi==0.141.1 pydantic==2.13.5 sqlalchemy==2.1.1 prometheus-client==0.26.0 httpx==0.28.1 pytest==9.1.1
pytest
```

To train the model and serve it on your own machine, use [uv](https://docs.astral.sh/uv/). The
first run downloads two pretrained models and translates the German tickets. On our laptop GPU
the whole run took about 9 minutes.

```bash
uv venv --python 3.12 .venv-train
VIRTUAL_ENV=.venv-train uv pip install -r docker/requirements-train.txt
PYTHONPATH=src .venv-train/bin/python -m mlops.train_pipeline --local-only
```

It prints the validation accuracy (about 90%) and saves the model to
`artifacts/<version>/router.joblib`. Then pick any key for your copy and start the API.

```bash
API_KEYS=me:my-local-key ARTIFACT_PATH=artifacts/<version>/router.joblib \
  PYTHONPATH=src .venv-train/bin/uvicorn serve.app:app --app-dir src
```

In a second terminal, `ROUTER_URL=http://127.0.0.1:8000 ROUTER_API_KEY=my-local-key scripts/try_endpoint.sh`
sends it the sample tickets. `API_KEYS` takes `name:key` pairs separated by commas. The name
shows up in the logs and the key never does. The test set was scored once by
`src/final/test_once.py german`, which refuses to run again while
`results/12b_stack_german.json` exists.

## Layout

```
src/final/          the model that ships, model.py is what the API serves
src/serve/app.py    the API
src/mlops/          training job, Spaces storage, audit database, drift
src/baselines/      where we started, the TF-IDF models
src/experiments/    everything we tried in between
tests/              the API, drift rules, preprocessing and the model parity check
deploy/             Docker Compose stack (API, Caddy, Prometheus) and the updater
infra/              Terraform for every DigitalOcean resource, plus helper scripts
docs/               architecture, security, observability, costs, model story, works cited
evidence/           scanner reports, the SBOM, training and rollout logs
examples/ scripts/  sample tickets and the endpoint test
results/            one JSON per model we tested
```

## Team

Nicky Rice (machine learning), Anshul Shah (cloud infrastructure and security) and Alejandro
Emmanuel (the API, containers and database). MIS 547, Professor Zara Ahmad-Post, University of
Arizona.
