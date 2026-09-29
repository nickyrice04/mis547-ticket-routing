# Observability

Lecture 15 separates telemetry (the data), monitoring (acting on known questions) and
observability (answering questions nobody predicted). This is how each pillar is covered,
how drift is detected, and how model quality is evaluated over time.

## Telemetry

### Logs, including the audit log

| Source | What it records | Where it lives |
| --- | --- | --- |
| Audit log, table `predictions` | Every routing decision: time, request id, API key name, model version, language, whether it was translated, the masked ticket text, queue, confidence, whether it was auto-routed, the threshold, top three queues, familiarity, total latency and time per stage | Managed PostgreSQL, private network |
| Corrections, table `feedback` | A person's confirmation or correction, who sent it, and whether the model agreed | PostgreSQL |
| Training runs, table `training_runs` | Every run: host, GPU, git commit, rows used, validation accuracy, macro-F1 and ECE, time per stage, status (promoted, rejected, failed), artifact | PostgreSQL, plus `metrics.json` next to each model in Spaces |
| Application log | One JSON object per line: model loaded, ready, routed, feedback, authentication failures, audit-write failures, model updates | Docker's log, rotated at 10 MB times 5 |
| Access log | Every HTTP request through Caddy, as JSON | Docker's log |

Because every decision row carries the request id, the key name, the model version and
the per-stage timings, questions nobody planned for can still be answered with one SQL
query. For example, which client sent the most German tickets last week, whether latency
rose after a particular model version, or how often tickets below 0.3 familiarity were
corrected.

### Metrics

The API exposes Prometheus metrics at `/metrics` on the private Docker network.
Prometheus on the same droplet scrapes them every 15 seconds and keeps 30 days. Caddy
returns 404 for `/metrics` from the internet. To look at them from a laptop:

```bash
ssh -L 9090:localhost:9090 nicky@<inference-ip>     # then open http://localhost:9090
```

| Metric | Meaning |
| --- | --- |
| `router_requests_total{endpoint,status}` | Requests by endpoint and HTTP status |
| `router_request_seconds` | Latency histogram per endpoint |
| `router_stage_seconds{stage}` | Time in translate and clean, embed, retrieve and stack |
| `router_predictions_total{queue,auto_routed}` | Decisions by queue and by automation |
| `router_confidence` | Distribution of the top probability |
| `router_nearest_similarity` | Distribution of familiarity, the leading drift signal |
| `router_translated_total` | Tickets that needed translation |
| `router_feedback_total{agreed}` | Corrections, and how many agreed with the model |
| `router_audit_failures_total` | Decisions that could not be logged, which forced them to a person |
| `router_rate_limited_total{key}` | Requests refused by the rate limit |
| `router_model_info{version}` | Which model version is serving |

DigitalOcean's monitoring agent adds host metrics (CPU, memory, disk, bandwidth) for both
droplets, and the managed database has its own metrics page.

### Traces

The service is one process, so a request's journey is recorded as stage timings on the
same audit row (`stage_ms`) and as the `router_stage_seconds` histogram, tied together by
the request id that also goes back to the caller in `X-Request-ID`. If the router is ever
split into services (translation, retrieval, stacking), OpenTelemetry would carry the same
request id across them as a distributed trace.

## Monitoring and alerts

| Check | Condition | Who hears about it |
| --- | --- | --- |
| Uptime check, from us_east and us_west | `GET https://<host>/healthz` fails for 2 minutes | Email |
| Uptime latency | Response slower than 2 seconds over 5 minutes | Email |
| Certificate | TLS certificate within 14 days of expiry | Email |
| CPU | Above 85% for 5 minutes | Email |
| Memory | Above 90% for 5 minutes | Email |
| Disk | Above 85% on either droplet | Email |
| Container health | `/healthz` failing three times in a row | The five-minute watchdog restarts the container, and Docker restarts one that exits |
| Image rollout | New image not healthy within 5 minutes | `deploy/update.sh` rolls back automatically |
| Model drift | `/v1/drift` reports drift | A GitHub issue opened by the daily drift workflow |

The external checks matter because a dead droplet cannot report its own death. That is the
answer to the midterm question "how will you know if your container has crashed?".

## Model drift

`GET /v1/drift?days=7` compares recent traffic in the audit log with the reference the
training job recorded on the validation slice. Signals and thresholds (src/mlops/drift.py):

| Signal | Why it was chosen | Warn | Drift |
| --- | --- | --- | --- |
| Unfamiliar share, tickets whose nearest labelled ticket is below 0.3 similarity | The router is right 96 to 100% of the time on familiar tickets and near a guess on unfamiliar ones, so a rise predicts an accuracy drop before corrections arrive | +5 points | +10 points |
| Queue mix, population stability index | A shift in what customers write about, or a model that started favouring one queue | 0.1 | 0.2 |
| Mean confidence | The model becoming less sure of itself | down 0.05 | |
| Live accuracy on reviewed tickets | The only direct accuracy measure, from corrections | | below 0.85 with at least 50 corrections |

The report needs 200 predictions in the window before it judges anything.

**Demonstration (evidence/drift-demo-local.json).** Against the containerised service with
PostgreSQL, 300 held-out test tickets were replayed with half of them corrected (the way an
agent would confirm them), then 120 off-topic tickets (recipes, sports, weather).

| Traffic | Status | Unfamiliar share change | Confidence change | Live accuracy |
| --- | --- | --- | --- | --- |
| 300 held-out tickets | ok | +0.5 points | +0.01 | 92.9% on 141 corrections |
| plus 120 off-topic tickets | **drift** | **+23.9 points** | **down 0.17** | 93.6% on 156 corrections |

Live accuracy stayed high in the second row because nobody corrected the off-topic
tickets. The familiarity signal caught the new kind of traffic anyway. That is why it
leads the list.

## Evaluating model quality

- **Before promotion.** Every training run scores the evaluated recipe on the validation
  slice (accuracy, macro-F1, calibration error, per-queue recall, threshold table) and the
  gate refuses a model below 0.85 or more than one point under production.
- **In production.** Corrections give a running accuracy on reviewed tickets, per queue and
  per language, straight from SQL. The drift report exposes it.
- **Stored for comparison.** `training_runs` and each version's `metrics.json` keep every
  run's numbers, so any two versions can be compared later.

## A real run, end to end

The first cloud training run (2026-09-29, evidence/gpu-training-run.log) produced version
`v20260929-033811` on the RTX 4000 Ada: validation accuracy 89.79%, macro-F1 0.900,
calibration error 0.020. It cleared the gate (floor 88.93%, one point under the version in
production), moved `latest.json`, and the live API loaded it within two minutes without a
restart. `GET /v1/model` now reports the GPU, the host and the git commit it came from.

## Live traffic on the production endpoint

On 2026-09-29, 250 held-out test tickets were replayed through https://146-190-188-160.sslip.io
with half of them confirmed through `/v1/feedback` (evidence/live-replay-and-drift.txt):

| Measure | Value |
| --- | --- |
| Accuracy of the live API on the 250 tickets | 90.4% |
| Live accuracy on 131 human-reviewed tickets | 91.6% |
| Queue mix PSI against the reference | 0.058 |
| Mean confidence change | +0.012 |
| Unfamiliar share change | +6.2 points |
| Drift report status | **warn** |

The warning is real and explainable. About 25 of the 275 tickets in the window were
hand-written demonstration tickets with no relative in the history, which is exactly what the
unfamiliar-share signal is there to catch. Replayed tickets carry the `replay` key and never
become training data.

## What is not covered yet

A central log store (Loki or a managed service) and Grafana dashboards. Prometheus' own UI
and SQL on the audit log cover the proof of concept.
