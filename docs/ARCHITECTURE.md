# Architecture

The end-to-end system that serves the ticket router on DigitalOcean. Every component
below exists in the DigitalOcean account (project **MIS547**, region **tor1**, tag
`team3`) and is defined in code in [infra/terraform](../infra/terraform).

![Architecture](architecture.png)

The diagram source is [architecture.mmd](architecture.mmd) (Mermaid), rendered to
`architecture.png` and `architecture.svg` for the report.

## What happens to one ticket

1. A client sends `POST /v1/route` with the ticket's subject and body over HTTPS to
   the API's reserved IP. The hostname is `<ip-with-dashes>.sslip.io`, which lets Caddy
   obtain a real Let's Encrypt certificate without buying a domain.
2. The cloud firewall admits only ports 80 and 443 from the internet. Caddy redirects
   HTTP to HTTPS, caps the request at 64 KB, adds security headers, blocks `/metrics`
   from outside, and forwards to the API container on the private Docker network.
3. The API checks the `X-API-Key` header and a per-key rate limit, validates the input,
   translates German sentences to English with the pinned opus-mt model, and scores the
   ticket against about 35,500 labelled tickets (final/model.py).
4. If the top queue's probability is at least 0.7 the ticket is routed automatically.
   Otherwise it goes to a person with the top three queues attached. At 0.7 that is 87%
   of tickets routed automatically at 98.3% accuracy.
5. The decision is written to the audit log in Managed PostgreSQL over the private
   network (TLS, least-privilege user). If that write fails, the answer is still
   returned but `auto_routed` is forced to false, so no automated decision is made
   without a record.
6. The response carries a `ticket_id`. A support agent who corrects the routing sends
   `POST /v1/feedback`, which is stored next to the prediction.

## How the model is trained and shipped

1. When a refresh is due, `terraform apply -var training_enabled=true` creates the GPU
   droplet. Cloud-init installs the team's logins and clones the repository.
2. `deploy/train.sh` builds the CUDA environment and runs the pipeline
   (src/mlops/train_pipeline.py): translate the German pool on the GPU, embed it,
   validate the evaluated recipe on the validation slice, pull corrected tickets from the
   audit log, fit the production model, and apply the quality gate.
3. The artifact (about 185 MB) and its metrics go to Spaces under `models/<version>/`.
   If the gate passes, `models/latest.json` is moved to the new version with its SHA-256.
   Every run, promoted or rejected, is recorded in the `training_runs` table.
4. The API polls `latest.json`, downloads the new artifact, checks the hash, and swaps
   the model in memory without a restart.
5. `terraform apply -var training_enabled=false` destroys the GPU droplet. It bills by the
   hour while it exists, even powered off.

Code ships separately from the model. A push to `main` runs CI (tests, Semgrep, Gitleaks,
pip-audit, Terraform validation), builds the image, scans it with Trivy, attaches an SBOM,
and publishes it to GitHub Container Registry. The inference droplet checks the registry
every five minutes and rolls to the new image, with a health check and automatic rollback.

## Components and why each one is there

| Component | What it is | Why it is needed | Sizing and justification |
| --- | --- | --- | --- |
| VPC `team3-ticket-routing-vpc` | Private network 10.140.0.0/24 in tor1 | Keeps database and droplet traffic off the public internet | Free. tor1 because it is the only region with the RTX 4000 Ada GPU, and every component must share the GPU's region to share its private network |
| Inference droplet | Ubuntu 24.04, 2 vCPU, 4 GB, Docker Compose (API, Caddy, Prometheus) | Always-on online inference, answers while a ticket is submitted | $24/month. The router holds about 2.3 GB resident (two pretrained models plus 185 MB of pools). 2 vCPU give about 35 ms for an English ticket and 0.5 to 1.5 s for a German one on CPU. A GPU here would sit idle, since serving is one ticket at a time |
| Reserved IP | Fixed public IPv4 attached to the inference droplet | The URL in the report survives a rebuilt droplet, which is also the disaster recovery path | Free while attached |
| GPU training droplet | 1x NVIDIA RTX 4000 Ada 20 GB, 8 vCPU, 32 GB, DigitalOcean AI/ML image | Burst compute for translation, embedding and fitting | $0.76/hour, only while a refresh runs. Translating 16,500 tickets is the heavy step (about 11 GPU minutes on the laptop). A CPU droplet would take hours, and a company still exploring models would run many such jobs |
| Managed PostgreSQL 16 | 1 vCPU, 1 GB, 10 GB disk, private network only | Audit log of every decision, human corrections, training runs. Structured records with one text field, which fits a relational database | $15/month. A year of traffic at 1,000 tickets a day is about 365,000 rows of roughly 2 KB, under 1 GB. Managed, so backups, patching and failover of the engine are DigitalOcean's job |
| Spaces bucket | S3-compatible object storage, private, versioned | Dataset and every model version, the source of truth the API loads from | $5/month for 250 GB. Each model version is 185 MB, so years of weekly versions fit |
| Cloud firewalls | Tag-based, one for inference, one for training | Only HTTPS in, SSH only from team addresses, egress limited to web, DNS, time and the database | Free |
| Monitoring | DigitalOcean agent alerts (CPU, memory, disk), uptime checks from two regions (down, latency, certificate expiry), Prometheus | Know when the service is down or degrading, from outside the droplet | Free |
| GitHub Actions and GHCR | CI with scanners, image registry | Every change is tested and scanned before it can reach production | Free for a public repository |

No serverless components are used. The one candidate was the daily drift check, and it
runs as a scheduled GitHub Actions workflow instead, which costs nothing and keeps the
API key out of another platform. DigitalOcean Functions would fit the same job.

## Where the heavy compute and bandwidth are

| Stage | Compute | Memory | Network | Frequency |
| --- | --- | --- | --- | --- |
| Translation of the German pool | GPU, about 11 minutes | 3 GB VRAM | 10 MB text in and out | Once per new batch of non-English tickets |
| Embedding the pools | GPU, about 1 minute | 2 GB VRAM | none | Each training run |
| Fitting the stacker | CPU, 2 to 3 minutes | 8 GB RAM | none | Each training run |
| Model download to the API | none | 185 MB | 185 MB from Spaces, same region | Each promoted version |
| Online inference | CPU, 35 ms (English), 0.5 to 1.5 s (German) | 2.3 GB | about 2 KB per request | Every ticket |
| Image pull | none | none | about 1.5 GB compressed from GHCR | Each code release |

Serving is light and constant. Training is heavy and rare. That split is the reason for
renting a GPU only for the burst and running inference on a small CPU droplet.

## What changed from the midterm diagram

The midterm diagram was a proposal. This table maps each part of it to what now exists.

| Midterm diagram | Now |
| --- | --- |
| Cloud firewall in front of the endpoint | Two tag-based firewalls, plus HTTPS through Caddy and a reserved IP |
| One inference droplet running a container | Same, with Compose running the API, Caddy and Prometheus, and pull-based updates |
| Managed PostgreSQL in a private subnet | Managed PostgreSQL reachable only from our two tagged droplets over the VPC, with two least-privilege users |
| Spaces with the dataset and every model version | Same, versioned, with bucket-scoped read-only and read-write keys and SHA-256 checks |
| Training droplet in its own private subnet, monthly retraining | GPU droplet that exists only during a training run, accepts no inbound traffic but SSH from team addresses, and publishes through a quality gate |
| TF-IDF plus logistic regression, DistilBERT explored | Retrieval plus a gradient-boosted stacker over English and translated German tickets |

DigitalOcean VPCs do not have separate public and private subnets. The isolation the
midterm drew as subnets is enforced by the firewalls and the database's trusted sources.

## High availability

See [HIGH_AVAILABILITY.md](HIGH_AVAILABILITY.md) for the answers to the midterm feedback.
In short, the proof of concept runs one inference droplet with Docker restart policies,
health and readiness checks, an outside uptime check and automatic image rollback. A
second droplet behind a DigitalOcean load balancer, polling `/readyz`, is the production
step. It is one Terraform change and $36 a month more.
