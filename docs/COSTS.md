# Costs and budget

DigitalOcean prices are the list prices the account reported through the API in September
2026. Other providers' prices are approximate public on-demand list prices (US East) and
must be checked against each vendor's pricing page, and cited, before they go in the report.

## What the proof of concept costs per month

| Resource | Size | Monthly |
| --- | --- | --- |
| Inference droplet | 2 vCPU, 4 GB, 80 GB disk, 4 TB transfer included | $24.00 |
| Managed PostgreSQL | 1 vCPU, 1 GB, 10 GB, single node | $15.00 |
| Spaces | 250 GB storage and 1 TB transfer included | $5.00 |
| Reserved IP | attached to a droplet | $0.00 |
| VPC, firewalls, monitoring alerts | | $0.00 |
| GPU training droplet | RTX 4000 Ada, $0.76 per hour, only during a run | see below |
| GitHub Actions, Container Registry | public repository | $0.00 |
| **Fixed monthly total** | | **$44.00** |

### Training bursts

The measured full retrain on the RTX 4000 Ada took 8 minutes, plus about half a minute to
build the Python environment and a few minutes of droplet boot. Call it 15 minutes billed.

| Scenario | GPU hours per month | GPU cost |
| --- | --- | --- |
| One measured retrain, droplet created and destroyed around it | 0.25 | $0.19 |
| Weekly refresh | 1 | $0.76 |
| Monthly refresh | 0.25 | $0.19 |
| An exploration month like this project's, 40 hours of experiments | 40 | $30.40 |
| The GPU droplet forgotten and left running | 730 | **$554.80** |

The last row is why the GPU droplet sits behind `training_enabled` in Terraform and
defaults to off. A GPU droplet bills while it exists, even powered off.

**Forecast for the proof of concept, weekly retraining: about $45 a month.**

## Optional, predictable latency

The shared-CPU inference droplet measured 20% CPU steal and 1 to 4 seconds per English ticket.
A CPU-Optimized droplet (`c-2`, 2 dedicated vCPU, 4 GB) costs $42 instead of $24, **+$18 a month**,
and is a one-line Terraform change (`inference_size = "c-2"`).

## Production with high availability

| Change | Monthly |
| --- | --- |
| Second inference droplet | +$24.00 |
| DigitalOcean load balancer (polls `/readyz`, TLS) | +$12.00 |
| PostgreSQL standby node for automatic failover, billed like a second node | +$15.00 to +$45.00 |
| **Production total, weekly retraining** | **about $97 to $127** |

The standby range depends on whether DigitalOcean allows a standby node on the smallest
database plan. If it requires the 2 GB plan ($30 a node), the database line becomes $60.
Check the current managed database pricing page before quoting a number.

## The same system elsewhere

Approximate, verify before citing.

| Option | What it would be | Monthly, HA production |
| --- | --- | --- |
| DigitalOcean IaaS (this project) | 2 droplets, load balancer, managed PostgreSQL with standby, Spaces, GPU by the hour | about $97 to $127 |
| AWS IaaS | 2 EC2 t3.medium (about $30 each), Application Load Balancer (about $18 plus usage), RDS PostgreSQL db.t4g.micro Multi-AZ (about $25), S3 (under $1), public IPv4 addresses (about $4 each), 3 hours of g6.xlarge with an L4 GPU (about $0.80 an hour) | about $115 to $130 |
| AWS SageMaker (managed MLOps) | Real-time endpoint on 2 ml.m5.large instances for HA (about $0.115 an hour each), training jobs on ml.g5.xlarge (about $1.41 an hour), RDS and S3 as above | about $190 to $210 |
| Google Vertex AI or Azure Machine Learning | Similar shape to SageMaker, a managed online endpoint on two small nodes plus GPU training jobs | about $150 to $250 |
| Databricks Model Serving | Serverless CPU serving billed per DBU, workspace and storage on the underlying cloud | about $150 to $300, highly usage dependent |
| SaaS helpdesk AI (Zendesk, Salesforce Service Cloud, Freshdesk add-ons) | Built-in triage, priced per agent seat, typically tens of dollars per agent per month | $500 to $1,500 for 20 agents, and the tickets leave the bank's network |

## Costs outside the cloud bill

- **Engineering time.** Building this took the team a few weeks. Running it is mainly
  reviewing drift issues, approving retraining, and patching. At a fully loaded $150,000 a
  year, a tenth of an engineer is about $1,250 a month, more than every cloud line combined.
- **Labelling.** The feedback loop depends on agents correcting tickets, which is time they
  spend anyway when a ticket lands in the wrong queue.
- **Compliance.** Audit log retention, access reviews, and a vendor review for any SaaS
  option, since complaint data would leave the bank.
- **Time to market.** SaaS triage is fastest to switch on. A managed platform like SageMaker
  trades money for less operations work. IaaS is cheapest per month and slowest to build,
  and that build is now done.

## Recommendation

Keep serving on DigitalOcean IaaS as built, add the second droplet and the load balancer
before real traffic, and keep the GPU strictly on demand. The reasons, in order:

1. **Data control.** Complaint text never leaves the bank's private network, which is the
   reason the business case gave for bringing this work in house.
2. **Cost.** About $100 to $130 a month for a highly available service, against more than that
   on a managed ML platform and five to fifteen times that for per-seat SaaS triage.
3. **The workload's shape.** Serving is small and constant, training is heavy and rare. Paying
   for a GPU by the hour and a small CPU droplet around the clock matches that exactly.

Revisit the choice if the team loses the operations skills to run it, or if the bank wants
many models, where a managed platform's registry, monitoring and deployment tooling start to
pay for themselves.
