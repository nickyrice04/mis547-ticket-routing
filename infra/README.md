# Infrastructure runbook

Everything in DigitalOcean is defined in [terraform/](terraform). Nothing is created by
clicking in the console, so the account always matches this code (Lecture 12). The design
is explained in [docs/ARCHITECTURE.md](../docs/ARCHITECTURE.md).

| Resource | Name | Notes |
| --- | --- | --- |
| VPC | `team3-ticket-routing-vpc` | tor1, 10.140.0.0/24 |
| Inference droplet | `team3-ticket-routing-inference` | 2 vCPU, 4 GB, always on, behind a reserved IP |
| GPU training droplet | `team3-ticket-routing-training-gpu` | RTX 4000 Ada, exists only while `training_enabled = true` |
| Managed PostgreSQL | `team3-ticket-routing-db` | database `routing`, users `router_api` and `router_trainer` |
| Spaces bucket | `team3-ticket-routing-mis547` | private, versioned |
| Firewalls | `team3-ticket-routing-inference-fw`, `-training-fw` | SSH only from `ssh_allowed_cidrs` |
| Alerts | CPU, memory, disk, uptime (down, latency, certificate) | email to `alert_email` |

All of it is tagged `team3` and filed under the **MIS547** project.

## One-time setup on a laptop

```bash
brew install doctl terraform jq
doctl auth init --context mis547            # paste the class API token when asked
```

Terraform also needs a Spaces key to manage the bucket. It lives outside the repository in
`~/.config/team3/spaces-terraform.env` (two lines, `SPACES_ACCESS_KEY_ID=` and
`SPACES_SECRET_ACCESS_KEY=`, file mode 600). Ask Nicky for it, or create your own with
`doctl spaces keys create`.

Copy `terraform/terraform.tfvars.example` to `terraform/terraform.tfvars` (git ignores it)
and fill in the team's public keys, your address for SSH, and the alert email.

## Deploying

```bash
cd infra/terraform
source ../scripts/tf-env.sh          # exports the DigitalOcean token and the Spaces key
terraform init
terraform plan -out tfplan           # read it
terraform apply tfplan
cd ../..
infra/scripts/bootstrap.sh           # secrets onto the droplets, database schema, dataset upload, start the API
```

`bootstrap.sh` generates the API keys once and keeps them in `~/.config/team3/api-keys.env`
on the laptop that ran it. `GRADER_KEY` is the one for the instructor. Keys never go in git.

Then train and publish the first model version on the GPU droplet:

```bash
ssh nicky@$(terraform -chdir=infra/terraform output -raw training_ip) 'sudo /srv/ticket-routing/deploy/train.sh'
```

The API picks the new version up within two minutes (it polls Spaces). Check it:

```bash
curl -s $(terraform -chdir=infra/terraform output -raw api_url)/readyz
```

Turn the uptime checks on once the endpoint answers with a certificate:

```bash
terraform -chdir=infra/terraform apply -var uptime_enabled=true
```

## The GPU switch

The GPU droplet costs $0.76 an hour while it exists, powered on or off. Turn it off as soon
as a run finishes and back on for the next one:

```bash
terraform -chdir=infra/terraform apply -var training_enabled=false    # destroy it
terraform -chdir=infra/terraform apply -var training_enabled=true     # recreate it, then re-run bootstrap.sh for trainer.env
```

## Access for teammates

Each teammate has their own login on both droplets (`nicky`, `anshul`, `alejandro`), with
their own key. Root login and passwords are disabled. SSH is dropped at the cloud firewall
unless it comes from an address in `ssh_allowed_cidrs`, so a teammate first sends their
address (`curl -4 ifconfig.me`), it is added as `x.x.x.x/32` in `terraform.tfvars`, and
someone runs `terraform apply`. Then:

```bash
ssh -i ~/.ssh/digitalocean_key anshul@<inference-ip>
```

The service lives in `/srv/ticket-routing` (a git checkout), its secrets in
`/etc/ticket-routing/*.env` (root only).

## Operating the service

```bash
sudo systemctl status ticket-router                 # the Compose stack
cd /srv/ticket-routing/deploy && sudo docker compose ps
sudo docker compose logs -f api                     # JSON logs, one line per decision
sudo systemctl list-timers ticket-router-update     # the five-minute updater and watchdog
sudo journalctl -u ticket-router-update             # what the updater did
```

Prometheus from a laptop: `ssh -L 9090:localhost:9090 nicky@<inference-ip>`, then
http://localhost:9090.

**Roll back the code.** Pin an older image in `/etc/ticket-routing/compose.env`
(`API_IMAGE=ghcr.io/nickyrice04/mis547-ticket-routing:<commit>`) and
`sudo systemctl restart ticket-router`.

**Roll back the model.** Point `models/latest.json` in Spaces at the previous version and
hash. The API follows within two minutes.

**Revoke an API key.** Remove it from `API_KEYS` in `/etc/ticket-routing/api.env` and
`sudo docker compose up -d api`.

**Rebuild a dead droplet.** `terraform apply -replace=digitalocean_droplet.inference`, then
`infra/scripts/bootstrap.sh`. The reserved IP moves to the new droplet, so the URL stays the same.

## Tearing it down after grading

```bash
terraform -chdir=infra/terraform destroy
```

The bucket has versioning on, so Terraform refuses to delete it while it holds objects.
Empty it in the console first, or keep it as the archive of every model version.
