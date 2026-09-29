# High availability, answering the midterm feedback

Professor Zara asked four questions about the midterm architecture. Here is what the
final system does, and what the production step adds.

## How does one container on one droplet support high availability?

It does not, and the report should say so. One droplet is a single point of failure. What
the proof of concept does is shrink how long a failure lasts and make sure someone knows:

- Docker restarts the API container if it crashes or exits (`restart: unless-stopped`).
  Docker alone does not restart a container that is running but unhealthy, so a watchdog in
  the five-minute timer (`deploy/update.sh`) restarts the API when its health check is
  failing. The systemd unit brings the whole stack back after a reboot.
- Readiness is separate from liveness. `/healthz` says the process is alive. `/readyz` says
  the model is loaded, warmed up and the audit database answers. A container whose model
  failed to load is alive but not ready.
- The API degrades rather than failing. If the database is down it still answers, but sends
  every ticket to a person, because no automated decision may be made without an audit record.
- If a new model fails to load, the service keeps serving the one it has.

**The production step** is a second inference droplet and a DigitalOcean load balancer
($12 a month) that polls `/readyz` and stops routing to a node that fails. Both droplets are
stateless (the model lives in Spaces, the decisions in PostgreSQL), so either can serve any
request. In Terraform that is a `count` on the droplet and one load balancer resource. With a
managed PostgreSQL standby node the database fails over on its own too.

## How will you know if your container has crashed?

Three layers, all running:

1. **Inside the container.** Docker polls `/healthz` every 30 seconds and marks the container
   unhealthy after three failures. The watchdog restarts an unhealthy container within five
   minutes, and Docker restarts one that crashed immediately.
2. **On the droplet.** DigitalOcean's agent alerts on CPU above 85%, memory above 90% and
   disk above 85%, by email.
3. **Outside the droplet.** DigitalOcean uptime checks call `https://<host>/healthz` from two
   regions and email when it is down for two minutes, slower than two seconds, or when the
   TLS certificate is within 14 days of expiry. A dead droplet cannot report itself, so this
   is the check that matters most.

## What will you do if the droplet is not responding?

With the proof of concept, rebuild it: `terraform apply -replace=digitalocean_droplet.inference`
creates a fresh droplet from the same cloud-init, and the reserved IP is reassigned to it,
so the URL and the HTTPS hostname do not change. `infra/scripts/bootstrap.sh` puts the
secrets back. The model comes from Spaces and the history is in PostgreSQL, so nothing is lost.
That takes about ten minutes.

With the production step, nothing needs doing in the moment. The load balancer drops the dead
node and the other droplet takes the traffic.

## How will you handle updates to the system and application?

Code, configuration and model versions move separately:

- **Code.** A push to `main` runs CI, which tests, scans and publishes a new image. The droplet
  checks the registry every five minutes, starts the new image, waits for it to report
  healthy, and puts the previous image back if it does not within five minutes. With two
  droplets, the load balancer turns this into a rolling update with no downtime.
- **Configuration.** Compose, Caddy and update-script changes arrive with the same `git pull`.
- **Model.** The training job publishes a new version only if it passes the quality gate. The
  API notices the new `latest.json`, verifies the file's hash, and swaps the model in memory
  without a restart. Rolling back is pointing `latest.json` at the previous version.
- **Infrastructure.** Terraform plan, review, apply. Every change is in git.
