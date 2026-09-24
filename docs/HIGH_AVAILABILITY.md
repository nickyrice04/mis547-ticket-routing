# Answering the high availability question

Professor Zara asked four questions about the midterm architecture. One
container on one droplet does not answer any of them, and the final report
should not pretend otherwise. Here is what is true today and what has to change.

## How does one container on one droplet support high availability?

It does not. A single droplet is a single point of failure. If it reboots, the
service is down. If the process dies, the service is down until something
restarts it. The midterm design was honest about being version one, but the
final report needs the real answer, which has three parts.

1. **More than one droplet.** Two droplets in different availability zones,
   each running the same container image.
2. **A load balancer in front of them.** A DigitalOcean regional load balancer
   costs $12 a month, polls a health check on each droplet, and stops sending
   traffic to one that fails. This is the piece that turns two droplets into one
   service.
3. **A health check the load balancer can poll.** Already built. The API serves
   `GET /health`, which returns the model version and whether the model actually
   finished loading, not just whether the process is up.

## How will we know if the container has crashed?

Three layers, and we have the first two running now.

- **The container restarts itself.** It runs with `--restart unless-stopped`, so
  Docker restarts it after a crash or a droplet reboot.
- **The container reports its own health.** The `HEALTHCHECK` line in the
  Dockerfile polls `/health` every 30 seconds. `docker ps` then shows the
  container as healthy or unhealthy rather than just running.
- **Something off the droplet has to watch too.** A crashed droplet cannot tell
  us it crashed. DigitalOcean monitoring can alert on CPU, memory and a droplet
  going unreachable, and the load balancer's health check is what actually pulls
  a bad droplet out of rotation.

The distinction that matters here is that a process can be running and still be
useless. A container whose model failed to load still answers on port 8000, so
the health check reports the model version and load state instead of a bare
"ok".

## What if the droplet is not responding?

With one droplet, someone has to notice and rebuild it. That is the current
state and it is not acceptable for a service with a 15 day regulatory clock.
With a load balancer and two droplets, the failure is automatic: the health
check fails, the load balancer stops routing there, and the other droplet takes
the traffic. Rebuilding the dead one is then routine rather than urgent.

Because the droplet holds no state, rebuilding it is quick. The model lives in
Spaces and the predictions live in the managed database, so a replacement
droplet pulls the image, downloads the model, and joins the pool.

## How will we handle updates?

Right now a deploy is a pull and a restart, which means a few seconds of
downtime. With two droplets behind a load balancer it becomes a rolling update:
take one out of rotation, update it, put it back, then do the other. No downtime
and an easy rollback, because the previous image is still in the registry.

The model and the code version separately. The image tag covers the code, and
`MODEL_VERSION` covers the weights, which is what the audit log records with
every prediction. That way a bad model can be rolled back without redeploying
the code.

## What this costs

| Setup | Monthly |
| --- | --- |
| Today: one 4 GB droplet | $24 |
| Two 4 GB droplets plus a load balancer | $60 |
| Managed PostgreSQL, smallest plan | $15 |
| Spaces for models and data | $5 |

High availability roughly doubles the compute bill. That is the honest tradeoff
to put in the report, next to the cost of missing a regulatory deadline because
one droplet was down.
