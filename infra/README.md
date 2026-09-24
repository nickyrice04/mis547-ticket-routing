# Team droplet: what exists and how to use it

## What is running right now

| Thing | Value |
| --- | --- |
| Droplet name | `mis547-project-team3` |
| Droplet ID | `602239964` |
| Public IP | `174.138.91.2` |
| Size | `s-2vcpu-4gb` (2 vCPU, 4 GB RAM, 80 GB disk) |
| Cost | $24/month, which is $0.036 per hour |
| Region | `nyc3` (same region the labs use) |
| Image | `ubuntu-24-04-x64` |
| Tags | `project`, `team3`, `nicholasrice` |
| Firewall | `mis547-project-team3-fw`, inbound SSH only |

It also has a 4 GB swap file. Swap does not make the droplet fast, but it turns
"the model did not fit and the process was killed" into "the model loaded and
ran slowly," which is a useful difference when we are measuring where the
memory ceiling actually is.

## Why this size

The labs only ever use `s-1vcpu-1gb` at $6/month. That is enough for Jupyter and
for the TF-IDF baseline, but a DistilBERT process needs roughly 1.5 GB resident
before it answers anything, so a 1 GB droplet cannot host it. 4 GB is the
smallest size that holds the models we are actually comparing.

This is the department's donated account, so keep an eye on it. Destroy the
droplet when we are done, because **a powered-off droplet still bills**. Only
destroying it stops the charge.

```bash
# take a snapshot first if we want to rebuild it later
doctl compute droplet-action snapshot 602239964 --snapshot-name team3-final --wait
doctl compute droplet delete 602239964
```

## Adding Anshul and Alejandro

Each of them runs this once on their own machine. The private key never leaves
their laptop.

```bash
ssh-keygen -t ed25519 -C "netid@arizona.edu" -f ~/.ssh/digitalocean_key
cat ~/.ssh/digitalocean_key.pub
```

They send the output of that second command. A public key is safe to paste into
Slack or email. Then Nicky runs this once per person, replacing the name and the
key text:

```bash
ssh -i ~/.ssh/digitalocean_key root@174.138.91.2 \
  "useradd -m -s /bin/bash -G sudo,docker anshul && \
   mkdir -p /home/anshul/.ssh && \
   echo 'ssh-ed25519 AAAA...their key... netid@arizona.edu' > /home/anshul/.ssh/authorized_keys && \
   chown -R anshul:anshul /home/anshul/.ssh && chmod 700 /home/anshul/.ssh && \
   chmod 600 /home/anshul/.ssh/authorized_keys && \
   echo 'anshul ALL=(ALL) NOPASSWD:ALL' > /etc/sudoers.d/anshul"
```

After that they connect with their own account and their own key:

```bash
ssh -i ~/.ssh/digitalocean_key anshul@174.138.91.2
```

Separate accounts are better than sharing one key. We can see who did what, and
if someone's laptop is lost we remove one account instead of rotating a key that
all three of us use. The shared work goes in `/srv/ticket-routing`, which is
group writable, so all three accounts can edit the same checkout.

## Getting the code and models onto the droplet

From the laptop, push the project up without the virtual environment or the raw
dataset:

```bash
rsync -avz --exclude .venv --exclude data_tickets.parquet --exclude '__pycache__' \
  -e "ssh -i ~/.ssh/digitalocean_key" \
  "/Users/nickyrice/Documents/Cloud Computing/ticket-routing/" \
  root@174.138.91.2:/srv/ticket-routing/
```

## Running the API on the droplet

```bash
ssh -i ~/.ssh/digitalocean_key root@174.138.91.2
cd /srv/ticket-routing
docker build --build-arg MODEL_DIR=models/1_tfidf_logreg.joblib -t ticket-api:tfidf .
docker run -d --name ticket-api -p 127.0.0.1:8000:8000 --restart unless-stopped ticket-api:tfidf
```

Note the `127.0.0.1:` in the port mapping. The API listens only on the droplet's
own loopback address, so nothing is exposed to the internet. That is why the
firewall only needs port 22 open.

## Reaching the API from a laptop

Same SSH tunnel idea as the Jupyter server in Lab 1:

```bash
ssh -i ~/.ssh/digitalocean_key -L 8000:localhost:8000 root@174.138.91.2
```

Leave that terminal open, then from a second terminal on the laptop:

```bash
curl -s localhost:8000/health
curl -s -X POST localhost:8000/predict \
  -H 'content-type: application/json' \
  -d '{"subject":"double charge","body":"I was billed twice for my subscription this month and need the second charge refunded."}'
```

## Resizing when we test the bigger model

A CPU and RAM resize is reversible as long as we do not grow the disk. Growing
the disk is permanent, so never pass `--resize-disk`.

```bash
doctl compute droplet-action power-off 602239964 --wait
doctl compute droplet-action resize 602239964 --size s-4vcpu-8gb --wait   # $48/month
doctl compute droplet-action power-on 602239964 --wait
# and back down afterwards
doctl compute droplet-action resize 602239964 --size s-2vcpu-4gb --wait
```

## Still to decide with Professor Zara

The labs never authorize anything above the $6/month droplet, and they never
mention GPU droplets. Before we spend more on the department's account we should
ask her about two things.

1. Whether a $24/month droplet for the rest of the term is acceptable, and
   whether we may resize to $48/month for a few hours of benchmarking.
2. Whether a GPU droplet is available to the class if we want to fine-tune the
   language model in the cloud instead of on a laptop. The cheapest one is an
   RTX 4000 Ada at $0.76/hour, and we would need it for under an hour.
