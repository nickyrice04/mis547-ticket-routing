# The whole architecture of the ticket router on DigitalOcean.
#
#   internet ── HTTPS ──> reserved IP ──> inference droplet (Caddy -> API container, Prometheus)
#                                               │ private network (VPC)
#                                               ├──> Managed PostgreSQL   audit log, corrections, training runs
#                                               └──> Spaces (S3 API)      dataset and model versions
#   GPU training droplet (on only for a training burst) ── private network ──> PostgreSQL, Spaces
#
# Everything is in one region (tor1) and one VPC. DigitalOcean VPCs have no public and
# private subnets the way AWS does, so isolation is enforced by cloud firewalls: the
# inference droplet accepts only HTTPS from the internet, the training droplet accepts
# nothing but SSH from the team, and the database accepts connections only from those two
# droplets over the private network.

data "digitalocean_ssh_key" "nicky" {
  name = var.do_ssh_key_name
}

data "digitalocean_project" "course" {
  name = var.do_project
}

locals {
  common_tags = ["team3", "project"]
  # A hostname that resolves to the reserved IP without buying a domain, so Caddy can
  # get a real Let's Encrypt certificate for it. 1.2.3.4 -> 1-2-3-4.sslip.io
  api_host   = "${replace(digitalocean_reserved_ip.api.ip_address, ".", "-")}.sslip.io"
  everywhere = ["0.0.0.0/0", "::/0"]
}

# ------------------------------------------------------------------------------------------ network
resource "digitalocean_vpc" "main" {
  name        = "${var.name}-vpc"
  region      = var.region
  ip_range    = var.vpc_cidr
  description = "Team 3 ticket router, private network for the API, the training node and the database"
}

resource "digitalocean_tag" "inference" {
  name = "team3-inference"
}

resource "digitalocean_tag" "training" {
  name = "team3-training"
}

resource "digitalocean_tag" "db" {
  name = "team3-db"
}

# A fixed public address for the API. If the droplet is rebuilt, the address and the
# HTTPS hostname stay the same, so the URL in the report never changes.
resource "digitalocean_reserved_ip" "api" {
  region = var.region
}

resource "digitalocean_reserved_ip_assignment" "api" {
  ip_address = digitalocean_reserved_ip.api.ip_address
  droplet_id = digitalocean_droplet.inference.id
}

# ------------------------------------------------------------------------------------------ compute
resource "digitalocean_droplet" "inference" {
  name              = "${var.name}-inference"
  region            = var.region
  size              = var.inference_size
  image             = "ubuntu-24-04-x64"
  vpc_uuid          = digitalocean_vpc.main.id
  ssh_keys          = [data.digitalocean_ssh_key.nicky.id]
  monitoring        = true # DigitalOcean's metrics agent: CPU, memory, disk, bandwidth
  droplet_agent     = true
  graceful_shutdown = true
  tags              = concat(local.common_tags, [digitalocean_tag.inference.name])
  user_data = templatefile("${path.module}/cloud-init/inference.yaml.tftpl", {
    team     = var.team_ssh_keys
    repo_url = var.repo_url
  })
  # Teammates are added over SSH afterwards, so a key change never rebuilds the server.
  lifecycle {
    ignore_changes = [user_data]
  }
}

resource "digitalocean_droplet" "training" {
  count             = var.training_enabled ? 1 : 0
  name              = "${var.name}-training-gpu"
  region            = var.region
  size              = var.training_size
  image             = var.training_image
  vpc_uuid          = digitalocean_vpc.main.id
  ssh_keys          = [data.digitalocean_ssh_key.nicky.id]
  monitoring        = true
  droplet_agent     = true
  graceful_shutdown = true
  tags              = concat(local.common_tags, [digitalocean_tag.training.name])
  user_data = templatefile("${path.module}/cloud-init/training.yaml.tftpl", {
    team     = var.team_ssh_keys
    repo_url = var.repo_url
  })
  lifecycle {
    ignore_changes = [user_data]
  }
}

# ------------------------------------------------------------------------------------------ database
resource "digitalocean_database_cluster" "tickets" {
  name                 = "${var.name}-db"
  engine               = "pg"
  version              = "16"
  size                 = var.db_size
  region               = var.region
  node_count           = 1
  private_network_uuid = digitalocean_vpc.main.id
  tags                 = concat(local.common_tags, [digitalocean_tag.db.name])
}

resource "digitalocean_database_db" "routing" {
  cluster_id = digitalocean_database_cluster.tickets.id
  name       = "routing"
}

# Two service accounts with narrow grants (see src/mlops/db.py GRANTS). The API can
# insert and read predictions and feedback. The trainer can read them and write runs.
resource "digitalocean_database_user" "api" {
  cluster_id = digitalocean_database_cluster.tickets.id
  name       = "router_api"
  # The API returns an empty settings block after creation, and the provider then tries to
  # "remove" it with an update the API rejects. Nothing here is managed through settings.
  lifecycle {
    ignore_changes = [settings]
  }
}

resource "digitalocean_database_user" "trainer" {
  cluster_id = digitalocean_database_cluster.tickets.id
  name       = "router_trainer"
  # The API returns an empty settings block after creation, and the provider then tries to
  # "remove" it with an update the API rejects. Nothing here is managed through settings.
  lifecycle {
    ignore_changes = [settings]
  }
}

# Trusted sources: only droplets carrying these tags can open a connection at all.
resource "digitalocean_database_firewall" "tickets" {
  cluster_id = digitalocean_database_cluster.tickets.id
  rule {
    type  = "tag"
    value = digitalocean_tag.inference.name
  }
  rule {
    type  = "tag"
    value = digitalocean_tag.training.name
  }
}

# ------------------------------------------------------------------------------------------ storage
resource "digitalocean_spaces_bucket" "artifacts" {
  name   = var.bucket_name
  region = var.region
  acl    = "private"
  versioning {
    enabled = true # an overwritten or deleted model can be recovered
  }
}

# Least privilege keys: the API can only read the bucket, the trainer can read and write it.
resource "digitalocean_spaces_key" "inference" {
  name = "${var.name}-inference-read"
  grant {
    bucket     = digitalocean_spaces_bucket.artifacts.name
    permission = "read"
  }
}

resource "digitalocean_spaces_key" "training" {
  name = "${var.name}-training-readwrite"
  grant {
    bucket     = digitalocean_spaces_bucket.artifacts.name
    permission = "readwrite"
  }
}

# ------------------------------------------------------------------------------------------ firewalls
resource "digitalocean_firewall" "inference" {
  name = "${var.name}-inference-fw"
  tags = [digitalocean_tag.inference.name]

  inbound_rule {
    protocol         = "tcp"
    port_range       = "22"
    source_addresses = var.ssh_allowed_cidrs
  }
  inbound_rule { # HTTP only answers the certificate challenge and redirects to HTTPS
    protocol         = "tcp"
    port_range       = "80"
    source_addresses = local.everywhere
  }
  inbound_rule {
    protocol         = "tcp"
    port_range       = "443"
    source_addresses = local.everywhere
  }

  # Egress is limited to what the service needs: web (image pulls, Spaces, certificates,
  # updates), DNS and time, and PostgreSQL on the private network only.
  outbound_rule {
    protocol              = "tcp"
    port_range            = "80"
    destination_addresses = local.everywhere
  }
  outbound_rule {
    protocol              = "tcp"
    port_range            = "443"
    destination_addresses = local.everywhere
  }
  outbound_rule {
    protocol              = "tcp"
    port_range            = "25060"
    destination_addresses = [var.vpc_cidr]
  }
  outbound_rule {
    protocol              = "udp"
    port_range            = "53"
    destination_addresses = local.everywhere
  }
  outbound_rule {
    protocol              = "udp"
    port_range            = "123"
    destination_addresses = local.everywhere
  }
}

resource "digitalocean_firewall" "training" {
  name = "${var.name}-training-fw"
  tags = [digitalocean_tag.training.name]

  inbound_rule { # nothing but SSH from the team, the node serves no traffic
    protocol         = "tcp"
    port_range       = "22"
    source_addresses = var.ssh_allowed_cidrs
  }

  outbound_rule {
    protocol              = "tcp"
    port_range            = "80"
    destination_addresses = local.everywhere
  }
  outbound_rule {
    protocol              = "tcp"
    port_range            = "443"
    destination_addresses = local.everywhere
  }
  outbound_rule {
    protocol              = "tcp"
    port_range            = "25060"
    destination_addresses = [var.vpc_cidr]
  }
  outbound_rule {
    protocol              = "udp"
    port_range            = "53"
    destination_addresses = local.everywhere
  }
  outbound_rule {
    protocol              = "udp"
    port_range            = "123"
    destination_addresses = local.everywhere
  }
}

# ------------------------------------------------------------------------------------------ monitoring
# Resource alerts from DigitalOcean's monitoring agent, for every team3 droplet.
resource "digitalocean_monitor_alert" "cpu" {
  alerts {
    email = [var.alert_email]
  }
  window      = "5m"
  type        = "v1/insights/droplet/cpu"
  compare     = "GreaterThan"
  value       = 85
  enabled     = true
  tags        = [digitalocean_tag.inference.name]
  description = "Team 3 inference CPU above 85 percent for 5 minutes"
}

resource "digitalocean_monitor_alert" "memory" {
  alerts {
    email = [var.alert_email]
  }
  window      = "5m"
  type        = "v1/insights/droplet/memory_utilization_percent"
  compare     = "GreaterThan"
  value       = 90
  enabled     = true
  tags        = [digitalocean_tag.inference.name]
  description = "Team 3 inference memory above 90 percent for 5 minutes"
}

resource "digitalocean_monitor_alert" "disk" {
  alerts {
    email = [var.alert_email]
  }
  window      = "5m"
  type        = "v1/insights/droplet/disk_utilization_percent"
  compare     = "GreaterThan"
  value       = 85
  enabled     = true
  tags        = [digitalocean_tag.inference.name, digitalocean_tag.training.name]
  description = "Team 3 disk above 85 percent"
}

# An outside check of the public endpoint, from two regions. This is how we know the
# container or the droplet has died even when the droplet itself cannot tell us.
resource "digitalocean_uptime_check" "api" {
  count   = var.uptime_enabled ? 1 : 0
  name    = "${var.name}-api"
  target  = "https://${local.api_host}/healthz"
  type    = "https"
  regions = ["us_east", "us_west"]
  enabled = true
}

resource "digitalocean_uptime_alert" "down" {
  count    = var.uptime_enabled ? 1 : 0
  check_id = digitalocean_uptime_check.api[0].id
  name     = "Team 3 ticket router is down"
  type     = "down"
  period   = "2m"
  notifications {
    email = [var.alert_email]
  }
}

resource "digitalocean_uptime_alert" "latency" {
  count      = var.uptime_enabled ? 1 : 0
  check_id   = digitalocean_uptime_check.api[0].id
  name       = "Team 3 ticket router is slow"
  type       = "latency"
  threshold  = 2000
  comparison = "greater_than"
  period     = "5m"
  notifications {
    email = [var.alert_email]
  }
}

resource "digitalocean_uptime_alert" "certificate" {
  count      = var.uptime_enabled ? 1 : 0
  check_id   = digitalocean_uptime_check.api[0].id
  name       = "Team 3 ticket router certificate expires soon"
  type       = "ssl_expiry"
  threshold  = 14
  comparison = "less_than"
  period     = "2m"
  notifications {
    email = [var.alert_email]
  }
}

# ------------------------------------------------------------------------------------------ project
# Filed under the course project, so the instructor sees exactly these resources.
resource "digitalocean_project_resources" "course" {
  project = data.digitalocean_project.course.id
  resources = concat(
    [
      digitalocean_droplet.inference.urn,
      digitalocean_database_cluster.tickets.urn,
      digitalocean_spaces_bucket.artifacts.urn,
      digitalocean_reserved_ip.api.urn,
    ],
    digitalocean_droplet.training[*].urn,
  )
}
