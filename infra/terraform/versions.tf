# Terraform and provider versions, pinned so every teammate plans against the same code.
terraform {
  required_version = ">= 1.6"
  required_providers {
    digitalocean = {
      source  = "digitalocean/digitalocean"
      version = "~> 2.102"
    }
  }
}

# Credentials come from the environment, never from files in this repository:
#   DIGITALOCEAN_TOKEN                               the API token (infra/scripts/tf-env.sh reads it from doctl)
#   SPACES_ACCESS_KEY_ID / SPACES_SECRET_ACCESS_KEY  a Spaces key, needed to manage the bucket
provider "digitalocean" {}
