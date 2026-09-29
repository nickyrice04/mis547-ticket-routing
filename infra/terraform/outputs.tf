output "api_url" {
  description = "The public HTTPS endpoint of the router"
  value       = "https://${local.api_host}"
}

output "api_ip" {
  value = digitalocean_reserved_ip.api.ip_address
}

output "inference_ip" {
  description = "The droplet's own address, for SSH (the reserved IP works too)"
  value       = digitalocean_droplet.inference.ipv4_address
}

output "inference_private_ip" {
  value = digitalocean_droplet.inference.ipv4_address_private
}

output "training_ip" {
  value = one(digitalocean_droplet.training[*].ipv4_address)
}

output "alert_email" {
  value = var.alert_email
}

output "region" {
  value = var.region
}

output "bucket" {
  value = digitalocean_spaces_bucket.artifacts.name
}

output "db_private_host" {
  value = digitalocean_database_cluster.tickets.private_host
}

output "db_port" {
  value = digitalocean_database_cluster.tickets.port
}

output "db_name" {
  value = digitalocean_database_db.routing.name
}

# Secrets below are read only by infra/scripts/bootstrap.sh, which writes them to root-only
# env files on the droplets. They live in the local terraform.tfstate, which is never committed.
output "db_admin_user" {
  value     = digitalocean_database_cluster.tickets.user
  sensitive = true
}

output "db_admin_password" {
  value     = digitalocean_database_cluster.tickets.password
  sensitive = true
}

output "db_api_password" {
  value     = digitalocean_database_user.api.password
  sensitive = true
}

output "db_trainer_password" {
  value     = digitalocean_database_user.trainer.password
  sensitive = true
}

output "spaces_inference_key" {
  value     = digitalocean_spaces_key.inference.access_key
  sensitive = true
}

output "spaces_inference_secret" {
  value     = digitalocean_spaces_key.inference.secret_key
  sensitive = true
}

output "spaces_training_key" {
  value     = digitalocean_spaces_key.training.access_key
  sensitive = true
}

output "spaces_training_secret" {
  value     = digitalocean_spaces_key.training.secret_key
  sensitive = true
}
