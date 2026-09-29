variable "name" {
  description = "Prefix for every resource, so the team's resources are easy to find in the shared account"
  type        = string
  default     = "team3-ticket-routing"
}

variable "region" {
  description = "tor1 is the only region that offers the RTX 4000 Ada GPU droplet, and everything else sits next to it on the same private network"
  type        = string
  default     = "tor1"
}

variable "vpc_cidr" {
  description = "Private address range of the project VPC, chosen not to overlap the account's other VPCs"
  type        = string
  default     = "10.140.0.0/24"
}

variable "inference_size" {
  description = "Always-on API droplet. 2 vCPU and 4 GB holds the router (about 2.3 GB resident) plus Caddy and Prometheus"
  type        = string
  default     = "s-2vcpu-4gb"
}

variable "training_enabled" {
  description = "The GPU training droplet exists only while this is true. It bills by the hour even when powered off, so the default is off"
  type        = bool
  default     = false
}

variable "training_size" {
  description = "Cheapest GPU droplet, one NVIDIA RTX 4000 Ada with 20 GB of VRAM, 8 vCPU, 32 GB RAM, $0.76 an hour"
  type        = string
  default     = "gpu-4000adax1-20gb"
}

variable "training_image" {
  description = "DigitalOcean's AI/ML ready image, Ubuntu with the NVIDIA driver and CUDA preinstalled"
  type        = string
  default     = "gpu-h100x1-base"
}

variable "db_size" {
  description = "Smallest managed PostgreSQL plan, 1 vCPU, 1 GB RAM, 10 GB disk"
  type        = string
  default     = "db-s-1vcpu-1gb"
}

variable "bucket_name" {
  description = "Spaces bucket for the dataset and every model version. Bucket names are global, so it carries the course name"
  type        = string
  default     = "team3-ticket-routing-mis547"
}

variable "repo_url" {
  description = "Public repository the droplets clone on first boot"
  type        = string
  default     = "https://github.com/nickyrice04/mis547-ticket-routing.git"
}

variable "team_ssh_keys" {
  description = "One login per teammate, name => public key. Public keys only, private keys never leave a laptop"
  type        = map(string)
}

variable "ssh_allowed_cidrs" {
  description = "Addresses allowed to reach SSH. Everyone else is dropped at the cloud firewall"
  type        = list(string)
}

variable "alert_email" {
  description = "Where monitoring and uptime alerts go"
  type        = string
}

variable "do_ssh_key_name" {
  description = "An SSH key already registered in the DigitalOcean account, attached at creation so the droplet can boot"
  type        = string
  default     = "Nicky_Rice_ssh_key"
}

variable "do_project" {
  description = "DigitalOcean project the resources are filed under, so the instructor finds them"
  type        = string
  default     = "MIS547"
}

variable "uptime_enabled" {
  description = "External uptime check and alerts on the public endpoint. Turned on once the endpoint has its certificate"
  type        = bool
  default     = true
}
