# Security and threat model

Method from Lecture 16: ask what the system was designed to do, what it can be made to
do, and what it can be made to do for evil, then walk every component through STRIDE and
map the controls to the NIST Cybersecurity Framework. Risks we chose not to fix are listed
at the end, in writing, as the lecture requires.

## What we are protecting

| Asset | Why it matters |
| --- | --- |
| Customer ticket text | Complaints name people, accounts and card fragments. In a bank this is regulated customer data |
| The routing decision and its audit trail | A misrouted complaint can miss a regulatory deadline, and the bank must be able to show why a ticket went where it went |
| The model artifact | It decides where tickets go, and it contains the translated German tickets verbatim plus TF-IDF vectors of every English ticket |
| Credentials | API keys, database passwords, Spaces keys, the DigitalOcean token, Terraform state |
| Availability | Tickets arrive all day, and a routing outage becomes a queue of unread complaints |

## Evidence that the controls are real

| Control | Evidence |
| --- | --- |
| No secrets in version control | Gitleaks over the full git history, no leaks (evidence/gitleaks-history.json), and on every push in CI |
| Static code scanning | Semgrep (Python, security audit and secrets rule packs) on the 41 shipped files, 0 findings (evidence/semgrep.json), and in CI |
| Container scanning | Trivy on the inference image, 0 fixable HIGH or CRITICAL vulnerabilities (evidence/trivy-image.txt), and in CI, which fails on fixable CRITICAL |
| Software bill of materials | CycloneDX SBOM of the image, 168 components (evidence/sbom-cyclonedx.json), and a CI artifact on every build |
| Least-privilege database user | `router_api` gets "permission denied" on DELETE and "must be owner" on DROP. Tested against PostgreSQL 16 |
| Hardened container | Runs as UID 10001, read-only root filesystem ("Read-only file system" on write), all Linux capabilities dropped, no-new-privileges |
| SSH exposure is real | The midterm droplet, with SSH open to the world, logged 16,547 rejected attempts from 399 addresses in a few days (evidence/old-droplet-ssh-probes.txt). The new firewalls allow SSH only from team addresses |
| PII masking in storage | A ticket containing a 16-digit card number is stored as `[number]` in the audit log |

## STRIDE by component

### Public API (Caddy and the router container)

| Threat | Example | Controls |
| --- | --- | --- |
| Spoofing | Someone calls the API as another client | Per-client API keys in the `X-API-Key` header, compared in constant time. Each key has a name that lands in the audit log, and one key can be revoked without touching the others |
| Tampering | Oversized or malformed input, SQL injection | Pydantic validation (subject up to 300 characters, body up to 8,000, unknown fields rejected), 64 KB body limit at Caddy, parameterised SQL through SQLAlchemy. There is no LLM prompt, so there is no prompt injection surface. The translator is a sequence-to-sequence model that cannot follow instructions |
| Repudiation | A client denies sending a ticket | Every request gets a request id, returned in `X-Request-ID` and stored with the key name, model version and time |
| Information disclosure | Ticket text sniffed in transit, internals in error messages | HTTPS only (HTTP redirects), HSTS. Errors return a code and message, never a stack trace. `/metrics` returns 404 from the internet |
| Denial of service | Flooding the endpoint | Per-key rate limit (60 per minute, 429 with Retry-After), request size limit, container memory cap so the host survives. Uptime alerts when it is down |
| Elevation of privilege | Escaping the container | Non-root user, read-only filesystem, no capabilities, no-new-privileges, no Docker socket inside |

### Managed PostgreSQL (audit log)

| Threat | Example | Controls |
| --- | --- | --- |
| Spoofing | Connecting as the application from elsewhere | Trusted sources: only droplets tagged `team3-inference` or `team3-training` can connect at all. Private network hostname. TLS required |
| Tampering | Rewriting the audit trail | The API's user can only SELECT and INSERT. It cannot UPDATE, DELETE or DROP. Proven by test |
| Repudiation | "The log was edited" | Same as above, plus DigitalOcean's managed backups |
| Information disclosure | Reading customer tickets | Two service users with narrow grants, no public access, digits masked before storage, encryption at rest managed by DigitalOcean |
| Denial of service | Exhausting connections | Connection pool of 5 plus 5 per API process |
| Elevation of privilege | Creating objects or roles | Service users have no CREATE rights. Only the admin, used once for the migration, can |

### Spaces (dataset and model versions)

| Threat | Example | Controls |
| --- | --- | --- |
| Tampering | Swapping in a malicious model file. Model files are Python pickles, which can run code when loaded | `latest.json` carries the SHA-256 of each artifact and the API refuses to load a file whose hash does not match. The API's key is read-only and scoped to this one bucket. Versioning keeps every earlier file |
| Information disclosure | The artifact holds the translated German tickets and word vectors of the English ones | Private bucket, no public ACLs, bucket-scoped keys, HTTPS only |
| Elevation of privilege | Using a leaked key elsewhere in the shared account | Keys are scoped to this bucket, read-only for the API, read-write only for the training job |

### GPU training droplet and pipeline

| Threat | Example | Controls |
| --- | --- | --- |
| Tampering (data poisoning) | Flooding the feedback endpoint with wrong corrections so the next model learns them | Feedback requires an API key and is stored with its key name. Replayed test traffic is excluded from training. The quality gate rejects a model whose validation accuracy drops below 0.85 or more than one point below production |
| Information disclosure | Reading the training data off the node | Node exists only during a run. No inbound traffic except SSH from team addresses. Credentials in a root-only env file |
| Spoofing | Pretending to be the trainer to publish a model | Only the training key can write to the bucket, and only the gate moves `latest.json` |

### Supply chain and CI/CD

| Threat | Example | Controls |
| --- | --- | --- |
| Tampering | A vulnerable or malicious dependency, a changed pretrained model | Exact version pins, pip-audit and Trivy in CI, SBOM on every build. Pretrained models pinned to commit hashes in code and baked into the image, so production never downloads models |
| Information disclosure | Secrets committed to the public repository | Gitleaks in CI, `.gitignore` for state, tfvars and env files. CI holds no cloud credentials |
| Elevation of privilege | CI pushing straight to servers | Pull-based deployment. The droplet pulls images, CI never connects to it, so there is no deploy key to steal. CI permissions are read-only except the image job's registry write |

### Administrative access

| Threat | Example | Controls |
| --- | --- | --- |
| Spoofing | Password guessing over SSH | Key-only SSH, one login per teammate, no root login, no passwords, SSH allowed only from known addresses at the cloud firewall |
| Repudiation | Who changed the server | Named user accounts instead of a shared key |
| Information disclosure | Terraform state contains database passwords | State stays on one laptop and is never committed. Remote encrypted state is the production step |

## ML-specific threats

| Threat | Why it applies here | Controls and residual risk |
| --- | --- | --- |
| Membership inference | The response's `familiarity` score tells a caller how close the nearest labelled ticket is, which hints whether a similar ticket exists in the history | Only authenticated clients see it. In production it would be internal-only or rounded. Accepted for the proof of concept, the data is synthetic |
| Model extraction | Many queries could approximate the router | Rate limits and per-key audit. Low value to an attacker, the queues are internal |
| Evasion (adversarial tickets) | A customer words a ticket to jump to a faster queue | Low-confidence tickets go to a person. Drift monitoring flags unusual traffic |
| Bias | German tickets pass through a translator, small queues have little data | Per-queue recall is 87 to 96%, macro-F1 0.917. Language mix is recorded per prediction and reported by `/v1/drift`, so per-language accuracy can be measured from corrections |
| Unintended use (Lecture 16) | Using routing data to profile customers | The service stores no customer identifiers beyond the text, and digits are masked |

## Shared responsibility

| Layer | DigitalOcean | Team 3 |
| --- | --- | --- |
| Droplets (IaaS) | Hardware, hypervisor, network isolation between customers | OS patching (unattended upgrades), SSH hardening, firewall rules, Docker, the application |
| Managed PostgreSQL (PaaS) | Engine patching, backups, encryption at rest, failover of the engine | Users and grants, trusted sources, TLS on connections, what data we store |
| Spaces | Durability, availability, the storage infrastructure | Bucket ACL, key scope, what we upload, integrity checks |

## NIST CSF 2.0 mapping

| Function | What we do |
| --- | --- |
| Govern | Risks accepted in writing below, one owner per component, least privilege as the default |
| Identify | Asset list above, SBOM for the image, Terraform as the inventory of cloud resources |
| Protect | TLS everywhere, firewalls, scoped keys, hardened containers, masked storage, pinned dependencies |
| Detect | Uptime checks from two regions, CPU, memory and disk alerts, Prometheus metrics, audit log, drift report, CI scanners |
| Respond | Revoke an API key by editing one env file, roll back an image automatically on failed health, reject a bad model at the gate |
| Recover | Reserved IP plus Terraform rebuilds a droplet at the same address. Spaces versioning restores any model. Managed database backups |

## Risks accepted for the proof of concept

Each of these is a conscious choice for a class project on synthetic data. Each would be
revisited before real customer data.

1. **One inference droplet.** A droplet failure is an outage until it is rebuilt. The
   production fix is a second droplet behind a load balancer, $36 a month more.
2. **SSH from team home addresses** rather than a VPN or bastion. Addresses change, so the
   allow list is edited in Terraform when a teammate moves.
3. **Terraform state on one laptop.** It holds database passwords. Production would use
   encrypted remote state with locking.
4. **Public repository and public image.** Nothing secret is in either, and it lets the
   instructor review everything.
5. **The familiarity score is visible to API clients**, as described under membership
   inference.
6. **Interactive API docs are public at /docs.** They describe the API and cannot call it
   without a key.
