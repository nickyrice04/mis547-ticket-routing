# Security

Our full threat model is in section 5 of the final report. This page is the short version, with
links to the proof.

## What we protect

Customer ticket text, the audit log of routing decisions, the model file (it holds translated
tickets word for word), our credentials, and uptime, since an outage turns into a pile of unread
complaints.

## API keys

Every call except the health checks needs a key in the `X-API-Key` header. The server reads its
keys as `name:key` pairs from a root-only file, `/etc/ticket-routing/api.env`, which is never in
git. The template is [deploy/api.env.example](../deploy/api.env.example). There are three keys,
`team`, `grader` and `replay`, and the grader's key is only given in our report.

The API compares keys in constant time, logs the key's name with every decision and never the
key itself, and allows 60 requests a minute per key. To revoke one we remove it from that file
and run `sudo docker compose up -d api`.

## Threats and controls

Below is the STRIDE summary, one main threat per part of the system. The report's Table 6 has
the full version.

| Part | Threat | Control |
| --- | --- | --- |
| Public API | Floods, junk input, SQL injection | API keys, 60 requests a minute per key, subject up to 300 characters and body up to 8,000, a 64 KB request cap, parameterized SQL, JSON errors with no stack traces |
| Traffic | Ticket text read in transit | HTTPS only with HSTS, through Caddy and a Let's Encrypt certificate |
| Container | Escaping to the server | Non-root user, read-only filesystem, all Linux capabilities dropped, no-new-privileges |
| PostgreSQL | Rewriting the audit log | Only our tagged droplets can connect, over the private network with TLS. The API's user can only read and insert |
| Spaces | Swapping in a bad model file (pickles can run code) | Private bucket, a read-only key for the API, a SHA-256 check before loading, versioning for rollback |
| Training | Poisoned corrections | Keyed and logged feedback, a quality gate at 85% validation accuracy, drift monitoring |
| CI/CD | A stolen deploy key | Pull-based deploys. GitHub holds no cloud credentials and never connects to the server |
| SSH | Password guessing | Key-only logins, one account per teammate, no root login, firewall open to team addresses only |

## Scans

Every push runs these in [CI](../.github/workflows/ci.yml). The saved reports are in
[evidence/](../evidence/README.md).

| Scanner | What it checks | Latest result |
| --- | --- | --- |
| Semgrep | The 44 code files that ship | 0 findings |
| Gitleaks | The whole git history | No secrets |
| pip-audit | The pinned runtime packages | No known vulnerabilities in the last CI run |
| Trivy | The inference image | 0 vulnerabilities, plus a CycloneDX SBOM of 168 components |

## Secrets

There are no secrets in this repository. API keys, database passwords and storage keys live in
root-only files on the droplets and on one teammate's laptop. `.gitignore` keeps env files and
Terraform state out of git, and Gitleaks checks every push.

## Risks we accepted

This is a class project on synthetic data, so we accepted a few things we would fix before real
customer data.

- One inference droplet. The fix is a second droplet behind a load balancer, about $36 a month more.
- SSH from teammates' home addresses instead of a VPN or bastion host.
- Terraform state, which holds database passwords, on one laptop instead of remote encrypted storage.
- The `familiarity` score in each response hints at whether a similar ticket exists. Only key holders see it.

If you find a security problem, please tell us directly instead of opening a public issue.
