# Evidence

Raw output from the scanners, the cloud training run and the live system, kept so every
claim in the report can be checked. Each file is what the tool printed, not a summary of it.

| File | What it shows | Rubric criterion |
| --- | --- | --- |
| [gitleaks-history.json](gitleaks-history.json) | Gitleaks over the whole git history found no secrets (an empty list) | Security, no secrets in version control |
| [semgrep.json](semgrep.json) | Semgrep 1.177 with the Python, security-audit and secrets rule packs, 44 shipped files scanned, 0 findings, 0 errors | Security, static code scanning |
| [trivy-image.txt](trivy-image.txt) | Trivy scan of the inference image, 0 vulnerabilities in the Debian base and every Python package | Security, container scanning |
| [sbom-cyclonedx.json](sbom-cyclonedx.json) | CycloneDX 1.7 software bill of materials for the image, 168 components, the list of everything that ships | Security and code repositories, DevSecOps tooling |
| [old-droplet-ssh-probes.txt](old-droplet-ssh-probes.txt) | 16,547 rejected SSH attempts from 399 addresses on the midterm droplet, whose firewall allowed SSH from anywhere. The reason SSH is now closed to everyone but the team | Security, the threat model uses real attack data |
| [gpu-training-run.log](gpu-training-run.log) | The full retrain on the rented RTX 4000 Ada GPU, every stage timed, validation 89.79%, the quality gate promoting the model, 185 MB artifact published to Spaces | Architecture (where heavy compute is needed), model training |
| [cd-rollout-and-latency.txt](cd-rollout-and-latency.txt) | A push to `main` reaching production with nobody logging in, healthy 19 seconds after the rollout started, and the live latency before and after a fix | Code repositories (GitHub Actions), deployment |
| [live-replay-and-drift.txt](live-replay-and-drift.txt) | 250 held-out tickets replayed through the live endpoint, 90.4% correct, and the drift report that followed. Live accuracy 91.6% over 131 human corrections, status "warn" because unfamiliar tickets rose 6.2 points | Observability, inference endpoint |
| [drift-demo-local.json](drift-demo-local.json) | Off-topic traffic sent on purpose to a local copy of the service, and the drift report flagging it (unfamiliar tickets up 23.9 points, confidence down 0.168) | Observability, how drift is detected |

Two notes for anyone reading the raw files.

- `gpu-training-run.log` ends with a database error, `relation "training_runs" does not
  exist`. That is expected. The run happened before the audit tables were created, so the
  pipeline could not record itself. Its row was written afterwards from the saved artifact by
  [scripts/record_run.py](../scripts/record_run.py). The model itself was published and
  promoted normally, and the job exited with status 0.
- CI regenerates the Semgrep, Gitleaks, Trivy and SBOM results on every push. The files here
  are local runs of the same tools, kept so the results can be read without opening GitHub.
  The CI runs are under the repository's Actions tab.
