#!/usr/bin/env bash
# Source this before running terraform:   source infra/scripts/tf-env.sh
#
# Exports the DigitalOcean token from doctl's config (the current context) and the Spaces
# key Terraform uses to manage the bucket, from a root-only file outside the repository.
# Nothing is printed and nothing is written into the repository.
_cfg="${HOME}/Library/Application Support/doctl/config.yaml"
[ -f "$_cfg" ] || _cfg="${HOME}/.config/doctl/config.yaml"
_ctx=$(doctl auth list 2>/dev/null | awk '/\(current\)/{print $1}')
export DIGITALOCEAN_TOKEN=$(python3 - "$_cfg" "$_ctx" <<'PY'
import sys, re
cfg, ctx = sys.argv[1], sys.argv[2]
text = open(cfg).read()
m = re.search(r"auth-contexts:\n((?:[ \t]+.*\n)+)", text)
if m and ctx and ctx != "default":
    for line in m.group(1).splitlines():
        k, _, v = line.strip().partition(":")
        if k == ctx:
            print(v.strip()); break
else:
    print(re.search(r"^access-token:\s*(\S+)", text, re.M).group(1))
PY
)
_spaces="${HOME}/.config/team3/spaces-terraform.env"
if [ -f "$_spaces" ]; then set -a; . "$_spaces"; set +a; fi
[ -n "$DIGITALOCEAN_TOKEN" ] && echo "DIGITALOCEAN_TOKEN set for doctl context '${_ctx}'" || echo "could not read the DigitalOcean token"
[ -n "$SPACES_ACCESS_KEY_ID" ] && echo "Spaces key set" || echo "no Spaces key at $_spaces"
unset _cfg _ctx _spaces
