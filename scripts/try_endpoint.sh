#!/usr/bin/env bash
# Send every sample ticket in examples/tickets.json to the router and print the decisions.
#
#   ROUTER_URL=https://1-2-3-4.sslip.io ROUTER_API_KEY=... scripts/try_endpoint.sh
set -euo pipefail
: "${ROUTER_URL:?set ROUTER_URL}" "${ROUTER_API_KEY:?set ROUTER_API_KEY}"
cd "$(dirname "$0")/.."
n=$(jq length examples/tickets.json)
for i in $(seq 0 $((n - 1))); do
  echo "--- $(jq -r ".[$i].why" examples/tickets.json)"
  jq -c ".[$i].request" examples/tickets.json | curl -s -X POST "$ROUTER_URL/v1/route" \
    -H "Content-Type: application/json" -H "X-API-Key: $ROUTER_API_KEY" -d @- \
    | jq -c '{queue, confidence, auto_routed, top_queues: [.top_queues[]?.queue], language, translated, familiarity, error}'
done
