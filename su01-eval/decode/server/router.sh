#!/usr/bin/env bash
set -euo pipefail

ROUTER_HOST="${ROUTER_HOST:-0.0.0.0}"
ROUTER_PORT="${ROUTER_PORT:-30000}"
ROUTER_POLICY="${ROUTER_POLICY:-power_of_two}"
WORKER_URLS_RAW="${WORKER_URLS:-}"
REQUEST_TIMEOUT_SECS="${REQUEST_TIMEOUT_SECS:-${REQUEST_TIMEOUT:-1800000}}"
QUEUE_TIMEOUT_SECS="${QUEUE_TIMEOUT_SECS:-180000}"
CB_FAILURE_THRESHOLD="${CB_FAILURE_THRESHOLD:-30}"
CB_WINDOW_DURATION_SECS="${CB_WINDOW_DURATION_SECS:-300}"
CB_TIMEOUT_DURATION_SECS="${CB_TIMEOUT_DURATION_SECS:-20}"
HEALTH_CHECK_TIMEOUT_SECS="${HEALTH_CHECK_TIMEOUT_SECS:-10}"
HEALTH_CHECK_INTERVAL_SECS="${HEALTH_CHECK_INTERVAL_SECS:-10}"
HEALTH_CHECK_ENDPOINT="${HEALTH_CHECK_ENDPOINT:-/model_info}"
SKIP_WORKER_CHECK="${SKIP_WORKER_CHECK:-0}"
SGLANG_INTERNAL_NO_PROXY="${SGLANG_INTERNAL_NO_PROXY:-localhost,127.0.0.1}"

WORKER_URLS=()

show_help() {
    cat <<'EOF'
Usage:
  router.sh --worker-url http://host1:port --worker-url http://host2:port [options]

Options:
      --worker-url URL       Add one SGLang worker URL. Repeatable.
      --worker-urls CSV      Comma-separated worker URLs.
      --host HOST            Router host. Default: 0.0.0.0.
      --port PORT            Router port. Default: 30000.
      --policy POLICY        Router policy. Default: power_of_two.
      --skip-worker-check    Do not pre-check TCP reachability.
  -h, --help                 Show this help.

Environment:
  WORKER_URLS, ROUTER_HOST, ROUTER_PORT, ROUTER_POLICY,
  REQUEST_TIMEOUT or REQUEST_TIMEOUT_SECS, QUEUE_TIMEOUT_SECS,
  SGLANG_INTERNAL_NO_PROXY.
EOF
}

add_worker_urls_csv() {
    local csv="$1"
    local old_ifs="$IFS"
    IFS=','
    read -r -a items <<< "$csv"
    IFS="$old_ifs"
    for item in "${items[@]}"; do
        item="${item//[[:space:]]/}"
        [ -n "$item" ] && WORKER_URLS+=("$item")
    done
}

if [ -n "$WORKER_URLS_RAW" ]; then
    add_worker_urls_csv "$WORKER_URLS_RAW"
fi

while [[ $# -gt 0 ]]; do
    case "$1" in
        --worker-url)
            WORKER_URLS+=("$2")
            shift 2
            ;;
        --worker-urls)
            add_worker_urls_csv "$2"
            shift 2
            ;;
        --host)
            ROUTER_HOST="$2"
            shift 2
            ;;
        --port)
            ROUTER_PORT="$2"
            shift 2
            ;;
        --policy)
            ROUTER_POLICY="$2"
            shift 2
            ;;
        --skip-worker-check)
            SKIP_WORKER_CHECK=1
            shift
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        *)
            echo "Unknown argument: $1" >&2
            show_help >&2
            exit 1
            ;;
    esac
done

require_uint() {
    local name="$1"
    local value="$2"
    if ! [[ "$value" =~ ^[0-9]+$ ]] || [ "$value" -lt 1 ]; then
        echo "Error: $name must be a positive integer, got: $value" >&2
        exit 1
    fi
}

require_uint "--port" "$ROUTER_PORT"
require_uint "REQUEST_TIMEOUT_SECS" "$REQUEST_TIMEOUT_SECS"
require_uint "QUEUE_TIMEOUT_SECS" "$QUEUE_TIMEOUT_SECS"
require_uint "CB_FAILURE_THRESHOLD" "$CB_FAILURE_THRESHOLD"
require_uint "CB_WINDOW_DURATION_SECS" "$CB_WINDOW_DURATION_SECS"
require_uint "CB_TIMEOUT_DURATION_SECS" "$CB_TIMEOUT_DURATION_SECS"
require_uint "HEALTH_CHECK_TIMEOUT_SECS" "$HEALTH_CHECK_TIMEOUT_SECS"
require_uint "HEALTH_CHECK_INTERVAL_SECS" "$HEALTH_CHECK_INTERVAL_SECS"

if [ "${#WORKER_URLS[@]}" -eq 0 ]; then
    echo "Error: at least one worker URL is required via --worker-url or WORKER_URLS" >&2
    exit 1
fi

_np_existing="${NO_PROXY:-${no_proxy:-}}"
if [ -n "$_np_existing" ]; then
    export NO_PROXY="${_np_existing},${SGLANG_INTERNAL_NO_PROXY}"
else
    export NO_PROXY="$SGLANG_INTERNAL_NO_PROXY"
fi
export no_proxy="$NO_PROXY"
unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy

if [ "$SKIP_WORKER_CHECK" -eq 0 ]; then
    for url in "${WORKER_URLS[@]}"; do
        parsed=$(python3 - "$url" <<'PY'
import sys
from urllib.parse import urlparse

parsed = urlparse(sys.argv[1])
if not parsed.hostname or not parsed.port:
    raise SystemExit(1)
print(parsed.hostname)
print(parsed.port)
PY
)
        host=$(printf '%s\n' "$parsed" | sed -n '1p')
        port=$(printf '%s\n' "$parsed" | sed -n '2p')
        echo "Checking worker $url ..."
        if ! timeout 2 bash -c ":</dev/tcp/$host/$port" 2>/dev/null; then
            echo "Error: worker is not reachable: $url" >&2
            exit 1
        fi
    done
fi

echo "Starting SGLang router on ${ROUTER_HOST}:${ROUTER_PORT}"
echo "Decode worker URLs: ${WORKER_URLS[*]}"

python3 -m sglang_router.launch_router \
    --worker-urls "${WORKER_URLS[@]}" \
    --host "$ROUTER_HOST" \
    --policy "$ROUTER_POLICY" \
    --port "$ROUTER_PORT" \
    --request-timeout-secs "$REQUEST_TIMEOUT_SECS" \
    --queue-timeout-secs "$QUEUE_TIMEOUT_SECS" \
    --cb-failure-threshold "$CB_FAILURE_THRESHOLD" \
    --cb-window-duration-secs "$CB_WINDOW_DURATION_SECS" \
    --cb-timeout-duration-secs "$CB_TIMEOUT_DURATION_SECS" \
    --health-check-timeout-secs "$HEALTH_CHECK_TIMEOUT_SECS" \
    --health-check-interval-secs "$HEALTH_CHECK_INTERVAL_SECS" \
    --health-check-endpoint "$HEALTH_CHECK_ENDPOINT"
