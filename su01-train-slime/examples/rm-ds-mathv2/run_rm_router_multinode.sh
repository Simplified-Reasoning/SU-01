#!/bin/bash

set -euo pipefail

DEFAULT_WORKER_PORT=34882
DEFAULT_ROUTER_PORT=34886
DEFAULT_REWARD_PORT=8006
MODEL_NAME="DeepSeek-Math-V2"
DEEPSEEK_MODEL_PATH="/root/models/DeepSeek-Math-V2"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
LAUNCH_MODEL_SCRIPT="${BASE_DIR}/launch_model.py"
DISCOVERY_DIR="${SCRIPT_DIR}/.router_multinode_discovery"
TIKTOKEN_ENCODINGS_BASE="${BASE_DIR}/encodings"

WORKER_PORT=$DEFAULT_WORKER_PORT
ROUTER_PORT=$DEFAULT_ROUTER_PORT
REWARD_PORT=$DEFAULT_REWARD_PORT
NUM_GPU=""
NNODES=1
NODE_RANK=0
MASTER_ADDR=""
ROUTER_POLICY="${ROUTER_POLICY:-cache_aware}"
REWARD_SERVER_WORKERS="${REWARD_SERVER_WORKERS:-64}"
PROOF_MAX_INFLIGHT="${PROOF_MAX_INFLIGHT:-64}"
PROOF_THREAD_WORKERS="${PROOF_THREAD_WORKERS:-64}"
REQUEST_LOG_EVERY="${REQUEST_LOG_EVERY:-1}"
SGLANG_STARTUP_TIMEOUT="${SGLANG_STARTUP_TIMEOUT:-1800}"
ROUTER_STARTUP_TIMEOUT="${ROUTER_STARTUP_TIMEOUT:-300}"

show_help() {
    cat <<EOF
Usage: $0 [OPTIONS]

DeepSeek multi-node integrated workers + router deployment.
Each node starts one independent DeepSeek worker.
Node 0 additionally starts the sglang router and reward server.

Options:
  -g, --num-gpu NUM         Number of GPUs per node
      --nnodes NUM          Total number of nodes (default: $NNODES)
      --node-rank NUM       Current node rank (default: $NODE_RANK)
      --master-addr ADDR    Rjob master address for node discovery
      --worker-port PORT    Worker port on every node (default: $DEFAULT_WORKER_PORT)
      --router-port PORT    Router port on node 0 (default: $DEFAULT_ROUTER_PORT)
  -r, --reward-port PORT    Reward server port on node 0 (default: $DEFAULT_REWARD_PORT)
      --router-policy NAME  Router dispatch policy (default: $ROUTER_POLICY)
  -h, --help                Show this help message

Example:
  $0 -g 8 --nnodes 4 --node-rank 0 --master-addr 10.0.0.1 \\
     --worker-port 34882 --router-port 34886 --reward-port 8006
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        -g|--num-gpu)
            NUM_GPU="$2"
            shift 2
            ;;
        --nnodes)
            NNODES="$2"
            shift 2
            ;;
        --node-rank)
            NODE_RANK="$2"
            shift 2
            ;;
        --master-addr)
            MASTER_ADDR="$2"
            shift 2
            ;;
        --worker-port)
            WORKER_PORT="$2"
            shift 2
            ;;
        --router-port)
            ROUTER_PORT="$2"
            shift 2
            ;;
        -r|--reward-port)
            REWARD_PORT="$2"
            shift 2
            ;;
        --router-policy)
            ROUTER_POLICY="$2"
            shift 2
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        *)
            echo "Error: Unknown parameter $1"
            show_help
            exit 1
            ;;
    esac
done

if [ -z "$NUM_GPU" ]; then
    echo "Error: Must specify GPU count per node through -g/--num-gpu"
    exit 1
fi

for value_name in NUM_GPU NNODES NODE_RANK WORKER_PORT ROUTER_PORT REWARD_PORT; do
    value="${!value_name}"
    if ! [[ "$value" =~ ^[0-9]+$ ]]; then
        echo "Error: $value_name must be an integer"
        exit 1
    fi
done

if [ "$NUM_GPU" -le 0 ] || [ "$NNODES" -le 0 ]; then
    echo "Error: NUM_GPU and NNODES must be positive integers"
    exit 1
fi

if [ "$NODE_RANK" -lt 0 ] || [ "$NODE_RANK" -ge "$NNODES" ]; then
    echo "Error: NODE_RANK must be in [0, NNODES)"
    exit 1
fi

if [ "$NNODES" -gt 1 ] && [ -z "$MASTER_ADDR" ]; then
    echo "Error: Multi-node mode must provide --master-addr"
    exit 1
fi

for port in "$WORKER_PORT" "$ROUTER_PORT" "$REWARD_PORT"; do
    if [ "$port" -lt 1024 ] || [ "$port" -gt 65535 ]; then
        echo "Error: Port must be between 1024-65535"
        exit 1
    fi
done

if [ "$WORKER_PORT" -eq "$ROUTER_PORT" ] || [ "$WORKER_PORT" -eq "$REWARD_PORT" ] || [ "$ROUTER_PORT" -eq "$REWARD_PORT" ]; then
    echo "Error: worker/router/reward ports cannot be repeated"
    exit 1
fi

if [ ! -f "$LAUNCH_MODEL_SCRIPT" ]; then
    echo "Error: Cannot find model launch script $LAUNCH_MODEL_SCRIPT"
    exit 1
fi

if [ ! -d "$DEEPSEEK_MODEL_PATH" ]; then
    echo "Error: Model path does not exist $DEEPSEEK_MODEL_PATH"
    exit 1
fi

TP_GPU="$NUM_GPU"
DP_GPU="$NUM_GPU"
MODEL_PATH="$DEEPSEEK_MODEL_PATH"
LAUNCH_EXTRA_ARGS=(
    --trust-remote-code
    --tool-call-parser deepseekv32
    --reasoning-parser deepseek-v3
    --context-length 52768
    --data-parallel-size "$DP_GPU"
    --enable-dp-attention
    --kv-cache-dtype fp8_e4m3
    --speculative-algorithm EAGLE --speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 4
)

export no_proxy="10.0.0.0/8,100.96.0.0/12,172.16.0.0/12,192.168.0.0/16,127.0.0.1,localhost"
if [ -n "$MASTER_ADDR" ]; then
    export no_proxy="${no_proxy},${MASTER_ADDR}"
fi

LOG_DIR="${SCRIPT_DIR}/logs"
mkdir -p "$LOG_DIR" "$DISCOVERY_DIR"
WORKER_LOG_FILE="${LOG_DIR}/$(date +%Y%m%d_%H%M%S)_worker_${WORKER_PORT}_rank${NODE_RANK}.log"
ROUTER_LOG_FILE="${LOG_DIR}/$(date +%Y%m%d_%H%M%S)_router_${ROUTER_PORT}.log"
REWARD_LOG_FILE="${LOG_DIR}/$(date +%Y%m%d_%H%M%S)_reward_${REWARD_PORT}.log"

if [ "$NODE_RANK" -eq 0 ]; then
    echo "Installing reward server dependencies on node 0..."
    pip install math_verify
    pip install fastapi uvicorn torch
fi

MY_IP=$(ifconfig bond0 2>/dev/null | awk '/inet / {print $2; exit}')
if [ -z "$MY_IP" ]; then
    MY_IP=$(hostname -I 2>/dev/null | awk '{print $1}')
fi
if [ -z "$MY_IP" ]; then
    echo "Error: Cannot get current node IP"
    exit 1
fi

echo "$MY_IP" > "${DISCOVERY_DIR}/node_${NODE_RANK}.tmp"
mv -f "${DISCOVERY_DIR}/node_${NODE_RANK}.tmp" "${DISCOVERY_DIR}/node_${NODE_RANK}"

echo "==========================================="
echo "DeepSeek Router Multinode Config"
echo "==========================================="
echo "  Model name:           $MODEL_NAME"
echo "  Model path:           $MODEL_PATH"
echo "  GPUs per node:        $NUM_GPU"
echo "  Total nodes:          $NNODES"
echo "  Node rank:            $NODE_RANK"
echo "  Local IP:             $MY_IP"
echo "  Master address:       ${MASTER_ADDR:-N/A}"
echo "  Worker port:          $WORKER_PORT"
echo "  Router port:          $ROUTER_PORT"
echo "  Reward port:          $REWARD_PORT"
echo "  Router policy:        $ROUTER_POLICY"
echo "  Startup timeout:      ${SGLANG_STARTUP_TIMEOUT}s"
echo "==========================================="

cd "$BASE_DIR" || {
    echo "Error: Cannot enter directory $BASE_DIR"
    exit 1
}

kill_port_owners() {
    local port="$1"
    local pids=""
    if command -v lsof >/dev/null 2>&1; then
        pids="$(lsof -ti "tcp:${port}" 2>/dev/null | tr '\n' ' ')"
    fi
    if [ -z "$pids" ] && command -v fuser >/dev/null 2>&1; then
        pids="$(fuser "${port}/tcp" 2>/dev/null | tr -s ' ' ' ')"
    fi
    if [ -n "$pids" ]; then
        echo "Killing processes on port ${port}: ${pids}"
        kill -9 $pids 2>/dev/null || true
    fi
}

safe_kill() {
    local pid="${1:-}"
    if [ -n "$pid" ] && [[ "$pid" =~ ^[0-9]+$ ]] && [ "$pid" -gt 0 ]; then
        kill "$pid" 2>/dev/null || true
    fi
}

wait_port_closed() {
    local port="$1"
    local retries="${2:-20}"
    local i=0
    while [ "$i" -lt "$retries" ]; do
        if ! timeout 1 bash -c "echo > /dev/tcp/127.0.0.1/${port}" >/dev/null 2>&1; then
            return 0
        fi
        sleep 1
        i=$((i + 1))
    done
    return 1
}

wait_for_discovery_file() {
    local filepath="$1"
    local timeout_s="${2:-600}"
    local elapsed=0
    while [ ! -s "$filepath" ]; do
        if [ "$elapsed" -ge "$timeout_s" ]; then
            echo "Error: Waiting for $filepath timed out (${timeout_s}s)"
            exit 1
        fi
        sleep 5
        elapsed=$((elapsed + 5))
        if [ $((elapsed % 30)) -eq 0 ]; then
            echo "[rank=$NODE_RANK] Still waiting for $filepath (${elapsed}/${timeout_s}s)"
        fi
    done
    cat "$filepath"
}

wait_for_healthy() {
    local url="$1"
    local timeout_s="${2:-600}"
    local interval="${3:-30}"
    local elapsed=0
    while [ "$elapsed" -lt "$timeout_s" ]; do
        if curl -s --connect-timeout 10 --max-time 15 "$url" >/dev/null 2>&1; then
            return 0
        fi
        sleep "$interval"
        elapsed=$((elapsed + interval))
        echo "  waiting for $url ... (${elapsed}/${timeout_s}s)"
    done
    return 1
}

wait_for_healthy_or_pid_exit() {
    local url="$1"
    local timeout_s="$2"
    local interval="$3"
    local pid="$4"
    local elapsed=0
    while [ "$elapsed" -lt "$timeout_s" ]; do
        if curl -s --connect-timeout 10 --max-time 15 "$url" >/dev/null 2>&1; then
            return 0
        fi
        if ! kill -0 "$pid" 2>/dev/null; then
            return 1
        fi
        sleep "$interval"
        elapsed=$((elapsed + interval))
        echo "  waiting for $url ... (${elapsed}/${timeout_s}s)"
    done
    return 1
}

start_worker_server() {
    echo "[rank=$NODE_RANK] Starting DeepSeek worker on port $WORKER_PORT..."
    python "$LAUNCH_MODEL_SCRIPT" \
        --port "$WORKER_PORT" \
        --model-path "$MODEL_PATH" \
        --host 0.0.0.0 \
        --mem-fraction-static 0.8 \
        --tensor-parallel-size "$TP_GPU" \
        --max-running-requests 64 \
        "${LAUNCH_EXTRA_ARGS[@]}" &
    MODEL_SERVER_PID=$!
    echo "[rank=$NODE_RANK] Worker PID=$MODEL_SERVER_PID"
}

restart_worker_server() {
    if [ -n "${MODEL_SERVER_PID:-}" ]; then
        safe_kill "$MODEL_SERVER_PID"
    fi
    kill_port_owners "$WORKER_PORT"
    wait_port_closed "$WORKER_PORT" 20 || echo "Warning: worker port $WORKER_PORT may still be occupied"
    sleep 5
    start_worker_server
}

collect_worker_urls() {
    WORKER_URLS=()
    local rank
    for ((rank = 0; rank < NNODES; rank++)); do
        local node_ip
        local node_url
        node_ip=$(wait_for_discovery_file "${DISCOVERY_DIR}/node_${rank}" "$SGLANG_STARTUP_TIMEOUT")
        node_url="http://${node_ip}:${WORKER_PORT}"
        echo "[node-0] Waiting for worker rank ${rank} (${node_url}) to become healthy..."
        if [ "$rank" -eq "$NODE_RANK" ]; then
            if ! wait_for_healthy_or_pid_exit "${node_url}/health" "$SGLANG_STARTUP_TIMEOUT" 30 "$MODEL_SERVER_PID"; then
                echo "Error: Local worker failed before becoming healthy"
                exit 1
            fi
        else
            if ! wait_for_healthy "${node_url}/health" "$SGLANG_STARTUP_TIMEOUT" 30; then
                echo "Warning: Worker rank ${rank} did not become healthy within timeout, still adding to router"
            fi
        fi
        WORKER_URLS+=("$node_url")
    done
}

start_router() {
    collect_worker_urls
    echo "[node-0] Starting sglang router on port $ROUTER_PORT..."
    echo "[node-0] Worker URLs: ${WORKER_URLS[*]}"
    NCCL_TIMEOUT=360000 python -m sglang_router.launch_router \
        --worker-urls "${WORKER_URLS[@]}" \
        --policy "$ROUTER_POLICY" \
        --host 0.0.0.0 \
        --port "$ROUTER_PORT" \
        >> "$ROUTER_LOG_FILE" 2>&1 &
    ROUTER_PID=$!
    echo "[node-0] Router PID=$ROUTER_PID"
}

start_reward_server() {
    echo "[node-0] Starting reward server on port $REWARD_PORT via router port $ROUTER_PORT..."
    MODEL_PORT="$ROUTER_PORT" \
    MODEL_NAME="$MODEL_NAME" \
    PROOF_MAX_INFLIGHT="$PROOF_MAX_INFLIGHT" \
    PROOF_THREAD_WORKERS="$PROOF_THREAD_WORKERS" \
    REQUEST_LOG_EVERY="$REQUEST_LOG_EVERY" \
    TIKTOKEN_ENCODINGS_BASE="$TIKTOKEN_ENCODINGS_BASE" \
    uvicorn reward_model_server:app \
        --host 0.0.0.0 \
        --port "$REWARD_PORT" \
        --timeout-keep-alive 30 \
        --log-level debug \
        --workers "$REWARD_SERVER_WORKERS" \
        >> "$REWARD_LOG_FILE" 2>&1 &
    REWARD_SERVER_PID=$!
    echo "[node-0] Reward PID=$REWARD_SERVER_PID"
}

restart_router_and_reward() {
    if [ -n "${REWARD_SERVER_PID:-}" ]; then
        safe_kill "$REWARD_SERVER_PID"
        wait "$REWARD_SERVER_PID" 2>/dev/null || true
    fi
    if [ -n "${ROUTER_PID:-}" ]; then
        safe_kill "$ROUTER_PID"
        wait "$ROUTER_PID" 2>/dev/null || true
    fi
    kill_port_owners "$REWARD_PORT"
    kill_port_owners "$ROUTER_PORT"
    wait_port_closed "$REWARD_PORT" 20 || true
    wait_port_closed "$ROUTER_PORT" 20 || true

    start_router
    echo "[node-0] Waiting for router to become healthy..."
    if ! wait_for_healthy "http://localhost:${ROUTER_PORT}/health" "$ROUTER_STARTUP_TIMEOUT" 10; then
        echo "Warning: router health check failed, proceeding anyway..."
    fi
    start_reward_server
}

monitor_node0() {
    while true; do
        if ! kill -0 "$MODEL_SERVER_PID" 2>/dev/null || ! curl -s --connect-timeout 10 --max-time 15 "http://localhost:${WORKER_PORT}/health" >/dev/null 2>&1; then
            echo "[node-0] Local worker is unhealthy, restarting worker/router/reward..."
            restart_worker_server
            if ! wait_for_healthy_or_pid_exit "http://localhost:${WORKER_PORT}/health" "$SGLANG_STARTUP_TIMEOUT" 30 "$MODEL_SERVER_PID"; then
                echo "[node-0] Worker is failed during restart, retrying later..."
                sleep 10
                continue
            fi
            restart_router_and_reward
        elif ! kill -0 "$ROUTER_PID" 2>/dev/null || ! curl -s --connect-timeout 10 --max-time 15 "http://localhost:${ROUTER_PORT}/health" >/dev/null 2>&1; then
            echo "[node-0] Router is unhealthy, restarting router/reward..."
            restart_router_and_reward
        elif ! kill -0 "$REWARD_SERVER_PID" 2>/dev/null; then
            echo "[node-0] Reward server is exited, restarting..."
            start_reward_server
        fi
        sleep 30
    done
}

start_worker_server

if [ "$NODE_RANK" -ne 0 ]; then
    echo "[rank=$NODE_RANK] Worker-only node; router and reward stay on node 0."
    trap 'echo "[rank='"$NODE_RANK"'] Shutting down worker..."; safe_kill "${MODEL_SERVER_PID:-}"; exit' SIGTERM SIGINT
    while true; do
        exit_code=0
        wait "$MODEL_SERVER_PID" || exit_code=$?
        echo "[rank=$NODE_RANK] Worker exited with code $exit_code, restarting in 10s..."
        sleep 10
        restart_worker_server
    done
fi

echo "[node-0] Waiting for local worker to become healthy..."
if ! wait_for_healthy_or_pid_exit "http://localhost:${WORKER_PORT}/health" "$SGLANG_STARTUP_TIMEOUT" 30 "$MODEL_SERVER_PID"; then
    echo "Error: local worker failed before becoming healthy"
    exit 1
fi

start_router
echo "[node-0] Waiting for router to become healthy..."
if ! wait_for_healthy "http://localhost:${ROUTER_PORT}/health" "$ROUTER_STARTUP_TIMEOUT" 10; then
    echo "Warning: Router health check failed, proceeding anyway..."
fi

start_reward_server
monitor_node0 &
MONITOR_PID=$!

trap 'echo "[node-0] Shutting down all services..."; safe_kill "${REWARD_SERVER_PID:-}"; safe_kill "${ROUTER_PID:-}"; safe_kill "${MODEL_SERVER_PID:-}"; safe_kill "${MONITOR_PID:-}"; exit' SIGTERM SIGINT

wait
