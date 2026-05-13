#!/usr/bin/env bash
set -euo pipefail

DEFAULT_MODEL_PORT=34883
DEFAULT_NCCL_PORT=15001
DEFAULT_DIST_INIT_ADDR=""
DEFAULT_HOST="127.0.0.1"
DEFAULT_MODEL_PATH=""
DEFAULT_SERVED_MODEL_NAME="SU01"
DEFAULT_TP=1
DEFAULT_MEM_FRACTION_STATIC=0.8
DEFAULT_ATTENTION_BACKEND="fa3"
DEFAULT_REASONING_PARSER=""
DEFAULT_CUDA_GRAPH_MAX_BS=256
DEFAULT_PIECEWISE_CUDA_GRAPH_MAX_TOKENS=8192
DEFAULT_ENABLE_FLASHINFER_ALLREDUCE_FUSION=0
DEFAULT_LOG_DIR="./logs"
DEFAULT_INTERNAL_NO_PROXY="localhost,127.0.0.1"

MODEL_PORT="${MODEL_PORT:-$DEFAULT_MODEL_PORT}"
NCCL_PORT="${NCCL_PORT:-$DEFAULT_NCCL_PORT}"
DIST_INIT_ADDR="${DIST_INIT_ADDR:-$DEFAULT_DIST_INIT_ADDR}"
HOST="${HOST:-$DEFAULT_HOST}"
MODEL_PATH="${MODEL_PATH:-$DEFAULT_MODEL_PATH}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-${MODEL_NAME:-$DEFAULT_SERVED_MODEL_NAME}}"
TP="${TP:-${TP_SIZE:-$DEFAULT_TP}}"
MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC:-$DEFAULT_MEM_FRACTION_STATIC}"
ATTENTION_BACKEND="${ATTENTION_BACKEND:-$DEFAULT_ATTENTION_BACKEND}"
REASONING_PARSER="${REASONING_PARSER:-$DEFAULT_REASONING_PARSER}"
CUDA_GRAPH_MAX_BS="${CUDA_GRAPH_MAX_BS:-$DEFAULT_CUDA_GRAPH_MAX_BS}"
PIECEWISE_CUDA_GRAPH_MAX_TOKENS="${PIECEWISE_CUDA_GRAPH_MAX_TOKENS:-$DEFAULT_PIECEWISE_CUDA_GRAPH_MAX_TOKENS}"
ENABLE_FLASHINFER_ALLREDUCE_FUSION="${ENABLE_FLASHINFER_ALLREDUCE_FUSION:-$DEFAULT_ENABLE_FLASHINFER_ALLREDUCE_FUSION}"
CONTEXT_LENGTH="${CONTEXT_LENGTH:-}"
ALLOW_AUTO_TRUNCATE="${ALLOW_AUTO_TRUNCATE:-0}"
DISABLE_CUDA_GRAPH="${DISABLE_CUDA_GRAPH:-0}"
DISABLE_RADIX_CACHE="${DISABLE_RADIX_CACHE:-0}"
DISABLE_OVERLAP_SCHEDULE="${DISABLE_OVERLAP_SCHEDULE:-0}"
AUTO_RESTART="${AUTO_RESTART:-1}"
DRY_RUN="${DRY_RUN:-0}"
LOG_DIR="${LOG_DIR:-$DEFAULT_LOG_DIR}"
SGLANG_INTERNAL_NO_PROXY="${SGLANG_INTERNAL_NO_PROXY:-$DEFAULT_INTERNAL_NO_PROXY}"
NCCL_PORT_SET=0
DIST_INIT_ADDR_SET=0
EXTRA_LAUNCH_ARGS=()
PASSTHROUGH_ARGS=()

show_help() {
    cat <<EOF
Usage:
  server.sh --model-path /path/to/model [options] [-- extra sglang args...]

Options:
  -m, --model-port PORT       SGLang service port. Default: $DEFAULT_MODEL_PORT.
      --port PORT             Alias for --model-port.
      --nccl-port PORT        NCCL port. Default: $DEFAULT_NCCL_PORT.
      --dist-init-addr ADDR   Distributed init address. Empty by default.
      --host HOST             Listen host. Default: $DEFAULT_HOST.
  -p, --model-path PATH       Model checkpoint directory. Required.
  -n, --served-model-name N   Served model name. Default: SU01.
      --model-name N          Alias for --served-model-name.
  -t, --tp N                  Tensor parallel size. Default: $DEFAULT_TP.
      --tp-size N             Alias for --tp.
      --mem-fraction-static F Static memory fraction. Default: $DEFAULT_MEM_FRACTION_STATIC.
      --attention-backend B   Attention backend. Default: $DEFAULT_ATTENTION_BACKEND.
      --reasoning-parser P    Optional reasoning parser.
      --enable-flashinfer-allreduce-fusion
      --disable-flashinfer-allreduce-fusion
      --context-length N      Explicit context length.
      --allow-auto-truncate   Pass --allow-auto-truncate.
      --disable-cuda-graph    Pass --disable-cuda-graph.
      --disable-radix-cache   Pass --disable-radix-cache.
      --disable-overlap-schedule
      --cuda-graph-max-bs N   CUDA graph max batch size. Default: $DEFAULT_CUDA_GRAPH_MAX_BS.
      --piecewise-cuda-graph-max-tokens N
                             Piecewise CUDA graph token limit. Default: $DEFAULT_PIECEWISE_CUDA_GRAPH_MAX_TOKENS.
      --no-auto-restart       Do not restart the server after process exit.
      --log-dir DIR           Log directory. Default: $DEFAULT_LOG_DIR.
      --dry-run               Print the launch command without executing it.
  -h, --help                  Show this help.

Environment:
  MODEL_PATH, MODEL_NAME, SERVED_MODEL_NAME, MODEL_PORT, HOST, TP, TP_SIZE,
  MEM_FRACTION_STATIC, ATTENTION_BACKEND, REASONING_PARSER, CONTEXT_LENGTH,
  ALLOW_AUTO_TRUNCATE, DISABLE_CUDA_GRAPH, DISABLE_RADIX_CACHE,
  DISABLE_OVERLAP_SCHEDULE, CUDA_GRAPH_MAX_BS,
  PIECEWISE_CUDA_GRAPH_MAX_TOKENS, AUTO_RESTART, LOG_DIR,
  SGLANG_INTERNAL_NO_PROXY.
EOF
}

require_value() {
    local option="$1"
    local value="${2:-}"
    if [ -z "$value" ]; then
        echo "Error: $option requires a value" >&2
        exit 1
    fi
}

require_uint() {
    local name="$1"
    local value="$2"
    if ! [[ "$value" =~ ^[0-9]+$ ]] || [ "$value" -lt 1 ]; then
        echo "Error: $name must be a positive integer, got: $value" >&2
        exit 1
    fi
}

require_port() {
    local name="$1"
    local value="$2"
    if ! [[ "$value" =~ ^[0-9]+$ ]] || [ "$value" -lt 1024 ] || [ "$value" -gt 65535 ]; then
        echo "Error: $name must be an integer between 1024 and 65535, got: $value" >&2
        exit 1
    fi
}

is_enabled() {
    case "${1,,}" in
        1|true|yes|on) return 0 ;;
        *) return 1 ;;
    esac
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        -m|--model-port|--port)
            require_value "$1" "${2:-}"
            MODEL_PORT="$2"
            shift 2
            ;;
        --nccl-port)
            require_value "$1" "${2:-}"
            NCCL_PORT="$2"
            NCCL_PORT_SET=1
            shift 2
            ;;
        --dist-init-addr)
            require_value "$1" "${2:-}"
            DIST_INIT_ADDR="$2"
            DIST_INIT_ADDR_SET=1
            shift 2
            ;;
        --host)
            require_value "$1" "${2:-}"
            HOST="$2"
            shift 2
            ;;
        -p|--model-path)
            require_value "$1" "${2:-}"
            MODEL_PATH="$2"
            shift 2
            ;;
        -n|--served-model-name|--model-name)
            require_value "$1" "${2:-}"
            SERVED_MODEL_NAME="$2"
            shift 2
            ;;
        -t|--tp|--tp-size)
            require_value "$1" "${2:-}"
            TP="$2"
            shift 2
            ;;
        --mem-fraction-static)
            require_value "$1" "${2:-}"
            MEM_FRACTION_STATIC="$2"
            shift 2
            ;;
        --attention-backend)
            require_value "$1" "${2:-}"
            ATTENTION_BACKEND="$2"
            shift 2
            ;;
        --reasoning-parser)
            require_value "$1" "${2:-}"
            REASONING_PARSER="$2"
            shift 2
            ;;
        --enable-flashinfer-allreduce-fusion)
            ENABLE_FLASHINFER_ALLREDUCE_FUSION=1
            shift
            ;;
        --disable-flashinfer-allreduce-fusion)
            ENABLE_FLASHINFER_ALLREDUCE_FUSION=0
            shift
            ;;
        --context-length)
            require_value "$1" "${2:-}"
            CONTEXT_LENGTH="$2"
            shift 2
            ;;
        --allow-auto-truncate)
            ALLOW_AUTO_TRUNCATE=1
            shift
            ;;
        --disable-cuda-graph)
            DISABLE_CUDA_GRAPH=1
            shift
            ;;
        --disable-radix-cache)
            DISABLE_RADIX_CACHE=1
            shift
            ;;
        --disable-overlap-schedule)
            DISABLE_OVERLAP_SCHEDULE=1
            shift
            ;;
        --cuda-graph-max-bs)
            require_value "$1" "${2:-}"
            CUDA_GRAPH_MAX_BS="$2"
            shift 2
            ;;
        --piecewise-cuda-graph-max-tokens)
            require_value "$1" "${2:-}"
            PIECEWISE_CUDA_GRAPH_MAX_TOKENS="$2"
            shift 2
            ;;
        --no-auto-restart)
            AUTO_RESTART=0
            shift
            ;;
        --log-dir)
            require_value "$1" "${2:-}"
            LOG_DIR="$2"
            shift 2
            ;;
        --dry-run)
            DRY_RUN=1
            shift
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        --)
            shift
            PASSTHROUGH_ARGS+=("$@")
            break
            ;;
        *)
            echo "Unknown argument: $1" >&2
            show_help >&2
            exit 1
            ;;
    esac
done

if [ -z "$MODEL_PATH" ]; then
    echo "Error: --model-path or MODEL_PATH is required" >&2
    exit 1
fi

require_port "--model-port" "$MODEL_PORT"
require_port "--nccl-port" "$NCCL_PORT"
require_uint "--tp" "$TP"
require_uint "--cuda-graph-max-bs" "$CUDA_GRAPH_MAX_BS"
require_uint "--piecewise-cuda-graph-max-tokens" "$PIECEWISE_CUDA_GRAPH_MAX_TOKENS"
[ -n "$CONTEXT_LENGTH" ] && require_uint "--context-length" "$CONTEXT_LENGTH"

if [ "$DRY_RUN" -eq 0 ] && [ ! -d "$MODEL_PATH" ]; then
    echo "Error: model path does not exist: $MODEL_PATH" >&2
    exit 1
fi

CONFIG_JSON="$MODEL_PATH/config.json"
if [ -f "$CONFIG_JSON" ]; then
    MODEL_CONFIG_INFO=$(python3 - "$CONFIG_JSON" <<'PY'
import json
import sys

try:
    with open(sys.argv[1], "r", encoding="utf-8") as handle:
        config = json.load(handle)
except Exception:
    config = {}

text_config = config.get("text_config")
if not isinstance(text_config, dict):
    text_config = {}

num_attention_heads = config.get("num_attention_heads")
if num_attention_heads is None:
    num_attention_heads = text_config.get("num_attention_heads")

model_type = config.get("model_type") or text_config.get("model_type") or ""
architectures = config.get("architectures") or []
if not isinstance(architectures, list):
    architectures = [architectures]

print("" if num_attention_heads is None else num_attention_heads)
print(model_type)
print(",".join(str(item) for item in architectures))
PY
)
    NUM_ATTENTION_HEADS=$(printf '%s\n' "$MODEL_CONFIG_INFO" | sed -n '1p')
    MODEL_TYPE=$(printf '%s\n' "$MODEL_CONFIG_INFO" | sed -n '2p')
    MODEL_ARCHITECTURES=$(printf '%s\n' "$MODEL_CONFIG_INFO" | sed -n '3p')

    if [ "$TP" -gt 1 ] && { [ "$MODEL_TYPE" = "gemma4" ] || [[ ",$MODEL_ARCHITECTURES," == *",Gemma4ForConditionalGeneration,"* ]]; }; then
        echo "Error: this SGLang setup does not support tensor parallel for Gemma4 fallback." >&2
        echo "       Use --tp 1 or a backend/version that supports this model." >&2
        exit 1
    fi

    if [[ "$NUM_ATTENTION_HEADS" =~ ^[0-9]+$ ]] && [ "$NUM_ATTENTION_HEADS" -gt 0 ]; then
        if [ $((NUM_ATTENTION_HEADS % TP)) -ne 0 ]; then
            VALID_TPS=()
            for ((i = 1; i <= NUM_ATTENTION_HEADS; i++)); do
                if [ $((NUM_ATTENTION_HEADS % i)) -eq 0 ]; then
                    VALID_TPS+=("$i")
                fi
            done
            echo "Error: --tp $TP is incompatible with num_attention_heads=$NUM_ATTENTION_HEADS." >&2
            echo "       Valid TP values: ${VALID_TPS[*]}" >&2
            exit 1
        fi
    fi
fi

if [ -n "$CONTEXT_LENGTH" ]; then
    EXTRA_LAUNCH_ARGS+=(--context-length "$CONTEXT_LENGTH")
fi
if is_enabled "$ALLOW_AUTO_TRUNCATE"; then
    EXTRA_LAUNCH_ARGS+=(--allow-auto-truncate)
fi
if [ "$TP" -gt 1 ] || [ "$NCCL_PORT_SET" -eq 1 ]; then
    EXTRA_LAUNCH_ARGS+=(--nccl-port "$NCCL_PORT")
fi
if [ -n "$DIST_INIT_ADDR" ] || [ "$DIST_INIT_ADDR_SET" -eq 1 ]; then
    EXTRA_LAUNCH_ARGS+=(--dist-init-addr "$DIST_INIT_ADDR")
fi
if is_enabled "$ENABLE_FLASHINFER_ALLREDUCE_FUSION"; then
    EXTRA_LAUNCH_ARGS+=(--enable-flashinfer-allreduce-fusion)
fi
if is_enabled "$DISABLE_CUDA_GRAPH"; then
    EXTRA_LAUNCH_ARGS+=(--disable-cuda-graph)
fi
if is_enabled "$DISABLE_RADIX_CACHE"; then
    EXTRA_LAUNCH_ARGS+=(--disable-radix-cache)
fi
if is_enabled "$DISABLE_OVERLAP_SCHEDULE"; then
    EXTRA_LAUNCH_ARGS+=(--disable-overlap-schedule)
fi
if [ -n "$REASONING_PARSER" ]; then
    EXTRA_LAUNCH_ARGS+=(--reasoning-parser "$REASONING_PARSER")
fi

_np_existing="${NO_PROXY:-${no_proxy:-}}"
if [ -n "$_np_existing" ]; then
    export NO_PROXY="${_np_existing},${SGLANG_INTERNAL_NO_PROXY}"
else
    export NO_PROXY="$SGLANG_INTERNAL_NO_PROXY"
fi
export no_proxy="$NO_PROXY"
unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy

if [ -n "${TIKTOKEN_RS_CACHE_DIR:-}" ]; then
    export TIKTOKEN_RS_CACHE_DIR
fi

LAUNCH_ARGS=(
    --model-path "$MODEL_PATH"
    --served-model-name "$SERVED_MODEL_NAME"
    --tp "$TP"
    --trust-remote-code
    --skip-server-warmup
    --mem-fraction-static "$MEM_FRACTION_STATIC"
    --moe-runner-backend deep_gemm
    --speculative-moe-runner-backend deep_gemm
    --attention-backend "$ATTENTION_BACKEND"
    --enable-memory-saver
    --enable-draft-weights-cpu-backup
    --cuda-graph-max-bs "$CUDA_GRAPH_MAX_BS"
    --piecewise-cuda-graph-max-tokens "$PIECEWISE_CUDA_GRAPH_MAX_TOKENS"
    "${EXTRA_LAUNCH_ARGS[@]}"
    "${PASSTHROUGH_ARGS[@]}"
    --host "$HOST"
    --port "$MODEL_PORT"
)

run_launch_server() {
    python3 -m sglang.launch_server "$@"
}

print_config() {
    echo "SGLang server configuration:"
    echo "  model path: $MODEL_PATH"
    echo "  served model name: $SERVED_MODEL_NAME"
    echo "  host: $HOST"
    echo "  port: $MODEL_PORT"
    if [ "$TP" -gt 1 ] || [ "$NCCL_PORT_SET" -eq 1 ]; then
        echo "  nccl port: $NCCL_PORT"
    else
        echo "  nccl port: disabled for tp=1"
    fi
    echo "  dist init addr: ${DIST_INIT_ADDR:-unset}"
    echo "  tp: $TP"
    echo "  mem fraction static: $MEM_FRACTION_STATIC"
    echo "  attention backend: $ATTENTION_BACKEND"
    echo "  reasoning parser: ${REASONING_PARSER:-disabled}"
    echo "  flashinfer allreduce fusion: $ENABLE_FLASHINFER_ALLREDUCE_FUSION"
    echo "  disable cuda graph: $DISABLE_CUDA_GRAPH"
    echo "  cuda graph max batch size: $CUDA_GRAPH_MAX_BS"
    echo "  piecewise cuda graph max tokens: $PIECEWISE_CUDA_GRAPH_MAX_TOKENS"
    echo "  disable radix cache: $DISABLE_RADIX_CACHE"
    echo "  disable overlap schedule: $DISABLE_OVERLAP_SCHEDULE"
    echo "  context length: ${CONTEXT_LENGTH:-inferred from model config}"
    echo "  allow auto truncate: $ALLOW_AUTO_TRUNCATE"
    echo "  auto restart: $AUTO_RESTART"
}

print_command() {
    printf 'Command: python3 -m sglang.launch_server'
    printf ' %q' "${LAUNCH_ARGS[@]}"
    printf '\n'
}

print_config
if [ "$DRY_RUN" -eq 1 ]; then
    print_command
    exit 0
fi

mkdir -p "$LOG_DIR"
LOCAL_IP=$(hostname -I 2>/dev/null | awk '{print $1}')
if [ -z "$LOCAL_IP" ]; then
    LOCAL_IP=$(hostname 2>/dev/null || echo "unknown_ip")
fi
SAFE_IP=$(printf '%s' "$LOCAL_IP" | sed 's/[^[:alnum:]._-]/_/g; s/\./_/g')
MODEL_BASENAME=$(basename "$MODEL_PATH")
SAFE_MODEL_BASENAME=$(printf '%s' "$MODEL_BASENAME" | sed 's/[^[:alnum:]._-]/_/g')
[ -n "$SAFE_MODEL_BASENAME" ] || SAFE_MODEL_BASENAME="model"
SAFE_MODEL_BASENAME_SHORT="${SAFE_MODEL_BASENAME:0:48}"
MODEL_PATH_HASH=$(printf '%s' "$MODEL_PATH" | cksum | awk '{print $1}')
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="$LOG_DIR/${SAFE_IP}_${SAFE_MODEL_BASENAME_SHORT}_${MODEL_PATH_HASH}_${TIMESTAMP}.log"

start_server() {
    echo "Starting SGLang model server on port $MODEL_PORT..."
    echo "Log file: $LOG_FILE"
    run_launch_server "${LAUNCH_ARGS[@]}" 2>&1 | tee -a "$LOG_FILE" &
    MODEL_SERVER_PID=$!
}

start_server

if [ "$AUTO_RESTART" -eq 0 ]; then
    trap 'echo "Shutting down..."; kill "$MODEL_SERVER_PID" 2>/dev/null; exit' SIGTERM SIGINT
    wait "$MODEL_SERVER_PID"
    exit $?
fi

monitor_server() {
    while true; do
        if ! kill -0 "$MODEL_SERVER_PID" 2>/dev/null; then
            echo "Model server exited, restarting..."
            start_server
            sleep 10
        fi
        sleep 30
    done
}

sleep 10
monitor_server &
MONITOR_PID=$!

trap 'echo "Shutting down..."; kill "$MODEL_SERVER_PID" "$MONITOR_PID" 2>/dev/null; exit' SIGTERM SIGINT

wait
