#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_RUN_SCRIPT="${SCRIPT_DIR}/run_rm_router_multinode.sh"

if [ ! -f "$BASE_RUN_SCRIPT" ]; then
    echo "Error: Cannot find script $BASE_RUN_SCRIPT"
    exit 1
fi

ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        *)
            ARGS+=("$1")
            shift
            ;;
    esac
done

has_flag() {
    local flag="$1"
    local arg
    for arg in "${ARGS[@]}"; do
        if [ "$arg" = "$flag" ]; then
            return 0
        fi
    done
    return 1
}

get_flag_value() {
    local flag="$1"
    local i
    for ((i = 0; i < ${#ARGS[@]}; i++)); do
        if [ "${ARGS[$i]}" = "$flag" ]; then
            if [ $((i + 1)) -lt ${#ARGS[@]} ]; then
                echo "${ARGS[$((i + 1))]}"
            fi
            return 0
        fi
    done
    return 1
}

NNODES_VAL="$(get_flag_value --nnodes || true)"
if [ -z "$NNODES_VAL" ]; then
    NNODES_VAL=1
fi

if [ "$NNODES_VAL" -gt 1 ]; then
    if ! has_flag --node-rank; then
        if [ -z "${NODE_RANK:-}" ]; then
            echo "Error: NODE_RANK is empty in multi-node mode"
            exit 1
        fi
        ARGS+=(--node-rank "$NODE_RANK")
    fi

    if ! has_flag --master-addr; then
        if [ -z "${MASTER_ADDR:-}" ]; then
            echo "Error: MASTER_ADDR is empty in multi-node mode"
            exit 1
        fi
        ARGS+=(--master-addr "$MASTER_ADDR")
    fi
fi

exec bash "$BASE_RUN_SCRIPT" "${ARGS[@]}"
