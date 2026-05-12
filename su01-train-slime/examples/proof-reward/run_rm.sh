#!/bin/bash

# Default port settings
DEFAULT_MODEL_PORT=34882
DEFAULT_REWARD_PORT=8001

# Parse command line arguments
MODEL_PORT=$DEFAULT_MODEL_PORT
REWARD_PORT=$DEFAULT_REWARD_PORT

# Show help information
show_help() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  -m, --model-port PORT    Specify 30B model server port (default: $DEFAULT_MODEL_PORT)"
    echo "  -r, --reward-port PORT   Specify reward model server port (default: $DEFAULT_REWARD_PORT)"
    echo "  -h, --help               Show this help information"
    echo ""
    echo "Examples:"
    echo "  $0                                    # Use default ports"
    echo "  $0 -m 34883 -r 8002                  # Specify custom ports"
    echo "  $0 --model-port 34883 --reward-port 8002"
    echo ""
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -m|--model-port)
            MODEL_PORT="$2"
            shift 2
            ;;
        -r|--reward-port)
            REWARD_PORT="$2"
            shift 2
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        *)
            echo "Unknown parameter: $1"
            show_help
            exit 1
            ;;
    esac
done

# Check port parameters
if ! [[ "$MODEL_PORT" =~ ^[0-9]+$ ]] || [ "$MODEL_PORT" -lt 1024 ] || [ "$MODEL_PORT" -gt 65535 ]; then
    echo "Error: Model port must be a number between 1024 and 65535"
    exit 1
fi

if ! [[ "$REWARD_PORT" =~ ^[0-9]+$ ]] || [ "$REWARD_PORT" -lt 1024 ] || [ "$REWARD_PORT" -gt 65535 ]; then
    echo "Error: Reward port must be a number between 1024 and 65535"
    exit 1
fi

if [ "$MODEL_PORT" -eq "$REWARD_PORT" ]; then
    echo "Error: Model port and reward port cannot be the same"
    exit 1
fi

echo "Configuration information:"
echo "  30B model server port: $MODEL_PORT"
echo "  Reward model server port: $REWARD_PORT"
echo ""

cd /root/slime/examples/proof-reward

echo "Installing dependencies..."
pip install math_verify
pip install fastapi uvicorn torch

echo "========== Running unit tests... =========="
python unit_test_sympy_verify.py
echo "========== Unit tests completed =========="

# Function to start gpt-oss-120b model server
start_model_server() {
    echo "Starting gpt-oss-120b model server with multi-GPU support on port $MODEL_PORT..."
    python launch_model.py --port $MODEL_PORT --model-path /root/models/gpt-oss-120b --host 0.0.0.0 --mem-fraction-static 0.8 --tensor-parallel-size 2 --max-concurrent-requests 64 --gpu-memory-utilization 0.9 &
    MODEL_SERVER_PID=$!
}

# Function to start reward model server
start_reward_server() {
    echo "Starting reward model server on port $REWARD_PORT..."
    echo "Using model server port: $MODEL_PORT"
    MODEL_PORT=$MODEL_PORT uvicorn reward_model_server:app --host 0.0.0.0 --port $REWARD_PORT --timeout-keep-alive 30 --log-level info --workers 32
}

# Check sglang HTTP health
check_model_healthy() {
    local port=$1
    local timeout=${2:-10}
    local http_code
    http_code=$(curl -s -o /dev/null -w "%{http_code}" --max-time $timeout "http://localhost:$port/health" 2>/dev/null)
    if [ "$http_code" = "200" ]; then
        return 0
    fi
    http_code=$(curl -s -o /dev/null -w "%{http_code}" --max-time $timeout "http://localhost:$port/v1/models" 2>/dev/null)
    if [ "$http_code" = "200" ]; then
        return 0
    fi
    return 1
}

# Completely clean up sglang related processes and GPU memory
kill_model_server() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Cleaning up model server processes..."
    pkill -f "launch_model.py.*--port.*$MODEL_PORT" 2>/dev/null
    pkill -f "sglang.launch_server.*--port.*$MODEL_PORT" 2>/dev/null
    pkill -f "sglang.*--port.*$MODEL_PORT" 2>/dev/null
    sleep 3
    # If it is not dead, send SIGKILL
    pkill -9 -f "launch_model.py.*--port.*$MODEL_PORT" 2>/dev/null
    pkill -9 -f "sglang.launch_server.*--port.*$MODEL_PORT" 2>/dev/null
    pkill -9 -f "sglang.*--port.*$MODEL_PORT" 2>/dev/null
    sleep 2
    # Try to release GPU memory (clean up possible residual CUDA processes)
    if command -v nvidia-smi >/dev/null 2>&1; then
        local zombie_pids
        zombie_pids=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | sort -u)
        for zpid in $zombie_pids; do
            if ! kill -0 "$zpid" 2>/dev/null; then
                echo "[$(date '+%Y-%m-%d %H:%M:%S')] Found zombie GPU process $zpid, force killing..."
                kill -9 "$zpid" 2>/dev/null
            fi
        done
    fi
    # Wait for processes to really exit
    local wait_count=0
    while pgrep -f "sglang.*--port.*$MODEL_PORT" >/dev/null 2>&1 && [ $wait_count -lt 15 ]; do
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] Waiting for sglang processes to exit..."
        sleep 2
        wait_count=$((wait_count + 1))
    done
}

# Wait for model server to start (HTTP health check)
wait_for_model_server() {
    local port=$1
    local max_attempts=120  # Maximum wait time is 20 minutes
    local attempt=0

    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Waiting for model server on port $port..."
    while [ $attempt -lt $max_attempts ]; do
        if check_model_healthy $port 15; then
            echo "[$(date '+%Y-%m-%d %H:%M:%S')] Model server is healthy on port $port"
            return 0
        fi
        sleep 10
        attempt=$((attempt + 1))
        if [ $((attempt % 6)) -eq 0 ]; then
            echo "[$(date '+%Y-%m-%d %H:%M:%S')] Still waiting for model server... ($attempt/$max_attempts)"
        fi
    done

    echo "[$(date '+%Y-%m-%d %H:%M:%S')] ERROR: Timeout waiting for model server to start"
    return 1
}

MODEL_RESTART_COUNT=0
MODEL_CONSECUTIVE_FAIL=0
MAX_CONSECUTIVE_FAIL=3

# Monitor function
monitor_server() {
    while true; do
        # === Check reward server ===
        if ! curl -s --max-time 10 http://localhost:$REWARD_PORT/health > /dev/null 2>&1; then
            echo "[$(date '+%Y-%m-%d %H:%M:%S')] Reward server is not responding, restarting..."
            pkill -f "uvicorn reward_model_server" 2>/dev/null
            sleep 2
            pkill -9 -f "uvicorn reward_model_server" 2>/dev/null
            sleep 1
            start_reward_server &
            REWARD_SERVER_PID=$!
        else
            echo "[$(date '+%Y-%m-%d %H:%M:%S')] Reward server is healthy"
        fi

        # === Check sglang model server (HTTP health check) ===
        local model_ok=false
        if check_model_healthy $MODEL_PORT 15; then
            model_ok=true
        else
            echo "[$(date '+%Y-%m-%d %H:%M:%S')] WARNING: Model server HTTP health check failed, retrying..."
            sleep 10
            if check_model_healthy $MODEL_PORT 30; then
                model_ok=true
                echo "[$(date '+%Y-%m-%d %H:%M:%S')] Model server recovered after retry"
            fi
        fi

        if $model_ok; then
            echo "[$(date '+%Y-%m-%d %H:%M:%S')] Model server is healthy (restarts so far: $MODEL_RESTART_COUNT)"
            MODEL_CONSECUTIVE_FAIL=0
        else
            MODEL_CONSECUTIVE_FAIL=$((MODEL_CONSECUTIVE_FAIL + 1))
            MODEL_RESTART_COUNT=$((MODEL_RESTART_COUNT + 1))
            echo "[$(date '+%Y-%m-%d %H:%M:%S')] Model server is DOWN! (consecutive failures: $MODEL_CONSECUTIVE_FAIL, total restarts: $MODEL_RESTART_COUNT)"

            if [ $MODEL_CONSECUTIVE_FAIL -ge $MAX_CONSECUTIVE_FAIL ]; then
                local backoff=$((MODEL_CONSECUTIVE_FAIL * 30))
                if [ $backoff -gt 300 ]; then
                    backoff=300
                fi
                echo "[$(date '+%Y-%m-%d %H:%M:%S')] Too many consecutive failures, backing off ${backoff}s before restart..."
                sleep $backoff
            fi

            kill_model_server
            start_model_server
            if wait_for_model_server $MODEL_PORT; then
                echo "[$(date '+%Y-%m-%d %H:%M:%S')] Model server restarted successfully (restart #$MODEL_RESTART_COUNT)"
                # Restart reward server after successful restart, because it may have cached connections
                echo "[$(date '+%Y-%m-%d %H:%M:%S')] Restarting reward server to refresh connections..."
                pkill -f "uvicorn reward_model_server" 2>/dev/null
                sleep 2
                start_reward_server &
                REWARD_SERVER_PID=$!
            else
                echo "[$(date '+%Y-%m-%d %H:%M:%S')] ERROR: Model server failed to restart, will retry next cycle"
            fi
        fi

        sleep 30
    done
}

# Start gpt-oss-120b model server
start_model_server

# Wait for model server to start
wait_for_model_server $MODEL_PORT


# Start reward server
start_reward_server &
REWARD_SERVER_PID=$!

# Wait for reward server to start, avoid monitor killing it immediately
echo "Waiting for reward server to initialize..."
sleep 10

# Start monitor
monitor_server &
MONITOR_PID=$!

# Wait for signal
trap 'echo "Shutting down servers..."; kill $REWARD_SERVER_PID $MODEL_SERVER_PID $MONITOR_PID 2>/dev/null; exit' SIGTERM SIGINT

# Wait for all processes
wait
