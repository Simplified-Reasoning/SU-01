#!/bin/bash

DEFAULT_MODEL_PORT=34882
DEFAULT_REWARD_PORT=8001

MODEL_PORT=$DEFAULT_MODEL_PORT
REWARD_PORT=$DEFAULT_REWARD_PORT
TIKTOKEN_ENCODINGS_BASE=/root/slime/examples/rm-gptoss/encodings
show_help() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  -m, --model-port PORT    Specify the 30B model server port (default: $DEFAULT_MODEL_PORT)"
    echo "  -r, --reward-port PORT   Specify the reward model server port (default: $DEFAULT_REWARD_PORT)"
    echo "  -h, --help               Show this help information"
    echo ""
    echo "Examples:"
    echo "  $0                                    # Use default ports"
    echo "  $0 -m 34883 -r 8002                  # Specify custom ports"
    echo "  $0 --model-port 34883 --reward-port 8002"
    echo ""
}

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
        -n|--model-name)
            MODEL_NAME="$2"
            shift 2
            ;;
        -g|--num-gpu)
            NUM_GPU="$2"
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


if [ "$MODEL_NAME" == "Qwen3-30B-A3B-Instruct-2507" ]; then
    MODEL_PATH="/root/Qwen/Qwen3-30B-A3B-Instruct-2507"
elif [ "$MODEL_NAME" == "gpt-oss-120b" ]; then
    MODEL_PATH="/root/Qwen/gpt-oss-120b"
else
    echo "Error: Model name not supported"
    show_help
    exit 1
fi

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

echo "Configuration:"
echo "  30B model server port: $MODEL_PORT"
echo "  Reward model server port: $REWARD_PORT"
echo ""

cd /root/slime/rm-gptoss

echo "Installing dependencies..."
pip install math_verify
pip install fastapi uvicorn torch

echo "Starting $MODEL_NAME model server with multi-GPU support on port $MODEL_PORT..."
python launch_model.py --port $MODEL_PORT --model-path $MODEL_PATH --host 0.0.0.0 --mem-fraction-static 0.8 --expert-parallel-size $NUM_GPU --max-concurrent-requests 200 --gpu-memory-utilization 0.9 &
MODEL_SERVER_PID=$!


sleep 10

start_reward_server() {
    echo "Starting reward model server on port $REWARD_PORT..."
    echo "Using model server port: $MODEL_PORT"
    MODEL_PORT=$MODEL_PORT uvicorn reward_model_server:app --host 0.0.0.0 --port $REWARD_PORT --timeout-keep-alive 30 --log-level info --workers 64
}


monitor_server() {
    while true; do
        if ! curl -s http://localhost:$REWARD_PORT/health > /dev/null 2>&1; then
            echo "Reward server is not responding, restarting..."
            pkill -f "uvicorn reward_model_server"
            sleep 2
            start_reward_server &
            REWARD_SERVER_PID=$!
        else
            echo "Reward server is healthy"
        fi
        
        if ! kill -0 $MODEL_SERVER_PID 2>/dev/null; then
            echo "Model server died, restarting..."
            python launch_model.py --port $MODEL_PORT --model-path $MODEL_PATH --host 0.0.0.0 --mem-fraction-static 0.8 --expert-parallel-size $NUM_GPU --max-concurrent-requests 200 --gpu-memory-utilization 0.9 &
            MODEL_SERVER_PID=$!
            sleep 10
        fi
        
        sleep 30
    done
}

start_reward_server &
REWARD_SERVER_PID=$!

monitor_server &
MONITOR_PID=$!

trap 'echo "Shutting down servers..."; kill $REWARD_SERVER_PID $MODEL_SERVER_PID $MONITOR_PID 2>/dev/null; exit' SIGTERM SIGINT

wait
