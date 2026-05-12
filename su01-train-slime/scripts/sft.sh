#!/bin/bash
export PARTITION=${GROUP}

cd /root/slime
pip install -e . --no-deps --no-index --disable-pip-version-check --no-build-isolation
# kill existing before rerun
pkill -9 sglang
sleep 3
ray stop --force
pkill -9 ray
pkill -9 python
sleep 3
pkill -9 ray
pkill -9 python



# Multi-node environment (defaults for single-node if not provided)
export RANK=${NODE_RANK:-0}
export NODE_COUNT=${KUBEBRAIN_REPLICA_TOTAL:-1}
export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
export PROC_PER_NODE=${PROC_PER_NODE:-8}
TP_SIZE=2
PP_SIZE=1
CP_SIZE=2
EP_SIZE=8
ETP_SIZE=1
MAX_LEN=$((1024 * 36))
MAX_TOKENS_PER_GPU=$((($MAX_LEN / $CP_SIZE) + 1024))

export WANDB_DIR="/root/slime/wandb"


if [ -z "$RANK" ]; then
  echo "RANK not set. Please set RANK=0 for master, RANK=1,2,... for workers"
  exit 1
fi

TAG=V2_reverse_16k_fix
EXP_NAME="qwen3-30b-thinking-sft-${TAG}-$(date "+%m%d-%H%M%S")"

SHARED_DIR="/root/slime"
READY_FLAG_FILE="$SHARED_DIR/ray_head_ready_30B"
IMO_PATH="/root/slime/data/test_jsonl"

RESUME_PATH="/root/models/Qwen3-30B-A3B-Thinking-2507_slime/resume"
EXP_NAME="qwen3-30b-thinking-sft-${TAG}-$(date "+%m%d-%H%M%S")"
SAVE_PATH="/root/slime/models/P1-30B-A3B/${EXP_NAME}"
# will prevent ray from buffering stdout/stderr
export PYTHONBUFFERED=16

NVLINK_COUNT=$(nvidia-smi | grep -o "NVLink" | wc -l)
if [ "$NVLINK_COUNT" -gt 0 ]; then
    HAS_NVLINK=1
else
    HAS_NVLINK=0
fi
echo "HAS_NVLINK: $HAS_NVLINK (detected $NVLINK_COUNT NVLink references)"

HAS_NVLINK=1

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
source "${SHARED_DIR}/scripts/models/qwen3-30B-A3B-2507.sh"

CKPT_ARGS=(
   --hf-checkpoint ${SHARED_DIR}/models/Qwen3-30B-A3B-Thinking-2507
   --ref-load ${SHARED_DIR}/models/Qwen3-30B-A3B-Thinking-2507_torch_dist_v2
   --save ${SAVE_PATH}
   --load ${RESUME_PATH}
   --save-interval 12800
)

SFT_ARGS=(
   --rollout-function-path slime.rollout.sft_rollout.generate_rollout
   --prompt-data /root/slime/data/sft-data.jsonl
   --input-key messages
  #  --apply-chat-template
  #  --max-rollout-context-len 9999
   --rollout-max-prompt-len 9999
   --rollout-max-response-len 99999
  #  --rollout-shuffle
   --num-epoch 4
   --rollout-batch-size 128
   --global-batch-size 128
   --loss-type sft_loss
   --calculate-per-token-loss
   --disable-compute-advantages-and-returns
   --debug-train-only
   --finetune
)

PERF_ARGS=(
   --tensor-model-parallel-size $TP_SIZE
   --sequence-parallel
   --pipeline-model-parallel-size $PP_SIZE
   --context-parallel-size $CP_SIZE
   --expert-model-parallel-size $EP_SIZE
   --expert-tensor-parallel-size $ETP_SIZE

   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1

   # --micro-batch-size 1
   --use-dynamic-batch-size
   --max-tokens-per-gpu 67000 # 9216
)


# ROLLOUT_ARGS=(
#    --num-rollout 3000
#    --n-samples-per-prompt 1
#    --rollout-max-response-len $MAX_LEN
# ) # for compatibility only


OPTIMIZER_ARGS=(
   --optimizer adam
  #  --lr 5e-6
  #  --lr-decay-style constant
   --lr 1e-5
   --lr-decay-style cosine
   --min-lr 1e-6
  #  --lr-decay-steps 100
   --lr-warmup-fraction 0.1
   --weight-decay 0.1
   --adam-beta1 0.9
   --adam-beta2 0.95

   --optimizer-cpu-offload
   --overlap-cpu-optimizer-d2h-h2d
   --use-precision-aware-optimizer
)

WANDB_ARGS=(
   --use-wandb
   --wandb-project IMO
   --wandb-group ${EXP_NAME}
   --wandb-key ""
   --wandb-dir ${WANDB_DIR}
   --wandb-mode offline
)
# WANDB_ARGS=(
# )



N_SAMPLES_PER_EVAL=(4 4 4 4 4 4)
EVAL_ARGS=(
   --eval-interval 20
  #  --eval-before-train
   --eval-use-xverify
   --eval-group
   --train-use-xverify 
   --eval-prompt-data apex_2025 $IMO_PATH/apex_2025.jsonl amo $IMO_PATH/amobench.jsonl answerbench $IMO_PATH/answerbench.jsonl beyondaime $IMO_PATH/beyondaime.jsonl aime24 $IMO_PATH/aime_2024.jsonl aime25 $IMO_PATH/aime_2025.jsonl 
   --n-samples-per-eval-prompt ${N_SAMPLES_PER_EVAL[@]}
   --eval-max-response-len 81920
   --eval-top-p 0.95
   --eval-temperature 0.6
)



MISC_ARGS=(
   # default dropout in megatron is 0.1
   --attention-dropout 0.0
   --hidden-dropout 0.0
   # should be good for model performance
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   # need to comment this when using model with MLA
   --attention-backend flash
)

# Build the runtime environment JSON with proper variable substitution
RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"/root/Megatron-LM/\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"NCCL_NVLS_ENABLE\": \"${HAS_NVLINK}\",
    \"MASTER_ADDR\": \"${MASTER_ADDR}\",
    \"PYTORCH_CUDA_ALLOC_CONF\": \"expandable_segments:True\"
  }
}"

# ========= 启动 Ray =========
if [ "$RANK" == "0" ]; then
  if [ -f "$READY_FLAG_FILE" ]; then
    rm -f "$READY_FLAG_FILE"
  fi
  echo "[RANK 0] Starting Ray Head node..."
  ray start --head --port=6379 --node-ip-address=$MASTER_ADDR --num-gpus=8 --disable-usage-stats
  echo "[RANK 0] Ray Head started successfully."
  touch "$READY_FLAG_FILE"
else
  echo "[RANK $RANK] Waiting for Ray Head to be ready..."
  sleep 10

  MAX_WAIT=120
  elapsed=0
  while [ ! -f "$READY_FLAG_FILE" ] && [ $elapsed -lt $MAX_WAIT ]; do
    echo "  ⏳ Still waiting... ($elapsed/$MAX_WAIT)"
    sleep 2
    elapsed=$((elapsed + 2))
  done

  if [ ! -f "$READY_FLAG_FILE" ]; then
    echo "❌ Timed out waiting for Ray Head to be ready."
    exit 1
  fi

  WORKER_IP=$(hostname -I | awk '{print $1}')

  echo "[RANK $RANK] Detected Ray Head at $MASTER_ADDR, starting worker at $WORKER_IP..."
  ray start --address=$MASTER_ADDR:6379 --node-ip-address=$WORKER_IP --num-gpus=8 --disable-usage-stats --block
  echo "[RANK $RANK] Worker started successfully."
fi

wait
# --colocate \ --rollout-num-gpus 48 \
if [ "$RANK" == "0" ]; then
   ray job submit --address="http://127.0.0.1:8265" \
      --runtime-env-json="${RUNTIME_ENV_JSON}" \
      -- python3 train_async.py \
      --actor-num-nodes ${NODE_COUNT} \
      --actor-num-gpus-per-node 8 \
      ${MODEL_ARGS[@]} \
      ${CKPT_ARGS[@]} \
      ${SFT_ARGS[@]} \
      ${ROLLOUT_ARGS[@]} \
      ${OPTIMIZER_ARGS[@]} \
      ${WANDB_ARGS[@]} \
      ${PERF_ARGS[@]} \
      ${EVAL_ARGS[@]} \
      ${MISC_ARGS[@]}
fi
