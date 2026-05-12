#!/bin/bash
cat "$0" 
export PARTITION=${GROUP}

export AWS_ACCESS_KEY_ID=""     #replace with your S3 access-key
export AWS_SECRET_ACCESS_KEY="" #replace with your S3 secret-key


cd /root/slime
pip install -e . --no-deps --no-index --disable-pip-version-check --no-build-isolation
pip install math_verify

export WANDB_MODE="offline"
export WANDB_KEY=""
export WANDB_DIR="/root/slime/wandb"
mkdir -p $WANDB_DIR

EXP_NAME="coarse-rl-$(date "+%m%d-%H%M%S")"

# Multi-node environment (defaults for single-node if not provided)
export RANK=${NODE_RANK:-0}
export NODE_COUNT=${KUBEBRAIN_REPLICA_TOTAL:-1}
export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
export PROC_PER_NODE=${PROC_PER_NODE:-8}

if [ -z "$RANK" ]; then
  echo "RANK not set. Please set RANK=0 for master, RANK=1,2,... for workers"
  exit 1
fi

SHARED_DIR="/root/slime"
READY_FLAG_FILE="$SHARED_DIR/ray_head_ready_30B"

# will prevent ray from buffering stdout/stderr
export PYTHONBUFFERED=16
# export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:128,expandable_segments:False"

NVLINK_COUNT=$(nvidia-smi | grep -o "NVLink" | wc -l)
if [ "$NVLINK_COUNT" -gt 0 ]; then
    HAS_NVLINK=1
else
    HAS_NVLINK=0
fi
echo "HAS_NVLINK: $HAS_NVLINK (detected $NVLINK_COUNT NVLink references)"

HAS_NVLINK=1

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
source "${SCRIPT_DIR}/models/qwen3-30B-A3B-2507.sh"

TP_SIZE=4
PP_SIZE=1
CP_SIZE=4
EP_SIZE=8
ETP_SIZE=1
MAX_LEN=$((1024 * 128))
MAX_TOKENS_PER_GPU=$((($MAX_LEN / $CP_SIZE) + 1024))
ROLLOUT_BATCH_SIZE=128
N_SAMPLES_PER_PROMPT=8
IMO_PATH=/root/slime/data/test_jsonl
TRAIN_DATA_PATH=/root/slime/data/rl_data
LOG_DIR=$SHARED_DIR/logs

RESUME_PATH=/root/slime/ckpt/Qwen3-30B-A3B-Thinking-2507_slime/coarse-rl

CKPT_ARGS=(
   --hf-checkpoint /root/slime/models/Qwen3-30B-A3B-Thinking-2507
   --ref-load /root/slime/models/Qwen3-30B-A3B-Thinking-2507_torch_dist_v2
   --load ${RESUME_PATH}
   --save $SHARED_DIR/ckpt/Qwen3-30B-A3B-Thinking-2507_slime/${EXP_NAME}
   --save-interval 16
)

ROLLOUT_ARGS=(
  --prompt-data aops.verify $TRAIN_DATA_PATH/aops.verify.jsonl book1224.verify $TRAIN_DATA_PATH/book-1224.verify.jsonl jc.verify $TRAIN_DATA_PATH/jc.verify.jsonl  skywork.verify $TRAIN_DATA_PATH/skywork.verify.jsonl sxzm.verify $TRAIN_DATA_PATH/sxzm.verify.jsonl physics.verify $TRAIN_DATA_PATH/physics.verify.jsonl
   --input-key prompt
   --label-key label
   --apply-chat-template
   --rollout-shuffle
   --shuffle-mode interleaved
   # --rm-type remote_rm_proof
   --rm-type remote_rm
   --physics-dataset-names physics
   --eval-rm-url "http://10.102.97.56:8001" 
   --rm-url "http://10.102.97.56:8001"
   --reward-key score
   --num-rollout 256
   --rollout-batch-size $ROLLOUT_BATCH_SIZE
   --n-samples-per-prompt $N_SAMPLES_PER_PROMPT
   --rollout-max-response-len $MAX_LEN
   --rollout-temperature 1.0
   # --global-batch-size 256
   # --use-token-output
   --num-steps-per-rollout 4
   --use-tis
   # --use-rollout-is
   --partial-rollout
   --over-sampling-batch-size $((ROLLOUT_BATCH_SIZE * 4))
   --dynamic-sampling-filter-path slime.rollout.filter_hub.dynamic_sampling_filters.check_reward_nonzero_std
   --balance-data
   --finetune
   --replay-filtering
)

# Number of samples per eval prompt for each dataset (in order of eval-prompt-data)
N_SAMPLES_PER_EVAL=(8 8 1 8 8 4 8 8)

EVAL_ARGS=(
   --eval-interval 8
   # --eval-before-train 
   --skip-eval-before-train
   --eval-use-xverify
   --eval-group
   --train-use-xverify
   --eval-prompt-data apex_2025 $IMO_PATH/apex_2025.jsonl amo $IMO_PATH/amobench.jsonl answerbench $IMO_PATH/answerbench.jsonl beyondaime $IMO_PATH/beyondaime.jsonl frontierscience $IMO_PATH/frontierscience_olympiad_physics.jsonl proofbench $IMO_PATH/proofbench.jsonl ipho24 $IMO_PATH/IPhO_2024.jsonl ipho25 $IMO_PATH/IPhO_2025.jsonl
   --n-samples-per-eval-prompt ${N_SAMPLES_PER_EVAL[@]}
   --eval-max-response-len 131072 #81920
   --eval-top-p 0.95
   --eval-temperature 1.0
)

PERF_ARGS=(
   --tensor-model-parallel-size $TP_SIZE
   --sequence-parallel
   --pipeline-model-parallel-size $PP_SIZE
   --context-parallel-size $CP_SIZE
   --expert-model-parallel-size $EP_SIZE
   --expert-tensor-parallel-size $ETP_SIZE
   # --overlap-param-gather
   # --overlap-grad-reduce
   # --tp-comm-overlap
   # --moe-grouped-gemm
   # --moe-token-dispatcher-type alltoall
   --moe-freeze-router
   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1
   # --micro-batch-size 1
   --use-dynamic-batch-size
   --max-tokens-per-gpu $MAX_TOKENS_PER_GPU

   # use deepep for megatron
   --moe-enable-deepep
   --moe-token-dispatcher-type flex

   # # fp8
   # --transformer-impl transformer_engine
   # --bf16
   # --fp8-format e4m3
   # --fp8-recipe blockwise

   --train-memory-margin-bytes $((1024**3))
)

GRPO_ARGS=(
   --advantage-estimator gspo
   # --use-kl-loss
   --kl-loss-coef 0.00
   --kl-loss-type low_var_kl
   --entropy-coef 0.00
   --eps-clip 1e-3
   --eps-clip-high 1e-3
)

OPTIMIZER_ARGS=(
   --optimizer adam
   --lr 1e-6
   --lr-decay-style constant
   --weight-decay 0.1
   --adam-beta1 0.9
   --adam-beta2 0.98

   --optimizer-cpu-offload
   --overlap-cpu-optimizer-d2h-h2d
   --use-precision-aware-optimizer
)

WANDB_ARGS=(
   --use-wandb
   --wandb-project slime-Qwen3-30B-A3B
   --wandb-group ${EXP_NAME}
   --wandb-key ${WANDB_KEY}
   --wandb-dir ${WANDB_DIR}
   --wandb-mode offline
)

SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 1
   --sglang-mem-fraction-static 0.8
   # --sglang-enable-ep-moe
   --sglang-expert-parallel-size 1
   # --sglang-kv-cache-dtype fp8_e4m3
   --sglang-moe-runner-backend deep_gemm
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
   --log-file-path ${LOG_DIR}
   # --train-env-vars '{"PYTORCH_CUDA_ALLOC_CONF":"expandable_segments:True"}'
)
#\"PYTORCH_CUDA_ALLOC_CONF\": \"max_split_size_mb:128,expandable_segments:False\",
# Build the runtime environment JSON with proper variable substitution
RUNTIME_ENV_JSON="{
    \"env_vars\": {
      \"PYTHONPATH\": \"/root/Megatron-LM/\",
      \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
      \"NCCL_NVLS_ENABLE\": \"${HAS_NVLINK}\",
      \"MASTER_ADDR\": \"${MASTER_ADDR}\",
      \"NVTE_FP8_BLOCK_SCALING_FP32_SCALES\": \"1\",
      \"NCCL_TIMEOUT_MS\":\"36000000\",
      \"PYTORCH_ALLOC_CONF\": \"max_split_size_mb:2048\"
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
     -- python3 train.py \
     --actor-num-nodes ${NODE_COUNT} \
     --actor-num-gpus-per-node 8 \
     --no-load-optim \
     --no-save-optim \
     --colocate \
     ${MODEL_ARGS[@]} \
     ${CKPT_ARGS[@]} \
     ${ROLLOUT_ARGS[@]} \
     ${OPTIMIZER_ARGS[@]} \
     ${GRPO_ARGS[@]} \
     ${WANDB_ARGS[@]} \
     ${PERF_ARGS[@]} \
     ${EVAL_ARGS[@]} \
     ${SGLANG_ARGS[@]} \
     ${MISC_ARGS[@]} 2>&1 | tee ${WANDB_DIR}/${EXP_NAME}.log
fi

