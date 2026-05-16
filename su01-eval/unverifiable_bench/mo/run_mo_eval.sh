#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

openai_client_base_url() {
  local text="${1:-}"
  text="${text%/}"
  if [[ "$text" == */chat/completions ]]; then
    text="${text%/chat/completions}"
  elif [[ "$text" == */completions ]]; then
    text="${text%/completions}"
  fi
  echo "$text"
}

PYTHON_BIN="${PYTHON_BIN:-python}"
OUTPUT_ROOT="${OUTPUT_ROOT:-results/mo}"
TASK_NAME="${TASK_NAME:-mo}"

DATA_PATH="${DATA_PATH:-}"
RESPONSE_DIR="${RESPONSE_DIR:-}"
META_JSONL="${META_JSONL:-}"
PREPARED_PATH="${PREPARED_PATH:-$OUTPUT_ROOT/prepared/${TASK_NAME}.jsonl}"
RESPONSE_PATTERN="${RESPONSE_PATTERN:-}"
if [[ -z "$RESPONSE_PATTERN" ]]; then
  RESPONSE_PATTERN="{idx}_out.txt"
fi
QUESTION_CYCLE="${QUESTION_CYCLE:-6}"
START_INDEX="${START_INDEX:-}"
END_INDEX="${END_INDEX:-}"
GUIDELINE_MD="${GUIDELINE_MD:-}"

JUDGE_MODEL="${JUDGE_MODEL:-gemini-2.5-pro}"
BASE_URL="${BASE_URL:-${MO_BASE_URL:-${OPENAI_BASE_URL:-http://localhost:8000/v1}}}"
BASE_URL="$(openai_client_base_url "$BASE_URL")"
API_KEY="${API_KEY:-${MO_API_KEY:-${OPENAI_API_KEY:-${OPENAI_API_TOKEN:-}}}}"
CONCURRENT="${CONCURRENT:-0}"
TEXT_ONLY="${TEXT_ONLY:-0}"
PRINT_FIRST_PROMPT="${PRINT_FIRST_PROMPT:-0}"
RUN_SUFFIX="${RUN_SUFFIX:-}"

NORMALIZE_POINTS="${NORMALIZE_POINTS:-1}"
PARSER_MODEL="${PARSER_MODEL:-gemini-2.5-flash}"
PARSER_CONCURRENT="${PARSER_CONCURRENT:-0}"
PARSER_MAX_WORKERS="${PARSER_MAX_WORKERS:-32}"
DRY_RUN="${DRY_RUN:-0}"

if [[ -z "$DATA_PATH" ]]; then
  if [[ -z "$RESPONSE_DIR" ]]; then
    echo "Error: RESPONSE_DIR is required when DATA_PATH is not set." >&2
    exit 1
  fi
  if [[ -z "$META_JSONL" ]]; then
    echo "Error: META_JSONL is required when DATA_PATH is not set." >&2
    exit 1
  fi

  prepare_cmd=(
    "$PYTHON_BIN" "$SCRIPT_DIR/prepare_mo_eval.py"
    --response-dir "$RESPONSE_DIR"
    --meta-jsonl "$META_JSONL"
    --output-path "$PREPARED_PATH"
    --response-pattern "$RESPONSE_PATTERN"
    --question-cycle "$QUESTION_CYCLE"
    --name "$TASK_NAME"
  )
  if [[ -n "$START_INDEX" ]]; then
    prepare_cmd+=(--start-index "$START_INDEX")
  fi
  if [[ -n "$END_INDEX" ]]; then
    prepare_cmd+=(--end-index "$END_INDEX")
  fi
  if [[ -n "$GUIDELINE_MD" ]]; then
    prepare_cmd+=(--guideline-md "$GUIDELINE_MD")
  fi

  echo "[INFO] Preparing MO eval input: $PREPARED_PATH"
  "${prepare_cmd[@]}"
  DATA_PATH="$PREPARED_PATH"
fi

if [[ -z "$API_KEY" && "$DRY_RUN" != "1" ]]; then
  echo "Error: API_KEY is required. Set API_KEY, MO_API_KEY, OPENAI_API_KEY, or OPENAI_API_TOKEN." >&2
  exit 1
fi

JUDGE_OUTPUT_DIR="$OUTPUT_ROOT/judge"
if [[ "$DRY_RUN" == "1" ]]; then
  echo "[DRY-RUN] DATA_PATH=$DATA_PATH"
  echo "[DRY-RUN] JUDGE_OUTPUT_DIR=$JUDGE_OUTPUT_DIR"
  echo "[DRY-RUN] BASE_URL=$BASE_URL"
  echo "[DRY-RUN] JUDGE_MODEL=$JUDGE_MODEL"
  echo "[DRY-RUN] NORMALIZE_POINTS=$NORMALIZE_POINTS"
  exit 0
fi

eval_cmd=(
  "$PYTHON_BIN" "$SCRIPT_DIR/eval_mo.py"
  --data-path "$DATA_PATH"
  --output-dir "$JUDGE_OUTPUT_DIR"
  --api-key "$API_KEY"
  --base-url "$BASE_URL"
  --model-name "$JUDGE_MODEL"
)
if [[ "$CONCURRENT" == "1" ]]; then
  eval_cmd+=(--concurrent)
fi
if [[ "$TEXT_ONLY" == "1" ]]; then
  eval_cmd+=(--text_only)
fi
if [[ "$PRINT_FIRST_PROMPT" == "1" ]]; then
  eval_cmd+=(--print_first_prompt)
fi
if [[ -n "$RUN_SUFFIX" ]]; then
  eval_cmd+=(--run_suffix "$RUN_SUFFIX")
fi

echo "[INFO] Running MO judge."
"${eval_cmd[@]}"

data_base="$(basename "$DATA_PATH")"
competition_name="${data_base%.*}"
judge_model_name="${JUDGE_MODEL##*/}"
suffix=""
if [[ -n "$RUN_SUFFIX" ]]; then
  suffix="-$RUN_SUFFIX"
fi
RESULT_PATH="$JUDGE_OUTPUT_DIR/$competition_name/$judge_model_name/$competition_name-$judge_model_name$suffix.json"

if [[ "$NORMALIZE_POINTS" == "1" ]]; then
  normalize_cmd=(
    "$PYTHON_BIN" "$SCRIPT_DIR/normalize_points.py"
    --input-file "$RESULT_PATH"
    --api-key "$API_KEY"
    --base-url "$BASE_URL"
    --model-name "$PARSER_MODEL"
  )
  if [[ "$PARSER_CONCURRENT" == "1" ]]; then
    normalize_cmd+=(--concurrent --max-workers "$PARSER_MAX_WORKERS")
  fi

  echo "[INFO] Normalizing point strings: $RESULT_PATH"
  "${normalize_cmd[@]}"
fi

echo "[DONE] MO eval result: $RESULT_PATH"
