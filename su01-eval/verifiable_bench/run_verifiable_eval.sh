#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ANSWER_SCRIPT="$SCRIPT_DIR/answer_verifiable_bench/eval_verifiable_answer.py"
FS_SCRIPT="$SCRIPT_DIR/fs_olympiad/run_frontierscience_eval.py"

PYTHON_BIN="${PYTHON_BIN:-python}"
INPUT_DIR="${INPUT_DIR:-${1:-}}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/tmp/su01-verifiable-eval}"
TASKS="${TASKS:-aime25,aime26,amobench,answerbench,fs_olympiad}"

AIME25_INPUT="${AIME25_INPUT:-${INPUT_DIR:+$INPUT_DIR/aime_2025.json}}"
AIME26_INPUT="${AIME26_INPUT:-${INPUT_DIR:+$INPUT_DIR/aime_2026.json}}"
AMOBENCH_INPUT="${AMOBENCH_INPUT:-${INPUT_DIR:+$INPUT_DIR/amobench.json}}"
ANSWERBENCH_INPUT="${ANSWERBENCH_INPUT:-${INPUT_DIR:+$INPUT_DIR/answerbench.json}}"
FS_OLYMPIAD_INPUT="${FS_OLYMPIAD_INPUT:-${INPUT_DIR:+$INPUT_DIR/frontierscience_olympiad.json}}"

RM_URL="${RM_URL:-}"
RM_MODE="${RM_MODE:-standard}"
CONCURRENCY="${CONCURRENCY:-32}"
RM_REQUEST_TIMEOUT="${RM_REQUEST_TIMEOUT:-0}"
MAX_ITEMS="${MAX_ITEMS:-}"
USE_XVERIFY="${USE_XVERIFY:-0}"
NO_PROGRESS="${NO_PROGRESS:-0}"
DRY_RUN="${DRY_RUN:-0}"

FRONTIER_OFFICIAL_DATA_PATH="${FRONTIER_OFFICIAL_DATA_PATH:-}"
FRONTIER_OUTPUT_ROOT="${FRONTIER_OUTPUT_ROOT:-$OUTPUT_ROOT/fs_olympiad}"
FRONTIER_DRY_RUN="${FRONTIER_DRY_RUN:-0}"
FRONTIER_CONCURRENT="${FRONTIER_CONCURRENT:-1}"
FRONTIER_STREAM="${FRONTIER_STREAM:-1}"
FRONTIER_RESUME="${FRONTIER_RESUME:-1}"
FRONTIER_MAX_WORKERS="${FRONTIER_MAX_WORKERS:-8}"
FRONTIER_MAX_TOKENS="${FRONTIER_MAX_TOKENS:-32768}"
FRONTIER_REQUEST_INTERVAL="${FRONTIER_REQUEST_INTERVAL:-0.0}"
OLYMPIAD_JUDGE_MODEL="${OLYMPIAD_JUDGE_MODEL:-gpt-oss}"
OLYMPIAD_REASONING_EFFORT="${OLYMPIAD_REASONING_EFFORT:-high}"
OLYMPIAD_BASE_URL="${OLYMPIAD_BASE_URL:-}"
OLYMPIAD_API_KEY="${OLYMPIAD_API_KEY:-}"

usage() {
  cat <<'EOF'
Usage:
  INPUT_DIR=/path/to/predictions RM_URL=http://host:port bash run_verifiable_eval.sh

Required:
  INPUT_DIR     Directory containing generated prediction files.
  RM_URL        Reward model server URL for AIME/AMO/AnswerBench tasks.

Common optional env:
  TASKS         Comma-separated tasks. Default: aime25,aime26,amobench,answerbench,fs_olympiad
                Supported: aime25,aime26,amobench,answerbench,fs_olympiad,answer,all
  OUTPUT_ROOT   Output directory. Default: /tmp/su01-verifiable-eval
  MAX_ITEMS     Only score the first N records for answer-verifiable tasks.
  CONCURRENCY   RM request concurrency. Default: 32
  USE_XVERIFY   Set to 1 to pass --use-xverify.
  DRY_RUN       Set to 1 to print commands without running them.

Input file env:
  AIME25_INPUT       Default: $INPUT_DIR/aime_2025.json
  AIME26_INPUT       Default: $INPUT_DIR/aime_2026.json
  AMOBENCH_INPUT     Default: $INPUT_DIR/amobench.json
  ANSWERBENCH_INPUT  Default: $INPUT_DIR/answerbench.json
  FS_OLYMPIAD_INPUT  Default: $INPUT_DIR/frontierscience_olympiad.json

FrontierScience Olympiad optional env:
  FRONTIER_OFFICIAL_DATA_PATH  Required to run fs_olympiad.
  OLYMPIAD_BASE_URL            Judge endpoint base URL.
  OLYMPIAD_API_KEY             Judge API key.
  FRONTIER_DRY_RUN             Set to 1 to pass --dry-run to the FS runner.
EOF
}

to_lower() {
  echo "${1:-}" | tr '[:upper:]' '[:lower:]'
}

truthy() {
  local value
  value="$(to_lower "${1:-}")"
  [[ "$value" == "1" || "$value" == "true" || "$value" == "yes" || "$value" == "y" || "$value" == "on" ]]
}

task_enabled() {
  local needle="$1"
  local normalized
  normalized=",${TASKS// /},"
  [[ "$normalized" == *",all,"* || "$normalized" == *",$needle,"* ]] && return 0
  if [[ "$needle" != "fs_olympiad" && "$normalized" == *",answer,"* ]]; then
    return 0
  fi
  return 1
}

print_command_redacted() {
  local -a redacted=()
  local redact_next=0
  local token
  for token in "$@"; do
    if [[ "$redact_next" == "1" ]]; then
      redacted+=("******")
      redact_next=0
      continue
    fi
    redacted+=("$token")
    case "$token" in
      --api-key|--olympiad-api-key|--research-api-key)
        redact_next=1
        ;;
    esac
  done
  echo "[RUN] ${redacted[*]}"
}

run_command() {
  print_command_redacted "$@"
  if truthy "$DRY_RUN"; then
    return 0
  fi
  "$@"
}

require_input_dir() {
  if [[ -z "$INPUT_DIR" ]]; then
    usage >&2
    echo "" >&2
    echo "Error: INPUT_DIR is required." >&2
    exit 1
  fi
  if [[ ! -d "$INPUT_DIR" ]]; then
    echo "Error: INPUT_DIR not found: $INPUT_DIR" >&2
    exit 1
  fi
}

answer_task_enabled() {
  task_enabled "aime25" || task_enabled "aime26" || task_enabled "amobench" || task_enabled "answerbench"
}

answer_input_for_task() {
  local task="$1"
  case "$task" in
    aime25)
      echo "$AIME25_INPUT"
      ;;
    aime26)
      echo "$AIME26_INPUT"
      ;;
    amobench)
      echo "$AMOBENCH_INPUT"
      ;;
    answerbench)
      echo "$ANSWERBENCH_INPUT"
      ;;
    *)
      return 1
      ;;
  esac
}

run_answer_task() {
  local task="$1"
  local input_file
  input_file="$(answer_input_for_task "$task")"
  if [[ -z "$input_file" || ! -f "$input_file" ]]; then
    echo "[SKIP] $task: prediction file not found: ${input_file:-<empty>}"
    return 0
  fi

  local output_dir="$OUTPUT_ROOT/answer_verifiable_bench"
  local output_file="$output_dir/${task}.remote_eval.json"
  mkdir -p "$output_dir"

  local -a cmd=(
    env PYTHONDONTWRITEBYTECODE=1
    "$PYTHON_BIN"
    "$ANSWER_SCRIPT"
    --input-json "$input_file"
    --output-json "$output_file"
    --rm-url "$RM_URL"
    --rm-mode "$RM_MODE"
    --concurrency "$CONCURRENCY"
    --rm-request-timeout "$RM_REQUEST_TIMEOUT"
  )
  if truthy "$USE_XVERIFY"; then
    cmd+=(--use-xverify)
  fi
  if [[ -n "$MAX_ITEMS" ]]; then
    cmd+=(--max-items "$MAX_ITEMS")
  fi
  if truthy "$NO_PROGRESS"; then
    cmd+=(--no-progress)
  fi

  run_command "${cmd[@]}"
}

run_fs_olympiad() {
  local input_file
  input_file="$FS_OLYMPIAD_INPUT"
  if [[ -z "$input_file" || ! -f "$input_file" ]]; then
    echo "[SKIP] fs_olympiad: prediction file not found: ${input_file:-<empty>}"
    return 0
  fi
  if [[ -z "$FRONTIER_OFFICIAL_DATA_PATH" ]]; then
    echo "[SKIP] fs_olympiad: FRONTIER_OFFICIAL_DATA_PATH is not set"
    return 0
  fi

  mkdir -p "$FRONTIER_OUTPUT_ROOT"
  local -a cmd=(
    env PYTHONDONTWRITEBYTECODE=1
    "$PYTHON_BIN"
    "$FS_SCRIPT"
    --olympiad-prediction-path "$input_file"
    --official-data-path "$FRONTIER_OFFICIAL_DATA_PATH"
    --output-root "$FRONTIER_OUTPUT_ROOT"
    --olympiad-judge-model "$OLYMPIAD_JUDGE_MODEL"
    --olympiad-reasoning-effort "$OLYMPIAD_REASONING_EFFORT"
    --max-workers "$FRONTIER_MAX_WORKERS"
    --max-tokens "$FRONTIER_MAX_TOKENS"
    --request-interval "$FRONTIER_REQUEST_INTERVAL"
  )
  if [[ -n "$OLYMPIAD_BASE_URL" ]]; then
    cmd+=(--olympiad-base-url "$OLYMPIAD_BASE_URL")
  fi
  if [[ -n "$OLYMPIAD_API_KEY" ]]; then
    cmd+=(--olympiad-api-key "$OLYMPIAD_API_KEY")
  fi
  if truthy "$FRONTIER_DRY_RUN"; then
    cmd+=(--dry-run)
  fi
  if truthy "$FRONTIER_CONCURRENT"; then
    cmd+=(--concurrent)
  fi
  if truthy "$FRONTIER_STREAM"; then
    cmd+=(--stream)
  fi
  if truthy "$FRONTIER_RESUME"; then
    cmd+=(--resume)
  fi

  run_command "${cmd[@]}"
}

main() {
  require_input_dir
  if answer_task_enabled && [[ -z "$RM_URL" ]]; then
    usage >&2
    echo "" >&2
    echo "Error: RM_URL is required for answer-verifiable tasks." >&2
    exit 1
  fi

  mkdir -p "$OUTPUT_ROOT"
  echo "[INFO] INPUT_DIR=$INPUT_DIR"
  echo "[INFO] OUTPUT_ROOT=$OUTPUT_ROOT"
  echo "[INFO] TASKS=$TASKS"

  for task in aime25 aime26 amobench answerbench; do
    if task_enabled "$task"; then
      run_answer_task "$task"
    fi
  done

  if task_enabled "fs_olympiad"; then
    run_fs_olympiad
  fi

  echo "[DONE] Verifiable evaluation outputs: $OUTPUT_ROOT"
}

main "$@"
