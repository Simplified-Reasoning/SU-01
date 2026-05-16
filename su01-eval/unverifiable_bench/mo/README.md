# MO Evaluation

This folder contains the release-facing scripts for model-judged mathematical
olympiad proof evaluation.

## Files

- `prepare_mo_eval.py`: converts one response txt file per problem into the
  JSON/JSONL format consumed by `eval_mo.py`.
- `eval_mo.py`: sends prepared samples to an OpenAI-compatible judge API and
  writes raw judge outputs.
- `normalize_points.py`: normalizes judge outputs into parseable
  `<points>X out of Y</points>` strings when the raw judge response is not
  already parseable.
- `metadata/`: problem metadata files, including the IMO 2025 prompt and
  reference-solution JSONL used by `META_JSONL`.
- `rubrics/`: optional per-problem rubric overrides, including the IMO 2025
  guideline file used by `GUIDELINE_MD`.
- `run_mo_eval.sh`: one-command wrapper for preparation, judging, and optional
  point normalization.

## Expected Inputs

`prepare_mo_eval.py` expects:

- `--response-dir`: a directory containing files such as `1_out.txt`,
  `13_out.txt`, or `imo01_out.txt`.
- `--meta-jsonl`: a JSONL file containing the problem metadata. Each row should
  include a problem statement (`question` or `prompt`) and a reference solution
  (`label`). Optional rubrics can be supplied through `rubrics` or
  `grading_guidelines`.

For IMO-style six-problem sets, `--question-cycle 6` maps response file indices
back to problem numbers, for example `13_out.txt -> problem 1`. Set
`--question-cycle 0` to disable this mapping.

## One-Command Usage

### IMO 2025

```bash
export RESPONSE_DIR=/path/to/responses/imo25
export META_JSONL=su01-eval/unverifiable_bench/mo/metadata/imo25.jsonl
export OUTPUT_ROOT=/path/to/output/mo
export TASK_NAME=imo25
export RESPONSE_PATTERN='imo{idx:02d}_out.txt'
export QUESTION_CYCLE=6
export GUIDELINE_MD=su01-eval/unverifiable_bench/mo/rubrics/imo25_guideline.md
export API_KEY="$OPENAI_API_KEY"
export BASE_URL=https://api.example.com/v1
export JUDGE_MODEL=gemini-2.5-pro

bash su01-eval/unverifiable_bench/mo/run_mo_eval.sh
```

### USAMO 2026

```bash
export RESPONSE_DIR=/path/to/responses/usamo_2026/out
export META_JSONL=su01-eval/unverifiable_bench/mo/metadata/usamo2026.jsonl
export OUTPUT_ROOT=/path/to/output/usamo2026
export TASK_NAME=usamo2026
export RESPONSE_PATTERN='USAMO-2026-P{idx}_out_s0.txt'
export QUESTION_CYCLE=6
export GUIDELINE_MD=su01-eval/unverifiable_bench/mo/rubrics/usamo2026_guideline.md
export API_KEY="$OPENAI_API_KEY"
export BASE_URL=https://api.example.com/v1
export JUDGE_MODEL=gemini-2.5-pro

bash su01-eval/unverifiable_bench/mo/run_mo_eval.sh
```

`BASE_URL` should normally be an OpenAI-compatible `/v1` base URL. The wrapper
also accepts a full `/v1/chat/completions` endpoint and trims it automatically.

Set `DRY_RUN=1` to run input preparation and print the judge configuration
without sending API requests.

If the eval input has already been prepared, skip the preparation step:

```bash
export DATA_PATH=/path/to/mo_eval_input.jsonl
export OUTPUT_ROOT=/path/to/output/mo
export API_KEY="$OPENAI_API_KEY"
export BASE_URL=https://api.example.com/v1

bash su01-eval/unverifiable_bench/mo/run_mo_eval.sh
```
