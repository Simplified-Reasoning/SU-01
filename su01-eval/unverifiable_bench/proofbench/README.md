# ProofBench Evaluation

This folder contains the release-facing ProofBench evaluation workflow.

## Files

- `proofbench.json`: ProofBench problem metadata, reference solutions, and
  grading guidelines.
- `prepare_proofbench_eval.py`: merges model predictions with the ProofBench
  metadata so the judge receives the original problem statement and rubric.
- `eval_mo.py`: sends prepared samples to an OpenAI-compatible judge API.
- `normalize_points.py`: normalizes judge outputs into parseable
  `<points>X out of Y</points>` strings when needed.
- `summarize_points.py`: summarizes normalized point strings into total score,
  score rate, and Basic/Advanced split metrics.
- `run_proofbench_eval.sh`: one-command wrapper for preparation, judging, and
  optional point normalization and summary generation.

## Expected Inputs

The recommended input is a prediction JSON/JSONL file containing a `response`
field for each ProofBench problem. If available, `metadata.problem_idx` or
`problem_id` is used to align predictions with `proofbench.json`; otherwise
rows are aligned by order.

You can also use a directory containing one text file per problem with names
such as `PB-Basic-001_out.txt`.

## One-Command Usage

```bash
export INPUT_FILE=/path/to/proofbench.json
export OUTPUT_ROOT=/path/to/output/proofbench
export API_KEY="$OPENAI_API_KEY"
export BASE_URL=https://api.example.com/v1
export JUDGE_MODEL=gemini-2.5-pro
export CONCURRENT=1

bash su01-eval/unverifiable_bench/proofbench/run_proofbench_eval.sh
```

`BASE_URL` should normally be an OpenAI-compatible `/v1` base URL. The wrapper
also accepts a full `/v1/chat/completions` endpoint and trims it automatically.

Set `DRY_RUN=1` to prepare inputs and print the judge configuration without
sending API requests.

By default, the wrapper writes a summary next to the judge result:

```text
<OUTPUT_ROOT>/judge/<task>/<judge-model>/<task>-<judge-model>.summary.json
```

Set `SUMMARIZE_POINTS=0` to skip summary generation. Set `SUMMARY_PATH` to
write the summary to a custom path.

## Text-File Responses

```bash
export RESPONSE_DIR=/path/to/proofbench/responses
export RESPONSE_PATTERN='{problem_id}_out.txt'
export OUTPUT_ROOT=/path/to/output/proofbench
export API_KEY="$OPENAI_API_KEY"
export BASE_URL=https://api.example.com/v1

bash su01-eval/unverifiable_bench/proofbench/run_proofbench_eval.sh
```

If the eval input has already been prepared, skip preparation:

```bash
export DATA_PATH=/path/to/proofbench_eval_input.json
export OUTPUT_ROOT=/path/to/output/proofbench
export API_KEY="$OPENAI_API_KEY"
export BASE_URL=https://api.example.com/v1

bash su01-eval/unverifiable_bench/proofbench/run_proofbench_eval.sh
```
