# Verifiable Bench Evaluation

This folder is the release-facing home for benchmarks that can be scored with
programmatic or verifier-style reward logic.

Current layout:

- `answer_verifiable_bench/`: reward-server client for answer-verifiable
  benchmarks, including AIME25/26, AMO-Bench, and AnswerBench.
- `fs_olympiad/`: FrontierScience Olympiad judge-only evaluation scripts.
- `run_verifiable_eval.sh`: one-click wrapper for the verifiable benchmark
  group.

Example smoke test:

```bash
INPUT_DIR=/path/to/predictions \
RM_URL=http://host:8001 \
TASKS=aime25 \
MAX_ITEMS=2 \
bash su01-eval/verifiable_bench/run_verifiable_eval.sh
```

By default the wrapper expects these files under `INPUT_DIR`:

- `aime_2025.json`
- `aime_2026.json`
- `amobench.json`
- `answerbench.json`
- `frontierscience_olympiad.json`

Each path can also be set directly with `AIME25_INPUT`, `AIME26_INPUT`,
`AMOBENCH_INPUT`, `ANSWERBENCH_INPUT`, or `FS_OLYMPIAD_INPUT`.

For FrontierScience Olympiad, set:

- `FRONTIER_OFFICIAL_DATA_PATH`: directory containing the processed official
  FrontierScience data.
- `OLYMPIAD_BASE_URL`: judge endpoint base URL.
- `OLYMPIAD_API_KEY`: judge API key.
- `OLYMPIAD_JUDGE_MODEL`: judge model name. Default: `gpt-oss`.
- `OLYMPIAD_REASONING_EFFORT`: default: `high`.
- `FRONTIER_CONCURRENT`: default: `1`.
- `FRONTIER_STREAM`: default: `1`.
- `FRONTIER_RESUME`: default: `1`.

The original `eval/` tree is kept untouched for now. Future cleanup should make
this folder self-contained and use environment variables or CLI arguments for
private endpoints, model names, and data paths.
