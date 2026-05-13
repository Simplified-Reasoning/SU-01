# Decode

This directory contains the SU-01 decoding entry points.

- `direct_gen.py`: direct single-pass decoding. It accepts either one `.txt`
  problem or a JSONL file.
- `tts_gen.py`: test-time scaling decoding with solution generation,
  verification, and correction loops.
- `decode.py`: batch runner over a `problems/` tree and `problem_list.txt`
  files.

## Input Layout

```text
<problems-root>/<dataset>/problem_list.txt
<problems-root>/<dataset>/<problem>.txt
<problems-root>/<dataset>/general_prompt.txt
<problems-root>/<dataset>/<problem>_instruct.txt
```

`problem_list.txt` may contain absolute paths or paths relative to the dataset
directory. `general_prompt.txt` and `*_instruct.txt` are optional for direct
decoding but are used when present.

## Default Settings

The decode scripts use these defaults unless overridden by the environment:

```bash
export MODEL_NAME="SU01"
export MAX_TOKENS=160000
export TEMPERATURE=1.0
export TOP_K=-1
export TOP_P=0.95
export API_TIMEOUT=432000000
export REQUEST_TIMEOUT=1800000
export NO_PROXY="localhost,127.0.0.1"
export no_proxy="$NO_PROXY"
unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy
```

## Single Problem

```bash
export API_URL="http://localhost:34883/v1/chat/completions"
export OPENAI_API_KEY="dummy"

python su01-eval/decode/direct_gen.py \
  /path/to/problems/amobench/amobench-1.txt \
  --log /tmp/su01-decode/direct_gen/amobench-1.log \
  --out /tmp/su01-decode/direct_gen/amobench-1_out.txt
```

For TTS decoding:

```bash
python su01-eval/decode/tts_gen.py \
  /path/to/problems/proofbench/proofbench-1.txt \
  --dataset_name proofbench \
  --max_runs 10 \
  --parallel_runs 3 \
  --log /tmp/su01-decode/tts_gen/proofbench-1.log \
  --out /tmp/su01-decode/tts_gen/proofbench-1_out.txt
```

## Batch Decode

```bash
python su01-eval/decode/decode.py \
  --problems-root /path/to/problems \
  --output-root /tmp/su01-decode \
  --datasets amobench,aime_2025 \
  --decode-method direct_gen \
  --model SU01 \
  --max-workers 8
```

Use `--decode-method tts_gen` to run the TTS decoding flow. Extra
method-specific flags can be passed with repeated `--decode-arg=...` values.

## Server Helpers

Launch one SGLang worker:

```bash
bash su01-eval/decode/server/server.sh \
  --model-path /path/to/model \
  --model-name SU01 \
  --port 34883
```

Launch a router over one or more workers:

```bash
bash su01-eval/decode/server/router.sh \
  --worker-url http://host1:34883 \
  --worker-url http://host2:34883 \
  --port 30000
```

## Smoke Test

All three decode scripts support `--dry-run`, which writes deterministic
outputs without calling `API_URL`.

```bash
python su01-eval/decode/decode.py \
  --problems-root /path/to/problems \
  --output-root /tmp/su01-decode-smoke \
  --datasets smoke \
  --decode-method direct_gen \
  --sample-counts smoke=1 \
  --limit-per-dataset 1 \
  --dry-run
```
