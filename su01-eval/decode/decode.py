#!/usr/bin/env python3
"""Batch runner for SU-01 decode scripts.

Expected input layout:

    <problems-root>/<dataset>/problem_list.txt
    <problems-root>/<dataset>/<problem>.txt
    <problems-root>/<dataset>/general_prompt.txt       optional
    <problems-root>/<dataset>/<problem>_instruct.txt    optional

Each line in ``problem_list.txt`` can be an absolute path or a path relative to
the dataset directory. Outputs are written as
``<output-root>/<model>/<decode-method>/<dataset>/out/<stem>_out_s<N>.txt``.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DATASETS = [
    "amobench",
    "answerbench",
    "aime_2025",
    "aime_2026",
    "IPhO_2024",
    "IPhO_2025",
    "proofbench",
    "frontierscience_olympiad",
    "fs-research",
    "usamo_2026",
]
DEFAULT_SAMPLE_COUNTS = {
    "amobench": 8,
    "answerbench": 4,
    "aime_2025": 8,
    "aime_2026": 8,
    "IPhO_2024": 8,
    "IPhO_2025": 8,
    "frontierscience_olympiad": 4,
    "proofbench": 1,
    "fs-research": 1,
    "usamo_2026": 1,
}
DECODE_SCRIPTS = {
    "direct_gen": SCRIPT_DIR / "direct_gen.py",
    "tts_gen": SCRIPT_DIR / "tts_gen.py",
}


@dataclass(frozen=True)
class Task:
    dataset: str
    item_index: int
    total_items: int
    sample_index: int
    problem_path: Path
    log_dir: Path
    out_dir: Path
    decode_method: str
    decode_script: Path
    dry_run: bool
    force_rerun: bool
    min_out_bytes: int
    extra_decode_args: tuple[str, ...]


def parse_csv(value: str | None) -> list[str]:
    if not value:
        return []
    return [part.strip() for part in value.split(",") if part.strip()]


def parse_sample_counts(raw: str | None) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in parse_csv(raw):
        if "=" not in item:
            raise ValueError(f"Invalid --sample-counts entry: {item!r}; expected dataset=N")
        dataset, value = item.split("=", 1)
        counts[dataset.strip()] = int(value)
    return counts


def is_truthy(value: str | None) -> bool:
    return (value or "").lower() in {"1", "true", "yes", "on"}


def valid_output(path: Path, min_bytes: int) -> bool:
    try:
        return path.is_file() and path.stat().st_size >= min_bytes
    except OSError:
        return False


def read_problem_list(dataset_dir: Path, limit: int | None) -> list[Path]:
    problem_list = dataset_dir / "problem_list.txt"
    if not problem_list.is_file():
        print(f"[SKIP] {dataset_dir.name}: missing {problem_list}")
        return []

    problems: list[Path] = []
    with problem_list.open("r", encoding="utf-8") as handle:
        for line in handle:
            entry = line.strip()
            if not entry:
                continue
            path = Path(entry)
            problems.append(path if path.is_absolute() else dataset_dir / path)
            if limit is not None and len(problems) >= limit:
                break
    return problems


def build_tasks(args: argparse.Namespace, decode_script: Path) -> list[Task]:
    datasets = parse_csv(args.datasets) or DEFAULT_DATASETS
    sample_counts = DEFAULT_SAMPLE_COUNTS.copy()
    sample_counts.update(parse_sample_counts(args.sample_counts))

    tasks: list[Task] = []
    for dataset in datasets:
        dataset_dir = args.problems_root / dataset
        problems = read_problem_list(dataset_dir, args.limit_per_dataset)
        if not problems:
            continue

        n_samples = sample_counts.get(dataset, args.default_samples)
        log_dir = args.output_root / args.model / args.decode_method / dataset
        out_dir = log_dir / "out"
        for item_index, problem_path in enumerate(problems, 1):
            for sample_index in range(n_samples):
                tasks.append(
                    Task(
                        dataset=dataset,
                        item_index=item_index,
                        total_items=len(problems),
                        sample_index=sample_index,
                        problem_path=problem_path,
                        log_dir=log_dir,
                        out_dir=out_dir,
                        decode_method=args.decode_method,
                        decode_script=decode_script,
                        dry_run=args.dry_run,
                        force_rerun=args.force_rerun,
                        min_out_bytes=args.min_out_bytes,
                        extra_decode_args=tuple(args.decode_arg or []),
                    )
                )
    return tasks


def command_for_task(task: Task) -> tuple[list[str], Path, Path]:
    stem = task.problem_path.stem
    tag = f"{stem}_s{task.sample_index}"
    log_path = task.log_dir / f"{tag}.log"
    out_path = task.out_dir / f"{stem}_out_s{task.sample_index}.txt"

    cmd = [
        sys.executable,
        str(task.decode_script),
        str(task.problem_path),
        "--log",
        str(log_path),
        "--out",
        str(out_path),
    ]
    if task.decode_method == "tts_gen":
        cmd.extend(["--dataset_name", task.dataset])
    if task.dry_run:
        cmd.append("--dry-run")
    cmd.extend(task.extra_decode_args)
    return cmd, log_path, out_path


def run_one(task: Task, env: dict[str, str]) -> tuple[str, str, int | str | None]:
    cmd, log_path, out_path = command_for_task(task)
    label = f"{task.dataset}:{task.problem_path.name}:s{task.sample_index}"

    if not task.problem_path.is_file():
        print(f"[SKIP] {label}: problem file not found: {task.problem_path}")
        return label, "missing", None

    if not task.force_rerun and valid_output(out_path, task.min_out_bytes):
        print(f"[SKIP] {label}: output exists")
        return label, "resume_skip", None

    task.log_dir.mkdir(parents=True, exist_ok=True)
    task.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[RUN] {label}")
    result = subprocess.run(cmd, cwd=SCRIPT_DIR, env=env)
    if result.returncode == 0:
        print(f"[DONE] {label} -> {out_path}")
        return label, "ok", 0

    print(f"[FAIL] {label}: exit {result.returncode}; log={log_path}")
    return label, "failed", result.returncode


def add_common_env(args: argparse.Namespace) -> dict[str, str]:
    env = os.environ.copy()
    env["MODEL_NAME"] = args.model
    if args.api_url:
        env["API_URL"] = args.api_url
    if args.api_key is not None:
        env["OPENAI_API_KEY"] = args.api_key
    if args.dry_run:
        env["DECODE_DRY_RUN"] = "1"
        env.setdefault("OPENAI_API_KEY", "dummy")
    return env


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def normalize_decode_method(value: str) -> str:
    if value not in DECODE_SCRIPTS:
        choices = ", ".join(sorted(DECODE_SCRIPTS))
        raise ValueError(f"Unknown decode method {value!r}; expected one of: {choices}")
    return value


def existing_decode_script(decode_method: str, decode_script: str | None) -> Path:
    if decode_script:
        path = Path(decode_script).expanduser().resolve()
    else:
        path = DECODE_SCRIPTS[decode_method]
    if not path.is_file():
        raise FileNotFoundError(f"Decode script not found: {path}")
    return path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Batch decode runner.")
    problems_root_env = os.environ.get("PROBLEMS_ROOT")
    parser.add_argument(
        "--problems-root",
        type=Path,
        default=Path(problems_root_env) if problems_root_env else None,
        required=not bool(problems_root_env),
        help="Root directory containing per-dataset problem folders.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(os.environ.get("OUTPUT_ROOT", "/tmp/su01-decode")),
        help="Directory for logs and decoded outputs.",
    )
    parser.add_argument(
        "--datasets",
        default=os.environ.get("DATASETS", ",".join(DEFAULT_DATASETS)),
        help="Comma-separated dataset names.",
    )
    parser.add_argument(
        "--decode-method",
        choices=sorted(DECODE_SCRIPTS),
        default=os.environ.get("DECODE_METHOD", "direct_gen"),
        help="Decoding method to run.",
    )
    parser.add_argument("--decode-script", default=os.environ.get("DECODE_SCRIPT"))
    parser.add_argument("--model", default=os.environ.get("MODEL_NAME", "SU01"))
    parser.add_argument("--api-url", default=os.environ.get("API_URL", "http://localhost:34883/v1/chat/completions"))
    parser.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY"))
    parser.add_argument("--max-workers", type=positive_int, default=int(os.environ.get("MAX_WORKERS", "4")))
    parser.add_argument("--default-samples", type=positive_int, default=int(os.environ.get("DEFAULT_SAMPLES", "1")))
    parser.add_argument("--sample-counts", default=os.environ.get("SAMPLE_COUNTS"))
    parser.add_argument("--limit-per-dataset", type=positive_int, default=None)
    parser.add_argument("--force-rerun", action="store_true", default=is_truthy(os.environ.get("FORCE_RERUN")))
    parser.add_argument("--min-out-bytes", type=int, default=int(os.environ.get("MIN_OUT_BYTES", "1")))
    parser.add_argument("--dry-run", action="store_true", default=is_truthy(os.environ.get("DECODE_DRY_RUN")))
    parser.add_argument(
        "--decode-arg",
        dest="decode_arg",
        action="append",
        help="Extra argument passed through to the selected decode script. Repeat for multiple args.",
    )
    return parser


def summarize(results: Iterable[tuple[str, str, int | str | None]]) -> int:
    counts: dict[str, int] = {}
    failed = 0
    for _, status, _ in results:
        counts[status] = counts.get(status, 0) + 1
        if status == "failed":
            failed += 1

    print("\nBatch summary:")
    for status in sorted(counts):
        print(f"  {status}: {counts[status]}")
    return 1 if failed else 0


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    args.problems_root = args.problems_root.expanduser().resolve()
    args.output_root = args.output_root.expanduser().resolve()

    try:
        args.decode_method = normalize_decode_method(args.decode_method)
        decode_script = existing_decode_script(args.decode_method, args.decode_script)
        tasks = build_tasks(args, decode_script)
    except Exception as exc:
        parser.error(str(exc))

    if not tasks:
        print("No decode tasks were created.")
        return 0

    print(f"Decode method: {args.decode_method}")
    print(f"Problems root: {args.problems_root}")
    print(f"Output root: {args.output_root}")
    print(f"Tasks: {len(tasks)}")
    if args.dry_run:
        print("Dry run: API requests are skipped.")

    env = add_common_env(args)
    results = []
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = [executor.submit(run_one, task, env) for task in tasks]
        for future in as_completed(futures):
            results.append(future.result())

    return summarize(results)


if __name__ == "__main__":
    raise SystemExit(main())
