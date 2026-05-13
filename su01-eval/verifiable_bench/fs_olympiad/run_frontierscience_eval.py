#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OFFICIAL_DATA_PATH = SCRIPT_DIR.parent / \
    "extract_data" / "frontierscience"
DEFAULT_OUTPUT_ROOT = SCRIPT_DIR / "results" / "frontierscience_eval"


def dump_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def format_command(cmd: list[str]) -> str:
    redacted: list[str] = []
    redact_next = False
    for token in cmd:
        if redact_next:
            redacted.append("******")
            redact_next = False
            continue
        redacted.append(shlex.quote(token))
        if token == "--api-key":
            redact_next = True
    return " ".join(redacted)


def build_track_command(
    python_bin: str,
    *,
    track: str,
    prediction_path: str,
    official_data_path: str,
    output_root: Path,
    args: argparse.Namespace,
) -> list[str]:
    judge_model = getattr(args, f"{track}_judge_model")
    api_key = getattr(args, f"{track}_api_key")
    base_url = getattr(args, f"{track}_base_url")
    reasoning_effort = getattr(args, f"{track}_reasoning_effort")
    response_key = getattr(args, f"{track}_response_key")
    match_by = getattr(args, f"{track}_match_by")

    cmd = [
        python_bin,
        str(SCRIPT_DIR / "eval_frontierscience.py"),
        "--track",
        track,
        "--prediction-path",
        prediction_path,
        "--official-data-path",
        official_data_path,
        "--output-dir",
        str(output_root),
        "--judge-model",
        judge_model,
        "--reasoning-effort",
        reasoning_effort,
        "--match-by",
        match_by,
        "--max-workers",
        str(args.max_workers),
        "--max-tokens",
        str(args.max_tokens),
        "--request-interval",
        str(args.request_interval),
    ]
    if response_key:
        cmd.extend(["--response-key", response_key])
    if base_url:
        cmd.extend(["--base-url", base_url])
    if api_key:
        cmd.extend(["--api-key", api_key])
    if args.stream:
        cmd.append("--stream")
    if args.concurrent:
        cmd.append("--concurrent")
    if args.resume:
        cmd.append("--resume")
    return cmd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Judge-only FrontierScience evaluator. "
            "This script only scores pre-generated prediction files/directories."
        )
    )
    parser.add_argument(
        "--olympiad-prediction-path",
        type=str,
        default=None,
        help="已生成好的 olympiad 预测文件（json/jsonl）或 trial 目录。",
    )
    parser.add_argument(
        "--research-prediction-path",
        type=str,
        default=None,
        help="已生成好的 research 预测文件（json/jsonl）或 trial 目录。",
    )
    parser.add_argument(
        "--official-data-path",
        type=str,
        default=str(DEFAULT_OFFICIAL_DATA_PATH),
        help="FrontierScience 官方处理后数据目录或文件。",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="评测输出根目录。",
    )
    parser.add_argument(
        "--python",
        type=str,
        default=sys.executable,
        help="用于执行子评测脚本的 Python。",
    )
    parser.add_argument(
        "--concurrent",
        action="store_true",
        help="对每个 subset 启用并发 judge。",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=32,
        help="并发 worker 数。",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=32768,
        help="judge 单次请求的最大输出 token 数。",
    )
    parser.add_argument(
        "--request-interval",
        type=float,
        default=0.0,
        help="并发请求启动间隔（秒）。",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="复用已有 judged 输出继续跑。",
    )
    parser.add_argument(
        "--stream",
        action="store_true",
        help="使用流式 chat completions 调 judge，并将 chunk 拼回完整 judge_response。",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只打印将执行的 judge 命令，不实际运行。",
    )

    parser.add_argument("--olympiad-judge-model", type=str, default="gpt-5")
    parser.add_argument("--research-judge-model", type=str, default="gpt-5")
    parser.add_argument("--olympiad-api-key", type=str, default=None)
    parser.add_argument("--research-api-key", type=str, default=None)
    parser.add_argument("--olympiad-base-url", type=str, default=None)
    parser.add_argument("--research-base-url", type=str, default=None)
    parser.add_argument("--olympiad-reasoning-effort",
                        type=str, default="high")
    parser.add_argument("--research-reasoning-effort",
                        type=str, default="high")
    parser.add_argument("--olympiad-response-key", type=str, default=None)
    parser.add_argument("--research-response-key", type=str, default=None)
    parser.add_argument("--olympiad-match-by", type=str, default="auto")
    parser.add_argument("--research-match-by", type=str, default="auto")

    args = parser.parse_args()
    if not args.olympiad_prediction_path and not args.research_prediction_path:
        parser.error(
            "至少提供 --olympiad-prediction-path 或 --research-prediction-path 之一。")
    return args


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)

    run_manifest: dict[str, Any] = {
        "official_data_path": str(Path(args.official_data_path).resolve()),
        "output_root": str(args.output_root.resolve()),
        "tracks": {},
    }

    for track, prediction_path in (
        ("olympiad", args.olympiad_prediction_path),
        ("research", args.research_prediction_path),
    ):
        if not prediction_path:
            continue

        cmd = build_track_command(
            args.python,
            track=track,
            prediction_path=prediction_path,
            official_data_path=args.official_data_path,
            output_root=args.output_root,
            args=args,
        )
        print(f"[RUN] {format_command(cmd)}", flush=True)
        if args.dry_run:
            run_manifest["tracks"][track] = {
                "prediction_path": str(Path(prediction_path).resolve()),
                "judge_model": getattr(args, f"{track}_judge_model"),
                "base_url": getattr(args, f"{track}_base_url"),
                "reasoning_effort": getattr(args, f"{track}_reasoning_effort"),
                "stream": args.stream,
                "dry_run": True,
            }
            continue

        subprocess.run(cmd, check=True, cwd=str(SCRIPT_DIR))

        aggregate_path = args.output_root / track / "aggregate_summary.json"
        aggregate_summary = load_json(aggregate_path)
        run_manifest["tracks"][track] = {
            "prediction_path": str(Path(prediction_path).resolve()),
            "judge_model": getattr(args, f"{track}_judge_model"),
            "base_url": getattr(args, f"{track}_base_url"),
            "reasoning_effort": getattr(args, f"{track}_reasoning_effort"),
            "stream": args.stream,
            "aggregate_summary_path": str(aggregate_path.resolve()),
            "aggregate_summary": aggregate_summary,
        }

    manifest_path = args.output_root / "frontierscience_eval_manifest.json"
    dump_json(manifest_path, run_manifest)
    print(f"\nManifest: {manifest_path.resolve()}", flush=True)


if __name__ == "__main__":
    main()
