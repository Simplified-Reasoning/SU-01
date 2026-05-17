#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Optional, Tuple


POINTS_PATTERN = re.compile(
    r"<points>\s*(\d+(?:\.\d+)?)\s+out\s+of\s+(\d+(?:\.\d+)?)\s*</points>",
    re.IGNORECASE,
)


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def extract_rows(data: Any) -> list[dict[str, Any]]:
    if isinstance(data, list):
        return [row for row in data if isinstance(row, dict)]
    if isinstance(data, dict):
        for key in ("results", "data", "items", "rows"):
            value = data.get(key)
            if isinstance(value, list):
                return [row for row in value if isinstance(row, dict)]
    return []


def parse_points(value: Any) -> Tuple[Optional[float], Optional[float]]:
    if not isinstance(value, str):
        return None, None
    match = POINTS_PATTERN.search(value)
    if match is None:
        return None, None
    return float(match.group(1)), float(match.group(2))


def nested_get(data: dict[str, Any], *keys: str) -> Any:
    cur: Any = data
    for key in keys:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(key)
    return cur


def resolve_problem_id(row: dict[str, Any]) -> str:
    for key in ("problem_id", "problem_idx", "source_id", "id"):
        value = row.get(key)
        if value is not None:
            return str(value)
    for key in ("problem_idx", "problem_id", "task_group_id", "source_id", "id"):
        value = nested_get(row, "metadata", key)
        if value is not None:
            return str(value)
    return ""


def summarize(rows: list[dict[str, Any]], result_file: Path) -> dict[str, Any]:
    parsed_count = 0
    failed_count = 0
    total_score = 0.0
    max_score = 0.0
    basic_count = 0
    basic_score = 0.0
    basic_max_score = 0.0
    advanced_count = 0
    advanced_score = 0.0
    advanced_max_score = 0.0

    for row in rows:
        score, max_value = parse_points(row.get("prediction"))
        if score is None or max_value is None:
            failed_count += 1
            continue

        parsed_count += 1
        total_score += score
        max_score += max_value

        problem_id = resolve_problem_id(row).lower()
        if "basic" in problem_id:
            basic_count += 1
            basic_score += score
            basic_max_score += max_value
        elif "advanced" in problem_id:
            advanced_count += 1
            advanced_score += score
            advanced_max_score += max_value

    summary: dict[str, Any] = {
        "status": "ok" if rows else "missing_result",
        "result_file": str(result_file.resolve()),
        "num_rows": len(rows),
        "parsed_count": parsed_count,
        "failed_count": failed_count,
        "total_score": total_score,
        "max_score": max_score,
        "score_rate_pct": (100.0 * total_score / max_score) if max_score else None,
    }
    if basic_max_score:
        summary.update(
            {
                "basic_count": basic_count,
                "basic_score": basic_score,
                "basic_max_score": basic_max_score,
                "basic_score_rate_pct": 100.0 * basic_score / basic_max_score,
            }
        )
    if advanced_max_score:
        summary.update(
            {
                "advanced_count": advanced_count,
                "advanced_score": advanced_score,
                "advanced_max_score": advanced_max_score,
                "advanced_score_rate_pct": 100.0 * advanced_score / advanced_max_score,
            }
        )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize ProofBench judge results from <points>X out of Y</points> predictions."
    )
    parser.add_argument("--input-file", type=Path, required=True, help="Judge result JSON file.")
    parser.add_argument(
        "--output-file",
        type=Path,
        default=None,
        help="Summary JSON path. Defaults to <input stem>.summary.json.",
    )
    args = parser.parse_args()

    rows = extract_rows(load_json(args.input_file))
    output_file = args.output_file or args.input_file.with_suffix(".summary.json")
    summary = summarize(rows, args.input_file)
    save_json(output_file, summary)

    if summary["score_rate_pct"] is not None:
        print(
            f"Summary: {summary['total_score']}/{summary['max_score']} "
            f"({summary['score_rate_pct']:.4f}%)"
        )
    else:
        print("Summary: no parseable points")
    print(f"Summary saved to: {output_file}")


if __name__ == "__main__":
    main()
