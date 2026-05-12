#!/usr/bin/env python3
"""
Interactive dashboard for training logs and eval JSON logs.

Usage:
    streamlit run tools/eval_log_dashboard.py -- --logs-root /path/to/logs --logs-root /path/to/other/logs
"""

from __future__ import annotations

import argparse
import ast
import functools
import json
import re
import sys
import time
import zlib
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence


DEFAULT_LOG_ROOTS = [
    "/root/p1-slime/logs",
]
EVAL_SUMMARY_CACHE_VERSION = 3
RECENT_ACTIVITY_SECONDS = 3 * 60 * 60
DEFAULT_EVAL_GLOB = "**/eval_*_rollout_*.json"
DEFAULT_TRAINING_GLOB = "**/training_*.log"
EVAL_FILE_RE = re.compile(r"^eval_(?P<dataset>.+)_rollout_(?P<rollout>\d+)\.json$")
ENTRY_RE = re.compile(
    r"(?P<entry_type>rollout|step|perf|eval|multi_turn|passrate)\s+"
    r"(?P<entry_num>\d+):\s*(?P<payload>\{.+\})"
)
TIMESTAMP_RE = re.compile(r"^\[(?P<timestamp>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\]\s*(?P<rest>.*)$")
ANSI_RE = re.compile(r"\x1b\[[0-9;]*m|\[[0-9;]*m")
EXPERIMENT_TS_RE = re.compile(r"-(?P<mmdd>\d{4})-(?P<hhmmss>\d{6})$")
THINK_OPEN_TOKEN = "<think>"
THINK_CLOSE_TOKEN = "</think>"
IM_START_TOKEN = "<|im_start|>"
IM_END_TOKEN = "<|im_end|>"
END_OF_TEXT_TOKEN = "<|endoftext|>"


@dataclass(frozen=True)
class TrainingLogMeta:
    source_root: str
    source_label: str
    experiment: str
    log_path: str
    log_name: str
    modified_ts: float


@dataclass(frozen=True)
class EvalFileMeta:
    source_root: str
    source_label: str
    experiment: str
    dataset: str
    rollout: int
    file_path: str
    modified_ts: float


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize training logs and eval JSON files with Streamlit.")
    parser.add_argument(
        "--logs-root",
        dest="logs_roots",
        action="append",
        default=None,
        help="Repeat this flag to scan multiple log roots.",
    )
    parser.add_argument(
        "--eval-glob",
        default=DEFAULT_EVAL_GLOB,
        help="Glob used to discover eval JSON files such as eval_amo_rollout_1.json.",
    )
    parser.add_argument(
        "--training-glob",
        default=DEFAULT_TRAINING_GLOB,
        help="Glob used to discover training log files such as training_6h2lm0rx.log.",
    )
    parser.add_argument(
        "--default-training-logs",
        type=int,
        default=3,
        help="How many recent training logs to preselect.",
    )
    parser.add_argument(
        "--default-eval-files",
        type=int,
        default=24,
        help="How many recent eval JSON files to preselect.",
    )
    args, _ = parser.parse_known_args(argv)
    args.logs_roots = args.logs_roots or list(DEFAULT_LOG_ROOTS)
    return args


def source_label_for_root(root: str | Path) -> str:
    path = Path(root).expanduser().resolve()
    parts = path.parts
    if len(parts) >= 2:
        return "/".join(parts[-2:])
    return str(path)


def extract_experiment_datetime(experiment_name: str) -> datetime | None:
    match = EXPERIMENT_TS_RE.search(experiment_name)
    if not match:
        return None
    mmdd = match.group("mmdd")
    hhmmss = match.group("hhmmss")
    try:
        return datetime.strptime(f"2026{mmdd}{hhmmss}", "%Y%m%d%H%M%S")
    except ValueError:
        return None


def experiment_sort_key(experiment_name: str) -> tuple[int, float, str]:
    parsed = extract_experiment_datetime(experiment_name)
    if parsed is None:
        return (1, 0.0, experiment_name)
    return (0, -parsed.timestamp(), experiment_name)


def meta_sort_key(experiment_name: str, modified_ts: float, tie_breaker: str) -> tuple[int, float, float, str]:
    parsed = extract_experiment_datetime(experiment_name)
    if parsed is None:
        return (1, -modified_ts, -modified_ts, tie_breaker)
    return (0, -parsed.timestamp(), -modified_ts, tie_breaker)


def is_recent_activity(modified_ts: float, now_ts: float | None = None, window_seconds: int = RECENT_ACTIVITY_SECONDS) -> bool:
    effective_now = time.time() if now_ts is None else now_ts
    return modified_ts >= effective_now - window_seconds


def default_recent_experiments(
    meta_items: Sequence[Any], experiment_options: list[str], fallback_limit: int, now_ts: float | None = None
) -> list[str]:
    recent_experiments = [
        experiment
        for experiment in experiment_options
        if any(item.experiment == experiment and is_recent_activity(item.modified_ts, now_ts=now_ts) for item in meta_items)
    ]
    if recent_experiments:
        return recent_experiments
    return experiment_options[: min(fallback_limit, len(experiment_options))]


def default_recent_labels(
    meta_items: Sequence[Any], labels: list[str], fallback_limit: int, now_ts: float | None = None
) -> list[str]:
    recent_labels = [
        label
        for item, label in zip(meta_items, labels)
        if is_recent_activity(item.modified_ts, now_ts=now_ts)
    ]
    if recent_labels:
        return recent_labels
    return labels[: min(fallback_limit, len(labels))]


def filter_in_order(values: Sequence[str], allowed_values: Sequence[str]) -> list[str]:
    allowed = set(allowed_values)
    return [value for value in values if value in allowed]


def discover_training_logs(roots: Sequence[str | Path], glob_pattern: str) -> list[TrainingLogMeta]:
    items: list[TrainingLogMeta] = []
    for root_value in roots:
        root = Path(root_value).expanduser()
        if not root.exists():
            continue
        root_resolved = root.resolve()
        source_label = source_label_for_root(root_resolved)
        for path in root.glob(glob_pattern):
            if not path.is_file():
                continue
            resolved = path.resolve()
            stat = resolved.stat()
            items.append(
                TrainingLogMeta(
                    source_root=str(root_resolved),
                    source_label=source_label,
                    experiment=resolved.parent.name,
                    log_path=str(resolved),
                    log_name=resolved.name,
                    modified_ts=stat.st_mtime,
                )
            )
    return sorted(
        items,
        key=lambda item: meta_sort_key(item.experiment, item.modified_ts, item.log_name),
    )


def discover_eval_files(roots: Sequence[str | Path], glob_pattern: str) -> list[EvalFileMeta]:
    items: list[EvalFileMeta] = []
    for root_value in roots:
        root = Path(root_value).expanduser()
        if not root.exists():
            continue
        root_resolved = root.resolve()
        source_label = source_label_for_root(root_resolved)
        for path in root.glob(glob_pattern):
            if not path.is_file():
                continue
            match = EVAL_FILE_RE.match(path.name)
            if not match:
                continue
            resolved = path.resolve()
            stat = resolved.stat()
            items.append(
                EvalFileMeta(
                    source_root=str(root_resolved),
                    source_label=source_label,
                    experiment=resolved.parent.name,
                    dataset=match.group("dataset"),
                    rollout=int(match.group("rollout")),
                    file_path=str(resolved),
                    modified_ts=stat.st_mtime,
                )
            )
    return sorted(
        items,
        key=lambda item: meta_sort_key(item.experiment, item.modified_ts, f"{item.dataset}:{item.rollout}:{item.file_path}"),
    )


def clean_training_log_line(line: str) -> str:
    return ANSI_RE.sub("", line).strip()


def parse_training_log_line(line: str) -> dict[str, Any] | None:
    clean_line = clean_training_log_line(line)
    if not clean_line:
        return None

    timestamp = None
    body = clean_line
    timestamp_match = TIMESTAMP_RE.match(clean_line)
    if timestamp_match:
        timestamp = timestamp_match.group("timestamp")
        body = timestamp_match.group("rest")

    entry_match = ENTRY_RE.search(body)
    if not entry_match:
        return None

    payload = ast.literal_eval(entry_match.group("payload"))
    if not isinstance(payload, dict):
        return None

    return {
        "timestamp": timestamp,
        "entry_type": entry_match.group("entry_type"),
        "entry_num": int(entry_match.group("entry_num")),
        "payload": payload,
    }


def coerce_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return None
        try:
            return float(stripped)
        except ValueError:
            return None
    return None


def coerce_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "1", "yes"}:
            return True
        if lowered in {"false", "0", "no"}:
            return False
    return None


def compression_ratio(data: str, level: int = 9) -> tuple[float, float]:
    raw = data.encode("utf-8")
    original = len(raw)
    if original == 0:
        return float("inf"), 0.0

    compressed = zlib.compress(raw, level)
    comp_len = len(compressed)
    if comp_len == 0:
        return float("inf"), 100.0

    ratio = original / comp_len
    savings_pct = 100.0 * (1.0 - comp_len / original)
    return ratio, savings_pct


def has_repetition(text: str) -> bool:
    return len(text) > 10000 and compression_ratio(text[-10000:])[0] > 10


def classify_generation_exception(response: str) -> dict[str, Any]:
    think_close_count = response.count(THINK_CLOSE_TOKEN)
    think_open_count = response.count(THINK_OPEN_TOKEN)
    im_end_count = response.count(IM_END_TOKEN)
    has_endoftext = END_OF_TEXT_TOKEN in response
    has_im_start = IM_START_TOKEN in response
    has_mid_im_end = IM_END_TOKEN in response and not response.rstrip().endswith(IM_END_TOKEN)
    has_multiple_im_end = im_end_count > 1
    has_bad_special = has_endoftext or has_im_start or has_mid_im_end or has_multiple_im_end
    has_repetition_raw = has_repetition(response)
    answer = response.split(THINK_CLOSE_TOKEN)[-1] if think_close_count else ""
    has_repetition_answer = has_repetition(answer) if answer else False

    if has_endoftext or has_im_start:
        anti_hack_reason = "bad_special_eot_or_im_start"
    elif has_multiple_im_end:
        anti_hack_reason = "bad_special_multiple_im_end"
    elif has_mid_im_end:
        anti_hack_reason = "bad_special_mid_im_end"
    elif think_close_count != 1:
        anti_hack_reason = "think_close_count_not_one"
    elif think_open_count > 0:
        anti_hack_reason = "contains_think_open"
    elif has_repetition_raw:
        anti_hack_reason = "repetition_raw"
    elif has_repetition_answer:
        anti_hack_reason = "repetition_answer"
    else:
        anti_hack_reason = "pass"

    return {
        "anti_hack_reason": anti_hack_reason,
        "anti_hack_reject": anti_hack_reason != "pass",
        "has_endoftext": has_endoftext,
        "endoftext_no_think": has_endoftext and think_close_count == 0,
        "endoftext_after_think": has_endoftext and think_close_count > 0,
        "think_close_count": think_close_count,
        "think_open_count": think_open_count,
        "think_close_count_not_one": think_close_count != 1,
        "think_close_zero": think_close_count == 0,
        "think_close_multi": think_close_count > 1,
        "contains_think_open": think_open_count > 0,
        "bad_special_token": has_bad_special,
        "has_im_start": has_im_start,
        "im_end_count": im_end_count,
        "bad_im_end": has_mid_im_end or has_multiple_im_end,
        "repetition_raw": has_repetition_raw,
        "repetition_answer": has_repetition_answer,
    }


def normalize_reward(reward: Any) -> dict[str, Any]:
    normalized = {
        "score": None,
        "point": None,
        "acc": None,
        "score_noxverify": None,
        "point_noxverify": None,
    }
    if isinstance(reward, dict):
        for key in normalized:
            value = reward.get(key)
            if key == "acc":
                normalized[key] = coerce_bool(value)
            else:
                normalized[key] = coerce_float(value)
    return normalized


def label_to_text(label: Any) -> str:
    if label is None:
        return ""
    if isinstance(label, list):
        return "\n".join(str(item) for item in label)
    if isinstance(label, dict):
        return json.dumps(label, ensure_ascii=True, sort_keys=True)
    return str(label)


def load_training_logs(log_paths: Sequence[str]) -> tuple[list[dict[str, Any]], list[str]]:
    rows: list[dict[str, Any]] = []
    warnings: list[str] = []
    for log_path in log_paths:
        path = Path(log_path)
        try:
            with path.open("r", encoding="utf-8", errors="replace") as handle:
                for line_number, line in enumerate(handle, start=1):
                    try:
                        parsed = parse_training_log_line(line)
                    except Exception as exc:
                        warnings.append(f"Failed to parse {path}:{line_number}: {exc}")
                        continue
                    if parsed is None:
                        continue
                    numeric_payload = {
                        key: coerce_float(value)
                        for key, value in parsed["payload"].items()
                    }
                    numeric_payload = {key: value for key, value in numeric_payload.items() if value is not None}
                    if not numeric_payload:
                        continue
                    rows.append(
                        {
                            "source_label": source_label_for_root(path.parent.parent),
                            "experiment": path.parent.name,
                            "log_path": str(path),
                            "log_name": path.name,
                            "entry_type": parsed["entry_type"],
                            "entry_num": parsed["entry_num"],
                            "timestamp": parsed["timestamp"],
                            "line_number": line_number,
                            **numeric_payload,
                        }
                    )
        except Exception as exc:
            warnings.append(f"Failed to read {path}: {exc}")
    return rows, warnings


@functools.lru_cache(maxsize=128)
def load_training_eval_metric_index(experiment_dir: str) -> dict[tuple[str, int], dict[str, float]]:
    experiment_path = Path(experiment_dir)
    training_logs = sorted(str(path) for path in experiment_path.glob("training_*.log") if path.is_file())
    if not training_logs:
        return {}

    rows, _ = load_training_logs(training_logs)
    index: dict[tuple[str, int], dict[str, float]] = {}
    for row in rows:
        if row.get("entry_type") != "eval":
            continue
        rollout = int(row["entry_num"])
        for key, value in row.items():
            if not key.startswith("eval/") or value is None:
                continue
            dataset_and_metric = key.removeprefix("eval/")
            metric_name: str | None = None
            dataset_name: str | None = None
            for suffix in ("-point", "-truncated_ratio", "/truncated_ratio", "/repetition_frac"):
                if dataset_and_metric.endswith(suffix):
                    dataset_name = dataset_and_metric[: -len(suffix)]
                    metric_name = suffix
                    break
            if metric_name is None and "/" not in dataset_and_metric and "-" not in dataset_and_metric:
                dataset_name = dataset_and_metric
                metric_name = "score"
            if dataset_name is None or metric_name is None:
                continue

            metrics = index.setdefault((dataset_name, rollout), {})
            if metric_name == "score":
                metrics["score_mean"] = float(value)
            elif metric_name == "-point":
                metrics["point"] = float(value)
                metrics["point_mean"] = float(value)
            elif metric_name in {"-truncated_ratio", "/truncated_ratio"}:
                metrics["truncated_ratio"] = float(value)
            elif metric_name == "/repetition_frac":
                metrics["repetition_frac"] = float(value)
    return index


def build_training_dataframe(training_rows: list[dict[str, Any]]):
    import pandas as pd

    frame = pd.DataFrame(training_rows)
    if frame.empty:
        return frame
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="coerce")
    frame = frame.sort_values(["source_label", "experiment", "log_name", "entry_type", "entry_num"]).reset_index(drop=True)

    rollout_mask = frame["entry_type"] == "rollout"
    rollout_frame = frame.loc[rollout_mask].copy()
    if not rollout_frame.empty:
        group_columns = ["source_label", "experiment", "log_name", "log_path"]
        rollout_frame["prev_timestamp"] = rollout_frame.groupby(group_columns)["timestamp"].shift(1)
        rollout_frame["prev_entry_num"] = rollout_frame.groupby(group_columns)["entry_num"].shift(1)
        rollout_delta_seconds = (rollout_frame["timestamp"] - rollout_frame["prev_timestamp"]).dt.total_seconds()
        rollout_delta_num = rollout_frame["entry_num"] - rollout_frame["prev_entry_num"]
        rollout_frame["rollout/training_time"] = rollout_delta_seconds.where(rollout_delta_num > 0) / rollout_delta_num.where(
            rollout_delta_num > 0
        )
        frame.loc[rollout_frame.index, "rollout/training_time"] = rollout_frame["rollout/training_time"]
    return frame


def available_training_metrics(training_df, entry_type: str) -> list[str]:
    base_columns = {
        "source_label",
        "experiment",
        "log_path",
        "log_name",
        "entry_type",
        "entry_num",
        "timestamp",
        "line_number",
    }
    subset = training_df[training_df["entry_type"] == entry_type]
    return sorted([column for column in subset.columns if column not in base_columns and subset[column].notna().any()])


def default_training_metrics(entry_type: str, metric_options: list[str]) -> list[str]:
    if entry_type == "eval":
        metric_set = set(metric_options)
        default_eval_metrics = [
            metric
            for metric in metric_options
            if metric.endswith("-point") or f"{metric}-point" in metric_set
        ]
        return default_eval_metrics or metric_options[: min(8, len(metric_options))]

    preferred_by_entry_type = {
        "rollout": [
            "rollout/raw_reward",
            "rollout/rewards",
            "rollout/log_probs",
            "rollout/training_time",
            "rollout/response_lengths",
            "rollout/total_lengths",
            "rollout/truncated",
        ],
        "step": [
            "train/entropy_loss",
            "train/tis",
            "train/tis_abs",
            "train/tis_clipfrac",
            "train/pg_clipfrac",
            "self_refine/refine_solve_rate",
            "train/loss",
        ],
        "perf": [
            "perf/train_time",
            "perf/step_time",
            "perf/train_wait_time",
            "perf/wait_time_ratio",
            "self_refine/refine_avg_reward",
            "self_refine/normal_avg_reward",
        ],
    }
    defaults = [metric for metric in preferred_by_entry_type.get(entry_type, []) if metric in metric_options]
    return defaults or metric_options[: min(6, len(metric_options))]


def compute_dynamic_y_range(values: Sequence[Any]) -> list[float] | None:
    numeric_values = [float(value) for value in values if value is not None]
    if not numeric_values:
        return None

    lower = min(numeric_values)
    upper = max(numeric_values)
    if lower == upper:
        padding = max(abs(lower) * 0.05, 0.1)
        return [lower - padding, upper + padding]

    padding = (upper - lower) * 0.08
    return [lower - padding, upper + padding]


def disable_shared_y_axes(fig) -> None:
    fig.update_yaxes(matches=None, title_text=None, autorange=True, showticklabels=True)
    fig.update_xaxes(matches=None, showticklabels=True)


def build_training_line_style_maps(px, line_labels: Sequence[str], source_labels: Sequence[str]) -> tuple[dict[str, str], dict[str, str]]:
    color_palette = px.colors.qualitative.Plotly
    color_map = {line_label: color_palette[index % len(color_palette)] for index, line_label in enumerate(line_labels)}
    dash_styles = ["solid", "dot", "dash", "dashdot"]
    dash_map = {label: dash_styles[index % len(dash_styles)] for index, label in enumerate(source_labels)}
    return color_map, dash_map


def make_metric_figure(
    px,
    frame,
    metrics: list[str],
    title: str,
    x_col: str = "entry_num",
    color_map: dict[str, str] | None = None,
    dash_map: dict[str, str] | None = None,
):
    import math
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    subplot_cols = 2
    subplot_rows = math.ceil(len(metrics) / subplot_cols)
    fig = make_subplots(
        rows=subplot_rows,
        cols=subplot_cols,
        subplot_titles=metrics,
        shared_xaxes=False,
        shared_yaxes=False,
    )

    frame = frame.copy()
    frame["line_label"] = frame.apply(
        lambda row: format_training_line_label(row["source_label"], row["experiment"], row["log_name"]),
        axis=1,
    )
    line_labels = frame["line_label"].dropna().unique().tolist()
    source_labels = sorted(frame["source_label"].dropna().unique().tolist())
    if color_map is None or dash_map is None:
        color_map, dash_map = build_training_line_style_maps(px, line_labels, source_labels)
    legend_seen: set[str] = set()

    for metric_index, metric in enumerate(metrics):
        row = (metric_index // subplot_cols) + 1
        col = (metric_index % subplot_cols) + 1
        metric_frame = frame.dropna(subset=[metric]).sort_values(["experiment", "log_path", x_col])
        for (source_label, experiment, log_name, log_path, line_label), group in metric_frame.groupby(
            ["source_label", "experiment", "log_name", "log_path", "line_label"],
            dropna=False,
        ):
            legend_key = str(line_label)
            fig.add_trace(
                go.Scatter(
                    x=group[x_col],
                    y=group[metric],
                    mode="lines+markers",
                    name=str(line_label),
                    legendgroup=str(line_label),
                    showlegend=legend_key not in legend_seen,
                    line={"color": color_map.get(line_label), "dash": dash_map.get(source_label, "solid")},
                    hovertemplate=(
                        f"metric={metric}<br>"
                        f"experiment={experiment}<br>"
                        f"source={source_label}<br>"
                        f"log={log_name}<br>"
                        f"{x_col}=%{{x}}<br>"
                        "value=%{y}<extra></extra>"
                    ),
                ),
                row=row,
                col=col,
            )
            legend_seen.add(legend_key)
        dynamic_range = compute_dynamic_y_range(metric_frame[metric].tolist())
        if dynamic_range is not None:
            fig.update_yaxes(range=dynamic_range, row=row, col=col)

    fig.update_layout(
        height=max(420, 320 * subplot_rows),
        title={"text": title, "y": 0.98},
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.06,
            "xanchor": "left",
            "x": 0.0,
        },
        margin={"t": 140},
        legend_title_text="Experiment",
    )
    return fig


def summarize_eval_json_file(file_path: str) -> tuple[dict[str, Any] | None, list[str]]:
    warnings: list[str] = []
    path = Path(file_path)
    match = EVAL_FILE_RE.match(path.name)
    if not match:
        return None, [f"Unsupported eval filename: {path}"]
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return None, [f"Failed to read {path}: {exc}"]
    if not isinstance(payload, list):
        return None, [f"Skipped non-list eval JSON: {path}"]

    dataset = match.group("dataset")
    rollout = int(match.group("rollout"))
    experiment = path.parent.name
    source_label = source_label_for_root(path.parent.parent)

    completed_count = 0
    truncated_count = 0
    score_values: list[float] = []
    point_values: list[float] = []
    acc_values: list[bool] = []
    response_lengths: list[float] = []
    repetition_flags: list[float] = []
    exception_counts = {
        "anti_hack_reject": 0,
        "endoftext": 0,
        "endoftext_no_think": 0,
        "endoftext_after_think": 0,
        "think_close_count_not_one": 0,
        "think_close_zero": 0,
        "think_close_multi": 0,
        "contains_think_open": 0,
        "bad_special_token": 0,
        "bad_im_end": 0,
        "repetition_raw": 0,
        "repetition_answer": 0,
    }
    anti_hack_reason_counts: dict[str, int] = {}

    for row in payload:
        if not isinstance(row, dict):
            continue
        reward_data = normalize_reward(row.get("reward"))
        status = row.get("status")
        if status == "completed":
            completed_count += 1
        elif status == "truncated":
            truncated_count += 1
        if reward_data["score"] is not None:
            score_values.append(reward_data["score"])
        if reward_data["point"] is not None:
            point_values.append(reward_data["point"])
        if reward_data["acc"] is not None:
            acc_values.append(reward_data["acc"])
        response_length = coerce_float(row.get("response_length"))
        if response_length is not None:
            response_lengths.append(response_length)
        response = row.get("response")
        if isinstance(response, str):
            exception_data = classify_generation_exception(response)
            repetition_flags.append(1.0 if exception_data["repetition_raw"] else 0.0)
            exception_counts["anti_hack_reject"] += int(exception_data["anti_hack_reject"])
            exception_counts["endoftext"] += int(exception_data["has_endoftext"])
            exception_counts["endoftext_no_think"] += int(exception_data["endoftext_no_think"])
            exception_counts["endoftext_after_think"] += int(exception_data["endoftext_after_think"])
            exception_counts["think_close_count_not_one"] += int(exception_data["think_close_count_not_one"])
            exception_counts["think_close_zero"] += int(exception_data["think_close_zero"])
            exception_counts["think_close_multi"] += int(exception_data["think_close_multi"])
            exception_counts["contains_think_open"] += int(exception_data["contains_think_open"])
            exception_counts["bad_special_token"] += int(exception_data["bad_special_token"])
            exception_counts["bad_im_end"] += int(exception_data["bad_im_end"])
            exception_counts["repetition_raw"] += int(exception_data["repetition_raw"])
            exception_counts["repetition_answer"] += int(exception_data["repetition_answer"])
            reason = str(exception_data["anti_hack_reason"])
            anti_hack_reason_counts[reason] = anti_hack_reason_counts.get(reason, 0) + 1

    sample_count = len([row for row in payload if isinstance(row, dict)])
    truncated_ratio = truncated_count / sample_count if sample_count else None
    point_value = (sum(point_values) / len(point_values) if point_values else None)
    exception_metrics: dict[str, Any] = {}
    for name, count in exception_counts.items():
        exception_metrics[f"{name}_count"] = count
        exception_metrics[f"{name}_frac"] = count / sample_count if sample_count else None
    for reason, count in anti_hack_reason_counts.items():
        exception_metrics[f"anti_hack_reason/{reason}"] = count
    summary_row = {
        "source_label": source_label,
        "experiment": experiment,
        "dataset": dataset,
        "rollout": rollout,
        "file_path": str(path),
        "sample_count": sample_count,
        "completed_count": completed_count,
        "truncated_count": truncated_count,
        "accuracy_rate": (sum(1.0 if item else 0.0 for item in acc_values) / len(acc_values) if acc_values else None),
        "score_mean": (sum(score_values) / len(score_values) if score_values else None),
        "point": point_value,
        "point_mean": point_value,
        "repetition_frac": (sum(repetition_flags) / len(repetition_flags) if repetition_flags else None),
        "truncated_ratio": truncated_ratio,
        "truncation_rate": truncated_ratio,
        "trunc_frac": truncated_ratio,
        "response_length_mean": (sum(response_lengths) / len(response_lengths) if response_lengths else None),
        **exception_metrics,
    }
    training_metric_index = load_training_eval_metric_index(str(path.parent))
    summary_row.update(training_metric_index.get((dataset, rollout), {}))
    return summary_row, warnings


def load_eval_json_summaries(file_paths: Sequence[str]) -> tuple[list[dict[str, Any]], list[str]]:
    summary_rows: list[dict[str, Any]] = []
    warnings: list[str] = []

    for file_path in file_paths:
        summary_row, file_warnings = summarize_eval_json_file(file_path)
        warnings.extend(file_warnings)
        if summary_row is not None:
            summary_rows.append(summary_row)

    return summary_rows, warnings


def load_eval_json_samples(file_path: str) -> tuple[list[dict[str, Any]], list[str]]:
    path = Path(file_path)
    match = EVAL_FILE_RE.match(path.name)
    if not match:
        return [], [f"Unsupported eval filename: {path}"]
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return [], [f"Failed to read {path}: {exc}"]
    if not isinstance(payload, list):
        return [], [f"Skipped non-list eval JSON: {path}"]

    dataset = match.group("dataset")
    rollout = int(match.group("rollout"))
    experiment = path.parent.name
    source_label = source_label_for_root(path.parent.parent)
    sample_rows: list[dict[str, Any]] = []

    for row_id, row in enumerate(payload):
        if not isinstance(row, dict):
            continue
        reward_data = normalize_reward(row.get("reward"))
        status = row.get("status")
        response_length = coerce_float(row.get("response_length"))
        response = row.get("response") or ""
        exception_data = classify_generation_exception(response)
        reward_raw = row.get("reward")
        reward_details = ""
        if reward_raw is not None:
            if isinstance(reward_raw, (dict, list)):
                reward_details = json.dumps(reward_raw, ensure_ascii=True, indent=2, sort_keys=isinstance(reward_raw, dict))
            else:
                reward_details = str(reward_raw)
        sample_rows.append(
            {
                "row_id": row_id,
                "source_label": source_label,
                "experiment": experiment,
                "dataset": dataset,
                "rollout": rollout,
                "file_path": str(path),
                "index": row.get("index"),
                "status": status,
                "response_length": response_length,
                "score": reward_data["score"],
                "point": reward_data["point"],
                "acc": reward_data["acc"],
                "label_text": label_to_text(row.get("label")),
                "prompt": row.get("prompt") or "",
                "response": response,
                "reward_details": reward_details,
                "is_truncated": status == "truncated",
                "anti_hack_reason": exception_data["anti_hack_reason"],
                "anti_hack_reject": exception_data["anti_hack_reject"],
                "has_endoftext": exception_data["has_endoftext"],
                "endoftext_no_think": exception_data["endoftext_no_think"],
                "endoftext_after_think": exception_data["endoftext_after_think"],
                "think_close_count": exception_data["think_close_count"],
                "think_open_count": exception_data["think_open_count"],
                "think_close_count_not_one": exception_data["think_close_count_not_one"],
                "think_close_zero": exception_data["think_close_zero"],
                "think_close_multi": exception_data["think_close_multi"],
                "contains_think_open": exception_data["contains_think_open"],
                "bad_special_token": exception_data["bad_special_token"],
                "bad_im_end": exception_data["bad_im_end"],
                "repetition_raw": exception_data["repetition_raw"],
                "repetition_answer": exception_data["repetition_answer"],
            }
        )

    return sample_rows, []


def build_eval_summary_dataframe(summary_rows: list[dict[str, Any]]):
    import pandas as pd

    frame = pd.DataFrame(summary_rows)
    if frame.empty:
        return frame
    frame = frame.sort_values(["source_label", "dataset", "rollout", "experiment"]).reset_index(drop=True)
    frame["aligned_rollout"] = frame.groupby(["source_label", "experiment", "dataset"])["rollout"].transform(
        lambda column: column - column.min()
    )
    return frame


def build_eval_samples_dataframe(sample_rows: list[dict[str, Any]]):
    import pandas as pd

    frame = pd.DataFrame(sample_rows)
    if frame.empty:
        return frame
    return frame.sort_values(["file_path", "row_id"]).reset_index(drop=True)


def format_training_label(meta: TrainingLogMeta) -> str:
    return f"{meta.source_label} | {meta.experiment} | {meta.log_name}"


def format_training_line_label(source_label: str, experiment: str, log_name: str) -> str:
    return f"{source_label} | {experiment} | {log_name}"


def format_eval_label(meta: EvalFileMeta) -> str:
    return f"{meta.source_label} | {meta.experiment} | {meta.dataset} | rollout {meta.rollout}"


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv or sys.argv[1:])
    try:
        render_streamlit_app(args)
    except ModuleNotFoundError as exc:
        if exc.name == "streamlit":
            raise SystemExit("streamlit is not installed in the active Python environment.") from exc
        raise


def render_streamlit_app(args: argparse.Namespace) -> None:
    import pandas as pd
    import plotly.express as px
    import streamlit as st

    st.set_page_config(page_title="Log Dashboard", layout="wide")
    st.title("Training And Eval Log Dashboard")
    st.caption("Notebook-style flow: select logs first, then parse only what you selected.")

    @st.cache_data(show_spinner=False)
    def cached_training_parse(log_paths: tuple[str, ...]):
        return load_training_logs(log_paths)

    @st.cache_data(show_spinner=False)
    def cached_eval_parse(file_paths: tuple[str, ...], cache_version: int):
        _ = cache_version
        return load_eval_json_summaries(file_paths)

    @st.cache_data(show_spinner=False)
    def cached_eval_samples(file_path: str, cache_version: int):
        _ = cache_version
        return load_eval_json_samples(file_path)

    if st.button("Refresh cache"):
        st.cache_data.clear()
        st.rerun()

    roots_tuple = tuple(args.logs_roots)
    training_meta = discover_training_logs(roots_tuple, args.training_glob)
    eval_meta = discover_eval_files(roots_tuple, args.eval_glob)
    training_default_experiments = default_recent_experiments(
        training_meta,
        sorted({item.experiment for item in training_meta}, key=experiment_sort_key),
        fallback_limit=8,
    )

    st.write(f"Discovered `{len(training_meta)}` training logs and `{len(eval_meta)}` eval JSON files.")

    training_tab, eval_tab = st.tabs(["Training Logs", "Eval JSON Samples"])

    with training_tab:
        render_training_tab(
            st,
            px,
            pd,
            training_meta,
            cached_training_parse,
            args.default_training_logs,
            training_default_experiments,
        )

    with eval_tab:
        render_eval_tab(
            st,
            px,
            pd,
            eval_meta,
            cached_eval_parse,
            cached_eval_samples,
            args.default_eval_files,
            training_default_experiments,
        )


def render_training_tab(
    st,
    px,
    pd,
    training_meta: list[TrainingLogMeta],
    cached_training_parse,
    default_count: int,
    preferred_experiments: list[str],
) -> None:
    if not training_meta:
        st.warning("No training logs found.")
        return

    st.subheader("Training Logs")
    st.caption("Default selections prefer logs active within the last 3 hours.")
    root_options = sorted({item.source_label for item in training_meta})
    selected_roots = st.multiselect("Roots", root_options, default=root_options, key="training_roots")

    filtered_meta = [item for item in training_meta if item.source_label in selected_roots]
    experiment_options = sorted({item.experiment for item in filtered_meta}, key=experiment_sort_key)
    default_experiments = filter_in_order(preferred_experiments, experiment_options)
    if not default_experiments:
        default_experiments = default_recent_experiments(filtered_meta, experiment_options, fallback_limit=8)
    selected_experiments = st.multiselect(
        "Experiments",
        experiment_options,
        default=default_experiments,
        key="training_experiments",
    )
    if selected_experiments:
        filtered_meta = [item for item in filtered_meta if item.experiment in selected_experiments]
    st.session_state["linked_training_roots"] = selected_roots
    st.session_state["linked_training_experiments"] = selected_experiments if selected_experiments else experiment_options

    labels = [format_training_label(item) for item in filtered_meta]
    default_labels = default_recent_labels(filtered_meta, labels, fallback_limit=default_count)
    selected_labels = st.multiselect("Training logs", labels, default=default_labels, key="training_logs")
    selected_meta = [item for item in filtered_meta if format_training_label(item) in selected_labels]
    selected_paths = tuple(item.log_path for item in selected_meta)

    if not selected_paths:
        st.info("Select at least one training log.")
        return

    with st.spinner(f"Parsing {len(selected_paths)} training log(s)..."):
        training_rows, training_warnings = cached_training_parse(selected_paths)
    training_df = build_training_dataframe(training_rows)

    if training_warnings:
        with st.expander("Training warnings"):
            for warning in training_warnings:
                st.write(f"- {warning}")

    if training_df.empty:
        st.warning("No numeric training metrics were parsed from the selected logs.")
        return

    k1, k2, k3 = st.columns(3)
    k1.metric("Selected logs", len(selected_paths))
    k2.metric("Parsed rows", len(training_df))
    k3.metric("Entry types", training_df["entry_type"].nunique())

    selected_line_labels = [format_training_label(item) for item in selected_meta]
    selected_source_labels = sorted({item.source_label for item in selected_meta})
    color_map, dash_map = build_training_line_style_maps(px, selected_line_labels, selected_source_labels)

    st.subheader("Default Overview")
    rollout_defaults = [
        metric
        for metric in [
            "rollout/raw_reward",
            "rollout/rewards",
            "rollout/log_probs",
            "rollout/training_time",
            "rollout/response_lengths",
            "rollout/total_lengths",
        ]
        if metric in available_training_metrics(training_df, "rollout")
    ]
    perf_defaults = [
        metric
        for metric in [
            "self_refine/refine_avg_reward",
            "self_refine/normal_avg_reward",
        ]
        if metric in available_training_metrics(training_df, "perf")
    ]
    step_defaults = [
        metric
        for metric in [
            "train/entropy_loss",
            "train/tis",
            "train/tis_abs",
            "train/tis_clipfrac",
            "train/pg_clipfrac",
        ]
        if metric in available_training_metrics(training_df, "step")
    ]
    overview_metrics = rollout_defaults + step_defaults + perf_defaults
    overview_entry_types = ["rollout", "step"]
    if perf_defaults:
        overview_entry_types.append("perf")
    overview_df = training_df[training_df["entry_type"].isin(overview_entry_types)].copy()
    if overview_metrics and not overview_df.empty:
        overview_fig = make_metric_figure(
            px,
            overview_df,
            overview_metrics,
            "",
            color_map=color_map,
            dash_map=dash_map,
        )
        st.plotly_chart(overview_fig, use_container_width=True)

    with st.expander("ExGRPO / Bucket metrics", expanded=True):
        perf_metric_options = available_training_metrics(training_df, "perf")
        bucket_metrics = sorted([metric for metric in perf_metric_options if metric.startswith("bucket/")])
        exgrpo_metrics_requested = [
            "exgrpo/pool_size",
            "exgrpo/retired_size",
            "exgrpo/total_experiences",
            "exgrpo/batch_experience_count",
            "exgrpo/selected_entropy_mean",
        ]
        exgrpo_metrics = [metric for metric in exgrpo_metrics_requested if metric in perf_metric_options]
        exgrpo_section_metrics = exgrpo_metrics + bucket_metrics

        if not exgrpo_section_metrics:
            st.info("No ExGRPO/Bucket metrics found under `perf` entries in the selected logs.")
        else:
            perf_df = training_df[training_df["entry_type"] == "perf"].copy()
            if perf_df.empty:
                st.info("No `perf` entries found in the selected logs.")
            else:
                exgrpo_fig = make_metric_figure(
                    px,
                    perf_df,
                    exgrpo_section_metrics,
                    "",
                    color_map=color_map,
                    dash_map=dash_map,
                )
                st.plotly_chart(exgrpo_fig, use_container_width=True)

                latest_perf = (
                    perf_df.sort_values(["log_path", "entry_num"])
                    .groupby(["source_label", "experiment", "log_name"], as_index=False)
                    .tail(1)
                    .reset_index(drop=True)
                )
                st.caption("Latest `perf` values for the selected logs.")
                st.dataframe(
                    latest_perf[
                        ["source_label", "experiment", "log_name", "entry_num"]
                        + exgrpo_section_metrics[: min(24, len(exgrpo_section_metrics))]
                    ],
                    use_container_width=True,
                    hide_index=True,
                    height=240,
                )

    entry_type = st.selectbox("Entry type", sorted(training_df["entry_type"].unique().tolist()), key="training_entry_type")
    metric_options = available_training_metrics(training_df, entry_type)
    metric_search = st.text_input("Metric search", key="training_metric_search")
    if metric_search:
        metric_options = [metric for metric in metric_options if metric_search.lower() in metric.lower()]
    default_metrics = default_training_metrics(entry_type, metric_options)
    selected_metrics = st.multiselect(
        "Metrics",
        metric_options,
        default=default_metrics,
        key="training_metrics",
    )
    if not selected_metrics:
        st.info("Select at least one metric.")
        return

    plot_df = training_df[training_df["entry_type"] == entry_type].copy()
    fig = make_metric_figure(
        px,
        plot_df,
        selected_metrics,
        "",
        color_map=color_map,
        dash_map=dash_map,
    )
    st.plotly_chart(fig, use_container_width=True)

    latest = (
        plot_df.sort_values(["log_path", "entry_num"])
        .groupby(["source_label", "experiment", "log_name"], as_index=False)
        .tail(1)
        .reset_index(drop=True)
    )
    st.subheader("Latest values")
    st.dataframe(
        latest[["source_label", "experiment", "log_name", "entry_num"] + selected_metrics[: min(8, len(selected_metrics))]],
        use_container_width=True,
        hide_index=True,
        height=280,
    )


def render_eval_tab(
    st,
    px,
    pd,
    eval_meta: list[EvalFileMeta],
    cached_eval_parse,
    cached_eval_samples,
    default_count: int,
    preferred_experiments: list[str],
) -> None:
    if not eval_meta:
        st.warning("No eval JSON files found.")
        return

    st.subheader("Eval JSON Samples")
    training_selected_roots = st.session_state.get("linked_training_roots")
    training_selected_experiments = st.session_state.get("linked_training_experiments")
    if training_selected_roots:
        filtered_meta = [item for item in eval_meta if item.source_label in training_selected_roots]
    else:
        filtered_meta = list(eval_meta)

    experiment_options = sorted({item.experiment for item in filtered_meta}, key=experiment_sort_key)
    if training_selected_experiments:
        filtered_meta = [item for item in filtered_meta if item.experiment in set(training_selected_experiments)]
        followed_experiments = [exp for exp in experiment_options if exp in set(training_selected_experiments)]
    else:
        followed_experiments = filter_in_order(preferred_experiments, experiment_options)
        if not followed_experiments:
            followed_experiments = default_recent_experiments(filtered_meta, experiment_options, fallback_limit=2)
        filtered_meta = [item for item in filtered_meta if item.experiment in set(followed_experiments)]

    st.caption("Following roots and experiments selected in the Training Logs tab.")
    if training_selected_roots:
        st.write("Training roots:", ", ".join(training_selected_roots))
    if training_selected_experiments:
        st.write("Training experiments:", ", ".join(training_selected_experiments))
    elif followed_experiments:
        st.write("Training experiments:", ", ".join(followed_experiments))

    dataset_options = sorted({item.dataset for item in filtered_meta})
    selected_datasets = st.multiselect("Datasets", dataset_options, default=dataset_options, key="eval_datasets")
    if selected_datasets:
        filtered_meta = [item for item in filtered_meta if item.dataset in selected_datasets]

    labels = [format_eval_label(item) for item in filtered_meta]
    default_labels = labels if labels else []
    selected_labels = st.multiselect("Eval files", labels, default=default_labels, key="eval_files")
    selected_paths = tuple(item.file_path for item in filtered_meta if format_eval_label(item) in selected_labels)

    if not selected_paths:
        st.info("Select at least one eval JSON file.")
        return

    with st.spinner(f"Parsing {len(selected_paths)} eval file(s)..."):
        summary_rows, warnings = cached_eval_parse(selected_paths, EVAL_SUMMARY_CACHE_VERSION)

    if warnings:
        with st.expander("Eval warnings"):
            for warning in warnings:
                st.write(f"- {warning}")

    summary_df = build_eval_summary_dataframe(summary_rows)

    if summary_df.empty:
        st.warning("No eval data parsed from the selected files.")
        return

    if "point" not in summary_df.columns and "point_mean" in summary_df.columns:
        summary_df["point"] = summary_df["point_mean"]
    display_columns = [column for column in summary_df.columns if column != "point_mean"]

    st.dataframe(summary_df[display_columns], use_container_width=True, hide_index=True, height=280)

    detail_metrics = [
        "score_mean",
        "point",
        "response_length_mean",
        "anti_hack_reject_frac",
        "endoftext_no_think_frac",
        "endoftext_after_think_frac",
        "think_close_count_not_one_frac",
        "think_close_multi_frac",
        "contains_think_open_frac",
        "bad_special_token_frac",
        "repetition_raw_frac",
        "repetition_frac",
        "truncated_ratio",
    ]
    for detail_metric in detail_metrics:
        if detail_metric not in summary_df.columns:
            summary_df[detail_metric] = None
    detail_df = (
        summary_df.melt(
            id_vars=["source_label", "experiment", "dataset", "aligned_rollout", "file_path"],
            value_vars=detail_metrics,
            var_name="metric_name",
            value_name="metric_value",
        )
        .dropna(subset=["metric_value"])
        .reset_index(drop=True)
    )
    detail_fig = px.line(
        detail_df,
        x="aligned_rollout",
        y="metric_value",
        color="experiment",
        line_dash="source_label",
        facet_col="dataset" if summary_df["dataset"].nunique() > 1 else None,
        facet_row="metric_name",
        facet_row_spacing=0.04,
        category_orders={"metric_name": detail_metrics},
        markers=True,
        title=None,
    )
    disable_shared_y_axes(detail_fig)
    detail_fig.update_layout(
        height=max(1120, 180 * len(detail_metrics)),
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.04,
            "xanchor": "left",
            "x": 0.0,
        },
        margin={"t": 90},
    )
    st.plotly_chart(detail_fig, use_container_width=True)

    sample_file = st.selectbox("Inspect file", summary_df["file_path"].tolist(), key="eval_inspect_file")
    sample_rows, sample_warnings = cached_eval_samples(sample_file, EVAL_SUMMARY_CACHE_VERSION)
    if sample_warnings:
        for warning in sample_warnings:
            st.warning(warning)
        return
    samples_df = build_eval_samples_dataframe(sample_rows)
    if samples_df.empty:
        st.info("No samples available for the selected file.")
        return
    file_samples = samples_df[samples_df["file_path"] == sample_file].copy()
    exception_metric_cols = [
        "anti_hack_reject",
        "has_endoftext",
        "endoftext_no_think",
        "endoftext_after_think",
        "think_close_count_not_one",
        "think_close_zero",
        "think_close_multi",
        "contains_think_open",
        "bad_special_token",
        "bad_im_end",
        "repetition_raw",
        "repetition_answer",
    ]
    if "anti_hack_reason" not in file_samples.columns:
        exception_rows = file_samples["response"].fillna("").map(classify_generation_exception)
        file_samples["anti_hack_reason"] = exception_rows.map(lambda item: item["anti_hack_reason"])
        file_samples["think_close_count"] = exception_rows.map(lambda item: item["think_close_count"])
        file_samples["think_open_count"] = exception_rows.map(lambda item: item["think_open_count"])
        for column in exception_metric_cols:
            file_samples[column] = exception_rows.map(lambda item, key=column: item[key])
    else:
        for column in ["think_close_count", "think_open_count"] + exception_metric_cols:
            if column not in file_samples.columns:
                file_samples[column] = False
    selected_file_summary = summary_df[summary_df["file_path"] == sample_file].iloc[0]
    metric_cols = st.columns(5)
    metric_cols[0].metric("Anti-hack rejects", int(selected_file_summary.get("anti_hack_reject_count") or 0))
    metric_cols[1].metric("<|endoftext|>", int(selected_file_summary.get("endoftext_count") or 0))
    metric_cols[2].metric("EOT no </think>", int(selected_file_summary.get("endoftext_no_think_count") or 0))
    metric_cols[3].metric("</think> count != 1", int(selected_file_summary.get("think_close_count_not_one_count") or 0))
    metric_cols[4].metric("Repetition raw", int(selected_file_summary.get("repetition_raw_count") or 0))

    reason_options = sorted(file_samples["anti_hack_reason"].dropna().unique().tolist())
    non_pass_reasons = [reason for reason in reason_options if reason != "pass"]
    default_reasons = non_pass_reasons or reason_options
    selected_reasons = st.multiselect(
        "Exception reason filter",
        reason_options,
        default=default_reasons,
        key="eval_exception_reason_filter",
    )
    filtered_samples = file_samples[file_samples["anti_hack_reason"].isin(selected_reasons)].copy()
    show_only_exceptions = st.checkbox(
        "Show only exception samples",
        value=bool(non_pass_reasons),
        key="eval_show_only_exceptions",
    )
    if show_only_exceptions:
        filtered_samples = filtered_samples[filtered_samples["anti_hack_reject"]]
    if filtered_samples.empty:
        st.info("No samples match the current exception filters.")
        return

    sample_table_columns = [
        "row_id",
        "index",
        "status",
        "response_length",
        "score",
        "point",
        "acc",
        "anti_hack_reason",
        "think_close_count",
        "think_open_count",
    ] + exception_metric_cols
    st.dataframe(
        filtered_samples[sample_table_columns],
        use_container_width=True,
        hide_index=True,
        height=280,
    )

    selected_row = st.selectbox("Inspect sample row", filtered_samples["row_id"].tolist(), key="eval_inspect_row")
    sample = filtered_samples[filtered_samples["row_id"] == selected_row].iloc[0]
    with st.expander("Exception details", expanded=True):
        st.json({column: sample[column] for column in ["anti_hack_reason", "think_close_count", "think_open_count"] + exception_metric_cols})
    with st.expander("Label"):
        st.code(sample["label_text"] or "", language="text")
    with st.expander("Prompt"):
        st.code(sample["prompt"] or "", language="text")
    with st.expander("Response", expanded=True):
        st.code(sample["response"] or "", language="text")
    with st.expander("Reward details"):
        st.code(sample["reward_details"] or "", language="json")


if __name__ == "__main__":
    main()
