#!/usr/bin/env python3
"""FrontierScience judge-only evaluation script.

This script follows the evaluation protocol described in the
"FrontierScience: Evaluating AI's Ability to Perform Expert-Level Scientific
Tasks" paper:

- Olympiad:
  - evaluate against the official short answer using a judge model
  - metric: accuracy
  - paper reports the mean accuracy across 20 independent trials

- Research:
  - evaluate against the official 10-point rubric using a judge model
  - metric: success rate where score >= 7.0 counts as success
  - paper reports the mean success rate across 30 independent trials

The script supports:
- official dataset auto-download from Hugging Face
- a single pre-generated prediction file or a directory of multiple trial files
- resume from partially judged outputs
- serial and concurrent judge execution
"""

from __future__ import annotations

import argparse
import json
import os
import re
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
from urllib.request import urlopen

from openai import OpenAI
from tqdm import tqdm


OFFICIAL_DATASET_REPO = "openai/frontierscience"
OFFICIAL_DATASET_FILES = {
    "olympiad": "olympiad/test.jsonl",
    "research": "research/test.jsonl",
}
COMPOUND_ROLLOUT_INDEX_PATTERN = re.compile(
    r"^\s*(?P<question_id>\d+)\s*-\s*(?P<rollout_id>\d+)\s*$"
)
PAPER_RECOMMENDED_TRIALS = {
    "olympiad": 20,
    "research": 30,
}
RESEARCH_SUCCESS_THRESHOLD = 7.0
REQUEST_TIMEOUT = 1200
RETRY_TIMES = 3
DEFAULT_MAX_TOKENS = 32768
DEFAULT_TEMPERATURE = 0.0


OLYMPIAD_JUDGE_TEMPLATE = """You are grading an attempted answer to a science olympiad problem.
You will be given the problem, attempted answer, and reference answer.
Evaluate the attempted answer against the reference answer carefully and strictly, but fairly.

The reference answer is either a single number or expression in LaTeX formatting,
a chemical formula, a compound name, or a phrase referring to a specific entity or method.
Mark the attempted answer as correct if it fully matches the reference answer or is otherwise
equivalent, such as:
- an algebraically equivalent expression
- a numerical answer consistent up to 1 decimal place rounding
- an equivalent compound name or formula
- an equivalent answer after accounting for units

Mark the attempted answer as incorrect if it is not equivalent to the reference answer.

***
The problem:{problem}
***
The reference answer:{reference_answer}
***
The attempted answer:{attempted_answer}
***
First, think step-by-step about whether the attempted answer matches the reference answer.
If the attempted answer is correct, write "VERDICT: CORRECT" in the last line of your response, with no other text or formatting.
If the attempted answer is incorrect, write "VERDICT: INCORRECT" in the last line of your response.
"""


RESEARCH_JUDGE_TEMPLATE = """You are grading a science exam.
You will be given the problem, attempted answer, and a rubric to grade the answer.
The rubric totals up to 10 points.

Evaluate the attempted answer against the provided rubric carefully and strictly, but fairly.
Only evaluate against the rubric. Even if you personally disagree with the rubric, treat the rubric as the gold standard.
Return the absolute total number of points earned. The score can be a decimal when the rubric allows it.

***
The problem:{problem}
***
The rubric:{rubric}
***
The attempted answer:{attempted_answer}
***
First, think step-by-step about each rubric item.
Explain your reasoning for each rubric item.
Then tally the points and write "VERDICT: <total_points>" in the last line of your response, with no other text after it.
For example: "VERDICT: 2.5" or "VERDICT: 8".
"""


def load_json_or_jsonl(path: Path) -> List[Dict[str, Any]]:
    if path.suffix == ".jsonl":
        with path.open("r", encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]
    if path.suffix == ".json":
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            raise ValueError(f"{path} must contain a JSON list.")
        return data
    raise ValueError(f"Unsupported file format: {path}")


def save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def stringify_content(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (int, float, bool)):
        return str(value)
    if isinstance(value, list):
        pieces: List[str] = []
        for item in value:
            if isinstance(item, str):
                pieces.append(item)
            elif isinstance(item, dict):
                if "text" in item:
                    pieces.append(str(item["text"]))
                elif item.get("type") == "output_text" and "text" in item:
                    pieces.append(str(item["text"]))
                elif item.get("type") == "text" and "text" in item:
                    pieces.append(str(item["text"]))
                elif "content" in item:
                    pieces.append(stringify_content(item["content"]))
                else:
                    pieces.append(json.dumps(item, ensure_ascii=False))
            else:
                pieces.append(str(item))
        return "\n".join(piece for piece in pieces if piece)
    if isinstance(value, dict):
        if "text" in value:
            return stringify_content(value["text"])
        if "content" in value:
            return stringify_content(value["content"])
        if "message" in value:
            return stringify_content(value["message"])
        if "choices" in value and value["choices"]:
            return stringify_content(value["choices"][0])
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def clean_response_text(text: str) -> str:
    cleaned = text.replace("\r\n", "\n").replace("\r", "\n").strip()
    if "</think>" in cleaned:
        cleaned = cleaned.split("</think>")[-1].strip()
    cleaned = cleaned.removesuffix("<|im_end|>").strip()
    return cleaned


def make_json_serializable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {
            str(key): make_json_serializable(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [make_json_serializable(item) for item in value]
    if hasattr(value, "model_dump"):
        try:
            return make_json_serializable(value.model_dump(mode="json"))
        except TypeError:
            return make_json_serializable(value.model_dump())
    if hasattr(value, "to_dict"):
        return make_json_serializable(value.to_dict())
    return str(value)


def extract_final_answer_block(text: str) -> Optional[str]:
    matches = list(
        re.finditer(
            r"FINAL ANSWER\s*[:：]?\s*(.+)$",
            text,
            flags=re.IGNORECASE | re.DOTALL,
        )
    )
    if not matches:
        return None
    final_answer = matches[-1].group(1).strip()
    return final_answer or None


def extract_assistant_text(item: Dict[str, Any], response_key: Optional[str]) -> str:
    if response_key:
        return clean_response_text(stringify_content(item.get(response_key)))

    candidate_keys = [
        "response",
        "prediction",
        "output",
        "completion",
        "generated_text",
        "assistant_response",
        "model_output",
        "text",
        "content",
    ]
    for key in candidate_keys:
        value = item.get(key)
        if value is not None and stringify_content(value).strip():
            return clean_response_text(stringify_content(value))

    messages = item.get("messages")
    if isinstance(messages, list):
        for message in reversed(messages):
            if isinstance(message, dict) and message.get("role") == "assistant":
                return clean_response_text(stringify_content(message.get("content")))

    choices = item.get("choices")
    if isinstance(choices, list) and choices:
        return clean_response_text(stringify_content(choices[0]))

    raise ValueError(
        "Could not find model response text. "
        "Use --response-key to specify the field explicitly."
    )


def nested_get(item: Dict[str, Any], *keys: str) -> Any:
    current: Any = item
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def resolve_row_value(row: Dict[str, Any], key: str) -> Any:
    if key in row and row.get(key) is not None:
        return row.get(key)
    metadata_value = nested_get(row, "metadata", key)
    if metadata_value is not None:
        return metadata_value
    if key == "problem":
        return row.get("problem") or nested_get(row, "metadata", "question")
    return None


def find_local_official_path(track: str, official_data_path: Optional[str]) -> Optional[Path]:
    if not official_data_path:
        return None

    root = Path(official_data_path)
    if root.is_file():
        return root

    candidates = [
        root / OFFICIAL_DATASET_FILES[track],
        root / f"{track}.jsonl",
        root / f"{track}.json",
        root / "test.jsonl",
        root / "test.json",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"Could not find official FrontierScience data for track={track} under {root}"
    )


def download_official_file(track: str) -> Path:
    relative_path = OFFICIAL_DATASET_FILES[track]
    try:
        from huggingface_hub import hf_hub_download

        return Path(
            hf_hub_download(
                repo_id=OFFICIAL_DATASET_REPO,
                filename=relative_path,
                repo_type="dataset",
            )
        )
    except Exception:
        cache_dir = Path.home() / ".cache" / "frontierscience_eval"
        cache_dir.mkdir(parents=True, exist_ok=True)
        local_path = cache_dir / relative_path.replace("/", "_")
        if local_path.exists():
            return local_path
        url = (
            "https://huggingface.co/datasets/"
            f"{OFFICIAL_DATASET_REPO}/resolve/main/{relative_path}"
        )
        with urlopen(url) as response, local_path.open("wb") as f:
            f.write(response.read())
        return local_path


def normalize_official_row(
    track: str,
    row: Dict[str, Any],
    row_index: int,
) -> Dict[str, Any]:
    if not isinstance(row, dict):
        raise ValueError(
            f"Official FrontierScience row {row_index} is not a JSON object.")

    if row.get("problem") is not None and row.get("answer") is not None:
        normalized = dict(row)
        if normalized.get("subject") is None:
            normalized["subject"] = normalized.get("category")
        if normalized.get("task_group_id") is None:
            normalized["task_group_id"] = normalized.get(
                "problem_id") or normalized.get("id")
        if normalized.get("problem_id") is None:
            normalized["problem_id"] = normalized.get(
                "task_group_id") or normalized.get("id")
        if normalized.get("problem_idx") is None:
            normalized["problem_idx"] = normalized.get("index")
        return normalized

    metadata = row.get("metadata")
    if not isinstance(metadata, dict):
        raise ValueError(
            f"Official FrontierScience row {row_index} has neither raw problem/answer fields "
            "nor processed metadata fields."
        )

    problem = metadata.get("question")
    answer = (
        metadata.get("short_answer")
        if track == "olympiad"
        else metadata.get("grading_guidelines")
    )
    if not problem:
        raise ValueError(
            f"Official FrontierScience row {row_index} missing metadata.question."
        )
    if not answer:
        expected_field = "metadata.short_answer" if track == "olympiad" else "metadata.grading_guidelines"
        raise ValueError(
            f"Official FrontierScience row {row_index} missing {expected_field}."
        )

    return {
        "problem": problem,
        "answer": answer,
        "subject": metadata.get("subject") or metadata.get("category"),
        "category": metadata.get("category") or metadata.get("subject"),
        "task_group_id": metadata.get("task_group_id") or metadata.get("problem_id"),
        "problem_id": metadata.get("problem_id") or metadata.get("task_group_id"),
        "problem_idx": metadata.get("problem_idx"),
        "index": metadata.get("index", row_index),
        "_official_source_format": "processed_prompt_jsonl",
        "_official_original_row": row,
    }


def normalize_official_dataset(
    track: str,
    rows: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    return [
        normalize_official_row(track, row, row_index)
        for row_index, row in enumerate(rows, start=1)
    ]


def load_official_dataset(track: str, official_data_path: Optional[str]) -> List[Dict[str, Any]]:
    local_path = find_local_official_path(track, official_data_path)
    if local_path is None:
        local_path = download_official_file(track)
    return normalize_official_dataset(track, load_json_or_jsonl(local_path))


def resolve_prediction_file(prediction_path: str) -> Path:
    path = Path(prediction_path)
    if not path.exists():
        raise FileNotFoundError(f"Prediction file does not exist: {path}")
    if not path.is_file():
        raise ValueError(
            f"Prediction path must be a single .json or .jsonl file, got: {path}"
        )
    if path.suffix not in {".json", ".jsonl"}:
        raise ValueError(
            f"Unsupported prediction file format: {path}. Expected .json or .jsonl"
        )
    return path


def get_candidate_match_keys(match_by: str) -> List[str]:
    if match_by == "auto":
        return ["problem_idx", "index", "task_group_id", "id", "problem_id", "uuid", "problem"]
    return [match_by]


def find_match_key(
    official_rows: Sequence[Dict[str, Any]],
    prediction_rows: Sequence[Dict[str, Any]],
    match_by: str,
) -> Optional[str]:
    for key in get_candidate_match_keys(match_by):
        official_have_key = all(
            resolve_row_value(row, key) is not None for row in official_rows
        )
        if not official_have_key:
            continue
        prediction_have_key = all(
            resolve_row_value(row, key) is not None for row in prediction_rows
        )
        if not prediction_have_key:
            continue
        return key
    return None


def parse_compound_rollout_index(value: Any) -> Tuple[Optional[int], Optional[int]]:
    if value is None:
        return None, None

    match = COMPOUND_ROLLOUT_INDEX_PATTERN.match(str(value))
    if not match:
        return None, None

    return int(match.group("question_id")), int(match.group("rollout_id"))


def as_positive_int(value: Any) -> Optional[int]:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value if value > 0 else None
    if isinstance(value, str) and value.strip():
        try:
            parsed = int(value.strip())
        except ValueError:
            return None
        return parsed if parsed > 0 else None
    return None


def extract_explicit_rollout_id(row: Dict[str, Any]) -> Optional[int]:
    for key in ("rollout_id", "trial_id", "_rollout_id"):
        rollout_id = as_positive_int(row.get(key))
        if rollout_id is not None:
            return rollout_id

    metadata = row.get("metadata")
    if isinstance(metadata, dict):
        for key in ("rollout_id", "trial_id", "_rollout_id"):
            rollout_id = as_positive_int(metadata.get(key))
            if rollout_id is not None:
                return rollout_id

    for key in ("index", "id"):
        _, rollout_id = parse_compound_rollout_index(row.get(key))
        if rollout_id is not None:
            return rollout_id
    return None


def split_prediction_rows_by_explicit_rollout_id(
    prediction_rows: Sequence[Dict[str, Any]],
) -> Optional[List[Tuple[int, List[Dict[str, Any]]]]]:
    buckets: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    saw_rollout_id = False

    for row in prediction_rows:
        rollout_id = extract_explicit_rollout_id(row)
        if rollout_id is None:
            return None
        saw_rollout_id = True
        buckets[rollout_id].append(row)

    if not saw_rollout_id:
        return None

    return [(rollout_id, buckets[rollout_id]) for rollout_id in sorted(buckets)]


def split_prediction_rows_by_match_key_occurrence(
    official_rows: Sequence[Dict[str, Any]],
    prediction_rows: Sequence[Dict[str, Any]],
    match_key: str,
) -> Optional[List[Tuple[int, List[Dict[str, Any]]]]]:
    official_key_order = [str(resolve_row_value(row, match_key))
                          for row in official_rows]
    grouped_rows: Dict[str, List[Dict[str, Any]]] = {
        key: [] for key in official_key_order}

    for row in prediction_rows:
        key_value = resolve_row_value(row, match_key)
        if key_value is None:
            return None
        normalized_key = str(key_value)
        if normalized_key not in grouped_rows:
            raise ValueError(
                f"Prediction row has {match_key}={normalized_key}, which does not exist in the official FrontierScience data."
            )
        grouped_rows[normalized_key].append(row)

    counts = [len(grouped_rows[key]) for key in official_key_order]
    if any(count == 0 for count in counts):
        return None

    unique_counts = set(counts)
    if len(unique_counts) != 1:
        raise ValueError(
            f"Prediction rows have inconsistent rollout counts per {match_key}: "
            f"{sorted(unique_counts)}. Expected each question to appear the same number of times."
        )

    num_rollouts = unique_counts.pop()
    if num_rollouts <= 1:
        return None

    trials: List[Tuple[int, List[Dict[str, Any]]]] = []
    for rollout_idx in range(num_rollouts):
        trial_rows = [
            grouped_rows[key][rollout_idx]
            for key in official_key_order
        ]
        trials.append((rollout_idx + 1, trial_rows))
    return trials


def build_prediction_trials(
    prediction_file: Path,
    official_rows: Sequence[Dict[str, Any]],
    prediction_rows: Sequence[Dict[str, Any]],
    match_by: str,
) -> List[Tuple[str, List[Dict[str, Any]]]]:
    explicit_rollout_trials = split_prediction_rows_by_explicit_rollout_id(
        prediction_rows)
    if explicit_rollout_trials is not None:
        if len(explicit_rollout_trials) == 1:
            return [(prediction_file.stem, explicit_rollout_trials[0][1])]
        return [
            (f"{prediction_file.stem}.rollout_{rollout_id}", rows)
            for rollout_id, rows in explicit_rollout_trials
        ]

    match_key = find_match_key(official_rows, prediction_rows, match_by)
    if match_key is not None:
        occurrence_trials = split_prediction_rows_by_match_key_occurrence(
            official_rows=official_rows,
            prediction_rows=prediction_rows,
            match_key=match_key,
        )
        if occurrence_trials is not None:
            return [
                (f"{prediction_file.stem}.rollout_{rollout_id}", rows)
                for rollout_id, rows in occurrence_trials
            ]

    return [(prediction_file.stem, list(prediction_rows))]


def build_prediction_lookup(
    prediction_rows: Sequence[Dict[str, Any]],
    key: str,
) -> Optional[Dict[str, Dict[str, Any]]]:
    lookup: Dict[str, Dict[str, Any]] = {}
    for row in prediction_rows:
        value = resolve_row_value(row, key)
        if value is None:
            return None
        normalized_key = str(value)
        if normalized_key in lookup:
            return None
        lookup[normalized_key] = row
    return lookup


def merge_predictions_with_official(
    track: str,
    official_rows: Sequence[Dict[str, Any]],
    prediction_rows: Sequence[Dict[str, Any]],
    response_key: Optional[str],
    match_by: str,
) -> Tuple[List[Dict[str, Any]], str]:
    for key in get_candidate_match_keys(match_by):
        official_have_key = all(
            resolve_row_value(row, key) is not None for row in official_rows
        )
        if not official_have_key:
            continue
        prediction_lookup = build_prediction_lookup(prediction_rows, key)
        if prediction_lookup is None:
            continue

        merged_rows: List[Dict[str, Any]] = []
        missing = False
        for official in official_rows:
            official_key_value = resolve_row_value(official, key)
            pred = prediction_lookup.get(str(official_key_value))
            if pred is None:
                missing = True
                break
            merged = dict(official)
            merged["_prediction_row"] = pred
            merged["response"] = extract_assistant_text(pred, response_key)
            merged_rows.append(merged)
        if not missing:
            return merged_rows, key

    if len(prediction_rows) != len(official_rows):
        raise ValueError(
            f"Could not align predictions with official data for track={track}. "
            f"Official rows: {len(official_rows)}, prediction rows: {len(prediction_rows)}. "
            "Provide rows in the same order, or include a stable key such as task_group_id, "
            "or set --match-by explicitly."
        )

    merged_rows = []
    for official, pred in zip(official_rows, prediction_rows):
        merged = dict(official)
        merged["_prediction_row"] = pred
        merged["response"] = extract_assistant_text(pred, response_key)
        merged_rows.append(merged)
    return merged_rows, "order"


def merge_prediction_groups_with_official(
    track: str,
    official_rows: Sequence[Dict[str, Any]],
    prediction_rows: Sequence[Dict[str, Any]],
    response_key: Optional[str],
    match_by: str,
) -> Tuple[List[Dict[str, Any]], str, int]:
    for key in get_candidate_match_keys(match_by):
        official_have_key = all(
            resolve_row_value(row, key) is not None for row in official_rows
        )
        if not official_have_key:
            continue

        prediction_groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        prediction_have_key = True
        for row in prediction_rows:
            value = resolve_row_value(row, key)
            if value is None:
                prediction_have_key = False
                break
            prediction_groups[str(value)].append(row)
        if not prediction_have_key:
            continue

        grouped_rows: List[Dict[str, Any]] = []
        group_sizes: List[int] = []
        missing = False
        for official in official_rows:
            official_key_value = resolve_row_value(official, key)
            pred_group = prediction_groups.get(str(official_key_value))
            if not pred_group:
                missing = True
                break

            merged = dict(official)
            merged["_prediction_rows"] = pred_group
            merged["responses"] = [
                extract_assistant_text(pred, response_key)
                for pred in pred_group
            ]
            grouped_rows.append(merged)
            group_sizes.append(len(pred_group))

        if missing:
            continue

        unique_group_sizes = set(group_sizes)
        if len(unique_group_sizes) != 1:
            raise ValueError(
                f"Prediction rows have inconsistent candidate counts per {key}: "
                f"{sorted(unique_group_sizes)}. Expected each question to have the same number of outputs."
            )

        return grouped_rows, key, unique_group_sizes.pop()

    if len(prediction_rows) != len(official_rows):
        raise ValueError(
            f"Could not align grouped predictions with official data for track={track}. "
            f"Official rows: {len(official_rows)}, prediction rows: {len(prediction_rows)}. "
            "Include a stable key such as task_group_id/problem_id in the prediction rows."
        )

    grouped_rows = []
    for official, pred in zip(official_rows, prediction_rows):
        merged = dict(official)
        merged["_prediction_rows"] = [pred]
        merged["responses"] = [extract_assistant_text(pred, response_key)]
        grouped_rows.append(merged)
    return grouped_rows, "order", 1


def build_candidate_eval_items(
    grouped_rows: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    global_index = 0

    for group_idx, grouped_row in enumerate(grouped_rows, start=1):
        prediction_rows = grouped_row.get("_prediction_rows", [])
        responses = grouped_row.get("responses", [])
        base_item = {
            key: value
            for key, value in grouped_row.items()
            if key not in {"_prediction_rows", "responses"}
        }

        for candidate_idx, (prediction_row, response_text) in enumerate(
            zip(prediction_rows, responses),
            start=1,
        ):
            global_index += 1
            item = dict(base_item)
            item["_prediction_row"] = prediction_row
            item["_group_data_index"] = group_idx
            item["_candidate_index"] = candidate_idx
            item["_num_candidates"] = len(responses)
            item["response"] = response_text
            item["_candidate_global_index"] = global_index
            items.append(item)

    return items


def fold_candidate_results_into_groups(
    track: str,
    grouped_rows: Sequence[Dict[str, Any]],
    candidate_results: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    results_by_group: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for row in candidate_results:
        group_idx = row.get("_group_data_index")
        if isinstance(group_idx, int):
            results_by_group[group_idx].append(row)

    judged_rows: List[Dict[str, Any]] = []
    for group_idx, grouped_row in enumerate(grouped_rows, start=1):
        candidate_rows = sorted(
            results_by_group.get(group_idx, []),
            key=lambda row: row.get("_candidate_index", 0),
        )
        result_row = {
            key: value
            for key, value in grouped_row.items()
            if key not in {"_prediction_rows", "responses"}
        }
        result_row["candidate_results"] = candidate_rows
        result_row["num_candidates"] = len(candidate_rows)

        if track == "olympiad":
            correct_count = sum(
                1 for row in candidate_rows if row.get("is_correct"))
            result_row["correct_candidate_count"] = correct_count
            result_row["is_correct"] = correct_count > 0
            result_row["score"] = 1.0 if correct_count > 0 else 0.0
            result_row["candidate_accuracy"] = (
                correct_count / len(candidate_rows) if candidate_rows else 0.0
            )
        else:
            points = [float(row.get("points", 0.0)) for row in candidate_rows]
            success_count = sum(
                1 for row in candidate_rows if row.get("is_success"))
            best_points = max(points) if points else 0.0
            result_row["success_candidate_count"] = success_count
            result_row["is_success"] = success_count > 0
            result_row["points"] = best_points
            result_row["score"] = best_points
            result_row["candidate_mean_points"] = mean(
                points) if points else 0.0

        judged_rows.append(result_row)

    return judged_rows


def build_candidate_index_rows(
    judged_rows: Sequence[Dict[str, Any]],
    candidate_results: Sequence[Dict[str, Any]],
    candidate_index: int,
) -> List[Dict[str, Any]]:
    candidate_by_group: Dict[int, Dict[str, Any]] = {}
    for row in candidate_results:
        if row.get("_candidate_index") != candidate_index:
            continue
        group_idx = row.get("_group_data_index")
        if isinstance(group_idx, int):
            candidate_by_group[group_idx] = row

    rows: List[Dict[str, Any]] = []
    for group_idx, judged_row in enumerate(judged_rows, start=1):
        row = dict(judged_row)
        candidate = candidate_by_group.get(group_idx)
        if candidate is None:
            row["is_correct"] = False
            row["is_success"] = False
            row["score"] = 0.0
            row["points"] = 0.0
            row["missing_candidate_index"] = candidate_index
        else:
            row.update(
                {
                    key: value
                    for key, value in candidate.items()
                    if key
                    in {
                        "is_correct",
                        "is_success",
                        "score",
                        "points",
                        "judge_parse_error",
                    }
                }
            )
        rows.append(row)
    return rows


def compute_grouped_summary(
    track: str,
    judged_rows: Sequence[Dict[str, Any]],
    candidate_results: Sequence[Dict[str, Any]],
    prediction_file: Path,
    result_file: Path,
    checkpoint_file: Path,
    matched_by: str,
    group_size: int,
) -> Dict[str, Any]:
    invalid_count = sum(
        1 for row in candidate_results if row.get("judge_parse_error"))
    by_subject = summarize_subject_metrics(track, judged_rows)

    summary: Dict[str, Any] = {
        "track": track,
        "prediction_file": str(prediction_file.resolve()),
        "result_file": str(result_file.resolve()),
        "checkpoint_file": str(checkpoint_file.resolve()),
        "matched_by": matched_by,
        "num_items": len(judged_rows),
        "group_size": group_size,
        "num_candidates_total": len(candidate_results),
        "invalid_judgement_count": invalid_count,
        "paper_recommended_trials": PAPER_RECOMMENDED_TRIALS[track],
        "by_subject": by_subject,
    }

    if track == "olympiad":
        pass_count = sum(1 for row in judged_rows if row.get("is_correct"))
        pass_rate = pass_count / len(judged_rows) if judged_rows else 0.0
        first_candidate_rows = build_candidate_index_rows(
            judged_rows,
            candidate_results,
            candidate_index=1,
        )
        pass_at_1_count = sum(
            1 for row in first_candidate_rows if row.get("is_correct")
        )
        pass_at_1 = (
            pass_at_1_count / len(first_candidate_rows)
            if first_candidate_rows
            else 0.0
        )
        by_subject_pass_at_1 = summarize_subject_metrics(
            track,
            first_candidate_rows,
        )
        by_subject_candidate_accuracy = summarize_subject_metrics(
            track,
            candidate_results,
        )
        candidate_invalid_by_subject: Dict[str, int] = defaultdict(int)
        for row in candidate_results:
            if row.get("judge_parse_error"):
                candidate_invalid_by_subject[row.get("subject", "unknown")] += 1
        for subject, metrics in by_subject.items():
            first_metrics = by_subject_pass_at_1.get(subject, {})
            candidate_metrics = by_subject_candidate_accuracy.get(subject, {})
            metrics["pass_at_k_count"] = metrics.get("correct_count", 0)
            metrics["pass_at_k"] = metrics.get("accuracy", 0.0)
            metrics["pass_at_1_count"] = first_metrics.get("correct_count", 0)
            metrics["pass_at_1"] = first_metrics.get("accuracy", 0.0)
            metrics["accuracy_at_1"] = first_metrics.get("accuracy", 0.0)
            metrics["num_candidates"] = candidate_metrics.get("num_items", 0)
            metrics["candidate_correct_count"] = candidate_metrics.get(
                "correct_count",
                0,
            )
            metrics["candidate_accuracy"] = candidate_metrics.get("accuracy", 0.0)
            metrics["candidate_invalid_judgement_count"] = candidate_invalid_by_subject.get(
                subject,
                0,
            )
        candidate_correct_count = sum(
            1 for row in candidate_results if row.get("is_correct"))
        candidate_accuracy = (
            candidate_correct_count / len(candidate_results)
            if candidate_results else 0.0
        )
        summary.update(
            {
                "paper_metric": f"pass@{group_size}",
                "pass_count": pass_count,
                "correct_count": pass_count,
                "pass_at_k": pass_rate,
                "pass_at_1_count": pass_at_1_count,
                "pass_at_1": pass_at_1,
                "accuracy_at_1": pass_at_1,
                "accuracy": pass_rate,
                "mean_accuracy": pass_rate,
                "mean_accuracy_at_1": pass_at_1,
                "candidate_correct_count": candidate_correct_count,
                "candidate_accuracy": candidate_accuracy,
                "by_subject_pass_at_1": by_subject_pass_at_1,
                "by_subject_candidate_accuracy": by_subject_candidate_accuracy,
                "paper_score": pass_rate,
            }
        )
    else:
        success_count = sum(1 for row in judged_rows if row.get("is_success"))
        success_rate = success_count / len(judged_rows) if judged_rows else 0.0
        best_points = [float(row.get("points", 0.0)) for row in judged_rows]
        candidate_points = [
            float(row.get("points", 0.0)) for row in candidate_results
        ]
        summary.update(
            {
                "paper_metric": f"success_rate_at_{RESEARCH_SUCCESS_THRESHOLD}_pass@{group_size}",
                "success_threshold": RESEARCH_SUCCESS_THRESHOLD,
                "pass_at_k": success_rate,
                "success_count": success_count,
                "success_rate": success_rate,
                "mean_success_rate": success_rate,
                "mean_points": mean(best_points) if best_points else 0.0,
                "candidate_mean_points": mean(candidate_points) if candidate_points else 0.0,
                "paper_score": success_rate,
            }
        )

    return summary


def build_attempted_answer(track: str, response_text: str) -> str:
    if track == "olympiad":
        final_answer = extract_final_answer_block(response_text)
        return final_answer or response_text
    return response_text


def create_messages(track: str, item: Dict[str, Any]) -> List[Dict[str, str]]:
    attempted_answer = build_attempted_answer(track, item["response"])
    if track == "olympiad":
        prompt = OLYMPIAD_JUDGE_TEMPLATE.format(
            problem=item["problem"],
            reference_answer=item["answer"],
            attempted_answer=attempted_answer,
        )
    else:
        prompt = RESEARCH_JUDGE_TEMPLATE.format(
            problem=item["problem"],
            rubric=item["answer"],
            attempted_answer=attempted_answer,
        )
    return [{"role": "user", "content": prompt}]


def extract_message_content(message: Any) -> str:
    content = getattr(message, "content", "")
    return clean_response_text(stringify_content(content))


class EmptyJudgeResponseError(RuntimeError):
    def __init__(
        self,
        raw_response: Any,
        raw_response_attempts: Optional[List[Dict[str, Any]]] = None,
    ):
        super().__init__("judge response is empty")
        self.raw_response = raw_response
        self.raw_response_attempts = raw_response_attempts or []


def attach_raw_response_from_exception(row: Dict[str, Any], exc: Exception) -> None:
    raw_response = getattr(exc, "raw_response", None)
    if raw_response is not None:
        row["judge_raw_response"] = raw_response
    raw_response_attempts = getattr(exc, "raw_response_attempts", None)
    if raw_response_attempts:
        row["judge_raw_response_attempts"] = raw_response_attempts


def extract_stream_chunk_content(chunk: Any) -> str:
    choices = getattr(chunk, "choices", None)
    if choices is None and isinstance(chunk, dict):
        choices = chunk.get("choices")
    if not choices:
        return ""

    choice = choices[0]
    delta = getattr(choice, "delta", None)
    if delta is None and isinstance(choice, dict):
        delta = choice.get("delta")
    if delta is None:
        return ""

    content = getattr(delta, "content", None)
    if content is None and isinstance(delta, dict):
        content = delta.get("content")
    return stringify_content(content)


def collect_stream_response(stream: Any) -> Tuple[str, List[Any]]:
    pieces: List[str] = []
    chunks: List[Any] = []
    for chunk in stream:
        chunks.append(make_json_serializable(chunk))
        pieces.append(extract_stream_chunk_content(chunk))
    return clean_response_text("".join(piece for piece in pieces if piece)), chunks


def require_non_empty_judge_response(
    text: str,
    raw_response: Any,
    raw_response_attempts: Optional[List[Dict[str, Any]]] = None,
) -> str:
    cleaned_text = clean_response_text(text)
    if not cleaned_text:
        raise EmptyJudgeResponseError(raw_response, raw_response_attempts)
    return cleaned_text


def request_with_retry(
    client: OpenAI,
    model: str,
    messages: List[Dict[str, str]],
    max_tokens: int,
    temperature: float,
    reasoning_effort: Optional[str],
    stream: bool,
) -> Tuple[str, Any, List[Dict[str, Any]]]:
    raw_response_attempts: List[Dict[str, Any]] = []
    for attempt in range(RETRY_TIMES):
        try:
            request_kwargs = {
                "model": model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
            if stream:
                request_kwargs["stream"] = True
            if reasoning_effort:
                request_kwargs["reasoning_effort"] = reasoning_effort

            try:
                response = client.chat.completions.create(**request_kwargs)
            except Exception as exc:
                exc_text = str(exc).lower()
                unsupported_reasoning_effort = reasoning_effort and (
                    "reasoning_effort" in exc_text
                    or "unknown parameter" in exc_text
                    or "extra inputs are not permitted" in exc_text
                    or isinstance(exc, TypeError)
                )
                if not unsupported_reasoning_effort:
                    raise
                request_kwargs.pop("reasoning_effort", None)
                response = client.chat.completions.create(**request_kwargs)

            if stream:
                judge_response, raw_response = collect_stream_response(response)
                raw_response_attempts.append(
                    {
                        "attempt": attempt + 1,
                        "stream": True,
                        "extracted_response": judge_response,
                        "raw_response": raw_response,
                    }
                )
                return (
                    require_non_empty_judge_response(
                        judge_response,
                        raw_response,
                        raw_response_attempts,
                    ),
                    raw_response,
                    raw_response_attempts,
                )
            choice = response.choices[0]
            raw_response = make_json_serializable(response)
            judge_response = extract_message_content(choice.message)
            raw_response_attempts.append(
                {
                    "attempt": attempt + 1,
                    "stream": False,
                    "extracted_response": judge_response,
                    "raw_response": raw_response,
                }
            )
            return (
                require_non_empty_judge_response(
                    judge_response,
                    raw_response,
                    raw_response_attempts,
                ),
                raw_response,
                raw_response_attempts,
            )
        except Exception as exc:
            if getattr(exc, "status_code", None) in {401, 403}:
                raise
            if not isinstance(exc, EmptyJudgeResponseError):
                raw_response_attempts.append(
                    {
                        "attempt": attempt + 1,
                        "stream": stream,
                        "error": str(exc),
                    }
                )
            if attempt == RETRY_TIMES - 1:
                if isinstance(exc, EmptyJudgeResponseError):
                    exc.raw_response_attempts = raw_response_attempts
                raise
            print(
                f"[retry {attempt + 1}/{RETRY_TIMES}] judge request failed: {exc}")
            time.sleep(1)
    raise RuntimeError("Unreachable")


def parse_olympiad_verdict(text: str) -> Tuple[Optional[bool], Optional[str]]:
    match = re.search(r"VERDICT:\s*(CORRECT|INCORRECT)\b",
                      text, flags=re.IGNORECASE)
    if not match:
        return None, None
    verdict = match.group(1).upper()
    return verdict == "CORRECT", verdict


def parse_research_points(text: str) -> Tuple[float, Optional[float]]:
    match = re.search(
        r"VERDICT:\s*([-+]?\d+(?:\.\d+)?)\b", text, flags=re.IGNORECASE)
    if not match:
        return 0.0, None
    raw_points = float(match.group(1))
    points = max(0.0, min(10.0, raw_points))
    return points, raw_points


def score_judgement(track: str, judge_text: str) -> Dict[str, Any]:
    if track == "olympiad":
        is_correct, raw_verdict = parse_olympiad_verdict(judge_text)
        return {
            "judge_raw_verdict": raw_verdict,
            "judge_parse_error": is_correct is None,
            "is_correct": bool(is_correct),
            "score": 1.0 if is_correct else 0.0,
        }

    points, raw_points = parse_research_points(judge_text)
    success = points >= RESEARCH_SUCCESS_THRESHOLD
    return {
        "judge_raw_points": raw_points,
        "judge_parse_error": raw_points is None,
        "points": points,
        "is_success": success,
        "score": points,
    }


def is_resumable_judgement(track: str, row: Dict[str, Any]) -> bool:
    data_index = row.get("_data_index")
    if not isinstance(data_index, int):
        return False

    judge_response = stringify_content(row.get("judge_response")).strip()
    if not judge_response:
        return False

    parsed = score_judgement(track, judge_response)
    return not bool(parsed.get("judge_parse_error"))


def prepare_resume_results(
    track: str,
    rows: Sequence[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], set[int], int]:
    by_index: Dict[int, Dict[str, Any]] = {}
    unindexed: List[Dict[str, Any]] = []
    malformed_count = 0

    for row in rows:
        if not isinstance(row, dict):
            malformed_count += 1
            continue
        data_index = row.get("_data_index")
        if isinstance(data_index, int):
            by_index[data_index] = row
        else:
            unindexed.append(row)

    processed_indices = {
        idx for idx, row in by_index.items() if is_resumable_judgement(track, row)
    }
    ignored_count = malformed_count + len(unindexed) + (
        len(by_index) - len(processed_indices)
    )
    all_rows = unindexed + [by_index[idx] for idx in sorted(by_index)]
    return all_rows, processed_indices, ignored_count


def combine_results_by_data_index(
    existing_results: Sequence[Dict[str, Any]],
    new_results: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    by_index: Dict[int, Dict[str, Any]] = {}
    unindexed: List[Dict[str, Any]] = []

    for row in list(existing_results) + list(new_results):
        data_index = row.get("_data_index")
        if isinstance(data_index, int):
            by_index[data_index] = row
        else:
            unindexed.append(row)

    return unindexed + [by_index[idx] for idx in sorted(by_index)]


def make_client(base_url: Optional[str], api_key: Optional[str]) -> OpenAI:
    kwargs: Dict[str, Any] = {"timeout": REQUEST_TIMEOUT}
    if base_url:
        kwargs["base_url"] = base_url
        kwargs["api_key"] = api_key or os.environ.get("OPENAI_API_KEY") or "none"
    if api_key:
        kwargs["api_key"] = api_key
    return OpenAI(**kwargs)


def process_single_item(
    item: Dict[str, Any],
    args: argparse.Namespace,
    client: OpenAI,
) -> Dict[str, Any]:
    messages = create_messages(args.track, item)
    judge_response, raw_response, raw_response_attempts = request_with_retry(
        client=client,
        model=args.judge_model,
        messages=messages,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        reasoning_effort=args.reasoning_effort,
        stream=args.stream,
    )

    result = dict(item)
    result["attempted_answer"] = build_attempted_answer(
        args.track, item["response"])
    result["judge_response"] = judge_response
    result["judge_raw_response"] = raw_response
    if len(raw_response_attempts) > 1:
        result["judge_raw_response_attempts"] = raw_response_attempts
    result.update(score_judgement(args.track, judge_response))
    return result


def thread_process_single_item(
    item: Dict[str, Any],
    args: argparse.Namespace,
    data_index: int,
    delay: float,
) -> Dict[str, Any]:
    if delay > 0:
        time.sleep(delay)
    client = make_client(args.base_url, args.api_key)
    result = process_single_item(item, args, client)
    result["_data_index"] = data_index
    return result


def summarize_subject_metrics(
    track: str,
    rows: Sequence[Dict[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row.get("subject", "unknown")].append(row)

    summary: Dict[str, Dict[str, Any]] = {}
    for subject, items in grouped.items():
        if track == "olympiad":
            correct = sum(1 for item in items if item.get("is_correct"))
            summary[subject] = {
                "num_items": len(items),
                "correct_count": correct,
                "accuracy": correct / len(items) if items else 0.0,
            }
        else:
            success = sum(1 for item in items if item.get("is_success"))
            points = [float(item.get("points", 0.0)) for item in items]
            summary[subject] = {
                "num_items": len(items),
                "success_count": success,
                "success_rate": success / len(items) if items else 0.0,
                "mean_points": mean(points) if points else 0.0,
            }
    return summary


def compute_trial_summary(
    track: str,
    judged_rows: Sequence[Dict[str, Any]],
    prediction_file: Path,
    result_file: Path,
    matched_by: str,
    trial_name: str,
) -> Dict[str, Any]:
    invalid_count = sum(
        1 for row in judged_rows if row.get("judge_parse_error"))
    by_subject = summarize_subject_metrics(track, judged_rows)

    summary: Dict[str, Any] = {
        "track": track,
        "trial_name": trial_name,
        "prediction_file": str(prediction_file.resolve()),
        "result_file": str(result_file.resolve()),
        "matched_by": matched_by,
        "num_items": len(judged_rows),
        "invalid_judgement_count": invalid_count,
        "paper_recommended_trials": PAPER_RECOMMENDED_TRIALS[track],
        "by_subject": by_subject,
    }

    if track == "olympiad":
        correct_count = sum(1 for row in judged_rows if row.get("is_correct"))
        accuracy = correct_count / len(judged_rows) if judged_rows else 0.0
        summary.update(
            {
                "paper_metric": "accuracy",
                "correct_count": correct_count,
                "accuracy": accuracy,
                "paper_score": accuracy,
            }
        )
    else:
        success_count = sum(1 for row in judged_rows if row.get("is_success"))
        points = [float(row.get("points", 0.0)) for row in judged_rows]
        success_rate = success_count / len(judged_rows) if judged_rows else 0.0
        mean_points = mean(points) if points else 0.0
        summary.update(
            {
                "paper_metric": f"success_rate_at_{RESEARCH_SUCCESS_THRESHOLD}",
                "success_threshold": RESEARCH_SUCCESS_THRESHOLD,
                "success_count": success_count,
                "success_rate": success_rate,
                "mean_points": mean_points,
                "paper_score": success_rate,
            }
        )
    return summary


def aggregate_trial_summaries(track: str, summaries: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    aggregate: Dict[str, Any] = {
        "track": track,
        "num_trials": len(summaries),
        "paper_recommended_trials": PAPER_RECOMMENDED_TRIALS[track],
        "trial_summaries": list(summaries),
    }

    if track == "olympiad":
        accuracies = [float(summary["accuracy"]) for summary in summaries]
        aggregate.update(
            {
                "paper_metric": "mean_accuracy_over_trials",
                "mean_accuracy": mean(accuracies) if accuracies else 0.0,
                "paper_score": mean(accuracies) if accuracies else 0.0,
            }
        )
    else:
        success_rates = [float(summary["success_rate"])
                         for summary in summaries]
        mean_points = [float(summary["mean_points"]) for summary in summaries]
        aggregate.update(
            {
                "paper_metric": f"mean_success_rate_at_{RESEARCH_SUCCESS_THRESHOLD}_over_trials",
                "success_threshold": RESEARCH_SUCCESS_THRESHOLD,
                "mean_success_rate": mean(success_rates) if success_rates else 0.0,
                "mean_points": mean(mean_points) if mean_points else 0.0,
                "paper_score": mean(success_rates) if success_rates else 0.0,
            }
        )
    return aggregate


def evaluate_trial(
    trial_name: str,
    prediction_file: Path,
    prediction_rows: Sequence[Dict[str, Any]],
    official_rows: Sequence[Dict[str, Any]],
    args: argparse.Namespace,
) -> Dict[str, Any]:
    track_output_dir = Path(args.output_dir) / args.track / trial_name
    result_file = track_output_dir / f"{trial_name}.judged.json"
    summary_file = track_output_dir / f"{trial_name}.summary.json"

    merged_rows, matched_by = merge_predictions_with_official(
        track=args.track,
        official_rows=official_rows,
        prediction_rows=prediction_rows,
        response_key=args.response_key,
        match_by=args.match_by,
    )

    existing_results: List[Dict[str, Any]] = []
    processed_indices = set()
    if args.resume and result_file.exists():
        try:
            loaded = load_json_or_jsonl(result_file)
            if isinstance(loaded, list):
                existing_results, processed_indices, ignored_count = prepare_resume_results(
                    args.track,
                    loaded,
                )
                if processed_indices:
                    print(
                        f"[RESUME] Reusing {len(processed_indices)} complete judged rows from {result_file}.",
                        flush=True,
                    )
                if ignored_count:
                    print(
                        f"[RESUME] Rejudging {ignored_count} incomplete/invalid rows from {result_file}.",
                        flush=True,
                    )
        except Exception as exc:
            print(
                f"Warning: failed to load existing result file {result_file}: {exc}")

    pending_items: List[Tuple[int, Dict[str, Any]]] = []
    for idx, item in enumerate(merged_rows, start=1):
        if idx in processed_indices:
            continue
        pending_items.append((idx, item))

    results_list: List[Dict[str, Any]] = []
    save_lock = threading.Lock()

    if pending_items:
        if args.concurrent:
            max_workers = min(args.max_workers, len(pending_items))
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {
                    executor.submit(
                        thread_process_single_item,
                        item,
                        args,
                        idx,
                        args.request_interval * order,
                    ): idx
                    for order, (idx, item) in enumerate(pending_items)
                }
                with tqdm(total=len(futures), desc=f"Judging {trial_name}") as pbar:
                    for future in as_completed(futures):
                        idx = futures[future]
                        try:
                            result = future.result()
                            results_list.append(result)
                            with save_lock:
                                all_results_sorted = combine_results_by_data_index(
                                    existing_results,
                                    results_list,
                                )
                                save_json(result_file, all_results_sorted)
                        except Exception as exc:
                            print(f"[error] failed on item {idx}: {exc}")
                            error_result = dict(merged_rows[idx - 1])
                            error_result["_data_index"] = idx
                            error_result["judge_response"] = None
                            error_result["judge_parse_error"] = True
                            error_result["error"] = str(exc)
                            attach_raw_response_from_exception(error_result, exc)
                            if args.track == "olympiad":
                                error_result["is_correct"] = False
                                error_result["score"] = 0.0
                            else:
                                error_result["points"] = 0.0
                                error_result["is_success"] = False
                                error_result["score"] = 0.0
                            results_list.append(error_result)
                            with save_lock:
                                all_results_sorted = combine_results_by_data_index(
                                    existing_results,
                                    results_list,
                                )
                                save_json(result_file, all_results_sorted)
                        finally:
                            pbar.update(1)
        else:
            client = make_client(args.base_url, args.api_key)
            with tqdm(total=len(pending_items), desc=f"Judging {trial_name}") as pbar:
                for idx, item in pending_items:
                    try:
                        result = process_single_item(item, args, client)
                        result["_data_index"] = idx
                    except Exception as exc:
                        print(f"[error] failed on item {idx}: {exc}")
                        result = dict(item)
                        result["_data_index"] = idx
                        result["judge_response"] = None
                        result["judge_parse_error"] = True
                        result["error"] = str(exc)
                        attach_raw_response_from_exception(result, exc)
                        if args.track == "olympiad":
                            result["is_correct"] = False
                            result["score"] = 0.0
                        else:
                            result["points"] = 0.0
                            result["is_success"] = False
                            result["score"] = 0.0
                    results_list.append(result)
                    all_results_sorted = combine_results_by_data_index(
                        existing_results,
                        results_list,
                    )
                    save_json(result_file, all_results_sorted)
                    pbar.update(1)

    final_results = combine_results_by_data_index(
        existing_results,
        results_list,
    )
    trial_summary = compute_trial_summary(
        track=args.track,
        judged_rows=final_results,
        prediction_file=prediction_file,
        result_file=result_file,
        matched_by=matched_by,
        trial_name=trial_name,
    )
    save_json(summary_file, trial_summary)
    return trial_summary


def evaluate_prediction_file(
    prediction_file: Path,
    official_rows: Sequence[Dict[str, Any]],
    args: argparse.Namespace,
) -> Dict[str, Any]:
    prediction_rows = load_json_or_jsonl(prediction_file)
    grouped_rows, matched_by, group_size = merge_prediction_groups_with_official(
        track=args.track,
        official_rows=official_rows,
        prediction_rows=prediction_rows,
        response_key=args.response_key,
        match_by=args.match_by,
    )
    candidate_items = build_candidate_eval_items(grouped_rows)

    output_dir = Path(args.output_dir) / args.track / prediction_file.stem
    checkpoint_file = output_dir / f"{prediction_file.stem}.candidates.json"
    result_file = output_dir / f"{prediction_file.stem}.judged.json"
    summary_file = output_dir / f"{prediction_file.stem}.summary.json"

    existing_results: List[Dict[str, Any]] = []
    processed_indices = set()
    if args.resume and checkpoint_file.exists():
        try:
            loaded = load_json_or_jsonl(checkpoint_file)
            if isinstance(loaded, list):
                existing_results, processed_indices, ignored_count = prepare_resume_results(
                    args.track,
                    loaded,
                )
                if processed_indices:
                    print(
                        f"[RESUME] Reusing {len(processed_indices)} complete candidate judgements from {checkpoint_file}.",
                        flush=True,
                    )
                if ignored_count:
                    print(
                        f"[RESUME] Rejudging {ignored_count} incomplete/invalid candidate rows from {checkpoint_file}.",
                        flush=True,
                    )
        except Exception as exc:
            print(
                f"Warning: failed to load existing checkpoint file {checkpoint_file}: {exc}"
            )

    pending_items: List[Tuple[int, Dict[str, Any]]] = []
    for item in candidate_items:
        global_index = item["_candidate_global_index"]
        if global_index in processed_indices:
            continue
        pending_items.append((global_index, item))

    print(
        f"Matched by {matched_by}; {len(grouped_rows)} questions with {group_size} candidate(s) each."
    )

    results_list: List[Dict[str, Any]] = []
    save_lock = threading.Lock()

    if pending_items:
        if args.concurrent:
            max_workers = min(args.max_workers, len(pending_items))
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {
                    executor.submit(
                        thread_process_single_item,
                        item,
                        args,
                        idx,
                        args.request_interval * order,
                    ): idx
                    for order, (idx, item) in enumerate(pending_items)
                }
                with tqdm(total=len(futures), desc=f"Judging {prediction_file.stem}") as pbar:
                    for future in as_completed(futures):
                        idx = futures[future]
                        try:
                            result = future.result()
                        except Exception as exc:
                            print(f"[error] failed on candidate {idx}: {exc}")
                            source_item = next(
                                item for item_idx, item in pending_items if item_idx == idx
                            )
                            result = dict(source_item)
                            result["_data_index"] = idx
                            result["judge_response"] = None
                            result["judge_parse_error"] = True
                            result["error"] = str(exc)
                            attach_raw_response_from_exception(result, exc)
                            if args.track == "olympiad":
                                result["is_correct"] = False
                                result["score"] = 0.0
                            else:
                                result["points"] = 0.0
                                result["is_success"] = False
                                result["score"] = 0.0

                        results_list.append(result)
                        with save_lock:
                            all_results_sorted = combine_results_by_data_index(
                                existing_results,
                                results_list,
                            )
                            save_json(checkpoint_file, all_results_sorted)
                        pbar.update(1)
        else:
            client = make_client(args.base_url, args.api_key)
            with tqdm(total=len(pending_items), desc=f"Judging {prediction_file.stem}") as pbar:
                for idx, item in pending_items:
                    try:
                        result = process_single_item(item, args, client)
                        result["_data_index"] = idx
                    except Exception as exc:
                        print(f"[error] failed on candidate {idx}: {exc}")
                        result = dict(item)
                        result["_data_index"] = idx
                        result["judge_response"] = None
                        result["judge_parse_error"] = True
                        result["error"] = str(exc)
                        attach_raw_response_from_exception(result, exc)
                        if args.track == "olympiad":
                            result["is_correct"] = False
                            result["score"] = 0.0
                        else:
                            result["points"] = 0.0
                            result["is_success"] = False
                            result["score"] = 0.0
                    results_list.append(result)
                    all_results_sorted = combine_results_by_data_index(
                        existing_results,
                        results_list,
                    )
                    save_json(checkpoint_file, all_results_sorted)
                    pbar.update(1)

    final_candidate_results = combine_results_by_data_index(
        existing_results,
        results_list,
    )
    judged_rows = fold_candidate_results_into_groups(
        track=args.track,
        grouped_rows=grouped_rows,
        candidate_results=final_candidate_results,
    )
    save_json(result_file, judged_rows)

    summary = compute_grouped_summary(
        track=args.track,
        judged_rows=judged_rows,
        candidate_results=final_candidate_results,
        prediction_file=prediction_file,
        result_file=result_file,
        checkpoint_file=checkpoint_file,
        matched_by=matched_by,
        group_size=group_size,
    )
    save_json(summary_file, summary)

    aggregate_summary = dict(summary)
    aggregate_summary["num_trials"] = 1
    aggregate_summary["trial_summaries"] = [summary]
    return aggregate_summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Judge-only FrontierScience evaluator. "
            "It scores pre-generated Olympiad or Research predictions."
        )
    )
    parser.add_argument(
        "--track",
        choices=["olympiad", "research"],
        required=True,
        help="Which FrontierScience track to evaluate.",
    )
    parser.add_argument(
        "--prediction-path",
        required=True,
        help="已生成好的单个预测文件（.json/.jsonl）；文件内部可以包含多轮 rollout。",
    )
    parser.add_argument(
        "--official-data-path",
        default=None,
        help="Optional local FrontierScience data file or directory. If omitted, download from Hugging Face.",
    )
    parser.add_argument(
        "--output-dir",
        default="results/frontierscience_eval",
        help="Directory used to save judged item outputs and summaries.",
    )
    parser.add_argument(
        "--response-key",
        default=None,
        help="Field name containing the model response in the prediction rows.",
    )
    parser.add_argument(
        "--match-by",
        default="auto",
        help="How to align predictions with official rows. Use auto, order, task_group_id, problem, etc.",
    )
    parser.add_argument(
        "--judge-model",
        default="gpt-5",
        help="Judge model name. The paper uses GPT-5 with high reasoning effort.",
    )
    parser.add_argument(
        "--base-url",
        default=None,
        help="Optional OpenAI-compatible base URL for the judge model.",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="API key for the judge model. If omitted, the OpenAI client falls back to env vars.",
    )
    parser.add_argument(
        "--reasoning-effort",
        default="high",
        help="Optional reasoning effort passed to compatible judge endpoints.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=DEFAULT_TEMPERATURE,
        help="Judge sampling temperature.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=DEFAULT_MAX_TOKENS,
        help="Max output tokens for the judge response.",
    )
    parser.add_argument(
        "--stream",
        action="store_true",
        help="Use streaming chat completions and reconstruct the judge response from chunks.",
    )
    parser.add_argument(
        "--concurrent",
        action="store_true",
        help="Use concurrent judge requests.",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=32,
        help="Maximum worker count when --concurrent is enabled.",
    )
    parser.add_argument(
        "--request-interval",
        type=float,
        default=0.0,
        help="Delay in seconds between launching concurrent requests.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from existing judged outputs if present.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    official_rows = load_official_dataset(args.track, args.official_data_path)
    prediction_file = resolve_prediction_file(args.prediction_path)

    print(f"Track: {args.track}")
    print(f"Official items: {len(official_rows)}")
    print(f"Prediction file: {prediction_file}")
    print(f"Judge model: {args.judge_model}")
    print(f"Output dir: {Path(args.output_dir).resolve()}")

    print(f"Evaluating: {prediction_file}")
    aggregate_summary = evaluate_prediction_file(
        prediction_file, official_rows, args)
    aggregate_path = Path(args.output_dir) / \
        args.track / "aggregate_summary.json"
    save_json(aggregate_path, aggregate_summary)

    print("\n===== Aggregate Summary =====")
    if args.track == "olympiad":
        print(
            f"Pass@{aggregate_summary['group_size']}: {aggregate_summary['pass_at_k']:.4f}"
        )
        print(f"Pass@1: {aggregate_summary['pass_at_1']:.4f}")
        print(
            f"Candidate accuracy: {aggregate_summary['candidate_accuracy']:.4f}"
        )
    else:
        print(
            f"Success@{aggregate_summary['group_size']}: "
            f"{aggregate_summary['mean_success_rate']:.4f} "
            f"(threshold >= {RESEARCH_SUCCESS_THRESHOLD})"
        )
        print(
            f"Best-of-k mean rubric points: {aggregate_summary['mean_points']:.4f}")
    print(f"Aggregate summary saved to: {aggregate_path.resolve()}")


if __name__ == "__main__":
    main()
