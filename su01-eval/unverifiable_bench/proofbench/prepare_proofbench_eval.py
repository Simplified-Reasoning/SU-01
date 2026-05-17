#!/usr/bin/env python3
import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_META_PATH = SCRIPT_DIR / "proofbench.json"
DEFAULT_OUTPUT = SCRIPT_DIR / "eval_input" / "proofbench_eval_input.json"


def read_csv_rows(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return [dict(row) for row in csv.DictReader(f)]


def read_json_rows(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        if path.suffix.lower() == ".jsonl":
            return [json.loads(line) for line in f if line.strip()]
        data = json.load(f)
    if isinstance(data, list):
        return data
    raise ValueError(f"Expected a list in {path}")


def read_rows(path: Path) -> List[Dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return read_csv_rows(path)
    if suffix in {".json", ".jsonl"}:
        return read_json_rows(path)
    raise ValueError(f"Unsupported metadata/input format: {path}")


def write_output(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".json":
        with path.open("w", encoding="utf-8") as f:
            json.dump(rows, f, ensure_ascii=False, indent=2)
        return

    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, list):
        return "\n\n".join(clean_text(v) for v in value if clean_text(v))
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False, indent=2)
    return str(value).strip()


def build_response_name(pattern: str, idx: int, row: Dict[str, Any]) -> str:
    problem_id = clean_text(row.get("Problem ID"))
    try:
        return pattern.format(idx=idx, problem_id=problem_id)
    except KeyError as exc:
        raise ValueError(
            f"--response_pattern 包含不支持的占位符: {exc}. "
            "仅支持 {idx} 和 {problem_id}。"
        ) from exc


def build_eval_item(
    idx: int,
    row: Dict[str, Any],
    response_text: str,
    name: str,
    original: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    problem = clean_text(row.get("Problem"))
    solution = clean_text(row.get("Solution"))
    rubrics = clean_text(row.get("Grading guidelines"))

    if not problem:
        raise ValueError(f"第 {idx} 题缺少 Problem")
    if not solution:
        raise ValueError(f"第 {idx} 题缺少 Solution")

    item: Dict[str, Any] = dict(original or {})
    item.update({
        "group_index": None,
        "index": idx,
        "name": name,
        "prompt": item.get("prompt", problem),
        "question": problem,
        "response": response_text,
        "response_length": len(response_text),
        # Keep the legacy ProofBench prompt shape: old eval inputs stored the
        # reference solution as a one-element list, which is rendered as
        # "['...']" by eval_mo.py.
        "label": [solution],
        "reward": {},
        "status": "",
        "is_proof": "yes",
        "question_number": idx,
        "problem_id": clean_text(row.get("Problem ID")),
        "rubrics": rubrics,
        "category": clean_text(row.get("Category")),
        "level": clean_text(row.get("Level")),
        "short_answer": clean_text(row.get("Short Answer")),
        "source": clean_text(row.get("Source")),
    })
    return item


def get_problem_id(item: Dict[str, Any]) -> str:
    candidates = [
        item.get("problem_id"),
        item.get("Problem ID"),
        item.get("metadata", {}).get("problem_idx")
        if isinstance(item.get("metadata"), dict)
        else None,
        item.get("metadata", {}).get("problem_id")
        if isinstance(item.get("metadata"), dict)
        else None,
    ]
    for value in candidates:
        text = clean_text(value)
        if text:
            return text
    return ""


def iter_selected_rows(
    rows: List[Dict[str, Any]],
    start_index: int,
    end_index: Optional[int],
) -> Iterable[tuple[int, Dict[str, Any]]]:
    end = end_index if end_index is not None else len(rows)
    if start_index < 1:
        raise ValueError("start_index must be at least 1")
    if start_index > end:
        raise ValueError("start_index cannot be greater than end_index")
    if end > len(rows):
        raise ValueError(f"end_index={end} exceeds metadata size {len(rows)}")
    for idx in range(start_index, end + 1):
        yield idx, rows[idx - 1]


def prepare_from_prediction_file(
    prediction_rows: List[Dict[str, Any]],
    meta_rows: List[Dict[str, Any]],
    start_index: int,
    end_index: Optional[int],
    name: str,
) -> List[Dict[str, Any]]:
    meta_by_id = {clean_text(row.get("Problem ID")): row for row in meta_rows}
    selected_meta = dict(iter_selected_rows(meta_rows, start_index, end_index))
    selected_ids = {clean_text(row.get("Problem ID")) for row in selected_meta.values()}

    output_rows: List[Dict[str, Any]] = []
    for order_idx, pred in enumerate(prediction_rows, start=1):
        problem_id = get_problem_id(pred)
        meta_row: Optional[Dict[str, Any]] = meta_by_id.get(problem_id) if problem_id else None

        if meta_row is None:
            if order_idx > len(meta_rows):
                raise ValueError(
                    f"Cannot align prediction row {order_idx}; no problem_id and no metadata row."
                )
            meta_row = meta_rows[order_idx - 1]
            problem_id = clean_text(meta_row.get("Problem ID"))

        if selected_ids and problem_id not in selected_ids:
            continue

        meta_idx = next(
            (idx for idx, row in selected_meta.items() if clean_text(row.get("Problem ID")) == problem_id),
            order_idx,
        )
        response_text = clean_text(pred.get("response"))
        output_rows.append(
            build_eval_item(
                idx=meta_idx,
                row=meta_row,
                response_text=response_text,
                name=name,
                original=pred,
            )
        )

    return output_rows


def prepare_from_response_dir(
    response_dir: Path,
    meta_rows: List[Dict[str, Any]],
    response_pattern: str,
    start_index: int,
    end_index: Optional[int],
    name: str,
) -> List[Dict[str, Any]]:
    output_rows: List[Dict[str, Any]] = []
    missing_txt: List[Path] = []

    for idx, row in iter_selected_rows(meta_rows, start_index, end_index):
        response_name = build_response_name(response_pattern, idx, row)
        response_path = response_dir / response_name
        if not response_path.exists():
            missing_txt.append(response_path)
            continue

        response_text = response_path.read_text(encoding="utf-8").strip()
        output_rows.append(
            build_eval_item(
                idx=idx,
                row=row,
                response_text=response_text,
                name=name,
            )
        )

    if missing_txt:
        details = "\n".join(str(path) for path in missing_txt)
        raise FileNotFoundError(f"Missing response files:\n{details}")

    return output_rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare ProofBench prediction files for eval_mo.py."
    )
    parser.add_argument(
        "--meta-path",
        "--meta_json",
        "--meta_csv",
        type=Path,
        default=DEFAULT_META_PATH,
        help="ProofBench metadata path (.json, .jsonl, or .csv).",
    )
    parser.add_argument(
        "--input-file",
        type=Path,
        default=None,
        help="Prediction JSON/JSONL file containing a response field.",
    )
    parser.add_argument(
        "--response-dir",
        "--response_dir",
        type=Path,
        default=None,
        help="Directory containing one response txt file per problem.",
    )
    parser.add_argument(
        "--output-path",
        "--output_path",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Output file (.jsonl or .json).",
    )
    parser.add_argument(
        "--response-pattern",
        "--response_pattern",
        type=str,
        default="{problem_id}_out.txt",
        help="Response filename template for --response-dir. Supports {idx} and {problem_id}.",
    )
    parser.add_argument(
        "--start-index",
        "--start_index",
        type=int,
        default=1,
        help="Start problem index, 1-based and inclusive.",
    )
    parser.add_argument(
        "--end-index",
        "--end_index",
        type=int,
        default=None,
        help="End problem index, 1-based and inclusive.",
    )
    parser.add_argument(
        "--name",
        type=str,
        default="proofbench",
        help="Value for the output name field.",
    )
    args = parser.parse_args()

    if not args.meta_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {args.meta_path}")
    if args.input_file is None and args.response_dir is None:
        raise ValueError("Set either --input-file or --response-dir.")
    if args.input_file is not None and args.response_dir is not None:
        raise ValueError("Set only one of --input-file or --response-dir.")
    if args.response_dir is not None:
        if "{idx" not in args.response_pattern and "{problem_id" not in args.response_pattern:
            raise ValueError("--response-pattern must contain {idx} or {problem_id}")
        if not args.response_dir.exists():
            raise FileNotFoundError(f"Response directory not found: {args.response_dir}")

    meta_rows = read_rows(args.meta_path)
    if not meta_rows:
        raise ValueError(f"Metadata file is empty: {args.meta_path}")

    if args.input_file is not None:
        if not args.input_file.exists():
            raise FileNotFoundError(f"Input file not found: {args.input_file}")
        output_rows = prepare_from_prediction_file(
            prediction_rows=read_rows(args.input_file),
            meta_rows=meta_rows,
            start_index=args.start_index,
            end_index=args.end_index,
            name=args.name,
        )
    else:
        output_rows = prepare_from_response_dir(
            response_dir=args.response_dir,
            meta_rows=meta_rows,
            response_pattern=args.response_pattern,
            start_index=args.start_index,
            end_index=args.end_index,
            name=args.name,
        )

    if not output_rows:
        raise ValueError("No rows were prepared.")

    write_output(args.output_path, output_rows)
    print(f"Prepared {len(output_rows)} rows: {args.output_path}")


if __name__ == "__main__":
    main()
