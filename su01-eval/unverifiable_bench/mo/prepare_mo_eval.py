#!/usr/bin/env python3
import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT = SCRIPT_DIR / "eval_input" / "mo_eval_input.jsonl"


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"JSONL 解析失败: {path}:{line_no}: {e}") from e
    return rows


def as_int(value: Any) -> Optional[int]:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.strip():
        try:
            return int(value.strip())
        except ValueError:
            return None
    return None


def extract_question_number(item: Dict[str, Any]) -> Optional[int]:
    # 先用“题号”语义明确的字段，避免误用全局索引字段。
    direct_keys = ["question_number", "question_num", "problem_number"]
    for key in direct_keys:
        val = as_int(item.get(key))
        if val is not None:
            return val

    metadata = item.get("metadata")
    if isinstance(metadata, dict):
        for key in direct_keys:
            val = as_int(metadata.get(key))
            if val is not None:
                return val

    problem_id = item.get("problem_id")
    if isinstance(problem_id, str):
        match = re.search(r"P(\d+)$", problem_id)
        if match:
            return int(match.group(1))

    # 最后兜底再尝试 problem_idx。
    val = as_int(item.get("problem_idx"))
    if val is not None:
        return val
    if isinstance(metadata, dict):
        val = as_int(metadata.get("problem_idx"))
        if val is not None:
            return val

    return None


def choose_item_for_question(candidates: List[Dict[str, Any]]) -> Dict[str, Any]:
    if len(candidates) == 1:
        return candidates[0]

    test_split = [x for x in candidates if x.get("split") == "test"]
    if len(test_split) == 1:
        return test_split[0]
    if len(test_split) > 1:
        return test_split[0]

    return candidates[0]


def extract_prompt_text(item: Dict[str, Any]) -> str:
    prompt = item.get("prompt")
    if isinstance(prompt, str):
        return prompt
    if isinstance(prompt, list):
        parts: List[str] = []
        for p in prompt:
            if isinstance(p, dict):
                content = p.get("content")
                if isinstance(content, str):
                    parts.append(content)
            elif isinstance(p, str):
                parts.append(p)
        if parts:
            return "\n".join(parts)
    question = item.get("question")
    if isinstance(question, str):
        return question
    return ""


def extract_label(item: Dict[str, Any]) -> Any:
    # 保持和源数据一致（通常是 list[str]），避免丢信息。
    return item.get("label", [])


def extract_rubrics(item: Dict[str, Any]) -> str:
    candidates = [
        item.get("rubrics"),
        item.get("grading_guidelines"),
        item.get("Grading guidelines"),
    ]
    metadata = item.get("metadata")
    if isinstance(metadata, dict):
        candidates.extend(
            [
                metadata.get("rubrics"),
                metadata.get("grading_guidelines"),
                metadata.get("Grading guidelines"),
            ]
        )
    for c in candidates:
        if isinstance(c, str) and c.strip():
            return c
    return ""


def parse_guideline_md(path: Path) -> Dict[int, str]:
    """
    解析类似如下格式的 markdown:
      1. "Guidelines": ...
      2. "Guidelines": ...
    返回 {题号: guideline 文本}。
    """
    text = path.read_text(encoding="utf-8")
    pattern = re.compile(r"^\s*(\d+)\.\s*\"Guidelines\":\s*(.+?)(?=^\s*\d+\.\s*\"Guidelines\":|\Z)", re.M | re.S)
    mapping: Dict[int, str] = {}
    for m in pattern.finditer(text):
        qn = int(m.group(1))
        guideline = m.group(2).strip()
        if guideline:
            mapping[qn] = guideline
    return mapping


def build_question_map(meta_rows: List[Dict[str, Any]]) -> Dict[int, Dict[str, Any]]:
    tmp: Dict[int, List[Dict[str, Any]]] = {}
    for row in meta_rows:
        qn = extract_question_number(row)
        if qn is None:
            continue
        tmp.setdefault(qn, []).append(row)

    result: Dict[int, Dict[str, Any]] = {}
    for qn, candidates in tmp.items():
        result[qn] = choose_item_for_question(candidates)
    return result


def build_eval_item(
    idx: int,
    question_number: int,
    response_text: str,
    source_item: Dict[str, Any],
    guideline_override: Optional[str],
    name: str,
) -> Dict[str, Any]:
    question = source_item.get("question")
    if not isinstance(question, str) or not question.strip():
        question = extract_prompt_text(source_item)

    out: Dict[str, Any] = {
        "group_index": None,
        "index": idx,
        "name": name,
        "prompt": extract_prompt_text(source_item),
        "question": question,
        "response": response_text,
        "response_length": len(response_text),
        "label": extract_label(source_item),
        "reward": {},
        "status": "",
        "is_proof": source_item.get("is_proof", "yes"),
        "question_number": question_number,
        "source_index": idx,
    }

    rubrics = guideline_override.strip() if isinstance(guideline_override, str) and guideline_override.strip() else extract_rubrics(source_item)
    if rubrics:
        out["rubrics"] = rubrics

    if "problem_id" in source_item:
        out["problem_id"] = source_item["problem_id"]

    if "year" in source_item:
        out["year"] = source_item["year"]

    return out


def normalize_question_number(idx: int, cycle_length: Optional[int]) -> int:
    if cycle_length is None or cycle_length <= 0:
        return idx
    return (idx - 1) % cycle_length + 1


def build_response_filename_regex(response_pattern: str) -> re.Pattern[str]:
    match = re.search(r"\{idx[^}]*\}", response_pattern)
    if not match:
        raise ValueError("--response_pattern 必须包含 {idx} 占位符")

    prefix = re.escape(response_pattern[:match.start()])
    suffix = re.escape(response_pattern[match.end():])
    return re.compile(rf"^{prefix}(?P<idx>\d+){suffix}$")


def collect_response_files(
    response_dir: Path,
    response_pattern: str,
    start_index: Optional[int],
    end_index: Optional[int],
) -> List[Tuple[int, Path]]:
    regex = build_response_filename_regex(response_pattern)

    matched: List[Tuple[int, Path]] = []
    for path in response_dir.iterdir():
        if not path.is_file():
            continue
        match = regex.match(path.name)
        if not match:
            continue
        matched.append((int(match.group("idx")), path))

    matched.sort(key=lambda x: x[0])
    if not matched:
        raise FileNotFoundError(
            f"在 {response_dir} 下没有找到匹配 {response_pattern} 的响应文件"
        )

    matched_map = {idx: path for idx, path in matched}

    if start_index is not None and end_index is not None:
        if start_index > end_index:
            raise ValueError("start_index 不能大于 end_index")

        missing = [
            response_dir / response_pattern.format(idx=idx)
            for idx in range(start_index, end_index + 1)
            if idx not in matched_map
        ]
        if missing:
            details = "\n".join(str(p) for p in missing)
            raise FileNotFoundError(f"以下 response 文件不存在:\n{details}")

        return [(idx, matched_map[idx]) for idx in range(start_index, end_index + 1)]

    filtered = matched
    if start_index is not None:
        filtered = [(idx, path)
                    for idx, path in filtered if idx >= start_index]
    if end_index is not None:
        filtered = [(idx, path) for idx, path in filtered if idx <= end_index]

    if not filtered:
        raise FileNotFoundError(
            "没有找到落在指定范围内的响应文件。"
        )

    return filtered


def write_output(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".json":
        with path.open("w", encoding="utf-8") as f:
            json.dump(rows, f, ensure_ascii=False, indent=2)
        return

    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert per-problem MO response txt files into eval_mo.py input format."
    )
    parser.add_argument(
        "--response_dir",
        "--response-dir",
        type=Path,
        required=True,
        help="Directory containing response txt files, such as 1_out.txt or imo01_out.txt.",
    )
    parser.add_argument(
        "--meta_jsonl",
        "--meta-jsonl",
        type=Path,
        required=True,
        help="JSONL file containing problem statements, reference solutions, and optional rubrics.",
    )
    parser.add_argument(
        "--output_path",
        "--output-path",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Output file (.jsonl or .json).",
    )
    parser.add_argument(
        "--start_index",
        "--start-index",
        type=int,
        default=None,
        help="Optional inclusive start response index.",
    )
    parser.add_argument(
        "--end_index",
        "--end-index",
        type=int,
        default=None,
        help="Optional inclusive end response index.",
    )
    parser.add_argument(
        "--response_pattern",
        "--response-pattern",
        type=str,
        default="{idx}_out.txt",
        help="Response filename pattern. Must contain {idx}, for example '{idx}_out.txt' or 'imo{idx:02d}_out.txt'.",
    )
    parser.add_argument(
        "--question_cycle",
        "--question-cycle",
        type=int,
        default=6,
        help="Map response index to problem number with a 1-based cycle. Use 0 to disable.",
    )
    parser.add_argument(
        "--name",
        type=str,
        default="mo",
        help="The name field written into each output sample.",
    )
    parser.add_argument(
        "--guideline_md",
        "--guideline-md",
        type=Path,
        default=None,
        help="Optional markdown file with per-problem guidelines to override rubrics.",
    )
    args = parser.parse_args()

    if not args.meta_jsonl.exists():
        raise FileNotFoundError(f"找不到 meta_jsonl: {args.meta_jsonl}")
    if not args.response_dir.exists():
        raise FileNotFoundError(f"找不到 response_dir: {args.response_dir}")

    meta_rows = read_jsonl(args.meta_jsonl)
    question_map = build_question_map(meta_rows)
    guideline_map: Dict[int, str] = {}
    if args.guideline_md is not None:
        if not args.guideline_md.exists():
            raise FileNotFoundError(f"找不到 guideline_md: {args.guideline_md}")
        guideline_map = parse_guideline_md(args.guideline_md)
        print(f"从 guideline md 解析到 {len(guideline_map)} 条题目 guideline")
    response_files = collect_response_files(
        response_dir=args.response_dir,
        response_pattern=args.response_pattern,
        start_index=args.start_index,
        end_index=args.end_index,
    )
    print(f"自动扫描到 {len(response_files)} 个响应文件")

    output_rows: List[Dict[str, Any]] = []
    missing_questions: List[int] = []

    for idx, response_path in response_files:
        question_number = normalize_question_number(idx, args.question_cycle)

        if question_number not in question_map:
            missing_questions.append(question_number)
            continue

        response_text = response_path.read_text(encoding="utf-8")
        response_text = response_text.strip()
        source_item = question_map[question_number]
        output_rows.append(
            build_eval_item(
                idx=idx,
                question_number=question_number,
                response_text=response_text,
                source_item=source_item,
                guideline_override=guideline_map.get(question_number),
                name=args.name,
            )
        )

    if missing_questions:
        raise KeyError(
            f"meta_jsonl 里找不到这些题号: {missing_questions}。"
            "请检查 question_number / metadata.question_number / problem_id。"
        )

    write_output(args.output_path, output_rows)
    print(f"已生成 {len(output_rows)} 条数据: {args.output_path}")


if __name__ == "__main__":
    main()
