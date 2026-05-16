#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from openai import OpenAI
from tqdm import tqdm

POINTS_PATTERN = re.compile(
    r"<points>\s*(\d+(?:\.\d+)?)\s+out\s+of\s+(\d+(?:\.\d+)?)\s*</points>",
    re.IGNORECASE,
)
OUT_OF_PATTERN = re.compile(
    r"(\d+(?:\.\d+)?)\s*(?:\\text\{\s*out\s+of\s*\}|out\s+of)\s*(\d+(?:\.\d+)?)",
    re.IGNORECASE,
)
BOXED_TEXT_OUT_PATTERN = re.compile(
    r"(\d+(?:\.\d+)?)\s*\\text\{\s*out\s+of\s+(\d+(?:\.\d+)?)\s*\}",
    re.IGNORECASE,
)
FINAL_SCORE_PATTERN = re.compile(
    r"(?:final\s+score\s+is|score\s*:?)\s*\$?\s*(?:\\?boxed\{\s*)?(\d+(?:\.\d+)?)",
    re.IGNORECASE,
)
CORRECT_PATTERN = re.compile(r"\[(Correct|Incorrect)\]", re.IGNORECASE)


def parse_points_prediction(value: Any) -> Tuple[Optional[float], Optional[float]]:
    if not isinstance(value, str):
        return None, None
    text = value.strip()
    if not text:
        return None, None

    matches = list(POINTS_PATTERN.finditer(text))
    if matches:
        last = matches[-1]
        return float(last.group(1)), float(last.group(2))

    matches = list(OUT_OF_PATTERN.finditer(text))
    if matches:
        last = matches[-1]
        return float(last.group(1)), float(last.group(2))

    matches = list(BOXED_TEXT_OUT_PATTERN.finditer(text))
    if matches:
        last = matches[-1]
        return float(last.group(1)), float(last.group(2))

    matches = list(FINAL_SCORE_PATTERN.finditer(text))
    if matches:
        score = float(matches[-1].group(1))
        return (0.0, 1.0) if score == 0 else (1.0, 1.0)

    match = CORRECT_PATTERN.search(text)
    if match:
        label = match.group(1).lower()
        return (1.0 if label == "correct" else 0.0), 1.0

    return None, None


def load_list_json(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"期望输入是 list JSON: {path}")
    return [x for x in data if isinstance(x, dict)]


def save_list_json(path: Path, data: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def _format_points(score: float, max_score: float) -> str:
    score_str = str(int(score)) if float(score).is_integer() else str(score)
    max_str = str(int(max_score)) if float(max_score).is_integer() else str(max_score)
    return f"<points>{score_str} out of {max_str}</points>"


def build_messages(prediction_text: str) -> List[Dict[str, Any]]:
    """
    二次 parser 只做“格式归一化”，不重新打分：
    - 输入：judge 的原始 prediction 文本（可能不严格遵守 <points>）
    - 输出：严格一行 <points>X out of Y</points>
    """
    prompt = (
        "你是一个严格的文本解析器。下面给你一段 judge 的 prediction 文本。\n"
        "你的任务：从中抽取最终得分 X 和满分 Y，并且只输出一行、严格使用下面格式：\n"
        "<points>X out of Y</points>\n"
        "\n"
        "约束：\n"
        "- 只输出这一行，不要任何解释、不要换行、不要代码块、不要 LaTeX。\n"
        "- X、Y 必须是数字（整数或小数）。\n"
        "- 文本里可能出现各种写法，你需要鲁棒处理，例如：\n"
        "  - $\\\\boxed{1 \\\\text{ out of } 7}$\n"
        "  - The final answer is $\\\\boxed{\\\\text{6 out of 7}}$\n"
        "  - score: 6 out of 7\n"
        "  - <points>6 out of 7</points>\n"
        "\n"
        "prediction 原文如下（请只基于它抽取分数）：\n"
        f"{prediction_text}"
    )
    return [{"role": "user", "content": prompt}]


def build_messages_infer(prediction_text: str) -> List[Dict[str, Any]]:
    """
    抽取式失败时使用：部分 judge 把几何答案写进 \\boxed{}，全文没有 “X out of 7”，
    但评语里仍有「完整正确 / complete」等。要求模型在 0–7 制下给出唯一一行 <points>。
    """
    prompt = (
        "下面是一段 ProofBench / IMO 风格的阅卷长文，满分一般为 7。\n"
        "任务：给出该生在本题应得的分数 X 与满分 Y（通常为 7），只输出一行：\n"
        "<points>X out of Y</points>\n"
        "\n"
        "规则：\n"
        "- 若文中已有明确分数（如 “6 out of 7”、<points>、\\\\boxed{\\\\text{n out of 7}} 等），必须按该分数输出。\n"
        "- 若全文没有任何数值分数，但明确认定解答完整、正确、严谨（如 complete / correct / full rigor），输出 <points>7 out of 7</points>。\n"
        "- 若明确认定基本错误、无实质进展，输出 <points>0 out of 7</points>。\n"
        "- 若仅有部分进展、无法确定具体分值，输出 <points>1 out of 7</points>（保守部分分）。\n"
        "- 只输出这一行，不要解释、不要 LaTeX、不要代码块。\n"
        "\n"
        "阅卷原文：\n"
        f"{prediction_text}"
    )
    return [{"role": "user", "content": prompt}]


def tail_text(text: str, max_chars: int) -> str:
    """分数通常在末尾；截断可降低超长文本导致的不稳定。"""
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    return text[-max_chars:]


def request_with_retry(
    client: OpenAI,
    *,
    model: str,
    messages: List[Dict[str, Any]],
    retry_times: int,
    timeout_s: int,
    max_tokens: int = 256,
) -> tuple[Optional[str], Optional[str]]:
    for attempt in range(retry_times):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.0,
                max_tokens=max_tokens,
                timeout=timeout_s,
            )
            return resp.choices[0].message.content, None
        except Exception as e:
            print(f"[重试 {attempt+1}/{retry_times}] 解析请求失败: {e}", flush=True)
            if attempt < retry_times - 1:
                time.sleep(1)
    return None, "request_failed_or_timed_out"


def _parse_llm_points_line(raw: Optional[str]) -> Tuple[Optional[float], Optional[float]]:
    if not raw or not isinstance(raw, str):
        return None, None
    text = raw.strip()
    s, m = parse_points_prediction(text)
    if s is not None and m is not None:
        return s, m
    match = list(OUT_OF_PATTERN.finditer(text))
    if match:
        last = match[-1]
        return float(last.group(1)), float(last.group(2))
    return None, None


def _llm_line_trivially_bad(raw: Optional[str]) -> bool:
    if not raw or not isinstance(raw, str):
        return True
    t = raw.strip()
    if len(t) < 12:
        return True
    low = t.lower()
    if "out of" not in low and "<points" not in low:
        return True
    return False


def normalize_one(
    item: Dict[str, Any],
    *,
    client: OpenAI,
    model: str,
    retry_times: int,
    timeout_s: int,
    tail_chars: int,
) -> Dict[str, Any]:
    # 只处理 prediction 为字符串且规则解析失败的条目
    pred = item.get("prediction")
    score, max_score = parse_points_prediction(pred)
    if score is not None and max_score is not None:
        return item

    if not isinstance(pred, str) or not pred.strip():
        return item

    text = tail_text(pred.strip(), max_chars=tail_chars)
    used_infer = False
    parsed_text, req_err = request_with_retry(
        client,
        model=model,
        messages=build_messages(text),
        retry_times=retry_times,
        timeout_s=timeout_s,
        max_tokens=256,
    )
    if not parsed_text:
        item = dict(item)
        # 进入过 parser 流程但请求失败：显式落盘，避免“无痕失败”
        item["parser_pred"] = None
        item["points_parse_error"] = {
            "status": "request_failed",
            "parser_model": model,
            "error": req_err or "unknown",
        }
        return item

    # 过短/乱码（如单个 “<”）再要一次，避免直接判死
    if _llm_line_trivially_bad(parsed_text):
        retry_msg, rerr = request_with_retry(
            client,
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": (
                        "上一次输出无效。请只输出一行，格式必须严格为：\n"
                        "<points>X out of Y</points>\n"
                        "X、Y 为数字。从下列文本抽取分数；若没有显式分数但明确说解答完整正确，"
                        "可输出 <points>7 out of 7</points>。\n\n"
                        f"{text}"
                    ),
                }
            ],
            retry_times=max(1, min(2, retry_times)),
            timeout_s=timeout_s,
            max_tokens=256,
        )
        if retry_msg and not _llm_line_trivially_bad(retry_msg):
            parsed_text = retry_msg

    s2, m2 = _parse_llm_points_line(parsed_text)

    # 抽取式仍失败：第二步 IMO 推断（处理 judge 未写 out of 7 但评语完整的情况）
    if s2 is None or m2 is None:
        infer_text, _ = request_with_retry(
            client,
            model=model,
            messages=build_messages_infer(text),
            retry_times=max(1, min(2, retry_times)),
            timeout_s=timeout_s,
            max_tokens=128,
        )
        if infer_text:
            s2, m2 = _parse_llm_points_line(infer_text)
            if s2 is not None and m2 is not None:
                parsed_text = infer_text
                used_infer = True

    if s2 is None or m2 is None:
        item = dict(item)
        # 保留 parser 原始输出，便于排查
        item["parser_pred"] = parsed_text
        item["points_parse_error"] = {
            "status": "llm_unparseable",
            "parser_model": model,
            "parser_output": parsed_text,
        }
        return item

    item = dict(item)
    item.setdefault("raw_prediction", item.get("prediction"))
    # 保留 parser 原始输出，便于追溯（即使最终能被规范化解析）
    item["parser_pred"] = parsed_text
    item["prediction"] = _format_points(s2, m2)
    item["points_parsed_by"] = {
        "method": "llm_normalize_infer" if used_infer else "llm_normalize",
        "parser_model": model,
    }
    item.pop("points_parse_error", None)
    return item


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="用 LLM 规范化 <points>X out of Y</points>，补齐规则解析覆盖面。")
    p.add_argument("--input-file", type=Path, required=True, help="eval_mo.py 的输出 JSON（list）")
    p.add_argument("--output-file", type=Path, default=None, help="输出路径（默认覆盖 input）")
    p.add_argument("--model-name", type=str, default="gemini-2.5-flash", help="用于解析 points 的模型名")
    p.add_argument("--tail-chars", type=int, default=8000, help="仅把 prediction 末尾 N 字符送入 parser（0 表示不截断）")
    p.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="API key. Defaults to OPENAI_API_KEY or OPENAI_API_TOKEN.",
    )
    p.add_argument(
        "--base-url",
        type=str,
        default="http://localhost:8000/v1",
        help="OpenAI-compatible API base URL. OPENAI_BASE_URL takes precedence.",
    )
    p.add_argument("--concurrent", action="store_true", help="并发解析（对失败项并发请求）")
    p.add_argument("--max-workers", type=int, default=32, help="并发最大线程数")
    p.add_argument("--retry-times", type=int, default=3, help="单条解析重试次数")
    p.add_argument("--timeout-s", type=int, default=120, help="单条请求超时秒数")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    output_file = args.output_file or args.input_file

    items = load_list_json(args.input_file)

    # 统计需要 LLM 处理的数量
    need = 0
    for it in items:
        s, m = parse_points_prediction(it.get("prediction"))
        if s is None or m is None:
            if isinstance(it.get("prediction"), str) and it.get("prediction").strip():
                need += 1

    print(f"加载 {len(items)} 条结果；需要 LLM 规范化的条目约 {need} 条。", flush=True)

    if need == 0:
        if output_file != args.input_file:
            save_list_json(output_file, items)
        return

    api_key = args.api_key or os.environ.get("OPENAI_API_KEY") or os.environ.get("OPENAI_API_TOKEN")
    base_url = os.environ.get("OPENAI_BASE_URL") or args.base_url
    if not api_key:
        raise ValueError("缺少 API key：请传 --api-key 或设置环境变量 OPENAI_API_KEY")
    print("api_key: ******")
    print(f"base_url: {base_url}")
    client = OpenAI(api_key=api_key, base_url=base_url, timeout=args.timeout_s)

    if not args.concurrent:
        out: List[Dict[str, Any]] = []
        for it in tqdm(items, desc="normalize points"):
            out.append(
                normalize_one(
                    it,
                    client=client,
                    model=args.model_name,
                    retry_times=args.retry_times,
                    timeout_s=args.timeout_s,
                    tail_chars=args.tail_chars,
                )
            )
        save_list_json(output_file, out)
        print(f"已写入: {output_file}", flush=True)
        return

    out = items[:]  # 保持原顺序，逐个回填
    lock = threading.Lock()

    def task(i: int, it: Dict[str, Any]) -> Tuple[int, Dict[str, Any]]:
        return (
            i,
            normalize_one(
                it,
                client=client,
                model=args.model_name,
                retry_times=args.retry_times,
                timeout_s=args.timeout_s,
                tail_chars=args.tail_chars,
            ),
        )

    with ThreadPoolExecutor(max_workers=max(1, min(args.max_workers, len(items)))) as ex:
        futures = {ex.submit(task, i, it): i for i, it in enumerate(items)}
        with tqdm(total=len(futures), desc="normalize points") as bar:
            for fut in as_completed(futures):
                i, normed = fut.result()
                with lock:
                    out[i] = normed
                bar.update(1)

    save_list_json(output_file, out)
    print(f"已写入: {output_file}", flush=True)


if __name__ == "__main__":
    main()
