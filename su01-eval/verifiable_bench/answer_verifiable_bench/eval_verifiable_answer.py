import argparse
import asyncio
import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import aiohttp
try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    tqdm = None


_SPECIAL_TOKENS_TO_STRIP = ("<|im_start|>", "<|im_end|>", "<|endoftext|>")


def strip_special_tokens(text: str) -> str:
    for tok in _SPECIAL_TOKENS_TO_STRIP:
        text = text.replace(tok, "")
    return text.strip()


def normalize_label(label: Any) -> Optional[List]:
    """Match rm_hub.remote_rm: wrap str in list, pass list as-is, else None."""
    if label is None:
        return None
    if isinstance(label, str):
        return [label]
    if isinstance(label, list):
        return label
    return None


def extract_rm_response(response_text: Any) -> str:
    if not isinstance(response_text, str):
        return "I don't know"
    if "</think>" in response_text:
        return strip_special_tokens(response_text.split("</think>")[-1])
    # Match rm_hub.remote_rm behavior: if missing </think>, send "I don't know".
    # return "I don't know"
    return response_text


def fallback_reward() -> Dict[str, Any]:
    return {
        "score": 0.0,
        "point": 0.0,
        "acc": False,
        "extracted_gt": "",
        "extracted_pred": "",
        "scored_by": "default_fallback",
        "score_noxverify": 0.0,
        "point_noxverify": 0.0,
    }


@dataclass(frozen=True)
class Config:
    rm_url: str
    concurrency: int
    rm_request_timeout: float
    max_retries: int
    base_delay: float
    use_xverify: bool
    rm_mode: str  # "auto" | "standard" | "proof"
    proof_reviewer: str
    proof_reviews: int


class _SimpleProgress:
    def __init__(self, total: int, desc: str = "progress") -> None:
        self.total = max(0, int(total))
        self.desc = desc
        self.done = 0
        self._last_len = 0
        self._print()

    def _line(self) -> str:
        if self.total <= 0:
            return f"[reward] {self.desc}: 0/0 (100.0%)"
        pct = 100.0 * self.done / self.total
        return f"[reward] {self.desc}: {self.done}/{self.total} ({pct:.1f}%)"

    def _print(self) -> None:
        line = self._line()
        pad = " " * max(0, self._last_len - len(line))
        print(f"\r{line}{pad}", end="", flush=True)
        self._last_len = len(line)

    def update(self, n: int = 1) -> None:
        self.done = min(self.total, self.done + max(0, int(n)))
        self._print()

    def close(self) -> None:
        self._print()
        print("", flush=True)


def _coerce_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    if isinstance(value, str):
        v = value.strip().lower()
        return v in {"1", "true", "t", "yes", "y", "on"}
    return False


def should_use_proof_mode(item: Dict[str, Any]) -> bool:
    """
    Decide proof-mode per item based on metadata, consistent with slime.rollout.rm_hub.remote_rm.

    Rule:
    - use proof mode iff item["metadata"]["is_proof"] is truthy.
    - if metadata is missing or is_proof is absent/falsey -> standard mode.

    Note: do NOT use input json's scored_by for routing.
    """
    metadata = item.get("metadata")
    if isinstance(metadata, dict):
        return _coerce_bool(metadata.get("is_proof", False))
    return False


def extract_problem_from_prompt(prompt_text: Any) -> Optional[str]:
    """
    Try to extract the "Problem: ..." section from the user prompt template.
    This is a best-effort heuristic for proof_verifier requests.
    """
    if not isinstance(prompt_text, str):
        return None
    s = prompt_text
    idx = s.find("Problem:")
    if idx < 0:
        # Fallback: whole prompt without special tokens.
        stripped = strip_special_tokens(s)
        return stripped or None

    end = len(s)
    for marker in ("<|im_end|>", "<|im_start|>"):
        j = s.find(marker, idx + len("Problem:") + 1)
        if j != -1:
            end = min(end, j)
    problem = s[idx + len("Problem:") : end]
    problem = strip_special_tokens(problem)
    return problem or None


async def _post_with_retries(
    session: aiohttp.ClientSession,
    cfg: Config,
    payload: Dict[str, Any],
) -> Dict[str, Any]:
    for attempt in range(cfg.max_retries):
        try:
            req_timeout = None
            if cfg.rm_request_timeout and cfg.rm_request_timeout > 0:
                req_timeout = aiohttp.ClientTimeout(total=cfg.rm_request_timeout)

            async with session.post(cfg.rm_url, json=payload, timeout=req_timeout) as resp:
                if resp.status == 200:
                    return await resp.json()

                # Non-200: try to read error for debug.
                try:
                    error_text = await resp.text()
                except Exception:
                    error_text = "<failed to read error body>"
                msg = (
                    f"[reward] HTTP {resp.status} attempt {attempt + 1}/{cfg.max_retries}, "
                    f"body={error_text}"
                )
                print(msg)
        except Exception as e:
            msg = f"[reward] Network/timeout error attempt {attempt + 1}/{cfg.max_retries}: {type(e).__name__}: {e!r}"
            print(msg)

        if attempt < cfg.max_retries - 1:
            delay = cfg.base_delay * (2**attempt)
            await asyncio.sleep(delay)

    print("[reward] all retries failed, using fallback reward")
    return fallback_reward()


async def evaluate_one(
    sem: asyncio.Semaphore,
    session: aiohttp.ClientSession,
    cfg: Config,
    item: Dict[str, Any],
    idx: int,
) -> Tuple[int, Dict[str, Any]]:
    async with sem:
        mode = cfg.rm_mode
        if mode == "auto":
            mode = "proof" if should_use_proof_mode(item) else "standard"

        payload: Dict[str, Any] = {
            # reward_model_server doesn't define "prompt" in some variants, but rm_hub.remote_rm sends it.
            # FastAPI/Pydantic will ignore extra keys.
            "prompt": item.get("prompt", ""),
            "response": extract_rm_response(item.get("response")),
            "use_xverify": cfg.use_xverify,
        }

        if mode == "proof":
            payload["is_proof"] = True
            payload["reviewer"] = cfg.proof_reviewer
            payload["reviews"] = int(cfg.proof_reviews)
            q = extract_problem_from_prompt(item.get("prompt"))
            if not q:
                return idx, fallback_reward()
            payload["question"] = q
        else:
            label = normalize_label(item.get("label"))
            if label is None:
                return idx, fallback_reward()

            metadata = item.get("metadata") or {}

            raw_points = metadata.get("points", None)
            points = (
                raw_points
                if isinstance(raw_points, list) and all(isinstance(p, float) for p in raw_points)
                else None
            )

            question_raw = metadata.get("question", None)
            question_clean = strip_special_tokens(question_raw) if isinstance(question_raw, str) else None

            payload["label"] = label
            payload["points"] = points
            payload["question"] = question_clean

        result = await _post_with_retries(session=session, cfg=cfg, payload=payload)
        return idx, result


def compute_summary(items: List[Dict[str, Any]]) -> Dict[str, Any]:
    scores: List[float] = []
    points: List[float] = []
    accs: List[bool] = []
    scored_by_hist: Dict[str, int] = {}

    for it in items:
        rr = it.get("remote_reward")
        if not isinstance(rr, dict):
            continue
        s = rr.get("score", 0.0)
        p = rr.get("point", 0.0)
        try:
            s = float(s)
        except Exception:
            s = 0.0
        try:
            p = float(p)
        except Exception:
            p = 0.0
        scores.append(s)
        points.append(p)
        acc = bool(rr.get("acc", False))
        accs.append(acc)

        sb = rr.get("scored_by", "unknown")
        sb = str(sb)
        scored_by_hist[sb] = scored_by_hist.get(sb, 0) + 1

    n = len(scores)
    if n == 0:
        return {"n": 0}

    mean_score = sum(scores) / n
    mean_point = sum(points) / n
    pass_rate = sum(1 for x in accs if x) / n

    # keep top-k scored_by to avoid huge output
    top_scored_by = sorted(scored_by_hist.items(), key=lambda x: x[1], reverse=True)[:10]

    return {
        "n": n,
        "mean_score": mean_score,
        "mean_point": mean_point,
        "pass_rate": pass_rate,
        "top_scored_by": top_scored_by,
    }


async def main_async(
    cfg: Config,
    input_path: str,
    output_path: str,
    max_items: Optional[int],
    show_progress: bool = True,
) -> None:
    with open(input_path, "r", encoding="utf-8") as f:
        items: List[Dict[str, Any]] = json.load(f)
    if not isinstance(items, list):
        raise ValueError(f"Expected top-level list in {input_path}, got {type(items)}")

    if max_items is not None:
        items = items[:max_items]

    sem = asyncio.Semaphore(cfg.concurrency)
    connector = aiohttp.TCPConnector(limit=cfg.concurrency, limit_per_host=0, force_close=False)
    # Keep timeout behavior close to rm_hub: total=None means no overall timeout.
    # If you set --rm-request-timeout, we will override per-request.
    timeout = aiohttp.ClientTimeout(total=None, sock_read=None)
    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        tasks = []
        for i, item in enumerate(items):
            tasks.append(evaluate_one(sem, session, cfg, item, i))

        progress = None
        if show_progress:
            if tqdm is not None:
                progress = tqdm(total=len(tasks), desc="reward", unit="item", dynamic_ncols=True)
            else:
                progress = _SimpleProgress(total=len(tasks), desc="reward")

        # Stream results as they finish to reduce memory spikes.
        done_results: List[Tuple[int, Dict[str, Any]]] = []
        try:
            for coro in asyncio.as_completed(tasks):
                done_results.append(await coro)
                if progress is not None:
                    progress.update(1)
        finally:
            if progress is not None:
                progress.close()

        # Apply results back.
        for idx, rr in done_results:
            items[idx]["remote_reward"] = rr

    summary = compute_summary(items)
    out = {"items": items, "summary": summary, "rm_url": cfg.rm_url}
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False)

    print(f"[reward] wrote: {output_path}")
    print(f"[reward] summary: {summary}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Request reward server and evaluate a rollout json file.")
    p.add_argument(
        "--input-json",
        required=True,
        type=str,
        help="Path to eval_answerbench_rollout_*.json (top-level list of sample dicts).",
    )
    p.add_argument(
        "--output-json",
        default=None,
        type=str,
        help="Output path. Default: <input>.remote_rm.json",
    )
    p.add_argument(
        "--rm-url",
        default=os.environ.get("RM_URL", "http://127.0.0.1:8001/"),
        type=str,
        help='Reward server base URL, e.g. "http://HOST:8001/". POST endpoint is "/".',
    )
    p.add_argument("--concurrency", type=int, default=32, help="Max concurrent HTTP requests.")
    p.add_argument(
        "--rm-request-timeout",
        type=float,
        default=0.0,
        help="Per-request timeout seconds. 0 means no timeout (match rm_hub default).",
    )
    p.add_argument("--use-xverify", action="store_true", help="Set use_xverify=true in reward request.")
    p.add_argument(
        "--rm-mode",
        type=str,
        default="auto",
        choices=["auto", "standard", "proof"],
        help="Reward mode. auto=per-item decide by metadata.is_proof (consistent with rm_hub.remote_rm).",
    )
    p.add_argument(
        "--proof-reviewer",
        type=str,
        default="ds_proof",
        help="Passed as `reviewer` when rm-mode=proof.",
    )
    p.add_argument(
        "--proof-reviews",
        type=int,
        default=1,
        help="Passed as `reviews` when rm-mode=proof.",
    )
    p.add_argument("--max-items", type=int, default=None, help="Only evaluate first N items.")
    p.add_argument("--no-progress", action="store_true", help="Disable progress bar.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    input_path = args.input_json
    output_path = args.output_json
    if output_path is None:
        output_path = input_path + ".remote_rm.json"

    rm_url = args.rm_url.strip()
    if not rm_url.endswith("/"):
        rm_url += "/"

    cfg = Config(
        rm_url=rm_url,
        concurrency=max(1, int(args.concurrency)),
        rm_request_timeout=float(args.rm_request_timeout),
        max_retries=3,
        base_delay=1.0,
        use_xverify=bool(args.use_xverify),
        rm_mode=str(args.rm_mode),
        proof_reviewer=args.proof_reviewer,
        proof_reviews=args.proof_reviews,
    )

    asyncio.run(
        main_async(
            cfg,
            input_path=input_path,
            output_path=output_path,
            max_items=args.max_items,
            show_progress=not args.no_progress,
        )
    )


if __name__ == "__main__":
    main()
