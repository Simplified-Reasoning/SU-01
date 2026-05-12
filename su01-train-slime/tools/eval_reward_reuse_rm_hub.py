"""
Offline evaluation script that directly reuses the training-time reward
functions (remote_rm / remote_rm_proof_only) from slime.rollout.rm_hub,
ensuring zero logic divergence with the online evaluation path.

Usage:
    python tools/eval_reward_reuse_rm_hub.py \
        --input-json eval_answerbench_rollout_0.json \
        --rm-url http://HOST:8001/ \
        --rm-func remote_rm \
        --eval-use-xverify

    python tools/eval_reward_reuse_rm_hub.py \
        --input-json eval_answerbench_rollout_0.json \
        --rm-url http://HOST:8001/ \
        --rm-func remote_rm_proof_only \
        --proof-rm-url http://HOST:8002/
"""

import argparse
import asyncio
import json
import os
import sys
from dataclasses import field
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from slime.rollout.rm_hub import remote_rm, remote_rm_proof_only
from slime.utils.types import Sample


def build_sample(item: Dict[str, Any]) -> Sample:
    """Convert a JSON dict (exported rollout item) into a Sample object."""
    label = item.get("label")
    metadata = item.get("metadata") or {}
    return Sample(
        group_index=item.get("group_index"),
        index=item.get("index"),
        name=item.get("name"),
        prompt=item.get("prompt", ""),
        response=item.get("response", ""),
        response_length=item.get("response_length", 0),
        label=label,
        metadata=metadata,
    )


def build_args(cli: argparse.Namespace) -> SimpleNamespace:
    """Build a fake args namespace that satisfies what rm_hub functions expect."""
    return SimpleNamespace(
        rm_url=cli.rm_url,
        eval_rm_url=cli.eval_rm_url or cli.rm_url,
        proof_rm_url=cli.proof_rm_url or cli.rm_url,
        eval_use_xverify=cli.eval_use_xverify,
        train_use_xverify=cli.train_use_xverify,
        rm_request_timeout=cli.rm_request_timeout,
        rm_concurrency=cli.concurrency,
        no_remote_rm_force_score_scale=cli.no_force_score_scale,
        proof_reviewer=cli.proof_reviewer,
        proof_reviews=cli.proof_reviews,
        physics_dataset_names=cli.physics_dataset_names,
        physics_rm_url=cli.physics_rm_url,
        rm_route_log_interval=200,
    )


RM_FUNCS = {
    "remote_rm": remote_rm,
    "remote_rm_proof_only": remote_rm_proof_only,
}


async def evaluate_one(
    sem: asyncio.Semaphore,
    args: SimpleNamespace,
    rm_func,
    sample: Sample,
    idx: int,
    is_evaluation: bool,
) -> Tuple[int, Any]:
    async with sem:
        result = await rm_func(args, sample, is_evaluation)
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
        try:
            s = float(rr.get("score", 0.0))
        except Exception:
            s = 0.0
        try:
            p = float(rr.get("point", 0.0))
        except Exception:
            p = 0.0
        scores.append(s)
        points.append(p)
        accs.append(bool(rr.get("acc", False)))

        sb = str(rr.get("scored_by", "unknown"))
        scored_by_hist[sb] = scored_by_hist.get(sb, 0) + 1

    n = len(scores)
    if n == 0:
        return {"n": 0}

    return {
        "n": n,
        "mean_score": sum(scores) / n,
        "mean_point": sum(points) / n,
        "pass_rate": sum(1 for x in accs if x) / n,
        "top_scored_by": sorted(scored_by_hist.items(), key=lambda x: x[1], reverse=True)[:10],
    }


async def main_async(cli: argparse.Namespace) -> None:
    with open(cli.input_json, "r", encoding="utf-8") as f:
        items: List[Dict[str, Any]] = json.load(f)
    if not isinstance(items, list):
        raise ValueError(f"Expected top-level list, got {type(items)}")

    if cli.max_items is not None:
        items = items[: cli.max_items]

    args = build_args(cli)
    rm_func = RM_FUNCS[cli.rm_func]
    is_evaluation = cli.is_evaluation

    print(f"[eval] rm_func={cli.rm_func}, is_evaluation={is_evaluation}, n_items={len(items)}")
    print(f"[eval] rm_url={args.rm_url}, eval_rm_url={args.eval_rm_url}, proof_rm_url={args.proof_rm_url}")

    sem = asyncio.Semaphore(cli.concurrency)
    samples = [build_sample(item) for item in items]

    tasks = [
        evaluate_one(sem, args, rm_func, sample, i, is_evaluation)
        for i, sample in enumerate(samples)
    ]

    done_results: List[Tuple[int, Any]] = []
    for coro in asyncio.as_completed(tasks):
        done_results.append(await coro)

    for idx, rr in done_results:
        items[idx]["remote_reward"] = rr

    summary = compute_summary(items)

    output_path = cli.output_json
    if output_path is None:
        output_path = cli.input_json + ".remote_rm.json"

    out = {"items": items, "summary": summary, "rm_url": args.rm_url, "rm_func": cli.rm_func}
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False)

    print(f"[eval] wrote: {output_path}")
    print(f"[eval] summary: {summary}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Offline eval that reuses rm_hub.remote_rm / remote_rm_proof_only directly.",
    )
    p.add_argument("--input-json", required=True, type=str)
    p.add_argument("--output-json", default=None, type=str)
    p.add_argument(
        "--rm-func",
        type=str,
        default="remote_rm",
        choices=list(RM_FUNCS.keys()),
        help="Which rm_hub function to call.",
    )
    p.add_argument(
        "--is-evaluation",
        action="store_true",
        default=True,
        help="Pass is_evaluation=True to the rm function (default True, matching eval path).",
    )
    p.add_argument("--no-is-evaluation", dest="is_evaluation", action="store_false")
    p.add_argument(
        "--rm-url",
        default=os.environ.get("RM_URL", "http://127.0.0.1:8001/"),
        type=str,
    )
    p.add_argument("--eval-rm-url", default=None, type=str, help="Defaults to --rm-url.")
    p.add_argument("--proof-rm-url", default=None, type=str, help="Defaults to --rm-url.")
    p.add_argument("--physics-rm-url", default=None, type=str)
    p.add_argument("--physics-dataset-names", nargs="*", default=None)
    p.add_argument("--concurrency", type=int, default=32)
    p.add_argument("--rm-request-timeout", type=float, default=0.0)
    p.add_argument("--eval-use-xverify", action="store_true")
    p.add_argument("--train-use-xverify", action="store_true")
    p.add_argument("--no-force-score-scale", action="store_true")
    p.add_argument("--proof-reviewer", type=str, default="ds_proof")
    p.add_argument("--proof-reviews", type=int, default=1)
    p.add_argument("--max-items", type=int, default=None)
    return p.parse_args()


def main() -> None:
    cli = parse_args()
    asyncio.run(main_async(cli))


if __name__ == "__main__":
    main()
