import asyncio
import random

import aiohttp

from slime.utils.metric_utils import has_repetition
from slime.utils.misc import load_function
from slime.utils.types import Sample

from .deepscaler import get_deepscaler_rule_based_reward
from .f1 import f1_score
from .gpqa import compute_gpqa_reward
from .math_dapo_utils import compute_score as compute_score_dapo
from .math_utils import extract_answer as extract_boxed_answer
from .math_utils import grade_answer_verl
from .p1 import compute_score_p1

_SPECIAL_TOKENS_TO_STRIP = ('<|im_start|>', '<|im_end|>', '<|endoftext|>')
_ANTI_HACK_FALLBACK_RESPONSE = "I don't know"

def _strip_special_tokens(text: str) -> str:
    """Remove specific chat-template special tokens that may leak into RM payloads."""
    for tok in _SPECIAL_TOKENS_TO_STRIP:
        text = text.replace(tok, '')
    return text.strip()


def _has_bad_special_token_leak(response: str) -> bool:
    if '<|endoftext|>' in response or '<|im_start|>' in response:
        return True

    # Normal completed generations in this setup may keep one trailing <|im_end|>
    # because sampling uses no_stop_trim=True. Other <|im_end|> placements are malformed.
    if response.count('<|im_end|>') > 1:
        return True
    if '<|im_end|>' in response and not response.rstrip().endswith('<|im_end|>'):
        return True

    return False


def _response_for_remote_rm(sample: Sample) -> str:
    """Return answer text for RM, or a safe fallback for malformed generations."""
    response = sample.response or ""

    if _has_bad_special_token_leak(response):
        return _ANTI_HACK_FALLBACK_RESPONSE

    if response.count("</think>") != 1:
        return _ANTI_HACK_FALLBACK_RESPONSE

    if "<think>" in response:
        return _ANTI_HACK_FALLBACK_RESPONSE

    if has_repetition(response):
        return _ANTI_HACK_FALLBACK_RESPONSE

    answer = response.split("</think>")[-1]
    if has_repetition(answer):
        return _ANTI_HACK_FALLBACK_RESPONSE

    return answer

_RM_SESSION: aiohttp.ClientSession | None = None


def _get_rm_session(args) -> aiohttp.ClientSession:
    """Create or reuse a process-local aiohttp session with connection-pool-based concurrency."""
    global _RM_SESSION
    if _RM_SESSION is None or _RM_SESSION.closed:
        limit = max(1, int(getattr(args, "rm_concurrency", 512)))
        connector = aiohttp.TCPConnector(limit=limit, limit_per_host=0, force_close=False)
        _RM_SESSION = aiohttp.ClientSession(
            connector=connector,
            timeout=aiohttp.ClientTimeout(total=None, sock_read=None),
        )
    return _RM_SESSION


def _maybe_force_remote_rm_score_scale(args, result):
    if getattr(args, "no_remote_rm_force_score_scale", False):
        return result
    if not isinstance(result, dict):
        return result
    try:
        score = float(result.get("score", 0.0))
        if score < 1.0:
            result["score"] = 0.0
    except Exception:
        print(f"Error parsing score from result: {result}")
    return result


async def remote_rm(args, sample: Sample, is_evaluation):
    if is_evaluation:
        remote_rm_url = args.eval_rm_url
    else:
        remote_rm_url = args.rm_url

    response = _response_for_remote_rm(sample)
    # print(f"sample.label: {sample.label}, len(sample.label): {len(sample.label)}")

    question_raw = sample.metadata.get("question", None)
    question_clean = _strip_special_tokens(question_raw) if isinstance(question_raw, str) else None

    if "is_proof" in sample.metadata:
        payload = {
            "prompt": sample.prompt,
            "response": response,
            "label": (
                [sample.label] if isinstance(sample.label, str) else sample.label
                if isinstance(sample.label, list) else None
            ),
            "points": (
                sample.metadata.get("points", None) if isinstance(sample.metadata.get("points", None), list)
                and all(isinstance(p, float) for p in sample.metadata.get("points", []))
                else None
            ),
            "question": question_clean,
            "use_xverify": args.eval_use_xverify if is_evaluation else args.train_use_xverify,
            "is_proof": sample.metadata.get("is_proof", False),
            "reviewer": sample.metadata.get("reviewer", "standard")
        }
    else:
        remote_rm_url = args.eval_rm_url
        payload = {
            "prompt": sample.prompt,
            "response": response,
            "label": (
                [sample.label] if isinstance(sample.label, str) else sample.label
                if isinstance(sample.label, list) else None
            ),
            "points": (
                sample.metadata.get("points", None) if isinstance(sample.metadata.get("points", None), list)
                and all(isinstance(p, float) for p in sample.metadata.get("points", []))
                else None
            ),
            "question": question_clean,
            "use_xverify": args.eval_use_xverify if is_evaluation else args.train_use_xverify
        }
    
    
    # Retry configuration
    max_retries = 3
    base_delay = 1.0  # seconds
    
    for attempt in range(max_retries):
        try:
            rm_request_timeout = float(getattr(args, "rm_request_timeout", 0) or 0)
            req_timeout = aiohttp.ClientTimeout(total=rm_request_timeout) if rm_request_timeout > 0 else None
            session = _get_rm_session(args)
            async with session.post(remote_rm_url, json=payload, timeout=req_timeout) as resp:
                if resp.status == 200:
                    return await resp.json()
                else:
                    # Get the error response for debugging
                    try:
                        error_text = await resp.text()
                        print(
                            f"Remote RM server error (status {resp.status}) on attempt {attempt + 1}/{max_retries}: {error_text}"
                        )
                        print(f"Payload sent: {payload}")
                    except Exception:
                        print(
                            f"Remote RM server error (status {resp.status}) on attempt {attempt + 1}/{max_retries}: Unable to read error response"
                        )
                        print(f"Payload sent: {payload}")

                    if attempt < max_retries - 1:
                        delay = base_delay * (2 ** attempt)  # Exponential backoff
                        print(f"Retrying in {delay} seconds...")
                        await asyncio.sleep(delay)
                    else:
                        # Final attempt failed, return default
                        print(f"All {max_retries} attempts failed. Returning default reward value.")
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
        except Exception as e:
            print(
                f"Network error on attempt {attempt + 1}/{max_retries} "
                f"(url={remote_rm_url}): {type(e).__name__}: {e!r}"
            )
            if attempt < max_retries - 1:
                delay = base_delay * (2 ** attempt)  # Exponential backoff
                print(f"Retrying in {delay} seconds...")
                await asyncio.sleep(delay)
            else:
                # Final attempt failed, return default
                print(f"All {max_retries} attempts failed due to network errors. Returning default reward value.")
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


def _build_payload_with_response(args, sample: Sample, is_evaluation: bool, response: str):
    question_raw = sample.metadata.get("question", None)
    question = _strip_special_tokens(question_raw) if isinstance(question_raw, str) else None
    payload = {
        "prompt": sample.prompt,
        "response": _strip_special_tokens(response),
        "label": (
            [sample.label] if isinstance(sample.label, str) else sample.label
            if isinstance(sample.label, list)
            else None
        ),
        "points": (
            sample.metadata.get("points", None)
            if isinstance(sample.metadata.get("points", None), list)
            and all(isinstance(p, float) for p in sample.metadata.get("points", []))
            else None
        ),
        "question": question,
        "use_xverify": args.eval_use_xverify if is_evaluation else args.train_use_xverify,
    }
    if "is_proof" in sample.metadata:
        payload["is_proof"] = sample.metadata.get("is_proof", False)
        payload["reviewer"] = sample.metadata.get("reviewer", "standard")
    return payload


async def remote_rm_proof(args, sample: Sample, is_evaluation):
    metadata = sample.metadata if isinstance(sample.metadata, dict) else {}
    physics_names = {str(x).strip().lower() for x in (getattr(args, "physics_dataset_names", []) or ["physics"])}
    sample_name = (sample.name or metadata.get("_dataset_name") or "").strip()
    is_physics = sample_name.lower() in physics_names

    if is_evaluation:
        response = _response_for_remote_rm(sample)
        payload = _build_payload_with_response(args, sample, is_evaluation, response)
        target_url = getattr(args, "eval_rm_url", None) or getattr(args, "physics_rm_url", None) or args.rm_url
    elif is_physics:
        response = _response_for_remote_rm(sample)
        payload = _build_payload_with_response(args, sample, is_evaluation, response)
        target_url = getattr(args, "physics_rm_url", None) or getattr(args, "eval_rm_url", None) or args.rm_url
    else:
        response = _response_for_remote_rm(sample)
        payload = _build_payload_with_response(args, sample, is_evaluation, response)
        target_url = getattr(args, "proof_rm_url", None) or args.rm_url
        payload["is_proof"] = True
        payload["reviewer"] = getattr(args, "proof_reviewer", "ds_proof")
        payload["reviews"] = int(getattr(args, "proof_reviews", 1))
        if not payload.get("question"):
            payload["question"] = _strip_special_tokens(str(sample.prompt))

    # Lightweight routing visibility for debugging in training logs.
    route_type = "eval_old_server" if is_evaluation else ("physics_old_server" if is_physics else "proof_new_server")
    log_interval = int(getattr(args, "rm_route_log_interval", 200))
    sample_index = getattr(sample, "index", None)
    should_log = sample_index is None or (isinstance(sample_index, int) and sample_index % max(1, log_interval) == 0)
    if should_log:
        print(
            f"[remote_rm_proof] route={route_type} dataset={sample_name or 'unknown'} "
            f"is_eval={is_evaluation} target_url={target_url}"
        )

    # Retry configuration
    max_retries = 3
    base_delay = 1.0  # seconds

    for attempt in range(max_retries):
        try:
            rm_request_timeout = float(getattr(args, "rm_request_timeout", 0) or 0)
            req_timeout = aiohttp.ClientTimeout(total=rm_request_timeout) if rm_request_timeout > 0 else None
            session = _get_rm_session(args)
            async with session.post(target_url, json=payload, timeout=req_timeout) as resp:
                if resp.status == 200:
                    result = await resp.json()
                    return _maybe_force_remote_rm_score_scale(args, result)
                else:
                    try:
                        error_text = await resp.text()
                        print(
                            f"Remote RM server error (status {resp.status}) on attempt {attempt + 1}/{max_retries}: {error_text}"
                        )
                        print(f"Target URL: {target_url}")
                        print(f"Payload sent: {payload}")
                    except Exception:
                        print(
                            f"Remote RM server error (status {resp.status}) on attempt {attempt + 1}/{max_retries}: Unable to read error response"
                        )
                        print(f"Target URL: {target_url}")
                        print(f"Payload sent: {payload}")

                    if attempt < max_retries - 1:
                        delay = base_delay * (2 ** attempt)
                        print(f"Retrying in {delay} seconds...")
                        await asyncio.sleep(delay)
                    else:
                        print(f"All {max_retries} attempts failed. Returning default reward value.")
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
        except Exception as e:
            print(
                f"Network error on attempt {attempt + 1}/{max_retries} "
                f"(url={target_url}): {type(e).__name__}: {e!r}"
            )
            if attempt < max_retries - 1:
                delay = base_delay * (2 ** attempt)
                print(f"Retrying in {delay} seconds...")
                await asyncio.sleep(delay)
            else:
                print(f"All {max_retries} attempts failed due to network errors. Returning default reward value.")
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


async def remote_rm_proof_only(args, sample: Sample, is_evaluation):
    if is_evaluation:
        remote_rm_url = args.eval_rm_url
    else:
        remote_rm_url = args.rm_url

    response = _response_for_remote_rm(sample)

    question_raw = sample.metadata.get("question", None)
    question_clean = _strip_special_tokens(question_raw) if isinstance(question_raw, str) else None

    if "is_proof" in sample.metadata:
        payload = {
            "prompt": sample.prompt,
            "response": response,
            "label": (
                [sample.label] if isinstance(sample.label, str) else sample.label
                if isinstance(sample.label, list) else None
            ),
            "points": (
                sample.metadata.get("points", None) if isinstance(sample.metadata.get("points", None), list)
                and all(isinstance(p, float) for p in sample.metadata.get("points", []))
                else None
            ),
            "question": question_clean,
            "use_xverify": args.eval_use_xverify if is_evaluation else args.train_use_xverify,
            "is_proof": sample.metadata.get("is_proof", False),
            "reviewer": sample.metadata.get("reviewer", "standard")
        }
        if not is_evaluation:
            remote_rm_url = args.proof_rm_url
            payload["is_proof"] = True
            payload["reviewer"] = "ds_proof"
            payload["reviews"] = int(getattr(args, "proof_reviews", 1))
            if not payload.get("question"):
                payload["question"] = _strip_special_tokens(str(sample.prompt))

    else:
        remote_rm_url = args.eval_rm_url
        payload = {
            "prompt": sample.prompt,
            "response": response,
            "label": (
                [sample.label] if isinstance(sample.label, str) else sample.label
                if isinstance(sample.label, list) else None
            ),
            "points": (
                sample.metadata.get("points", None) if isinstance(sample.metadata.get("points", None), list)
                and all(isinstance(p, float) for p in sample.metadata.get("points", []))
                else None
            ),
            "question": question_clean,
            "use_xverify": args.eval_use_xverify if is_evaluation else args.train_use_xverify
        }
    
    # Retry configuration
    max_retries = 3
    base_delay = 1.0  # seconds

    for attempt in range(max_retries):
        try:
            rm_request_timeout = float(getattr(args, "rm_request_timeout", 0) or 0)
            req_timeout = aiohttp.ClientTimeout(total=rm_request_timeout) if rm_request_timeout > 0 else None
            session = _get_rm_session(args)
            async with session.post(remote_rm_url, json=payload, timeout=req_timeout) as resp:
                if resp.status == 200:
                    result = await resp.json()
                    return _maybe_force_remote_rm_score_scale(args, result)
                else:
                    try:
                        error_text = await resp.text()
                        print(
                            f"Remote RM server error (status {resp.status}) on attempt {attempt + 1}/{max_retries}: {error_text}"
                        )
                        print(f"Remote RM URL: {remote_rm_url}")
                        print(f"Payload sent: {payload}")
                    except Exception:
                        print(
                            f"Remote RM server error (status {resp.status}) on attempt {attempt + 1}/{max_retries}: Unable to read error response"
                        )
                        print(f"Remote RM URL: {remote_rm_url}")
                        print(f"Payload sent: {payload}")

                    if attempt < max_retries - 1:
                        delay = base_delay * (2 ** attempt)
                        print(f"Retrying in {delay} seconds...")
                        await asyncio.sleep(delay)
                    else:
                        print(f"All {max_retries} attempts failed. Returning default reward value.")
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
        except Exception as e:
            print(
                f"Network error on attempt {attempt + 1}/{max_retries} "
                f"(url={remote_rm_url}): {type(e).__name__}: {e!r}"
            )
            if attempt < max_retries - 1:
                delay = base_delay * (2 ** attempt)
                print(f"Retrying in {delay} seconds...")
                await asyncio.sleep(delay)
            else:
                print(f"All {max_retries} attempts failed due to network errors. Returning default reward value.")
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


async def async_rm(args, sample: Sample, **kwargs):
    if args.custom_rm_path is not None:
        rm_function = load_function(args.custom_rm_path)
        return await rm_function(args, sample, **kwargs)

    is_evalution = kwargs.get("evaluation", False)
    
    metadata = sample.metadata if isinstance(sample.metadata, dict) else {}
    rm_type = (metadata.get("rm_type") or args.rm_type or "").strip()
    response = sample.response
    label = sample.label
    if rm_type.startswith("boxed_"):
        response = extract_boxed_answer(response) or ""
        rm_type = rm_type[len("boxed_") :]

    # This function is intended for remote or time-consuming reward model evaluation.
    # Implement the actual logic as needed.
    if rm_type == "remote_rm":
        return await remote_rm(args, sample, is_evalution)
    elif rm_type == "remote_rm_proof":
        return await remote_rm_proof(args, sample, is_evalution)
    elif rm_type == "remote_rm_proof_only":
        return await remote_rm_proof_only(args, sample, is_evalution)
    elif rm_type == "p1":
        return compute_score_p1(response, 
            label if isinstance(label, list) else [label], 
            points=sample.metadata.get("points", None) if isinstance(sample.metadata.get("points", None), list)
            and all(isinstance(p, float) for p in sample.metadata.get("points", []))
            else None,
            use_xverify=args.eval_use_xverify if is_evalution else args.train_use_xverify, 
            base_url=args.rm_url,
            question=sample.metadata.get("question", None) if isinstance(sample.metadata.get("question", None), str) else None
        )
    elif rm_type == "deepscaler":
        return get_deepscaler_rule_based_reward(response, label)
    elif rm_type == "dapo":
        return compute_score_dapo(response, label)
    elif rm_type == "math":
        return 1 if grade_answer_verl(response, label) else 0
    elif rm_type == "f1":
        return f1_score(response, label)[0]
    elif rm_type == "gpqa":
        return compute_gpqa_reward(response, label, metadata=metadata)
    elif rm_type == "ifbench":
        from .ifbench import compute_ifbench_reward

        return compute_ifbench_reward(response, label, metadata=metadata)
    elif rm_type == "random":
        return random.randint(0, 1)
    elif rm_type:
        raise NotImplementedError(f"Rule-based RM for {rm_type} is not implemented.")
    else:
        raise NotImplementedError("Rule-based RM type is not specified.")


async def batched_async_rm(
    args,
    samples: list[Sample],
    **kwargs,
) -> list[int | float]:
    if args.custom_rm_path is not None:
        # Ensure the custom reward function is implemented in batch mode
        rm_function = load_function(args.custom_rm_path)
        return await rm_function(args, samples, **kwargs)
    tasks = [async_rm(args, sample, **kwargs) for sample in samples]
    rewards = await asyncio.gather(*tasks)
    return rewards
