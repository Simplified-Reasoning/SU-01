import asyncio
import itertools
import logging
import math
import multiprocessing
import os
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import ray
import torch
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
from sglang.srt.constants import GPU_MEMORY_TYPE_CUDA_GRAPH, GPU_MEMORY_TYPE_KV_CACHE, GPU_MEMORY_TYPE_WEIGHTS

from slime.backends.sglang_utils.sglang_engine import SGLangEngine
from slime.rollout.base_types import call_rollout_fn
from slime.utils import logging_utils
from slime.utils.health_monitor import RolloutHealthMonitor
from slime.utils.async_utils import run
from slime.utils.http_utils import _wrap_ipv6, find_available_port, get_host_info, init_http_client, post
from slime.utils.logging_utils import configure_logger, init_tracking
from slime.utils.metric_utils import (
    MetricChecker,
    compute_pass_rate,
    compute_rollout_step,
    compute_statistics,
    dict_add_prefix,
)
from slime.utils.misc import Box, group_by, load_function
from slime.utils.processing_utils import encode_image_for_rollout_engine
from slime.utils.seqlen_balancing import get_seqlen_balanced_partitions
from slime.utils.types import Sample

from ..utils.metric_utils import has_repetition
from .utils import NOSET_VISIBLE_DEVICES_ENV_VARS_LIST, Lock

logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

from datetime import datetime
import os

# Global variable for default log file path
_default_log_file = None

def log_with_file(message, log_file=None, args=None):
    """Log message to both console and file with timestamp."""
    global _default_log_file
    
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_message = f"[{timestamp}] {message}"
    print(log_message)
    
    # Get log file path from args if not specified
    if log_file is None:
        if args is not None and hasattr(args, 'log_file_path') and args.log_file_path is not None:
            log_file = args.log_file_path
        else:
            # Create default path with timestamp (once per program run)
            if _default_log_file is None:
                timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
                _default_log_file = f"training_metrics_{timestamp_str}.log"
            log_file = _default_log_file
    
    # Ensure log directory exists
    os.makedirs(os.path.dirname(os.path.abspath(log_file)) if os.path.dirname(log_file) else ".", exist_ok=True)
    
    # Append to log file
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(log_message + "\n")
logger = logging.getLogger(__name__)


@ray.remote
class RolloutManager:
    """The class to run rollout and convert rollout data to training data."""

    def __init__(self, args, pg):
        configure_logger()

        self.args = args
        self.pg = pg
        _start_router(args)
        # TODO make args immutable
        init_tracking(args, primary=False, router_addr=f"http://{args.sglang_router_ip}:{args.sglang_router_port}")
        init_http_client(args)

        self.best_val = 0
        data_source_cls = load_function(self.args.data_source_path)
        self.data_source = data_source_cls(args)

        self.generate_rollout = load_function(self.args.rollout_function_path)
        self.eval_generate_rollout = load_function(self.args.eval_function_path)
        self.custom_reward_post_process_func = None
        if self.args.custom_reward_post_process_path is not None:
            self.custom_reward_post_process_func = load_function(self.args.custom_reward_post_process_path)
        self.custom_convert_samples_to_train_data_func = None
        if self.args.custom_convert_samples_to_train_data_path is not None:
            self.custom_convert_samples_to_train_data_func = load_function(
                self.args.custom_convert_samples_to_train_data_path
            )
        logger.info(f"import {self.args.rollout_function_path} as generate_rollout function.")
        logger.info(f"import {self.args.eval_function_path} as eval_generate_rollout function.")

        if self.args.debug_train_only:
            self.all_rollout_engines = []
        else:
            num_gpu_per_engine = min(args.rollout_num_gpus_per_engine, args.num_gpus_per_node)
            num_engines = args.rollout_num_gpus // num_gpu_per_engine
            self.all_rollout_engines = [None] * num_engines
        self.num_new_engines = init_rollout_engines(args, pg, self.all_rollout_engines)
        self.nodes_per_engine = max(1, args.rollout_num_gpus_per_engine // args.num_gpus_per_node)
        self.rollout_engine_lock = Lock.options(num_cpus=1, num_gpus=0).remote()
        self.rollout_id = -1
        self.actor_entropy_worker = None

        self._metric_checker = MetricChecker.maybe_create(args)
        self._health_monitor = None
        if self.args.use_fault_tolerance:
            self._health_monitor = RolloutHealthMonitor(self, args)
            self._health_monitor.start()  # Start the monitor thread (in paused state)
            self._ci_fault_injection_pending = self.args.ci_test  # Flag for CI fault injection

    def _try_ci_fault_injection(self):
        """Try to inject fault during generate (when health monitor is running)."""
        if not self._ci_fault_injection_pending:
            return

        # Only inject fault once
        self._ci_fault_injection_pending = False

        if self.all_rollout_engines and self.all_rollout_engines[0]:
            logger.info("CI Fault Injection: Simulating crash on engine 0 during generate")
            try:
                # This will cause the ray actor to exit
                self.all_rollout_engines[0].simulate_crash.remote()
                # Wait for health monitor to detect the crash and mark engine as None
                # health_check_interval + health_check_timeout + buffer
                wait_time = self.args.rollout_health_check_interval + self.args.rollout_health_check_timeout + 5
                logger.info(f"CI Fault Injection: Waiting {wait_time}s for health monitor to detect crash")
                time.sleep(wait_time)
            except Exception as e:
                logger.warning(f"CI Fault Injection failed: {e}")

    def dispose(self):
        if self._metric_checker is not None:
            self._metric_checker.dispose()
        if self._health_monitor is not None:
            self._health_monitor.stop()

    # TODO maybe rename "rollout_engines" and "all_rollout_engines" later
    @property
    def rollout_engines(self):
        # when doing multi-node serving, we will only send request to node-0 for each engine.
        return self.all_rollout_engines[:: self.nodes_per_engine]

    def get_rollout_engines_and_lock(self):
        return self.rollout_engines, self.rollout_engine_lock, self.num_new_engines

    def get_num_rollout_per_epoch(self):
        assert self.args.rollout_global_dataset
        return len(self.data_source) // self.args.rollout_batch_size

    def set_actor_entropy_worker(self, actor_worker):
        self.actor_entropy_worker = actor_worker

    @staticmethod
    def _compute_topk_entropy_lower_bound(
        sampled_logprob: float | None,
        sampled_token_id: int | None,
        top_logprobs: list[float] | None,
        top_token_ids: list[int] | None,
    ) -> tuple[float, float]:
        probs_by_token = {}
        if sampled_logprob is not None and sampled_token_id is not None:
            probs_by_token[int(sampled_token_id)] = math.exp(float(sampled_logprob))

        if top_logprobs is not None:
            top_token_ids = top_token_ids or []
            for idx, logprob in enumerate(top_logprobs):
                token_id = top_token_ids[idx] if idx < len(top_token_ids) else None
                if token_id is None:
                    continue
                prob = math.exp(float(logprob))
                prev_prob = probs_by_token.get(int(token_id))
                if prev_prob is None or prob > prev_prob:
                    probs_by_token[int(token_id)] = prob

        covered_mass = min(max(sum(probs_by_token.values()), 0.0), 1.0)
        entropy = 0.0
        for prob in probs_by_token.values():
            if prob > 0.0:
                entropy -= prob * math.log(prob)

        tail_mass = max(0.0, 1.0 - covered_mass)
        if tail_mass > 0.0:
            entropy -= tail_mass * math.log(tail_mass)

        return entropy, covered_mass

    def _extract_sglang_input_logprob_payload(self, output: dict) -> tuple[list, list, list, list]:
        meta_info = output.get("meta_info", output)
        input_logprobs = meta_info.get("input_logprobs")
        if isinstance(input_logprobs, dict):
            return (
                input_logprobs.get("token_logprobs_val") or [],
                input_logprobs.get("token_logprobs_idx") or [],
                input_logprobs.get("top_logprobs_val") or [],
                input_logprobs.get("top_logprobs_idx") or [],
            )

        input_token_logprobs = meta_info.get("input_token_logprobs") or []
        token_logprobs = []
        token_ids = []
        for item in input_token_logprobs:
            if isinstance(item, (list, tuple)):
                token_logprobs.append(item[0] if len(item) > 0 else None)
                token_ids.append(item[1] if len(item) > 1 else None)
            else:
                token_logprobs.append(item)
                token_ids.append(None)

        return (
            token_logprobs,
            token_ids,
            meta_info.get("input_top_logprobs") or [],
            meta_info.get("input_top_logprobs_idx") or [],
        )

    async def _score_exgrpo_sample_with_sglang_topk(self, sample: Sample) -> float:
        if sample.response_length <= 0:
            return 0.0

        prompt_length = len(sample.tokens) - sample.response_length
        if prompt_length < 0:
            raise RuntimeError(
                "ExGRPO SGLang entropy scoring received an invalid sample with "
                f"prompt_length={prompt_length}, total_tokens={len(sample.tokens)}, "
                f"response_length={sample.response_length}"
            )

        payload = {
            "input_ids": sample.tokens,
            "sampling_params": {
                "temperature": 0,
                "top_p": 1.0,
                "top_k": 1,
                "max_new_tokens": 1,
                "skip_special_tokens": False,
                "no_stop_trim": True,
                "spaces_between_special_tokens": False,
            },
            "return_logprob": True,
            "logprob_start_len": prompt_length,
            "top_logprobs_num": getattr(self.args, "exgrpo_sglang_topk_num", 32),
            "stream": False,
        }

        if sample.multimodal_inputs and sample.multimodal_inputs.get("images"):
            payload["image_data"] = [
                encode_image_for_rollout_engine(image) for image in sample.multimodal_inputs["images"]
            ]

        output = await post(
            f"http://{self.args.sglang_router_ip}:{self.args.sglang_router_port}/generate",
            payload,
        )

        token_logprobs, token_ids, top_logprobs, top_token_ids = self._extract_sglang_input_logprob_payload(output)
        if len(token_logprobs) != sample.response_length:
            raise RuntimeError(
                "ExGRPO SGLang top-k entropy returned a mismatched number of input logprobs: "
                f"expected {sample.response_length}, got {len(token_logprobs)}"
            )

        token_entropies = []
        covered_masses = []
        for idx in range(sample.response_length):
            entropy, covered_mass = self._compute_topk_entropy_lower_bound(
                token_logprobs[idx] if idx < len(token_logprobs) else None,
                token_ids[idx] if idx < len(token_ids) else None,
                top_logprobs[idx] if idx < len(top_logprobs) else None,
                top_token_ids[idx] if idx < len(top_token_ids) else None,
            )
            token_entropies.append(entropy)
            covered_masses.append(covered_mass)

        mean_entropy = float(sum(token_entropies) / max(len(token_entropies), 1))
        mean_covered_mass = float(sum(covered_masses) / max(len(covered_masses), 1))
        if not np.isfinite(mean_entropy):
            raise RuntimeError("ExGRPO SGLang top-k entropy produced a non-finite sample score")
        if not np.isfinite(mean_covered_mass):
            raise RuntimeError("ExGRPO SGLang top-k entropy produced a non-finite coverage score")
        return mean_entropy

    async def _score_exgrpo_chunk_with_sglang_topk(self, chunk: list[Sample]) -> list[float]:
        return await asyncio.gather(*[self._score_exgrpo_sample_with_sglang_topk(sample) for sample in chunk])

    def _prepare_exgrpo_experience_selection(self):
        if not hasattr(self.data_source, "collect_experience_candidates"):
            return

        all_candidate_samples, candidate_indices = self.data_source.collect_experience_candidates()
        if not all_candidate_samples:
            return

        batch_size = max(1, getattr(self.args, "exgrpo_entropy_batch_size", 64))
        entropy_source = getattr(self.args, "exgrpo_entropy_source", "megatron_exact")
        logger.info(
            "ExGRPO entropy scoring start: "
            f"candidate_count={len(all_candidate_samples)}, batch_size={batch_size}, source={entropy_source}"
        )
        entropies = []
        for start in range(0, len(all_candidate_samples), batch_size):
            chunk = all_candidate_samples[start : start + batch_size]
            if entropy_source == "megatron_exact":
                if self.actor_entropy_worker is None:
                    raise RuntimeError(
                        "ExGRPO requires Megatron actor-backed true entropy scoring, "
                        "but no actor entropy worker is registered."
                    )
                chunk_entropies = ray.get(self.actor_entropy_worker.compute_experience_entropy.remote(chunk))
            elif entropy_source == "sglang_topk":
                chunk_entropies = run(self._score_exgrpo_chunk_with_sglang_topk(chunk))
            else:
                raise RuntimeError(f"Unknown ExGRPO entropy source: {entropy_source}")
            if len(chunk_entropies) != len(chunk):
                raise RuntimeError(
                    "ExGRPO entropy scorer returned mismatched score count: "
                    f"expected {len(chunk)}, got {len(chunk_entropies)} "
                    f"for chunk starting at candidate index {start}."
                )
            for i, entropy in enumerate(chunk_entropies):
                if not np.isfinite(entropy):
                    raise RuntimeError(
                        "ExGRPO entropy scorer returned a non-finite entropy score: "
                        f"value={entropy} at chunk candidate offset {start + i}."
                    )
            entropies.extend(chunk_entropies)

        if len(entropies) != len(candidate_indices):
            raise RuntimeError(
                "ExGRPO entropy scorer returned mismatched total score count: "
                f"expected {len(candidate_indices)}, got {len(entropies)}."
            )
        logger.info(
            "ExGRPO entropy scoring complete: "
            f"scored_candidates={len(entropies)}, source={entropy_source}"
        )
        self.data_source.set_precomputed_entropies(candidate_indices, entropies)

    def generate(self, rollout_id):
        start_time = time.time()
        self.rollout_id = rollout_id
        self.health_monitoring_resume()
        if self.args.ci_test and self.args.use_fault_tolerance and rollout_id >= 2:
            self._try_ci_fault_injection()
        # Pre-compute entropy for ExGRPO experience selection (must be in sync context,
        # before _get_rollout_data which enters async event loop)
        if hasattr(self.data_source, "collect_experience_candidates"):
            print(f"[ExGRPO] rollout {rollout_id}: entering entropy precompute", flush=True)
            self._prepare_exgrpo_experience_selection()
        data, metrics = self._get_rollout_data(rollout_id=rollout_id)
        # self._update_accuracy_tracking(data)
        self._save_debug_rollout_data(data, rollout_id=rollout_id, evaluation=False)
        # Update accuracy tracking for replay filtering
        self._update_accuracy_tracking(data)
        if hasattr(self.data_source, 'inject_replay_trajectories'):
            data = self.data_source.inject_replay_trajectories(data)
        # Collect self-refine samples from unsolved queries for next batch
        if hasattr(self.data_source, 'add_self_refine_samples'):
            self._collect_self_refine_samples(data)
        # Collect successful samples into ExGRPO experience pool
        if hasattr(self.data_source, 'add_experience_samples'):
            self._collect_experience_samples(data)
        _log_rollout_data(
            rollout_id, self.args, data, metrics, time.time() - start_time,
            self_refine_metrics=self._get_self_refine_metrics(),
            exgrpo_metrics=self._get_exgrpo_metrics(),
        )
        data = self._convert_samples_to_train_data(data)
        return self._split_train_data_by_dp(data, self.train_parallel_config["dp_size"])

    def eval(self, rollout_id):
        if self.args.debug_train_only:
            # if debug train only, we don't generate evaluation data
            return
        self.health_monitoring_resume()

        result = call_rollout_fn(self.eval_generate_rollout, self.args, rollout_id, self.data_source, evaluation=True)
        data = result.data
        # self._save_debug_rollout_data(data, rollout_id=rollout_id, evaluation=True)
        val, metrics = _log_eval_rollout_data(rollout_id, self.args, data, result.metrics)
        if self._metric_checker is not None:
            self._metric_checker.on_eval(metrics)
        best = False

        if val > self.best_val:
            self.best_val = val
            best = True
            print(f"eval {rollout_id}: best val {val}")
        return best

    def save(self, rollout_id):
        self.data_source.save(rollout_id)

    def load(self, rollout_id=None):
        self.data_source.load(rollout_id)

    def offload(self):
        self.health_monitoring_pause()
        return ray.get(
            [engine.release_memory_occupation.remote() for engine in self.rollout_engines if engine is not None]
        )

    def onload(self, tags: list[str] | None = None):
        return ray.get(
            [
                engine.resume_memory_occupation.remote(tags=tags)
                for engine in self.rollout_engines
                if engine is not None
            ]
        )

    def onload_weights(self):
        self.onload(tags=[GPU_MEMORY_TYPE_WEIGHTS])

    def onload_kv(self):
        self.onload(tags=[GPU_MEMORY_TYPE_KV_CACHE, GPU_MEMORY_TYPE_CUDA_GRAPH])

    def recover_rollout_engines(self):
        """Restart any dead rollout engines and update num_new_engines for update_weights detection."""
        self.health_monitoring_pause()
        if self.rollout_id == -1:
            return self.rollout_engines, self.rollout_engine_lock, self.num_new_engines

        dead_indices = [i for i, engine in enumerate(self.all_rollout_engines) if engine is None]
        self.num_new_engines = init_rollout_engines(self.args, self.pg, self.all_rollout_engines)
        logger.info(f"Recovered {self.num_new_engines} dead rollout engines")
        assert self.num_new_engines == len(dead_indices), "num_new_engines does not match dead_indices length"
        if self.args.offload_rollout and dead_indices:
            new_engines = [self.all_rollout_engines[i] for i in dead_indices]
            ray.get([engine.release_memory_occupation.remote() for engine in new_engines])
            ray.get([engine.resume_memory_occupation.remote(tags=[GPU_MEMORY_TYPE_WEIGHTS]) for engine in new_engines])

        return self.rollout_engines, self.rollout_engine_lock, self.num_new_engines

    def clear_num_new_engines(self):
        # when fault tolerance is not enabled, we need to manually clear num_new_engines after update_weights
        self.num_new_engines = 0

    def health_monitoring_pause(self) -> None:
        if self._health_monitor is not None:
            self._health_monitor.pause()

    def health_monitoring_resume(self) -> None:
        if self._health_monitor is not None:
            self._health_monitor.resume()

    def check_weights(self, action: str):
        return ray.get([engine.check_weights.remote(action=action) for engine in self.rollout_engines])

    def _get_rollout_data(self, rollout_id):
        if self.args.load_debug_rollout_data:
            data = torch.load(
                self.args.load_debug_rollout_data.format(rollout_id=rollout_id),
                weights_only=False,
            )["samples"]
            data = [Sample.from_dict(sample) for sample in data]
            if (ratio := self.args.load_debug_rollout_data_subsample) is not None:
                original_num_rows = len(data)
                rough_subsample_num_rows = int(original_num_rows * ratio)
                data = data[: rough_subsample_num_rows // 2] + data[-rough_subsample_num_rows // 2 :]
                logger.info(
                    f"Subsample loaded debug rollout data using {ratio=} and change num rows {original_num_rows} -> {len(data)}"
                )
            metrics = None
        else:
            data = call_rollout_fn(self.generate_rollout, self.args, rollout_id, self.data_source, evaluation=False)
            metrics = data.metrics
            data = data.samples
            # flatten the data if it is a list of lists
            while isinstance(data[0], list):
                data = list(itertools.chain.from_iterable(data))

            if not self.args.disable_rollout_trim_samples:
                global_batch_size = self.args.global_batch_size
                if self.args.use_dynamic_global_batch_size:
                    logger.info(f"Collected {len(data)} samples from rollout to train with dynamic global batch size")
                    # TODO: this is a temporary solution, we should directly save dynamic_global_batch_size to rollout data
                    self._dynamic_global_batch_size = self._compute_dynamic_global_batch_size(len(data))
                    global_batch_size = self._dynamic_global_batch_size

                if len(data) % global_batch_size != 0:
                    trim_len = (len(data) // global_batch_size) * global_batch_size
                    if trim_len == 0:
                        raise ValueError(f"Not enough samples {len(data)} for global_batch_size {global_batch_size}")
                    origin_data_length = len(data)
                    data = data[:trim_len]
                    logger.info(f"trim number of samples from {origin_data_length} to {trim_len}")
                logger.info(f"Final collected {len(data)} samples from rollout to train")

        return data, metrics

    def _update_accuracy_tracking(self, samples: list[Sample]):
        """Update accuracy tracking for each query based on rollout results.
        
        This is used for replay filtering to skip queries that are too easy.
        """
        if not hasattr(self.data_source, 'update_query_accuracy'):
            return
        
        # Determine correctness based on reward
        reward_threshold = getattr(self.args, 'accuracy_reward_threshold', 0.9)
        
        # Track statistics for logging
        num_correct = 0
        num_tracked = 0
        
        for sample in samples:
            if sample.query_id is None:
                continue

            if not self.data_source.should_track_query_accuracy(sample):
                continue
            
            # Determine if this rollout was correct
            reward = sample.get_reward_value(self.args) if hasattr(sample, 'get_reward_value') else sample.reward
            if isinstance(reward, dict):
                # If reward is a dict, use a specific key or check for correctness
                is_correct = reward.get('correct', False) if 'correct' in reward else reward.get('score', 0) >= reward_threshold
            else:
                # If reward is a scalar, use threshold
                is_correct = reward >= reward_threshold if reward is not None else False
            
            # Update accuracy for EVERY single rollout result
            self.data_source.update_query_accuracy(sample.query_id, is_correct)
            num_tracked += 1
            if is_correct:
                num_correct += 1
        
        # Log summary statistics
        if num_tracked > 0:
            accuracy_rate = num_correct / num_tracked
            logger.info(f"Accuracy tracking: {num_correct}/{num_tracked} rollouts correct ({accuracy_rate:.2%})")
            
            # Log how many queries are at risk of being filtered
            if getattr(self.data_source, 'replay_filtering', False):
                queries_at_threshold = 0
                for query_id, history in self.data_source.query_accuracy.items():
                    window_size = getattr(self.args, 'accuracy_window_size', 3)
                    if len(history) >= window_size and all(history[-window_size:]):
                        queries_at_threshold += 1
                logger.info(f"Replay filtering: {queries_at_threshold} queries eligible for filtering")
    
    def _collect_self_refine_samples(self, samples: list[Sample]):
        """Collect wrong responses from unsolved query groups for self-refine.

        Groups samples by group_index, identifies groups whose average reward
        is below the threshold, and adds wrong individual samples to the
        data_source self-refine buffer for mixing into future batches.

        Also computes metrics that split current-batch results into refine vs
        normal samples so we can track whether the model actually solves
        previously-failed questions after being prompted to self-refine.
        """
        if not hasattr(self.data_source, 'add_self_refine_samples'):
            return

        group_reward_threshold = getattr(self.args, 'self_refine_group_reward_threshold', 0.5)
        individual_reward_threshold = getattr(self.args, 'accuracy_reward_threshold', 0.9)

        def _reward_scalar(s):
            r = s.get_reward_value(self.args) if hasattr(s, 'get_reward_value') else s.reward
            if isinstance(r, dict):
                r = r.get('score', 0)
            return r if r is not None else 0.0

        # --- Step 1: split current batch into refine vs normal ---
        refine_samples = []
        normal_samples = []
        for s in samples:
            if s.metadata and s.metadata.get('_is_self_refine', False):
                refine_samples.append(s)
            elif s.metadata and s.metadata.get('is_off_policy', False):
                continue
            else:
                normal_samples.append(s)

        refine_rewards = [_reward_scalar(s) for s in refine_samples]
        normal_rewards = [_reward_scalar(s) for s in normal_samples]

        refine_solved = sum(1 for r in refine_rewards if r >= individual_reward_threshold)
        normal_solved = sum(1 for r in normal_rewards if r >= individual_reward_threshold)

        # --- Step 2: collect new wrong samples for next batch ---
        # Only collect from normal (non-refine) samples. Refine samples that
        # still fail are NOT re-collected, so each question gets at most one
        # refine attempt — preventing hard questions from looping indefinitely.
        normal_sample_groups = group_by(normal_samples, lambda s: s.group_index)

        wrong_samples = []
        num_unsolved_groups = 0

        for _group_idx, group in normal_sample_groups.items():
            rewards = [_reward_scalar(s) for s in group]
            avg_reward = sum(rewards) / len(rewards) if rewards else 0.0

            if avg_reward < group_reward_threshold:
                num_unsolved_groups += 1
                for s, r in zip(group, rewards):
                    if r < individual_reward_threshold and s.response:
                        s.metadata['_pre_refine_reward'] = r
                        wrong_samples.append(s)

        self.data_source.add_self_refine_samples(wrong_samples)

        # --- Step 3: store metrics ---
        num_total_groups = len(normal_sample_groups)

        # Reward improvement: avg reward after refine minus avg reward before refine
        pre_refine_rewards = [
            s.metadata.get('_pre_refine_reward', 0.0)
            for s in refine_samples if s.metadata
        ]
        refine_avg_post = sum(refine_rewards) / len(refine_rewards) if refine_rewards else 0.0
        refine_avg_pre = sum(pre_refine_rewards) / len(pre_refine_rewards) if pre_refine_rewards else 0.0
        refine_reward_delta = refine_avg_post - refine_avg_pre if refine_samples else 0.0

        self._last_self_refine_metrics = {
            # Current batch: refine vs normal breakdown
            "self_refine/batch_refine_count": len(refine_samples),
            # "self_refine/batch_normal_count": len(normal_samples),
            "self_refine/batch_refine_ratio": len(refine_samples) / len(samples) if samples else 0.0,
            # Key effectiveness metric: did refine queries get solved?
            # "self_refine/refine_solved": refine_solved,
            # "self_refine/refine_solve_rate": refine_solved / len(refine_samples) if refine_samples else 0.0,
            "self_refine/refine_avg_reward": refine_avg_post,
            # "self_refine/refine_avg_reward_pre": refine_avg_pre,
            "self_refine/refine_reward_delta": refine_reward_delta,
            # Normal samples for comparison
            # "self_refine/normal_solved": normal_solved,
            # "self_refine/normal_solve_rate": normal_solved / len(normal_samples) if normal_samples else 0.0,
            "self_refine/normal_avg_reward": sum(normal_rewards) / len(normal_rewards) if normal_rewards else 0.0,
            # Collection for next batch
            # "self_refine/unsolved_groups": num_unsolved_groups,
            # "self_refine/unsolved_ratio": num_unsolved_groups / num_total_groups if num_total_groups > 0 else 0.0,
            "self_refine/wrong_samples_collected": len(wrong_samples),
            "self_refine/buffer_queries_next": self.data_source.get_self_refine_buffer_length(),
            "self_refine/normal_buffer_size": self.data_source.get_buffer_length(),
        }

        refine_rate_str = f"{refine_solved/len(refine_samples):.2%}" if refine_samples else "N/A"
        normal_rate_str = f"{normal_solved/len(normal_samples):.2%}" if normal_samples else "N/A"
        delta_str = f"{refine_reward_delta:+.4f}" if refine_samples else "N/A"
        logger.info(
            f"Self-refine stats: "
            f"batch=[refine={len(refine_samples)} (solved={refine_solved}, rate={refine_rate_str}, "
            f"reward {refine_avg_pre:.4f}->{refine_avg_post:.4f}, delta={delta_str}), "
            f"normal={len(normal_samples)} (solved={normal_solved}, rate={normal_rate_str})], "
            f"next_buffer={self.data_source.get_self_refine_buffer_length()} queries "
            f"({len(wrong_samples)} collected from {num_unsolved_groups}/{num_total_groups} unsolved groups)"
        )

    def _collect_experience_samples(self, samples: list[Sample]):
        """Group rollout samples and store into ExGRPO experience pool."""
        from collections import defaultdict
        groups = defaultdict(list)
        for s in samples:
            groups[s.group_index].append(s)

        sample_groups = list(groups.values())
        self.data_source.add_experience_samples(sample_groups)

    def _get_self_refine_metrics(self) -> dict:
        """Return self-refine metrics for the current rollout, if available."""
        return getattr(self, '_last_self_refine_metrics', {})

    def _get_exgrpo_metrics(self) -> dict:
        """Return ExGRPO metrics for the current rollout, if available."""
        if hasattr(self.data_source, 'get_exgrpo_metrics'):
            return self.data_source.get_exgrpo_metrics()
        return {}

    def _compute_dynamic_global_batch_size(self, num_samples: int) -> int:
        """Calculate dynamic global_batch_size to ensure only one training step.

        Strategy: global_batch_size = num_samples rounded down to a multiple of dp_size
        This ensures num_steps_per_rollout = num_samples // global_batch_size = 1
        """
        dp_size = self.train_parallel_config["dp_size"]
        original_gbs = self.args.global_batch_size

        # Round down to a multiple of dp_size to ensure only one training step
        dynamic_gbs = (num_samples // dp_size) * dp_size

        if dynamic_gbs == 0:
            # Too few samples, use at least dp_size
            dynamic_gbs = dp_size
            logger.warning(f"num_samples={num_samples} < dp_size={dp_size}, using dp_size as global_batch_size")

        # Calculate how many samples will be discarded
        wasted = num_samples - dynamic_gbs

        if dynamic_gbs != original_gbs or wasted > 0:
            logger.info(
                f"Dynamic global_batch_size: {original_gbs} -> {dynamic_gbs} "
                f"(num_samples={num_samples}, dp_size={dp_size}, "
                f"num_steps=1, wasted={wasted})"
            )

        return dynamic_gbs

    def _save_debug_rollout_data(self, data, rollout_id, evaluation: bool):
        # TODO to be refactored (originally Buffer._set_data)
        if (path_template := self.args.save_debug_rollout_data) is not None:
            path = Path(path_template.format(rollout_id=("eval_" if evaluation else "") + str(rollout_id)))
            logger.info(f"Save debug rollout data to {path}")
            path.parent.mkdir(parents=True, exist_ok=True)

            # TODO may improve the format
            if evaluation:
                dump_data = dict(
                    samples=[sample.to_dict() for dataset_name, info in data.items() for sample in info["samples"]]
                )
            else:
                dump_data = dict(
                    samples=[sample.to_dict() for sample in data],
                )

            torch.save(dict(rollout_id=rollout_id, **dump_data), path)

    def _post_process_rewards(self, samples: list[Sample] | list[list[Sample]]):
        if self.custom_reward_post_process_func is not None:
            return self.custom_reward_post_process_func(self.args, samples)

        raw_rewards = [sample.get_reward_value(self.args) for sample in samples]
        if (
            self.args.advantage_estimator in ["grpo", "gspo", "reinforce_plus_plus_baseline"]
            and self.args.rewards_normalization
        ):
            # group norm
            rewards = torch.tensor(raw_rewards, dtype=torch.float)
            if rewards.shape[-1] == self.args.n_samples_per_prompt * self.args.rollout_batch_size:
                rewards = rewards.reshape(-1, self.args.n_samples_per_prompt)
            else:
                # when samples count are not equal in each group
                rewards = rewards.view(-1, rewards.shape[-1])
            mean = rewards.mean(dim=-1, keepdim=True)
            rewards = rewards - mean

            if self.args.advantage_estimator in ["grpo", "gspo"] and self.args.grpo_std_normalization:
                std = rewards.std(dim=-1, keepdim=True)
                rewards = rewards / (std + 1e-6)

            return raw_rewards, rewards.flatten().tolist()

        return raw_rewards, raw_rewards

    def _convert_samples_to_train_data(self, samples: list[Sample] | list[list[Sample]]):
        """
        Convert inference generated samples to training data.
        """
        if self.custom_convert_samples_to_train_data_func is not None:
            return self.custom_convert_samples_to_train_data_func(self.args, samples)

        raw_rewards, rewards = self._post_process_rewards(samples)

        assert len(raw_rewards) == len(samples)
        assert len(rewards) == len(samples)
        
        train_data = {
            "tokens": [sample.tokens for sample in samples],
            "response_lengths": [sample.response_length for sample in samples],
            "rewards": rewards,
            "raw_reward": raw_rewards,
            "truncated": [1 if sample.status == Sample.Status.TRUNCATED else 0 for sample in samples],
            "sample_indices": [sample.index for sample in samples],
        }
        
        loss_masks = []
        for sample in samples:
            # always instantiate loss_mask if not provided
            if sample.loss_mask is None:
                sample.loss_mask = [1] * sample.response_length

            assert (
                len(sample.loss_mask) == sample.response_length
            ), f"loss mask length {len(sample.loss_mask)} != response length {sample.response_length}"
            if sample.remove_sample:
                sample.loss_mask = [0] * sample.response_length
            loss_masks.append(sample.loss_mask)
        train_data["loss_masks"] = loss_masks
            
        if samples[0].metadata and "raw_reward" in samples[0].metadata:
            train_data["raw_reward"] = [sample.metadata["raw_reward"] for sample in samples]

        # For rollout buffer
        if samples[0].metadata and "round_number" in samples[0].metadata:
            train_data["round_number"] = [sample.metadata["round_number"] for sample in samples]

        # Add tool_call_count for logging
        if hasattr(samples[0], "tool_call_count") and samples[0].tool_call_count is not None:
            train_data["tool_call_count"] = [getattr(sample, "tool_call_count", 0) for sample in samples]

        # Add rollout log probabilities for off-policy correction
        if any(sample.rollout_log_probs is not None for sample in samples):
            train_data["rollout_log_probs"] = [sample.rollout_log_probs for sample in samples]

        # Add recorded old log probs for ExGRPO experience samples
        if any(sample.metadata and "recorded_old_log_prob" in sample.metadata for sample in samples):
            train_data["recorded_old_log_probs"] = [
                sample.metadata.get("recorded_old_log_prob") for sample in samples
            ]
            train_data["is_off_policy"] = [
                sample.metadata.get("is_off_policy", False) for sample in samples
            ]

        if samples[0].rollout_routed_experts is not None:
            train_data["rollout_routed_experts"] = [sample.rollout_routed_experts for sample in samples]

        if samples[0].train_metadata is not None:
            train_data["metadata"] = [sample.train_metadata for sample in samples]
        
        # Add points for logging
        if samples[0].reward is not None and isinstance(samples[0].reward, dict) and "point" in samples[0].reward:
            train_data["points"] = [sample.reward.get("point", 0.0) if isinstance(sample.reward, dict) else 0.0 for sample in samples]

        if samples[0].multimodal_train_inputs is not None:
            train_data["multimodal_train_inputs"] = [sample.multimodal_train_inputs for sample in samples]

        if samples[0].teacher_log_probs is not None:
            train_data["teacher_log_probs"] = [sample.teacher_log_probs for sample in samples]

        return train_data

    def set_train_parallel_config(self, config: dict):
        self.train_parallel_config = config

    def _split_train_data_by_dp(self, data, dp_size):
        """Split the train data by data parallel size."""
        rollout_data = {}

        if "prompt" in data:
            rollout_data["prompt"] = data["prompt"]

        total_lengths = [len(t) for t in data["tokens"]]
        data["total_lengths"] = total_lengths

        if self.args.balance_data:
            partitions = get_seqlen_balanced_partitions(total_lengths, dp_size, equal_size=True)
        else:
            partitions = [range(i, len(total_lengths), dp_size) for i in range(dp_size)]

        rollout_data_refs = []

        for i in range(dp_size):
            rollout_data = {}
            partition = partitions[i]
            rollout_data["partition"] = partition
            for key in [
                "tokens",
                "multimodal_train_inputs",
                "response_lengths",
                "rewards",
                "truncated",
                "loss_masks",
                "round_number",
                "sample_indices",
                "rollout_log_probs",
                "recorded_old_log_probs",
                "is_off_policy",
                "rollout_routed_experts",
                "prompt",
                "teacher_log_probs",
            ]:
                if key not in data:
                    continue
                val = [data[key][j] for j in partition]
                rollout_data[key] = val
            # keys that need to be splited at train side
            for key in [
                "raw_reward",
                "total_lengths",
            ]:
                if key not in data:
                    continue
                rollout_data[key] = data[key]
            # Pass dynamic global_batch_size to training side
            if hasattr(self, "_dynamic_global_batch_size"):
                rollout_data["dynamic_global_batch_size"] = self._dynamic_global_batch_size
            rollout_data_refs.append(Box(ray.put(rollout_data)))
        return rollout_data_refs


def init_rollout_engines(args, pg, all_rollout_engines):
    if args.debug_train_only:
        return 0

    num_gpu_per_engine = min(args.rollout_num_gpus_per_engine, args.num_gpus_per_node)
    num_engines = args.rollout_num_gpus // num_gpu_per_engine
    assert len(all_rollout_engines) == num_engines
    if args.prefill_num_servers is not None:
        prefill_num_servers = args.prefill_num_servers * args.rollout_num_gpus_per_engine // num_gpu_per_engine
        assert (
            num_engines > prefill_num_servers
        ), f"num_engines {num_engines} should be larger than prefill_num_servers {prefill_num_servers}"

    pg, reordered_bundle_indices, reordered_gpu_ids = pg

    RolloutRayActor = ray.remote(SGLangEngine)

    rollout_engines = []
    for i in range(num_engines):
        if all_rollout_engines[i] is not None:
            continue

        num_gpus = 0.2
        num_cpus = num_gpus

        # Get the base GPU ID from placement group
        base_gpu_id = int(reordered_gpu_ids[i * num_gpu_per_engine])

        scheduling_strategy = PlacementGroupSchedulingStrategy(
            placement_group=pg,
            placement_group_capture_child_tasks=True,
            placement_group_bundle_index=reordered_bundle_indices[i * num_gpu_per_engine],
        )

        env_vars = {name: "1" for name in NOSET_VISIBLE_DEVICES_ENV_VARS_LIST} | {
            key: os.environ.get(key, default_val)
            for key, default_val in {
                "SGLANG_JIT_DEEPGEMM_PRECOMPILE": "false",
                "SGL_DISABLE_TP_MEMORY_INBALANCE_CHECK": "true",
                "SGLANG_DISABLE_TP_MEMORY_INBALANCE_CHECK": "true",
                "SGLANG_MEMORY_SAVER_CUDA_GRAPH": "true",
                "SGLANG_BATCH_INVARIANT_OPS_ENABLE_MM_FALLBACK_VARIANT": "true",
                "SGLANG_ENABLE_HEALTH_ENDPOINT_GENERATION": "false",
                "SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_IDLE": "false",
            }.items()
        }

        worker_type = "regular"
        if args.prefill_num_servers is not None:
            if i < prefill_num_servers:
                worker_type = "prefill"
            else:
                worker_type = "decode"

        rollout_engine = RolloutRayActor.options(
            num_cpus=num_cpus,
            num_gpus=num_gpus,
            scheduling_strategy=scheduling_strategy,
            runtime_env={
                "env_vars": env_vars,
            },
        ).remote(args, rank=i, worker_type=worker_type, base_gpu_id=base_gpu_id)

        rollout_engines.append((i, rollout_engine))
        all_rollout_engines[i] = rollout_engine

    num_new_engines = len(rollout_engines)

    if num_new_engines == 0:
        return num_new_engines

    if args.rollout_external:
        addr_and_ports = _allocate_rollout_engine_addr_and_ports_external(args=args, rollout_engines=rollout_engines)
    else:
        addr_and_ports = _allocate_rollout_engine_addr_and_ports_normal(
            args=args, num_engines=num_engines, rollout_engines=rollout_engines
        )

    # TODO: don't ray.get here to overlap train actor init with rollout engine init.
    # somehow if we don't sync here, the --debug-rollout-only mode will crash.
    init_handles = [engine.init.remote(**(addr_and_ports[rank])) for rank, engine in rollout_engines]
    ray.get(init_handles)

    return num_new_engines


def _allocate_rollout_engine_addr_and_ports_external(args, rollout_engines):
    addr_and_ports = []
    for rank, _ in rollout_engines:
        addr = args.rollout_external_engine_addrs[rank]
        [host, port] = addr.split(":")
        addr_and_ports.append(
            dict(
                dist_init_addr=addr,
                nccl_port=None,
                host=host,
                port=int(port),
            )
        )
    return addr_and_ports


def _allocate_rollout_engine_addr_and_ports_normal(*, args, num_engines, rollout_engines):
    # get ports
    # there are 4 ports we need to allocate
    # 1. server port
    # 2. nccl port
    # 3. dist_init_addr port
    # 4. other ports for dp_attention, which is of size 4 + dp_size
    num_engines_per_node = max(
        1, min(args.num_gpus_per_node, args.rollout_num_gpus) // args.rollout_num_gpus_per_engine
    )
    addr_and_ports = [{} for _ in range(num_engines)]

    # Calculate prefill limit to identify prefill engines
    prefill_limit = 0
    if args.prefill_num_servers is not None:
        num_gpu_per_engine = min(args.rollout_num_gpus_per_engine, args.num_gpus_per_node)
        prefill_limit = args.prefill_num_servers * args.rollout_num_gpus_per_engine // num_gpu_per_engine

    visited_nodes = set()
    for rank, engine in rollout_engines:
        if rank // num_engines_per_node in visited_nodes:
            continue
        visited_nodes.add(rank // num_engines_per_node)
        # TODO: currently when restarting engines, we will set port for all engines on this node starting with this rank.
        # e.g. for 8 gpus, if we are restarting engine on gpu 3, we will set port for engine 3,4,5,6,7 on this node.
        num_engines_on_this_node = num_engines_per_node - (rank % num_engines_per_node)

        def get_addr_and_ports(engine):
            # use small ports to prevent ephemeral port between 32768 and 65536.
            # also, ray uses port 10002-19999, thus we avoid near-10002 to avoid racing condition
            start_port = 15000

            def port(consecutive=1):
                nonlocal start_port
                _, port = ray.get(
                    engine._get_current_node_ip_and_free_port.remote(
                        start_port=start_port,
                        consecutive=consecutive,
                    )
                )
                start_port = port + consecutive
                return port

            def addr():
                addr, _ = ray.get(engine._get_current_node_ip_and_free_port.remote())
                return addr

            return addr, port

        get_addr, get_port = get_addr_and_ports(engine)

        for i in range(num_engines_on_this_node):
            current_rank = rank + i
            addr_and_ports[current_rank]["host"] = get_addr()
            addr_and_ports[current_rank]["port"] = get_port()
            addr_and_ports[current_rank]["nccl_port"] = get_port()

            if args.prefill_num_servers is not None and current_rank < prefill_limit:
                addr_and_ports[current_rank]["disaggregation_bootstrap_port"] = get_port()

        if args.rollout_num_gpus_per_engine > args.num_gpus_per_node:
            num_node_per_engine = args.rollout_num_gpus_per_engine // args.num_gpus_per_node
            if rank % num_node_per_engine == 0:
                # this is the first node in the engine, we need to allocate the dist_init_addr port
                dist_init_addr = f"{get_addr()}:{get_port(30 + args.sglang_dp_size)}"
                for i in range(num_node_per_engine):
                    addr_and_ports[rank + i]["dist_init_addr"] = dist_init_addr
        else:
            for i in range(num_engines_on_this_node):
                addr_and_ports[rank + i]["dist_init_addr"] = f"{get_addr()}:{get_port(30 + args.sglang_dp_size)}"

    for i, _ in rollout_engines:
        for key in ["port", "nccl_port", "dist_init_addr"]:
            assert key in addr_and_ports[i], f"Engine {i} {key} is not set."
        logger.info(f"Ports for engine {i}: {addr_and_ports[i]}")

    return addr_and_ports


def _start_router(args):
    """start sgl router and slime router"""
    if args.sglang_router_ip is not None:
        return

    args.sglang_router_ip = _wrap_ipv6(get_host_info()[1])
    if args.sglang_router_port is None:
        args.sglang_router_port = find_available_port(random.randint(3000, 4000))

    if args.use_slime_router:
        assert args.prefill_num_servers is None, "slime router does not support prefill_num_servers."
        from slime.router.router import run_router

        router_args = args

    else:
        from sglang_router.launch_router import RouterArgs

        from slime.utils.http_utils import run_router

        router_args = RouterArgs.from_cli_args(args, use_router_prefix=True)
        router_args.host = args.sglang_router_ip
        router_args.port = args.sglang_router_port
        router_args.prometheus_port = find_available_port(random.randint(4000, 5000))
        router_args.log_level = "warn"
        router_args.request_timeout_secs = args.sglang_router_request_timeout_secs

        if args.prefill_num_servers is not None:
            router_args.pd_disaggregation = True

        logger.info(f"Launch router with args: {router_args}")

    process = multiprocessing.Process(
        target=run_router,
        args=(router_args,),
    )
    process.daemon = True  # Set the process as a daemon
    process.start()
    # Wait 3 seconds
    time.sleep(3)
    assert process.is_alive()
    logger.info(f"Router launched at {args.sglang_router_ip}:{args.sglang_router_port}")


def _log_eval_rollout_data(rollout_id, args, data, extra_metrics: dict[str, Any] | None = None):
    val = 0
    if args.custom_eval_rollout_log_function_path is not None:
        custom_log_func = load_function(args.custom_eval_rollout_log_function_path)
        if custom_log_func(rollout_id, args, data, extra_metrics):
            return

    log_dict = extra_metrics or {}

    # Helper function to get n_samples for a specific dataset
    def get_n_samples_for_dataset(dataset_key, dataset_index):
        if isinstance(args.n_samples_per_eval_prompt, int):
            return args.n_samples_per_eval_prompt
        elif isinstance(args.n_samples_per_eval_prompt, list):
            if len(args.n_samples_per_eval_prompt) == 1:
                return args.n_samples_per_eval_prompt[0]
            elif dataset_index < len(args.n_samples_per_eval_prompt):
                return args.n_samples_per_eval_prompt[dataset_index]
            else:
                # Fallback to first value if index is out of range
                return args.n_samples_per_eval_prompt[0]
        else:
            return 1  # Default fallback

    for idx, key in enumerate(data.keys()):

        rewards = data[key]["rewards"]
        
        log_dict[f"eval/{key}"] = sum(rewards) / len(rewards)
        if (samples := data[key].get("samples")) is not None:
            log_dict |= dict_add_prefix(compute_metrics_from_samples(args, samples), f"eval/{key}/")
            log_dict[f"eval/{key}-tool_call_count"] = sum([sample.tool_call_count for sample in samples]) / len(samples)
        if "truncated" in data[key]:
            truncated = data[key]["truncated"]
            log_dict[f"eval/{key}-truncated_ratio"] = sum(truncated) / len(truncated)
        # Prefer n_samples_per_prompt from data if available, otherwise calculate from args
        n_samples_for_dataset = data[key].get("n_samples_per_prompt")
        if n_samples_for_dataset is None:
            n_samples_for_dataset = get_n_samples_for_dataset(key, idx)
        
        val += sum(data[key]['scores']) / n_samples_for_dataset
        if "points" in data[key]:
            points = data[key]['points']
            log_dict[f"eval/{key}-point"] = sum(points) / n_samples_for_dataset
            # val += sum(points) / n_samples_for_dataset
        if "points_noxverify" in data[key]:
            points_noxverify = data[key]['points_noxverify']
            log_dict[f"eval/{key}-points_noxverify"] = sum(points_noxverify) / n_samples_for_dataset

    
    if args.log_passrate:
        log_dict |= dict_add_prefix(
            compute_pass_rate(
                flat_rewards=rewards,
                group_size=n_samples_for_dataset,
            ),
            f"eval/{key}-",
        )

    # print(f"eval {rollout_id}: {log_dict}")
    # logger.info(f"eval {rollout_id}: {log_dict}")
    log_with_file(f"eval {rollout_id}: {log_dict}", args=args)

    step = compute_rollout_step(args, rollout_id)
    log_dict["eval/step"] = step
    logging_utils.log(args, log_dict, step_key="eval/step")
    
    return val, log_dict



def _log_rollout_data(rollout_id, args, samples, rollout_extra_metrics, rollout_time, self_refine_metrics=None, exgrpo_metrics=None):
    if args.custom_rollout_log_function_path is not None:
        custom_log_func = load_function(args.custom_rollout_log_function_path)
        if custom_log_func(rollout_id, args, samples, rollout_extra_metrics, rollout_time):
            return

    if args.load_debug_rollout_data:
        return

    log_dict = {**(rollout_extra_metrics or {})}
    log_dict |= dict_add_prefix(compute_metrics_from_samples(args, samples), "rollout/")
    log_dict |= dict_add_prefix(compute_perf_metrics_from_samples(args, samples, rollout_time), "perf/")
    if self_refine_metrics:
        log_dict |= self_refine_metrics
    if exgrpo_metrics:
        log_dict |= exgrpo_metrics
    log_with_file(f"perf {rollout_id}: {log_dict}", args=args)
    step = compute_rollout_step(args, rollout_id)
    log_dict["rollout/step"] = step
    logging_utils.log(args, log_dict, step_key="rollout/step")


def compute_metrics_from_samples(args, samples):
    response_lengths = [sample.effective_response_length for sample in samples]

    log_dict = {}
    log_dict |= dict_add_prefix(compute_statistics(response_lengths), "response_len/")
    log_dict |= _compute_zero_std_metrics(args, samples)
    log_dict |= _compute_reward_cat_metrics(args, samples)
    log_dict |= _compute_spec_metrics(args, samples)
    log_dict |= _compute_prefix_cache_metrics(args, samples)
    log_dict["repetition_frac"] = np.mean([int(has_repetition(s.response)) for s in samples]).item()
    log_dict["truncated_ratio"] = np.mean([int(s.status == Sample.Status.TRUNCATED) for s in samples]).item()
    return log_dict


def compute_perf_metrics_from_samples(args, samples, rollout_time):
    non_generation_time = [sample.non_generation_time for sample in samples]

    log_dict = {}
    log_dict["rollout_time"] = rollout_time
    if max(non_generation_time) > 0:
        log_dict |= dict_add_prefix(compute_statistics(non_generation_time), "non_generation_time/")

    def token_perf(response_lengths, non_generation_time, key=""):
        max_response_length = max(response_lengths)
        if args.rollout_num_gpus:
            log_dict[f"{key}tokens_per_gpu_per_sec"] = sum(response_lengths) / rollout_time / args.rollout_num_gpus
        log_dict[f"longest_{key}sample_tokens_per_sec"] = max_response_length / rollout_time

        if max(non_generation_time) == 0:
            return

        non_generation_time = [
            t for t, length in zip(non_generation_time, response_lengths, strict=True) if length == max_response_length
        ]
        mean_non_generation_time = sum(non_generation_time) / len(non_generation_time)

        log_dict[f"longest_{key}sample_non_generation_time"] = mean_non_generation_time
        log_dict[f"longest_{key}sample_tokens_per_sec_without_non_generation"] = max_response_length / (
            rollout_time - mean_non_generation_time
        )

    token_perf([sample.response_length for sample in samples], non_generation_time, key="")
    token_perf([sample.effective_response_length for sample in samples], non_generation_time, key="effective_")

    return log_dict


def _compute_zero_std_metrics(args, all_samples: list[Sample]):
    # only compute in GRPO-like algorithms where one prompt has multiple responses
    if args.advantage_estimator == "ppo":
        return {}

    def _is_zero_std(samples: list[Sample]):
        rewards = [sample.get_reward_value(args) for sample in samples]
        return len(rewards) == 0 or all(rewards[0] == r for r in rewards)

    all_sample_groups = group_by(all_samples, lambda s: s.group_index)
    interesting_sample_groups = [g for g in all_sample_groups.values() if _is_zero_std(g)]

    interesting_rewards = [str(round(g[0].get_reward_value(args), 1)) for g in interesting_sample_groups]

    return {f"zero_std/count_{reward}": len(items) for reward, items in group_by(interesting_rewards).items()}


def _compute_spec_metrics(args, all_samples: list[Sample]):
    if args.sglang_speculative_algorithm is None:
        return {}
    num_samples = len(all_samples)
    metrics = {}
    metrics["spec_accept_rate"] = sum(sample.spec_info.spec_accept_rate for sample in all_samples) / num_samples
    metrics["spec_accept_length"] = sum(sample.spec_info.spec_accept_length for sample in all_samples) / num_samples
    return metrics


def _compute_prefix_cache_metrics(args, all_samples: list[Sample]):
    if not getattr(args, "sglang_enable_cache_report", False):
        return {}
    num_samples = len(all_samples)
    metrics = {}
    total_cached_tokens = sum(sample.prefix_cache_info.cached_tokens for sample in all_samples)
    total_prompt_tokens = sum(sample.prefix_cache_info.total_prompt_tokens for sample in all_samples)

    metrics["prefix_cache_hit_rate"] = total_cached_tokens / total_prompt_tokens if total_prompt_tokens > 0 else 0.0
    metrics["avg_cached_tokens_per_sample"] = total_cached_tokens / num_samples
    return metrics


def _compute_reward_cat_metrics(args, all_samples: list[Sample]):
    reward_cat_key = args.log_reward_category
    if reward_cat_key is None:
        return {}

    samples_of_reward_cat = group_by(all_samples, lambda s: s.reward[reward_cat_key])

    return {f"error_cat/{reward_cat}": len(s) / len(all_samples) for reward_cat, s in samples_of_reward_cat.items()}
