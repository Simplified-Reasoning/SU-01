"""ExGRPO DataSource that extends SelfRefineDataSource with experience replay."""

import copy
import hashlib
import logging
from collections import defaultdict
from typing import Optional

import numpy as np

from slime.rollout.data_source import RolloutDataSource, RolloutDataSourceWithBuffer, SelfRefineDataSource
from slime.utils.types import Sample
from slime.utils.experience_manager import ExperienceBucketManager, WeightedBucketSampler

logger = logging.getLogger(__name__)


def _debug_print(message: str):
    print(f"[ExGRPO] {message}", flush=True)


def _reward_scalar(s, args):
    if s.reward is None:
        return 0.0
    r = s.get_reward_value(args) if hasattr(s, 'get_reward_value') else s.reward
    if isinstance(r, dict):
        r = r.get('score', 0)
    return r if r is not None else 0.0


class ExGRPODataSource(SelfRefineDataSource):
    """DataSource with experience replay for ExGRPO.

    Extends SelfRefineDataSource to maintain compatibility with self-refine mechanism.
    Adds experience pool management and mixing logic.
    """

    def __init__(self, args):
        super().__init__(args)

        # ExGRPO parameters
        self.experience_ratio = getattr(args, 'experience_ratio', 0.0)
        self.experience_min_reward = getattr(args, 'experience_min_reward', 0.0)
        self.experience_weighting_method = getattr(args, 'experience_weighting_method', 'linear')
        self.experience_metric = getattr(args, 'experience_metric', 'ent')  # ent or ppl
        self.experience_select_mode = getattr(args, 'experience_select_mode', 'argmin')  # argmin for ent
        self.exgrpo_replay_k = getattr(args, 'exgrpo_replay_k', 1)
        self.experience_in_threshold = getattr(args, 'experience_in_threshold', None)
        if self.experience_in_threshold is None:
            self.experience_in_threshold = args.n_samples_per_prompt
        if self.experience_in_threshold <= 1:
            raise ValueError(
                "experience_in_threshold must be greater than 1 so that 0 < num_success < threshold "
                f"has a non-empty valid range, but got {self.experience_in_threshold}"
            )
        self.exgrpo_retire_threshold = getattr(args, 'exgrpo_retire_threshold', None)
        if self.exgrpo_retire_threshold is None:
            self.exgrpo_retire_threshold = args.n_samples_per_prompt
        if not 1 <= self.exgrpo_retire_threshold <= args.n_samples_per_prompt:
            raise ValueError(
                "exgrpo_retire_threshold must be in [1, n_samples_per_prompt], but got "
                f"{self.exgrpo_retire_threshold} with n_samples_per_prompt={args.n_samples_per_prompt}"
            )

        # Experience storage: query_id -> list[dict] where each element is one successful trajectory.
        self.experience_pool = {}
        self.experience_signatures = {}  # query_id -> set[str], used for per-query trajectory dedup
        self.bucket_manager = ExperienceBucketManager()

        # Retired set: query_ids that are too easy (all rollouts correct)
        self.retired_set = set()

        logger.info(
            f"ExGRPO enabled: ratio={self.experience_ratio}, "
            f"metric={self.experience_metric}, select={self.experience_select_mode}, "
            f"retire_threshold={self.exgrpo_retire_threshold}"
        )

        # Metrics from the latest get_samples call
        self._last_batch_metrics = {}
        self._rollout_batch_metrics = {}

        # Entropy pre-computed by Megatron actor with current model,
        # cleared and recomputed at the start of each rollout
        self._precomputed_entropies = {}  # (query_id, exp_idx) -> entropy
        self._last_candidate_count = 0
        self._last_valid_query_count = 0
        self._last_scored_candidate_count = 0
        self._pending_replay_groups = {}

    def _prune_empty_experience_entries(self):
        """Remove query ids whose experience list is empty."""
        empty_query_ids = [qid for qid, experiences in self.experience_pool.items() if len(experiences) == 0]
        for query_id in empty_query_ids:
            del self.experience_pool[query_id]
            self.experience_signatures.pop(query_id, None)
            self.bucket_manager.remove_from_bucket(query_id)
        if empty_query_ids:
            _debug_print(
                "pruned empty experience entries: "
                f"count={len(empty_query_ids)}, sample_query_ids={empty_query_ids[:8]}"
            )
            logger.info(
                "ExGRPO pruned empty experience entries: "
                f"count={len(empty_query_ids)}, sample_query_ids={empty_query_ids[:8]}"
            )

    @staticmethod
    def _trajectory_signature(sample: Sample) -> str:
        """Build a stable fingerprint for one trajectory within the same query."""
        response = getattr(sample, "response", "") or ""
        if response:
            payload = response.encode("utf-8", errors="ignore")
        else:
            tokens = getattr(sample, "tokens", None) or []
            response_length = getattr(sample, "response_length", 0) or 0
            if response_length > 0 and len(tokens) >= response_length:
                response_tokens = tokens[-response_length:]
            else:
                response_tokens = tokens
            payload = ",".join(str(token) for token in response_tokens).encode("utf-8")
        return hashlib.sha1(payload).hexdigest()

    def _build_signature_cache_for_query(self, query_id: str):
        """Populate per-query signature cache from existing stored experiences."""
        signatures = set()
        for exp in self.experience_pool.get(query_id, []):
            sample = exp.get("sample")
            if sample is None:
                continue
            signatures.add(self._trajectory_signature(sample))
        self.experience_signatures[query_id] = signatures

    def _rebuild_signature_cache(self):
        """Rebuild signature cache and drop duplicated trajectories within each query."""
        removed_duplicates = 0
        rebuilt_cache = {}
        for query_id, experiences in list(self.experience_pool.items()):
            deduped_experiences = []
            signatures = set()
            for exp in experiences:
                sample = exp.get("sample")
                if sample is None:
                    continue
                signature = self._trajectory_signature(sample)
                if signature in signatures:
                    removed_duplicates += 1
                    continue
                signatures.add(signature)
                deduped_experiences.append(exp)
            self.experience_pool[query_id] = deduped_experiences
            rebuilt_cache[query_id] = signatures
        self.experience_signatures = rebuilt_cache
        self._prune_empty_experience_entries()
        if removed_duplicates > 0:
            _debug_print(f"rebuilt experience signature cache and removed duplicated trajectories={removed_duplicates}")
            logger.info(
                "ExGRPO rebuilt experience signature cache and removed "
                f"{removed_duplicates} duplicated trajectories"
            )

    def collect_experience_candidates(self) -> tuple[list[Sample], list[tuple[str, int]]]:
        """Collect one stored successful trajectory per candidate for scoring."""
        self._prune_empty_experience_entries()
        self._precomputed_entropies = {}
        self._last_candidate_count = 0
        self._last_valid_query_count = 0
        self._last_scored_candidate_count = 0
        self._pending_replay_groups = {}
        self._rollout_batch_metrics = {
            "exgrpo/batch_fresh_count": 0,
            "exgrpo/batch_experience_count": 0,
            "exgrpo/batch_self_refine_count": 0,
            "exgrpo/displaced_count": 0,
            "exgrpo/injected_trajectory_count": 0,
        }

        if self.experience_ratio == 0.0 or len(self.experience_pool) == 0:
            return [], []

        valid_query_ids = [
            qid for qid, experiences in self.experience_pool.items()
            if qid not in self.retired_set and len(experiences) > 0
        ]
        self._last_valid_query_count = len(valid_query_ids)
        if not valid_query_ids:
            _debug_print(
                "candidate collection skipped: "
                f"pool_size={len(self.experience_pool)}, retired_size={len(self.retired_set)}, valid_query_ids=0"
            )
            logger.info(
                "ExGRPO candidate collection skipped: "
                f"pool_size={len(self.experience_pool)}, retired_size={len(self.retired_set)}, valid_query_ids=0"
            )
            return [], []

        all_candidate_samples = []
        candidate_indices = []

        for query_id in valid_query_ids:
            for exp_idx, exp in enumerate(self.experience_pool[query_id]):
                all_candidate_samples.append(exp['sample'])
                candidate_indices.append((query_id, exp_idx))

        self._last_candidate_count = len(all_candidate_samples)
        _debug_print(
            "candidate collection: "
            f"pool_size={len(self.experience_pool)}, retired_size={len(self.retired_set)}, "
            f"valid_query_ids={len(valid_query_ids)}, candidate_count={len(all_candidate_samples)}"
        )
        logger.info(
            "ExGRPO candidate collection: "
            f"pool_size={len(self.experience_pool)}, retired_size={len(self.retired_set)}, "
            f"valid_query_ids={len(valid_query_ids)}, candidate_count={len(all_candidate_samples)}"
        )
        return all_candidate_samples, candidate_indices

    def set_precomputed_entropies(self, candidate_indices: list[tuple[str, int]], entropies: list[float]):
        """Store externally computed entropy values for candidate experiences."""
        if len(candidate_indices) != len(entropies):
            raise RuntimeError(
                "ExGRPO precomputed entropy count mismatch: "
                f"expected {len(candidate_indices)}, got {len(entropies)}."
            )
        for i, (query_id, exp_idx) in enumerate(candidate_indices):
            self._precomputed_entropies[(query_id, exp_idx)] = entropies[i]
        self._last_scored_candidate_count = len(entropies)
        _debug_print(f"stored precomputed entropy scores: count={len(entropies)}")
        logger.info(f"ExGRPO stored {len(entropies)} precomputed entropy scores")


    @staticmethod
    def _is_self_refine_group(group: list[Sample]) -> bool:
        return bool(
            group
            and hasattr(group[0], 'metadata')
            and group[0].metadata
            and group[0].metadata.get('_is_self_refine', False)
        )

    @staticmethod
    def _is_off_policy_sample(sample: Sample) -> bool:
        return bool(sample.metadata and sample.metadata.get('is_off_policy', False))

    @staticmethod
    def _group_query_id(group: list[Sample]) -> Optional[str]:
        if not group:
            return None
        return group[0].query_id

    def should_bypass_replay_filtering(self, sample: Sample) -> bool:
        """Keep active replay candidates available even if they look easy."""
        query_id = sample.query_id
        return (
            query_id is not None
            and query_id not in self.retired_set
            and query_id in self.experience_pool
            and len(self.experience_pool[query_id]) > 0
        )

    def should_track_query_accuracy(self, sample: Sample) -> bool:
        """Replay-query groups should not push a query into replay filtering."""
        metadata = sample.metadata or {}
        return not metadata.get("_exgrpo_replay_query", False)

    @staticmethod
    def _original_query_id_for_group(group: list[Sample]) -> Optional[str]:
        if not group:
            return None
        first_sample = group[0]
        if first_sample.metadata and first_sample.metadata.get('_is_self_refine', False):
            return first_sample.metadata.get('_original_query_id', first_sample.query_id)
        return first_sample.query_id

    def _extract_valid_old_log_prob(self, sample: Sample, query_id: str):
        rollout_log_probs = getattr(sample, 'rollout_log_probs', None)
        if rollout_log_probs is None:
            rollout_log_probs = getattr(sample, 'logprobs', None)

        if rollout_log_probs is None:
            return None

        import torch

        log_prob = torch.as_tensor(rollout_log_probs, dtype=torch.float32)
        response_len = getattr(sample, 'response_length', 0) or log_prob.numel()
        valid_log_prob = log_prob[:response_len]
        pad_part = log_prob[response_len:]

        if pad_part.numel() > 0 and not torch.allclose(pad_part, pad_part[0].expand_as(pad_part)):
            raise ValueError(
                f"ExGRPO expects padded old_log_probs tail to be constant for query_id={query_id}"
            )

        return valid_log_prob

    def _make_fresh_rollout_sample(self, template_sample: Sample, *, group_index: int, sample_index: int) -> Sample:
        sample = copy.deepcopy(template_sample)
        sample.group_index = group_index
        sample.index = sample_index
        sample.tokens = []
        sample.logprobs = []
        sample.response = ""
        sample.response_length = 0
        sample.reward = None
        sample.loss_mask = None
        sample.weight_versions = []
        sample.rollout_log_probs = None
        sample.rollout_routed_experts = None
        sample.remove_sample = False
        sample.teacher_log_probs = None
        sample.multimodal_train_inputs = None
        sample.status = Sample.Status.PENDING
        sample.spec_info = Sample.SpecInfo()
        sample.prefix_cache_info = Sample.PrefixCacheInfo()
        sample.metadata = copy.deepcopy(sample.metadata) if sample.metadata is not None else {}
        sample.metadata.pop("is_off_policy", None)
        sample.metadata.pop("recorded_old_log_prob", None)
        sample.metadata.pop("_exgrpo_injected", None)
        sample.metadata.pop("_exgrpo_replay_query", None)
        sample.metadata.pop("_exgrpo_replay_query_id", None)
        sample.metadata.pop("_exgrpo_replay_k", None)
        sample.metadata["_exgrpo_replay_query"] = True
        return sample

    def _sample_query_ids_without_replacement(
        self,
        num_samples: int,
        excluded_query_ids: Optional[set[str]] = None,
    ) -> list[str]:
        excluded_query_ids = excluded_query_ids or set()
        if num_samples <= 0:
            return []

        valid_buckets = {}
        for bucket_key, query_ids in self.bucket_manager.experience_bucket.items():
            filtered_query_ids = [
                query_id
                for query_id in query_ids
                if query_id in self.experience_pool
                and query_id not in self.retired_set
                and query_id not in excluded_query_ids
                and len(self.experience_pool[query_id]) > 0
            ]
            if filtered_query_ids:
                valid_buckets[bucket_key] = filtered_query_ids

        if not valid_buckets:
            return []

        sampler = WeightedBucketSampler(
            buckets=valid_buckets,
            n_rollout=self.args.n_samples_per_prompt,
            weighting_method=self.experience_weighting_method,
        )

        local_buckets = {bucket_key: list(items) for bucket_key, items in sampler.valid_buckets.items()}
        selected_query_ids = []
        while len(selected_query_ids) < num_samples:
            available_bucket_keys = [bucket_key for bucket_key, items in local_buckets.items() if items]
            if not available_bucket_keys:
                break

            bucket_weights = np.array(
                [sampler.valid_weights[bucket_key] for bucket_key in available_bucket_keys],
                dtype=np.float64,
            )
            bucket_weights /= bucket_weights.sum()
            bucket_key = int(np.random.choice(available_bucket_keys, p=bucket_weights))

            bucket_items = local_buckets[bucket_key]
            item_index = int(np.random.randint(len(bucket_items)))
            selected_query_ids.append(bucket_items.pop(item_index))

        return selected_query_ids

    def _select_replay_trajectory_indices(self, query_id: str, replay_k: int) -> tuple[list[int], list[float]]:
        experiences = self.experience_pool.get(query_id, [])
        scored_entries = []
        missing_entropy_indices = []
        for exp_idx, _exp in enumerate(experiences):
            entropy_key = (query_id, exp_idx)
            if entropy_key not in self._precomputed_entropies:
                missing_entropy_indices.append(exp_idx)
                continue
            scored_entries.append((self._precomputed_entropies[entropy_key], exp_idx))

        if not scored_entries:
            raise RuntimeError(
                "ExGRPO failed to select replay trajectories for query_id="
                f"{query_id}. total_experiences={len(experiences)}, "
                f"missing_entropy_indices={missing_entropy_indices}, "
                f"precomputed_entropy_count={len(self._precomputed_entropies)}"
            )

        reverse = self.experience_select_mode == 'argmax'
        scored_entries.sort(key=lambda item: item[0], reverse=reverse)
        replay_k = min(replay_k, len(scored_entries), max(self.args.n_samples_per_prompt - 1, 0))
        selected_entries = scored_entries[:replay_k]
        return [exp_idx for _entropy, exp_idx in selected_entries], [entropy for entropy, _exp_idx in selected_entries]

    def _build_replay_query_group(self, query_id: str, selected_exp_indices: list[int]) -> list[Sample]:
        if not selected_exp_indices:
            return []

        group_index = self.sample_group_index
        self.sample_group_index += 1

        template_sample = self.experience_pool[query_id][selected_exp_indices[0]]['sample']
        group = []
        for _ in range(self.args.n_samples_per_prompt):
            sample = self._make_fresh_rollout_sample(
                template_sample,
                group_index=group_index,
                sample_index=self.sample_index,
            )
            sample.metadata["_exgrpo_replay_query_id"] = query_id
            sample.metadata["_exgrpo_replay_k"] = len(selected_exp_indices)
            self.sample_index += 1
            group.append(sample)

        self._pending_replay_groups[group_index] = {
            "query_id": query_id,
            "selected_exp_indices": list(selected_exp_indices),
        }
        return group

    def _pop_allowed_buffer_groups(self, num_needed: int, blocked_query_ids: set[str]) -> list[list[Sample]]:
        if num_needed <= 0 or not self.buffer:
            return []

        kept_buffer = []
        selected_groups = []
        for group in self.buffer:
            query_id = self._group_query_id(group)
            if len(selected_groups) < num_needed and (query_id is None or query_id not in blocked_query_ids):
                selected_groups.append(group)
            else:
                kept_buffer.append(group)
        self.buffer = kept_buffer
        return selected_groups

    def _fill_additional_fresh_groups(self, num_needed: int, blocked_query_ids: set[str]) -> list[list[Sample]]:
        if num_needed <= 0:
            return []

        local_blocked_query_ids = set(blocked_query_ids)
        extra_groups = self._pop_allowed_buffer_groups(num_needed, local_blocked_query_ids)
        for group in extra_groups:
            query_id = self._group_query_id(group)
            if query_id is not None:
                local_blocked_query_ids.add(query_id)

        if len(extra_groups) >= num_needed or self.dataset is None:
            return extra_groups

        # Continue normal dataset iteration to fill the remaining fresh groups.
        # We allow scanning at most one dataset epoch worth of query groups here;
        # if that still cannot satisfy the request, the caller should fail loudly.
        scanned_dataset_groups = 0
        max_dataset_scan = len(self.dataset)
        while len(extra_groups) < num_needed and scanned_dataset_groups < max_dataset_scan:
            fetched_groups = RolloutDataSource.get_samples(self, 1)
            if not fetched_groups:
                break

            group = fetched_groups[0]
            scanned_dataset_groups += 1
            query_id = self._group_query_id(group)
            if query_id is not None and query_id in local_blocked_query_ids:
                self.buffer.append(group)
                continue

            extra_groups.append(group)
            if query_id is not None:
                local_blocked_query_ids.add(query_id)

        return extra_groups

    def get_samples(self, num_samples: int) -> list[list[Sample]]:
        """Get samples with three-way mixing: self-refine + replay-query + fresh query."""
        if self.experience_ratio == 0.0 or len(self.experience_pool) == 0:
            # No experience, use parent (handles self-refine + fresh)
            return super().get_samples(num_samples)

        parent_samples = super().get_samples(num_samples)
        self_refine_samples = [group for group in parent_samples if self._is_self_refine_group(group)]
        all_fresh = [group for group in parent_samples if not self._is_self_refine_group(group)]

        num_remaining = num_samples - len(self_refine_samples)
        desired_num_experience = int(num_remaining * self.experience_ratio)

        excluded_query_ids = {
            self._original_query_id_for_group(group)
            for group in self_refine_samples
            if self._original_query_id_for_group(group) is not None
        }

        replay_query_ids = self._sample_query_ids_without_replacement(desired_num_experience, excluded_query_ids)
        experience_samples = []
        selected_entropies = []
        blocked_query_ids = set(replay_query_ids) | excluded_query_ids
        for query_id in replay_query_ids:
            if len(self._precomputed_entropies) == 0:
                raise RuntimeError(
                    "ExGRPO replay query sampling was requested, but no precomputed entropies are available. "
                    f"experience_pool={len(self.experience_pool)}, requested_queries={desired_num_experience}, "
                    f"last_valid_query_count={self._last_valid_query_count}, "
                    f"last_candidate_count={self._last_candidate_count}, "
                    f"last_scored_candidate_count={self._last_scored_candidate_count}"
                )

            selected_indices, entropies = self._select_replay_trajectory_indices(query_id, self.exgrpo_replay_k)
            if not selected_indices:
                continue
            experience_samples.append(self._build_replay_query_group(query_id, selected_indices))
            selected_entropies.extend(entropies)

        actual_num_experience = len(experience_samples)
        num_fresh = max(0, num_remaining - actual_num_experience)
        fresh_samples = []
        displaced = []
        for group in all_fresh:
            query_id = self._group_query_id(group)
            if query_id is not None and query_id in blocked_query_ids:
                displaced.append(group)
                continue
            if len(fresh_samples) < num_fresh:
                fresh_samples.append(group)
            else:
                displaced.append(group)

        if len(fresh_samples) < num_fresh:
            blocked_for_top_up = blocked_query_ids | {
                self._group_query_id(group) for group in fresh_samples if self._group_query_id(group) is not None
            }
            fresh_samples.extend(
                self._fill_additional_fresh_groups(num_fresh - len(fresh_samples), blocked_for_top_up)
            )

        if len(fresh_samples) < num_fresh:
            raise RuntimeError(
                "ExGRPO failed to assemble enough fresh query groups after replay selection: "
                f"needed {num_fresh}, got {len(fresh_samples)}, "
                f"desired_replay_queries={desired_num_experience}, actual_replay_queries={actual_num_experience}, "
                f"blocked_query_count={len(blocked_for_top_up)}, buffer_size={len(self.buffer)}"
            )

        for group in displaced:
            self.buffer.append(group)

        mixed = self_refine_samples + experience_samples + fresh_samples

        # Record batch metrics
        self._last_batch_metrics = {
            "exgrpo/batch_fresh_count": len(fresh_samples),
            "exgrpo/batch_experience_count": len(experience_samples),
            "exgrpo/batch_self_refine_count": len(self_refine_samples),
            "exgrpo/displaced_count": len(displaced),
        }
        if selected_entropies:
            self._last_batch_metrics["exgrpo/selected_entropy_mean"] = float(np.mean(selected_entropies))
        for key, value in self._last_batch_metrics.items():
            self._rollout_batch_metrics[key] = self._rollout_batch_metrics.get(key, 0) + value

        logger.debug(f"Batch: {len(self_refine_samples)} self-refine + "
                    f"{len(experience_samples)} experience + {len(fresh_samples)} fresh")

        return mixed

    def inject_replay_trajectories(self, samples: list[Sample]) -> list[Sample]:
        """Inject selected replay trajectories into replay query groups after fresh rollout."""
        if not self._pending_replay_groups:
            return samples

        grouped_samples = {}
        for sample in samples:
            grouped_samples.setdefault(sample.group_index, []).append(sample)

        injected_samples = []
        injected_count = 0
        for group_index, group in grouped_samples.items():
            group = sorted(group, key=lambda sample: sample.index if sample.index is not None else -1)
            replay_plan = self._pending_replay_groups.get(group_index)
            if replay_plan is None:
                injected_samples.extend(group)
                continue

            query_id = replay_plan["query_id"]
            selected_exp_indices = replay_plan["selected_exp_indices"]
            replace_count = min(len(selected_exp_indices), len(group))
            replacement_positions = sorted(
                range(len(group)),
                key=lambda position: (
                    _reward_scalar(group[position], self.args),
                    group[position].index if group[position].index is not None else position,
                ),
            )[:replace_count]

            for position, exp_idx in zip(replacement_positions, selected_exp_indices, strict=False):
                if query_id not in self.experience_pool or exp_idx >= len(self.experience_pool[query_id]):
                    raise RuntimeError(
                        "ExGRPO replay injection plan referenced a missing stored trajectory: "
                        f"query_id={query_id}, exp_idx={exp_idx}"
                    )

                stored_trajectory = self.experience_pool[query_id][exp_idx]
                slot_sample = group[position]
                replay_sample = copy.deepcopy(stored_trajectory["sample"])
                replay_sample.group_index = slot_sample.group_index
                replay_sample.index = slot_sample.index
                replay_sample.metadata = copy.deepcopy(replay_sample.metadata) if replay_sample.metadata else {}
                replay_sample.metadata["is_off_policy"] = True
                replay_sample.metadata["recorded_old_log_prob"] = stored_trajectory["old_log_prob"]
                replay_sample.metadata["_exgrpo_injected"] = True
                replay_sample.metadata["_exgrpo_replay_query"] = True
                replay_sample.metadata["_exgrpo_replay_query_id"] = query_id
                group[position] = replay_sample
                injected_count += 1

            injected_samples.extend(group)

        self._pending_replay_groups = {}
        self._last_batch_metrics["exgrpo/injected_trajectory_count"] = injected_count
        self._rollout_batch_metrics["exgrpo/injected_trajectory_count"] = injected_count
        return sorted(injected_samples, key=lambda sample: sample.index if sample.index is not None else -1)

    def add_samples(self, samples: list[list[Sample]]):
        """Handle aborted samples - parent puts them back to buffer."""
        super().add_samples(samples)

    def add_experience_samples(self, samples: list[list[Sample]]):
        """Store successful rollout samples into experience pool."""
        ingested_rewards = []

        for sample_group in samples:
            if not sample_group:
                continue

            on_policy_samples = [
                sample for sample in sample_group
                if not self._is_off_policy_sample(sample)
            ]
            if not on_policy_samples:
                continue

            # Skip groups with incomplete on-policy samples
            if any(s.reward is None for s in on_policy_samples):
                continue

            # Skip self-refine samples and replayed off-policy samples.
            # ExGRPO should only ingest fresh rollout results into the experience pool.
            first_sample = on_policy_samples[0]
            if (hasattr(first_sample, 'metadata') and
                first_sample.metadata and
                first_sample.metadata.get('_is_self_refine', False)):
                continue

            query_id = first_sample.query_id
            if query_id is None:
                continue
            if query_id in self.retired_set:
                continue

            successful_samples = [
                sample for sample in on_policy_samples
                if _reward_scalar(sample, self.args) >= self.experience_min_reward
                and _reward_scalar(sample, self.args) > 0
            ]
            num_success = len(successful_samples)

            # Retire once a fresh group is easy enough by the configured threshold.
            # Replay-injected trajectories are guaranteed successful by construction.
            if num_success > self.exgrpo_retire_threshold:
                self.retired_set.add(query_id)
                if query_id in self.experience_pool:
                    del self.experience_pool[query_id]
                    self.experience_signatures.pop(query_id, None)
                    self.bucket_manager.remove_from_bucket(query_id)
                continue

            # Store if success count falls into configured admission range.
            # Default keeps the previous behavior: 0 < num_success < n_samples_per_prompt.
            if 0 < num_success < self.experience_in_threshold:
                if query_id not in self.experience_pool:
                    self.experience_pool[query_id] = []
                if query_id not in self.experience_signatures:
                    self._build_signature_cache_for_query(query_id)

                for sample in successful_samples:
                    signature = self._trajectory_signature(sample)
                    if signature in self.experience_signatures[query_id]:
                        continue
                    self.experience_pool[query_id].append({
                        'sample': copy.deepcopy(sample),
                        'old_log_prob': self._extract_valid_old_log_prob(sample, query_id),
                    })
                    self.experience_signatures[query_id].add(signature)
                    ingested_rewards.append(_reward_scalar(sample, self.args))

                self.bucket_manager.update_bucket(query_id, num_success)
            elif query_id in self.experience_pool:
                self.bucket_manager.update_bucket(query_id, num_success)

        # Record ingested reward stats
        if ingested_rewards:
            self._last_batch_metrics["exgrpo/ingested_reward_mean"] = float(np.mean(ingested_rewards))
        else:
            self._last_batch_metrics["exgrpo/ingested_reward_mean"] = 0.0

        self.bucket_manager.cleanup_consistency(self.experience_pool)
        logger.debug(f"Experience pool: {len(self.experience_pool)}, retired: {len(self.retired_set)}")
        self.bucket_manager.print_bucket_stats()

    def get_exgrpo_metrics(self) -> dict:
        """Collect ExGRPO metrics for logging."""
        self._prune_empty_experience_entries()
        metrics = {}

        # Pool & retired set
        metrics["exgrpo/pool_size"] = len(self.experience_pool)
        metrics["exgrpo/retired_size"] = len(self.retired_set)
        metrics["exgrpo/total_experiences"] = sum(
            len(exps) for exps in self.experience_pool.values()
        )

        # Bucket distribution
        for key, count in self.bucket_manager.get_bucket_stats().items():
            metrics[key] = count

        # Batch composition (from latest get_samples)
        metrics.update(self._rollout_batch_metrics)

        # Selection/ingestion metrics from the most recent rollout pass
        for key, value in self._last_batch_metrics.items():
            if (
                key.startswith("exgrpo/selected_")
                or key.startswith("exgrpo/ingested_")
                or key.startswith("exgrpo/injected_")
            ):
                metrics[key] = value

        return metrics

    def save(self, rollout_id):
        """Save state including experience pool and retired set."""
        super().save(rollout_id)

        import torch
        import os
        state_dict = {
            "experience_pool": self.experience_pool,
            "experience_signatures": self.experience_signatures,
            "retired_set": self.retired_set,
            "bucket_manager_buckets": dict(self.bucket_manager.experience_bucket),
            "bucket_manager_uid_to_bucket": self.bucket_manager.uid_to_bucket,
        }
        path = os.path.join(self.args.save, f"rollout/exgrpo_state_{rollout_id}.pt")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(state_dict, path)
        logger.info(f"Saved ExGRPO: pool={len(self.experience_pool)}, retired={len(self.retired_set)}")

    def load(self, rollout_id=None):
        """Load state including experience pool and retired set."""
        super().load(rollout_id)

        import torch
        import os
        if self.args.load is None:
            return

        path = os.path.join(self.args.load, f"rollout/exgrpo_state_{rollout_id}.pt")
        if not os.path.exists(path):
            logger.info(f"ExGRPO checkpoint {path} does not exist")
            return

        state_dict = torch.load(path)
        self.experience_pool = state_dict.get("experience_pool", {})
        loaded_signatures = state_dict.get("experience_signatures")
        if loaded_signatures is None:
            self.experience_signatures = {}
            self._rebuild_signature_cache()
        else:
            self.experience_signatures = {qid: set(signatures) for qid, signatures in loaded_signatures.items()}
            self._rebuild_signature_cache()
        self.retired_set = state_dict.get("retired_set", set())
        self.bucket_manager.experience_bucket = defaultdict(
            list,
            state_dict.get("bucket_manager_buckets", {}),
        )
        self.bucket_manager.uid_to_bucket = state_dict.get("bucket_manager_uid_to_bucket", {})
        self._prune_empty_experience_entries()

        logger.info(f"Loaded ExGRPO: pool={len(self.experience_pool)}, retired={len(self.retired_set)}")
        self.bucket_manager.print_bucket_stats()
