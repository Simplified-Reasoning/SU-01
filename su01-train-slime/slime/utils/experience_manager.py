"""Experience management for ExGRPO algorithm.

This module provides experience pool management with bucketing and weighted sampling.
Ported from verl's ExGRPO implementation.
"""

import random
import numpy as np
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import logging

logger = logging.getLogger(__name__)

CONSTANT_DATA_UID = -90000


class ExperienceBucketManager:
    """Manages experience buckets organized by rollout correctness.

    Buckets are keyed by num_success (number of correct rollouts per prompt).
    Enables targeted sampling from specific correctness levels.
    """

    def __init__(self):
        self.experience_bucket = defaultdict(list)  # num_success -> list of df_uid
        self.uid_to_bucket = {}  # df_uid -> current bucket

    def update_bucket(self, df_uid: int, num_success: int):
        """Update bucket assignment for a dataframe UID."""
        if df_uid == CONSTANT_DATA_UID:
            return

        num_success_item = num_success if isinstance(num_success, int) else num_success.item()

        # Remove from old bucket if changed
        if df_uid in self.uid_to_bucket:
            old_bucket = self.uid_to_bucket[df_uid]
            if old_bucket != num_success_item:
                if df_uid in self.experience_bucket[old_bucket]:
                    self.experience_bucket[old_bucket].remove(df_uid)

        # Add to new bucket
        if df_uid not in self.experience_bucket[num_success_item]:
            self.experience_bucket[num_success_item].append(df_uid)
        self.uid_to_bucket[df_uid] = num_success_item

    def remove_from_bucket(self, df_uid: int):
        """Remove a UID from all bucket structures."""
        if df_uid in self.uid_to_bucket:
            bucket_key = self.uid_to_bucket[df_uid]
            if df_uid in self.experience_bucket[bucket_key]:
                self.experience_bucket[bucket_key].remove(df_uid)
            del self.uid_to_bucket[df_uid]

    def cleanup_consistency(self, experience_pool: Dict):
        """Remove UIDs not present in experience pool."""
        to_remove = []
        for bucket_key, df_uid_list in self.experience_bucket.items():
            for df_uid in df_uid_list:
                if df_uid not in experience_pool:
                    to_remove.append((bucket_key, df_uid))

        for bucket_key, df_uid in to_remove:
            self.experience_bucket[bucket_key].remove(df_uid)
            if df_uid in self.uid_to_bucket:
                del self.uid_to_bucket[df_uid]

        # Clean uid_to_bucket
        to_remove_mapping = [uid for uid in self.uid_to_bucket if uid not in experience_pool]
        for df_uid in to_remove_mapping:
            del self.uid_to_bucket[df_uid]

    def get_bucket_stats(self) -> Dict[str, int]:
        """Get bucket statistics for logging."""
        if not self.experience_bucket:
            return {}
        return {f'bucket/{k}_count': len(v) for k, v in self.experience_bucket.items() if len(v) > 0}

    def print_bucket_stats(self):
        """Print bucket distribution."""
        if self.experience_bucket:
            bucket_info = [f"bucket_{k}:{len(self.experience_bucket[k])}" for k in sorted(self.experience_bucket.keys()) if len(self.experience_bucket[k]) > 0]
            if bucket_info:
                logger.info(f"[Bucket Stats] {', '.join(bucket_info)}")


class WeightedBucketSampler:
    """Samples experiences from buckets with configurable weighting strategies."""

    def __init__(self, buckets: Dict[int, List[Any]], n_rollout: int,
                 weighting_method: str = "linear", beta: float = 0.01):
        """
        Args:
            buckets: bucket_key -> list of data
            n_rollout: number of rollouts per prompt (for weight normalization)
            weighting_method: linear/normal/sqrt/linear_clip
            beta: unused, kept for API compatibility
        """
        self.buckets = buckets.copy()
        self.beta = float(beta)

        # Compute weights based on method
        if weighting_method == "linear":
            self.weights = {k: k / n_rollout for k in buckets.keys()}
        elif weighting_method == "normal":
            self.weights = {k: self._normal_mapping(k, n_rollout) for k in buckets.keys()}
        elif weighting_method == "sqrt":
            self.weights = {k: np.sqrt(k / n_rollout) for k in buckets.keys()}
        elif weighting_method == "linear_clip":
            self.weights = {k: min(k / n_rollout, 0.5) if k > n_rollout / 2 else k / n_rollout
                           for k in buckets.keys()}
        else:
            raise ValueError(f"Invalid weighting method: {weighting_method}")

        # Filter empty buckets
        self.valid_buckets = {k: v for k, v in self.buckets.items() if len(v) > 0}
        self.valid_weights = {k: self.weights.get(k, 0) for k in self.valid_buckets.keys()}

        self.total_weight = sum(self.valid_weights.values())
        if self.total_weight == 0:
            raise ValueError("All buckets empty or zero weight")

        self.probabilities = {k: w / self.total_weight for k, w in self.valid_weights.items()}
        self.total_valid_items = sum(len(v) for v in self.valid_buckets.values())

        logger.info(f"WeightedBucketSampler: {len(self.valid_buckets)} buckets, "
                   f"sizes={[(k, len(v)) for k, v in self.valid_buckets.items()]}, "
                   f"probs={[(k, f'{p:.3f}') for k, p in self.probabilities.items()]}")

    @staticmethod
    def _normal_mapping(x: float, rollout: int) -> float:
        """Gaussian weighting centered at rollout/2."""
        mu = rollout / 2
        sigma = 1
        coeff = 1 / np.sqrt(2 * np.pi * sigma**2)
        exponent = -((x - mu) ** 2) / (2 * sigma**2)
        return coeff * np.exp(exponent)

    def sample(self, n: int, seed: Optional[int] = None) -> Tuple[List[Any], List[float]]:
        """Sample n items with replacement using multinomial distribution.

        Returns:
            (sampled_items, importance_weights)
        """
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        if not self.valid_buckets:
            return [], []

        # Multinomial sampling across buckets
        bucket_names = list(self.valid_buckets.keys())
        bucket_probs = [self.probabilities[name] for name in bucket_names]

        counts = np.random.multinomial(n, bucket_probs)

        sampled_items = []
        importance_weights = []

        for bucket_name, count in zip(bucket_names, counts):
            if count == 0:
                continue

            bucket_data = self.valid_buckets[bucket_name]
            bucket_size = len(bucket_data)
            bucket_prob = self.probabilities[bucket_name]

            # Sample with replacement
            sampled_indices = np.random.choice(bucket_size, size=count, replace=True)

            for idx in sampled_indices:
                item = bucket_data[idx]
                # Importance weight: 1 / (bucket_prob * 1/bucket_size) = bucket_size / bucket_prob
                is_weight = bucket_size / (bucket_prob * self.total_valid_items)
                sampled_items.append(item)
                importance_weights.append(is_weight)

        return sampled_items, importance_weights
