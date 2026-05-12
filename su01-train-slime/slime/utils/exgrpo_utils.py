"""Helper functions for ExGRPO off-policy correction."""

import torch
import logging

logger = logging.getLogger(__name__)


def replace_old_log_probs_for_experience(rollout_data: dict) -> dict:
    """Replace computed log_probs with recorded old_log_probs for experience samples.

    This enables off-policy correction for experience replay.

    Args:
        rollout_data: Dict containing 'log_probs' and sample metadata

    Returns:
        Modified rollout_data with replaced log_probs for experience samples
    """
    # Check if we have experience samples
    if 'samples' not in rollout_data:
        return rollout_data

    samples = rollout_data.get('samples', [])
    log_probs = rollout_data.get('log_probs', [])

    if not samples or not log_probs:
        return rollout_data

    # Replace log_probs for off-policy samples
    replaced_count = 0
    for i, sample_group in enumerate(samples):
        if not sample_group:
            continue

        # Check if this is an experience sample
        first_sample = sample_group[0]
        if (hasattr(first_sample, 'metadata') and
            first_sample.metadata and
            first_sample.metadata.get('is_off_policy', False)):

            # Use recorded old_log_prob
            for j, sample in enumerate(sample_group):
                recorded_log_prob = sample.metadata.get('recorded_old_log_prob')
                if recorded_log_prob is not None and i < len(log_probs):
                    # Replace with recorded value
                    log_probs[i] = recorded_log_prob
                    replaced_count += 1
                    break  # One per group

    if replaced_count > 0:
        logger.debug(f"Replaced log_probs for {replaced_count} experience samples")

    rollout_data['log_probs'] = log_probs
    return rollout_data
