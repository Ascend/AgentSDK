# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co.,Ltd.
"""Fail-closed checks for trajectory-level GRPO (not stepwise advantages)."""
from collections import Counter, defaultdict


class PromptTooLongError(ValueError):
    """A deterministic input error; retrying generation cannot repair it."""


def validate_group_ids(prompt_ids, n_samples, expected_size=None):
    if not isinstance(n_samples, int) or n_samples < 1:
        raise ValueError("n_samples must be a positive integer")
    if len(prompt_ids) == 0 or (expected_size is not None and len(prompt_ids) != expected_size):
        raise ValueError("GRPO batch has an unexpected number of trajectories")
    if len(prompt_ids) % n_samples:
        raise ValueError("GRPO batch must contain whole groups")
    if any(prompt_id is None or str(prompt_id) == "" for prompt_id in prompt_ids):
        raise ValueError("GRPO prompt_id must not be empty")
    counts = Counter(str(prompt_id) for prompt_id in prompt_ids)
    if any(count != n_samples for count in counts.values()):
        raise ValueError(f"GRPO group sizes must all equal {n_samples}: {dict(counts)}")


def _values(value):
    return value.tolist() if hasattr(value, "tolist") else list(value)


def validate_trajectory_batch(trajectories, n_samples, expected_size=None):
    validate_group_ids([t["prompt_id"] for t in trajectories], n_samples, expected_size)
    groups = defaultdict(list)
    probabilities_present = []
    for trajectory in trajectories:
        groups[str(trajectory["prompt_id"])].append(trajectory)
        prompt = _values(trajectory["prompt_tokens"])
        response = _values(trajectory["response_tokens"])
        mask = _values(trajectory["response_masks"])
        if not prompt or not response or len(response) != len(mask):
            raise ValueError("GRPO requires nonempty real trajectories with aligned response masks")
        if any(value not in (0, 1) for value in mask):
            raise ValueError("GRPO response masks must be binary")
        if trajectory.get("trajectory", {}).get("data_id") == "0" * 33:
            raise ValueError("Dummy trajectories must not enter trajectory-level GRPO")
        log_probs = trajectory.get("logprobs")
        has_probs = log_probs is not None and len(log_probs) > 0
        probabilities_present.append(has_probs)
        if has_probs and len(log_probs) != len(response):
            raise ValueError("rollout log-probs must align with response tokens")
    if any(probabilities_present) and not all(probabilities_present):
        raise ValueError("Mixed missing rollout log-probs must not silently disable batch correction")
    for prompt_id, group in groups.items():
        if len({str(t["idx"]) for t in group}) != n_samples:
            raise ValueError(f"Duplicate trajectory/sample ids in GRPO group {prompt_id}")
        reference = _values(group[0]["prompt_tokens"])
        if any(_values(t["prompt_tokens"]) != reference for t in group[1:]):
            raise ValueError(f"Different initial prompts share GRPO uid {prompt_id}")
