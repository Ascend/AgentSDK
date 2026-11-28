# Copyright (c) 2026 Huawei Technologies Co.,Ltd.
# Licensed under Mulan PSL v2 (http://license.coscl.org.cn/MulanPSL2).
"""Real trajectory conversion and length-based (not token-ID-based) repadding."""
from collections import defaultdict

from .codec import digest
from .schema import ReplayGroup, ReplaySample, ReplayTensor


def trajectory_groups(trajectories, config, batch_id, weight_version):
    """All validation finishes before any durable batch is submitted."""
    from aura.trainer.rollout.trajectory_validation import validate_trajectory_batch
    validate_trajectory_batch(trajectories, config.group_size, config.batch_groups * config.group_size)
    grouped = defaultdict(list)
    for trajectory in trajectories:
        grouped[str(trajectory["prompt_id"])].append(trajectory)
    groups = []
    for prompt_id, members in grouped.items():
        group_id = digest([config.store_id, config.source_run_id, batch_id, prompt_id])
        samples = []
        for index, t in enumerate(sorted(members, key=lambda t: str(t["idx"]))):
            def tensor(value, dtype=None):
                import torch
                if not isinstance(value, torch.Tensor):
                    value = torch.as_tensor(value, dtype=dtype)
                elif dtype is not None:
                    value = value.to(dtype=dtype)
                return ReplayTensor.from_torch(value.detach().cpu())
            import torch
            response = tensor(t["response_tokens"])
            reward = float(t["trajectory_reward"])
            rewards = torch.zeros(response.shape[0], dtype=torch.float32)
            rewards[-1] = reward
            sample = ReplaySample(
                source_store_id=config.store_id, trajectory_id=digest([group_id, index]),
                prompt_id=prompt_id, group_id=group_id, sample_index=index, group_size=config.group_size,
                source_experiment_id=config.source_experiment_id, source_run_id=config.source_run_id,
                policy_lineage_id=config.policy_lineage_id, weight_version=weight_version, reward=reward,
                payload={"prompts": tensor(t["prompt_tokens"]), "responses": response,
                         "response_mask": tensor(t["response_masks"]),
                         "token_level_rewards": tensor(rewards), "rm_scores": tensor(rewards)},
                old_log_probs=tensor(t["logprobs"], torch.float32),
                model_signature=config.model_signature, tokenizer_signature=config.tokenizer_signature,
                task_signature=config.task_signature, reward_signature=config.reward_signature,
                source_batch_id=batch_id)
            samples.append(sample)
        groups.append(ReplayGroup(tuple(samples)))
    return tuple(groups)


def training_batch(groups, pad_token_id):
    """Rebuild verl inputs, preserving tool masks and original behavior probabilities."""
    import torch
    samples = [s for group in groups for s in group.samples]
    if not samples:
        raise ValueError("Cannot build an empty training batch")
    if len({s.sample_key for s in samples}) != len(samples):
        raise ValueError("Duplicate trajectory in mixed batch")
    count = len(samples)
    p = max(s.payload["prompts"].shape[0] for s in samples)
    r = max(s.payload["responses"].shape[0] for s in samples)
    result = {
        "prompts": torch.full((count, p), pad_token_id, dtype=torch.int64),
        "responses": torch.full((count, r), pad_token_id, dtype=torch.int64),
        "attention_mask": torch.zeros((count, p + r), dtype=torch.int64),
        "response_mask": torch.zeros((count, r), dtype=torch.int64),
        "rm_scores": torch.zeros((count, r), dtype=torch.float32),
        "token_level_rewards": torch.zeros((count, r), dtype=torch.float32),
        "rollout_log_probs": torch.zeros((count, r), dtype=torch.float32),
        "prompt_ids": [], "replay_trajectory_ids": [], "replay_weight_versions": [],
        "replay_source_store_ids": [],
    }
    # Fill the owned CPU tensors through NumPy views. This avoids a temporary
    # Torch tensor and Torch slice assignment for every source field/row.
    # ReplayTensor.to_numpy still copies the immutable source bytes.
    arrays = {name: result[name].numpy() for name in (
        "prompts", "responses", "attention_mask", "response_mask", "rm_scores",
        "token_level_rewards", "rollout_log_probs")}
    for row, sample in enumerate(samples):
        a, b = sample.payload["prompts"].shape[0], sample.payload["responses"].shape[0]
        arrays["prompts"][row, p-a:] = sample.payload["prompts"].to_numpy()
        arrays["responses"][row, :b] = sample.payload["responses"].to_numpy()
        arrays["attention_mask"][row, p-a:p+b] = 1  # Includes genuine eos/pad-valued tokens.
        arrays["response_mask"][row, :b] = sample.payload["response_mask"].to_numpy()
        for name in ("rm_scores", "token_level_rewards"):
            arrays[name][row, :b] = sample.payload[name].to_numpy()
        arrays["rollout_log_probs"][row, :b] = sample.old_log_probs.to_numpy()
        result["prompt_ids"].append(digest([sample.source_store_id, sample.group_id]))
        result["replay_trajectory_ids"].append(sample.trajectory_id)
        result["replay_weight_versions"].append(sample.weight_version)
        result["replay_source_store_ids"].append(sample.source_store_id)
    result["input_ids"] = torch.cat([result["prompts"], result["responses"]], dim=1)
    result["position_ids"] = (result["attention_mask"].cumsum(-1) - 1) * result["attention_mask"]
    return result
