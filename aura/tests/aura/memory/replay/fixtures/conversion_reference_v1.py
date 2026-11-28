"""Frozen pre-tail-fix CPU repadding implementation for bit-exact comparison."""
from aura.memory.replay.codec import digest

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
    for row, sample in enumerate(samples):
        a, b = sample.payload["prompts"].shape[0], sample.payload["responses"].shape[0]
        result["prompts"][row, p-a:] = sample.payload["prompts"].to_torch()
        result["responses"][row, :b] = sample.payload["responses"].to_torch()
        result["attention_mask"][row, p-a:p+b] = 1  # Includes genuine eos/pad-valued tokens.
        result["response_mask"][row, :b] = sample.payload["response_mask"].to_torch()
        for name in ("rm_scores", "token_level_rewards"):
            result[name][row, :b] = sample.payload[name].to_torch()
        result["rollout_log_probs"][row, :b] = sample.old_log_probs.to_torch()
        result["prompt_ids"].append(digest([sample.source_store_id, sample.group_id]))
        result["replay_trajectory_ids"].append(sample.trajectory_id)
        result["replay_weight_versions"].append(sample.weight_version)
        result["replay_source_store_ids"].append(sample.source_store_id)
    result["input_ids"] = torch.cat([result["prompts"], result["responses"]], dim=1)
    result["position_ids"] = (result["attention_mask"].cumsum(-1) - 1) * result["attention_mask"]
    return result
