# Copyright (c) 2026 Huawei Technologies Co.,Ltd.
# Licensed under Mulan PSL v2 (http://license.coscl.org.cn/MulanPSL2).
"""Stable old-policy/behavior IS for the pinned verl native vanilla PPO loss."""
import math


def validate_loss_config(config):
    actor = config.actor_rollout_ref.actor
    if actor.get("strategy", "fsdp") not in ("fsdp", "fsdp2"):
        raise ValueError("Replay P3 loss is validated for FSDP/FSDP2 only")
    if config.trainer.get("use_legacy_worker_impl", "disable") != "disable":
        raise ValueError("Replay P3 requires the new verl engine worker")
    if config.trainer.get("critic_warmup", 0) != 0:
        raise ValueError("Replay cannot commit batches during critic-only warmup")
    if (actor.get("policy_loss") or {}).get("loss_mode", "vanilla") != "vanilla":
        raise ValueError("Replay P3 supports native vanilla PPO loss only")
    correction = config.algorithm.get("rollout_correction") or {}
    if (correction.get("bypass_mode", False) or correction.get("rollout_is") is not None
            or correction.get("rollout_rs") is not None or correction.get("rollout_is_batch_normalize", False)):
        raise ValueError("Replay P3 owns behavior IS; disable bypass, other IS/RS and weight normalization")


def validate_native_loss():
    import inspect
    from verl.trainer.ppo.core_algos import get_policy_loss_fn
    native = get_policy_loss_fn("vanilla")
    if (native.__module__ != "verl.trainer.ppo.core_algos"
            or native.__name__ != "compute_policy_loss_vanilla"
            or "rollout_is_weights" not in inspect.signature(native).parameters):
        raise RuntimeError("Unsupported verl native vanilla loss; expected rollout_is_weights consumption")


def apply_behavior_correction(batch):
    """Detached rho=clip(exp(log pi_old-log pi_behavior),.5,2), once per batch.

    PPO's evolving current/old ratio is left to native verl. This is NOT a
    sampling-probability/PER beta weight, and immutable behavior fields stay intact.
    """
    import torch
    data = batch.batch
    if "rollout_is_weights" in data:
        raise ValueError("Behavior correction already exists; refusing double IS")
    required = ("old_log_probs", "rollout_log_probs", "response_mask", "advantages")
    if any(key not in data for key in required):
        raise ValueError("Missing actual old/behavior log-probs, mask or native advantages")
    old, behavior, mask, advantages = (data[key] for key in required)
    if old.ndim != 2 or any(value.shape != old.shape for value in (behavior, mask, advantages)):
        raise ValueError("Behavior correction requires matching 2D response tensors")
    if not torch.all((mask == 0) | (mask == 1)):
        raise ValueError("Behavior correction requires binary response masks")
    if any(not value.is_floating_point() or not torch.isfinite(value).all()
           for value in (old, behavior, advantages)):
        raise ValueError("Nonfinite/nonfloating native policy/advantage tensors")
    active = mask.bool()
    # Float64 subtraction also handles finite extreme float32 log probabilities.
    delta = old.detach().to(torch.float64) - behavior.detach().to(torch.float64)
    limit = torch.finfo(torch.float64).max
    delta = torch.nan_to_num(delta, posinf=limit, neginf=-limit)
    lo, hi = math.log(0.5), math.log(2.0)
    weights = torch.exp(torch.clamp(delta, lo, hi)).to(torch.float32)
    weights = torch.where(active, weights, torch.ones_like(weights)).detach()
    data["rollout_is_weights"] = weights
    selected = weights[active]
    if selected.numel() == 0:
        # No token contributes to loss; all stored IS weights are neutral ones.
        return {"replay/is_weight_min": 1.0, "replay/is_weight_max": 1.0, "replay/is_clip_fraction": 0.0}
    return {"replay/is_weight_min": float(selected.min().item()),
            "replay/is_weight_max": float(selected.max().item()),
            "replay/is_clip_fraction": float(((delta[active] < lo) | (delta[active] > hi)).double().mean().item())}
