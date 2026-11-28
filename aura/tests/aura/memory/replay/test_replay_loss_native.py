# Copyright (c) 2026 Huawei Technologies Co.,Ltd.
# Licensed under Mulan PSL v2 (http://license.coscl.org.cn/MulanPSL2).
"""CPU tests of the pinned verl engine's real loss and autograd, not a mock loss."""
import math
import copy
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
from omegaconf import OmegaConf
from verl import DataProto
from verl.utils import tensordict_utils as tu
from verl.workers.utils.losses import ppo_loss
from verl.workers.utils.padding import left_right_2_no_padding, no_padding_2_padding

from aura.trainer.train_adapter.verl.full_async.replay_loss import (
    apply_behavior_correction, validate_loss_config, validate_native_loss,
)


def batch():
    mask = torch.tensor([[1, 0, 1], [1, 1, 0], [1, 0, 1], [1, 1, 1]])
    old = torch.full((4, 3), -1.0)
    delta = torch.tensor([math.log(4), math.log(.25), 0, math.log(1.5)])[:, None]
    attention = torch.tensor([[0, 1, 1, 1, 1], [1, 1, 1, 1, 0],
                              [0, 1, 1, 1, 1], [1, 1, 1, 1, 1]])
    return DataProto.from_dict(tensors={
        "prompts": torch.ones((4, 2), dtype=torch.int64),
        "responses": torch.ones((4, 3), dtype=torch.int64),
        "input_ids": torch.ones((4, 5), dtype=torch.int64),
        "position_ids": (attention.cumsum(-1)-1) * attention,
        "attention_mask": attention, "response_mask": mask,
        "old_log_probs": old, "rollout_log_probs": old-delta,
        "advantages": torch.tensor([1., -1., 2., -2.])[:, None].expand(4, 3).clone(),
    })


def loss_config():
    return OmegaConf.create({"global_batch_info": {}, "loss_scale_factor": None,
        "loss_agg_mode": "token-mean", "policy_loss": {"loss_mode": "vanilla"},
        "entropy_coeff": 0., "use_kl_loss": False, "clip_ratio": .2,
        "clip_ratio_low": None, "clip_ratio_high": None, "clip_ratio_c": 3.})


def engine_data(b, global_tokens=None):
    td = left_right_2_no_padding(b.to_tensordict())
    for key, value in {"dp_size": 1, "global_batch_size": 4,
                       "batch_num_tokens": global_tokens or int(b.batch["response_mask"].sum())}.items():
        tu.assign_non_tensor_data(td, key, value)
    return td


@pytest.mark.parametrize("microbatches", [1, 2, 4])
def test_native_engine_loss_gradient_and_global_normalization(microbatches):
    validate_native_loss()
    b = batch()
    original = {k: b.batch[k].clone() for k in ("old_log_probs", "rollout_log_probs", "advantages", "response_mask")}
    metrics = apply_behavior_correction(b)
    expected_weights = torch.tensor([2., .5, 1., 1.5])[:, None].expand(4, 3)
    expected_weights = torch.where(b.batch["response_mask"].bool(), expected_weights, 1.)
    torch.testing.assert_close(b.batch["rollout_is_weights"], expected_weights)
    assert not b.batch["rollout_is_weights"].requires_grad
    assert metrics["replay/is_weight_min"] == .5 and metrics["replay/is_weight_max"] == 2.
    assert metrics["replay/is_clip_fraction"] == pytest.approx(4/9)
    total_tokens = int(b.batch["response_mask"].sum())
    total_loss = 0.
    for part in b.chunk(microbatches):
        td = engine_data(part, total_tokens)
        torch.testing.assert_close(td["rollout_is_weights"], part.batch["rollout_is_weights"])
        current = torch.full_like(td["input_ids"].values(), -1., dtype=torch.float32, requires_grad=True)
        actual, _ = ppo_loss(loss_config(), {"log_probs": current}, td)
        response = no_padding_2_padding(current, td)
        manual = -(td["advantages"] * td["rollout_is_weights"] *
                   torch.exp(response - td["old_log_probs"]) * td["response_mask"]).sum()/total_tokens
        expected_grad, = torch.autograd.grad(manual, current, retain_graph=True)
        actual.backward()
        torch.testing.assert_close(actual, manual, rtol=0, atol=1e-7)
        torch.testing.assert_close(current.grad, expected_grad, rtol=0, atol=1e-7)
        assert torch.isfinite(current.grad).all()
        total_loss += float(actual.detach())
    expected = -(b.batch["advantages"] * expected_weights * b.batch["response_mask"]).sum()/total_tokens
    assert total_loss == pytest.approx(float(expected), abs=1e-7)
    for key, value in original.items():
        assert torch.equal(b.batch[key], value)


def test_unit_weights_match_native_unweighted_loss_and_gradient():
    b = batch()
    b.batch["rollout_log_probs"] = b.batch["old_log_probs"].clone()
    plain = copy.deepcopy(b)
    apply_behavior_correction(b)
    values = []
    for item in (plain, b):
        td = engine_data(item)
        current = torch.full_like(td["input_ids"].values(), -1., dtype=torch.float32, requires_grad=True)
        loss, _ = ppo_loss(loss_config(), {"log_probs": current}, td)
        loss.backward()
        values.append((loss.detach(), current.grad))
    for plain_value, weighted_value in zip(*values):
        assert torch.equal(plain_value, weighted_value)


def test_nonunit_weights_really_change_native_gradient():
    plain, weighted = batch(), batch()
    apply_behavior_correction(weighted)
    gradients = []
    for item in (plain, weighted):
        td = engine_data(item)
        current = torch.full_like(td["input_ids"].values(), -1., dtype=torch.float32, requires_grad=True)
        loss, _ = ppo_loss(loss_config(), {"log_probs": current}, td)
        loss.backward()
        gradients.append(current.grad)
    assert not torch.equal(*gradients)


@pytest.mark.parametrize("ratio", [.25, 1., 4.])
def test_native_ppo_clip_remains_separate_from_behavior_clip(ratio):
    b = batch()
    apply_behavior_correction(b)
    td = engine_data(b)
    current = torch.full_like(td["input_ids"].values(), -1.+math.log(ratio),
                              dtype=torch.float32, requires_grad=True)
    actual, _ = ppo_loss(loss_config(), {"log_probs": current}, td)
    adv = td["advantages"]
    first = -adv * ratio
    clipped = -adv * min(max(ratio, .8), 1.2)
    upper = torch.maximum(first, clipped)
    dual = torch.where(adv < 0, torch.minimum(-adv * 3., upper), upper)
    expected = (dual * td["rollout_is_weights"] * td["response_mask"]).sum()/9
    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-7)
    actual.backward()
    assert torch.isfinite(current.grad).all()


@pytest.mark.parametrize("enabled,fail", [(False,False),(True,False),(True,True)])
def test_production_fit_step_weights_before_actor_and_commits_only_after_success(enabled, fail):
    import asyncio
    import contextlib
    from unittest.mock import Mock, AsyncMock
    from test_replay_backpressure import method
    b = batch()
    events = []
    session = Mock()
    session.priority_updates.side_effect = lambda value: events.append("priority-prepared") or {("s", "g"): 1.}
    session.commit_success.side_effect = lambda *args: events.append("commit") or {}
    trainer = SimpleNamespace(global_steps=1, total_train_steps=2, epoch=0, current_param_version=3,
        local_trigger_step=0, config=OmegaConf.create({"algorithm": {"norm_adv_by_std_in_grpo": True}}),
        _replay_session=session if enabled else None, _fit_generate=AsyncMock(return_value=b))
    for name in ("_fit_start_profile", "_fit_update_local_step", "_fit_dump_data", "_fit_save_checkpoint",
                 "_fit_stop_profile", "_fit_collect_metrics", "_fit_torch_memory", "_fit_postprocess_step"):
        setattr(trainer, name, Mock())
    for name in ("_fit_compute_reward", "_fit_compute_log_prob", "_fit_compute_ref_log_prob",
                 "_fit_compute_critic", "_fit_compute_advantage", "_fit_update_critic"):
        setattr(trainer, name, lambda value: value)
    trainer._fit_update_weights = AsyncMock()
    def update(value):
        events.append("actor")
        assert ("rollout_is_weights" in value.batch) is enabled
        if fail:
            raise RuntimeError("injected actor update failure")
        return value
    trainer._fit_update_actor = update
    fn = method("aura/trainer/train_adapter/verl/full_async/full_async_trainer.py", "FullyAsyncTrainer", "fit_step",
        logger=Mock(), OmegaConf=OmegaConf, marked_timer=lambda *args: contextlib.nullcontext(),
        TrainingStopException=RuntimeError)
    if fail:
        with pytest.raises(RuntimeError, match="injected actor"):
            asyncio.run(fn(trainer))
        session.commit_success.assert_not_called()
    else:
        asyncio.run(fn(trainer))
        assert events == (["priority-prepared", "actor", "commit"] if enabled else ["actor"])
        if enabled:
            session.commit_success.assert_called_once_with(3, {("s", "g"): 1.})


def test_extreme_behavior_ratios_are_finite_clipped_and_detached():
    b = batch()
    b.batch["old_log_probs"].requires_grad_()
    b.batch["rollout_log_probs"] = torch.tensor([-1000., 1000., -1., -1.])[:, None].expand(4, 3)
    apply_behavior_correction(b)
    w = b.batch["rollout_is_weights"]
    assert not w.requires_grad and torch.isfinite(w).all()
    assert w.min() == .5 and w.max() == 2.
    assert torch.equal(w[~b.batch["response_mask"].bool()], torch.ones(3))


def test_double_correction_fails_without_overwriting_weights():
    b = batch()
    apply_behavior_correction(b)
    w = b.batch["rollout_is_weights"].clone()
    with pytest.raises(ValueError, match="double IS"):
        apply_behavior_correction(b)
    assert torch.equal(w, b.batch["rollout_is_weights"])


@pytest.mark.parametrize("key", ["old_log_probs", "rollout_log_probs", "advantages"])
@pytest.mark.parametrize("position", [(0, 0), (0, 1)])
def test_nonfinite_even_masked_policy_inputs_fail_closed(key, position):
    b = batch()
    b.batch[key][position] = float("nan")
    with pytest.raises(ValueError, match="Nonfinite"):
        apply_behavior_correction(b)
    assert "rollout_is_weights" not in b.batch


@pytest.mark.parametrize("kind", ["missing", "shape", "mask"])
def test_malformed_loss_inputs_fail_before_update(kind):
    # Plain dictionaries permit constructing invalid shapes that TensorDict itself rejects.
    b = SimpleNamespace(batch=dict(batch().batch.items()))
    if kind == "missing":
        del b.batch["rollout_log_probs"]
    elif kind == "shape":
        b.batch["advantages"] = torch.ones(4, 2)
    elif kind == "mask":
        b.batch["response_mask"][0, 0] = 2
    with pytest.raises(ValueError):
        apply_behavior_correction(b)


@pytest.mark.parametrize("all_rows", [False, True])
def test_filtered_real_rows_keep_neutral_is_weights(all_rows):
    b = batch()
    b.batch["response_mask"][:(4 if all_rows else 1)] = 0
    metrics = apply_behavior_correction(b)
    assert torch.all(b.batch["rollout_is_weights"][b.batch["response_mask"] == 0] == 1)
    assert all(math.isfinite(value) for value in metrics.values())
    if all_rows:
        assert metrics == {"replay/is_weight_min": 1., "replay/is_weight_max": 1., "replay/is_clip_fraction": 0.}


def adapter_config():
    return OmegaConf.create({"actor_rollout_ref": {"actor": {"strategy": "fsdp2",
        "policy_loss": {"loss_mode": "vanilla"}}}, "trainer": {
        "use_legacy_worker_impl": "disable", "critic_warmup": 0},
        "algorithm": {"rollout_correction": {"bypass_mode": False}}})


def test_supported_adapter_and_native_capability():
    validate_loss_config(adapter_config())
    validate_native_loss()


@pytest.mark.parametrize("path,value", [
    ("actor_rollout_ref.actor.strategy", "megatron"),
    ("actor_rollout_ref.actor.policy_loss.loss_mode", "gspo"),
    ("trainer.use_legacy_worker_impl", "auto"), ("trainer.critic_warmup", 1),
    ("algorithm.rollout_correction.bypass_mode", True),
    ("algorithm.rollout_correction.rollout_is", "token"),
    ("algorithm.rollout_correction.rollout_rs", "token"),
    ("algorithm.rollout_correction.rollout_is_batch_normalize", True),
])
def test_unsupported_or_duplicate_loss_modes_rejected(path, value):
    config = adapter_config()
    OmegaConf.update(config, path, value)
    with pytest.raises(ValueError):
        validate_loss_config(config)
