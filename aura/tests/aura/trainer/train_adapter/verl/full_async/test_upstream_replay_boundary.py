# Copyright (c) 2026 Huawei Technologies Co.,Ltd.
# Licensed under Mulan PSL v2 (http://license.coscl.org.cn/MulanPSL2).
"""Dependency-isolated contracts for the !1073 / !1085 replay integration boundary."""
import ast
import asyncio
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from omegaconf import OmegaConf

SRC = Path(__file__).resolve().parents[6] / "aura/trainer/train_adapter/verl"


class Base:
    def _compute_old_log_prob(self, batch):
        return self.forward(batch)


def load_class(relative, name, methods):
    tree = ast.parse((SRC / relative).read_text())
    node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == name)
    node.decorator_list = []
    node.bases = [ast.Name(id="Base", ctx=ast.Load())]
    node.body = [n for n in node.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
                 and n.name in methods]
    scope = {"Base": Base, "OmegaConf": OmegaConf, "DataProto": object,
             "logger": MagicMock(), "ray": SimpleNamespace(get=MagicMock())}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])),
                 str(relative), "exec"), scope)
    return scope[name], scope


def config(mode="one_step_off_policy"):
    return OmegaConf.create({"extras": {"work_mode": mode}})


@pytest.mark.parametrize('mode,rows', [('fully_async', 16), ('one_step_off', 32)])
def test_actual_generate_uses_mode_specific_complete_group_count(mode, rows):
    from contextlib import nullcontext
    import numpy as np
    import torch
    cls, scope = load_class('full_async/full_async_trainer.py', 'FullyAsyncTrainer', {'_fit_generate'})
    scope.update(np=np, torch=torch, marked_timer=lambda *a, **k: nullcontext(),
                 DataProto=SimpleNamespace(from_dict=lambda **k: SimpleNamespace(non_tensor_batch={}, meta_info={})))
    obj = cls()
    obj.config = OmegaConf.create({'extras': {'work_mode': mode}, 'data': {'train_batch_size': 16},
        'algorithm': {'adv_estimator': 'grpo'}, 'actor_rollout_ref': {'rollout': {'n': 2, 'temperature': 1.}}})
    obj.required_samples, obj._replay_session, obj.timing_raw = 8, None, {}
    raw = {'prompt_ids': [str(i // 2) for i in range(rows)], 'attention_mask': torch.ones(rows, 3)}
    obj._get_samples_from_queue = lambda: (0, raw)
    obj._extract_generate_tensors = lambda value: {}
    assert len(asyncio.run(obj._fit_generate()).non_tensor_batch['uid']) == rows
    raw['prompt_ids'].pop()
    with pytest.raises(ValueError):
        asyncio.run(obj._fit_generate())


def test_failed_actual_update_confirmation_prevents_ack_and_version_advance():
    from contextlib import nullcontext
    from unittest.mock import AsyncMock
    from aura.memory.replay.transaction import TransactionalReplaySession
    from aura.trainer.train_adapter.verl.full_async import replay_loss
    cls, scope = load_class('full_async/full_async_trainer.py', 'FullyAsyncTrainer', {'fit_step'})
    scope['marked_timer'] = lambda *a, **k: nullcontext()
    obj = cls()
    obj.global_steps, obj.total_train_steps, obj.epoch, obj.current_param_version = 1, 2, 0, 0
    obj._fit_start_profile = MagicMock()
    obj._fit_generate = AsyncMock(return_value=object())
    for name in ('_fit_compute_reward', '_fit_compute_log_prob', '_fit_compute_ref_log_prob',
                 '_fit_compute_critic', '_fit_compute_advantage', '_fit_update_critic', '_fit_update_actor'):
        setattr(obj, name, lambda batch: batch)
    session = TransactionalReplaySession.__new__(TransactionalReplaySession)
    session.mixer, session.priority_updates, session.ack_update = MagicMock(), MagicMock(), MagicMock()
    obj._replay_session = session
    obj._replay_model = SimpleNamespace(confirm_update=MagicMock(side_effect=RuntimeError('optimizer skipped')))
    obj._fit_update_local_step = MagicMock()
    with patch.object(replay_loss, 'apply_behavior_correction', return_value={}):
        with pytest.raises(RuntimeError, match='optimizer skipped'):
            asyncio.run(obj.fit_step())
    session.ack_update.assert_not_called()
    obj._fit_update_local_step.assert_not_called()
    assert obj.current_param_version == 0


@pytest.mark.parametrize("mode", ["one_step_off_policy", "one_step_off", None])
def test_one_step_never_looks_up_existing_streaming_queue(mode):
    cls, _ = load_class("full_async/train_main.py", "FullyAsyncTaskRunner", {"_inject_sample_queue"})
    queue = MagicMock()
    trainer = MagicMock()
    with patch.dict(sys.modules, {"aura.controllers.rollout_controller.sample_queue": queue}):
        cls()._inject_sample_queue(trainer, config(mode))
    queue.get_sample_queue.assert_not_called()
    trainer.set_sample_queue.remote.assert_not_called()


def test_fully_async_injects_required_queue():
    cls, scope = load_class("full_async/train_main.py", "FullyAsyncTaskRunner", {"_inject_sample_queue"})
    queue, trainer = MagicMock(), MagicMock()
    with patch.dict(sys.modules, {"aura.controllers.rollout_controller.sample_queue": queue}):
        cls()._inject_sample_queue(trainer, config("fully_async"))
    queue.get_sample_queue.assert_called_once()
    trainer.set_sample_queue.remote.assert_called_once_with(queue.get_sample_queue.return_value)
    scope["ray"].get.assert_called_once()


def test_fully_async_missing_queue_is_not_silently_fallback():
    cls, _ = load_class("full_async/train_main.py", "FullyAsyncTaskRunner", {"_inject_sample_queue"})
    queue = MagicMock()
    queue.get_sample_queue.side_effect = RuntimeError("missing queue")
    with patch.dict(sys.modules, {"aura.controllers.rollout_controller.sample_queue": queue}):
        with pytest.raises(RuntimeError, match="missing queue"):
            cls()._inject_sample_queue(MagicMock(), config("fully_async"))


def trainer(mode):
    cls, _ = load_class("full_async/full_async_trainer.py", "FullyAsyncTrainer",
                        {"_compute_old_log_prob", "_maybe_sync_final_weights", "_should_save_now",
                         "set_sample_queue"})
    obj = cls()
    obj.config = config(mode)
    obj.forward = MagicMock(return_value=("logp", .1))
    obj.actor_rollout_wg = MagicMock()
    obj.current_param_version = 5
    obj.local_trigger_step = 1
    obj.trigger_parameter_sync_step = 1
    obj.last_synced_param_version = 0
    obj._replay_session = None
    return obj


def test_one_step_native_logprob_does_not_add_mis_snapshots():
    obj = trainer("one_step_off_policy")
    assert obj._compute_old_log_prob("batch") == ("logp", .1)
    obj.forward.assert_called_once_with("batch")
    obj.actor_rollout_wg.save_model_to_cpu.assert_not_called()
    obj.actor_rollout_wg.restore_model_from_cpu.assert_not_called()


def test_fully_async_first_trigger_snapshots_current_weights():
    obj = trainer("fully_async")
    assert obj._compute_old_log_prob("batch") == ("logp", .1)
    obj.actor_rollout_wg.save_model_to_cpu.assert_called_once_with(1)
    obj.actor_rollout_wg.restore_model_from_cpu.assert_not_called()


@pytest.mark.parametrize("fails", [False, True])
def test_mis_restores_working_weights_even_if_forward_fails(fails):
    obj = trainer("fully_async")
    obj.local_trigger_step = 3
    if fails:
        obj.forward.side_effect = RuntimeError("forward failed")
        with pytest.raises(RuntimeError, match="forward failed"):
            obj._compute_old_log_prob("batch")
    else:
        assert obj._compute_old_log_prob("batch") == ("logp", .1)
    assert obj.actor_rollout_wg.method_calls == [
        ("save_model_to_cpu", (3,), {}), ("restore_model_from_cpu", (1,), {}),
        ("restore_model_from_cpu", (3,), {}), ("clear_cpu_model", (3,), {})]


def test_missing_mis_apis_explains_dependency_mismatch():
    obj = trainer("fully_async")
    obj.actor_rollout_wg = object()
    with pytest.raises(RuntimeError, match="pinned verl"):
        obj._compute_old_log_prob("batch")
    obj.forward.assert_not_called()


@pytest.mark.parametrize("frequency", [0, -1, 5, 10])
def test_final_export_does_not_depend_on_validation_frequency(frequency):
    obj = trainer("one_step_off_policy")
    obj.config.trainer = {"test_freq": frequency}
    from unittest.mock import AsyncMock
    obj._fit_update_weights = AsyncMock()
    asyncio.run(obj._maybe_sync_final_weights())
    obj._fit_update_weights.assert_awaited_once()
    obj.last_synced_param_version = obj.current_param_version
    obj._fit_update_weights.reset_mock()
    asyncio.run(obj._maybe_sync_final_weights())
    obj._fit_update_weights.assert_not_awaited()


@pytest.mark.parametrize("force,version,frequency,esi,expected", [
    (True, 6, 10, False, True), (False, 10, 10, False, True),
    (False, 6, 10, False, False), (False, 6, 10, True, True),
    (True, 6, 0, False, False), (True, 6, -1, True, False)])
def test_checkpoint_final_periodic_disabled_contract(force, version, frequency, esi, expected):
    obj = trainer("one_step_off_policy")
    obj.current_param_version = version
    obj.config.trainer = {"save_freq": frequency}
    assert obj._should_save_now(force, esi) is expected


def test_streaming_queue_cannot_replace_replay_session():
    obj = trainer("one_step_off_policy")
    obj._replay_session = object()
    with pytest.raises(ValueError, match="ReplaySession"):
        obj.set_sample_queue(object())


@pytest.mark.parametrize("recovery_fails", [False, True])
def test_paired_recovery_publishes_sync_version_only_after_success(tmp_path, recovery_fails):
    cls, _ = load_class("full_async/full_async_trainer.py", "FullyAsyncTrainer",
                        {"set_replay_session", "_maybe_sync_final_weights"})
    obj = cls()
    obj._replay_session = None
    obj.sample_queue = None
    obj.tokenizer = SimpleNamespace(pad_token_id=0)
    obj.config = SimpleNamespace(algorithm=SimpleNamespace(adv_estimator="grpo"))
    obj.current_param_version = 0
    obj.global_steps = 0
    obj.last_synced_param_version = 0
    obj._checkpoint_loaded = False
    obj.actor_rollout_wg = MagicMock()
    obj.data_manager = MagicMock()
    source = obj.data_manager.replay_source_adapter.return_value
    from unittest.mock import AsyncMock
    obj._fit_update_weights = AsyncMock()

    modules = {name: MagicMock() for name in [
        "aura.memory.replay.config", "aura.memory.replay.runtime",
        "aura.memory.replay.transaction",
        "aura.trainer.train_adapter.verl.full_async.replay_loss",
        "aura.trainer.train_adapter.verl.full_async.replay_recovery",
    ]}
    recovery = modules["aura.trainer.train_adapter.verl.full_async.replay_recovery"]
    recovery.recovery_settings.return_value = {
        "root": str(tmp_path), "branch": "resumed", "writer_request": "writer",
        "from_branch": "original", "ledger_path": str(tmp_path / "ledger.sqlite"),
    }
    transaction = modules["aura.memory.replay.transaction"].TransactionalReplaySession
    session = MagicMock()
    transaction.recover.return_value = (session, {
        "body": {"policy_version": 3, "step": 3, "commit_id": "paired-3"},
    })
    if recovery_fails:
        transaction.recover.side_effect = RuntimeError("paired recovery failed")
    with patch.dict(sys.modules, modules):
        if recovery_fails:
            with pytest.raises(RuntimeError, match="paired recovery failed"):
                obj.set_replay_session({})
            assert obj.current_param_version == obj.last_synced_param_version == 0
            assert obj.global_steps == 0 and not obj._checkpoint_loaded
            source.resume.assert_not_called()
        else:
            obj.set_replay_session({})
            assert obj._replay_session is session
            assert obj.current_param_version == obj.last_synced_param_version == 3
            assert obj.global_steps == 4 and obj._checkpoint_loaded
            source.resume.assert_called_once_with()
            asyncio.run(obj._maybe_sync_final_weights())
            obj._fit_update_weights.assert_not_awaited()
