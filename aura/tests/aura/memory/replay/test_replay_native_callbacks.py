"""Server CPU tests for the required native save/load and version boundaries."""
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from aura.memory.replay.codec import canonical
from aura.memory.replay.schema import ReplayIntegrityError
from aura.trainer.train_adapter.verl.full_async.replay_recovery import FSDPCallbacks
from test_replay_backpressure import method


@pytest.mark.parametrize('fault', ['none', 'save', 'missing_model', 'missing_optimizer', 'missing_extra', 'missing_receipt', 'missing_export'])
def test_native_callback_keeps_required_artifacts_and_propagates_failure(tmp_path, fault):
    receipt = dict(step=3, policy_version=3, rank=0, world_size=1)

    def write(path, step, version, owner):
        assert (step, version, owner) == (3, 3, 'owner')
        if fault == 'save':
            raise OSError('native save failed')
        folder = Path(path)
        for kind, label in [('model', 'model'), ('optim', 'optimizer'), ('extra_state', 'extra')]:
            if fault != 'missing_' + label:
                (folder / f'{kind}_world_size_1_rank_0.pt').write_bytes(b'native state')
        if fault != 'missing_receipt':
            (folder / 'replay_rank_0.json').write_bytes(canonical(receipt))
        if fault != 'missing_export':
            (folder / 'huggingface').mkdir()
            (folder / 'huggingface' / 'model.safetensors').write_bytes(b'inference export')
        return [receipt]

    workers = SimpleNamespace(bind_replay_owner=Mock(), write_replay_checkpoint=write)
    callbacks = FSDPCallbacks(workers, tmp_path, 'branch', 'owner', Mock())
    if fault != 'none':
        with pytest.raises((OSError, ReplayIntegrityError)):
            callbacks.save(3, 3)
    else:
        result = callbacks.save(3, 3)
        assert type(result) is dict
        assert set(result['artifacts']) == {'model', 'optimizer', 'scheduler', 'rng'}
        assert all(result['artifacts'].values())
        assert result['artifacts']['scheduler'] == result['artifacts']['rng']


@pytest.mark.parametrize('fault', ['none', 'load', 'version', 'rank'])
def test_restore_loads_native_state_before_confirming_inference(tmp_path, fault):
    folder = tmp_path / 'actor'
    folder.mkdir()
    saved = dict(step=3, policy_version=3, rank=0, world_size=1)
    # Older paired points also contain optional full-state diagnostic hashes.
    (folder / 'replay_rank_0.json').write_bytes(canonical({**saved, 'model_sha256': 'a' * 64}))
    events = []

    def load(path, step, version, owner):
        events.append('load')
        assert (path, step, version, owner) == (str(folder), 3, 3, 'owner')
        if fault == 'load':
            raise OSError('native load failed')
        value = dict(saved)
        if fault in ('version', 'rank'):
            value['policy_version' if fault == 'version' else 'rank'] = 8
        return [value]

    workers = SimpleNamespace(bind_replay_owner=Mock(), load_replay_checkpoint=load)
    check = Mock(side_effect=lambda *args: events.append('inference'))
    callbacks = FSDPCallbacks(workers, tmp_path, 'new', 'owner', check)
    paths = {'model': [str(folder / 'model_world_size_1_rank_0.pt')]}
    if fault == 'none':
        assert callbacks.load(paths, 3, 3) == {'step': 3, 'policy_version': 3}
        assert events == ['load', 'inference']
        check.assert_called_once_with(str(folder / 'huggingface'), 3)
    else:
        with pytest.raises((OSError, ReplayIntegrityError)):
            callbacks.load(paths, 3, 3)
        check.assert_not_called()


@pytest.mark.parametrize('fault', ['none', 'optimizer', 'scheduler', 'step'])
def test_native_watermark_uses_actual_cpu_optimizer_and_scheduler(fault):
    import torch

    model = torch.nn.Linear(2, 1, device='cpu')
    optimizer = torch.optim.AdamW(model.parameters(), foreach=False, fused=False)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1)
    model(torch.ones(1, 2, device='cpu')).sum().backward()
    optimizer.step()
    scheduler.step()
    if fault == 'optimizer':
        next(iter(optimizer.state.values()))['step'].fill_(8)
    if fault == 'scheduler':
        scheduler.last_epoch = 8
    engine = SimpleNamespace(optimizer=optimizer, lr_scheduler=scheduler, is_optimizer_offload_enabled=False,
                             checkpoint_manager=SimpleNamespace(rank=0, world_size=1))
    worker = SimpleNamespace(actor=SimpleNamespace(engine=engine))
    function = method('aura/trainer/train_adapter/verl/full_async/workers/fsdp_workers.py',
                      'FsdpDetachActorWorker', '_replay_checkpoint_receipt')
    if fault == 'none':
        assert function(worker, 1, 1) == dict(step=1, policy_version=1, rank=0, world_size=1)
    else:
        with pytest.raises(RuntimeError):
            function(worker, 2 if fault == 'step' else 1, 1)


@pytest.mark.parametrize('skip', [False, True])
def test_ack_confirmation_rejects_actual_skipped_optimizer_step(skip):
    import torch
    model = torch.nn.Linear(2, 1, device='cpu')
    optimizer = torch.optim.AdamW(model.parameters(), foreach=False, fused=False)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1)
    model(torch.ones(1, 2)).sum().backward()
    # Match the pinned FSDP engine: nonfinite grad norm skips optimizer.step;
    # the surrounding training API may still return normally and advance LR.
    if not skip:
        optimizer.step()
    scheduler.step()
    engine = SimpleNamespace(optimizer=optimizer, lr_scheduler=scheduler, is_optimizer_offload_enabled=False,
                             checkpoint_manager=SimpleNamespace(rank=0, world_size=1))
    worker = SimpleNamespace(actor=SimpleNamespace(engine=engine), _check_replay_owner=Mock())
    receipt = method('aura/trainer/train_adapter/verl/full_async/workers/fsdp_workers.py',
                     'FsdpDetachActorWorker', '_replay_checkpoint_receipt')
    worker._replay_checkpoint_receipt = lambda step, version: receipt(worker, step, version)
    confirm = method('aura/trainer/train_adapter/verl/full_async/workers/fsdp_workers.py',
                     'FsdpDetachActorWorker', 'confirm_replay_update')
    workers = SimpleNamespace(bind_replay_owner=Mock(),
        confirm_replay_update=lambda version, owner: [confirm(worker, version, owner)])
    from aura.trainer.train_adapter.verl.full_async.replay_recovery import FSDPCallbacks
    callbacks = FSDPCallbacks.__new__(FSDPCallbacks)
    callbacks.workers, callbacks.owner = workers, 'owner'
    if skip:
        with pytest.raises(RuntimeError, match='optimizer/scheduler'):
            callbacks.confirm_update(1)
    else:
        callbacks.confirm_update(1)


@pytest.mark.parametrize('offload,actual_step', [(False, 1), (True, 1), (True, 0)])
def test_native_receipt_waits_for_offload_before_reading_step(offload, actual_step):
    events = []

    class Step:
        def __int__(self):
            assert events == (['sync'] if offload else [])
            events.append('read')
            return actual_step

    device = SimpleNamespace(synchronize=Mock(side_effect=lambda: events.append('sync')))
    engine = SimpleNamespace(
        is_optimizer_offload_enabled=offload,
        optimizer=SimpleNamespace(state={'parameter': {'step': Step()}}),
        lr_scheduler=SimpleNamespace(state_dict=lambda: {'last_epoch': 1}),
        checkpoint_manager=SimpleNamespace(rank=0, world_size=1))
    worker = SimpleNamespace(actor=SimpleNamespace(engine=engine))
    receipt = method('aura/trainer/train_adapter/verl/full_async/workers/fsdp_workers.py',
                     'FsdpDetachActorWorker', '_replay_checkpoint_receipt', get_torch_device=lambda: device)
    if actual_step == 1:
        assert receipt(worker, 1, 1) == dict(step=1, policy_version=1, rank=0, world_size=1)
    else:
        with pytest.raises(RuntimeError, match='optimizer/scheduler version differs'):
            receipt(worker, 1, 1)
    assert events == (['sync', 'read'] if offload else ['read'])


def test_native_receipt_propagates_offload_sync_failure_without_reading_state():
    state = Mock()
    device = SimpleNamespace(synchronize=Mock(side_effect=RuntimeError('device copy failed')))
    engine = SimpleNamespace(is_optimizer_offload_enabled=True, optimizer=SimpleNamespace(state=state))
    worker = SimpleNamespace(actor=SimpleNamespace(engine=engine))
    receipt = method('aura/trainer/train_adapter/verl/full_async/workers/fsdp_workers.py',
                     'FsdpDetachActorWorker', '_replay_checkpoint_receipt', get_torch_device=lambda: device)
    with pytest.raises(RuntimeError, match='device copy failed'):
        receipt(worker, 1, 1)
    state.values.assert_not_called()
