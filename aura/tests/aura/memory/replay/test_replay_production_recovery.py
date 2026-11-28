"""Server-only real source/callback contracts; NPU validation is a separate gate."""
import asyncio
import copy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from aura.memory.replay.codec import canonical, digest
from aura.memory.replay.source import PromptCursor, DataManagerSource
from aura.memory.replay.schema import ReplayIntegrityError
from test_replay_backpressure import method


@pytest.mark.parametrize('count', [0, 1, 3, 6])
def test_produced_cursor_not_train_step(count):
    data = [{'prompts': [[i, i+1]], 'id': [str(i)]} for i in range(6)]
    a = PromptCursor(lambda: iter(data), digest('dataset'))
    for _ in range(count):
        next(a)
    b = PromptCursor(lambda: iter(data), digest('dataset'))
    b.restore(a.state())
    assert list(b.iterator) == list(a.iterator)
    assert b.state()['produced_batches'] == count


@pytest.mark.parametrize('kind', ['content', 'order', 'short', 'identity', 'count', 'hash', 'exhaustion'])
def test_cursor_faults_do_not_publish_partial_cursor(kind):
    data = [{'id': i} for i in range(4)]
    a = PromptCursor(lambda: iter(data), digest('dataset'))
    next(a); next(a)
    state = a.state()
    changed = copy.deepcopy(data)
    if kind == 'content': changed[0]['id'] = 7
    if kind == 'order': changed.reverse()
    if kind == 'short': changed = changed[:1]
    if kind == 'identity': state['identity'] = digest('other')
    if kind == 'count': state['produced_batches'] = True
    if kind == 'hash': state['prefix_sha256'] = digest('wrong')
    if kind == 'exhaustion': state['exhausted'] = True
    b = PromptCursor(lambda: iter(changed), digest('dataset'))
    before = b.state()
    with pytest.raises((ReplayIntegrityError, ValueError)):
        b.restore(state)
    assert b.state() == before


def test_exhausted_cursor_reconstructs_exactly():
    a = PromptCursor(lambda: iter([{'a': 1}]), digest('dataset'))
    next(a)
    with pytest.raises(StopIteration): next(a)
    b = PromptCursor(lambda: iter([{'a': 1}]), digest('dataset'))
    b.restore(a.state())
    assert b.state() == a.state()
    with pytest.raises(StopIteration): next(b)


CONTROLLER = 'aura/controllers/train_controller/train_controller.py'


@pytest.mark.parametrize('pending', [False, True])
@pytest.mark.parametrize('fail_at', ['none', 'dispatch', 'generation', 'barrier', 'freeze', 'credit_mismatch'])
def test_real_controller_drain_orders_pause_join_barrier_freeze(pending, fail_at):
    events = []
    def point(name, result=None):
        def run(*args, **kwargs):
            events.append(name)
            if fail_at == name: raise TimeoutError(name)
            return result
        return run
    cursor = {'identity': digest('source'), 'produced_batches': int(pending)}
    dispatch = SimpleNamespace(pause_replay_source=SimpleNamespace(remote=point('dispatch', cursor)))
    queue = SimpleNamespace(quiesce_replay=SimpleNamespace(remote=point('barrier', True)))
    ray = SimpleNamespace(get=lambda ref, **kw: ref, get_actor=lambda *a, **kw: queue)
    batches = [{'replay_refs': []}] if pending else []
    if fail_at == 'credit_mismatch': batches = [] if pending else [{'replay_refs': []}]
    obj = SimpleNamespace(dispatch_actor=dispatch, replay_queue_timeout=2, _replay_credit_pending=pending,
        wait_for_online_batch=point('generation'),
        _replay_queue_rpc=point('freeze', {'batches': batches, 'metrics': [{}]*len(batches)}))
    import time
    fn = method(CONTROLLER, 'TrainController', 'quiesce_replay_source', ray=ray, time=time,
                TRAIN_CONTROLLER_NAMESPACE='controller_raygroup')
    failing = fail_at not in ('none', 'generation') or (pending and fail_at == 'generation')
    if failing:
        with pytest.raises((TimeoutError, RuntimeError)): fn(obj)
    else:
        assert fn(obj) == (cursor, batches, [{}]*len(batches))
        assert events == ['dispatch'] + (['generation'] if pending else []) + ['barrier', 'freeze']
    assert obj._recovery_paused


@pytest.mark.parametrize('stage', ['snapshot', 'restore', 'resume'])
def test_ambiguous_source_adapter_is_poisoned(stage):
    controller = Mock()
    adapter = DataManagerSource(controller, None, None)
    if stage == 'snapshot':
        controller.quiesce_replay_source.side_effect = TimeoutError('pending generation')
        action = adapter.snapshot
    elif stage == 'restore':
        action = lambda: adapter.restore({})
    else:
        adapter.paused = True
        controller.resume_replay_source.side_effect = TimeoutError('dispatch')
        action = adapter.resume
    with pytest.raises((TimeoutError, KeyError)): action()
    assert adapter.failed
    with pytest.raises(RuntimeError): adapter.resume()


def test_atomic_rollout_barrier_waits_for_real_generation():
    from aura.controllers.rollout_controller.rollout_queue import RolloutQueueActor
    cls = RolloutQueueActor.__ray_metadata__.modified_class
    queue = cls()
    assert queue.begin_generation()
    assert not queue.quiesce_replay()
    queue.add_queue({'prompt': 1})
    queue.end_generation()
    assert not queue.quiesce_replay()
    queue.pop_queue()
    assert queue.quiesce_replay()
    assert not queue.begin_generation()
    with pytest.raises(RuntimeError): queue.add_queue({'late': True})
    queue.resume_replay()
    assert queue.begin_generation()
    queue.end_generation()
    with pytest.raises(RuntimeError): queue.end_generation()


def test_actual_trainqueue_http_freeze_restore_and_late_arrival():
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from aura.controllers.train_controller.train_server import TrainServer
    from aura.controllers.train_controller.train_queue import TrainQueue
    server = TrainServer.__new__(TrainServer)
    server.queue_instance = TrainQueue(8, 2, 2)
    server.replay_demand_mode, server._recovery_frozen = True, False
    server._batch_ready = asyncio.Event()
    server.logger = Mock()
    server.queue_instance.add_minibatch({'replay_refs': [{'test': 1}], 'tensor-not-persisted': object()}, {'latency': 1.5})
    app = FastAPI()
    app.post('/freeze')(server.freeze_replay_queue)
    app.post('/restore')(server.restore_replay_queue)
    app.post('/resume')(server.resume_replay_queue)
    app.get('/pop')(server.pop_minibatch)
    with TestClient(app) as client:
        snapshot = client.post('/freeze').json()
        assert snapshot == {'batches': [{'replay_refs': [{'test': 1}]}], 'metrics': [{'latency': 1.5}]}
        assert client.get('/pop').status_code == 409
        with pytest.raises(RuntimeError): server.put_minibatch_to_queue({}, {})
        assert client.post('/restore', json=snapshot).status_code == 409
        server.queue_instance.pop_batch()
        assert client.post('/restore', json=snapshot).json() == snapshot
        assert client.post('/resume').json() == {'resumed': True}
        assert server.queue_instance.pop_batch() == ({'replay_refs': [{'test': 1}]}, {'latency': 1.5})


@pytest.mark.parametrize('kind,detail', [('missing_metric', 'lengths differ'),
    ('too_many', 'at most one batch'), ('wrong_type', 'metric 0 must be an object'),
    ('fields', 'batches and metrics fields'), ('list', 'must be lists'), ('refs', 'requires replay_refs')])
def test_restore_queue_malformed_never_mutates_fifo(kind, detail):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from aura.controllers.train_controller.train_server import TrainServer
    from aura.controllers.train_controller.train_queue import TrainQueue
    server = TrainServer.__new__(TrainServer)
    server.queue_instance = TrainQueue(8, 2, 2)
    server.replay_demand_mode, server._recovery_frozen = True, True
    server._batch_ready = asyncio.Event()
    app = FastAPI(); app.post('/restore')(server.restore_replay_queue)
    value = {'batches': [{'replay_refs': []}], 'metrics': [{}]}
    if kind == 'missing_metric': value['metrics'] = []
    if kind == 'too_many': value = {'batches': value['batches']*2, 'metrics': [{}]*2}
    if kind == 'wrong_type': value['metrics'] = [None]
    if kind == 'fields': value.pop('batches')
    if kind == 'list': value['batches'] = {}
    if kind == 'refs': value['batches'] = [{}]
    with TestClient(app) as client:
        response = client.post('/restore', json=value)
        assert response.status_code == 400 and detail in response.json()['detail']
    assert not server.queue_instance.queue and not server.queue_instance.metric_queue


@pytest.mark.parametrize('kind', ['ranks', 'step', 'version', 'world', 'empty'])
def test_fsdp_partial_or_wrong_watermark_receipts_rejected(kind):
    from aura.trainer.train_adapter.verl.full_async.replay_recovery import FSDPCallbacks
    receipts = [dict(rank=0, world_size=1, step=3, policy_version=3)]
    if kind == 'empty': receipts = []
    else: receipts[0][{'ranks': 'rank', 'step': 'step', 'version': 'policy_version', 'world': 'world_size'}[kind]] = 9
    with pytest.raises(ReplayIntegrityError): FSDPCallbacks._receipts(receipts, 3, 3)


def test_datamanager_adapter_uses_held_controller_not_trainer_copy():
    from aura.data_manager.verl_data import VerlDataManager
    manager = VerlDataManager()
    controller = Mock()
    manager.controller = controller
    source = manager.replay_source_adapter('config', 'actor')
    assert source.controller is controller
    controller.enable_replay_recovery.assert_called_once_with()


def recovery_config(tmp_path):
    from omegaconf import OmegaConf
    return OmegaConf.create({'replay': {'replay_ratio': .3}, 'replay_recovery': {
        'enabled': True, 'root': str(tmp_path), 'ledger_path': str(tmp_path/'ledger.db'),
        'branch': 'new', 'writer_request': 'new-worker-owner'},
        'data': {'train_batch_size': 2}, 'async_training': {'trigger_parameter_sync_step': 1},
        'actor_rollout_ref': {'actor': {'strategy': 'fsdp', 'ppo_epochs': 1, 'ppo_mini_batch_size': 2,
            'checkpoint': {'save_contents': ['model', 'optimizer', 'extra', 'hf_model'],
                           'load_contents': ['model', 'optimizer', 'extra']}}},
        'trainer': {'resume_mode': 'disable', 'default_hdfs_dir': None,
                    'default_local_dir': str(tmp_path), 'del_local_ckpt_after_load': False}})


@pytest.mark.parametrize('key,value', [
    ('replay.replay_ratio', 0), ('actor_rollout_ref.actor.strategy', 'megatron'),
    ('async_training.trigger_parameter_sync_step', 2), ('actor_rollout_ref.actor.ppo_epochs', 2),
    ('actor_rollout_ref.actor.ppo_mini_batch_size', 1), ('trainer.default_hdfs_dir', 'hdfs://example'),
    ('trainer.del_local_ckpt_after_load', True), ('trainer.resume_mode', 'auto'),
    ('replay_recovery.root', 'relative'), ('replay_recovery.ledger_path', 'relative'),
    ('replay_recovery.from_branch', 'new'),
    ('replay_recovery.branch', '../outside'), ('replay_recovery.branch', '/absolute'),
    ('actor_rollout_ref.actor.checkpoint.save_contents', ['model', 'optimizer', 'extra']),
    ('actor_rollout_ref.actor.checkpoint.load_contents', ['model'])])
def test_unsafe_production_config_rejected(tmp_path, key, value):
    from omegaconf import OmegaConf
    from aura.trainer.train_adapter.verl.full_async.replay_recovery import recovery_settings
    cfg = recovery_config(tmp_path)
    assert recovery_settings(cfg)['branch'] == 'new'
    OmegaConf.update(cfg, key, value)
    with pytest.raises(ValueError): recovery_settings(cfg)


def test_worker_owner_cannot_be_rebound():
    path = 'aura/trainer/train_adapter/verl/full_async/workers/fsdp_workers.py'
    bind = method(path, 'FsdpDetachActorWorker', 'bind_replay_owner')
    check = method(path, 'FsdpDetachActorWorker', '_check_replay_owner')
    worker = SimpleNamespace()
    bind(worker, 'old-worker-owner'); check(worker, 'old-worker-owner')
    with pytest.raises(RuntimeError): bind(worker, 'new-worker-owner')
    with pytest.raises(RuntimeError): check(worker, 'new-worker-owner')


@pytest.mark.parametrize('branch', ['../outside', '/absolute'])
def test_model_callbacks_reject_branch_path_escape_before_binding(tmp_path, branch):
    from aura.trainer.train_adapter.verl.full_async.replay_recovery import FSDPCallbacks
    workers = Mock()
    with pytest.raises(ReplayIntegrityError): FSDPCallbacks(workers, tmp_path, branch, 'owner', Mock())
    workers.bind_replay_owner.assert_not_called()
    assert list(tmp_path.iterdir()) == []
