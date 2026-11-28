"""Exercise production controller methods without HTTP/Ray services."""
import ast
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock
import io
import time

import pytest


def method(relative, cls_name, name, **namespace):
    root = Path(__file__).resolve().parents[4]
    tree = ast.parse((root / relative).read_text(encoding="utf-8"))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == cls_name)
    fn = next(n for n in cls.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == name)
    fn.decorator_list = []
    module = ast.Module(body=[fn], type_ignores=[])
    compiled = dict(namespace)
    exec(compile(ast.fix_missing_locations(module), relative, "exec"), compiled)
    return compiled[name]


def test_replay_demand_initialization_does_not_prefetch():
    obj = SimpleNamespace(replay_demand_mode=True, send_initial_batch_groups_to_rollout=Mock(),
                          unlock_rollout_unit=Mock(), clean_train_updated_weights=Mock())
    fn = method("aura/controllers/train_controller/train_controller.py", "TrainController", "initialize_rollout")
    fn(obj)
    obj.send_initial_batch_groups_to_rollout.assert_not_called()
    obj.unlock_rollout_unit.assert_called_once()


@pytest.mark.parametrize("replay", [False, True])
def test_train_server_only_replenishes_legacy_path(replay):
    obj = SimpleNamespace(logger=Mock(), replay_demand_mode=replay, queue_instance=Mock(),
                          dispatch_actor=Mock(), _batch_ready=Mock())
    obj.queue_instance.add_minibatch.return_value = True
    fn = method("aura/controllers/train_controller/train_server.py", "TrainServer", "put_minibatch_to_queue",
                DEFAULT_BATCH_SIZE=2)
    fn(obj, {}, {})
    assert obj.dispatch_actor.send_batch_groups.remote.call_count == (0 if replay else 1)
    obj.queue_instance.add_minibatch.assert_called_once()


def test_replay_fetch_requests_one_credit_and_has_bounded_wait():
    remote = Mock()
    remote.get.side_effect = lambda value, **kwargs: value
    obj = SimpleNamespace(replay_demand_mode=True, replay_queue_timeout=0, dispatch_actor=Mock(), logger=Mock(),
                          training_batch_queue_ready=lambda: False, unlock_rollout_unit=Mock(),
                          _replay_credit_pending=False)
    obj.dispatch_actor.send_batch_groups.remote.return_value = 1
    request = method("aura/controllers/train_controller/train_controller.py", "TrainController",
                     "request_online_batch", ray=remote)
    obj.request_online_batch = lambda: request(obj)
    obj.wait_for_online_batch = Mock(side_effect=TimeoutError("requested online batch"))
    fn = method("aura/controllers/train_controller/train_controller.py", "TrainController", "get_next_training_batch",
                ray=remote, time=time, DEFAULT_SLEEP_TIME=0, logger=Mock())
    with pytest.raises(TimeoutError):
        fn(obj)
    obj.dispatch_actor.send_batch_groups.remote.assert_called_once_with(1)
    obj.dispatch_actor.unlock_rollout_unit.remote.assert_called_once()
    assert obj._replay_credit_pending
    obj._replay_credit_pending = False
    obj.dispatch_actor.send_batch_groups.remote.return_value = 0
    with pytest.raises(RuntimeError, match="exhausted"):
        fn(obj)


def test_replay_controller_switch_is_explicit():
    obj = SimpleNamespace(train_server=SimpleNamespace(replay_demand_mode=False))
    fn = method("aura/controllers/train_controller/train_controller.py", "TrainController", "enable_replay_demand")
    fn(obj, 45)
    assert obj.replay_demand_mode and obj.train_server.replay_demand_mode and obj.replay_queue_timeout == 45


def test_replay_failure_notifies_own_idle_rollouter():
    ray = Mock()
    ray.wait.return_value = ([object()], [])
    ray.get.side_effect = ValueError("pure replay pool expired")
    trainer = Mock()
    controller = Mock()
    obj = SimpleNamespace(running=False, components={"trainer": trainer, "replay_controller": controller})
    relative = "aura/trainer/train_adapter/verl/full_async/train_main.py"
    drain = method(relative, "FullyAsyncTaskRunner", "_drain_futures", ray=ray)
    handle = method(relative, "FullyAsyncTaskRunner", "_handle_completed_future", ray=ray, logger=Mock())
    cancel = method(relative, "FullyAsyncTaskRunner", "_cancel_all", ray=ray)
    obj._drain_futures = lambda futures: drain(obj, futures)
    obj._handle_completed_future = lambda future, remaining: handle(obj, future, remaining)
    obj._cancel_all = cancel
    fn = method("aura/trainer/train_adapter/verl/full_async/train_main.py",
                "FullyAsyncTaskRunner", "_run_training_loop", ray=ray, logger=Mock())
    with pytest.raises(ValueError, match="expired"):
        fn(obj)
    controller.finish_training.assert_called_once()
