"""Bounded prefetch, actual production HTTP notifications and mixer contracts."""
import asyncio
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from unittest.mock import Mock
import io
import json
import time

import pytest

from test_replay_backpressure import method
from test_replay_mixing import settings, group_batch
from aura.memory.replay.mixing import GroupMixer, ReplayUnavailableError, group_key

CONTROLLER = "aura/controllers/train_controller/train_controller.py"


@pytest.mark.parametrize("ratio", [0.3, 1.0])
def test_prefetch_preserves_draws_and_pending_bound(settings, make_group, ratio):
    mixers = [GroupMixer(settings(batch_groups=2, replay_ratio=ratio,
                                 warmup_online_steps=2, online_prefetch=enabled))
              for enabled in (True, False)]
    calls = [0, 0]
    reserved = []
    credits = 0

    def produce(index):
        calls[index] += 1
        return group_batch(make_group, str(calls[index]), 2)

    def fetch(index):
        if index == 0 and reserved:
            return reserved.pop()
        return produce(index)

    for step in range(20):
        results = [m.next(lambda i=i: fetch(i), 3) for i, m in enumerate(mixers)]
        assert results[0].groups == results[1].groups
        assert results[0].draw_probabilities == results[1].draw_probabilities
        rng = mixers[0].rng.getstate()
        carry = mixers[0].carry
        if mixers[0].wants_online_prefetch(last_step=step == 19) and not reserved:
            reserved.append(produce(0))
            credits += 1
        assert len(mixers[0].pending) + 2 * len(reserved) <= 4
        assert mixers[0].rng.getstate() == rng and mixers[0].carry == carry
        for mixer in mixers:
            mixer.commit_success(3)
    assert calls[0] == calls[1] and not reserved
    assert (credits > 0) == (ratio == 0.3)
    if ratio == 1:
        assert calls == [2, 2]


@pytest.mark.parametrize("kind", ["shortage", "expired"])
def test_mixed_fallback_still_whole_and_fresh(settings, make_group, kind):
    mixer = GroupMixer(settings(batch_groups=10))
    groups = group_batch(make_group, "past", 2, version=3)
    mixer.window.update((group_key(g), g) for g in groups)
    version = 11 if kind == "expired" else 4
    result = mixer.next(lambda: group_batch(make_group, "fresh", 10, version), version)
    assert result.metrics["replay/actual_groups"] == (0 if kind == "expired" else 2)
    assert len(result.groups) == 10 and all(g.complete for g in result.groups)
    assert len({group_key(g) for g in result.groups}) == 10
    assert mixer.wants_online_prefetch()
    mixer.commit_success(version)


@pytest.mark.parametrize("kind", ["shortage", "expired"])
def test_pure_negative_never_prefetches_or_falls_back(settings, make_group, kind):
    mixer = GroupMixer(settings(batch_groups=2, replay_ratio=1))
    groups = group_batch(make_group, "past", 1 if kind == "shortage" else 2)
    mixer.window.update((group_key(g), g) for g in groups)
    fetch = Mock(side_effect=AssertionError("No online fallback"))
    rng = mixer.rng.getstate()
    with pytest.raises(ReplayUnavailableError):
        mixer.next(fetch, 11 if kind == "expired" else 3)
    fetch.assert_not_called()
    assert not mixer.wants_online_prefetch() and mixer.rng.getstate() == rng


def test_stale_prefetched_payload_cannot_train(settings, make_group):
    mixer = GroupMixer(settings(batch_groups=2))
    with pytest.raises(ReplayUnavailableError):
        mixer.next(lambda: group_batch(make_group, "late", 2), 11)
    assert mixer.inflight is None and mixer.completed_steps == 0 and not mixer.window


@pytest.mark.parametrize("changes", [{"online_prefetch": False}, {"max_lag": 1}])
def test_prefetch_can_be_disabled_and_never_advances_final_step(settings, make_group, changes):
    mixer = GroupMixer(settings(batch_groups=2, **changes))
    mixer.next(lambda: group_batch(make_group, "a", 2), 3)
    assert not mixer.wants_online_prefetch()
    assert not mixer.wants_online_prefetch(last_step=True)


def controller():
    ray = Mock()
    ray.get.side_effect = lambda ref, **kw: ref
    obj = SimpleNamespace(replay_demand_mode=True, _replay_credit_pending=False,
                          replay_queue_timeout=3, dispatch_actor=Mock())
    obj.dispatch_actor.send_batch_groups.remote.return_value = 1
    request = method(CONTROLLER, "TrainController", "request_online_batch", ray=ray)
    obj.request_online_batch = lambda: request(obj)
    return obj, ray


def test_one_credit_until_successful_dequeue_then_replenish():
    obj, _ = controller()
    assert obj.request_online_batch()
    assert not obj.request_online_batch()
    obj.wait_for_online_batch = Mock()
    obj.training_batch_queue_ready = Mock(side_effect=AssertionError("No poll"))
    obj.train_server_addr = "localhost:1"
    requests, torch = Mock(), Mock()
    requests.get.return_value.headers = {}
    requests.get.return_value.content = b"batch"
    torch.load.return_value = {"real": True}
    fetch = method(CONTROLLER, "TrainController", "get_next_training_batch",
                   logger=Mock(), time=time, requests=requests, torch=torch,
                   io=io, json=json, DEFAULT_URL_METHOD="http", DEFAULT_SLEEP_TIME=2)
    assert fetch(obj)[0] == {"real": True}
    assert not obj._replay_credit_pending
    assert obj.request_online_batch()
    assert obj.dispatch_actor.send_batch_groups.remote.call_count == 2


def test_ambiguous_send_failure_keeps_credit_no_retry():
    obj, ray = controller()
    ray.get.side_effect = TimeoutError("ambiguous dispatch")
    with pytest.raises(TimeoutError):
        obj.request_online_batch()
    assert obj._replay_credit_pending and not obj.request_online_batch()
    assert obj.dispatch_actor.send_batch_groups.remote.call_count == 1


def test_zero_path_does_not_construct_prefetch_or_change_fetch():
    fetch = Mock(return_value=({"unchanged": True}, []))
    dm = SimpleNamespace(get_data=fetch, prefetch_online_batch=Mock(side_effect=AssertionError()))
    obj = SimpleNamespace(required_samples=4, data_manager=dm, _replay_session=None, sample_queue=None)
    fn = method("aura/trainer/train_adapter/verl/full_async/full_async_trainer.py",
                "FullyAsyncTrainer", "_get_samples_from_queue", logger=Mock(), Any=object)
    assert fn(obj) == (0, {"unchanged": True})
    fetch.assert_called_once_with(experience_consumer_stage="train", experience_columns=None,
                                  experience_count=4)
    obj2, _ = controller()
    obj2.replay_demand_mode = False
    assert not obj2.request_online_batch()
    obj2.dispatch_actor.send_batch_groups.remote.assert_not_called()


def test_real_http_wait_is_not_blocking_receive_and_no_lost_ready(monkeypatch):
    import ray
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from aura.controllers.train_controller.train_server import TrainServer
    monkeypatch.setattr(ray, "get_actor", lambda *a, **kw: Mock())
    server = TrainServer(2, 2, 2)
    server.replay_demand_mode = True
    app = FastAPI()
    app.include_router(server.router)

    @app.post("/test/arrive")
    async def arrive():
        server.put_minibatch_to_queue({"complete_groups": 2}, {})
        return {"ok": True}

    with TestClient(app) as client, ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(client.post, "/train/wait_batch_ready?timeout=2")
        client.post("/test/arrive").raise_for_status()
        assert future.result(timeout=3).json() == {"is_ready": True}
        assert client.post("/train/wait_batch_ready?timeout=0.01").json()["is_ready"]
        server.queue_instance.pop_batch()
        assert not client.post("/train/wait_batch_ready?timeout=0.01").json()["is_ready"]
        assert client.post("/train/wait_batch_ready?timeout=31").status_code == 400
    server.dispatch_actor.send_batch_groups.remote.assert_not_called()
