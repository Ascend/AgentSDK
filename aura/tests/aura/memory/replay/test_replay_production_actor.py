"""Real CPU Ray Dispatch/Episode and native HTTP TrainQueue source recovery."""
import os
from dataclasses import replace
from types import SimpleNamespace

import pytest

from test_replay_actor_integration import real_ray
from test_replay_mixing import settings
from test_replay_reuse import reuse_case

pytestmark = pytest.mark.skipif(os.environ.get('RUN_REPLAY_RAY_TESTS') != '1', reason='Explicit CPU Ray opt-in')


def prompts(config, validate, consumed):
    return iter([{'prompts': [[10+i]], 'prompt_id': [i]} for i in range(6)]), None, None


@pytest.mark.parametrize('backend', ['sqlite', 'json'])
@pytest.mark.parametrize('reuse_enabled', [False, True])
def test_real_dispatch_cursor_datamanager_and_http_queue_roundtrip(real_ray, settings, make_group,
                                                                 tmp_path, monkeypatch, backend, reuse_enabled, reuse_case):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from aura.controllers.train_controller.dispatch_actor import DispatchActor
    from aura.controllers.train_controller.train_controller import TrainController
    from aura.controllers.train_controller.train_server import TrainServer
    from aura.controllers.rollout_controller.rollout_queue import RolloutQueueActor
    from aura.data_manager.verl_data import VerlDataManager
    from aura.memory.episode.episode import Episode
    from aura.memory.replay.runtime import ReplaySession
    ray = real_ray
    if reuse_enabled:
        _, config, _ = reuse_case(backend)
    else:
        config = settings(path=str(tmp_path/'episodes'), backend=backend, batch_groups=2)
    actor = Episode.options(num_cpus=0).remote('source-episode', replay_store_config={
        'backend': backend, 'path': config.path, 'store_id': config.store_id})
    if reuse_enabled:
        from dataclasses import asdict
        ray.get(actor.configure_replay_sources.remote(asdict(config)))
    def dispatch():
        from pathlib import Path
        return DispatchActor.options(name='dispatch', namespace='controller_raygroup', num_cpus=0,
            runtime_env={'env_vars': {'PYTHONPATH': str(Path(__file__).parent)+':'+os.environ.get('PYTHONPATH', '')}}).remote(
            2, 0, 2, 6, SimpleNamespace(seed=42), prompts, 0, False, '127.0.0.1:1')
    producer = dispatch()
    rollout = RolloutQueueActor.options(name='rollout_queue', namespace='controller_raygroup', num_cpus=0).remote()
    server = TrainServer(8, 2, 2)
    server.replay_demand_mode = True
    app = FastAPI(); app.include_router(server.router)
    controller = TrainController.__new__(TrainController)
    controller.dispatch_actor, controller.train_server_addr = producer, 'testserver'
    controller.replay_demand_mode, controller._replay_credit_pending = True, False
    controller.replay_queue_timeout = 20
    manager = VerlDataManager(); manager.controller = controller
    targets = [actor, producer, rollout]
    try:
        source = manager.replay_source_adapter(config, actor)
        with TestClient(app) as client:
            import aura.controllers.train_controller.train_controller as module
            monkeypatch.setattr(module.requests, 'post', lambda url, **kw: client.post(
                url.split('testserver')[-1], params=kw.get('params'), json=kw.get('json')))
            monkeypatch.setattr(module.requests, 'get', lambda url, **kw: client.get(url.split('testserver')[-1]))
            ray.get(producer._get_from_data_iters.remote())
            members = [replace(s, source_batch_id='prefetched-0', source_experiment_id=config.source_experiment_id,
                               source_run_id=config.source_run_id, policy_lineage_id=config.policy_lineage_id) for i in range(2)
                       for s in make_group(group_id=f'group-{i}', version=3)]
            receipt = ray.get(actor.append_replay_batch.remote(members, 'prefetched-0'))
            raw = {'replay_refs': [dict(trajectory_id=t, checksum=h, created_seq=n, source_store_id='store')
                                  for t,h,n in receipt.sample_refs]}
            server.put_minibatch_to_queue(raw, {'rollout_cost': .25})
            controller._replay_credit_pending = True
            saved = source.snapshot()
            assert saved['producer']['state']['cursor']['produced_batches'] == 1
            assert len(saved['queued_batches']) == 1
            source.resume()
            expected = manager.get_data(None, None, None)[0]
            # Recreate the ACTUAL dispatch actor: not a cursor on a synthetic test source.
            ray.kill(producer, no_restart=True)
            import time
            for _ in range(200):
                try: ray.get_actor('dispatch', namespace='controller_raygroup')
                except ValueError: break
                time.sleep(.01)
            producer = dispatch(); targets.append(producer)
            controller.dispatch_actor = producer
            source = manager.replay_source_adapter(config, actor)
            result = source.restore(saved)
            assert result['source_sha256']
            source.resume()
            restored = manager.get_data(None, None, None)[0]
            assert restored == expected
            assert len(ReplaySession(config, 0, actor).load_online(restored)) == 2
            assert ray.get(producer._get_from_data_iters.remote()) == {'prompts': [[11]], 'prompt_id': [1]}
            assert not controller._replay_credit_pending and server.queue_instance.size() == 0
    finally:
        ray.get(actor.close_replay_store.remote())
        for target in targets:
            ray.kill(target, no_restart=True)
