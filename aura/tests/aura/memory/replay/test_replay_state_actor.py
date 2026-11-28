"""Opt-in CPU Ray actor restart + mixer-sidecar roundtrip, not model recovery."""
import os
import pytest
from test_replay_actor_integration import real_ray
from test_replay_mixing import settings

pytestmark=pytest.mark.skipif(os.environ.get('RUN_REPLAY_RAY_TESTS')!='1',reason='CPU Ray opt-in')


@pytest.mark.parametrize('backend',['sqlite','json'])
@pytest.mark.parametrize('strategy',['uniform','priority','reward_weighted'])
def test_actor_restart_preserves_complete_mixer_draw(real_ray,settings,make_group,tmp_path,backend,strategy):
    from aura.memory.episode.episode import Episode
    from aura.memory.replay.mixing import GroupMixer,group_key
    from aura.memory.replay.schema import ReplayGroup
    from aura.memory.replay.state import snapshot_mixer,restore_mixer,write_state_once
    ray=real_ray
    config=settings(batch_groups=2,replay_buffer_size=12,replay_strategy=strategy,replay_ratio=.3,
                    path=str(tmp_path/'store'),backend=backend)
    spec=dict(backend=backend,path=config.path,store_id='store')
    actor=Episode.remote('p4-state-writer',replay_store_config=spec);reader=None
    calls=[]
    def fetch():
        b=len(calls);calls.append(b);samples=[]
        for i in range(2):samples.extend(make_group(group_id=f'b{b}-{i}'))
        receipt=ray.get(actor.append_replay_batch.remote(samples,f'batch-{b}'),timeout=30)
        values=ray.get(actor.get_replay_batch.remote([r[0] for r in receipt.sample_refs]),timeout=30)
        return tuple(ReplayGroup(tuple(values[i:i+2])) for i in range(0,4,2))
    def commit(mixer,version):
        updates={group_key(g):.7 for g in mixer.inflight.groups} if strategy=='priority' else None
        return mixer.commit_success(version,updates)
    try:
        mixer=GroupMixer(config)
        for _ in range(3):mixer.next(fetch,3);commit(mixer,3)
        data=snapshot_mixer(mixer,policy_version=3);path=tmp_path/'state.json';write_state_once(path,data)
        # Deliberate abrupt death, not graceful close. No existing user actor is touched.
        ray.kill(actor,no_restart=True);actor=None
        reader=Episode.remote('p4-state-reader',replay_store_config={**spec,'read_only':True})
        restored=restore_mixer(path.read_bytes(),config,
            lambda tid:ray.get(reader.get_trajectory.remote(tid),timeout=30),expected_policy_version=3)
        assert snapshot_mixer(restored,policy_version=3)==data
        def no_fetch():pytest.fail('pending complete group should satisfy online demand')
        a=mixer.next(no_fetch,4);b=restored.next(no_fetch,4)
        assert [group_key(g) for g in a.groups]==[group_key(g) for g in b.groups]
        assert a.draw_probabilities==b.draw_probabilities
        assert commit(mixer,4)==commit(restored,4)
        assert snapshot_mixer(restored,policy_version=4)==snapshot_mixer(mixer,policy_version=4)
        with pytest.raises(ray.exceptions.RayTaskError):
            ray.get(reader.append_replay_batch.remote(make_group(group_id='forbidden'),'denied'),timeout=30)
    finally:
        for target in (actor,reader):
            if target is not None:
                try:ray.get(target.close_replay_store.remote(),timeout=10)
                finally:ray.kill(target,no_restart=True)
