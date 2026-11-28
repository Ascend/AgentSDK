"""Opt-in real Ray transport test for the bounded batch read capability."""
import os
import pytest
from test_replay_actor_integration import real_ray

pytestmark=pytest.mark.skipif(os.environ.get('RUN_REPLAY_RAY_TESTS')!='1',reason='Explicit real Ray opt-in')


@pytest.mark.parametrize('backend',['sqlite','json'])
def test_real_actor_order_missing_and_bounds(real_ray,tmp_path,make_group,backend):
    from aura.memory.episode.episode import Episode
    actor=Episode.remote('batch-read',replay_store_config={'backend':backend,'path':str(tmp_path/'store'),'store_id':'store'})
    samples=make_group();ids=[s.trajectory_id for s in samples]
    try:
        real_ray.get(actor.append_replay_batch.remote(samples,'r'))
        values=real_ray.get(actor.get_replay_batch.remote([ids[1],'missing',ids[0]]))
        assert [x.trajectory_id if x else None for x in values]==[ids[1],None,ids[0]]
        assert values[0].old_log_probs.data==samples[1].old_log_probs.data
        with pytest.raises(real_ray.exceptions.RayTaskError):
            real_ray.get(actor.get_replay_batch.remote([ids[0],ids[0]]))
    finally:
        real_ray.get(actor.close_replay_store.remote());real_ray.kill(actor,no_restart=True)


def controlled_config(tmp_path,sample,batch_groups):
    from aura.memory.replay.config import ReplayConfig
    return ReplayConfig(replay_ratio=1,replay_buffer_size=64,group_size=2,batch_groups=batch_groups,
        max_lag=2 if batch_groups==2 else 3,seed=42,store_id=sample.source_store_id,
        source_experiment_id=sample.source_experiment_id,source_run_id=sample.source_run_id,
        policy_lineage_id=sample.policy_lineage_id,model_signature=sample.model_signature,
        tokenizer_signature=sample.tokenizer_signature,task_signature=sample.task_signature,
        reward_signature=sample.reward_signature,path=str(tmp_path/'store'),warmup_online_steps=1)


@pytest.mark.parametrize('batch_groups',[1,2])
def test_actual_actor_single_candidate_with_legal_capacity(real_ray,tmp_path,make_group,batch_groups):
    """Controlled version fixtures through real service; not an NPU training claim."""
    from aura.memory.episode.episode import Episode
    from aura.memory.replay.runtime import ReplaySession
    from aura.memory.replay.mixing import ReplayUnavailableError
    samples=make_group(group_id='old',version=0)
    if batch_groups==2:samples+=make_group(group_id='recent',version=1)
    from dataclasses import replace
    # The shared storage fixture is not a complete training payload; add its
    # explicit outcome reward tensor rather than weakening training validation.
    samples=[replace(s,payload={**s.payload,'rm_scores':s.payload['token_level_rewards']}) for s in samples]
    config=controlled_config(tmp_path,samples[0],batch_groups)
    actor=Episode.remote('single-candidate',replay_store_config={'backend':'sqlite','path':config.path,'store_id':'store'})
    calls=[]
    try:
        receipt=real_ray.get(actor.append_replay_batch.remote(samples,'warmup'))
        refs=[{'trajectory_id':key,'checksum':checksum,'created_seq':seq,'source_store_id':'store'}
              for key,checksum,seq in receipt.sample_refs]
        def fetch():calls.append(1);return {'replay_refs':refs}
        session=ReplaySession(config,0,actor)
        start_version=1 if batch_groups==2 else 0
        session.next_batch(fetch,start_version);session.commit_success(start_version)
        if batch_groups==1:
            for version in (1,2):
                batch=session.next_batch(fetch,version)
                assert not session.mixer.inflight.online_groups
                assert session.mixer.inflight.draw_probabilities == (1.,)
                session.commit_success(version)
            failed_version=3
        else:
            # At version2, version0 is expired and exactly one version1 group remains.
            assert sum(0<=2-g.samples[0].weight_version<2 for g in session.mixer.window.values())==1
            failed_version=2
        state=session.mixer.rng.getstate();steps=session.mixer.completed_steps
        with pytest.raises(ReplayUnavailableError):session.next_batch(fetch,failed_version)
        assert session.mixer.rng.getstate()==state and session.mixer.completed_steps==steps
        assert session.mixer.inflight is None and len(calls)==1
        assert len(real_ray.get(actor.query_replay.remote()))==len(samples)
    finally:
        real_ray.get(actor.close_replay_store.remote());real_ray.kill(actor,no_restart=True)
