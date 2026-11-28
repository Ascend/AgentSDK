"""P4 sampler primitive only; does not claim model/queue paired recovery."""
from dataclasses import replace
import json
import random

import pytest

from aura.memory.replay.codec import canonical, digest
from aura.memory.replay.json_store import JsonReplayStore
from aura.memory.replay.mixing import GroupMixer, group_key
from aura.memory.replay.schema import ReplayGroup, ReplayIntegrityError
from aura.memory.replay.sqlite_store import SQLiteReplayStore
from aura.memory.replay.state import snapshot_mixer, restore_mixer, write_state_once
from test_replay_mixing import settings


@pytest.fixture
def prepared(settings,make_group,tmp_path):
    def create(strategy='uniform',backend='sqlite'):
        config=settings(batch_groups=2,replay_buffer_size=12,replay_strategy=strategy,
                        replay_ratio=.3,path=str(tmp_path/('source-'+backend)),backend=backend)
        cls=SQLiteReplayStore if backend=='sqlite' else JsonReplayStore
        store=cls(config.path,store_id='store')
        batches=[]
        for b in range(10):
            groups=[]
            for i in range(2):
                samples=make_group(group_id=f'{b}-{i}',reward=(b%3)/2)
                store.append_trajectories(samples,f'{b}-{i}')
                groups.append(ReplayGroup(tuple(store.get_trajectory(s.trajectory_id) for s in samples)))
            batches.append(tuple(groups))
        mixer=GroupMixer(config)
        calls=[]
        def fetch():
            index=len(calls); calls.append(index); return batches[index]
        for _ in range(3):
            result=mixer.next(fetch,3)
            update={group_key(g):float(int(g.samples[0].group_id[0])%3) for g in result.groups} if strategy=='priority' else None
            mixer.commit_success(3,update)
        data=snapshot_mixer(mixer,policy_version=3)
        store.close()
        return config,cls,mixer,data,batches,len(calls)
    return create


@pytest.mark.parametrize('strategy',['uniform','priority','reward_weighted'])
@pytest.mark.parametrize('backend',['sqlite','json'])
def test_reopen_preserves_rng_carry_pending_fifo_and_future_draws(prepared,strategy,backend):
    config,cls,original,data,batches,index=prepared(strategy,backend)
    before=random.getstate()
    with_source=cls(config.path,store_id='store',read_only=True)
    try:
        restored=restore_mixer(data,replace(config,actor_name='new-actor'),with_source.get_trajectory,expected_policy_version=3)
        assert snapshot_mixer(restored,policy_version=3)==data
        assert restored.pending and str(restored.carry)==str(original.carry)
        cursors=[index,index]
        def source(n):
            def fetch():
                value=batches[cursors[n]];cursors[n]+=1;return value
            return fetch
        for version in [4,4,5,5,6,7]:
            a=original.next(source(0),version);b=restored.next(source(1),version)
            assert [group_key(g) for g in a.groups]==[group_key(g) for g in b.groups]
            assert a.draw_probabilities==b.draw_probabilities
            update={group_key(g):.25 for g in a.groups} if strategy=='priority' else None
            assert original.commit_success(version,update)==restored.commit_success(version,update)
            assert snapshot_mixer(original,policy_version=version)==snapshot_mixer(restored,policy_version=version)
        assert cursors[0]==cursors[1] and random.getstate()==before
    finally: with_source.close()


def mutate(data,change):
    envelope=json.loads(data);change(envelope['state']);envelope['checksum']=digest(envelope['state'])
    return canonical(envelope)


@pytest.mark.parametrize('field,value',[
    ('completed_steps',True),('completed_groups',999),('replayed_groups',999),
    ('quarantined_online',-1),('carry','NaN'),('carry','1'),('carry','-0.1'),
    ('rng',[3,[0]*624,None]),('seen_online',[]),('window',{}),('policy_version',True)])
def test_malformed_state_rejected_without_source_mutation(prepared,field,value):
    config,cls,original,data,_,_=prepared()
    store=cls(config.path,read_only=True)
    try:
        with pytest.raises(ReplayIntegrityError):
            restore_mixer(mutate(data,lambda s:s.update({field:value})),config,store.get_trajectory,expected_policy_version=3)
        assert snapshot_mixer(original,policy_version=3)==data
    finally: store.close()


@pytest.mark.parametrize('change',[
    lambda s:s['window'].append(s['window'][0]),
    lambda s:s['pending'].append(s['window'][0]),
    lambda s:s['window'][0]['samples'][0].update(created_seq=999),
    lambda s:s['window'][0]['samples'][0].update(checksum='bad'),
    lambda s:s['window'][0]['samples'][0].update(trajectory_id='missing'),
    lambda s:s.update(priorities=[{'key':s['window'][0]['key'],'value':1,'version':3}]),
])
def test_references_duplicate_priority_or_missing_rejected(prepared,change):
    config,cls,_,data,_,_=prepared();store=cls(config.path,read_only=True)
    try:
        with pytest.raises(ReplayIntegrityError):
            restore_mixer(mutate(data,change),config,store.get_trajectory,expected_policy_version=3)
    finally: store.close()


@pytest.mark.parametrize('change',[
    {'seed':44},{'replay_ratio':.5},{'max_lag':2},{'store_id':'other'},
    {'replay_strategy':'priority'},{'policy_lineage_id':'other'},{'online_prefetch':False}])
def test_contract_changes_rejected(prepared,change):
    config,_,_,data,_,_=prepared()
    with pytest.raises(ReplayIntegrityError):
        restore_mixer(data,replace(config,**change),lambda _:pytest.fail('must fail before read'),expected_policy_version=3)


def test_bad_checksum_and_wrong_model_watermark(prepared):
    config,_,_,data,_,_=prepared()
    with pytest.raises(ReplayIntegrityError): restore_mixer(data+b'x',config,None,expected_policy_version=3)
    with pytest.raises(ReplayIntegrityError): restore_mixer(data,config,None,expected_policy_version=4)
    document=json.loads(data);document['state']['carry']='0.125'
    with pytest.raises(ReplayIntegrityError): restore_mixer(canonical(document),config,None,expected_policy_version=3)


def test_snapshot_refuses_inflight_and_undurable(prepared):
    config,_,mixer,_,batches,index=prepared()
    mixer.next(lambda:batches[index],3)
    with pytest.raises(RuntimeError): snapshot_mixer(mixer,policy_version=3)
    mixer.inflight=None
    key=next(iter(mixer.window));group=mixer.window[key]
    mixer.window[key]=ReplayGroup(tuple(replace(s,created_seq=None) for s in group.samples))
    with pytest.raises(ValueError): snapshot_mixer(mixer,policy_version=3)


def test_atomic_sidecar_no_overwrite_and_incomplete_json_rejected(prepared,tmp_path):
    _,_,_,data,_,_=prepared();target=tmp_path/'state.json'
    write_state_once(target,data);assert target.read_bytes()==data
    with pytest.raises(FileExistsError): write_state_once(target,data)
    assert target.read_bytes()==data and not list(tmp_path.glob('*.tmp'))
    with pytest.raises(ReplayIntegrityError):write_state_once(tmp_path/'broken.json',data[:-1])
    assert not (tmp_path/'broken.json').exists()
    with pytest.raises(ReplayIntegrityError):write_state_once(tmp_path/'not-state.json',b'{}')


@pytest.mark.parametrize('field,value', [('value',float('nan')),('value',-1),('version',99)])
def test_priority_corruption_rejected(prepared,field,value):
    config,cls,_,data,_,_=prepared('priority');store=cls(config.path,read_only=True)
    try:
        if isinstance(value,float):
            # Nonfinite JSON is rejected even before the state decoder.
            broken=json.loads(data);broken['state']['priorities'][0][field]=value
            payload=json.dumps(broken).encode()
        else: payload=mutate(data,lambda s:s['priorities'][0].update({field:value}))
        with pytest.raises(ReplayIntegrityError):restore_mixer(payload,config,store.get_trajectory,expected_policy_version=3)
    finally:store.close()


def test_stale_window_preserved_until_normal_consumption_filter(prepared):
    config,cls,mixer,_,_,_=prepared();data=snapshot_mixer(mixer,policy_version=12)
    store=cls(config.path,read_only=True)
    try:
        restored=restore_mixer(data,config,store.get_trajectory,expected_policy_version=12)
        assert len(restored.window)==len(mixer.window)>0
        assert snapshot_mixer(restored,policy_version=12)==data
    finally:store.close()


def test_failed_publication_does_not_create_destination(prepared,tmp_path,monkeypatch):
    import os
    _,_,_,data,_,_=prepared();target=tmp_path/'state.json'
    def failure(*args):raise OSError('injected link failure')
    monkeypatch.setattr(os,'link',failure)
    with pytest.raises(OSError):write_state_once(target,data)
    assert not target.exists() and not list(tmp_path.glob('*.tmp'))


def test_snapshot_does_not_repeat_218112_historical_group_ids(prepared):
    config, cls, mixer, _, _, _ = prepared()
    baseline = snapshot_mixer(mixer, policy_version=3)
    mixer.seen_online.update(('store', 'archived-' + str(i).zfill(64)) for i in range(218112))
    assert snapshot_mixer(mixer, policy_version=3) == baseline
    assert len(json.loads(baseline)['state']['seen_online']) <= (
        config.replay_buffer_size // config.group_size + 2*config.batch_groups)
