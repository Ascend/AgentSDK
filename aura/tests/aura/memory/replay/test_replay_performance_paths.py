"""Integrity/compatibility regressions for optimized persistence and batch reads."""
from dataclasses import replace
import json
import struct
import sys
from types import SimpleNamespace
import pytest

from aura.memory.replay.codec import ReplayCodec
from aura.memory.replay.config import ReplayConfig
from aura.memory.replay.json_store import JsonReplayStore
from aura.memory.replay.sqlite_store import SQLiteReplayStore
from aura.memory.replay.schema import ReplayIntegrityError
from aura.memory.replay.store_common import prepare_batch


def test_prepare_exact_canonical_bytes_and_metadata_ownership(make_group):
    sample=replace(make_group()[0],created_seq=99,group_complete=True)
    snapshot,data,checksum=ReplayCodec.prepare(sample)
    assert data==ReplayCodec.encode(sample)
    assert checksum==json.loads(data)['checksum']
    assert snapshot==ReplayCodec.decode(data)
    assert snapshot.created_seq is snapshot.group_complete is None
    assert sample.created_seq==99 and sample.group_complete is True
    assert snapshot.payload['responses'] is not sample.payload['responses']
    assert snapshot.old_log_probs.data==sample.old_log_probs.data


@pytest.mark.parametrize('mutation',['nan','dtype','shape','negative_tokens','boolean_version'])
def test_prepare_revalidates_forged_frozen_objects(make_group,mutation):
    sample=make_group()[0]
    if mutation=='nan':object.__setattr__(sample.old_log_probs,'data',struct.pack('<fff',float('nan'),0,-.5))
    elif mutation=='dtype':object.__setattr__(sample.old_log_probs,'dtype','object')
    elif mutation=='shape':object.__setattr__(sample.payload['responses'],'shape',(100,))
    elif mutation=='negative_tokens':object.__setattr__(sample.payload['prompts'],'data',struct.pack('<qq',-1,3))
    else:object.__setattr__(sample,'weight_version',True)
    with pytest.raises((ValueError,TypeError)):prepare_batch([sample],'store')


@pytest.mark.parametrize('backend',[SQLiteReplayStore,JsonReplayStore])
def test_batch_read_order_missing_source_bounds_and_reopen(tmp_path,make_group,backend):
    path=tmp_path/'store';samples=make_group();ids=[s.trajectory_id for s in samples]
    store=backend(path,store_id='store')
    store.append_trajectories(samples,'request')
    result=store.get_trajectories([ids[1],'missing',ids[0]])
    assert [s.trajectory_id if s else None for s in result]==[ids[1],None,ids[0]]
    assert all(s.group_complete for s in result if s)
    assert store.get_trajectories(ids,'another-store')==[None,None]
    for values in ([],[ids[0],ids[0]],[''] ,list(map(str,range(257)))):
        with pytest.raises(ValueError):store.get_trajectories(values)
    store.close();reader=backend(path,read_only=True)
    assert reader.get_trajectories(ids)[0].old_log_probs.data==samples[0].old_log_probs.data
    reader.close()


def test_json_point_read_does_not_call_history_query(tmp_path,make_group,monkeypatch):
    store=JsonReplayStore(tmp_path/'store',store_id='store')
    samples=make_group();store.append_trajectories(samples,'r')
    def forbidden(*args,**kwargs):raise AssertionError('Point lookup must not scan/query full history')
    monkeypatch.setattr(store,'query_page',forbidden)
    assert store.get_trajectory(samples[0].trajectory_id).created_seq==1
    assert store.get_trajectories([samples[1].trajectory_id])[0].created_seq==2
    store.close()


def test_derived_compatibility_cache_preserves_full_filter_and_checksum(tmp_path,make_group):
    from aura.memory.replay.mixing import compatible
    from aura.memory.replay.schema import ReplayGroup
    sample=make_group()[0]
    settings=ReplayConfig(replay_ratio=.3,replay_buffer_size=64,group_size=2,batch_groups=2,max_lag=8,
        seed=42,store_id=sample.source_store_id,source_experiment_id=sample.source_experiment_id,
        source_run_id=sample.source_run_id,policy_lineage_id=sample.policy_lineage_id,
        model_signature=sample.model_signature,tokenizer_signature=sample.tokenizer_signature,
        task_signature=sample.task_signature,reward_signature=sample.reward_signature,path=str(tmp_path/'store'))
    group=ReplayGroup(tuple(make_group()))
    assert compatible(group,settings,3) and compatible(group,settings,10)
    assert not compatible(group,settings,2) and not compatible(group,settings,11)
    for name in ('store_id','source_run_id','source_experiment_id','policy_lineage_id','model_signature',
                 'tokenizer_signature','task_signature','reward_signature'):
        assert not compatible(group,replace(settings,**{name:'different'}),3)
    assert not compatible(group,replace(settings,group_size=3),3)
    encoded=ReplayCodec.encode(sample)
    object.__setattr__(sample,'_compatibility_key',('forged-cache',))
    snapshot,data,_=ReplayCodec.prepare(sample)
    assert data==encoded and snapshot._compatibility_key!=sample._compatibility_key


def test_store_metadata_projection_owns_new_object_and_validates_metadata(make_group):
    from aura.memory.replay.store_common import stored_sample
    entry=prepare_batch(make_group(),'store')[0][0]
    entry=replace(entry,seq=7)
    view=stored_sample(entry,True)
    assert view is not entry.sample and view.created_seq==7 and view.group_complete is True
    assert entry.sample.created_seq is entry.sample.group_complete is None
    assert ReplayCodec.encode(view)==entry.encoded
    with pytest.raises(ValueError):stored_sample(replace(entry,seq=-1),True)
    with pytest.raises(ValueError):stored_sample(entry,1)


@pytest.mark.parametrize('bad_length',[False,True])
def test_runtime_bounded_batch_rpc_and_short_response_rejection(tmp_path,make_group,monkeypatch,bad_length):
    from aura.memory.replay.runtime import ReplaySession
    first=make_group()[0]
    config=ReplayConfig(replay_ratio=.3,replay_buffer_size=1000,group_size=2,batch_groups=130,max_lag=8,
        seed=42,store_id='store',source_experiment_id=first.source_experiment_id,source_run_id=first.source_run_id,
        policy_lineage_id=first.policy_lineage_id,model_signature=first.model_signature,
        tokenizer_signature=first.tokenizer_signature,task_signature=first.task_signature,
        reward_signature=first.reward_signature,path=str(tmp_path/'store'))
    samples=[replace(s,created_seq=2*i+s.sample_index+1,group_complete=True)
             for i in range(130) for s in make_group(group_id=f'g{i}')]
    by_id={s.trajectory_id:s for s in samples};calls=[]
    def load(ids):
        calls.append(len(ids));return [] if bad_length else [by_id[x] for x in ids]
    actor=SimpleNamespace(get_replay_batch=SimpleNamespace(remote=load))
    monkeypatch.setitem(sys.modules,'ray',SimpleNamespace(get=lambda value,timeout:value))
    session=ReplaySession(config,0,actor)
    refs=[{'trajectory_id':s.trajectory_id,'created_seq':s.created_seq,
           'checksum':s.payload_checksum,'source_store_id':'store'} for s in samples]
    if bad_length:
        with pytest.raises(ReplayIntegrityError):session.load_online({'replay_refs':refs})
    else:
        assert len(session.load_online({'replay_refs':refs}))==130
        assert calls==[256,4]
