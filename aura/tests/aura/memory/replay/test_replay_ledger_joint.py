"""Server CPU-only P4 protocol and fault matrix. Tiny artifacts, no NPU model."""
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path

import pytest

from aura.memory.replay.codec import canonical, digest
from aura.memory.replay.joint import publish_commit, read_commit
from aura.memory.replay.ledger import ReplayLedger, ReplayFencedError, batch_document
from aura.memory.replay.mixing import GroupMixer, group_key
from aura.memory.replay.schema import ReplayConflictError, ReplayGroup, ReplayIntegrityError
from aura.memory.replay.sqlite_store import SQLiteReplayStore
from aura.memory.replay.json_store import JsonReplayStore
from aura.memory.replay.state import snapshot_mixer, restore_mixer
from test_replay_mixing import settings


class Crash(RuntimeError):pass


def record_evidence(name,value):
    directory=os.environ.get('P4_PROTOCOL_EVIDENCE_ROOT')
    if directory:
        root=Path(directory)
        if not root.is_absolute():raise ValueError('Evidence root must be absolute')
        root.mkdir(parents=True,exist_ok=True)
        with (root/(name+'.json')).open('xb') as stream:stream.write(canonical(value))


class Harness:
    def __init__(self,config,make_group,root):
        self.config,self.make_group,self.root=config,make_group,root
        self.store=(SQLiteReplayStore if config.backend=='sqlite' else JsonReplayStore)(config.path,store_id='store')
        self.ledger=ReplayLedger(root/'ledger.db',config,self.store.get_trajectory)
        self.writer=self.ledger.acquire_writer('first-owner');self.branch='first'
        self.ledger.initial_branch(self.branch,self.writer)
        self.mixer=GroupMixer(config);self.version=0;self.cursor=0;self.value=0;self.saves=0
        self.bind_history()

    def bind_history(self):
        self.mixer.history_contains = lambda keys: self.ledger.seen_online(
            self.branch, self.mixer.completed_steps, keys, self.writer)

    def source_state(self):
        return {'format':'aura.replay.source-state.v1','quiescent':True,'prefetch':None,
                'producer':{'identity':digest('frozen-unit-source'),'state':{'next_batch':self.cursor}},'queued_batches':[]}

    def fetch(self,fail=False):
        number=self.cursor;self.cursor+=1;groups=[]
        for i in range(self.config.batch_groups):
            samples=self.make_group(group_id=f'b{number}-{i}',version=self.version,reward=(number+i)%2)
            self.store.append_trajectories(samples,f'batch-{number}-{i}')
            groups.append(ReplayGroup(tuple(self.store.get_trajectory(s.trajectory_id) for s in samples)))
        if fail:raise Crash('after_persist_before_enqueue')
        return tuple(groups)

    def save(self,step,version):
        folder=self.root/'models'/f'{self.branch}-{step}-{self.saves}';self.saves+=1;folder.mkdir(parents=True)
        artifacts={}
        for role in ('model','optimizer','scheduler','rng'):
            path=folder/(role+'.json')
            path.write_bytes(canonical({'role':role,'step':step,'policy_version':version,'value':self.value}))
            artifacts[role]=[path.relative_to(self.root).as_posix()]
        return dict(step=step,policy_version=version,artifacts=artifacts)

    def checkpoint(self,commit,stage=None):
        def hit(name):
            if name==stage:raise Crash(name)
        return publish_commit(self.root,self.config,self.store.get_trajectory,self.ledger,self.writer,self.branch,
            snapshot_mixer(self.mixer,policy_version=self.version),self.source_state(),self.save,commit_id=commit,failpoint=hit)

    def update(self,stage=None):
        self.mixer.next(lambda:self.fetch(stage=='after_persist_before_enqueue'),self.version)
        if stage=='after_plan':raise Crash(stage)
        step=self.mixer.completed_steps+1;facts=batch_document(self.mixer)
        self.ledger.enqueue(self.branch,step,self.version,facts,f'update-{step}',self.writer)
        if stage=='after_enqueue':raise Crash(stage)
        receipt=self.ledger.lease(self.branch,step,f'lease-{step}',self.writer)
        if stage=='after_lease':raise Crash(stage)
        self.value+=1
        if stage=='after_update_before_ack':raise Crash(stage)
        updates={group_key(g):float(g.samples[0].reward) for g in self.mixer.inflight.groups} if self.config.replay_strategy=='priority' else None
        self.mixer.commit_success(self.version,updates);self.version+=1
        state=snapshot_mixer(self.mixer,policy_version=self.version)
        self.ledger.ack(receipt['lease_id'],state,self.writer)
        if stage=='after_ack':raise Crash(stage)
        return facts,receipt,state

    def recover(self,branch='recovered',commit=None):
        self.writer=self.ledger.acquire_writer('owner-'+branch)
        selected=self.ledger.recover_branch(str(self.root),self.branch,branch,self.writer,commit)
        self.branch=branch;self.version=selected['body']['policy_version']
        self.mixer=restore_mixer(selected['snapshot'],self.config,self.store.get_trajectory,expected_policy_version=self.version)
        self.bind_history()
        self.cursor=selected['source']['producer']['state']['next_batch']
        model=next(a for a in selected['body']['artifacts'] if 'model' in a['roles'])
        self.value=json.loads((self.root/model['path']).read_bytes())['value']
        return selected

    def close(self):self.ledger.close();self.store.close()


def test_compact_history_preserves_evicted_duplicates_and_recovery_cut(harness):
    h = harness(ratio=0.3, name='compact-history')
    h.checkpoint('genesis')
    h.update()
    first = tuple(h.mixer.window.values())
    sizes = []
    for _ in range(25):
        h.update()
        sizes.append(len(snapshot_mixer(h.mixer, policy_version=h.version)))
    assert group_key(first[0]) not in h.mixer.window
    assert h.ledger.seen_online(h.branch, h.mixer.completed_steps, [group_key(first[0])], h.writer)
    assert len(h.mixer.seen_online) <= h.config.replay_buffer_size // h.config.group_size + 2*h.config.batch_groups
    assert max(sizes[-10:]) - min(sizes[-10:]) < 2048
    source = h.source_state()
    source['queued_batches'] = [{'batch_id': 'repeated-old', 'weight_version': 0, 'replay_refs': [
        {'source_store_id': s.source_store_id, 'trajectory_id': s.trajectory_id,
         'checksum': s.payload_checksum, 'created_seq': s.created_seq} for g in first for s in g.samples]}]
    with pytest.raises(ReplayIntegrityError, match='previously fetched'):
        publish_commit(h.root, h.config, h.store.get_trajectory, h.ledger, h.writer, h.branch,
                       snapshot_mixer(h.mixer, policy_version=h.version), source, h.save, commit_id='bad-queue')
    h.checkpoint('cut')
    cut_step = h.mixer.completed_steps
    for _ in range(3): h.update()
    future = h.ledger.conn.execute('SELECT source_id,group_id FROM p4_online_history WHERE branch=? AND step>?',
                                  (h.branch, cut_step)).fetchall()
    assert future
    h.ledger.close()
    h.ledger = ReplayLedger(h.root/'ledger.db', h.config, h.store.get_trajectory)
    h.recover('after-cut', 'cut')
    assert h.ledger.seen_online(h.branch, cut_step, [group_key(first[0])], h.writer)
    assert not h.ledger.seen_online(h.branch, cut_step, [tuple(row) for row in future], h.writer)
    # Exercise the restored mixer's actual duplicate guard, not only the SQL query.
    h.mixer.pending.clear()
    with pytest.raises(ValueError, match='repeated online generation'):
        h.mixer.next(lambda: first, h.version)


@pytest.fixture
def harness(settings,make_group,tmp_path):
    opened=[]
    def create(strategy='uniform',backend='sqlite',ratio=.3,name='case'):
        root=tmp_path/name;root.mkdir()
        config=settings(path=str(root/'trajectories'),batch_groups=2,replay_buffer_size=12,
                        replay_strategy=strategy,backend=backend,replay_ratio=ratio,warmup_online_steps=1)
        value=Harness(config,make_group,root);opened.append(value);return value
    yield create
    for h in reversed(opened):h.close()


@pytest.mark.parametrize('strategy',['uniform','priority','reward_weighted'])
@pytest.mark.parametrize('backend',['sqlite','json'])
def test_paired_recovery_ignores_newer_ack_and_preserves_draws(harness,strategy,backend):
    h=harness(strategy,backend);h.checkpoint('genesis');h.update();h.checkpoint('cut1')
    expected2=h.update();expected3=h.update();old_writer=h.writer
    assert h.ledger.inspect('first')['branch']['head_step']==3
    chosen=h.recover();assert chosen['body']['step']==1 and h.value==1
    assert h.ledger.inspect('first')['branch']['head_step']==3
    with pytest.raises(ReplayFencedError):h.ledger.lease('first',4,'stale',old_writer)
    actual2=h.update();actual3=h.update()
    for a,b in ((expected2,actual2),(expected3,actual3)):
        assert a[0]==b[0] and a[2]==b[2]
    assert h.value==3 and h.ledger.inspect('recovered')['counts']=={'acked':2}


UPDATE_FAILURES=['after_persist_before_enqueue','after_plan','after_enqueue','after_lease','after_update_before_ack','after_ack']
SAVE_FAILURES=['after_model_save','after_prepare','after_replay_sidecar','after_source_sidecar','before_marker','after_marker','after_seal']


@pytest.mark.parametrize('stage',UPDATE_FAILURES+SAVE_FAILURES)
def test_fault_matrix_restores_only_complete_paired_cut(harness,stage):
    h=harness();h.checkpoint('genesis')
    with pytest.raises(Crash):
        if stage in UPDATE_FAILURES:h.update(stage)
        else:h.update();h.checkpoint('cut1',stage)
    history=h.ledger.inspect('first');chosen=h.recover()
    expected=1 if stage in ('after_marker','after_seal') else 0
    assert chosen['body']['step']==h.value==h.mixer.completed_steps==expected
    assert h.ledger.inspect('first')==history
    assert h.ledger.inspect('recovered')['branch']['head_step']==expected
    h.update();assert h.value==expected+1
    record_evidence('fault-'+stage,dict(stage=stage,selected_step=expected,old_branch=history,
        first_update_after_recovery=expected+1,original_history_preserved=True))


def test_rpc_retries_idempotent_and_conflicts_not_reapplied(harness):
    h=harness();h.checkpoint('genesis');facts,receipt,state=h.update()
    assert h.ledger.acquire_writer('first-owner')==h.writer
    assert h.ledger.enqueue('first',1,0,facts,'update-1',h.writer)['status']=='acked'
    assert h.ledger.lease('first',1,'lease-1',h.writer)['lease_id']==receipt['lease_id']
    assert h.ledger.ack(receipt['lease_id'],state,h.writer)['status']=='acked'
    assert h.ledger.inspect('first')['branch']['head_step']==1
    with pytest.raises(ReplayConflictError):h.ledger.enqueue('first',1,0,facts,'other',h.writer)
    with pytest.raises(ReplayConflictError):h.ledger.lease('first',1,'other',h.writer)


def test_no_ttl_or_new_writer_reclaims_uncertain_lease(harness):
    h=harness();h.checkpoint('genesis')
    with pytest.raises(Crash):h.update('after_lease')
    writer=h.ledger.acquire_writer('second-owner')
    with pytest.raises(ReplayFencedError):h.ledger.lease('first',1,'lease-1',writer)
    with pytest.raises(ReplayFencedError):h.ledger.acquire_writer('first-owner')
    assert h.ledger.inspect('first')['counts']=={'leased':1}


def test_initial_model_only_or_ack_only_is_not_a_recovery_point(harness):
    h=harness();h.mixer.next(h.fetch,0)
    with pytest.raises(RuntimeError):h.ledger.enqueue('first',1,0,batch_document(h.mixer),'u',h.writer)
    (h.root/'latest_checkpointed_iteration.txt').write_text('999')
    with pytest.raises(ReplayIntegrityError):h.recover()


@pytest.mark.parametrize('target',['model','optimizer','scheduler','rng','mixer','source','marker'])
def test_corrupt_latest_pair_falls_back_but_explicit_path_fails(harness,target):
    h=harness();h.checkpoint('genesis');h.update();cut=h.checkpoint('cut1')
    if target in ('mixer','source','marker'):
        name={'mixer':'mixer.json','source':'source.json','marker':'COMMITTED.json'}[target]
        path=h.root/'replay-commits'/'cut1'/name
    else:path=h.root/next(a['path'] for a in cut['body']['artifacts'] if target in a['roles'])
    path.write_bytes(path.read_bytes()+b'x')
    with pytest.raises((ReplayIntegrityError,ValueError)):h.recover('bad-explicit','cut1')
    selected=h.recover('fallback');assert selected['body']['step']==0
    assert any(r['commit_id']=='cut1' for r in selected['rejected'])


def test_unknown_lineage_or_missing_model_role_rejected(harness):
    h=harness();h.checkpoint('genesis');other=replace(h.config,policy_lineage_id='unrelated')
    with pytest.raises(ReplayIntegrityError):read_commit(h.root,'genesis',other,h.store.get_trajectory,h.ledger.lookup_commit)
    h.update()
    def bad_save(step,version):
        r=h.save(step,version);del r['artifacts']['optimizer'];return r
    with pytest.raises(ReplayIntegrityError):publish_commit(h.root,h.config,h.store.get_trajectory,h.ledger,h.writer,'first',
        snapshot_mixer(h.mixer,policy_version=1),h.source_state(),bad_save,commit_id='bad')
    assert not (h.root/'replay-commits'/'bad'/'COMMITTED.json').exists()


def test_source_inflight_or_wrong_cursor_receipt_not_accepted(harness):
    from aura.memory.replay.joint import validate_source
    h=harness();state=h.source_state();state['quiescent']=False
    with pytest.raises(ReplayIntegrityError):validate_source(state,h.config,h.store.get_trajectory,0,set())
    state=h.source_state();state['prefetch']={'not_drained':True}
    with pytest.raises(ReplayIntegrityError):validate_source(state,h.config,h.store.get_trajectory,0,set())


def test_unsafe_artifacts_and_symlinks_rejected(harness,tmp_path):
    from aura.memory.replay.joint import file_info
    h=harness();outside=tmp_path/'outside';outside.write_bytes(b'x')
    for relative in ('../outside','/etc/passwd','a/../outside','x\\y'):
        with pytest.raises(ReplayIntegrityError):file_info(h.root,relative)
    (h.root/'alias').symlink_to(outside)
    with pytest.raises(ReplayIntegrityError):file_info(h.root,'alias')


def test_reopened_ledger_fences_old_connection_and_keeps_history(harness):
    h=harness();h.checkpoint('genesis');h.update();h.checkpoint('cut1')
    second=ReplayLedger(h.root/'ledger.db',h.config,h.store.get_trajectory)
    try:
        new=second.acquire_writer('new-process')
        with pytest.raises(ReplayFencedError):h.ledger.initial_branch('first',h.writer)
        cut=second.recover_branch(h.root,'first','new-process-branch',new)
        assert cut['body']['step']==1 and h.ledger.inspect('first')['counts']=={'acked':1}
    finally:second.close()


@pytest.mark.parametrize('field',['carry','rng','window','pending'])
def test_ack_cannot_substitute_a_different_valid_snapshot(harness,field):
    h=harness();h.checkpoint('genesis')
    h.mixer.next(h.fetch,0);facts=batch_document(h.mixer)
    h.ledger.enqueue('first',1,0,facts,'u',h.writer);lease=h.ledger.lease('first',1,'l',h.writer)
    h.mixer.commit_success(0);snapshot=snapshot_mixer(h.mixer,policy_version=1)
    obj=json.loads(snapshot);state=obj['state']
    if field=='carry':state['carry']='0.25'
    elif field=='rng':state['rng'][1][0]^=1
    elif field=='window':state['window'].reverse()
    else:
        # Reassign consumed groups to pending while preserving schema counters.
        state['pending']=state['window'][:1];state['window']=state['window'][1:]
    obj['checksum']=digest(state)
    with pytest.raises(ReplayIntegrityError):h.ledger.ack(lease['lease_id'],canonical(obj),h.writer)
    assert h.ledger.inspect('first')['counts']=={'leased':1}


def test_marker_before_sql_seal_is_committed_but_prepare_without_marker_is_not(harness):
    h=harness();h.checkpoint('genesis');h.update()
    def crash():raise Crash('marker durable before SQL seal')
    h.ledger._after_marker_written=crash
    with pytest.raises(Crash):h.checkpoint('cut1')
    assert h.ledger.lookup_commit('cut1')['sealed'] is False
    del h.ledger._after_marker_written
    assert h.recover()['body']['step']==1


def test_superseded_writer_cannot_publish_a_prepared_marker(harness):
    h=harness();h.checkpoint('genesis');h.update()
    with pytest.raises(Crash):h.checkpoint('cut1','before_marker')
    old=h.writer;h.ledger.acquire_writer('replacement')
    with pytest.raises(ReplayFencedError):h.ledger.publish_marker('cut1',old,str(h.root))
    assert not (h.root/'replay-commits/cut1/COMMITTED.json').exists()


def test_checkpoint_barrier_blocks_enqueues_and_unpublished_seal(harness):
    h=harness();h.checkpoint('genesis');h.update()
    with pytest.raises(Crash):h.checkpoint('cut1','before_marker')
    h.mixer.next(h.fetch,1)
    with pytest.raises(RuntimeError):h.ledger.enqueue('first',2,1,batch_document(h.mixer),'u2',h.writer)
    with pytest.raises(FileNotFoundError):h.ledger.seal_commit('cut1',h.writer,str(h.root))


def test_legacy_launcher_resume_stays_fail_closed():
    from omegaconf import OmegaConf
    from aura.memory.replay.launcher import prepare_replay
    config=OmegaConf.create({'replay':{'replay_ratio':.3},'trainer':{'resume_mode':'auto'}})
    with pytest.raises(ValueError,match='explicit model/source adapters'):prepare_replay(config,object())


def test_start_retry_after_initial_pair_requires_recovery_not_fresh_mixer(harness):
    h=harness()
    assert h.ledger.initial_branch('first',h.writer)=='first'  # Lost create response, before commit.
    h.checkpoint('genesis')
    with pytest.raises(ReplayConflictError):h.ledger.initial_branch('first',h.writer)
    assert h.recover()['body']['step']==0
