"""Opt-in real CPU Ray transport, persistent leases, paired cut and source queue."""
from dataclasses import replace
import hashlib
import json
import os

import pytest

from aura.memory.replay.codec import canonical,digest
from aura.memory.replay.mixing import group_key
from aura.memory.replay.state import snapshot_mixer
from test_replay_actor_integration import real_ray
from test_replay_mixing import settings
from test_replay_ledger_joint import record_evidence

pytestmark=pytest.mark.skipif(os.environ.get('RUN_REPLAY_RAY_TESTS')!='1',reason='Explicit CPU Ray opt-in')


class Driver:
    def __init__(self,ray,actor,config,make_group,root):
        self.ray,self.actor,self.config,self.make_group,self.root=ray,actor,config,make_group,root
        self.value=0;self.cursor=0;self.queue=[];self.saves=0;self.session=None

    def produce(self):
        n=self.cursor;self.cursor+=1;samples=[]
        version=self.session.policy_version
        for i in range(self.config.batch_groups):
            members=self.make_group(group_id=f'b{n}-{i}',version=version,reward=(n+i)%2)
            samples.extend(replace(s,payload={**s.payload,'rm_scores':s.payload['token_level_rewards']}) for s in members)
        receipt=self.ray.get(self.actor.append_replay_batch.remote(samples,f'batch-{n}'),timeout=30)
        return {'batch_id':f'batch-{n}','weight_version':version,'replay_refs':[
            dict(trajectory_id=tid,checksum=sha,created_seq=seq,source_store_id='store') for tid,sha,seq in receipt.sample_refs]}

    def fetch(self):return self.queue.pop(0) if self.queue else self.produce()

    def source(self):
        return {'format':'aura.replay.source-state.v1','quiescent':True,'prefetch':None,
                'producer':{'identity':digest('fixed-actor-source'),'state':{'next_batch':self.cursor}},
                'queued_batches':list(self.queue)}

    def save(self,step,version):
        folder=self.root/'models'/f'{self.session.branch}-{step}-{self.saves}';self.saves+=1;folder.mkdir(parents=True)
        artifacts={}
        for role in ('model','optimizer','scheduler','rng'):
            p=folder/(role+'.json');p.write_bytes(canonical(dict(role=role,step=step,policy_version=version,value=self.value)))
            artifacts[role]=[p.relative_to(self.root).as_posix()]
        return dict(step=step,policy_version=version,artifacts=artifacts)

    def load(self,paths,step,version):
        from pathlib import Path
        values=[]
        for role,files in paths.items():
            assert len(files)==1
            data=json.loads(Path(files[0]).read_bytes())
            assert data['role']==role and data['step']==step and data['policy_version']==version
            values.append(data['value'])
        assert len(set(values))==1;self.value=values[0]
        return dict(step=step,policy_version=version)

    def restore(self,source):
        assert source['producer']['identity']==digest('fixed-actor-source')
        self.cursor=source['producer']['state']['next_batch'];self.queue=list(source['queued_batches'])
        return {'source_sha256':hashlib.sha256(canonical(source)).hexdigest()}

    def update(self):
        session=self.session;batch=session.next_batch(self.fetch,session.policy_version)
        self.value+=1
        priority={group_key(g):float(g.samples[0].reward) for g in session.mixer.inflight.groups} if self.config.replay_strategy=='priority' else None
        probabilities=session.mixer.inflight.draw_probabilities
        session.ack_update(session.policy_version+1,priority)
        return batch['replay_trajectory_ids'],probabilities,snapshot_mixer(session.mixer,policy_version=session.policy_version)


@pytest.mark.parametrize('strategy',['uniform','priority','reward_weighted'])
@pytest.mark.parametrize('ratio',[.3,1.])
@pytest.mark.parametrize('backend',['sqlite','json'])
def test_real_actor_death_joint_cut_not_latest_ack_and_queue_restore(real_ray,settings,make_group,tmp_path,strategy,ratio,backend):
    from aura.memory.episode.episode import Episode
    from aura.memory.replay.transaction import TransactionalReplaySession
    config=settings(path=str(tmp_path/'store'),backend=backend,batch_groups=2,replay_buffer_size=12,
                    replay_ratio=ratio,replay_strategy=strategy,warmup_online_steps=1)
    spec=dict(backend=backend,path=config.path,store_id='store');ledger_path=str(tmp_path/'consumer.db')
    actor=Episode.remote('p4-txn-writer',replay_store_config=spec);reopened=None
    driver=Driver(real_ray,actor,config,make_group,tmp_path)
    try:
        driver.session=TransactionalReplaySession.start(config,0,actor,ledger_path,branch='first',writer_request='owner-one')
        driver.session.checkpoint(tmp_path,driver.source(),driver.save,commit_id='genesis')
        driver.update();driver.queue.append(driver.produce())
        driver.session.checkpoint(tmp_path,driver.source(),driver.save,commit_id='cut1')
        expected=[driver.update(),driver.update()]
        history=driver.session.ledger.inspect('first');assert history['branch']['head_step']==3
        real_ray.kill(actor,no_restart=True);actor=None
        reopened=Episode.remote('p4-txn-reopened',replay_store_config=spec);driver.actor=reopened
        driver.session,selected=TransactionalReplaySession.recover(config,0,reopened,ledger_path,tmp_path,
            from_branch='first',new_branch='recovered',writer_request='owner-two',load_model=driver.load,restore_source=driver.restore)
        assert selected['body']['step']==1 and driver.value==1 and len(driver.queue)==1 and driver.cursor==2
        assert driver.session.ledger.inspect('first')==history
        assert [driver.update(),driver.update()]==expected
        driver.session.checkpoint(tmp_path,driver.source(),driver.save,commit_id='cut3')
        assert driver.value==3 and driver.session.ledger.lookup_commit('cut3')['sealed']
        record_evidence('actor-'+backend+'-'+strategy+'-'+str(ratio),dict(backend=backend,strategy=strategy,
            replay_ratio=ratio,old_unpaired_head=3,selected_step=1,final_step=3,restored_queued_batches=1,
            source_cursor_at_restore=2,consumption_ids_probabilities_sampler_snapshots_exact=True,
            old_history_preserved=True,checkpoint_cut3_sealed=True,
            final_snapshot_sha256=hashlib.sha256(snapshot_mixer(driver.session.mixer,policy_version=3)).hexdigest(),
            scope='Actual CPU Ray Episode actor death/reopen; model adapter uses synthetic JSON role files'))
        with pytest.raises(RuntimeError):driver.session.commit_success(3)
    finally:
        for target in (actor,reopened):
            if target is not None:
                try:real_ray.get(target.close_replay_store.remote(),timeout=10)
                finally:real_ray.kill(target,no_restart=True)


@pytest.mark.parametrize('stage',['after_enqueue','after_lease','after_update_before_ack','after_ack'])
def test_uncertain_transaction_poisoned_until_paired_recovery(real_ray,settings,make_group,tmp_path,stage):
    from aura.memory.episode.episode import Episode
    from aura.memory.replay.transaction import TransactionalReplaySession
    config=settings(path=str(tmp_path/'store'),batch_groups=2,replay_buffer_size=12,warmup_online_steps=1)
    actor=Episode.remote('p4-poison',replay_store_config=dict(backend='sqlite',path=config.path,store_id='store'))
    driver=Driver(real_ray,actor,config,make_group,tmp_path);ledger_path=str(tmp_path/'control.db')
    class Injected(RuntimeError):pass
    try:
        driver.session=TransactionalReplaySession.start(config,0,actor,ledger_path,branch='first',writer_request='owner')
        driver.session.checkpoint(tmp_path,driver.source(),driver.save,commit_id='genesis')
        def fail(name):
            if name==stage:raise Injected(stage)
        driver.session.failpoint=fail
        with pytest.raises(Injected):driver.update()
        assert driver.session.failed
        with pytest.raises(RuntimeError):driver.session.next_batch(driver.fetch,0)
        driver.session,chosen=TransactionalReplaySession.recover(config,0,actor,ledger_path,tmp_path,
            from_branch='first',new_branch='recovered',writer_request='new-owner',load_model=driver.load,restore_source=driver.restore)
        assert chosen['body']['step']==driver.value==0
        driver.update();assert driver.value==1
        record_evidence('poison-'+stage,dict(stage=stage,session_blocked_before_restore=True,
            selected_step=chosen['body']['step'],continued_step=1,old_history=driver.session.ledger.inspect('first')))
    finally:
        try:real_ray.get(actor.close_replay_store.remote(),timeout=10)
        finally:real_ray.kill(actor,no_restart=True)


def test_acked_reply_never_authorizes_a_second_optimizer_update(real_ray,settings,make_group,tmp_path,monkeypatch):
    from aura.memory.episode.episode import Episode
    from aura.memory.replay.transaction import TransactionalReplaySession
    from aura.memory.replay.schema import ReplayIntegrityError
    config=settings(path=str(tmp_path/'store'),batch_groups=2,replay_buffer_size=12,warmup_online_steps=1)
    actor=Episode.remote('p4-receipt',replay_store_config=dict(backend='sqlite',path=config.path,store_id='store'))
    driver=Driver(real_ray,actor,config,make_group,tmp_path)
    try:
        driver.session=TransactionalReplaySession.start(config,0,actor,str(tmp_path/'control.db'),branch='first',writer_request='owner')
        driver.session.checkpoint(tmp_path,driver.source(),driver.save,commit_id='genesis')
        original=driver.session.ledger.lease
        def wrong_reply(*args,**kwargs):return dict(original(*args,**kwargs),status='acked')
        monkeypatch.setattr(driver.session.ledger,'lease',wrong_reply)
        with pytest.raises(ReplayIntegrityError,match='ACK is not a lease'):driver.update()
        assert driver.value==0 and driver.session.failed
        assert driver.session.ledger.inspect('first')['counts']=={'leased':1}
    finally:
        try:real_ray.get(actor.close_replay_store.remote(),timeout=10)
        finally:real_ray.kill(actor,no_restart=True)
