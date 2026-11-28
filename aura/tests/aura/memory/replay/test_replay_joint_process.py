"""Real CPU optimizer/RNG continuity and abrupt-process crash tests, server only."""
import copy
import json
import hashlib
import multiprocessing
import os
import random

import pytest

from aura.memory.replay.codec import canonical
from aura.memory.replay.ledger import ReplayLedger,batch_document
from aura.memory.replay.mixing import group_key
from aura.memory.replay.sqlite_store import SQLiteReplayStore
from aura.memory.replay.state import snapshot_mixer,restore_mixer
from test_replay_ledger_joint import harness,Crash,UPDATE_FAILURES,record_evidence
from test_replay_mixing import settings


@pytest.mark.parametrize('strategy',['uniform','priority','reward_weighted'])
@pytest.mark.parametrize('ratio',[.3,1.])
def test_real_cpu_model_optimizer_scheduler_and_rng_match_uninterrupted(harness,strategy,ratio):
    import torch
    torch.random.default_generator.manual_seed(42);random.seed(42)
    h=harness(strategy,ratio=ratio)
    model=torch.nn.Linear(3,2,device='cpu')
    # torch_npu patches automatic foreach detection, even for CPU parameters.
    # Explicit scalar CPU kernels avoid probing any accelerator in this test.
    optimizer=torch.optim.AdamW(model.parameters(),lr=.01,foreach=False,fused=False)
    scheduler=torch.optim.lr_scheduler.StepLR(optimizer,step_size=1,gamma=.9)
    def save(step,version):
        folder=h.root/'torch-models'/f'{h.branch}-{step}';folder.mkdir(parents=True)
        path=folder/'state.pt'
        torch.save({'step':step,'policy_version':version,'model':model.state_dict(),
            'optimizer':optimizer.state_dict(),'scheduler':scheduler.state_dict(),
            'torch_rng':torch.get_rng_state(),'python_rng':random.getstate()},path)
        return dict(step=step,policy_version=version,
                    artifacts={k:[path.relative_to(h.root).as_posix()] for k in ('model','optimizer','scheduler','rng')})
    h.save=save
    def update():
        plan=h.mixer.next(h.fetch,h.version);step=h.mixer.completed_steps+1;facts=batch_document(h.mixer)
        h.ledger.enqueue(h.branch,step,h.version,facts,f'u-{step}',h.writer)
        lease=h.ledger.lease(h.branch,step,f'l-{step}',h.writer)
        optimizer.zero_grad(set_to_none=True)
        output=model(torch.randn(4,3,device='cpu'));loss=output.square().mean()+random.random()*.1*output.mean()
        loss.backward();assert any(p.grad.abs().sum()>0 for p in model.parameters())
        optimizer.step();scheduler.step()
        priority={group_key(g):float(g.samples[0].reward) for g in plan.groups} if strategy=='priority' else None
        h.mixer.commit_success(h.version,priority);h.version+=1
        state=snapshot_mixer(h.mixer,policy_version=h.version)
        h.ledger.ack(lease['lease_id'],state,h.writer)
        return facts,state
    h.checkpoint('genesis');update();h.checkpoint('cut1')
    expected=[update() for _ in range(3)]
    expected_state=copy.deepcopy({'model':model.state_dict(),'optimizer':optimizer.state_dict(),
        'scheduler':scheduler.state_dict(),'torch_rng':torch.get_rng_state(),'python_rng':random.getstate()})
    h.writer=h.ledger.acquire_writer('recovery-owner')
    chosen=h.ledger.recover_branch(h.root,'first','resume',h.writer)
    assert chosen['body']['step']==1 and h.ledger.inspect('first')['branch']['head_step']==4
    path=h.root/chosen['body']['artifacts'][0]['path']
    # Trusted test-generated artifact; the actor never unpickles model files.
    loaded=torch.load(path,map_location='cpu',weights_only=False)
    assert loaded['step']==loaded['policy_version']==1
    model.load_state_dict(loaded['model']);optimizer.load_state_dict(loaded['optimizer']);scheduler.load_state_dict(loaded['scheduler'])
    torch.set_rng_state(loaded['torch_rng']);random.setstate(loaded['python_rng'])
    h.branch='resume';h.version=1;h.cursor=chosen['source']['producer']['state']['next_batch']
    h.mixer=restore_mixer(chosen['snapshot'],h.config,h.store.get_trajectory,expected_policy_version=1)
    assert [update() for _ in range(3)]==expected
    def equal(a,b):
        if isinstance(a,torch.Tensor):assert torch.equal(a,b)
        elif isinstance(a,dict):
            assert a.keys()==b.keys()
            for k in a:equal(a[k],b[k])
        elif isinstance(a,(list,tuple)):
            assert len(a)==len(b)
            for x,y in zip(a,b):equal(x,y)
        else:assert a==b
    equal(expected_state,{'model':model.state_dict(),'optimizer':optimizer.state_dict(),
        'scheduler':scheduler.state_dict(),'torch_rng':torch.get_rng_state(),'python_rng':random.getstate()})
    record_evidence('cpu-model-'+strategy+'-'+str(ratio),dict(strategy=strategy,replay_ratio=ratio,
        restored_step=1,unpaired_old_head=4,final_step=4,model_optimizer_scheduler_rng_exact=True,
        consumption_and_sampler_state_exact=True,scheduler_last_epoch=scheduler.last_epoch,
        optimizer_steps=[int(v['step'].item()) for v in optimizer.state.values()],
        model_tensor_sha256={k:hashlib.sha256(v.detach().contiguous().numpy().tobytes()).hexdigest() for k,v in model.state_dict().items()},
        torch_rng_sha256=hashlib.sha256(torch.get_rng_state().numpy().tobytes()).hexdigest(),
        scope='Real CPU Linear/AdamW/StepLR protocol probe; not NPU FSDP/GRPO training'))


def crash_child(h,stage):
    # Open independent post-fork SQLite handles; no inherited connection is used.
    h.ledger.close();h.store.close()
    h.store=SQLiteReplayStore(h.config.path,store_id='store')
    h.ledger=ReplayLedger(h.root/'ledger.db',h.config,h.store.get_trajectory)
    h.recover('child')
    try:
        if stage in UPDATE_FAILURES:h.update(stage)
        else:
            h.update()
            if stage=='marker_before_sql_seal':
                h.ledger._after_marker_written=lambda:os._exit(73)
                h.checkpoint('child-cut')
            else:h.checkpoint('child-cut',stage)
    except Crash:os._exit(73)
    os._exit(74)


@pytest.mark.skipif(os.name!='posix',reason='Server Linux abrupt process tests')
@pytest.mark.parametrize('stage',['after_persist_before_enqueue','after_enqueue','after_lease',
    'after_update_before_ack','after_ack','after_model_save','after_prepare','before_marker','marker_before_sql_seal'])
def test_abrupt_os_exit_wal_and_marker_recovery(harness,stage):
    h=harness();h.checkpoint('genesis')
    process=multiprocessing.get_context('fork').Process(target=crash_child,args=(h,stage))
    process.start();process.join(20)
    if process.is_alive():process.kill();process.join();pytest.fail('Own crash-test subprocess hung')
    assert process.exitcode==73
    h.branch='child';chosen=h.recover('after-process-death')
    expected=1 if stage=='marker_before_sql_seal' else 0
    assert chosen['body']['step']==h.value==expected
    assert h.ledger.conn.execute('PRAGMA integrity_check').fetchall()[0][0]=='ok'
    if stage=='marker_before_sql_seal':assert h.ledger.lookup_commit('child-cut')['sealed'] is False
    h.update();assert h.value==expected+1
    record_evidence('process-'+stage,dict(stage=stage,child_exitcode=process.exitcode,selected_step=expected,
        continued_step=expected+1,integrity_check='ok',old_history=h.ledger.inspect('child'),
        marker_before_sql_seal_recovered=stage=='marker_before_sql_seal'))
