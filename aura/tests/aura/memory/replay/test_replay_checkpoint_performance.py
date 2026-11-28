"""SERVER ONLY: preserve checkpoint bytes, barriers and fault semantics."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier, Event, Lock
from types import SimpleNamespace

import pytest

from aura.memory.replay.joint import file_info, file_infos
from aura.memory.replay.schema import ReplayIntegrityError
from aura.trainer.train_adapter.verl.full_async.replay_recovery import save_while_draining


def test_parallel_artifact_hashes_match_serial_order_and_sync(tmp_path):
    files=[]
    for i in range(4):
        p=tmp_path/(str(i)+'.bin');p.write_bytes(bytes([i])*16384);files.append(p.name)
    assert file_infos(tmp_path,files,sync=True)==[file_info(tmp_path,p,sync=True) for p in files]
    assert file_infos(tmp_path,[])==[]
    assert file_infos(tmp_path,files[:1])==[file_info(tmp_path,files[0])]


def test_artifact_reads_overlap_with_three_worker_bound(tmp_path,monkeypatch):
    import aura.memory.replay.joint as joint
    barrier=Barrier(3,timeout=5);lock=Lock();active=0;peak=0
    def read(root,path,*,sync=False):
        nonlocal active,peak
        assert sync
        with lock:active+=1;peak=max(peak,active)
        barrier.wait()
        with lock:active-=1
        return dict(path=path,bytes=1,sha256='a'*64)
    monkeypatch.setattr(joint,'file_info',read)
    rows=joint.file_infos(tmp_path,['c','a','b'],sync=True)
    assert [x['path'] for x in rows]==['c','a','b'] and peak==3


@pytest.mark.parametrize('bad',['missing','outside','mutated'])
def test_parallel_reads_retain_artifact_rejection(tmp_path,bad):
    path=tmp_path/'good';path.write_bytes(b'unchanged')
    expected=file_info(tmp_path,'good')
    if bad=='missing':relative='missing'
    elif bad=='outside':relative='../outside'
    else:path.write_bytes(b'corrupt');relative='good'
    if bad=='mutated':assert file_infos(tmp_path,[relative])[0]!=expected
    else:
        with pytest.raises((OSError,ValueError)):file_infos(tmp_path,['good',relative])


class Session:
    def __init__(self):
        self.lease_receipt=None;self.mixer=SimpleNamespace(inflight=None,completed_steps=6)
        self.policy_version=6;self.failed=False;self.calls=[]
    def _guard(self):
        if self.failed:raise RuntimeError('poisoned')
    def checkpoint(self,root,state,save):
        self.calls.append((root,state))
        return save(self.mixer.completed_steps,self.policy_version)


def test_native_save_overlaps_real_drain_before_publication():
    started=Event();drained=Event();finished=Event();session=Session()
    receipt=dict(step=6,policy_version=6,artifacts={})
    def save(step,version):
        assert (step,version)==(6,6);started.set()
        assert drained.wait(5);finished.set();return receipt
    def snapshot():
        assert started.wait(5);drained.set();return {'quiescent':True}
    source=SimpleNamespace(snapshot=snapshot)
    model=SimpleNamespace(save=save)
    assert save_while_draining(session,source,model,'root') is receipt
    assert finished.is_set() and len(session.calls)==1


@pytest.mark.parametrize('fault',['save','drain','boundary'])
def test_concurrent_save_fault_never_publishes_or_resumes(fault):
    session=Session();started=Event();finished=Event();save_count=0
    def save(step,version):
        nonlocal save_count
        save_count+=1;started.set()
        try:
            if fault=='save':raise OSError('save failed')
            return dict(step=step,policy_version=version)
        finally:finished.set()
    def snapshot():
        assert started.wait(5)
        if fault=='drain':raise TimeoutError('generation active')
        if fault=='boundary':session.policy_version=7
        return {'quiescent':True}
    with pytest.raises((OSError,TimeoutError,ReplayIntegrityError)):
        save_while_draining(session,SimpleNamespace(snapshot=snapshot),SimpleNamespace(save=save),'root')
    assert save_count==1 and finished.is_set() and not session.calls


@pytest.mark.parametrize('unsafe',['lease','inflight','poisoned'])
def test_concurrent_save_rejects_unsafe_training_boundary(unsafe):
    session=Session()
    if unsafe=='lease':session.lease_receipt={'id':'live'}
    elif unsafe=='inflight':session.mixer.inflight=object()
    else:session.failed=True
    def forbidden(*a):raise AssertionError('save/drain must not start')
    with pytest.raises(RuntimeError):
        save_while_draining(session,SimpleNamespace(snapshot=forbidden),SimpleNamespace(save=forbidden),'root')
