# Copyright (c) 2026 Huawei Technologies Co.,Ltd.
# Licensed under Mulan PSL v2 (http://license.coscl.org.cn/MulanPSL2).
"""Explicit P4 runtime opt-in; the legacy launcher is deliberately unchanged.

One acknowledged optimizer update per policy version. Real trainer/producer
adapters must provide quiescent source save/load and opaque model callbacks;
this module never guesses a data cursor or resolves a model-only latest pointer.
"""
from dataclasses import asdict
import hashlib

from .codec import canonical, digest
from .joint import publish_commit, read_commit
from .ledger import batch_document
from .runtime import ReplaySession
from .schema import ReplayIntegrityError
from .state import restore_mixer, snapshot_mixer


class ActorLedger:
    def __init__(self, actor, timeout):
        self.actor,self.timeout=actor,timeout

    def __getattr__(self, operation):
        if operation not in {'acquire_writer','initial_branch','enqueue','lease','ack',
                             'prepare_commit','lookup_commit','seal_commit','publish_marker','recover_branch','inspect',
                             'seen_online'}:
            raise AttributeError(operation)
        def call(*args,**kwargs):
            import ray
            return ray.get(self.actor.replay_ledger_call.remote(operation,args,kwargs),timeout=self.timeout)
        return call


class TransactionalReplaySession(ReplaySession):
    def __init__(self, config, pad_token_id, actor, ledger, writer, branch, policy_version, *, failpoint=None):
        super().__init__(config,pad_token_id,actor)
        self.ledger,self.writer,self.branch=ledger,writer,branch
        self._bind_history()
        self.policy_version=policy_version
        self.lease_receipt=None
        self.failed=False
        self.failpoint=failpoint or (lambda stage:None)

    def _bind_history(self):
        self.mixer.history_contains = lambda keys: self.ledger.seen_online(
            self.branch, self.mixer.completed_steps, sorted(keys), self.writer)

    @staticmethod
    def _open(config,actor,ledger_path,writer_request):
        import ray
        ray.get(actor.configure_replay_ledger.remote(ledger_path,asdict(config)),timeout=config.rpc_timeout)
        ledger=ActorLedger(actor,config.rpc_timeout)
        return ledger,ledger.acquire_writer(writer_request)

    @classmethod
    def start(cls,config,pad_token_id,actor,ledger_path,*,branch,writer_request,policy_version=0,failpoint=None):
        ledger,writer=cls._open(config,actor,ledger_path,writer_request)
        ledger.initial_branch(branch,writer,policy_version)
        return cls(config,pad_token_id,actor,ledger,writer,branch,policy_version,failpoint=failpoint)

    def _guard(self):
        if self.failed:raise RuntimeError('Uncertain transaction: recover model and replay from a paired commit')

    def abort_uncertain(self):
        """Call after an external model-update/source failure; never retry it."""
        self.failed=True

    def _load_sample(self,tid,store_id=None):
        import ray
        return ray.get(self.actor.get_trajectory.remote(tid,store_id),timeout=self.config.rpc_timeout)

    def next_batch(self,fetch_raw,current_version):
        self._guard()
        if self.lease_receipt is not None or current_version!=self.policy_version:
            raise RuntimeError('Outstanding lease / model policy version mismatch')
        try:
            batch=super().next_batch(fetch_raw,current_version)
            self.failpoint('after_plan')
            step=self.mixer.completed_steps+1
            facts=batch_document(self.mixer)
            self.ledger.enqueue(self.branch,step,current_version,facts,f'update-{step}',self.writer)
            self.failpoint('after_enqueue')
            self.lease_receipt=self.ledger.lease(self.branch,step,f'lease-{step}',self.writer)
            receipt=self.lease_receipt
            if (not isinstance(receipt,dict) or receipt.get('status')!='leased'
                    or receipt.get('branch')!=self.branch or receipt.get('step')!=step
                    or receipt.get('version')!=current_version or receipt.get('facts_digest')!=digest(facts)
                    or not isinstance(receipt.get('lease_id'),str) or not receipt['lease_id']):
                raise ReplayIntegrityError('Only a matching live lease authorizes a model update; ACK is not a lease')
            self.failpoint('after_lease')
            return batch
        except BaseException:
            self.failed=True;raise

    def commit_success(self,*args,**kwargs):
        raise RuntimeError('Transactional sessions require ack_update with the actual post-update policy version')

    def ack_update(self,new_policy_version,priority_updates=None):
        self._guard()
        if self.lease_receipt is None or type(new_policy_version) is not int or new_policy_version!=self.policy_version+1:
            raise RuntimeError('ACK requires a lease and exactly one successful optimizer update')
        try:
            self.failpoint('after_update_before_ack')
            metrics=self.mixer.commit_success(self.policy_version,priority_updates)
            snapshot=snapshot_mixer(self.mixer,policy_version=new_policy_version)
            ack=self.ledger.ack(self.lease_receipt['lease_id'],snapshot,self.writer)
            if (not isinstance(ack,dict) or ack.get('status')!='acked' or ack.get('branch')!=self.branch
                    or ack.get('lease_id')!=self.lease_receipt['lease_id']
                    or ack.get('step')!=self.mixer.completed_steps or ack.get('model_version')!=new_policy_version
                    or ack.get('snapshot_sha')!=hashlib.sha256(snapshot).hexdigest()):
                raise ReplayIntegrityError('ACK receipt does not match the actual updated mixer snapshot')
            self.failpoint('after_ack')
            self.policy_version=new_policy_version;self.lease_receipt=None
            return metrics
        except BaseException:
            self.failed=True;raise

    def checkpoint(self,root,source,save_model,*,commit_id=None):
        self._guard()
        if self.lease_receipt is not None or self.mixer.inflight is not None:
            raise RuntimeError('Checkpoint requires a post-ACK quiescent point')
        try:
            return publish_commit(root,self.config,self._load_sample,self.ledger,self.writer,self.branch,
                snapshot_mixer(self.mixer,policy_version=self.policy_version),source,save_model,
                commit_id=commit_id,failpoint=self.failpoint)
        except BaseException:
            self.failed=True;raise

    @classmethod
    def recover(cls,config,pad_token_id,actor,ledger_path,root,*,from_branch,new_branch,writer_request,
                load_model,restore_source,commit_id=None):
        ledger,writer=cls._open(config,actor,ledger_path,writer_request)
        chosen=ledger.recover_branch(str(root),from_branch,new_branch,writer,commit_id)
        session=cls(config,pad_token_id,actor,ledger,writer,new_branch,chosen['body']['policy_version'])
        session.failed=True  # Nothing may run until BOTH callbacks and post-load hashes pass.
        before=read_commit(root,chosen['body']['commit_id'],config,session._load_sample,ledger.lookup_commit)
        body=before['body']
        paths={r:[] for r in ('model','optimizer','scheduler','rng')}
        from pathlib import Path
        for item in body['artifacts']:
            for role in item['roles']:paths[role].append(str(Path(before['root'])/item['path']))
        receipt=load_model(paths,body['step'],body['policy_version'])
        if (not isinstance(receipt,dict) or set(receipt)!={'step','policy_version'}
                or receipt!={k:body[k] for k in ('step','policy_version')}
                or any(type(receipt[k]) is not int for k in ('step','policy_version'))):
            raise ReplayIntegrityError('Loaded model/optimizer watermark was not confirmed')
        source_receipt=restore_source(before['source'])
        if source_receipt!={'source_sha256':hashlib.sha256(canonical(before['source'])).hexdigest()}:
            raise ReplayIntegrityError('Producer cursor and queue restoration was not confirmed')
        after=read_commit(root,body['commit_id'],config,session._load_sample,ledger.lookup_commit)
        if before['marker_sha256']!=after['marker_sha256']:
            raise ReplayIntegrityError('Joint commit changed during model/source restore')
        session.mixer=restore_mixer(after['snapshot'],config,session._load_sample,expected_policy_version=body['policy_version'])
        session._bind_history()
        session.failed=False
        return session,chosen
