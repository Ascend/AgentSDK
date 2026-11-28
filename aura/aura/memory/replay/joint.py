# Copyright (c) 2026 Huawei Technologies Co.,Ltd.
# Licensed under Mulan PSL v2 (http://license.coscl.org.cn/MulanPSL2).
"""Local-filesystem P4 paired commit protocol. The final marker is authoritative.

No 'latest consumed' or model-only pointer is used for recovery. Model/source
callbacks must implement their real save/load and drain semantics. No pickle is
parsed here; opaque model/RNG files are hashed, not deserialized by the actor.
"""
from pathlib import Path, PurePosixPath
import hashlib
import hmac
import os
import re
import uuid

from .codec import canonical, digest, strict_json
from .schema import ReplayGroup, ReplayIntegrityError, identifier, integer
from .state import MAX_STATE_BYTES, _contract, _reference, _unpack, restore_mixer

FORMAT='aura.replay.joint.v1'
ROLES={'model','optimizer','scheduler','rng'}


def _hex(value):
    if not isinstance(value,str) or not re.fullmatch('[0-9a-f]{64}',value):
        raise ReplayIntegrityError('Invalid SHA256')


def _commit_id(value):
    if not isinstance(value,str) or not re.fullmatch('[a-zA-Z0-9_-]{1,128}',value):
        raise ReplayIntegrityError('Unsafe commit identity')
    return value


def _root(root):
    root=Path(root)
    if not root.is_absolute() or root.is_symlink():
        raise ReplayIntegrityError('Commit root must be an absolute non-symlink directory')
    return root.resolve(strict=True)


def _safe(root, relative):
    if (not isinstance(relative,str) or not relative or '\\' in relative or ':' in relative
            or PurePosixPath(relative).is_absolute() or any(p in ('..','.') for p in relative.split('/'))
            or any(not p for p in relative.split('/'))):
        raise ReplayIntegrityError('Unsafe checkpoint relative path')
    path=root
    for part in relative.split('/'):
        path=path/part
        if path.is_symlink():raise ReplayIntegrityError('Symlink checkpoint component')
    resolved=path.resolve(strict=True)
    if not resolved.is_relative_to(root) or not resolved.is_file():
        raise ReplayIntegrityError('Checkpoint file outside root / not a regular file')
    return resolved


def _fsync_dir(path):
    descriptor=os.open(path,os.O_RDONLY|getattr(os,'O_DIRECTORY',0))
    try:os.fsync(descriptor)
    finally:os.close(descriptor)


def _read_small(path):
    if path.stat().st_size>MAX_STATE_BYTES:raise ReplayIntegrityError('Oversized joint sidecar')
    with path.open('rb') as stream:data=stream.read(MAX_STATE_BYTES+1)
    if len(data)>MAX_STATE_BYTES:raise ReplayIntegrityError('Growing/oversized joint sidecar')
    return data


def _write_once(path,data):
    temporary=path.with_name('.'+path.name+'.'+uuid.uuid4().hex+'.tmp')
    try:
        with temporary.open('xb') as stream:
            stream.write(data);stream.flush();os.fsync(stream.fileno())
        os.link(temporary,path)
        _fsync_dir(path.parent)
    finally:temporary.unlink(missing_ok=True)


def file_info(root,relative,*,sync=False):
    path=_safe(root,relative);hasher=hashlib.sha256()
    with path.open('rb') as stream:
        before=os.fstat(stream.fileno())
        # Reuse one bounded buffer rather than allocate/copy a fresh bytes
        # object for each block. Every publication/recovery still reads EOF.
        buffer=bytearray(min(8*1024*1024,max(1,before.st_size)));view=memoryview(buffer)
        while count:=stream.readinto(buffer):hasher.update(view[:count])
        if sync:os.fsync(stream.fileno())
        after=os.fstat(stream.fileno())
    if (before.st_size,before.st_mtime_ns,before.st_ino)!=(after.st_size,after.st_mtime_ns,after.st_ino):
        raise ReplayIntegrityError('Checkpoint changed while hashing')
    if sync:
        parent=path.parent
        while True:
            _fsync_dir(parent)
            if parent==root:break
            parent=parent.parent
    return {'path':relative,'bytes':after.st_size,'sha256':hasher.hexdigest()}


def file_infos(root, relatives, *, sync=False):
    """Bounded parallel full-byte verification; preserve input order and fsync.

    There is deliberately no cross-call checksum cache: publication and every
    independent recovery still read each artifact and enforce the same guards.
    """
    from concurrent.futures import ThreadPoolExecutor
    relatives = tuple(relatives)
    if len(relatives) < 2:
        return [file_info(root, path, sync=sync) for path in relatives]
    with ThreadPoolExecutor(max_workers=min(3, len(relatives)),
                            thread_name_prefix='replay-artifact-hash') as pool:
        return list(pool.map(lambda path: file_info(root, path, sync=sync), relatives))


def validate_body(body,config):
    fields={'format','commit_id','branch','step','policy_version','contract_digest',
            'replay_sha256','source_sha256','artifacts','model_receipt'}
    if not isinstance(body,dict) or set(body)!=fields or body['format']!=FORMAT:
        raise ReplayIntegrityError('Unsupported joint commit body')
    _commit_id(body['commit_id']);identifier(body['branch'],'branch')
    integer(body['step'],'step');integer(body['policy_version'],'policy version')
    for k in ('contract_digest','replay_sha256','source_sha256'):_hex(body[k])
    if body['contract_digest']!=digest(_contract(config)):
        raise ReplayIntegrityError('Joint commit run/algorithm contract mismatch')
    receipt=body['model_receipt']
    if (not isinstance(receipt,dict) or set(receipt)!={'step','policy_version'}
            or type(receipt['step']) is not int or type(receipt['policy_version']) is not int
            or receipt!={k:body[k] for k in ('step','policy_version')}):
        raise ReplayIntegrityError('Model/replay watermark mismatch')
    if not isinstance(body['artifacts'],list) or not 1<=len(body['artifacts'])<=10000:
        raise ReplayIntegrityError('Missing or excessive model artifacts')
    paths,roles=set(),set()
    for item in body['artifacts']:
        if not isinstance(item,dict) or set(item)!={'path','bytes','sha256','roles'}:
            raise ReplayIntegrityError('Invalid model artifact record')
        identifier(item['path'],'artifact path');integer(item['bytes'],'artifact bytes',1);_hex(item['sha256'])
        if (not isinstance(item['roles'],list) or not item['roles'] or any(r not in ROLES for r in item['roles'])
                or len(set(item['roles']))!=len(item['roles']) or item['path'] in paths):
            raise ReplayIntegrityError('Duplicate artifact or invalid roles')
        paths.add(item['path']);roles.update(item['roles'])
    if roles!=ROLES:raise ReplayIntegrityError('Model, optimizer, scheduler and RNG artifacts are all required')


def validate_source(value,config,load_sample,policy_version,seen_online):
    fields={'format','quiescent','producer','queued_batches','prefetch'}
    if (not isinstance(value,dict) or set(value)!=fields or value['format']!='aura.replay.source-state.v1'
            or value['quiescent'] is not True or value['prefetch'] is not None
            or not isinstance(value['queued_batches'],list) or len(value['queued_batches'])>4096):
        raise ReplayIntegrityError('Source must be drained, quiescent and explicit about its queue')
    producer=value['producer']
    if not isinstance(producer,dict) or set(producer)!={'identity','state'} or not isinstance(producer['state'],dict):
        raise ReplayIntegrityError('Producer cursor/identity required')
    _hex(producer['identity'])
    batches,ids,group_keys=set(),set(),set()
    for batch in value['queued_batches']:
        if not isinstance(batch,dict) or set(batch)!={'batch_id','weight_version','replay_refs'}:
            raise ReplayIntegrityError('Invalid durable queue batch')
        identifier(batch['batch_id'],'batch id');integer(batch['weight_version'],'queue version')
        if batch['batch_id'] in batches or batch['weight_version']>policy_version:
            raise ReplayIntegrityError('Duplicate/future queued batch')
        batches.add(batch['batch_id']);refs=batch['replay_refs']
        if not isinstance(refs,list) or len(refs)!=config.group_size*config.batch_groups:
            raise ReplayIntegrityError('Incomplete durable queue batch')
        groups={}
        for ref in refs:
            if not isinstance(ref,dict) or set(ref)!={'trajectory_id','checksum','created_seq','source_store_id'}:
                raise ReplayIntegrityError('Malformed queued reference')
            integer(ref['created_seq'],'created_seq',1);_hex(ref['checksum'])
            tid=identifier(ref['trajectory_id'],'trajectory id');sample=load_sample(tid)
            if (tid in ids or sample is None or ref['source_store_id']!=config.store_id
                    or sample.weight_version!=batch['weight_version'] or sample.source_store_id!=config.store_id
                    or sample.payload_checksum!=ref['checksum'] or sample.created_seq!=ref['created_seq']):
                raise ReplayIntegrityError('Changed/missing/duplicate queued trajectory')
            ids.add(tid);groups.setdefault((sample.source_store_id,sample.group_id),[]).append(sample)
        if len(groups)!=config.batch_groups:raise ReplayIntegrityError('Invalid queue group count')
        for key,samples in groups.items():
            group=ReplayGroup(tuple(samples))
            # Full signatures and eligibility are checked by the same restore validator.
            from .mixing import compatible
            if key in seen_online or key in group_keys or not compatible(group,config,batch['weight_version']):
                raise ReplayIntegrityError('Queue overlaps mixer or has incompatible groups')
            group_keys.add(key)
    canonical(value)  # Reject nonfinite/non-JSON producer state.
    return group_keys


def read_commit(root,commit_id,config,load_sample,lookup):
    root=_root(root);commit_id=_commit_id(commit_id)
    relative='replay-commits/'+commit_id+'/'
    marker=strict_json(_read_small(_safe(root,relative+'COMMITTED.json')),max_bytes=MAX_STATE_BYTES)
    if not isinstance(marker,dict) or set(marker)!={'body','checksum'} or not isinstance(marker['checksum'],str):
        raise ReplayIntegrityError('Malformed joint commit marker')
    body=marker['body'];validate_body(body,config)
    if body['commit_id']!=commit_id or not hmac.compare_digest(marker['checksum'],digest(body)):
        raise ReplayIntegrityError('Joint marker checksum/identity mismatch')
    return verify_prepared(root,body,config,load_sample,lookup)


def verify_prepared(root,body,config,load_sample,lookup):
    """Verify prepared files; NOT a committed-recovery selector on its own."""
    checked=_prepared_metadata(root,body,config,load_sample,lookup)
    root=Path(checked['root'])
    actual_files=file_infos(root,(item['path'] for item in body['artifacts']))
    for expected,actual in zip(body['artifacts'],actual_files):
        if actual!={k:expected[k] for k in ('path','bytes','sha256')}:
            raise ReplayIntegrityError('Joint model artifact changed or incomplete')
    return checked


def _prepared_metadata(root,body,config,load_sample,lookup):
    """Replay/source/ledger checks shared by full reads and publication return."""
    root=_root(root);validate_body(body,config);commit_id=body['commit_id']
    relative='replay-commits/'+commit_id+'/'
    registered=lookup(commit_id)
    if registered['body']!=body or registered['body_digest']!=digest(body):
        raise ReplayIntegrityError('Joint marker does not match ledger prepare')
    snapshot=_read_small(_safe(root,relative+'mixer.json'))
    source_bytes=_read_small(_safe(root,relative+'source.json'))
    if (hashlib.sha256(snapshot).hexdigest()!=body['replay_sha256']
            or hashlib.sha256(source_bytes).hexdigest()!=body['source_sha256']):
        raise ReplayIntegrityError('Joint replay/source checksum mismatch')
    mixer=restore_mixer(snapshot,config,load_sample,expected_policy_version=body['policy_version'])
    if mixer.completed_steps!=body['step']:raise ReplayIntegrityError('Joint mixer step mismatch')
    source=strict_json(source_bytes,max_bytes=MAX_STATE_BYTES)
    queued = validate_source(source,config,load_sample,body['policy_version'],mixer.seen_online)
    if queued:
        lookup(commit_id, online_keys=sorted(queued))
    return {'body':body,'sequence':registered['sequence'],'snapshot':snapshot,'source':source,
            'root':str(root),'marker_sha256':hashlib.sha256(canonical({'body':body,'checksum':digest(body)})).hexdigest()}


def select_commit(root,config,load_sample,lookup,*,from_branch,base_commit=None,commit_id=None):
    root=_root(root)
    if commit_id is not None:
        chosen=read_commit(root,commit_id,config,load_sample,lookup)
        if chosen['body']['branch']!=from_branch and commit_id!=base_commit:
            raise ReplayIntegrityError('Explicit commit is outside recovery branch ancestry')
        chosen['rejected']=[];return chosen
    candidates,rejected=[],[]
    for path in sorted((root/'replay-commits').glob('*/COMMITTED.json')):
        try:
            candidate=read_commit(root,path.parent.name,config,load_sample,lookup)
            if candidate['body']['branch']==from_branch or candidate['body']['commit_id']==base_commit:
                candidates.append(candidate)
        except (OSError,ValueError,TypeError,KeyError) as error:
            rejected.append({'commit_id':path.parent.name,'reason':type(error).__name__})
    if not candidates:raise ReplayIntegrityError('No complete paired commit available; model-only/ACK-only recovery forbidden')
    chosen=max(candidates,key=lambda c:(c['body']['step'],c['sequence']))
    chosen['rejected']=rejected
    return chosen


def publish_commit(root,config,load_sample,ledger,writer,branch,snapshot,source,save_model,*,commit_id=None,failpoint=None):
    """Save callbacks first, prepare/freeze ACK head, sidecars, marker LAST, seal.

    save_model(step,version) -> {step,policy_version,artifacts:{role:[relative]}}.
    Callers MUST poison uncertain sessions and recover; no retrying an update.
    """
    root=_root(root);commit_id=_commit_id(commit_id or uuid.uuid4().hex)
    hit=failpoint or (lambda stage:None)
    state=_unpack(snapshot);version=state['policy_version'];step=state['completed_steps']
    mixer=restore_mixer(snapshot,config,load_sample,expected_policy_version=version)
    queued = validate_source(source,config,load_sample,version,mixer.seen_online)
    if queued and ledger.seen_online(branch,step,sorted(queued),writer):
        raise ReplayIntegrityError('Checkpoint queue repeats previously fetched online generation')
    source_bytes=canonical(source)
    if len(source_bytes)>MAX_STATE_BYTES:raise ValueError('Source state exceeds bounded size')
    receipt=save_model(step,version)
    if (not isinstance(receipt,dict) or set(receipt)!={'step','policy_version','artifacts'}
            or type(receipt['step']) is not int or type(receipt['policy_version']) is not int
            or receipt['step']!=step or receipt['policy_version']!=version
            or not isinstance(receipt['artifacts'],dict) or set(receipt['artifacts'])!=ROLES):
        raise ReplayIntegrityError('Model save receipt/watermark/roles mismatch')
    inventory={}
    for role,paths in receipt['artifacts'].items():
        if not isinstance(paths,list) or not paths:raise ReplayIntegrityError('Missing model role artifacts')
        for relative in paths:
            if relative not in inventory:inventory[relative]={'roles':[]}
            if role in inventory[relative]['roles']:raise ReplayIntegrityError('Repeated artifact role')
            inventory[relative]['roles'].append(role)
    reads=file_infos(root,inventory,sync=True)
    for actual in reads:
        inventory[actual['path']].update(actual)
    hit('after_model_save')
    body={'format':FORMAT,'commit_id':commit_id,'branch':branch,'step':step,'policy_version':version,
          'contract_digest':digest(_contract(config)),'replay_sha256':hashlib.sha256(snapshot).hexdigest(),
          'source_sha256':hashlib.sha256(source_bytes).hexdigest(),
          'artifacts':[dict(v,roles=sorted(v['roles'])) for k,v in sorted(inventory.items())],
          'model_receipt':{'step':step,'policy_version':version}}
    validate_body(body,config)
    ledger.prepare_commit(body,snapshot,writer);hit('after_prepare')
    parent=root/'replay-commits'
    if parent.is_symlink():raise ReplayIntegrityError('Symlink commit parent')
    parent.mkdir(exist_ok=True);_fsync_dir(root)
    destination=parent/commit_id;destination.mkdir(exist_ok=False);_fsync_dir(parent)
    _write_once(destination/'mixer.json',snapshot);hit('after_replay_sidecar')
    _write_once(destination/'source.json',source_bytes);hit('after_source_sidecar');hit('before_marker')
    ledger.publish_marker(commit_id,writer,str(root))
    hit('after_marker');hit('after_seal')
    return read_commit(root,commit_id,config,load_sample,ledger.lookup_commit)
