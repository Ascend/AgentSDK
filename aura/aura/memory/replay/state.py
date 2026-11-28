# Copyright (c) 2026 Huawei Technologies Co.,Ltd.
# Licensed under Mulan PSL v2 (http://license.coscl.org.cn/MulanPSL2).
"""P4 primitive: safe quiescent mixer snapshots, NOT a joint training checkpoint.

The launcher resume guard intentionally remains enabled. The caller must first
drain generation/prefetch and later pair this sidecar with a committed model,
optimizer, data cursor and consumption ledger. A checksum is not authentication.
"""
from collections import OrderedDict, deque
from dataclasses import asdict
from decimal import Decimal, InvalidOperation
import hmac
import math
import os
from pathlib import Path
import uuid

from .codec import ReplayCodec, canonical, digest, strict_json
from .mixing import GroupMixer, group_key
from .schema import COMPATIBILITY_FIELDS, ReplayGroup, ReplayIntegrityError, identifier, integer
from .reuse import source_for, load_reference, effective_version

FORMAT = 'aura.replay.mixer-state.v2'
MAX_STATE_BYTES = 16 * 1024 * 1024
TRANSPORT_FIELDS = {'path', 'actor_name', 'actor_namespace', 'rpc_timeout'}


def _contract(config):
    value = {k:v for k,v in asdict(config).items() if k not in TRANSPORT_FIELDS}
    if not value.get('reuse_sources'):
        value.pop('reuse_sources', None)  # Preserve existing single-store P4 commits.
    return strict_json(canonical(value), MAX_STATE_BYTES)


def _key(value):
    if not isinstance(value, list) or len(value) != 2:
        raise ReplayIntegrityError('Invalid group key')
    return tuple(identifier(v, 'group key') for v in value)


def _reference(group):
    refs = []
    for sample in group.samples:
        integer(sample.created_seq, 'durable created_seq', 1)
        _, _, checksum = ReplayCodec.prepare(sample)
        refs.append({'trajectory_id':sample.trajectory_id, 'checksum':checksum,
                     'created_seq':sample.created_seq})
    return {'key':list(group_key(group)), 'samples':refs}


def _rng(value):
    if (not isinstance(value, list) or len(value)!=3 or type(value[0]) is not int or value[0]!=3
            or not isinstance(value[1], list) or len(value[1])!=625):
        raise ReplayIntegrityError('Unsupported RNG state')
    for n in value[1][:-1]: integer(n, 'RNG word', 0, 2**32-1)
    integer(value[1][-1], 'RNG index', 0, 624)
    cached = value[2]
    if cached is not None and (type(cached) not in (int,float) or not math.isfinite(cached)):
        raise ReplayIntegrityError('Invalid RNG cache')
    return value[0], tuple(value[1]), cached


def _unpack(data):
    envelope = strict_json(data,max_bytes=MAX_STATE_BYTES)
    if (not isinstance(envelope,dict) or set(envelope)!={'format','checksum','state'}
            or envelope['format']!=FORMAT or not isinstance(envelope['checksum'],str)
            or not isinstance(envelope['state'],dict)
            or not hmac.compare_digest(envelope['checksum'],digest(envelope['state']))):
        raise ReplayIntegrityError('Mixer state checksum/format mismatch')
    return envelope['state']


def snapshot_mixer(mixer, *, policy_version):
    """Capture a post-ack, no-inflight point; only durable immutable references."""
    integer(policy_version, 'policy version')
    if mixer.inflight is not None:
        raise RuntimeError('Cannot snapshot an inflight mixed batch')
    body = {'contract':_contract(mixer.config), 'policy_version':policy_version,
            'window':[_reference(g) for g in mixer.window.values()],
            'pending':[_reference(g) for g in mixer.pending],
            'seen_online':[list(k) for k in sorted(mixer.live_online_keys())],
            'carry':str(mixer.carry), 'rng':mixer.rng.getstate(),
            'completed_steps':mixer.completed_steps, 'completed_groups':mixer.completed_groups,
            'replayed_groups':mixer.replayed_groups, 'quarantined_online':mixer.quarantined_online,
            'priorities':[{'key':list(k),'value':v,'version':mixer.priority_versions[k]}
                          for k,v in sorted(mixer.priorities.items())]}
    data = canonical({'format':FORMAT,'checksum':digest(body),'state':body})
    if len(data)>MAX_STATE_BYTES:
        raise ValueError('Mixer state exceeds bounded JSON size')
    return data


def restore_mixer(data, config, load_sample, *, expected_policy_version):
    """Build a new mixer or fail without altering an existing mixer/source store.

    load_sample takes a trajectory id and must read from the snapshot's store.
    The original created_seq and canonical payload must match. Old but valid
    window entries are preserved: normal consumption performs max_lag filtering.
    """
    integer(expected_policy_version, 'expected policy version')
    try:
        body = _unpack(data)
        fields = {'contract','policy_version','window','pending','seen_online','carry','rng',
                  'completed_steps','completed_groups','replayed_groups','quarantined_online','priorities'}
        if not isinstance(body,dict) or set(body)!=fields or body['contract']!=_contract(config):
            raise ReplayIntegrityError('Mixer state contract mismatch')
        integer(body['policy_version'],'policy version')
        if body['policy_version']!=expected_policy_version:
            raise ReplayIntegrityError('Model/mixer policy watermark mismatch')
        mixer = GroupMixer(config)
        for name in ('completed_steps','completed_groups','replayed_groups','quarantined_online'):
            setattr(mixer,name,integer(body[name],name))
        if (mixer.completed_groups!=mixer.completed_steps*config.batch_groups
                or mixer.replayed_groups>mixer.completed_groups):
            raise ReplayIntegrityError('Inconsistent completed counters')
        if not isinstance(body['carry'],str) or len(body['carry'])>128:
            raise ReplayIntegrityError('Invalid decimal carry')
        mixer.carry = Decimal(body['carry'])
        if not mixer.carry.is_finite() or not 0<=mixer.carry<1:
            raise ReplayIntegrityError('Invalid decimal carry')
        mixer.rng.setstate(_rng(body['rng']))
        if not all(isinstance(body[k],list) for k in ('window','pending','seen_online','priorities')):
            raise ReplayIntegrityError('Invalid state collections')
        if (len(body['window'])*config.group_size>config.replay_buffer_size
                or len(body['pending'])>2*config.batch_groups):
            raise ReplayIntegrityError('State exceeds configured window/pending bound')
        seen = [_key(k) for k in body['seen_online']]
        if (len(set(seen))!=len(seen) or any(k[0]!=config.store_id for k in seen)
                or len(seen)>mixer.completed_groups-mixer.replayed_groups
                    +len(body['pending'])+mixer.quarantined_online):
            raise ReplayIntegrityError('Inconsistent online generation ledger')
        mixer.seen_online = set(seen)
        mixer.history_complete = len(seen) == (mixer.completed_groups-mixer.replayed_groups
                                              +len(body['pending'])+mixer.quarantined_online)
        keys, trajectory_ids = set(), set()
        def group(entry, *, pending=False):
            if not isinstance(entry,dict) or set(entry)!={'key','samples'}:
                raise ReplayIntegrityError('Invalid group reference')
            key = _key(entry['key'])
            refs = entry['samples']
            source = source_for(config, key[0])
            foreign = source is not None and key[1] in source.group_ids
            if (key in keys or (key not in mixer.seen_online and not foreign) or (pending and foreign)
                    or not isinstance(refs,list) or len(refs)!=config.group_size):
                raise ReplayIntegrityError('Duplicate/unknown/incomplete group reference')
            samples = []
            for ref in refs:
                if not isinstance(ref,dict) or set(ref)!={'trajectory_id','checksum','created_seq'}:
                    raise ReplayIntegrityError('Invalid sample reference')
                tid = identifier(ref['trajectory_id'],'trajectory id')
                integer(ref['created_seq'],'created_seq',1)
                identity = (key[0], tid)
                if identity in trajectory_ids:
                    raise ReplayIntegrityError('Repeated trajectory reference')
                sample = load_reference(load_sample, tid, key[0], config)
                _,_,checksum = ReplayCodec.prepare(sample)
                if (sample.trajectory_id!=tid or sample.created_seq!=ref['created_seq']
                        or checksum!=ref['checksum']):
                    raise ReplayIntegrityError('Missing or changed immutable sample')
                if (not sample.train_eligible or effective_version(sample,config)>expected_policy_version
                        or (not foreign and any(getattr(sample,k)!=getattr(config,'store_id' if k=='source_store_id' else k)
                               for k in COMPATIBILITY_FIELDS))):
                    raise ReplayIntegrityError('Incompatible/future replay sample')
                samples.append(sample); trajectory_ids.add(identity)
            value = ReplayGroup(tuple(samples))
            if group_key(value)!=key:
                raise ReplayIntegrityError('Group identity mismatch')
            keys.add(key)
            return value
        mixer.window = OrderedDict((group_key(g),g) for g in map(group,body['window']))
        mixer.pending = deque(group(entry,pending=True) for entry in body['pending'])
        if mixer.seen_online != mixer.live_online_keys():
            raise ReplayIntegrityError('Compact online keys must match the live window and pending groups')
        for entry in body['priorities']:
            if not isinstance(entry,dict) or set(entry)!={'key','value','version'}:
                raise ReplayIntegrityError('Invalid priority entry')
            key = _key(entry['key']); value = entry['value']
            version = integer(entry['version'],'priority version')
            if (key not in mixer.window or key in mixer.priorities
                    or type(value) not in (int,float) or not math.isfinite(value) or value<0
                    or version>expected_policy_version or version<effective_version(mixer.window[key].samples[0],config)):
                raise ReplayIntegrityError('Invalid priority value/version/identity')
            mixer.priorities[key]=value; mixer.priority_versions[key]=version
        if ((config.replay_strategy=='priority' and set(mixer.priorities)!=set(mixer.window))
                or (config.replay_strategy!='priority' and mixer.priorities)):
            raise ReplayIntegrityError('Strategy/priority state mismatch')
        return mixer
    except ReplayIntegrityError:
        raise
    except (ValueError,TypeError,KeyError,AttributeError,InvalidOperation,OverflowError) as error:
        raise ReplayIntegrityError('Malformed or unavailable mixer state') from error


def write_state_once(path, data):
    """Durable no-overwrite sidecar publication on a local hardlink-capable FS.

    Does not publish a joint checkpoint marker or make resume safe by itself.
    A failed directory fsync may leave the complete destination, never truncate it.
    """
    _unpack(data)
    path = Path(path)
    temporary = path.with_name('.'+path.name+'.'+uuid.uuid4().hex+'.tmp')
    try:
        with temporary.open('xb') as stream:
            stream.write(data); stream.flush(); os.fsync(stream.fileno())
        os.link(temporary,path)  # Atomic and fails if path already exists.
        descriptor = os.open(path.parent,os.O_RDONLY | getattr(os,'O_DIRECTORY',0))
        try: os.fsync(descriptor)
        finally: os.close(descriptor)
    finally:
        temporary.unlink(missing_ok=True)
