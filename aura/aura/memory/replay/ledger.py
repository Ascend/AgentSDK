# Copyright (c) 2026 Huawei Technologies Co.,Ltd.
# Licensed under Mulan PSL v2 (http://license.coscl.org.cn/MulanPSL2).
"""Actor-owned, run-private P4 consumption ledger; ACK is not a model commit.

No lease TTL/reclaim: an uncertain update requires recovery from a paired commit
on a new fenced branch. Old pending/leased/acked evidence is never overwritten.
"""
from contextlib import contextmanager
from collections import OrderedDict
from decimal import Decimal
from pathlib import Path
import hashlib
import sqlite3
import threading
import uuid
import random

from .codec import canonical, digest, strict_json
from .mixing import compatible, group_key
from .schema import ReplayConflictError, ReplayGroup, ReplayIntegrityError, identifier, integer
from .state import MAX_STATE_BYTES, _contract, _reference, restore_mixer


class ReplayFencedError(RuntimeError):
    """A stale writer or an old recovery branch cannot mutate current state."""


def blob_sha(data):
    return hashlib.sha256(data).hexdigest()


def batch_document(mixer):
    if mixer.inflight is None:
        raise RuntimeError('A lease requires an actual inflight batch')
    count = len(mixer.inflight.online_groups)
    return {'groups':[dict(_reference(g), replay=i>=count)
                      for i,g in enumerate(mixer.inflight.groups)]}


class ReplayLedger:
    """SQLite control state for either SQLite or JSON trajectory storage.

    Production writes are exposed by Episode, not by trainer-side direct DB use.
    load_sample is an actor-local, read-only immutable trajectory lookup.
    """
    def __init__(self, path, config, load_sample):
        path = Path(path)
        if not path.is_absolute() or path.resolve()==Path(config.path).resolve():
            raise ValueError('Ledger needs its own absolute, run-private path')
        if any(path.resolve().is_relative_to(Path(s.path).resolve().parent) for s in config.reuse_sources):
            raise ValueError('Ledger must be outside read-only source directories')
        if path.exists() and any(path.samefile(s.path) for s in config.reuse_sources):
            raise ValueError('Ledger aliases a read-only source inode')
        self.config, self.load_sample = config, load_sample
        self.contract = _contract(config)
        self.contract_digest = digest(self.contract)
        self.lock = threading.RLock()
        self.conn = sqlite3.connect(str(path),isolation_level=None,check_same_thread=False,timeout=30)
        self.conn.row_factory = sqlite3.Row
        try:
            tables = {r[0] for r in self.conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
            if tables and 'p4_meta' not in tables:
                raise ReplayIntegrityError('Refusing to add a ledger to an unrelated database')
            if 'p4_meta' in tables and self.conn.execute(
                    'SELECT schema_version FROM p4_meta WHERE id=1').fetchone()[0] != 2:
                raise ReplayIntegrityError('Ledger v2 is required for compact online history; v1 is not migrated')
            self.conn.execute('PRAGMA journal_mode=WAL')
            self.conn.execute('PRAGMA synchronous=FULL')
            self.conn.execute('PRAGMA foreign_keys=ON')
            with self._tx():
                for sql in (
                    'CREATE TABLE IF NOT EXISTS p4_meta(id INTEGER PRIMARY KEY,schema_version INTEGER NOT NULL,contract BLOB NOT NULL,epoch INTEGER NOT NULL)',
                    'CREATE TABLE IF NOT EXISTS p4_owners(request_id TEXT PRIMARY KEY,epoch INTEGER NOT NULL UNIQUE)',
                    'CREATE TABLE IF NOT EXISTS p4_branches(id TEXT PRIMARY KEY,owner_epoch INTEGER NOT NULL,base_commit TEXT,base_step INTEGER NOT NULL,head_step INTEGER NOT NULL,head_version INTEGER NOT NULL,blocked TEXT)',
                    '''CREATE TABLE IF NOT EXISTS p4_batches(branch TEXT NOT NULL REFERENCES p4_branches(id),step INTEGER NOT NULL,
                        request_id TEXT NOT NULL,version INTEGER NOT NULL,facts BLOB NOT NULL,facts_digest TEXT NOT NULL,
                        status TEXT NOT NULL,lease_id TEXT UNIQUE,lease_request TEXT,lease_epoch INTEGER,
                        snapshot BLOB,snapshot_sha TEXT,model_version INTEGER,PRIMARY KEY(branch,step),UNIQUE(branch,request_id))''',
                    '''CREATE TABLE IF NOT EXISTS p4_commits(sequence INTEGER PRIMARY KEY AUTOINCREMENT,id TEXT NOT NULL UNIQUE,
                        branch TEXT NOT NULL REFERENCES p4_branches(id),step INTEGER NOT NULL,version INTEGER NOT NULL,
                        owner_epoch INTEGER NOT NULL,body BLOB NOT NULL,body_digest TEXT NOT NULL,snapshot BLOB NOT NULL,sealed INTEGER NOT NULL)''',
                    '''CREATE TABLE IF NOT EXISTS p4_online_history(branch TEXT NOT NULL REFERENCES p4_branches(id),
                        source_id TEXT NOT NULL,group_id TEXT NOT NULL,step INTEGER NOT NULL,
                        PRIMARY KEY(branch,source_id,group_id))'''):
                    self.conn.execute(sql)
                self.conn.execute('INSERT OR IGNORE INTO p4_meta VALUES(1,2,?,0)',(canonical(self.contract),))
                row = self.conn.execute('SELECT * FROM p4_meta WHERE id=1').fetchone()
                if row['schema_version']!=2 or row['contract']!=canonical(self.contract):
                    raise ReplayIntegrityError('Ledger run/algorithm contract mismatch')
        except BaseException:
            self.close(); raise

    @contextmanager
    def _tx(self):
        with self.lock:
            self.conn.execute('BEGIN IMMEDIATE')
            try:
                yield
                self.conn.execute('COMMIT')
            except BaseException:
                self.conn.execute('ROLLBACK'); raise

    def close(self):
        with self.lock:
            if self.conn is not None:
                self.conn.close(); self.conn=None

    def _fence(self, writer):
        integer(writer,'writer epoch',1)
        if self.conn.execute('SELECT epoch FROM p4_meta WHERE id=1').fetchone()[0]!=writer:
            raise ReplayFencedError('Writer was superseded; recover a committed branch')

    def _branch(self, branch, writer):
        self._fence(writer)
        identifier(branch,'branch')
        row = self.conn.execute('SELECT * FROM p4_branches WHERE id=?',(branch,)).fetchone()
        if row is None or row['owner_epoch']!=writer:
            raise ReplayFencedError('Branch is not owned by this writer; fork from a paired point')
        return row

    def acquire_writer(self, request_id):
        identifier(request_id,'writer request')
        with self._tx():
            old = self.conn.execute('SELECT epoch FROM p4_owners WHERE request_id=?',(request_id,)).fetchone()
            if old:
                self._fence(old[0]); return old[0]
            epoch = self.conn.execute('SELECT epoch FROM p4_meta WHERE id=1').fetchone()[0]+1
            self.conn.execute('UPDATE p4_meta SET epoch=? WHERE id=1',(epoch,))
            self.conn.execute('INSERT INTO p4_owners VALUES(?,?)',(request_id,epoch))
            return epoch

    def initial_branch(self, branch, writer, policy_version=0):
        identifier(branch,'branch'); integer(policy_version,'policy version')
        with self._tx():
            self._fence(writer)
            old = self.conn.execute('SELECT * FROM p4_branches WHERE id=?',(branch,)).fetchone()
            if old:
                if (old['owner_epoch']!=writer or old['head_step']!=0 or old['head_version']!=policy_version
                        or old['base_commit'] is not None or old['blocked'] is not None
                        or self.conn.execute('SELECT 1 FROM p4_commits WHERE branch=?',(branch,)).fetchone()
                        or self.conn.execute('SELECT 1 FROM p4_batches WHERE branch=?',(branch,)).fetchone()):
                    raise ReplayConflictError('Initial branch is no longer an empty startup')
                return branch
            if self.conn.execute('SELECT 1 FROM p4_branches').fetchone():
                raise ReplayConflictError('Existing runs must fork a verified paired commit, not reset to zero')
            self.conn.execute('INSERT INTO p4_branches VALUES(?,?,NULL,0,0,?,NULL)',(branch,writer,policy_version))
            return branch

    def _validate_batch(self, facts, version):
        if not isinstance(facts,dict) or set(facts)!={'groups'} or not isinstance(facts['groups'],list):
            raise ReplayIntegrityError('Malformed consumption batch')
        if len(facts['groups'])!=self.config.batch_groups:
            raise ReplayIntegrityError('Batch must contain exactly the configured complete groups')
        keys, ids = set(), set()
        replay_started=False
        for entry in facts['groups']:
            if (not isinstance(entry,dict) or set(entry)!={'key','samples','replay'}
                    or type(entry['replay']) is not bool or not isinstance(entry['samples'],list)
                    or len(entry['samples'])!=self.config.group_size):
                raise ReplayIntegrityError('Malformed complete group facts')
            if replay_started and not entry['replay']:
                raise ReplayIntegrityError('Online/replay group order mismatch')
            replay_started |= entry['replay']
            samples=[]
            from .state import _key
            source_id = _key(entry['key'])[0]
            if not entry['replay'] and source_id != self.config.store_id:
                raise ReplayIntegrityError('Foreign history cannot be leased as fresh online generation')
            for ref in entry['samples']:
                if not isinstance(ref,dict) or set(ref)!={'trajectory_id','checksum','created_seq'}:
                    raise ReplayIntegrityError('Malformed durable reference')
                tid=identifier(ref['trajectory_id'],'trajectory id')
                integer(ref['created_seq'],'created_seq',1)
                from .joint import _hex
                _hex(ref['checksum'])
                identity = (source_id, tid)
                if identity in ids: raise ReplayIntegrityError('Duplicate trajectory within an update')
                from .reuse import load_reference
                sample=load_reference(self.load_sample,tid,source_id,self.config)
                if sample is None: raise ReplayIntegrityError('Missing durable trajectory')
                samples.append(sample);ids.add(identity)
            group=ReplayGroup(tuple(samples));key=group_key(group)
            if (key in keys or _reference(group)!={k:entry[k] for k in ('key','samples')}
                    or not compatible(group,self.config,version)):
                raise ReplayIntegrityError('Changed, incompatible, stale or repeated group')
            keys.add(key)

    def _seen_at(self, branch, step, keys):
        return any(self.conn.execute(
            'SELECT 1 FROM p4_online_history WHERE branch=? AND source_id=? AND group_id=? AND step<=?',
            (branch, source, group, step)).fetchone() is not None for source, group in keys)

    def seen_online(self, branch, step, keys, writer):
        from .state import _key
        keys = [_key(list(key)) for key in keys]
        with self.lock:
            current = self._branch(branch, writer)
            if step != current['head_step']:
                raise ReplayConflictError('Online history query must use the acknowledged branch head')
            return self._seen_at(branch, step, keys)

    def _restore_head(self, branch):
        state = restore_mixer(self._snapshot_at(branch, branch['head_step']), self.config, self.load_sample,
                              expected_policy_version=branch['head_version'])
        state.history_contains = lambda keys: self._seen_at(branch['id'], branch['head_step'], keys)
        return state

    def enqueue(self, branch, step, policy_version, facts, request_id, writer):
        integer(step,'step',1);integer(policy_version,'policy version');identifier(request_id,'enqueue request')
        encoded=canonical(facts)
        if len(encoded)>MAX_STATE_BYTES:raise ValueError('Consumption facts exceed limit')
        self._validate_batch(facts,policy_version)
        with self._tx():
            b=self._branch(branch,writer)
            old=self.conn.execute('SELECT * FROM p4_batches WHERE branch=? AND step=?',(branch,step)).fetchone()
            if old:
                if old['request_id']!=request_id or old['version']!=policy_version or old['facts']!=encoded:
                    raise ReplayConflictError('Update identity reused with different facts')
                return self._receipt(old)
            if b['blocked'] is not None or b['base_commit'] is None:
                raise RuntimeError('Publish a complete initial commit / finish the checkpoint barrier first')
            if step!=b['head_step']+1 or policy_version!=b['head_version']:
                raise ReplayConflictError('Update skipped or model/ledger version mismatch')
            if self.conn.execute('SELECT 1 FROM p4_batches WHERE branch=? AND request_id=?',(branch,request_id)).fetchone():
                raise ReplayConflictError('Enqueue request reused for another step')
            old_state=self._restore_head(b)
            self._selection(old_state,facts,policy_version)
            self.conn.execute('INSERT INTO p4_batches(branch,step,request_id,version,facts,facts_digest,status) VALUES(?,?,?,?,?,?,?)',
                              (branch,step,request_id,policy_version,encoded,digest(facts),'pending'))
            return self._receipt(self.conn.execute('SELECT * FROM p4_batches WHERE branch=? AND step=?',(branch,step)).fetchone())

    @staticmethod
    def _receipt(row):
        return {k:row[k] for k in ('branch','step','version','facts_digest','status','lease_id','snapshot_sha','model_version')}

    def lease(self, branch, step, request_id, writer):
        integer(step,'step',1);identifier(request_id,'lease request')
        with self._tx():
            b=self._branch(branch,writer)
            if b['blocked'] is not None:raise RuntimeError('Checkpoint barrier active')
            row=self.conn.execute('SELECT * FROM p4_batches WHERE branch=? AND step=?',(branch,step)).fetchone()
            if row is None:raise ReplayConflictError('Update was not durably enqueued')
            if row['status']!='pending':
                if row['lease_request']!=request_id or row['lease_epoch']!=writer:
                    raise ReplayConflictError('Cannot reclaim another or uncertain lease')
                return self._receipt(row)
            if step!=b['head_step']+1:raise ReplayConflictError('Out-of-order lease')
            self.conn.execute("UPDATE p4_batches SET status='leased',lease_id=?,lease_request=?,lease_epoch=? WHERE branch=? AND step=?",
                              (uuid.uuid4().hex,request_id,writer,branch,step))
            return self._receipt(self.conn.execute('SELECT * FROM p4_batches WHERE branch=? AND step=?',(branch,step)).fetchone())

    def ack(self, lease_id, snapshot, writer):
        identifier(lease_id,'lease id')
        if not isinstance(snapshot,bytes):raise TypeError('Snapshot must be bytes')
        with self._tx():
            self._fence(writer)
            row=self.conn.execute('SELECT * FROM p4_batches WHERE lease_id=?',(lease_id,)).fetchone()
            if row is None:raise ReplayConflictError('Unknown lease')
            b=self._branch(row['branch'],writer)
            if row['lease_epoch']!=writer:raise ReplayFencedError('Lease belongs to a stale writer')
            state=restore_mixer(snapshot,self.config,self.load_sample,expected_policy_version=row['version']+1)
            if state.completed_steps!=row['step']:raise ReplayIntegrityError('ACK snapshot/update mismatch')
            checksum=blob_sha(snapshot)
            if row['status']=='acked':
                if row['snapshot']!=snapshot:raise ReplayConflictError('ACK changed after commit')
                return self._receipt(row)
            if row['status']!='leased' or b['head_step']+1!=row['step'] or b['blocked'] is not None:
                raise ReplayConflictError('Invalid ACK state/sequence')
            old=self._restore_head(b)
            replay_count=sum(g['replay'] for g in strict_json(row['facts'])['groups'])
            if state.replayed_groups!=old.replayed_groups+replay_count:
                raise ReplayIntegrityError('ACK replay counter does not match leased facts')
            fetched=self._validate_transition(old,state,strict_json(row['facts']),row['version'])
            self.conn.executemany('INSERT INTO p4_online_history VALUES(?,?,?,?)',
                                  [(row['branch'], source, group, row['step']) for source, group in fetched])
            self.conn.execute("UPDATE p4_batches SET status='acked',snapshot=?,snapshot_sha=?,model_version=? WHERE lease_id=?",
                              (snapshot,checksum,row['version']+1,lease_id))
            self.conn.execute('UPDATE p4_branches SET head_step=?,head_version=? WHERE id=?',
                              (row['step'],row['version']+1,row['branch']))
            return self._receipt(self.conn.execute('SELECT * FROM p4_batches WHERE lease_id=?',(lease_id,)).fetchone())

    def _selection(self,old,facts,version):
        from .sampling import draw_groups
        config=self.config
        pending=[g for g in old.pending if compatible(g,config,version)]
        excluded={group_key(g) for g in pending}
        candidates=[g for key,g in old.window.items() if key not in excluded and compatible(g,config,version)
                    and (config.replay_strategy!='priority' or key in old.priorities)]
        warm=old.completed_steps<config.warmup_online_steps
        target=Decimal(config.batch_groups)*Decimal(str(config.replay_ratio))+old.carry
        requested=0 if warm else int(target)
        carry=old.carry if warm else target-requested
        rng=random.Random(0);rng.setstate(old.rng.getstate())
        chosen,_=draw_groups(candidates,min(requested,len(candidates)),rng,config,old.priorities)
        online=[tuple(g['key']) for g in facts['groups'] if not g['replay']]
        actual=[tuple(g['key']) for g in facts['groups'] if g['replay']]
        if (actual!=[group_key(g) for g in chosen]
                or (not warm and config.replay_ratio==1 and len(chosen)!=config.batch_groups)):
            raise ReplayIntegrityError('Lease sampling sequence does not match the committed RNG/window')
        prefix=[group_key(g) for g in pending[:len(online)]]
        if (online[:len(prefix)]!=prefix or any(k in old.window for k in online)
                or old.history_contains(set(online)-excluded)):
            raise ReplayIntegrityError('Lease online FIFO/repeated generation mismatch')
        return pending,carry,rng.getstate()

    def _validate_transition(self,old,new,facts,version):
        pending,carry,rng_state=self._selection(old,facts,version)
        online=[g for g in facts['groups'] if not g['replay']]
        consumed={tuple(g['key']) for g in facts['groups']}
        expected=OrderedDict(old.window)
        for entry in online:
            from .reuse import load_reference
            group=ReplayGroup(tuple(load_reference(self.load_sample,ref['trajectory_id'],entry['key'][0],self.config)
                                    for ref in entry['samples']))
            expected[group_key(group)]=group
        while len(expected)*self.config.group_size>self.config.replay_buffer_size:expected.popitem(last=False)
        if ([_reference(g) for g in expected.values()]!=[_reference(g) for g in new.window.values()]
                or new.carry!=carry or new.rng.getstate()!=rng_state
                or new.quarantined_online!=old.quarantined_online+len(old.pending)-len(pending)):
            raise ReplayIntegrityError('ACK changed window/order, RNG, carry or quarantine state')
        old_keys={group_key(g) for g in pending}
        new_keys={group_key(g) for g in new.pending}
        needed=len(online)
        fetch_required=len(pending)<needed
        expected_new={tuple(g['key']) for g in online if tuple(g['key']) not in old_keys}|(new_keys-old_keys)
        left=pending[needed:]
        if (old.history_contains(expected_new)
                or len(expected_new)!=(self.config.batch_groups if fetch_required else 0)
                or len(new.pending)!=len(pending)+(self.config.batch_groups if fetch_required else 0)-needed
                or [_reference(g) for g in list(new.pending)[:len(left)]]!=[_reference(g) for g in left]
                or new_keys&consumed or any(not compatible(g,self.config,version) for g in new.pending)):
            raise ReplayIntegrityError('ACK lost/reordered pending or generation state')
        if self.config.replay_strategy=='priority':
            for key in new.window:
                if key in consumed:
                    if new.priority_versions[key]!=version:raise ReplayIntegrityError('Priority update version mismatch')
                elif (new.priorities[key]!=old.priorities[key] or new.priority_versions[key]!=old.priority_versions[key]):
                    raise ReplayIntegrityError('Unconsumed priority changed')
        return expected_new

    def _snapshot_at(self, branch, step):
        if step==branch['base_step']:
            row=self.conn.execute('SELECT snapshot FROM p4_commits WHERE id=?',(branch['base_commit'],)).fetchone()
        else:
            row=self.conn.execute("SELECT snapshot FROM p4_batches WHERE branch=? AND step=? AND status='acked'",(branch['id'],step)).fetchone()
        if row is None:raise ReplayIntegrityError('Missing authoritative replay state')
        return row[0]

    def prepare_commit(self, body, snapshot, writer):
        from .joint import validate_body
        validate_body(body,self.config)
        state=restore_mixer(snapshot,self.config,self.load_sample,expected_policy_version=body['policy_version'])
        if state.completed_steps!=body['step'] or blob_sha(snapshot)!=body['replay_sha256']:
            raise ReplayIntegrityError('Joint replay snapshot mismatch')
        encoded=canonical(body)
        with self._tx():
            b=self._branch(body['branch'],writer)
            old=self.conn.execute('SELECT * FROM p4_commits WHERE id=?',(body['commit_id'],)).fetchone()
            if old:
                if old['body']!=encoded or old['snapshot']!=snapshot or old['owner_epoch']!=writer:
                    raise ReplayConflictError('Joint commit identity conflict')
                return self.lookup_commit(body['commit_id'])
            if (b['blocked'] is not None or body['step']!=b['head_step'] or body['policy_version']!=b['head_version']
                    or self.conn.execute("SELECT 1 FROM p4_batches WHERE branch=? AND status!='acked'",(b['id'],)).fetchone()):
                raise ReplayConflictError('Joint commit requires an acknowledged quiescent head')
            if b['base_commit'] is None:
                from .reuse import seeded_mixer
                from .state import snapshot_mixer
                expected = seeded_mixer(self.config, self.load_sample)
                if (state.completed_steps or state.seen_online or state.pending
                        or snapshot != snapshot_mixer(expected, policy_version=body['policy_version'])):
                    raise ReplayIntegrityError('Initial commit must precede any generated/consumed data')
            elif snapshot!=self._snapshot_at(b,b['head_step']):
                raise ReplayIntegrityError('Joint snapshot is not the ledger ACK state')
            self.conn.execute('INSERT INTO p4_commits(id,branch,step,version,owner_epoch,body,body_digest,snapshot,sealed) VALUES(?,?,?,?,?,?,?,?,0)',
                              (body['commit_id'],b['id'],body['step'],body['policy_version'],writer,encoded,digest(body),snapshot))
            self.conn.execute('UPDATE p4_branches SET blocked=? WHERE id=?',(body['commit_id'],b['id']))
            return self.lookup_commit(body['commit_id'])

    def lookup_commit(self, commit_id, *, online_keys=()):
        identifier(commit_id,'commit id')
        with self.lock:
            row=self.conn.execute('SELECT * FROM p4_commits WHERE id=?',(commit_id,)).fetchone()
            if row is None:raise ReplayIntegrityError('Joint marker is not registered in this ledger')
            body=strict_json(row['body'])
            if digest(body)!=row['body_digest'] or blob_sha(row['snapshot'])!=body['replay_sha256']:
                raise ReplayIntegrityError('Corrupt ledger commit record')
            from .state import _key
            keys = [_key(list(key)) for key in online_keys]
            if self._seen_at(row['branch'], row['step'], keys):
                raise ReplayIntegrityError('Checkpoint queue repeats previously fetched online generation')
            return {'body':body,'body_digest':row['body_digest'],'sequence':row['sequence'],'sealed':bool(row['sealed'])}

    def seal_commit(self, commit_id, writer, root):
        from .joint import read_commit
        read_commit(root,commit_id,self.config,self.load_sample,self.lookup_commit)
        with self._tx():
            row=self.conn.execute('SELECT * FROM p4_commits WHERE id=?',(commit_id,)).fetchone()
            if row is None:raise ReplayConflictError('Unknown prepared commit')
            b=self._branch(row['branch'],writer)
            if row['owner_epoch']!=writer:raise ReplayFencedError('Prepared commit owner was superseded')
            if not row['sealed']:
                if b['blocked']!=commit_id:raise ReplayConflictError('Checkpoint barrier changed')
                self.conn.execute('UPDATE p4_commits SET sealed=1 WHERE id=?',(commit_id,))
                self.conn.execute('UPDATE p4_branches SET blocked=NULL,base_commit=COALESCE(base_commit,?) WHERE id=?',(commit_id,b['id']))
            return self.lookup_commit(commit_id)

    def _after_marker_written(self):
        """No-op seam for server fault tests, including abrupt process death."""

    def publish_marker(self, commit_id, writer, root):
        """Fence check + final marker publication under one SQLite writer lock.

        A death after durable marker publication but before SQL seal is recoverable
        from the registered prepare. A superseded writer cannot publish later.
        """
        from .joint import verify_prepared, _read_small, _write_once, validate_body
        with self._tx():
            row=self.conn.execute('SELECT * FROM p4_commits WHERE id=?',(commit_id,)).fetchone()
            if row is None:raise ReplayConflictError('Unknown prepared commit')
            b=self._branch(row['branch'],writer)
            if row['owner_epoch']!=writer:raise ReplayFencedError('Prepared commit owner was superseded')
            body=strict_json(row['body'])
            validate_body(body,self.config)
            checked=verify_prepared(root,body,self.config,self.load_sample,self.lookup_commit)
            path=Path(checked['root'])/'replay-commits'/commit_id/'COMMITTED.json'
            marker=canonical({'body':body,'checksum':digest(body)})
            if row['sealed']:
                if not path.exists() or _read_small(path)!=marker:raise ReplayIntegrityError('Sealed marker missing or changed')
                return self.lookup_commit(commit_id)
            if b['blocked']!=commit_id:raise ReplayConflictError('Checkpoint barrier changed')
            if path.exists():
                if _read_small(path)!=marker:raise ReplayIntegrityError('Marker conflict')
            else:_write_once(path,marker)
            self._after_marker_written()
            self.conn.execute('UPDATE p4_commits SET sealed=1 WHERE id=?',(commit_id,))
            self.conn.execute('UPDATE p4_branches SET blocked=NULL,base_commit=COALESCE(base_commit,?) WHERE id=?',(commit_id,b['id']))
            return self.lookup_commit(commit_id)

    def recover_branch(self, root, from_branch, new_branch, writer, commit_id=None):
        from .joint import select_commit
        identifier(from_branch,'source branch');identifier(new_branch,'new branch')
        with self.lock:
            source=self.conn.execute('SELECT * FROM p4_branches WHERE id=?',(from_branch,)).fetchone()
            if source is None:raise ReplayIntegrityError('Unknown recovery branch')
            base=source['base_commit']
        chosen=select_commit(root,self.config,self.load_sample,self.lookup_commit,
                             from_branch=from_branch,base_commit=base,commit_id=commit_id)
        body=chosen['body']
        with self._tx():
            self._fence(writer)
            if self.conn.execute('SELECT 1 FROM p4_branches WHERE id=?',(new_branch,)).fetchone():
                raise ReplayConflictError('Recovery must create a fresh branch; old history is immutable')
            self.conn.execute('INSERT INTO p4_branches VALUES(?,?,?,?,?,?,NULL)',
                              (new_branch,writer,body['commit_id'],body['step'],body['step'],body['policy_version']))
            self.conn.execute('INSERT INTO p4_online_history SELECT ?,source_id,group_id,step '
                              'FROM p4_online_history WHERE branch=? AND step<=?',
                              (new_branch,body['branch'],body['step']))
        chosen['branch']=new_branch
        return chosen

    def inspect(self, branch):
        with self.lock:
            row=self.conn.execute('SELECT * FROM p4_branches WHERE id=?',(branch,)).fetchone()
            if row is None:raise ReplayIntegrityError('Unknown branch')
            counts=dict(self.conn.execute('SELECT status,COUNT(*) FROM p4_batches WHERE branch=? GROUP BY status',(branch,)).fetchall())
            return {'branch':dict(row),'counts':counts}
