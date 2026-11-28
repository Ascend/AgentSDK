# Copyright (c) 2026 Huawei Technologies Co.,Ltd.
# Licensed under Mulan PSL v2 (http://license.coscl.org.cn/MulanPSL2).
"""Explicit, bounded, read-only foreign replay views (no provenance rewrite).

Version labels are NOT globally comparable. Each allowed version pair requires
two operator-attested policy artifacts with identical SHA256. Artifacts must
contain the full policy, or a canonical manifest covering ALL policy tensors;
an architecture config is not policy identity. This is integrity, not trust or
authentication of the operator's historical version-to-artifact association.
"""
from dataclasses import dataclass
from collections.abc import Sequence
from pathlib import Path
import hashlib

from .schema import ReplayGroup, ReplayIntegrityError, identifier, integer

SEMANTICS = ('group_size', 'model_signature', 'tokenizer_signature', 'task_signature', 'reward_signature')


def sha256_file(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def _sha(value):
    if not isinstance(value, str) or len(value) != 64 or any(c not in '0123456789abcdef' for c in value):
        raise ValueError('Expected lowercase SHA256')
    return value


def _path(value):
    if not isinstance(value, str) or not Path(value).is_absolute():
        raise ValueError('Reuse paths must be absolute')
    return value


@dataclass(frozen=True)
class ReuseVersion:
    source_version: int
    target_version: int
    source_policy_path: str
    target_policy_path: str
    policy_sha256: str

    def __post_init__(self):
        integer(self.source_version, 'source version', maximum=2**63-1)
        integer(self.target_version, 'target version', maximum=2**63-1)
        _path(self.source_policy_path); _path(self.target_policy_path); _sha(self.policy_sha256)


@dataclass(frozen=True)
class ReuseSource:
    backend: str
    path: str
    sha256: str
    store_id: str
    source_experiment_id: str
    source_run_id: str
    policy_lineage_id: str
    group_ids: tuple[str, ...]
    versions: tuple[ReuseVersion, ...]

    def __post_init__(self):
        if self.backend not in ('sqlite', 'json'):
            raise ValueError('Unsupported reuse backend')
        _path(self.path); _sha(self.sha256)
        for name in ('store_id', 'source_experiment_id', 'source_run_id', 'policy_lineage_id'):
            identifier(getattr(self, name), name)
        if (not isinstance(self.group_ids, Sequence) or isinstance(self.group_ids, (str, bytes))
                or not 1 <= len(self.group_ids) <= 4096
                or len(set(self.group_ids)) != len(self.group_ids)):
            raise ValueError('Reuse requires explicit unique bounded group IDs')
        for value in self.group_ids: identifier(value, 'reuse group ID')
        if (not isinstance(self.versions, Sequence) or isinstance(self.versions, (str, bytes))
                or not 1 <= len(self.versions) <= 256):
            raise ValueError('Reuse needs bounded explicit policy-version pairs')
        versions = tuple(v if isinstance(v, ReuseVersion) else ReuseVersion(**dict(v)) for v in self.versions)
        if any(a.source_version >= b.source_version or a.target_version >= b.target_version
               for a, b in zip(versions, versions[1:])):
            raise ValueError('Version mapping must be strictly increasing and injective; no extrapolation')
        object.__setattr__(self, 'group_ids', tuple(self.group_ids))
        object.__setattr__(self, 'versions', versions)


def normalize_sources(value, config):
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) > 16:
        raise ValueError('At most 16 explicit read-only reuse sources')
    sources = tuple(v if isinstance(v, ReuseSource) else ReuseSource(**dict(v)) for v in value)
    ids = [s.store_id for s in sources]
    if config.store_id in ids or len(set(ids)) != len(ids):
        raise ValueError('Foreign source store IDs must be distinct from each other and the destination')
    if sum(len(s.group_ids) for s in sources)*config.group_size > config.replay_buffer_size:
        raise ValueError('Reuse seed groups exceed the replay window capacity')
    paths = [Path(s.path).resolve() for s in sources]
    if len(set(paths)) != len(paths):
        raise ValueError('Duplicate reuse path')
    for path in paths:
        # Source directory is reserved read-only, including control sidecars.
        if Path(config.path).resolve().is_relative_to(path.parent):
            raise ValueError('Destination must be outside each read-only source directory')
        if Path(config.path).exists() and path.exists() and Path(config.path).samefile(path):
            raise ValueError('Destination aliases a read-only source inode')
    return sources


def source_for(config, store_id):
    return next((s for s in config.reuse_sources if s.store_id == store_id), None)


def effective_version(sample, config):
    if sample.source_store_id == config.store_id:
        return sample.weight_version
    source = source_for(config, sample.source_store_id)
    if source is None or sample.group_id not in source.group_ids:
        raise ReplayIntegrityError('Foreign group not explicitly authorized')
    if any(getattr(sample, k) != getattr(source, k) for k in
           ('source_experiment_id', 'source_run_id', 'policy_lineage_id')):
        raise ReplayIntegrityError('Foreign provenance mismatch')
    if any(getattr(sample, k) != getattr(config, k) for k in SEMANTICS):
        raise ReplayIntegrityError('Foreign semantic compatibility mismatch')
    for mapping in source.versions:
        if sample.weight_version == mapping.source_version:
            return mapping.target_version
    raise ReplayIntegrityError('No attested mapping for this behavior-policy version')


def load_reference(load_sample, tid, store_id, config):
    # Preserve the original one-argument loader API for single-store callers.
    return load_sample(tid) if store_id == config.store_id else load_sample(tid, store_id)


def seed_groups(config, load_sample):
    groups = []
    for source in config.reuse_sources:
        for gid in source.group_ids:
            # Fixed member IDs come from the reader, not naming conventions.
            members = load_sample(None, source.store_id, gid)
            group = ReplayGroup(tuple(members))
            if (not group.complete or any(not s.group_complete or not s.train_eligible for s in group.samples)
                    or group.samples[0].source_store_id != source.store_id or group.samples[0].group_id != gid):
                raise ReplayIntegrityError('Missing/incomplete/ineligible reuse group')
            for sample in group.samples: effective_version(sample, config)
            groups.append(group)
    return tuple(groups)


def seeded_mixer(config, load_sample):
    from .mixing import GroupMixer, group_key
    mixer = GroupMixer(config)
    for group in seed_groups(config, load_sample):
        key = group_key(group)
        mixer.window[key] = group
        if config.replay_strategy == 'priority':
            # Neutral cold-start score, NOT a stored/native advantage. Zero
            # would starve foreign seeds behind positive online priorities.
            # Recompute from
            # actual GRPO advantages on the first successful destination update.
            mixer.priorities[key] = 1.0
            mixer.priority_versions[key] = effective_version(group.samples[0], config)
    return mixer


class ReadOnlySources:
    """Actor-local frozen readers; no source migration, WAL or writer lock.

    Mount-time SHA + per-read inode/size/mtime/ctime guards. Sources must be
    offline frozen exports. Concurrent writers are unsupported, not snapshotted.
    """
    def __init__(self, config):
        self.config = config
        self.stores, self.stamps, self.policy_stamps = {}, {}, {}
        try:
            for source in config.reuse_sources:
                path = Path(source.path)
                if path.is_symlink(): raise ReplayIntegrityError('Reuse source symlink forbidden')
                self._verify(source)
                from .sqlite_store import SQLiteReplayStore
                from .json_store import JsonReplayStore
                cls = SQLiteReplayStore if source.backend == 'sqlite' else JsonReplayStore
                extra = {'immutable': True} if source.backend == 'sqlite' else {}
                store = cls(source.path, read_only=True, store_id=source.store_id, **extra)
                self.stores[source.store_id] = store
                if getattr(store, 'ignored_tail_bytes', 0):
                    raise ReplayIntegrityError('Uncommitted JSON tail is not a frozen source')
                self.stamps[source.store_id] = self._stamp(path)
                self.policy_stamps[source.store_id] = {
                    p: self._stamp(p) for v in source.versions
                    for p in (v.source_policy_path, v.target_policy_path)}
                self._verify(source)
            seed_groups(config, self.load)
        except BaseException:
            self.close(); raise

    @staticmethod
    def _stamp(path):
        st = Path(path).stat()
        return st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns, st.st_ctime_ns

    def _verify(self, source):
        if sha256_file(source.path) != source.sha256:
            raise ReplayIntegrityError('Frozen source SHA256 mismatch')
        for version in source.versions:
            for path in (version.source_policy_path, version.target_policy_path):
                if sha256_file(path) != version.policy_sha256:
                    raise ReplayIntegrityError('Source/target policy artifacts differ from attested mapping')

    def verify(self):
        for source in self.config.reuse_sources:
            self._guard(source.store_id); self._verify(source)
        return {'sources': len(self.stores), 'unchanged': True}

    def _guard(self, store_id):
        source = source_for(self.config, store_id)
        if source is None or self._stamp(source.path) != self.stamps[store_id]:
            raise ReplayIntegrityError('Read-only source was changed/replaced while mounted')
        if any(self._stamp(p) != stamp for p, stamp in self.policy_stamps[store_id].items()):
            raise ReplayIntegrityError('Attested policy artifact changed while mounted')
        if source.backend == 'sqlite' and any(Path(source.path+x).exists() for x in ('-wal', '-shm', '-journal')):
            raise ReplayIntegrityError('Live SQLite sidecar is forbidden for frozen reuse')

    def load(self, tid, store_id, group_id=None):
        self._guard(store_id)
        store = self.stores[store_id]
        result = (store.query(group_id=group_id, train_eligible=True, limit=256)
                  if group_id is not None else store.get_trajectory(tid))
        self._guard(store_id)
        return result

    def close(self):
        for store in self.stores.values(): store.close()
        self.stores.clear()
