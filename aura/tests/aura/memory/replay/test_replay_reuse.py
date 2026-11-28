"""Server-only frozen foreign replay compatibility, provenance and recovery."""
from dataclasses import asdict, replace
import hashlib
import json
import os
from pathlib import Path

import pytest

from aura.memory.replay.codec import ReplayCodec, canonical, digest
from aura.memory.replay.config import ReplayConfig
from aura.memory.replay.joint import publish_commit
from aura.memory.replay.ledger import ReplayLedger, batch_document
from aura.memory.replay.mixing import compatible, group_key, ReplayUnavailableError
from aura.memory.replay.reuse import (ReadOnlySources, ReuseSource, ReuseVersion, effective_version,
                                     seeded_mixer, sha256_file)
from aura.memory.replay.schema import ReplayIntegrityError, ReplayGroup
from aura.memory.replay.sqlite_store import SQLiteReplayStore
from aura.memory.replay.json_store import JsonReplayStore
from aura.memory.replay.state import snapshot_mixer, restore_mixer
from test_replay_mixing import settings


def image(root):
    """Read-only evidence excludes read-induced atime, includes directory entries."""
    result = {}
    for path in sorted(Path(root).rglob('*')):
        st = path.stat()
        result[path.relative_to(root).as_posix()] = dict(
            bytes=st.st_size, mode=st.st_mode, mtime_ns=st.st_mtime_ns,
            ino=st.st_ino, sha256=sha256_file(path) if path.is_file() else None)
    return result


@pytest.fixture
def reuse_case(settings, make_group, tmp_path):
    opened = []
    def create(backend='sqlite', strategy='uniform', ratio=1, suffix='case'):
        root = tmp_path/suffix; root.mkdir()
        source_dir = root/'source'; source_dir.mkdir()
        target_dir = root/'destination'; target_dir.mkdir()
        path = source_dir/'episodes'
        cls = SQLiteReplayStore if backend == 'sqlite' else JsonReplayStore
        store = cls(path, store_id='foreign')
        for n in range(2):
            samples = make_group(store_id='foreign', group_id=f'g{n}', version=11, reward=.2+n*.6)
            from aura.memory.replay.schema import ReplayTensor
            samples = [replace(s, reward=s.reward+i*.1, payload=dict(s.payload,
                rm_scores=ReplayTensor.from_values([0., 0., s.reward+i*.1], 'float32'),
                token_level_rewards=ReplayTensor.from_values([0., 0., s.reward+i*.1], 'float32')),
                old_log_probs=ReplayTensor.from_values([-3., 0., -4.], 'float32'))
                for i, s in enumerate(samples)]
            store.append_trajectories(samples, f'create-{n}')
        store.close()  # SQLite WAL is checkpointed; never copy just a live main DB.
        source_policy = source_dir/'policy.bin'
        source_policy.write_bytes(canonical({'all_toy_policy_tensors': [[1.0, 2.0], [3.0]]}))
        target_policy = target_dir/'policy.bin'; target_policy.write_bytes(source_policy.read_bytes())
        mapping = ReuseVersion(11, 0, str(source_policy), str(target_policy), sha256_file(source_policy))
        source = ReuseSource(backend, str(path), sha256_file(path), 'foreign', 'experiment-a',
                             'run-a', 'lineage-a', ('g0', 'g1'), (mapping,))
        config = settings(path=str(target_dir/'episodes'), backend=backend, store_id='store', batch_groups=2,
                          source_experiment_id='experiment-b', source_run_id='run-b', policy_lineage_id='lineage-b',
                          replay_strategy=strategy, replay_ratio=ratio, reuse_sources=(source,))
        readers = ReadOnlySources(config); opened.append(readers)
        return root, config, readers
    yield create
    for reader in opened: reader.close()


@pytest.mark.parametrize('backend', ['sqlite', 'json'])
@pytest.mark.parametrize('strategy', ['uniform', 'priority', 'reward_weighted'])
def test_foreign_complete_groups_consume_and_restore_without_changing_source(reuse_case, backend, strategy):
    root, config, reader = reuse_case(backend, strategy)
    before = image(root/'source')
    mixer = seeded_mixer(config, reader.load)
    raw = [ReplayCodec.encode(s) for g in mixer.window.values() for s in g.samples]
    snapshot = snapshot_mixer(mixer, policy_version=0)
    restored = restore_mixer(snapshot, config, reader.load, expected_policy_version=0)
    assert snapshot_mixer(restored, policy_version=0) == snapshot
    result = mixer.next(lambda: pytest.fail('Pure replay must not request online data'), 0)
    assert len(result.replay_groups) == 2 and not result.online_groups
    assert all(0 < p <= 1 for p in result.draw_probabilities)
    assert all(s.weight_version == 11 and effective_version(s, config) == 0 for g in result.groups for s in g.samples)
    updates = {group_key(g): .75 for g in result.groups} if strategy == 'priority' else None
    mixer.commit_success(0, updates)
    state = snapshot_mixer(mixer, policy_version=1)
    continued = restore_mixer(state, config, reader.load, expected_policy_version=1)
    a = mixer.next(lambda: pytest.fail('No fetch'), 1)
    b = continued.next(lambda: pytest.fail('No fetch'), 1)
    assert [group_key(g) for g in a.groups] == [group_key(g) for g in b.groups]
    assert a.draw_probabilities == b.draw_probabilities
    assert raw == [ReplayCodec.encode(s) for g in mixer.window.values() for s in g.samples]
    assert reader.verify() == {'sources': 1, 'unchanged': True}
    with pytest.raises(PermissionError):
        reader.stores['foreign'].append_trajectories(list(a.groups[0].samples), 'forbidden')
    reader.close()
    assert image(root/'source') == before  # Includes WAL/SHM/lock names and file mtimes.
    record('consume-'+backend+'-'+strategy, dict(source_before=before, source_after=image(root/'source'),
        mapped_version=0, original_version=11, draw_probabilities=a.draw_probabilities,
        preserved_payload=True, preserved_rng=True, strategy=strategy, backend=backend))


def record(name, value):
    root = os.environ.get('P4_REUSE_EVIDENCE_ROOT')
    if root:
        path = Path(root); path.mkdir(parents=True, exist_ok=True)
        (path/(name+'.json')).write_bytes(canonical(value))


@pytest.mark.parametrize('field', ['group_size', 'model_signature', 'tokenizer_signature', 'task_signature', 'reward_signature'])
def test_semantic_mismatch_rejected_before_training(reuse_case, field):
    _, config, _ = reuse_case()
    value = 3 if field == 'group_size' else 'wrong'
    bad = replace(config, **{field: value})
    with pytest.raises((ValueError, ReplayIntegrityError)):
        ReadOnlySources(bad)


@pytest.mark.parametrize('field', ['source_experiment_id', 'source_run_id', 'policy_lineage_id'])
def test_false_foreign_provenance_rejected(reuse_case, field):
    _, config, _ = reuse_case()
    source = replace(config.reuse_sources[0], **{field: 'wrong'})
    with pytest.raises(ReplayIntegrityError): ReadOnlySources(replace(config, reuse_sources=(source,)))


@pytest.mark.parametrize('defect', ['source_hash', 'unmapped', 'missing_group', 'wrong_store', 'policy_hash'])
def test_missing_or_unattested_source_rejected(reuse_case, defect):
    root, config, _ = reuse_case()
    source = config.reuse_sources[0]
    if defect == 'source_hash': source = replace(source, sha256='0'*64)
    elif defect == 'unmapped': source = replace(source, versions=(replace(source.versions[0], source_version=12),))
    elif defect == 'missing_group': source = replace(source, group_ids=('unknown',))
    elif defect == 'wrong_store': source = replace(source, store_id='other')
    else: (root/'destination'/'policy.bin').write_bytes(b'not the same weights')
    before = image(root/'source')
    with pytest.raises((ValueError, ReplayIntegrityError)):
        ReadOnlySources(replace(config, reuse_sources=(source,)))
    assert image(root/'source') == before


@pytest.mark.parametrize('kind', ['duplicate_source', 'duplicate_target', 'backwards', 'bool_version'])
def test_ambiguous_or_nonmonotonic_version_mapping_rejected(reuse_case, kind):
    _, config, _ = reuse_case()
    v = config.reuse_sources[0].versions[0]
    with pytest.raises(ValueError):
        if kind == 'bool_version': replace(v, source_version=True)
        else:
            second = replace(v, source_version=11 if kind == 'duplicate_source' else 12,
                             target_version=0 if kind == 'duplicate_target' else (1 if kind == 'duplicate_source' else 0))
            versions = (second, v) if kind == 'backwards' else (v, second)
            replace(config.reuse_sources[0], versions=versions)


def test_future_and_max_lag_use_mapped_coordinate_not_foreign_counter(reuse_case):
    _, config, reader = reuse_case()
    group = seeded_mixer(config, reader.load).window[('foreign', 'g0')]
    assert compatible(group, config, 0) and compatible(group, config, config.max_lag-1)
    assert not compatible(group, config, config.max_lag)
    future_source = replace(config.reuse_sources[0], versions=(replace(config.reuse_sources[0].versions[0], target_version=1),))
    future = replace(config, reuse_sources=(future_source,))
    assert not compatible(group, future, 0)
    with pytest.raises(ReplayIntegrityError):
        restore_mixer(snapshot_mixer(seeded_mixer(future, reader.load), policy_version=0),
                      future, reader.load, expected_policy_version=0)
    expired = seeded_mixer(config, reader.load)
    with pytest.raises(ReplayUnavailableError): expired.next(lambda: (), config.max_lag)


@pytest.mark.parametrize('sidecar', ['-wal', '-shm', '-journal'])
def test_live_sqlite_sidecars_rejected_without_touching_source(reuse_case, sidecar):
    root, config, reader = reuse_case()
    reader.close()
    Path(config.reuse_sources[0].path+sidecar).write_bytes(b'live')
    before = image(root/'source')
    with pytest.raises(ReplayIntegrityError): ReadOnlySources(config)
    assert image(root/'source') == before


@pytest.mark.parametrize('changed', ['source', 'policy', 'replace_source'])
def test_changed_or_replaced_mounted_input_rejected(reuse_case, changed):
    root, config, reader = reuse_case()
    path = Path(config.reuse_sources[0].path if changed != 'policy' else config.reuse_sources[0].versions[0].target_policy_path)
    if changed == 'replace_source':
        data = path.read_bytes(); path.unlink(); path.write_bytes(data)
    else:
        with path.open('ab') as stream: stream.write(b'changed')
    with pytest.raises(ReplayIntegrityError): reader.load('g0/trajectory/0', 'foreign')


def test_target_and_ledger_must_not_write_source_directory_or_alias_source(reuse_case):
    root, config, reader = reuse_case()
    with pytest.raises(ValueError): replace(config, path=str(root/'source'/'target.db'))
    with pytest.raises(ValueError): ReplayLedger(root/'source'/'ledger.db', config, reader.load)
    alias = root/'destination'/'alias'; os.link(config.reuse_sources[0].path, alias)
    with pytest.raises(ValueError): replace(config, path=str(alias))
    with pytest.raises(ValueError): ReplayLedger(alias, config, reader.load)
    assert not (root/'source'/'ledger.db').exists()


def test_mapping_is_frozen_in_contract_and_old_single_store_contract_remains_readable(reuse_case):
    _, config, reader = reuse_case()
    data = snapshot_mixer(seeded_mixer(config, reader.load), policy_version=0)
    assert ReplayConfig.from_mapping(asdict(config)) == config
    changed = replace(config, reuse_sources=(replace(config.reuse_sources[0], group_ids=('g0',)),))
    with pytest.raises(ReplayIntegrityError): restore_mixer(data, changed, reader.load, expected_policy_version=0)
    with pytest.raises(ValueError): replace(config, reuse_sources=config.reuse_sources*2)


def test_zero_ratio_bypasses_even_malformed_reuse_without_touching_sources(monkeypatch):
    import sys
    from unittest.mock import Mock
    from aura.memory.replay.launcher import prepare_replay
    import aura.memory.replay.reuse as reuse
    fake = Mock(); monkeypatch.setitem(sys.modules, 'ray', fake)
    monkeypatch.setattr(reuse, 'ReadOnlySources', Mock(side_effect=AssertionError('Must not mount')))
    assert prepare_replay({'replay': {'replay_ratio': 0, 'reuse_sources': 'invalid'}}, object()) is None
    assert not fake.mock_calls


def test_ledger_rejects_foreign_groups_mislabeled_as_online(reuse_case):
    root, config, reader = reuse_case()
    ledger = ReplayLedger(root/'destination'/'ledger', config, reader.load)
    mixer = seeded_mixer(config, reader.load); mixer.next(lambda: (), 0)
    facts = batch_document(mixer)
    for entry in facts['groups']: entry['replay'] = False
    try:
        with pytest.raises(ReplayIntegrityError, match='fresh online'):
            ledger._validate_batch(facts, 0)
    finally: ledger.close()


@pytest.mark.parametrize('backend', ['sqlite', 'json'])
def test_partial_group_is_rejected_without_any_source_repair(reuse_case, backend):
    root, config, reader = reuse_case(backend)
    reader.close()
    cls = SQLiteReplayStore if backend == 'sqlite' else JsonReplayStore
    store = cls(config.reuse_sources[0].path, store_id='foreign')
    first = store.get_trajectory('g0/trajectory/0')
    partial = replace(first, group_id='partial', trajectory_id='partial/0', group_complete=None, created_seq=None,
                      train_eligible=False, invalid_reason='incomplete-generation')
    store.append_trajectories([partial], 'partial'); store.close()
    source = replace(config.reuse_sources[0], sha256=sha256_file(config.reuse_sources[0].path), group_ids=('partial',))
    before = image(root/'source')
    with pytest.raises((ReplayIntegrityError, ValueError)): ReadOnlySources(replace(config, reuse_sources=(source,)))
    assert image(root/'source') == before


def test_json_uncommitted_tail_rejected_and_not_truncated(reuse_case):
    root, config, reader = reuse_case('json'); reader.close()
    path = Path(config.reuse_sources[0].path)
    with path.open('ab') as stream: stream.write(b'{incomplete')
    source = replace(config.reuse_sources[0], sha256=sha256_file(path))
    before = image(root/'source')
    with pytest.warns(RuntimeWarning), pytest.raises(ReplayIntegrityError):
        ReadOnlySources(replace(config, reuse_sources=(source,)))
    assert image(root/'source') == before


@pytest.mark.parametrize('strategy', ['uniform', 'priority', 'reward_weighted'])
def test_foreign_seed_genesis_lease_ack_and_paired_recovery(reuse_case, strategy):
    root, config, reader = reuse_case(strategy=strategy)
    mixer = seeded_mixer(config, reader.load)
    before = image(root/'source')
    ledger = ReplayLedger(root/'destination'/'ledger', config, reader.load)
    owner = ledger.acquire_writer('first'); ledger.initial_branch('first', owner)
    cursor = {'format': 'aura.replay.source-state.v1', 'quiescent': True, 'prefetch': None,
              'producer': {'identity': digest('source'), 'state': {}}, 'queued_batches': []}
    def save(step, version):
        path = root/'destination'/f'model-{step}'; path.write_bytes(canonical({'step': step, 'version': version}))
        return {'step': step, 'policy_version': version,
                'artifacts': {r: ['destination/'+path.name] for r in ('model', 'optimizer', 'scheduler', 'rng')}}
    try:
        publish_commit(root, config, reader.load, ledger, owner, 'first', snapshot_mixer(mixer, policy_version=0),
                       cursor, save, commit_id='genesis')
        mixer.next(lambda: pytest.fail('Must replay'), 0)
        facts = batch_document(mixer); ledger.enqueue('first', 1, 0, facts, 'update', owner)
        lease = ledger.lease('first', 1, 'lease', owner)
        updates = {group_key(g): .5 for g in mixer.inflight.groups} if strategy == 'priority' else None
        mixer.commit_success(0, updates)
        data = snapshot_mixer(mixer, policy_version=1)
        ledger.ack(lease['lease_id'], data, owner)
        publish_commit(root, config, reader.load, ledger, owner, 'first', data, cursor, save, commit_id='cut1')
        new_owner = ledger.acquire_writer('second')
        recovered = ledger.recover_branch(str(root), 'first', 'second', new_owner)
        assert recovered['body']['step'] == recovered['body']['policy_version'] == 1
        assert snapshot_mixer(restore_mixer(recovered['snapshot'], config, reader.load,
            expected_policy_version=1), policy_version=1) == data
        assert ledger.inspect('first')['counts'] == {'acked': 1}
        assert image(root/'source') == before
    finally: ledger.close()
