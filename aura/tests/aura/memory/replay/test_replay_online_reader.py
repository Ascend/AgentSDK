"""Avoid repeated foreign reseeding at real source snapshot/restore boundaries."""
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock
import sys

import pytest

from aura.memory.replay.codec import digest
from aura.memory.replay.reuse import ReuseSource, ReuseVersion
from aura.memory.replay.runtime import OnlineReplayReader, ReplaySession
from aura.memory.replay.source import DataManagerSource
from aura.memory.replay.schema import ReplayIntegrityError
from test_replay_mixing import settings


@pytest.fixture
def online_reader_case(settings, make_group, tmp_path, monkeypatch):
    source = ReuseSource('sqlite', str(tmp_path/'frozen'/'never-opened'), '0'*64, 'foreign',
        'experiment-a', 'run-a', 'lineage-a', ('unused',),
        (ReuseVersion(0, 0, str(tmp_path/'a'), str(tmp_path/'b'), '0'*64),))
    cfg = settings(path=str(tmp_path/'private'/'episodes'), batch_groups=2, reuse_sources=(source,))
    members = [replace(s, created_seq=2*g+s.sample_index+1, group_complete=True, source_batch_id='fixed-batch')
               for g in range(2) for s in make_group(group_id=f'online-{g}', version=0)]
    refs = [dict(trajectory_id=s.trajectory_id, checksum=s.payload_checksum, created_seq=s.created_seq,
                 source_store_id='store') for s in members]
    actor = SimpleNamespace(get_replay_batch=SimpleNamespace(remote=Mock(return_value=members)),
        verify_replay_sources=SimpleNamespace(remote=Mock(return_value={'sources':1,'unchanged':True})),
        configure_replay_sources=SimpleNamespace(remote=Mock(side_effect=AssertionError('No remount'))),
        get_reuse_group=SimpleNamespace(remote=Mock(side_effect=AssertionError('No reseed'))))
    monkeypatch.setitem(sys.modules, 'ray', SimpleNamespace(get=lambda value, **kw:value))
    monkeypatch.setattr(ReplaySession, '__init__', Mock(side_effect=AssertionError('No sampler/RNG creation')))
    controller = Mock()
    cursor = dict(identity=digest('fixed-cursor'), produced_batches=1)
    controller.quiesce_replay_source.return_value = (cursor, [{'replay_refs':refs}], [{}])
    return cfg, actor, refs, members, controller


def test_snapshot_restore_share_exact_validator_without_window_or_rng(online_reader_case):
    cfg, actor, refs, members, controller = online_reader_case
    assert OnlineReplayReader.load_online is ReplaySession.load_online
    source = DataManagerSource(controller, cfg, actor)
    state = source.snapshot(); source.resume()
    assert state['queued_batches'][0]['replay_refs'] == refs
    restored = DataManagerSource(controller, cfg, actor)
    assert restored.restore(state)['source_sha256']
    controller.restore_replay_source.assert_called_once()
    assert actor.verify_replay_sources.remote.call_count == 2
    assert actor.get_replay_batch.remote.call_count == 2
    assert not actor.configure_replay_sources.remote.called and not actor.get_reuse_group.remote.called


@pytest.mark.parametrize('defect', ['checksum','created_seq','source_store','incomplete','short'])
def test_optimized_reader_keeps_all_durable_reference_rejections(online_reader_case, defect):
    cfg, actor, refs, members, controller = online_reader_case
    if defect == 'checksum': refs[0]['checksum'] = 'f'*64
    elif defect == 'created_seq': refs[0]['created_seq'] += 1
    elif defect == 'source_store': refs[0]['source_store_id'] = 'foreign'
    elif defect == 'incomplete': actor.get_replay_batch.remote.return_value = [replace(members[0], group_complete=False)]+members[1:]
    else: actor.get_replay_batch.remote.return_value = members[:-1]
    source = DataManagerSource(controller, cfg, actor)
    with pytest.raises(ReplayIntegrityError): source.snapshot()
    assert source.failed
    assert not controller.resume_replay_source.called


@pytest.mark.parametrize('receipt', [{'sources':0,'unchanged':True}, {'sources':1,'unchanged':False}])
def test_unmounted_or_changed_sources_are_not_silently_skipped(online_reader_case, receipt):
    cfg, actor, _, _, _ = online_reader_case
    actor.verify_replay_sources.remote.return_value = receipt
    with pytest.raises(ReplayIntegrityError): OnlineReplayReader(cfg, actor)


def test_no_reuse_empty_source_has_no_actor_or_rng_side_effects(settings, monkeypatch):
    import random
    monkeypatch.setattr(random, 'Random', Mock(side_effect=AssertionError('No RNG')))
    cfg = settings(batch_groups=2)
    controller = Mock()
    controller.quiesce_replay_source.return_value = (dict(identity=digest('empty')), [], [])
    actor = Mock()
    source = DataManagerSource(controller, cfg, actor)
    assert source.snapshot()['queued_batches'] == []
    assert not actor.mock_calls
