"""Server-only value equivalence and corruption checks for batched decode."""
import json
import struct

import pytest

from aura.memory.replay.codec import ReplayCodec
from aura.memory.replay.schema import DTYPES, ReplayIntegrityError, ReplayTensor
from aura.memory.replay.sqlite_store import SQLiteReplayStore


@pytest.mark.parametrize('dtype', list(DTYPES))
@pytest.mark.parametrize('size', [0, 1, 63, 64, 65, 511, 1024])
def test_bounded_scalar_blocks_equal_original_unpack(dtype, size):
    source = [i % 5 for i in range(size)]
    value = ReplayTensor.from_values(source, dtype)
    expected = [row[0] for row in struct.iter_unpack('<' + DTYPES[dtype], value.data)]
    assert list(value.values()) == expected


def test_sqlite_decodes_each_payload_once_and_checks_indexed_checksum(tmp_path, make_group, monkeypatch):
    store = SQLiteReplayStore(tmp_path / 'store.sqlite', store_id='store')
    samples = make_group()
    store.append_trajectories(samples, 'write')
    from aura.memory.replay import codec
    original = codec.strict_json
    calls = []

    def parse(data, *args, **kwargs):
        calls.append(1)
        return original(data, *args, **kwargs)

    monkeypatch.setattr(codec, 'strict_json', parse)
    assert len(store.get_trajectories([sample.trajectory_id for sample in samples])) == 2
    assert len(calls) == 2
    assert len(store.get_trajectories([sample.trajectory_id for sample in samples])) == 2
    assert len(calls) == 2
    store.conn.execute("UPDATE replay_samples SET checksum=? WHERE trajectory_id=?",
                       ('0'*64, samples[0].trajectory_id))
    with pytest.raises(ReplayIntegrityError):
        store.get_trajectory(samples[0].trajectory_id)
    store.close()


@pytest.mark.parametrize('column,value', [('reward', 7.), ('weight_version', 99),
                                         ('prompt_id', 'changed')])
def test_warm_decode_still_rejects_changed_indexed_metadata(tmp_path, make_group, column, value):
    store = SQLiteReplayStore(tmp_path / 'store.sqlite', store_id='store')
    samples = make_group()
    sample = samples[0]
    store.append_trajectories(samples, 'first')
    store.get_trajectory(sample.trajectory_id)
    store.conn.execute(f'UPDATE replay_samples SET {column}=? WHERE trajectory_id=?',
                       (value, sample.trajectory_id))
    with pytest.raises(ReplayIntegrityError):
        store.get_trajectory(sample.trajectory_id)
    store.close()


def test_warm_decode_does_not_hide_changed_blob_or_deleted_row(tmp_path, make_group):
    store = SQLiteReplayStore(tmp_path / 'store.sqlite', store_id='store')
    samples = make_group()
    store.append_trajectories(samples, 'first')
    store.get_trajectories([sample.trajectory_id for sample in samples])
    data = store.conn.execute('SELECT payload FROM replay_samples WHERE trajectory_id=?',
                              (samples[0].trajectory_id,)).fetchone()[0]
    store.conn.execute('UPDATE replay_samples SET payload=? WHERE trajectory_id=?',
                       (data.replace(b'"schema_version":1', b'"schema_version":2'), samples[0].trajectory_id))
    with pytest.raises(ReplayIntegrityError):
        store.get_trajectory(samples[0].trajectory_id)
    store.conn.execute('DELETE FROM replay_samples WHERE trajectory_id=?', (samples[1].trajectory_id,))
    assert store.get_trajectory(samples[1].trajectory_id) is None
    store.close()


def test_changed_payload_format_is_revalidated_and_preserves_text_compatibility(tmp_path, make_group):
    store = SQLiteReplayStore(tmp_path / 'store.sqlite', store_id='store')
    samples = make_group()
    store.append_trajectories(samples, 'first')
    first = store.get_trajectory(samples[0].trajectory_id)
    text = json.dumps(ReplayCodec.document(samples[0]), indent=2)
    store.conn.execute('UPDATE replay_samples SET payload=? WHERE trajectory_id=?',
                       (text, samples[0].trajectory_id))
    assert store.get_trajectory(samples[0].trajectory_id) == first
    store.close()


def test_cache_keeps_group_completion_fresh_and_evicts_by_payload_bytes(tmp_path, make_group, monkeypatch):
    from aura.memory.replay import sqlite_store
    store = SQLiteReplayStore(tmp_path / 'store.sqlite', store_id='store')
    samples = make_group(eligible=False)
    monkeypatch.setattr(sqlite_store, 'MAX_BATCH_BYTES', len(ReplayCodec.encode(samples[0])))
    store.append_trajectories(samples[:1], 'partial')
    first = store.get_trajectory(samples[0].trajectory_id)
    assert first.group_complete is False
    store.append_trajectories(samples[1:], 'complete')
    assert store.get_trajectory(samples[0].trajectory_id).group_complete is True
    assert first.group_complete is False
    store.get_trajectory(samples[1].trajectory_id)
    assert store._decoded_bytes <= sqlite_store.MAX_BATCH_BYTES
    assert store.get_trajectory(samples[0].trajectory_id).payload_checksum == samples[0].payload_checksum
    store.close()
    assert store._decoded_bytes == 0 and not store._decoded


def test_checksum_decode_preserves_public_sample_result(make_group):
    value = ReplayCodec.encode(make_group()[0])
    sample, checksum = ReplayCodec.decode_with_checksum(value)
    assert sample == ReplayCodec.decode(value)
    assert checksum == ReplayCodec.document(sample)['checksum']
