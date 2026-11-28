"""Server-only paired checkpoints retain full artifact checks after simplification."""
from collections import Counter
import os

import pytest

from aura.memory.replay import joint
from aura.memory.replay.schema import ReplayIntegrityError
from aura.memory.replay.state import snapshot_mixer
from test_replay_ledger_joint import harness
from test_replay_mixing import settings


@pytest.mark.parametrize('backend', ['sqlite', 'json'])
def test_publication_and_independent_recovery_read_every_artifact(harness, monkeypatch, backend):
    h = harness(backend=backend)
    calls = Counter()
    original = joint.file_info

    def read(root, path, **kwargs):
        calls[path] += 1
        return original(root, path, **kwargs)

    monkeypatch.setattr(joint, 'file_info', read)
    cut = h.checkpoint('cut')
    paths = [item['path'] for item in cut['body']['artifacts']]
    assert [calls[path] for path in paths] == [3] * len(paths)
    assert joint.read_commit(h.root, 'cut', h.config, h.store.get_trajectory,
                             h.ledger.lookup_commit) == cut
    assert [calls[path] for path in paths] == [4] * len(paths)


@pytest.mark.parametrize('stage', ['before_marker', 'after_marker'])
@pytest.mark.parametrize('fault', ['same_size_restore_mtime', 'missing', 'symlink', 'marker'])
def test_changed_artifacts_or_marker_never_return_success(harness, stage, fault):
    h = harness()
    h.checkpoint('genesis')
    h.update()

    def change(point):
        if point != stage:
            return
        if fault == 'marker':
            path = h.root / 'replay-commits' / 'cut' / 'COMMITTED.json'
            path.write_bytes(b'changed marker')
            return
        path = next((h.root / 'models').glob('first-1-*/model.json'))
        if fault == 'missing':
            path.unlink()
        elif fault == 'symlink':
            other = path.with_suffix('.alias')
            other.write_bytes(path.read_bytes())
            path.unlink()
            path.symlink_to(other)
        else:
            data, info = path.read_bytes(), path.stat()
            path.write_bytes(bytes([data[0] ^ 1]) + data[1:])
            os.utime(path, ns=(info.st_atime_ns, info.st_mtime_ns))

    with pytest.raises((ReplayIntegrityError, OSError, ValueError)):
        joint.publish_commit(h.root, h.config, h.store.get_trajectory, h.ledger, h.writer,
                             h.branch, snapshot_mixer(h.mixer, policy_version=h.version),
                             h.source_state(), h.save, commit_id='cut', failpoint=change)
