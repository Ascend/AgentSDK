# SPDX-License-Identifier: MulanPSL-2.0
# Copyright (c) 2026 Huawei Technologies Co.,Ltd.
import os
from types import SimpleNamespace

import pytest

from aura.trainer.train_adapter.verl.full_async.checkpoint_utils import checkpoint_version, resolve_checkpoint


@pytest.mark.parametrize('mode', ['disable', 'auto'])
def test_no_checkpoint_is_cold_start(tmp_path, mode):
    config = SimpleNamespace(resume_mode=mode, default_local_dir=str(tmp_path), default_hdfs_dir=None)
    assert resolve_checkpoint(config) is None


@pytest.mark.parametrize('version', ['0', '3', '150'])
def test_resume_path_and_auto_resolve_same_checkpoint(tmp_path, version):
    path = tmp_path / ('global_step_' + version)
    path.mkdir()
    (tmp_path / 'latest_checkpointed_iteration.txt').write_text(version + '\n')
    config = SimpleNamespace(resume_mode='auto', default_local_dir=str(tmp_path), default_hdfs_dir=None)
    assert resolve_checkpoint(config) == str(path)
    config.resume_mode = 'resume_path'
    config.resume_from_path = str(path) + os.sep
    assert resolve_checkpoint(config) == str(path)
    assert checkpoint_version(config.resume_from_path) == int(version)


@pytest.mark.parametrize('pointer', ['-1', 'bad', '', '../global_step_3'])
def test_invalid_committed_pointer_is_rejected(tmp_path, pointer):
    (tmp_path / 'latest_checkpointed_iteration.txt').write_text(pointer)
    config = SimpleNamespace(resume_mode='auto', default_local_dir=str(tmp_path), default_hdfs_dir=None)
    with pytest.raises(ValueError):
        resolve_checkpoint(config)


def test_missing_committed_checkpoint_is_not_silently_cold_started(tmp_path):
    (tmp_path / 'latest_checkpointed_iteration.txt').write_text('3')
    config = SimpleNamespace(resume_mode='auto', default_local_dir=str(tmp_path), default_hdfs_dir=None)
    with pytest.raises(FileNotFoundError):
        resolve_checkpoint(config)


def test_unknown_resume_mode_is_rejected():
    with pytest.raises(ValueError):
        resolve_checkpoint(SimpleNamespace(resume_mode='unknown'))
