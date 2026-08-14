# -------------------------------------------------------------------------
# This file is part of the AgentSDK project.
# Copyright (c) 2026 Huawei Technologies Co.,Ltd.
#
# AgentSDK is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#
#          http://license.coscl.org.cn/MulanPSL2
#
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
# -------------------------------------------------------------------------
#
# Copyright (c) 2026 Clawd Codex Team
# SPDX-License-Identifier: MIT
# Source: https://github.com/agentforce314/clawcodex
# ClawCodex-derived portions remain licensed under the MIT License.
# See clawcodex-ascend/LICENSE.clawcodex.
#
"""Tests for ``extensions.daemon.config``."""

from __future__ import annotations

from pathlib import Path

import pytest

from extensions.daemon.config import (
    DEFAULT_DAEMON_NAME,
    DEFAULT_WORKER_KINDS,
    DaemonConfig,
)
from extensions.daemon.errors import InvalidDaemonConfigError


def test_default_config_validates():
    cfg = DaemonConfig()
    cfg.validate()  # no raise
    assert cfg.name == DEFAULT_DAEMON_NAME
    assert cfg.worker_kinds == DEFAULT_WORKER_KINDS
    assert cfg.spawn_mode == "same-dir"
    assert cfg.capacity == 4


def test_validate_rejects_empty_name():
    cfg = DaemonConfig(name="")
    with pytest.raises(InvalidDaemonConfigError):
        cfg.validate()


def test_validate_rejects_path_separator_in_name():
    with pytest.raises(InvalidDaemonConfigError):
        DaemonConfig(name="a/b").validate()
    with pytest.raises(InvalidDaemonConfigError):
        DaemonConfig(name="a\\b").validate()
    with pytest.raises(InvalidDaemonConfigError):
        DaemonConfig(name="../escape").validate()


def test_validate_rejects_no_workers():
    cfg = DaemonConfig(worker_kinds=())
    with pytest.raises(InvalidDaemonConfigError):
        cfg.validate()


def test_validate_rejects_empty_worker_kind():
    with pytest.raises(InvalidDaemonConfigError):
        DaemonConfig(worker_kinds=("a", "")).validate()


def test_validate_rejects_bad_spawn_mode():
    with pytest.raises(InvalidDaemonConfigError):
        DaemonConfig(spawn_mode="hostile").validate()


def test_validate_rejects_capacity_below_one():
    with pytest.raises(InvalidDaemonConfigError):
        DaemonConfig(capacity=0).validate()


def test_validate_rejects_short_timeout():
    with pytest.raises(InvalidDaemonConfigError):
        DaemonConfig(timeout_ms=500).validate()


def test_validate_rejects_backoff_inversion():
    with pytest.raises(InvalidDaemonConfigError):
        DaemonConfig(backoff_initial_ms=1000, backoff_cap_ms=500).validate()


def test_with_workers_returns_new_instance():
    cfg = DaemonConfig()
    new = cfg.with_workers(("a", "b"))
    assert cfg.worker_kinds == DEFAULT_WORKER_KINDS
    assert new.worker_kinds == ("a", "b")


def test_with_dir_resolves_to_absolute():
    cfg = DaemonConfig().with_dir(Path("."))
    assert cfg.dir.is_absolute()


def test_with_name_returns_new_instance():
    cfg = DaemonConfig()
    new = cfg.with_name("alpha")
    assert cfg.name == DEFAULT_DAEMON_NAME
    assert new.name == "alpha"


def test_frozen_dataclass_immutable():
    cfg = DaemonConfig()
    with pytest.raises(Exception):  # FrozenInstanceError
        cfg.name = "hacked"  # type: ignore[misc]
