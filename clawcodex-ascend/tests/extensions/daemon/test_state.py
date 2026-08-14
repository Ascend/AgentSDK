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
"""Tests for ``extensions.daemon.state``."""

from __future__ import annotations

import json
import os


from extensions.daemon.state import (
    DaemonState,
    DaemonStatus,
    get_state_path,
    is_process_alive,
    make_state,
    query_daemon_status,
    read_daemon_state,
    remove_daemon_state,
    write_daemon_state,
)


def test_get_state_path_default_uses_home(state_dir, monkeypatch):
    monkeypatch.setenv("HOME", str(state_dir.parent.parent))
    from extensions.daemon import state as state_mod

    monkeypatch.setattr(state_mod.Path, "home", lambda: state_dir.parent.parent)
    path = state_mod.get_state_path("alpha")
    assert path == state_dir.parent.parent / ".clawcodex" / "daemon" / "alpha.json"


def test_get_state_path_with_state_dir(state_dir):
    path = get_state_path("alpha", state_dir=state_dir)
    assert path == state_dir / "alpha.json"


def test_make_state_fills_started_at():
    state = make_state(pid=42, worker_kinds=["remoteControl"], name="r")
    assert state.pid == 42
    assert state.worker_kinds == ["remoteControl"]
    assert state.name == "r"
    assert state.last_status == DaemonStatus.RUNNING
    assert state.started_at.endswith("Z")
    assert "T" in state.started_at


def test_write_and_read_roundtrip(state_dir):
    state = make_state(pid=os.getpid(), worker_kinds=["a", "b"], name="rt")
    target = write_daemon_state(state, state_dir=state_dir)
    assert target.exists()
    loaded = read_daemon_state("rt", state_dir=state_dir)
    assert loaded is not None
    assert loaded.pid == state.pid
    assert loaded.worker_kinds == ["a", "b"]
    assert loaded.name == "rt"
    assert loaded.last_status == DaemonStatus.RUNNING


def test_write_is_atomic_no_tmp_file_left(state_dir):
    state = make_state(pid=1234, worker_kinds=["a"], name="atomic")
    write_daemon_state(state, state_dir=state_dir)
    target = state_dir / "atomic.json"
    tmp = target.with_suffix(target.suffix + ".tmp")
    assert target.exists()
    assert not tmp.exists()


def test_write_creates_state_dir(tmp_path):
    state = make_state(pid=1, worker_kinds=["a"], name="x")
    nested = tmp_path / "deep" / "down" / "daemon"
    write_daemon_state(state, state_dir=nested)
    assert nested.is_dir()
    assert (nested / "x.json").exists()


def test_read_missing_returns_none(state_dir):
    assert read_daemon_state("absent", state_dir=state_dir) is None


def test_read_corrupt_json_returns_none(state_dir, caplog):
    state_dir.joinpath("bad.json").write_text("{not-json", encoding="utf-8")
    with caplog.at_level("WARNING"):
        result = read_daemon_state("bad", state_dir=state_dir)
    assert result is None


def test_read_missing_required_field_returns_none(state_dir, caplog):
    state_dir.joinpath("partial.json").write_text(
        json.dumps({"pid": 1, "started_at": "2026-01-01T00:00:00Z", "worker_kinds": []}),
        encoding="utf-8",
    )
    with caplog.at_level("WARNING"):
        result = read_daemon_state("partial", state_dir=state_dir)
    assert result is None


def test_remove_state_idempotent(state_dir):
    remove_daemon_state("never-existed", state_dir=state_dir)  # no raise
    state = make_state(pid=1, worker_kinds=["a"], name="r")
    write_daemon_state(state, state_dir=state_dir)
    remove_daemon_state("r", state_dir=state_dir)
    assert read_daemon_state("r", state_dir=state_dir) is None


def test_is_process_alive_current_pid():
    assert is_process_alive(os.getpid()) is True


def test_is_process_alive_bogus_pid():
    assert is_process_alive(2_000_000_000) is False


def test_is_process_alive_zero_and_negative():
    assert is_process_alive(0) is False
    assert is_process_alive(-1) is False


def test_query_status_stopped_when_no_state(state_dir):
    status, state = query_daemon_status("ghost", state_dir=state_dir)
    assert status == DaemonStatus.STOPPED
    assert state is None


def test_query_status_running_when_pid_alive(state_dir):
    state = make_state(pid=os.getpid(), worker_kinds=["a"], name="alive")
    write_daemon_state(state, state_dir=state_dir)
    status, loaded = query_daemon_status("alive", state_dir=state_dir)
    assert status == DaemonStatus.RUNNING
    assert loaded is not None
    assert loaded.pid == os.getpid()


def test_query_status_stale_auto_cleans(state_dir):
    state = make_state(pid=2_000_000_000, worker_kinds=["a"], name="stale")
    write_daemon_state(state, state_dir=state_dir)
    status, loaded = query_daemon_status("stale", state_dir=state_dir)
    assert status == DaemonStatus.STALE
    assert loaded is None
    assert not (state_dir / "stale.json").exists()


def test_state_dict_serialization_roundtrip():
    state = make_state(pid=1, worker_kinds=["a"], name="x")
    d = state.to_dict()
    assert d["last_status"] == "running"
    restored = DaemonState.from_dict(d)
    assert restored.last_status == DaemonStatus.RUNNING
    assert restored.pid == 1
    assert restored.worker_kinds == ["a"]


def test_state_dict_unknown_status_defaults_to_running():
    state = DaemonState.from_dict(
        {
            "pid": 1,
            "cwd": ".",
            "started_at": "2026-01-01T00:00:00Z",
            "worker_kinds": ["a"],
            "last_status": "bogus-state",
        }
    )
    assert state.last_status == DaemonStatus.RUNNING
