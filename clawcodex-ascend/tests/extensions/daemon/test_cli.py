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
"""Tests for ``extensions.daemon.cli`` — argparse + verbs."""

from __future__ import annotations

import argparse
import os
import sys

import pytest

from extensions.daemon.cli import build_parser, cmd_attach, cmd_kill, cmd_logs, cmd_status
from extensions.daemon.state import make_state, write_daemon_state


def _parsed(*argv: str) -> argparse.Namespace:
    return build_parser().parse_args(list(argv))


def test_parser_requires_verb():
    with pytest.raises(SystemExit):
        build_parser().parse_args([])


def test_parser_start_minimal():
    ns = _parsed("start", "--name", "alpha", "--foreground")
    assert ns.verb == "start"
    assert ns.name == "alpha"
    assert ns.workers == "remoteControl"
    assert ns.spawn_mode == "same-dir"
    assert ns.capacity == 4
    assert ns.foreground is True


def test_parser_start_full():
    ns = _parsed(
        "start",
        "--name",
        "alpha",
        "--workers",
        "remoteControl,cron",
        "--spawn-mode",
        "worktree",
        "--capacity",
        "8",
        "--permission-mode",
        "bypassPermissions",
        "--sandbox",
        "--foreground",
    )
    assert ns.workers == "remoteControl,cron"
    assert ns.spawn_mode == "worktree"
    assert ns.capacity == 8
    assert ns.permission_mode == "bypassPermissions"
    assert ns.sandbox is True


def test_parser_stop_accepts_force():
    ns = _parsed("stop", "--force")
    assert ns.verb == "stop"
    assert ns.force is True


def test_parser_status_accepts_json():
    ns = _parsed("status", "--json")
    assert ns.verb == "status"
    assert ns.json is True


def test_parser_ps_alias():
    ns = _parsed("ps")
    assert ns.verb == "ps"


def test_parser_ps_accepts_status_args():
    ns = _parsed("ps", "--name", "alpha", "--json")
    assert ns.verb == "ps"
    assert ns.name == "alpha"
    assert ns.json is True


def test_parser_bg_alias():
    ns = _parsed("bg", "--foreground")
    assert ns.verb == "bg"


def test_parser_kill():
    ns = _parsed("kill")
    assert ns.verb == "kill"


# ---------------------------------------------------------------------------
# Status verb — output formatting + JSON
# ---------------------------------------------------------------------------


def test_cmd_status_stopped(state_dir, capsys):
    ns = _parsed("status", "--name", "ghost")
    ns.state_dir = state_dir
    rc = cmd_status(ns)
    captured = capsys.readouterr()
    assert rc == 1
    assert "stopped" in captured.out
    assert "ghost" in captured.out


def test_cmd_status_running(state_dir, capsys):
    state = make_state(pid=os.getpid(), worker_kinds=["a", "b"], name="alive")
    write_daemon_state(state, state_dir=state_dir)
    ns = _parsed("status", "--name", "alive")
    ns.state_dir = state_dir
    rc = cmd_status(ns)
    captured = capsys.readouterr()
    assert rc == 0
    assert "running" in captured.out
    assert f"PID:     {os.getpid()}" in captured.out
    assert "a, b" in captured.out


def test_cmd_status_json(state_dir, capsys):
    state = make_state(pid=os.getpid(), worker_kinds=["a"], name="j")
    write_daemon_state(state, state_dir=state_dir)
    ns = _parsed("status", "--name", "j", "--json")
    ns.state_dir = state_dir
    cmd_status(ns)
    captured = capsys.readouterr()
    import json

    payload = json.loads(captured.out)
    assert payload["name"] == "j"
    assert payload["status"] == "running"
    assert payload["state"]["pid"] == os.getpid()


# ---------------------------------------------------------------------------
# Attach / logs / kill verbs
# ---------------------------------------------------------------------------


def test_cmd_attach_prints_notice(capsys):
    ns = _parsed("attach")
    rc = cmd_attach(ns)
    assert rc == 1
    out = capsys.readouterr().err
    assert "attach" in out


def test_cmd_logs_missing_file(state_dir, capsys):
    ns = _parsed("logs", "--name", "nope")
    ns.state_dir = state_dir
    rc = cmd_logs(ns)
    assert rc == 1
    err = capsys.readouterr().err
    assert "no log file" in err


def test_cmd_logs_reads_tail(state_dir, capsys):
    log = state_dir / "alpha.log"
    log.write_text("\n".join(f"line {i}" for i in range(50)) + "\n", encoding="utf-8")
    ns = _parsed("logs", "--name", "alpha", "--tail", "5")
    ns.state_dir = state_dir
    rc = cmd_logs(ns)
    assert rc == 0
    out = capsys.readouterr().out
    lines = out.strip().splitlines()
    assert lines == ["line 45", "line 46", "line 47", "line 48", "line 49"]


def test_cmd_kill_uses_force_flag(state_dir, monkeypatch, capsys):
    """cmd_kill should behave like cmd_stop --force."""
    sent: list[tuple[int, int]] = []

    def fake_kill(pid, sig):
        # Liveness probe (sig=0) — pretend the PID is alive.
        if sig == 0:
            return None
        sent.append((pid, sig))
        raise ProcessLookupError

    import extensions.daemon.cli as cli_mod

    monkeypatch.setattr(cli_mod.os, "kill", fake_kill)
    # Use the *current* process's PID so the liveness probe (sig=0) returns
    # successfully and the kill path actually runs.
    state = make_state(pid=os.getpid(), worker_kinds=["a"], name="killme")
    write_daemon_state(state, state_dir=state_dir)
    ns = _parsed("kill", "--name", "killme")
    ns.state_dir = state_dir
    rc = cmd_kill(ns)
    assert rc == 0
    assert sent and sent[0][1] == 9  # SIGKILL on Linux/Darwin
    out = capsys.readouterr().out
    assert "SIGKILL" in out


# ---------------------------------------------------------------------------
# run_daemon — top-level dispatcher
# ---------------------------------------------------------------------------


def test_run_daemon_unknown_verb(capsys):
    from extensions.daemon.cli import run_daemon

    with pytest.raises(SystemExit):
        run_daemon(["no-such-verb"])
    err = capsys.readouterr().err
    assert "unknown verb" in err or "invalid choice" in err


def test_run_daemon_status_dispatches_to_cmd_status(state_dir, capsys, monkeypatch):
    from extensions.daemon.cli import run_daemon

    rc = run_daemon(["--state-dir", str(state_dir), "status", "--name", "ghost"])
    assert rc == 1
    out = capsys.readouterr().out
    assert "stopped" in out


def test_run_daemon_ps_dispatches_to_cmd_status(state_dir, capsys):
    from extensions.daemon.cli import run_daemon

    rc = run_daemon(["--state-dir", str(state_dir), "ps", "--name", "ghost"])
    assert rc == 1
    out = capsys.readouterr().out
    assert "stopped" in out


def test_run_daemon_defaults_to_sys_argv(state_dir, capsys, monkeypatch):
    from extensions.daemon.cli import run_daemon

    monkeypatch.setattr(
        sys,
        "argv",
        ["clawcodex", "--state-dir", str(state_dir), "status", "--name", "ghost"],
    )
    rc = run_daemon()
    assert rc == 1
    out = capsys.readouterr().out
    assert "stopped" in out
