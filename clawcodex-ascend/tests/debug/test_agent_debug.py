#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# -------------------------------------------------------------------------
# This file is part of the AgentSDK project.
#
# Originally from Clawd Codex:
# https://github.com/agentforce314/clawcodex
# Copyright (c) 2026 Clawd Codex Team
# Licensed under the MIT License. See clawcodex-ascend/LICENSE.clawcodex.
#
# Portions copyright (c) 2026 Huawei Technologies Co.,Ltd.
# Licensed under Mulan PSL v2. You may obtain a copy of Mulan PSL v2 at:
#
#          http://license.coscl.org.cn/MulanPSL2
#
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
# -------------------------------------------------------------------------

from __future__ import annotations

import io
import json
from pathlib import Path

from clawcodex_ext.debug.agent_debug import (
    agent_debug_enabled,
    apply_agent_debug_environment,
    emit_agent_debug_marker,
    resolve_repl_history_file,
)


def test_agent_debug_enabled_accepts_truthy_values() -> None:
    assert agent_debug_enabled({"CLAWCODEX_AGENT_DEBUG": "1"}) is True
    assert agent_debug_enabled({"CLAWCODEX_AGENT_DEBUG": "true"}) is True
    assert agent_debug_enabled({"CLAWCODEX_AGENT_DEBUG": "yes"}) is True
    assert agent_debug_enabled({"CLAWCODEX_AGENT_DEBUG": "on"}) is True


def test_agent_debug_enabled_rejects_false_values() -> None:
    assert agent_debug_enabled({}) is False
    assert agent_debug_enabled({"CLAWCODEX_AGENT_DEBUG": ""}) is False
    assert agent_debug_enabled({"CLAWCODEX_AGENT_DEBUG": "0"}) is False
    assert agent_debug_enabled({"CLAWCODEX_AGENT_DEBUG": "false"}) is False


def test_resolve_repl_history_file_uses_home_when_debug_is_disabled(tmp_path: Path) -> None:
    assert resolve_repl_history_file({}, home=tmp_path) == tmp_path / ".clawcodex" / "history"


def test_resolve_repl_history_file_uses_explicit_file(tmp_path: Path) -> None:
    explicit = tmp_path / "custom-history"

    assert (
        resolve_repl_history_file(
            {
                "CLAWCODEX_AGENT_DEBUG": "1",
                "CLAWCODEX_HISTORY_FILE": str(explicit),
            },
            home=tmp_path,
        )
        == explicit
    )


def test_resolve_repl_history_file_uses_debug_dir(tmp_path: Path) -> None:
    assert (
        resolve_repl_history_file(
            {
                "CLAWCODEX_AGENT_DEBUG": "1",
                "CLAWCODEX_AGENT_DEBUG_DIR": str(tmp_path / "debug-home"),
            },
            home=tmp_path,
        )
        == tmp_path / "debug-home" / "history"
    )


def test_apply_agent_debug_environment_sets_state_paths(tmp_path: Path) -> None:
    env: dict[str, str] = {}

    apply_agent_debug_environment(env, debug_dir=tmp_path / "state")

    assert env["CLAWCODEX_AGENT_DEBUG"] == "1"
    assert env["CLAWCODEX_AGENT_DEBUG_DIR"] == str(tmp_path / "state")
    assert env["CLAWCODEX_HOME"] == str(tmp_path / "state")
    assert env["CLAWCODEX_HISTORY_FILE"] == str(tmp_path / "state" / "history")
    assert env["CLAWCODEX_SESSIONS_DIR"] == str(tmp_path / "state" / "sessions")
    assert env["CLAW_TELEMETRY_STORAGE_DIR"] == str(tmp_path / "state" / "telemetry")


def test_apply_agent_debug_environment_overrides_inherited_state_paths(tmp_path: Path) -> None:
    env = {
        "CLAWCODEX_HISTORY_FILE": "/real/home/history",
        "CLAWCODEX_HOME": "/real/home/clawcodex",
        "CLAWCODEX_SESSIONS_DIR": "/real/home/sessions",
        "CLAW_TELEMETRY_STORAGE_DIR": "/real/home/telemetry",
    }

    apply_agent_debug_environment(env, debug_dir=tmp_path / "state")

    assert env["CLAWCODEX_HOME"] == str(tmp_path / "state")
    assert env["CLAWCODEX_HISTORY_FILE"] == str(tmp_path / "state" / "history")
    assert env["CLAWCODEX_SESSIONS_DIR"] == str(tmp_path / "state" / "sessions")
    assert env["CLAW_TELEMETRY_STORAGE_DIR"] == str(tmp_path / "state" / "telemetry")


def test_session_storage_honors_agent_debug_sessions_dir(tmp_path: Path, monkeypatch) -> None:
    from clawcodex_ext.services.session_storage import SessionStorage

    sessions_dir = tmp_path / "state" / "sessions"
    monkeypatch.setenv("CLAWCODEX_SESSIONS_DIR", str(sessions_dir))

    storage = SessionStorage(session_id="debug-session")

    assert storage.sessions_dir == sessions_dir


def test_legacy_session_save_honors_agent_debug_sessions_dir(tmp_path: Path, monkeypatch) -> None:
    from clawcodex_ext.agent.session import Session

    sessions_dir = tmp_path / "state" / "sessions"
    home_dir = tmp_path / "home"
    monkeypatch.setenv("CLAWCODEX_SESSIONS_DIR", str(sessions_dir))
    monkeypatch.setenv("HOME", str(home_dir))

    session = Session(session_id="debug-save-session", provider="stub", model="stub")

    session.save()

    assert (sessions_dir / "debug-save-session" / "transcript.jsonl").exists()
    assert not (home_dir / ".clawcodex" / "sessions" / "debug-save-session").exists()


def test_emit_agent_debug_marker_is_silent_when_disabled() -> None:
    stream = io.StringIO()

    emit_agent_debug_marker("repl.ready", {"session_id": "s1"}, stream=stream, environ={})

    assert stream.getvalue() == ""


def test_emit_agent_debug_marker_writes_json_line_when_enabled() -> None:
    stream = io.StringIO()

    emit_agent_debug_marker(
        "repl.ready",
        {"session_id": "s1"},
        stream=stream,
        environ={"CLAWCODEX_AGENT_DEBUG": "1"},
    )

    line = stream.getvalue().strip()
    prefix, marker, payload = line.split("::", 2)
    assert prefix == "CLAWCODEX_AGENT_DEBUG"
    assert marker == "repl.ready"
    assert json.loads(payload) == {"session_id": "s1"}
