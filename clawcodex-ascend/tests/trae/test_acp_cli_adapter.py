#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# -------------------------------------------------------------------------
#  This file is part of the AgentSDK project.
# Copyright (c) 2026 Huawei Technologies Co.,Ltd.
#
# AgentSDK is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#
#           http://license.coscl.org.cn/MulanPSL2
#
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
# -------------------------------------------------------------------------

"""P66-F Tests for acp cli adapter."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from extensions.capabilities.acp_protocol import (
    ACPMessage,
    ACPMessageType,
)
from extensions.trae.acp_cli_adapter import (
    TraeCliACPAdapter,
    TraeCliConfig,
)

# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------


class _FakeProc:
    """Tests for _FakeProc."""

    def __init__(self, cmd: list[str], env: dict | None = None, **kwargs: Any) -> None:
        self.cmd = cmd
        self.env = env
        self.pid = 12345
        self._alive = True
        self.stdout = None
        self.stderr = None
        _FakeProc.last_cmd = cmd
        _FakeProc.last_env = env
        _FakeProc.instances.append(self)

    def poll(self) -> int | None:
        return None if self._alive else 0

    def terminate(self) -> None:
        self._alive = False

    def kill(self) -> None:
        self._alive = False

    def wait(self, timeout: float | None = None) -> int:
        self._alive = False
        return 0


@pytest.fixture(autouse=True)
def _reset_fake_proc() -> Any:
    _FakeProc.instances = []
    _FakeProc.last_cmd = None
    _FakeProc.last_env = None
    yield
    _FakeProc.instances = []
    _FakeProc.last_cmd = None
    _FakeProc.last_env = None


# ---------------------------------------------------------------------------
# create_session
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_create_session_generates_sid_and_trajectory_path(tmp_path: Path) -> None:
    """Verify create session generates sid and trajectory path."""
    adapter = TraeCliACPAdapter(TraeCliConfig(), str(tmp_path), process_factory=_FakeProc)
    session = await adapter.create_session(str(tmp_path))
    assert session.id
    assert session.workspace_path == str(tmp_path)
    traj_dir = tmp_path / ".trae" / "trajectories"
    assert traj_dir.is_dir()
    traj = traj_dir / f"{session.id}.jsonl"
    assert adapter._trajectories[session.id] == traj
    assert _FakeProc.instances == []


@pytest.mark.asyncio
async def test_create_session_unique_ids(tmp_path: Path) -> None:
    adapter = TraeCliACPAdapter(TraeCliConfig(), str(tmp_path), process_factory=_FakeProc)
    s1 = await adapter.create_session(str(tmp_path))
    s2 = await adapter.create_session(str(tmp_path))
    assert s1.id != s2.id


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_build_run_cmd_matches_trae_agent_interface(tmp_path: Path) -> None:
    """Verify build run cmd matches trae agent interface."""
    cfg = TraeCliConfig(
        trae_cli_path="trae-cli",
        provider="anthropic",
        model="claude-sonnet-4-6",
        extra_flags=["--no-color"],
    )
    adapter = TraeCliACPAdapter(cfg, str(tmp_path), process_factory=_FakeProc)
    session = await adapter.create_session(str(tmp_path))
    cmd = adapter._build_run_cmd(session.id, "fix the bug")

    assert cmd[0] == "trae-cli"
    assert cmd[1] == "run"
    assert cmd[2] == "fix the bug"
    assert "--working-dir" in cmd
    assert str(tmp_path) in cmd
    assert "--trajectory-file" in cmd
    assert str(adapter._trajectories[session.id]) in cmd
    assert "--provider" in cmd and "anthropic" in cmd
    assert "--model" in cmd and "claude-sonnet-4-6" in cmd
    assert "--no-color" in cmd


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_process_message_spawns_run_and_streams_trajectory(tmp_path: Path) -> None:
    """Verify process message spawns run and streams trajectory."""
    adapter = TraeCliACPAdapter(TraeCliConfig(), str(tmp_path), process_factory=_FakeProc)
    session = await adapter.create_session(str(tmp_path))
    traj = adapter._trajectories[session.id]

    traj.write_text(
        json.dumps({"id": "e1", "step": "think", "content": "analyzing", "model": "claude"})
        + "\n"
        + json.dumps(
            {
                "id": "e2",
                "step": "act",
                "tool_name": "edit_file",
                "tool_input": {"path": "a.py"},
                "content": "edit",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    msg = ACPMessage(
        type=ACPMessageType.MESSAGE_SEND,
        session_id=session.id,
        content="do the task",
    )
    msgs = []
    async for m in adapter.process_message(msg):
        msgs.append(m)
        if len(msgs) >= 2:
            _FakeProc.instances[0]._alive = False

    assert len(msgs) == 2
    assert msgs[0].type == ACPMessageType.MESSAGE_STREAM
    assert msgs[0].content == "analyzing"
    assert msgs[0].metadata["model"] == "claude"
    assert msgs[1].type == ACPMessageType.TOOL_CALL
    assert msgs[1].tool_calls[0]["name"] == "edit_file"
    assert msgs[1].tool_calls[0]["arguments"] == {"path": "a.py"}


@pytest.mark.asyncio
async def test_process_message_no_session_id_yields_nothing(tmp_path: Path) -> None:
    """Verify process message no session id yields nothing."""
    adapter = TraeCliACPAdapter(TraeCliConfig(), str(tmp_path), process_factory=_FakeProc)
    msg = ACPMessage(type=ACPMessageType.MESSAGE_SEND, session_id="", content="x")
    msgs = []
    async for m in adapter.process_message(msg):
        msgs.append(m)
    assert msgs == []
    assert _FakeProc.instances == []


@pytest.mark.asyncio
async def test_process_message_skips_unparseable_trajectory_lines(tmp_path: Path) -> None:
    """Verify process message skips unparseable trajectory lines."""
    adapter = TraeCliACPAdapter(TraeCliConfig(), str(tmp_path), process_factory=_FakeProc)
    session = await adapter.create_session(str(tmp_path))
    traj = adapter._trajectories[session.id]
    traj.write_text(
        "not-json-line\n" + json.dumps({"id": "e1", "content": "good"}) + "\n" + "{broken\n",
        encoding="utf-8",
    )
    msg = ACPMessage(type=ACPMessageType.MESSAGE_SEND, session_id=session.id, content="t")
    msgs = []
    async for m in adapter.process_message(msg):
        msgs.append(m)
        if len(msgs) >= 1:
            _FakeProc.instances[0]._alive = False
    assert len(msgs) == 1
    assert msgs[0].content == "good"


# ---------------------------------------------------------------------------
# resume_session
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_resume_session_spawns_interactive_with_trajectory(tmp_path: Path) -> None:
    adapter = TraeCliACPAdapter(TraeCliConfig(), str(tmp_path), process_factory=_FakeProc)
    session = await adapter.create_session(str(tmp_path))
    traj = adapter._trajectories[session.id]
    traj.write_text(json.dumps({"id": "e1", "content": "prev"}) + "\n", encoding="utf-8")

    restored = await adapter.resume_session(session.id)
    assert restored is not None
    assert restored.id == session.id
    assert _FakeProc.last_cmd is not None
    assert "interactive" in _FakeProc.last_cmd
    assert "--resume-trajectory" in _FakeProc.last_cmd
    assert str(traj) in _FakeProc.last_cmd


@pytest.mark.asyncio
async def test_resume_session_unknown_returns_none(tmp_path: Path) -> None:
    adapter = TraeCliACPAdapter(TraeCliConfig(), str(tmp_path), process_factory=_FakeProc)
    assert await adapter.resume_session("never-created") is None


@pytest.mark.asyncio
async def test_resume_session_missing_trajectory_file_returns_none(tmp_path: Path) -> None:
    adapter = TraeCliACPAdapter(TraeCliConfig(), str(tmp_path), process_factory=_FakeProc)
    session = await adapter.create_session(str(tmp_path))
    assert await adapter.resume_session(session.id) is None
    assert _FakeProc.instances == []


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_end_session_terminates_process_and_removes_trajectory(tmp_path: Path) -> None:
    adapter = TraeCliACPAdapter(TraeCliConfig(), str(tmp_path), process_factory=_FakeProc)
    session = await adapter.create_session(str(tmp_path))
    traj = adapter._trajectories[session.id]
    traj.write_text("x\n", encoding="utf-8")
    msg = ACPMessage(type=ACPMessageType.MESSAGE_SEND, session_id=session.id, content="t")
    gen = adapter.process_message(msg)
    traj.write_text(json.dumps({"id": "e1", "content": "c"}) + "\n", encoding="utf-8")
    async for _ in gen:
        break
    assert session.id in adapter._procs

    await adapter.end_session(session.id)
    assert session.id not in adapter._procs
    assert not traj.exists()
    assert session.id not in adapter._sessions


@pytest.mark.asyncio
async def test_end_session_unknown_id_is_noop(tmp_path: Path) -> None:
    """Verify end session unknown id is noop."""
    adapter = TraeCliACPAdapter(TraeCliConfig(), str(tmp_path), process_factory=_FakeProc)
    await adapter.end_session("never-existed")


@pytest.mark.asyncio
async def test_end_session_kill_after_terminate_timeout(tmp_path: Path, monkeypatch) -> None:
    """Verify end session kill after terminate timeout."""

    class _StubbornProc(_FakeProc):
        def wait(self, timeout: float | None = None) -> int:
            import subprocess

            raise subprocess.TimeoutExpired(cmd=self.cmd, timeout=timeout or 0)

    adapter = TraeCliACPAdapter(TraeCliConfig(), str(tmp_path), process_factory=_StubbornProc)
    session = await adapter.create_session(str(tmp_path))
    adapter._procs[session.id] = _StubbornProc(["trae-cli"])
    await adapter.end_session(session.id)
    assert session.id not in adapter._procs


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------


def test_trajectory_to_acp_tool_event() -> None:
    adapter = TraeCliACPAdapter(TraeCliConfig(), "/tmp")
    evt = {
        "id": "e1",
        "step": "act",
        "tool_name": "bash",
        "tool_input": {"cmd": "ls"},
        "content": "running",
    }
    msg = adapter._trajectory_to_acp("sid", evt)
    assert msg.type == ACPMessageType.TOOL_CALL
    assert msg.tool_calls == [{"name": "bash", "arguments": {"cmd": "ls"}}]
    assert msg.content == "running"
    assert msg.metadata["step"] == "act"


def test_trajectory_to_acp_text_event() -> None:
    adapter = TraeCliACPAdapter(TraeCliConfig(), "/tmp")
    evt = {"id": "e1", "step": "think", "content": "analyzing", "model": "claude"}
    msg = adapter._trajectory_to_acp("sid", evt)
    assert msg.type == ACPMessageType.MESSAGE_STREAM
    assert msg.content == "analyzing"
    assert msg.metadata["model"] == "claude"


def test_trajectory_to_acp_missing_fields_degrade() -> None:
    """Verify trajectory to acp missing fields degrade."""
    adapter = TraeCliACPAdapter(TraeCliConfig(), "/tmp")
    msg = adapter._trajectory_to_acp("sid", {})
    assert msg.type == ACPMessageType.MESSAGE_STREAM
    assert msg.content == ""
    assert msg.metadata["step"] == "unknown"
    assert msg.id


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------


def test_env_includes_provider_and_model() -> None:
    cfg = TraeCliConfig(provider="openai", model="gpt-4")
    adapter = TraeCliACPAdapter(cfg, "/tmp")
    env = adapter._env()
    assert env["TRAE_PROVIDER"] == "openai"
    assert env["TRAE_MODEL"] == "gpt-4"


def test_env_includes_mcp_servers_json() -> None:
    cfg = TraeCliConfig(
        mcp_servers=[{"name": "clawcodex", "command": "python", "args": ["-m", "extensions.trae.mcp_bridge"]}]
    )
    adapter = TraeCliACPAdapter(cfg, "/tmp")
    env = adapter._env()
    servers = json.loads(env["TRAE_MCP_SERVERS"])
    assert servers[0]["name"] == "clawcodex"


def test_env_omits_mcp_servers_when_empty() -> None:
    adapter = TraeCliACPAdapter(TraeCliConfig(), "/tmp")
    env = adapter._env()
    assert "TRAE_MCP_SERVERS" not in env


def test_env_inherits_os_environ() -> None:
    adapter = TraeCliACPAdapter(TraeCliConfig(), "/tmp")
    env = adapter._env()
    assert "PATH" in env or any(k for k in env if k.upper() == "PATH")


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------


def test_trae_cli_config_defaults() -> None:
    cfg = TraeCliConfig()
    assert cfg.trae_cli_path == "trae-cli"
    assert cfg.provider == "anthropic"
    assert cfg.model == "claude-sonnet-4-6"
    assert cfg.api_key_env == "ANTHROPIC_API_KEY"
    assert cfg.mcp_servers == []
    assert cfg.extra_flags == []
