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

"""P66-F — TraeCliACPAdapter 单元测试 (mock subprocess, 不依赖真 trae-cli)."""

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
# Mock subprocess.Popen — 模拟 trae-cli 进程
# ---------------------------------------------------------------------------


class _FakeProc:
    """模拟 subprocess.Popen: poll()/terminate()/wait()/kill() 行为。"""

    def __init__(self, cmd: list[str], env: dict | None = None, **kwargs: Any) -> None:
        self.cmd = cmd
        self.env = env
        self.pid = 12345
        self._alive = True
        self.stdout = None
        self.stderr = None
        # 测试可挂钩记录启动
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
    """session/create 生成 sid + .trae/trajectories/<sid>.jsonl 路径，无 CLI 调用。"""
    adapter = TraeCliACPAdapter(TraeCliConfig(), str(tmp_path), process_factory=_FakeProc)
    session = await adapter.create_session(str(tmp_path))
    assert session.id  # UUID 非空
    assert session.workspace_path == str(tmp_path)
    # trajectory 目录已创建
    traj_dir = tmp_path / ".trae" / "trajectories"
    assert traj_dir.is_dir()
    # trajectory 文件路径已记录（文件尚未创建，由 trae-cli 写）
    traj = traj_dir / f"{session.id}.jsonl"
    assert adapter._trajectories[session.id] == traj
    # create_session 不应启动任何子进程
    assert _FakeProc.instances == []


@pytest.mark.asyncio
async def test_create_session_unique_ids(tmp_path: Path) -> None:
    adapter = TraeCliACPAdapter(TraeCliConfig(), str(tmp_path), process_factory=_FakeProc)
    s1 = await adapter.create_session(str(tmp_path))
    s2 = await adapter.create_session(str(tmp_path))
    assert s1.id != s2.id


# ---------------------------------------------------------------------------
# _build_run_cmd — 接口变化集中改这一处
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_build_run_cmd_matches_trae_agent_interface(tmp_path: Path) -> None:
    """§1.10.6 验收: _build_run_cmd 生成的命令行符合 trae-agent v0.x 接口。"""
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
    assert cmd[2] == "fix the bug"  # task 作为位置参数
    assert "--working-dir" in cmd
    assert str(tmp_path) in cmd
    assert "--trajectory-file" in cmd
    assert str(adapter._trajectories[session.id]) in cmd
    assert "--provider" in cmd and "anthropic" in cmd
    assert "--model" in cmd and "claude-sonnet-4-6" in cmd
    assert "--no-color" in cmd  # extra_flags 透传


# ---------------------------------------------------------------------------
# process_message — 启动进程 + tail trajectory
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_process_message_spawns_run_and_streams_trajectory(tmp_path: Path) -> None:
    """message/stream: 启动 trae-cli run + tail jsonl 投影为 ACP 消息流。"""
    adapter = TraeCliACPAdapter(TraeCliConfig(), str(tmp_path), process_factory=_FakeProc)
    session = await adapter.create_session(str(tmp_path))
    traj = adapter._trajectories[session.id]

    # 写入 trajectory 事件 (tool + text 两种)，然后让进程退出
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
            # 让 fake proc 退出以结束 tail 循环
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
    """无 session_id 的消息不应启动进程或抛错。"""
    adapter = TraeCliACPAdapter(TraeCliConfig(), str(tmp_path), process_factory=_FakeProc)
    msg = ACPMessage(type=ACPMessageType.MESSAGE_SEND, session_id="", content="x")
    msgs = []
    async for m in adapter.process_message(msg):
        msgs.append(m)
    assert msgs == []
    assert _FakeProc.instances == []


@pytest.mark.asyncio
async def test_process_message_skips_unparseable_trajectory_lines(tmp_path: Path) -> None:
    """§1.10.7 风险缓解: 坏行降级跳过而非抛 JSONDecodeError。"""
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
    # 只产出 1 条有效消息 (两条坏行被跳过)
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
    # interactive 命令已启动
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
    # trajectory 文件从未写入
    assert await adapter.resume_session(session.id) is None
    assert _FakeProc.instances == []


# ---------------------------------------------------------------------------
# end_session — 进程清理 (§1.10.6 验收: 无残留)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_end_session_terminates_process_and_removes_trajectory(tmp_path: Path) -> None:
    adapter = TraeCliACPAdapter(TraeCliConfig(), str(tmp_path), process_factory=_FakeProc)
    session = await adapter.create_session(str(tmp_path))
    traj = adapter._trajectories[session.id]
    traj.write_text("x\n", encoding="utf-8")
    # 启动一个进程
    msg = ACPMessage(type=ACPMessageType.MESSAGE_SEND, session_id=session.id, content="t")
    gen = adapter.process_message(msg)
    # 推进一次以触发 spawn
    traj.write_text(json.dumps({"id": "e1", "content": "c"}) + "\n", encoding="utf-8")
    async for _ in gen:
        break
    assert session.id in adapter._procs

    await adapter.end_session(session.id)
    # 进程已从注册表移除
    assert session.id not in adapter._procs
    # trajectory 文件已清理
    assert not traj.exists()
    # session 状态已移除
    assert session.id not in adapter._sessions


@pytest.mark.asyncio
async def test_end_session_unknown_id_is_noop(tmp_path: Path) -> None:
    """end_session 对未知 sid 不抛错 (幂等清理)。"""
    adapter = TraeCliACPAdapter(TraeCliConfig(), str(tmp_path), process_factory=_FakeProc)
    await adapter.end_session("never-existed")  # 不应抛错


@pytest.mark.asyncio
async def test_end_session_kill_after_terminate_timeout(tmp_path: Path, monkeypatch) -> None:
    """terminate 后 wait 超时应升级到 kill (避免残留进程)。"""

    class _StubbornProc(_FakeProc):
        def wait(self, timeout: float | None = None) -> int:
            import subprocess

            raise subprocess.TimeoutExpired(cmd=self.cmd, timeout=timeout or 0)

    adapter = TraeCliACPAdapter(TraeCliConfig(), str(tmp_path), process_factory=_StubbornProc)
    session = await adapter.create_session(str(tmp_path))
    # 手动注入一个进程
    adapter._procs[session.id] = _StubbornProc(["trae-cli"])
    await adapter.end_session(session.id)  # 不应抛错
    assert session.id not in adapter._procs


# ---------------------------------------------------------------------------
# _trajectory_to_acp — 容错降级
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
    """§1.10.7: 字段缺失降级为通用消息，不抛 KeyError。"""
    adapter = TraeCliACPAdapter(TraeCliConfig(), "/tmp")
    # 几乎空的事件
    msg = adapter._trajectory_to_acp("sid", {})
    assert msg.type == ACPMessageType.MESSAGE_STREAM
    assert msg.content == ""
    assert msg.metadata["step"] == "unknown"
    assert msg.id  # 自动生成 UUID


# ---------------------------------------------------------------------------
# _env — provider/model/mcp_servers 注入
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
    # PATH 等基础变量应被继承
    assert "PATH" in env or any(k for k in env if k.upper() == "PATH")


# ---------------------------------------------------------------------------
# TraeCliConfig 默认值
# ---------------------------------------------------------------------------


def test_trae_cli_config_defaults() -> None:
    cfg = TraeCliConfig()
    assert cfg.trae_cli_path == "trae-cli"
    assert cfg.provider == "anthropic"
    assert cfg.model == "claude-sonnet-4-6"
    assert cfg.api_key_env == "ANTHROPIC_API_KEY"
    assert cfg.mcp_servers == []
    assert cfg.extra_flags == []
