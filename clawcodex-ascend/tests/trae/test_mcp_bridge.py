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

"""P66-E Tests for mcp bridge."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest
from extensions.trae.mcp_bridge import (
    MCP_UNAVAILABLE,
    TOOL_ORCH_RUN,
    TOOL_SKILL_INVOKE,
    TOOL_SOP_COMPILE,
    TOOL_STABILITY_GATE,
    BridgeConfig,
    TraeMcpBridge,
    _win_to_wsl,
    build_tool_specs,
    mcp_available,
)

# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------


def test_build_tool_specs_returns_4_tools() -> None:
    """Verify build tool specs returns 4 tools."""
    specs = build_tool_specs()
    names = [s.name for s in specs]
    assert names == [
        TOOL_ORCH_RUN,
        TOOL_SOP_COMPILE,
        TOOL_SKILL_INVOKE,
        TOOL_STABILITY_GATE,
    ]
    for spec in specs:
        assert spec.description, f"{spec.name} missing description"
        assert isinstance(spec.input_schema, dict)


def test_orch_run_tool_requires_issue_url() -> None:
    specs = build_tool_specs()
    orch = next(s for s in specs if s.name == TOOL_ORCH_RUN)
    assert orch.input_schema["required"] == ["issue_url"]
    assert "issue_url" in orch.input_schema["properties"]


def test_stability_gate_tool_has_empty_schema() -> None:
    specs = build_tool_specs()
    gate = next(s for s in specs if s.name == TOOL_STABILITY_GATE)
    assert gate.input_schema.get("properties", {}) == {}


# ---------------------------------------------------------------------------
# BridgeConfig.from_env
# ---------------------------------------------------------------------------


def test_bridge_config_from_env_reads_workspace() -> None:
    cfg = BridgeConfig.from_env({"CLAWCODEX_WORKSPACE": "/tmp/ws", "CLAWCODEX_REPORTS_DIR": "/tmp/ws/.reports/"})
    assert cfg.workspace == "/tmp/ws"
    assert cfg.reports_dir == "/tmp/ws/.reports/"


def test_bridge_config_from_env_defaults_empty() -> None:
    cfg = BridgeConfig.from_env({})
    assert cfg.workspace == ""
    assert cfg.reports_dir == ""


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------


def test_win_to_wsl_basic_drive() -> None:
    assert _win_to_wsl("C:\\WorkSpace\\clawcodex") == "/mnt/c/WorkSpace/clawcodex"
    assert _win_to_wsl("D:\\proj") == "/mnt/d/proj"


def test_win_to_wsl_forward_slash() -> None:
    assert _win_to_wsl("C:/WorkSpace/clawcodex") == "/mnt/c/WorkSpace/clawcodex"


def test_win_to_wsl_posix_passthrough() -> None:
    assert _win_to_wsl("/mnt/c/WorkSpace/clawcodex") == "/mnt/c/WorkSpace/clawcodex"
    assert _win_to_wsl("/tmp/ws") == "/tmp/ws"


def test_win_to_wsl_unc_passthrough() -> None:
    assert _win_to_wsl("\\\\wsl$\\Ubuntu-24.04\\home") == "\\\\wsl$\\Ubuntu-24.04\\home"


def test_win_to_wsl_empty_and_quoted() -> None:
    assert _win_to_wsl("") == ""
    assert _win_to_wsl('"C:\\WorkSpace"') == "/mnt/c/WorkSpace"
    assert _win_to_wsl("'C:\\WorkSpace'") == "/mnt/c/WorkSpace"


def test_bridge_config_from_env_converts_windows_paths() -> None:
    """Verify bridge config from env converts windows paths."""
    cfg = BridgeConfig.from_env(
        {
            "CLAWCODEX_WORKSPACE": "C:\\WorkSpace\\clawcodex",
            "CLAWCODEX_REPORTS_DIR": "C:\\WorkSpace\\clawcodex\\.reports\\",
        }
    )
    assert cfg.workspace == "/mnt/c/WorkSpace/clawcodex"
    assert cfg.reports_dir == "/mnt/c/WorkSpace/clawcodex/.reports/"
    assert cfg.stability_gate_cwd == "/mnt/c/WorkSpace/clawcodex"


def test_bridge_config_from_env_disabled_conversion() -> None:
    """Verify bridge config from env disabled conversion."""
    cfg = BridgeConfig.from_env(
        {
            "CLAWCODEX_WORKSPACE": "C:\\foo",
            "CLAWCODEX_AUTO_WIN_TO_WSL": "0",
        }
    )
    assert cfg.workspace == "C:\\foo"


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_call_tool_unknown_raises() -> None:
    bridge = TraeMcpBridge()
    with pytest.raises(ValueError, match="unknown tool"):
        await bridge.call_tool("bogus", {})


@pytest.mark.asyncio
async def test_call_tool_orch_run_fire_and_forget(tmp_path: Path) -> None:
    """Verify call tool orch run fire and forget."""
    enqueued: list[tuple[str, str | None]] = []

    def fake_enqueue(issue_url: str, workflow_path: str | None) -> str:
        enqueued.append((issue_url, workflow_path))
        return "run-123"

    bridge = TraeMcpBridge(
        config=BridgeConfig(reports_dir=str(tmp_path / ".reports")),
        orchestrator_enqueue=fake_enqueue,
    )
    result = await bridge.call_tool(TOOL_ORCH_RUN, {"issue_url": "https://gitcode.com/x/y/issues/1"})
    assert "run-123" in result
    assert "queued" in result
    assert enqueued == [("https://gitcode.com/x/y/issues/1", None)]
    assert "run-123" in bridge._runs


@pytest.mark.asyncio
async def test_call_tool_orch_run_missing_issue_url() -> None:
    bridge = TraeMcpBridge()
    result = await bridge.call_tool(TOOL_ORCH_RUN, {})
    assert "error" in result
    assert "issue_url" in result


@pytest.mark.asyncio
async def test_call_tool_orch_run_enqueue_exception_surfaces() -> None:
    """Verify call tool orch run enqueue exception surfaces."""

    def boom(issue_url: str, workflow_path: str | None) -> str:
        raise RuntimeError("daemon down")

    bridge = TraeMcpBridge(orchestrator_enqueue=boom)
    result = await bridge.call_tool(TOOL_ORCH_RUN, {"issue_url": "x"})
    assert "error" in result
    assert "daemon down" in result


@pytest.mark.asyncio
async def test_call_tool_sop_compile_success() -> None:
    def fake_compile(**kwargs):
        return {
            "status": "converted",
            "agent_type": "video-ops-agent",
            "skills": [{"name": "s1"}, {"name": "s2"}],
            "persist_status": "saved",
        }

    bridge = TraeMcpBridge(sop_compiler=fake_compile)
    result = await bridge.call_tool(TOOL_SOP_COMPILE, {"sdk_spec": "{}"})
    assert "video-ops-agent" in result
    assert "skills=2" in result
    assert "persist=saved" in result


@pytest.mark.asyncio
async def test_call_tool_sop_compile_missing_sdk_spec() -> None:
    bridge = TraeMcpBridge()
    result = await bridge.call_tool(TOOL_SOP_COMPILE, {})
    assert "error" in result
    assert "sdk_spec" in result


@pytest.mark.asyncio
async def test_call_tool_sop_compile_status_error_propagates() -> None:
    def fake_compile(**kwargs):
        return {"status": "error", "error": "No SDK methods parsed"}

    bridge = TraeMcpBridge(sop_compiler=fake_compile)
    result = await bridge.call_tool(TOOL_SOP_COMPILE, {"sdk_spec": "garbage"})
    assert "error" in result
    assert "No SDK methods parsed" in result


@pytest.mark.asyncio
async def test_call_tool_skill_invoke_success() -> None:
    def fake_invoke(name: str, params: dict) -> str:
        return f"prompt-for-{name}-with-{json.dumps(params)}"

    bridge = TraeMcpBridge(skill_invoker=fake_invoke)
    result = await bridge.call_tool(TOOL_SKILL_INVOKE, {"skill_name": "dream", "params": {"x": 1}})
    assert "prompt-for-dream" in result
    assert '"x": 1' in result


@pytest.mark.asyncio
async def test_call_tool_skill_invoke_missing_name() -> None:
    bridge = TraeMcpBridge()
    result = await bridge.call_tool(TOOL_SKILL_INVOKE, {})
    assert "error" in result
    assert "skill_name" in result


@pytest.mark.asyncio
async def test_call_tool_skill_invoke_exception_surfaces() -> None:
    def boom(name: str, params: dict) -> str:
        raise KeyError(name)

    bridge = TraeMcpBridge(skill_invoker=boom)
    result = await bridge.call_tool(TOOL_SKILL_INVOKE, {"skill_name": "missing"})
    assert "error" in result
    assert "missing" in result


@pytest.mark.asyncio
async def test_call_tool_stability_gate_success() -> None:
    def fake_runner() -> str:
        return "exit=0 | 16 passed in 1.20s"

    bridge = TraeMcpBridge(stability_runner=fake_runner)
    result = await bridge.call_tool(TOOL_STABILITY_GATE, {})
    assert "exit=0" in result
    assert "16 passed" in result


@pytest.mark.asyncio
async def test_call_tool_stability_gate_exception_surfaces() -> None:
    def boom() -> str:
        raise RuntimeError("pytest missing")

    bridge = TraeMcpBridge(stability_runner=boom)
    result = await bridge.call_tool(TOOL_STABILITY_GATE, {})
    assert "error" in result
    assert "pytest missing" in result


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_default_orchestrator_enqueue_writes_ndjson(tmp_path: Path) -> None:
    """Verify default orchestrator enqueue writes ndjson."""
    bridge = TraeMcpBridge(config=BridgeConfig(reports_dir=str(tmp_path / ".reports")))
    result = await bridge.call_tool(
        TOOL_ORCH_RUN,
        {"issue_url": "https://example.com/i/1", "workflow_path": "./w.md"},
    )
    assert "queued run_id=" in result
    ndjson_files = list((tmp_path / ".reports").glob("*.ndjson"))
    assert len(ndjson_files) == 1
    record = json.loads(ndjson_files[0].read_text(encoding="utf-8"))
    assert record["issue_url"] == "https://example.com/i/1"
    assert record["workflow_path"] == "./w.md"
    assert record["event"] == "queued"
    assert "run_id" in record


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------


def test_default_stability_runner_runs_subprocess(tmp_path: Path) -> None:
    """Verify default stability runner runs subprocess."""
    import sys

    cfg = BridgeConfig(
        stability_gate_args=[sys.executable, "-c", "print('1 passed in 0.01s')"],
        stability_gate_cwd=str(tmp_path),
        stability_gate_timeout_s=10.0,
    )
    bridge = TraeMcpBridge(config=cfg)
    result = bridge._default_stability_runner()
    assert "exit=0" in result
    assert "1 passed" in result


def test_default_stability_runner_timeout(tmp_path: Path) -> None:
    """Verify default stability runner timeout."""
    import sys

    cfg = BridgeConfig(
        stability_gate_args=[sys.executable, "-c", "import time; time.sleep(5)"],
        stability_gate_cwd=str(tmp_path),
        stability_gate_timeout_s=0.3,
    )
    bridge = TraeMcpBridge(config=cfg)
    result = bridge._default_stability_runner()
    assert "error" in result
    assert "timed out" in result


def test_default_stability_runner_not_found() -> None:
    """Verify default stability runner not found."""
    cfg = BridgeConfig(stability_gate_args=["definitely-not-a-real-binary-xyz"])
    bridge = TraeMcpBridge(config=cfg)
    result = bridge._default_stability_runner()
    assert "error" in result
    assert "not found" in result


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------


def test_mcp_available_returns_bool() -> None:
    """Verify mcp available returns bool."""
    assert isinstance(mcp_available(), bool)


def test_build_mcp_server_raises_when_mcp_missing() -> None:
    """Verify build mcp server raises when mcp missing."""
    bridge = TraeMcpBridge()
    if mcp_available():
        pytest.skip("mcp installed — skip the missing-mcp path")
    with pytest.raises(ImportError, match="pip install mcp"):
        bridge._build_mcp_server()


def test_main_returns_2_when_mcp_missing(capsys) -> None:
    """Verify main returns 2 when mcp missing."""
    if mcp_available():
        pytest.skip("mcp installed — skip the missing-mcp path")
    from extensions.trae.mcp_bridge import _main

    rc = _main()
    assert rc == 2
    captured = capsys.readouterr()
    assert MCP_UNAVAILABLE in captured.err or MCP_UNAVAILABLE in captured.out


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------


def test_list_tools_returns_4_specs() -> None:
    bridge = TraeMcpBridge()
    specs = bridge.list_tools()
    assert len(specs) == 4
    assert all(s.description for s in specs)


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------


def test_call_tool_async_runs_in_new_loop() -> None:
    """Verify call tool async runs in new loop."""
    bridge = TraeMcpBridge(stability_runner=lambda: "exit=0 | ok")
    result = asyncio.run(bridge.call_tool(TOOL_STABILITY_GATE, {}))
    assert "exit=0" in result
