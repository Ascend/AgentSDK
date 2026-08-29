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


# pylint: disable=E0611
from pathlib import Path
from types import SimpleNamespace

import pytest

from clawcodex_ext.agent.tool_authoring.call_handlers import bash as bash_handler
from clawcodex_ext.agent.tool_authoring.factory import _catalog_execution_context
from clawcodex_ext.agent.tool_authoring.spec import AgentToolSpec
from clawcodex_ext.tool_system.context import ToolContext


def test_catalog_execution_context_reads_session_and_dual_write_from_env(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle_dir = tmp_path / "bundle"
    bundle_dir.mkdir()
    monkeypatch.setenv("CLAWCODEX_SESSION_ID", "  sess-42  ")
    monkeypatch.setenv("CLAWCODEX_CATALOG_DUAL_WRITE", "YES")
    context = ToolContext(
        workspace_root=tmp_path,
        bundle_context=SimpleNamespace(
            bundle_path=bundle_dir,
            bundle_name="demo-bundle",
        ),
    )
    spec = AgentToolSpec(
        name="demo-workflow",
        description="demo",
        input_schema={"type": "object", "properties": {}},
        call_type="workflow",
        call_impl={},
    )

    catalog_ctx = _catalog_execution_context(spec, context)

    assert catalog_ctx.session_id == "sess-42"
    assert catalog_ctx.dual_write is True
    assert catalog_ctx.bundle_path == bundle_dir
    assert catalog_ctx.bundle_id == "demo-bundle"


def test_catalog_execution_context_prefers_tool_context_session_id(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle_dir = tmp_path / "bundle"
    bundle_dir.mkdir()
    monkeypatch.setenv("CLAWCODEX_SESSION_ID", "env-session")
    context = ToolContext(
        workspace_root=tmp_path,
        session_id="ctx-session",
        bundle_context=SimpleNamespace(
            bundle_path=bundle_dir,
            bundle_name="demo-bundle",
        ),
    )
    spec = AgentToolSpec(
        name="demo-workflow",
        description="demo",
        input_schema={"type": "object", "properties": {}},
        call_type="workflow",
        call_impl={},
    )

    catalog_ctx = _catalog_execution_context(spec, context)

    assert catalog_ctx.session_id == "ctx-session"


def test_execute_bash_injects_session_id_from_tool_context(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("CLAWCODEX_SESSION_ID", raising=False)
    captured: dict[str, object] = {}

    def fake_run(argv, **kwargs):
        captured["kwargs"] = kwargs
        return 0, "ok", "", False, False

    monkeypatch.setattr(bash_handler, "_run_subprocess_with_abort", fake_run)

    bash_handler.execute_bash(
        "echo ok",
        {},
        context=ToolContext(workspace_root=tmp_path, session_id="from-context"),
    )

    env = captured["kwargs"]["env"]
    assert env["CLAWCODEX_SESSION_ID"] == "from-context"


def test_execute_bash_forwards_catalog_env_without_bundle_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CLAWCODEX_SESSION_ID", "sess-forward")
    monkeypatch.setenv("CLAWCODEX_CATALOG_DUAL_WRITE", "1")
    captured: dict[str, object] = {}

    def fake_run(argv, **kwargs):
        captured["kwargs"] = kwargs
        return 0, "ok", "", False, False

    monkeypatch.setattr(bash_handler, "_run_subprocess_with_abort", fake_run)

    output = bash_handler.execute_bash(
        "echo ok",
        {},
        context=ToolContext(workspace_root=tmp_path),
    )

    assert output == "ok"
    env = captured["kwargs"]["env"]
    assert env["CLAWCODEX_SESSION_ID"] == "sess-forward"
    assert env["CLAWCODEX_CATALOG_DUAL_WRITE"] == "1"


def test_execute_bash_forwards_catalog_env_with_bundle_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle_dir = tmp_path / "bundle"
    bundle_dir.mkdir()
    monkeypatch.setenv("CLAWCODEX_SESSION_ID", "sess-bundle")
    monkeypatch.setenv("CLAWCODEX_CATALOG_DUAL_WRITE", "true")
    captured: dict[str, object] = {}

    def fake_run(argv, **kwargs):
        captured["kwargs"] = kwargs
        return 0, "ok", "", False, False

    monkeypatch.setattr(bash_handler, "_run_subprocess_with_abort", fake_run)

    bash_handler.execute_bash(
        "echo ok",
        {},
        context=ToolContext(
            workspace_root=tmp_path,
            bundle_context=SimpleNamespace(bundle_path=bundle_dir),
        ),
    )

    env = captured["kwargs"]["env"]
    assert env["CLAWCODEX_BUNDLE_PATH"] == str(bundle_dir)
    assert env["CLAWCODEX_SESSION_ID"] == "sess-bundle"
    assert env["CLAWCODEX_CATALOG_DUAL_WRITE"] == "true"
