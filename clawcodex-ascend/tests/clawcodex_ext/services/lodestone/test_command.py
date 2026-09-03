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

"""Tests for /link command + LodestoneTool."""
# pylint: disable=no-name-in-module

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from clawcodex_ext.command_system.engine import CommandContext
from clawcodex_ext.command_system.lodestone_commands import (
    LODESTONE_COMMAND,
    register_lodestone_commands,
)
from clawcodex_ext.command_system.registry import CommandRegistry
from clawcodex_ext.services.lodestone import (
    reset_default_service,
)


def _ctx(workspace_root: Path | None = None, *, session_id: str | None = "sess-1") -> CommandContext:
    ctx = CommandContext(workspace_root=workspace_root, cwd=workspace_root)
    setattr(ctx, "session_id", session_id)  # tolerate test mocks
    return ctx


@pytest.fixture(autouse=True)
def _reset_lodestone_singleton(tmp_path, monkeypatch):
    """Tests below mutate the singleton's config; isolate them."""
    monkeypatch.setenv("CLAWCODEX_CONFIG_DIR", str(tmp_path))
    reset_default_service()
    yield
    reset_default_service()


async def test_register_lodestone_commands_adds_to_registry():
    reg = CommandRegistry()
    register_lodestone_commands(reg)
    cmd = reg.get("link")
    assert cmd is not None


def test_link_parse_outputs_anchor_payload():
    ctx = _ctx(workspace_root=Path("/abs").resolve())
    result = asyncio.run(LODESTONE_COMMAND.call("parse src/foo.py:42:13 and #123", ctx))
    payload = json.loads(result.value)
    kinds = [a["kind"] for a in payload]
    assert "file_path" in kinds
    assert "tracker_issue" in kinds


def test_link_resolve_renders_anchors():
    ctx = _ctx(workspace_root=Path("/abs").resolve())
    result = asyncio.run(LODESTONE_COMMAND.call("resolve src/foo.py:42:13", ctx))
    assert "vscode" in result.value or "file" in result.value


def test_link_status_lists_current_config():
    ctx = _ctx()
    result = asyncio.run(LODESTONE_COMMAND.call("status", ctx))
    assert "default_editor" in result.value
    assert "renderer" in result.value


def test_link_targets_list_lists_built_ins():
    ctx = _ctx()
    result = asyncio.run(LODESTONE_COMMAND.call("targets list", ctx))
    assert "vscode" in result.value
    assert "gitcode" in result.value


def test_link_config_mutates_and_persists(tmp_path, monkeypatch):
    monkeypatch.setenv("CLAWCODEX_CONFIG_DIR", str(tmp_path))
    ctx = _ctx()
    result = asyncio.run(LODESTONE_COMMAND.call("config editor=cursor", ctx))
    assert "default_editor" in result.value
    # Read back via load_config
    from clawcodex_ext.services.lodestone.config import load_config

    cfg = load_config(path=tmp_path / "lodestone.json")
    assert cfg is not None
    assert cfg.default_editor == "cursor"


def test_link_targets_register_then_unregister():
    ctx = _ctx()
    asyncio.run(LODESTONE_COMMAND.call("targets register my-vscode file_path vscode://file/{abs}:{line}:{col}", ctx))
    result = asyncio.run(LODESTONE_COMMAND.call("targets list", ctx))
    assert "my-vscode" in result.value
    asyncio.run(LODESTONE_COMMAND.call("targets unregister my-vscode", ctx))
    result = asyncio.run(LODESTONE_COMMAND.call("targets list", ctx))
    assert "my-vscode" not in result.value


def test_link_targets_register_rejects_bad_placeholder():
    ctx = _ctx()
    result = asyncio.run(LODESTONE_COMMAND.call("targets register bad file_path vscode://file/{nope}:{line}", ctx))
    assert "disallowed placeholder" in result.value or "register failed" in result.value


def test_link_targets_test_renders_via_overridden_target():
    ctx = _ctx(workspace_root=Path("/abs").resolve())
    result = asyncio.run(LODESTONE_COMMAND.call("targets test cursor src/foo.py:42", ctx))
    assert "cursor://file/" in result.value
