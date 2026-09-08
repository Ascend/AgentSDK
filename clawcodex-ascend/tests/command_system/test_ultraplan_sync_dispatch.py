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

"""Regression tests for the /ultraplan SYNC dispatch slot.

Headless non-interactive mode (and the REPL's synchronous fallback) executes
local commands through ``execute_command_sync``, which invokes ``_call_impl``
WITHOUT awaiting it. Ultraplan's body is async, so the command publishes a
sync adapter in that slot that drives its own event loop — it must never leak
a bare coroutine (the historical "'coroutine' object has no attribute
'value'" crash). The ``/ultra`` / ``/up`` aliases are wired here too.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from clawcodex_ext.command_system.builtins import (
    execute_command_sync,
    register_builtin_commands,
)
from clawcodex_ext.command_system.engine import create_command_context
from clawcodex_ext.command_system.registry import CommandRegistry
from clawcodex_ext.command_system.types import LocalCommandResult
from clawcodex_ext.command_system.ultraplan_command import (
    ULTRAPLAN_COMMAND,
    run_ultraplan_command_sync,
)


@pytest.fixture
def registry() -> CommandRegistry:
    """Fresh private registry with built-in commands only (no runtime)."""
    reg = CommandRegistry()
    register_builtin_commands(reg)
    return reg


@pytest.fixture
def ctx(tmp_path: Path):
    """Minimal command context for sync execution."""
    return create_command_context(workspace_root=tmp_path)


def _patch_registry(monkeypatch: pytest.MonkeyPatch, registry: CommandRegistry) -> None:
    monkeypatch.setattr(
        "clawcodex_ext.command_system.builtins.get_command_registry",
        lambda: registry,
    )


# ---------------------------------------------------------------------------
# The sync slot itself (defect 1 regression)
# ---------------------------------------------------------------------------


def test_ultraplan_sync_slot_returns_result_not_coroutine(ctx) -> None:
    """``_call_impl`` must produce a completed result, never a coroutine."""
    result = ULTRAPLAN_COMMAND._call_impl("help", ctx)  # type: ignore[arg-type]
    assert isinstance(result, LocalCommandResult)
    assert not asyncio.iscoroutine(result)
    assert result.value.startswith("Usage:")


def test_ultraplan_command_call_impl_is_the_sync_adapter() -> None:
    """The published sync slot drives the async body to completion."""
    assert ULTRAPLAN_COMMAND._call_impl is run_ultraplan_command_sync  # type: ignore[arg-type]


async def test_ultraplan_sync_slot_tolerates_a_running_loop(ctx) -> None:
    """Inside an async context ``asyncio.run`` is illegal — the adapter must
    fall back to a worker-thread loop instead of raising.
    """
    # This test body itself runs on the pytest-asyncio event loop, so a
    # naive ``asyncio.run`` inside the slot would raise RuntimeError.
    result = ULTRAPLAN_COMMAND._call_impl("help", ctx)  # type: ignore[arg-type]
    assert isinstance(result, LocalCommandResult)
    assert result.value.startswith("Usage:")


# ---------------------------------------------------------------------------
# execute_command_sync (the headless non-interactive path)
# ---------------------------------------------------------------------------


def test_execute_command_sync_ultraplan_help(
    monkeypatch: pytest.MonkeyPatch,
    registry: CommandRegistry,
    ctx,
) -> None:
    _patch_registry(monkeypatch, registry)
    success, text, error = execute_command_sync("ultraplan", "help", ctx)
    assert success, f"/ultraplan help failed: {error}"
    assert error is None
    assert text is not None and text.startswith("Usage:")


def test_execute_command_sync_ultraplan_ls_empty(
    monkeypatch: pytest.MonkeyPatch,
    registry: CommandRegistry,
    ctx,
    tmp_path: Path,
) -> None:
    """``/ultraplan ls`` works end-to-end through the sync dispatcher."""
    monkeypatch.setenv("CLAWCODEX_ULTRAPLAN_DIR", str(tmp_path / "ultraplan"))
    _patch_registry(monkeypatch, registry)
    success, text, error = execute_command_sync("ultraplan", "ls", ctx)
    assert success, f"/ultraplan ls failed: {error}"
    assert error is None
    assert text is not None and "No ultraplan plans found." in text


def test_execute_command_sync_ultraplan_create_gate_off_is_friendly(
    monkeypatch: pytest.MonkeyPatch,
    registry: CommandRegistry,
    ctx,
) -> None:
    """A gated-off LLM planner must surface a friendly message, not an
    exception traceback, through the sync dispatcher.
    """
    monkeypatch.setenv("ULTRAPLAN_LLM_PLANNER", "off")
    _patch_registry(monkeypatch, registry)
    success, text, error = execute_command_sync("ultraplan", "create some goal", ctx)
    assert success
    assert error is None
    assert text is not None and "ULTRAPLAN_LLM_PLANNER is disabled" in text


# ---------------------------------------------------------------------------
# /ultra and /up aliases (defect 3 wiring)
# ---------------------------------------------------------------------------


def test_ultraplan_aliases_registered_on_command() -> None:
    """The aliases stay aligned with the trigger vocabulary."""
    assert set(ULTRAPLAN_COMMAND.aliases) >= {"ultra", "up"}


def test_command_registry_resolves_ultraplan_aliases() -> None:
    reg = CommandRegistry()
    reg.register(ULTRAPLAN_COMMAND)
    assert reg.get("ultra") is ULTRAPLAN_COMMAND
    assert reg.get("up") is ULTRAPLAN_COMMAND
    assert reg.get("Ultra") is ULTRAPLAN_COMMAND


@pytest.mark.parametrize("alias", ["ultra", "up"])
def test_execute_command_sync_ultraplan_alias_help(
    monkeypatch: pytest.MonkeyPatch,
    registry: CommandRegistry,
    ctx,
    alias: str,
) -> None:
    """``/ultra`` and ``/up`` resolve to the ultraplan command."""
    _patch_registry(monkeypatch, registry)
    success, text, error = execute_command_sync(alias, "help", ctx)
    assert success, f"/{alias} help failed: {error}"
    assert error is None
    assert text is not None and text.startswith("Usage:")


def test_execute_command_sync_alias_works_without_registry_entry(
    monkeypatch: pytest.MonkeyPatch,
    ctx,
) -> None:
    """The builtins fallback scan also matches aliases, so headless runs whose
    workspace registry lacks the alias still dispatch it.
    """
    empty = CommandRegistry()
    _patch_registry(monkeypatch, empty)
    success, text, error = execute_command_sync("ultra", "help", ctx)
    assert success, f"/ultra help failed: {error}"
    assert error is None
    assert text is not None and text.startswith("Usage:")
