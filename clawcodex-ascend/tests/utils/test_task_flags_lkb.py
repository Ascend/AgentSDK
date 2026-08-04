#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# -------------------------------------------------------------------------
#  This file is part of the AgentSDK project.
# Copyright (c) 2026 Huawei Technologies Co.,Ltd.
# Copyright (c) 2026 Clawd Codex Team
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

"""Tests for the LKB-driven TaskV2 gating in headless sessions.

Merged-flag semantics: enabling the single ``LKB_PLAN_GRAPH`` feature flag
must make :func:`clawcodex_ext.utils.task_flags.is_todo_v2_enabled` return
True even in non-interactive (headless/SDK) sessions without
``CLAUDE_CODE_ENABLE_TASKS``; with the flag off, headless sessions get False.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from clawcodex_ext.utils.task_flags import is_todo_v2_enabled


@pytest.fixture(autouse=True)
def _clean_state(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Isolate env vars, feature-gate tiers, and session interactivity."""
    monkeypatch.delenv("CLAUDE_CODE_ENABLE_TASKS", raising=False)
    monkeypatch.delenv("CLAWCODEX_FEATURE_LKB_PLAN_GRAPH", raising=False)
    # Fallback-registry env read by ``lkb.flags`` when the clawcodex feature
    # gate is unavailable.
    monkeypatch.delenv("LKB_FEATURE_LKB_PLAN_GRAPH", raising=False)
    # Force a non-interactive (headless) session: task_flags imported
    # get_is_non_interactive_session by name, so patch it there.
    monkeypatch.setattr("clawcodex_ext.utils.task_flags.get_is_non_interactive_session", lambda: True)
    from clawcodex_ext.feature_gate import get_registry
    from clawcodex_ext.feature_gate.config import ConfigStore

    registry = get_registry()
    registry._overrides.pop("LKB_PLAN_GRAPH", None)
    # Tier-3 persisted config (~/.clawcodex/features.json) is machine-local
    # state; swap in an empty store so the tests are hermetic.
    monkeypatch.setattr(registry, "_config_store", ConfigStore(config_dir=tmp_path))


def test_headless_flag_off_returns_false() -> None:
    """Non-interactive session, flag off → TaskV2 tools hidden."""
    assert is_todo_v2_enabled() is False


def test_headless_flag_on_via_override_returns_true(monkeypatch: pytest.MonkeyPatch) -> None:
    """Non-interactive session + LKB_PLAN_GRAPH override → TaskV2 enabled."""
    from clawcodex_ext.feature_gate import get_registry

    monkeypatch.setitem(get_registry()._overrides, "LKB_PLAN_GRAPH", True)
    assert is_todo_v2_enabled() is True


def test_headless_flag_on_via_env_returns_true(monkeypatch: pytest.MonkeyPatch) -> None:
    """Non-interactive session + CLAWCODEX_FEATURE_LKB_PLAN_GRAPH=1 → enabled."""
    monkeypatch.setenv("CLAWCODEX_FEATURE_LKB_PLAN_GRAPH", "1")
    assert is_todo_v2_enabled() is True


def test_interactive_session_unaffected_by_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    """Interactive sessions keep TaskV2 enabled regardless of the flag."""
    monkeypatch.setattr("clawcodex_ext.utils.task_flags.get_is_non_interactive_session", lambda: False)
    assert is_todo_v2_enabled() is True
