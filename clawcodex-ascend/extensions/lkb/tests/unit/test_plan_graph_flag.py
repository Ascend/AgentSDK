#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# -------------------------------------------------------------------------
# This file is part of the AgentSDK project.
# Copyright (c) 2026 Huawei Technologies Co.,Ltd.
# Copyright (c) 2026 Clawd Codex Team
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

"""Tests for the single current LKB feature flag."""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("CLAWCODEX_FEATURE_LKB_PLAN_GRAPH", raising=False)
    monkeypatch.delenv("LKB_FEATURE_LKB_PLAN_GRAPH", raising=False)


def test_plan_graph_can_be_disabled_by_host_override(monkeypatch: pytest.MonkeyPatch) -> None:
    from clawcodex_ext.feature_gate import get_registry
    from lkb.flags import is_plan_graph_enabled

    monkeypatch.setitem(get_registry()._overrides, "LKB_PLAN_GRAPH", False)
    assert is_plan_graph_enabled() is False


def test_plan_graph_constant_name() -> None:
    from lkb.flags import PLAN_GRAPH_FEATURE_NAME

    assert PLAN_GRAPH_FEATURE_NAME == "LKB_PLAN_GRAPH"


def test_plan_graph_enabled_via_host_env(monkeypatch: pytest.MonkeyPatch) -> None:
    from lkb.flags import is_plan_graph_enabled

    monkeypatch.setenv("CLAWCODEX_FEATURE_LKB_PLAN_GRAPH", "1")
    assert is_plan_graph_enabled() is True


def test_programmatic_override(monkeypatch: pytest.MonkeyPatch) -> None:
    from clawcodex_ext.feature_gate import get_registry
    from lkb.flags import is_plan_graph_enabled

    registry = get_registry()
    monkeypatch.setitem(registry._overrides, "LKB_PLAN_GRAPH", True)
    assert is_plan_graph_enabled() is True
