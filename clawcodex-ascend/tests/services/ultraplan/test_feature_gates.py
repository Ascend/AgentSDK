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

# pylint: disable=no-name-in-module

from __future__ import annotations

from clawcodex_ext.services.ultraplan.feature_gates import (
    is_ccr_endpoint_allowed,
    is_ultraplan_llm_enabled,
    is_ultraplan_rainbow_enabled,
    is_ultraplan_remote_enabled,
)


def test_ultraplan_feature_gate_defaults(monkeypatch) -> None:
    monkeypatch.delenv("ULTRAPLAN_LLM_PLANNER", raising=False)
    monkeypatch.delenv("ULTRAPLAN_REMOTE", raising=False)
    monkeypatch.delenv("ULTRAPLAN_RAINBOW", raising=False)
    assert is_ultraplan_llm_enabled()
    assert not is_ultraplan_remote_enabled()
    assert is_ultraplan_rainbow_enabled()


def test_ultraplan_legacy_env_overrides(monkeypatch) -> None:
    monkeypatch.setenv("ULTRAPLAN_LLM_PLANNER", "off")
    monkeypatch.setenv("ULTRAPLAN_REMOTE", "on")
    monkeypatch.setenv("ULTRAPLAN_RAINBOW", "0")
    assert not is_ultraplan_llm_enabled()
    assert is_ultraplan_remote_enabled()
    assert not is_ultraplan_rainbow_enabled()


def test_ccr_allowlist(monkeypatch) -> None:
    monkeypatch.setenv("CCR_ALLOWLIST", "good.example:443,http://localhost:9999")
    assert is_ccr_endpoint_allowed("https://good.example:443")
    assert is_ccr_endpoint_allowed("http://localhost:9999")
    assert not is_ccr_endpoint_allowed("https://bad.example")
