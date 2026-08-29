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

import pytest

from clawcodex_ext.skills.bundled.debug import register_debug_skill
from clawcodex_ext.skills.bundled.loop import register_loop_skill
from clawcodex_ext.skills.bundled_skills import (
    clear_bundled_skills,
    get_registered_bundled_skills,
)


@pytest.fixture(autouse=True)
def _reset_bundled_skills():
    clear_bundled_skills()
    yield
    clear_bundled_skills()


def _registered_skill(name: str):
    return next(skill for skill in get_registered_bundled_skills() if skill.name == name)


def test_loop_is_unavailable_when_cron_is_disabled(monkeypatch) -> None:
    monkeypatch.setenv("CLAWCODEX_DISABLE_CRON", "true")
    monkeypatch.delenv("CLAUDE_CODE_DISABLE_CRON", raising=False)
    assert register_loop_skill() is True

    assert _registered_skill("loop").is_enabled() is False


def test_loop_primary_gate_takes_precedence_over_legacy_gate(monkeypatch) -> None:
    monkeypatch.setenv("CLAWCODEX_DISABLE_CRON", "false")
    monkeypatch.setenv("CLAUDE_CODE_DISABLE_CRON", "true")
    assert register_loop_skill() is True

    assert _registered_skill("loop").is_enabled() is True


def test_debug_does_not_claim_a_noop_enabled_logging(tmp_path, monkeypatch) -> None:
    debug_log = tmp_path / "missing-debug.log"
    monkeypatch.setenv("CLAUDE_CODE_DEBUG_LOG_PATH", str(debug_log))
    assert register_debug_skill() is True

    prompt = _registered_skill("debug").get_prompt()

    assert "No debug logging was enabled" in prompt
    assert "logging was just enabled" not in prompt
    assert "logging is now active" not in prompt
