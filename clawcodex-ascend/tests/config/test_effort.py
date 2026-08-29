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

"""Tests for R2-WS-9: Effort system."""

from __future__ import annotations


from src.utils.effort import (
    EffortLevel,
    detect_effort_from_text,
    get_max_tokens_for_effort,
    resolve_applied_effort,
)


class TestEffortLevel:
    def test_enum_values(self):
        assert EffortLevel.LOW.value == "low"
        assert EffortLevel.MEDIUM.value == "medium"
        assert EffortLevel.HIGH.value == "high"
        assert EffortLevel.MAX.value == "max"


class TestResolveEffort:
    def test_user_effort_takes_priority(self):
        result = resolve_applied_effort(user_effort="high", config_effort="low")
        assert result == EffortLevel.HIGH

    def test_config_fallback(self):
        result = resolve_applied_effort(config_effort="max")
        assert result == EffortLevel.MAX

    def test_model_default_fallback(self):
        result = resolve_applied_effort(model_default="low")
        assert result == EffortLevel.LOW

    def test_default_medium(self):
        result = resolve_applied_effort()
        assert result == EffortLevel.MEDIUM

    def test_keyword_resolution(self):
        result = resolve_applied_effort(user_effort="thorough")
        assert result == EffortLevel.HIGH

        result = resolve_applied_effort(user_effort="quick")
        assert result == EffortLevel.LOW


class TestMaxTokensForEffort:
    def test_low(self):
        assert get_max_tokens_for_effort(EffortLevel.LOW) == 4_096

    def test_medium(self):
        assert get_max_tokens_for_effort(EffortLevel.MEDIUM) == 8_192

    def test_high(self):
        assert get_max_tokens_for_effort(EffortLevel.HIGH) == 16_384

    def test_max(self):
        assert get_max_tokens_for_effort(EffortLevel.MAX) == 32_768


class TestDetectEffort:
    def test_effort_flag(self):
        assert detect_effort_from_text("please do this --effort high") == EffortLevel.HIGH

    def test_thorough_flag(self):
        assert detect_effort_from_text("do this --thorough") == EffortLevel.HIGH

    def test_quick_flag(self):
        assert detect_effort_from_text("just --quick check") == EffortLevel.LOW

    def test_max_flag(self):
        assert detect_effort_from_text("go --max on this") == EffortLevel.MAX

    def test_no_effort_indicator(self):
        assert detect_effort_from_text("just a normal request") is None

    def test_effort_flag_with_value(self):
        assert detect_effort_from_text("--effort low do stuff") == EffortLevel.LOW
