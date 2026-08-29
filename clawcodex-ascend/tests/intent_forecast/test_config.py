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

from clawcodex_ext.intent_forecast.config import IntentForecastConfig


def test_config_defaults() -> None:
    cfg = IntentForecastConfig()
    assert cfg.enabled is True
    assert cfg.idle_seconds == 120
    assert cfg.max_sessions == 12
    assert cfg.response_language == "auto"
    assert cfg.intent_strategy == "user"


def test_config_from_mapping_clamps_values() -> None:
    cfg = IntentForecastConfig.from_mapping(
        {
            "enabled": "off",
            "idle_seconds": 0,
            "min_confidence": 2,
            "feedback_enabled": "false",
        }
    )
    assert cfg.enabled is False
    assert cfg.idle_seconds == 1
    assert cfg.min_confidence == 1.0
    assert cfg.feedback_enabled is False


def test_config_response_language_override() -> None:
    assert IntentForecastConfig.from_mapping({"response_language": "zh-CN"}).response_language == "Chinese"
    assert IntentForecastConfig.from_mapping({"response_language": "english"}).response_language == "English"
    assert IntentForecastConfig.from_mapping({"response_language": "klingon"}).response_language == "auto"


def test_config_intent_strategy_is_one_of_supported_values() -> None:
    assert IntentForecastConfig.from_mapping({"intent_strategy": "workspace"}).intent_strategy == "workspace"
    assert IntentForecastConfig.from_mapping({"intent_strategy": "history-first"}).intent_strategy == "history"
    assert IntentForecastConfig.from_mapping({"intent_strategy": "project"}).intent_strategy == "workspace"
    assert IntentForecastConfig.from_mapping({"intent_strategy": "mixed"}).intent_strategy == "user"
