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

# pylint: disable=no-name-in-module
from clawcodex_ext.away_summary.config import AwaySummaryConfig


def test_default_config_values() -> None:
    cfg = AwaySummaryConfig.from_mapping({})
    assert cfg.enabled is True
    assert cfg.idle_seconds == 300
    assert cfg.recap_command_enabled is True
    assert cfg.min_turns == 1


def test_config_accepts_idle_minutes_fallback() -> None:
    cfg = AwaySummaryConfig.from_mapping({"idle_minutes": 3})
    assert cfg.idle_seconds == 180


def test_config_normalizes_invalid_values() -> None:
    cfg = AwaySummaryConfig.from_mapping(
        {
            "enabled": "false",
            "idle_seconds": -5,
            "recap_command_enabled": "off",
            "min_turns": -1,
            "max_input_tokens": 10,
            "max_output_tokens": 10,
        }
    )
    assert cfg.enabled is False
    assert cfg.idle_seconds == 1
    assert cfg.recap_command_enabled is False
    assert cfg.min_turns == 0
    assert cfg.max_input_tokens == 256
    assert cfg.max_output_tokens == 64


def test_config_accepts_common_simplified_chinese_aliases() -> None:
    aliases = [
        "zh-hans",
        "zh-Hans",
        "chinese simplified",
        "zhongwen",
        "zhong wen",
    ]
    for value in aliases:
        cfg = AwaySummaryConfig.from_mapping({"response_language": value})
        assert cfg.response_language == "Chinese"
