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

from clawcodex_ext.services.ultraplan.keyword_detector import (
    find_ultraplan_trigger_positions,
    is_ultraplan_command,
    replace_ultraplan_keyword,
)


def test_detects_command_at_line_start_after_spaces() -> None:
    hits = find_ultraplan_trigger_positions("  /ultraplan refactor executor.py")
    assert hits[0].start == 2
    assert hits[0].keyword == "/ultraplan"
    assert is_ultraplan_command("  /ultraplan refactor executor.py")


def test_does_not_treat_middle_text_as_submit_command() -> None:
    assert find_ultraplan_trigger_positions("echo /ultraplan") != []
    assert not is_ultraplan_command("echo /ultraplan")


def test_skips_escaped_and_quoted_triggers() -> None:
    assert find_ultraplan_trigger_positions("\\/ultraplan foo") == []
    assert find_ultraplan_trigger_positions('"/ultraplan"') == []
    assert find_ultraplan_trigger_positions("`/ultraplan`") == []


def test_skips_when_inside_code_fence() -> None:
    assert find_ultraplan_trigger_positions("/ultraplan foo", inside_code_fence=True) == []


def test_replace_only_detected_keywords() -> None:
    text = "/up foo and '/up bar'"
    assert replace_ultraplan_keyword(text, "/up", "/ultraplan") == "/ultraplan foo and '/up bar'"
