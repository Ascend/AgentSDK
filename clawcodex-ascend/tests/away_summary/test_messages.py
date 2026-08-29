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

"""Tests for away-summary message helpers."""

from __future__ import annotations

from clawcodex_ext.away_summary.messages import (
    create_away_summary_message,
    format_away_summary_for_display,
)
from clawcodex_ext.types.messages import SystemMessage


def test_format_away_summary_for_display_strips_metadata() -> None:
    """The persisted system message starts with '[AWAY SUMMARY]...' metadata;
    the display formatter must strip that and expose only the recap body.
    """
    msg = create_away_summary_message(
        summary="You were fixing login.",
        trigger="manual",
        fingerprint="abc",
        message_count=5,
    )
    text = format_away_summary_for_display(msg)
    assert "[AWAY SUMMARY]" not in text
    assert "fingerprint=" not in text
    assert "You were fixing login." in text


def test_format_away_summary_for_display_uses_double_newline() -> None:
    """Rich Markdown folds a single newline into a space, which caused
    'Recapitulate' and the recap body to render on the same line.
    The formatter must insert a blank line (double newline) so the
    prefix and body become two distinct paragraphs.
    """
    text = format_away_summary_for_display("Fix the login bug.")
    assert text.startswith("Recapitulate\n\n")
    assert "Recapitulate\nFix" not in text
    assert text == "Recapitulate\n\nFix the login bug."


def test_format_away_summary_for_display_handles_away_summary_prefix() -> None:
    """If the input is a persisted system message that already carries the
    '[AWAY SUMMARY]' metadata block, the formatter must strip that block
    and add the display prefix with the double-newline paragraph break.
    """
    raw = "[AWAY SUMMARY]\ntrigger=auto fingerprint=fp1 model=\n\nAlready prefixed."
    text = format_away_summary_for_display(raw)
    assert text == "Recapitulate\n\nAlready prefixed."


def test_format_away_summary_for_display_handles_object_with_content() -> None:
    """Passing an object with a ``content`` attribute works like a string."""
    msg = SystemMessage(content="Check the dashboard.")
    text = format_away_summary_for_display(msg)
    assert text == "Recapitulate\n\nCheck the dashboard."


def test_create_away_summary_message_stores_metadata() -> None:
    msg = create_away_summary_message(
        summary="Fix login.",
        trigger="auto",
        fingerprint="fp1",
        message_count=3,
        model="test-model",
    )
    assert msg.role == "system"
    assert msg.subtype == "away_summary"
    assert "[AWAY SUMMARY]" in msg.content
    assert "trigger=auto" in msg.content
    assert "fingerprint=fp1" in msg.content
    assert "model=test-model" in msg.content
    assert "Fix login." in msg.content
    assert msg.data["away_summary"]["trigger"] == "auto"
    assert msg.data["away_summary"]["fingerprint"] == "fp1"
    assert msg.data["away_summary"]["message_count"] == 3
    assert msg.data["away_summary"]["model"] == "test-model"
