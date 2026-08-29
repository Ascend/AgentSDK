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

"""Tests for src/services/chrome/models.py."""
# pylint: disable=no-name-in-module

from __future__ import annotations

import json

import pytest

from clawcodex_ext.services.chrome.models import ChromeActionResult, ChromeActionType


def test_enum_completeness() -> None:
    """Every action the spec documents must be in the enum."""
    expected = {
        "navigate",
        "click",
        "type",
        "select",
        "screenshot",
        "eval_js",
        "get_html",
        "get_text",
        "hover",
        "scroll",
    }
    actual = {member.value for member in ChromeActionType}
    assert actual == expected


def test_enum_is_str() -> None:
    """The enum should serialise as its value (string)."""
    assert ChromeActionType.NAVIGATE == "navigate"
    assert json.dumps(ChromeActionType.CLICK) == '"click"'


def test_result_minimal_construction() -> None:
    r = ChromeActionResult(success=True)
    assert r.success is True
    assert r.data is None
    assert r.error is None
    assert r.url == ""
    assert r.screenshot_path is None
    assert r.elapsed_ms == 0.0
    assert r.action_type is None
    assert r.metadata is None


def test_result_is_frozen() -> None:
    r = ChromeActionResult(success=True, data="hello")
    with pytest.raises((AttributeError, Exception)):
        r.success = False  # type: ignore[misc]


def test_result_carries_bytes_data() -> None:
    payload = b"\x89PNG\r\n\x1a\n"  # PNG magic
    r = ChromeActionResult(success=True, data=payload, action_type=ChromeActionType.SCREENSHOT)
    assert r.data == payload
    assert r.action_type is ChromeActionType.SCREENSHOT


def test_result_carries_error_message() -> None:
    r = ChromeActionResult(
        success=False,
        error="connection refused",
        action_type=ChromeActionType.NAVIGATE,
    )
    assert r.success is False
    assert r.error == "connection refused"
    assert r.action_type is ChromeActionType.NAVIGATE


def test_result_metadata_dict() -> None:
    r = ChromeActionResult(
        success=True,
        data={"url": "https://example.com"},
        metadata={"selector": "body", "attempt": 1},
    )
    assert r.metadata == {"selector": "body", "attempt": 1}


def test_result_equality() -> None:
    """Two results with the same fields are equal (frozen dataclass)."""
    a = ChromeActionResult(success=True, data="x", action_type=ChromeActionType.CLICK)
    b = ChromeActionResult(success=True, data="x", action_type=ChromeActionType.CLICK)
    assert a == b


def test_result_hashable() -> None:
    """Frozen dataclass → usable as dict key / in sets."""
    a = ChromeActionResult(success=True, data="x")
    b = ChromeActionResult(success=True, data="y")
    bucket = {a, b}
    assert len(bucket) == 2
