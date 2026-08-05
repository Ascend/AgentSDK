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

"""Tests for the channel send-result / health / validation contract."""

from __future__ import annotations

# Sibling clawcodex_ext packages (channels source) are migrated in separate
# branches; suppress E0611 until the full series lands.
# pylint: disable=no-name-in-module

import pytest

from clawcodex_ext.services.channels.results import (
    ChannelHealth,
    ChannelSendResult,
    ErrorCategory,
    SendStatus,
    ValidationResult,
)


def test_send_result_success_factory() -> None:
    r = ChannelSendResult.success("wechat-main", provider_receipt="mid_123")
    assert r.ok is True
    assert r.status is SendStatus.SUCCESS
    assert r.retryable is False
    assert r.provider_receipt == "mid_123"
    assert r.error_category is ErrorCategory.NONE


def test_send_result_retryable_error_requires_retryable_category() -> None:
    r = ChannelSendResult.retryable_error("wechat-main", message="boom", category=ErrorCategory.SERVER_ERROR)
    assert r.ok is False
    assert r.retryable is True
    assert r.status is SendStatus.RETRYABLE_ERROR

    with pytest.raises(ValueError):
        ChannelSendResult.retryable_error("wechat-main", message="x", category=ErrorCategory.AUTH)


def test_send_result_nonretryable_error() -> None:
    r = ChannelSendResult.nonretryable_error("wechat-main", message="nope", category=ErrorCategory.AUTH)
    assert r.ok is False
    assert r.retryable is False
    assert r.status is SendStatus.NONRETRYABLE_ERROR


def test_send_result_rate_limited_is_visible_and_retryable() -> None:
    r = ChannelSendResult.rate_limited("wechat-main", message="rate limited", raw={"retry_after_seconds": 10})
    assert r.ok is False
    assert r.retryable is True
    assert r.status is SendStatus.RATE_LIMITED
    assert r.error_category is ErrorCategory.RATE_LIMIT
    assert r.raw == {"retry_after_seconds": 10}


def test_send_result_unsupported() -> None:
    r = ChannelSendResult.unsupported("wechat-main", message="no media")
    assert r.ok is False
    assert r.status is SendStatus.UNSUPPORTED
    assert r.retryable is False


def test_send_result_retryable_property_matches_category() -> None:
    assert ChannelSendResult(
        ok=False,
        status=SendStatus.RETRYABLE_ERROR,
        channel_id="c",
        error_category=ErrorCategory.RATE_LIMIT,
    ).retryable
    assert not ChannelSendResult(
        ok=False,
        status=SendStatus.NONRETRYABLE_ERROR,
        channel_id="c",
        error_category=ErrorCategory.FORMAT,
    ).retryable
    # ok result is never retryable even if category somehow retryable
    assert not ChannelSendResult(
        ok=True,
        status=SendStatus.SUCCESS,
        channel_id="c",
    ).retryable


def test_send_result_rejects_bad_types() -> None:
    with pytest.raises(TypeError):
        ChannelSendResult(ok=True, status="success", channel_id="c")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        ChannelSendResult(ok=True, status=SendStatus.SUCCESS, channel_id="c", attempts=0)


def test_send_result_to_dict_roundtrip() -> None:
    r = ChannelSendResult.success("c", provider_receipt="r", raw={"k": 1})
    d = r.to_dict()
    assert d["ok"] is True
    assert d["status"] == "success"
    assert d["provider_receipt"] == "r"
    assert d["raw"] == {"k": 1}


def test_channel_health_to_dict() -> None:
    h = ChannelHealth(
        healthy=True,
        channel_id="wechat-main",
        circuit_state="closed",
        consecutive_failures=2,
    )
    d = h.to_dict()
    assert d["healthy"] is True
    assert d["circuit_state"] == "closed"
    assert d["consecutive_failures"] == 2
    assert d["extra"] == {}


def test_validation_result_ok_and_fail() -> None:
    assert ValidationResult.ok_result().ok is True
    fail = ValidationResult.fail(["a", "b"])
    assert fail.ok is False
    assert fail.errors == ["a", "b"]
    assert ValidationResult.fail("one").errors == ["one"]
