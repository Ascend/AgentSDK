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

"""Tests for retry policy and error classification."""

from __future__ import annotations

# Sibling clawcodex_ext packages (channels source) are migrated in separate
# branches; suppress E0611 until the full series lands.
# pylint: disable=no-name-in-module

import pytest

from clawcodex_ext.services.channels.results import ErrorCategory
from clawcodex_ext.services.channels.retry import (
    DEFAULT_RETRY_POLICY,
    RetryPolicy,
    classify_exception,
    classify_http_status,
    compute_backoff,
)
from clawcodex_ext.services.channels.transport import TransportError


def test_classify_http_status() -> None:
    assert classify_http_status(200) is ErrorCategory.NONE
    assert classify_http_status(204) is ErrorCategory.NONE
    assert classify_http_status(429) is ErrorCategory.RATE_LIMIT
    assert classify_http_status(401) is ErrorCategory.AUTH
    assert classify_http_status(403) is ErrorCategory.AUTH
    assert classify_http_status(404) is ErrorCategory.NOT_FOUND
    assert classify_http_status(400) is ErrorCategory.CLIENT_ERROR
    assert classify_http_status(422) is ErrorCategory.CLIENT_ERROR
    assert classify_http_status(500) is ErrorCategory.SERVER_ERROR
    assert classify_http_status(503) is ErrorCategory.SERVER_ERROR
    assert classify_http_status(700) is ErrorCategory.UNKNOWN


def test_classify_exception() -> None:
    assert classify_exception(TimeoutError()) is ErrorCategory.TIMEOUT
    assert classify_exception(TransportError("network down")) is ErrorCategory.NETWORK
    assert classify_exception(TransportError("transport timeout: x")) is ErrorCategory.TIMEOUT
    assert classify_exception(ConnectionError("x")) is ErrorCategory.NETWORK
    assert classify_exception(ValueError("bad")) is ErrorCategory.FORMAT
    assert classify_exception(RuntimeError("??")) is ErrorCategory.UNKNOWN


def test_retry_policy_validation() -> None:
    with pytest.raises(ValueError):
        RetryPolicy(max_attempts=0)
    with pytest.raises(ValueError):
        RetryPolicy(base_seconds=0)
    with pytest.raises(ValueError):
        RetryPolicy(base_seconds=10, max_seconds=5)
    with pytest.raises(ValueError):
        RetryPolicy(jitter=1.5)


def test_retry_policy_is_retryable() -> None:
    p = DEFAULT_RETRY_POLICY
    assert p.is_retryable(ErrorCategory.SERVER_ERROR)
    assert p.is_retryable(ErrorCategory.RATE_LIMIT)
    assert not p.is_retryable(ErrorCategory.AUTH)
    assert not p.is_retryable(ErrorCategory.FORMAT)


def test_compute_backoff_monotonic_and_capped_without_jitter() -> None:
    policy = RetryPolicy(max_attempts=6, base_seconds=1.0, max_seconds=10.0, jitter=0.0)
    vals = [compute_backoff(a, policy) for a in range(1, 7)]
    # exponential: 1, 2, 4, 8, 10 (capped), 10 (capped)
    assert vals[0] == pytest.approx(1.0)
    assert vals[1] == pytest.approx(2.0)
    assert vals[2] == pytest.approx(4.0)
    assert vals[3] == pytest.approx(8.0)
    assert vals[4] == pytest.approx(10.0)
    assert vals[5] == pytest.approx(10.0)


def test_compute_backoff_jitter_within_bounds() -> None:
    class _Rng:
        def uniform(self, a: float, b: float) -> float:
            return (a + b) / 2  # midpoint

    policy = RetryPolicy(base_seconds=2.0, max_seconds=60.0, jitter=0.25)
    # attempt 1: base=2, delta=0.5, midpoint => 2.0
    assert compute_backoff(1, policy, rng=_Rng()) == pytest.approx(2.0)
    # attempt 3: base=8, delta=2, midpoint => 8.0
    assert compute_backoff(3, policy, rng=_Rng()) == pytest.approx(8.0)


def test_compute_backoff_rejects_bad_attempt() -> None:
    with pytest.raises(ValueError):
        compute_backoff(0, DEFAULT_RETRY_POLICY)
