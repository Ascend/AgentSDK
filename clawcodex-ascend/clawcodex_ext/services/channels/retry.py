#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# -------------------------------------------------------------------------
# This file is part of the AgentSDK project.
#
# Originally from Clawd Codex:
# https://github.com/agentforce314/clawcodex
# Copyright (c) 2026 Clawd Codex Team
# Licensed under the MIT License. See clawcodex-ascend/LICENSES/Clawd-Codex-MIT.txt.
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

"""Default retry policy and error classification for channels.

The gateway's :class:`OutboundDispatcher` and each adapter share this
classification so retry/no-retry decisions are consistent across
Feishu/Slack/Discord/WeChat. Classification is HTTP-status and
exception based; adapters may override the category on a per-result
basis when a platform returns a domain-specific signal (e.g. WeChat
401 → session expired → ``AUTH``, which pauses the account rather than
consuming the circuit-breaker budget).
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

from .results import ErrorCategory
from .transport import TransportError


@dataclass(frozen=True)
class RetryPolicy:
    """Per-channel retry policy.

    ``max_attempts`` is the total attempt count (1 == no retry).
    Backoff is exponential with optional jitter, capped at
    ``max_seconds``.
    """

    max_attempts: int = 5
    base_seconds: float = 2.0
    max_seconds: float = 60.0
    jitter: float = 0.25
    retryable_categories: frozenset[ErrorCategory] = frozenset(
        {
            ErrorCategory.NETWORK,
            ErrorCategory.TIMEOUT,
            ErrorCategory.RATE_LIMIT,
            ErrorCategory.SERVER_ERROR,
        }
    )

    def __post_init__(self) -> None:
        if self.max_attempts < 1:
            raise ValueError("RetryPolicy.max_attempts must be >= 1")
        if self.base_seconds <= 0:
            raise ValueError("RetryPolicy.base_seconds must be > 0")
        if self.max_seconds < self.base_seconds:
            raise ValueError("RetryPolicy.max_seconds must be >= base_seconds")
        if not (0.0 <= self.jitter <= 1.0):
            raise ValueError("RetryPolicy.jitter must be in [0.0, 1.0]")

    def is_retryable(self, category: ErrorCategory) -> bool:
        return category in self.retryable_categories


DEFAULT_RETRY_POLICY = RetryPolicy()


def classify_http_status(status: int) -> ErrorCategory:
    """Map an HTTP status code to an :class:`ErrorCategory`."""
    if 200 <= status < 300:
        return ErrorCategory.NONE
    if status == 429:
        return ErrorCategory.RATE_LIMIT
    if status in (401, 403):
        return ErrorCategory.AUTH
    if status == 404:
        return ErrorCategory.NOT_FOUND
    if 400 <= status < 500:
        return ErrorCategory.CLIENT_ERROR
    if 500 <= status < 600:
        return ErrorCategory.SERVER_ERROR
    return ErrorCategory.UNKNOWN


def classify_exception(exc: BaseException) -> ErrorCategory:
    """Map a raised exception to an :class:`ErrorCategory`."""
    if isinstance(exc, TimeoutError):
        return ErrorCategory.TIMEOUT
    if isinstance(exc, TransportError):
        msg = str(exc).lower()
        if "timeout" in msg:
            return ErrorCategory.TIMEOUT
        return ErrorCategory.NETWORK
    if isinstance(exc, (ConnectionError, OSError)):
        return ErrorCategory.NETWORK
    if isinstance(exc, ValueError):
        return ErrorCategory.FORMAT
    return ErrorCategory.UNKNOWN


def compute_backoff(
    attempt: int,
    policy: RetryPolicy,
    *,
    rng: Any = None,
) -> float:
    """Return the backoff seconds before retrying ``attempt`` (1-based).

    Exponential: ``base * 2**(attempt-1)``, capped at ``max_seconds``,
    with symmetric jitter of ``+/- jitter * value``.
    """
    if attempt < 1:
        raise ValueError("attempt must be >= 1")
    rng = rng if rng is not None else random
    base = min(policy.base_seconds * (2 ** (attempt - 1)), policy.max_seconds)
    if policy.jitter <= 0:
        return base
    delta = base * policy.jitter
    return base + rng.uniform(-delta, delta)


__all__ = [
    "DEFAULT_RETRY_POLICY",
    "RetryPolicy",
    "classify_exception",
    "classify_http_status",
    "compute_backoff",
]
