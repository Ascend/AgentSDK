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

"""Send results, health, and validation results for the channel contract.

All channels — both the new capability-based adapters and the legacy
webhook channels wrapped via :class:`WebhookChannelAdapter` — must align
on these types so the gateway can classify errors, retry uniformly, and
track provider receipts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class SendStatus(str, Enum):
    """Outcome of a single send attempt (or the terminal attempt)."""

    SUCCESS = "success"
    FAILED = "failed"  # terminal, non-retryable
    RETRYABLE_ERROR = "retryable_error"
    NONRETRYABLE_ERROR = "nonretryable_error"
    UNSUPPORTED = "unsupported"  # capability not declared (fail-closed)
    RATE_LIMITED = "rate_limited"
    ENQUEUED = "enqueued"


class ErrorCategory(str, Enum):
    """Coarse error classification driving retry decisions."""

    NONE = "none"
    NETWORK = "network"
    TIMEOUT = "timeout"
    RATE_LIMIT = "rate_limit"  # 429
    SERVER_ERROR = "server_error"  # 5xx
    CLIENT_ERROR = "client_error"  # 4xx other than below
    AUTH = "auth"  # 401 / 403 / session expired
    NOT_FOUND = "not_found"  # 404
    FORMAT = "format"  # malformed payload / response
    UNSUPPORTED_MEDIA = "unsupported_media"
    CIRCUIT_OPEN = "circuit_open"  # adapter fuse blown
    UNKNOWN = "unknown"


class CircuitState(str, Enum):
    """Adapter connection-fuse state (used by :class:`ChannelHealth`)."""

    CLOSED = "closed"
    OPEN = "circuit_open"


# Categories that may be retried under the default policy.
RETRYABLE_CATEGORIES: frozenset[ErrorCategory] = frozenset(
    {
        ErrorCategory.NETWORK,
        ErrorCategory.TIMEOUT,
        ErrorCategory.RATE_LIMIT,
        ErrorCategory.SERVER_ERROR,
    }
)

# Categories that must not be retried.
NONRETRYABLE_CATEGORIES: frozenset[ErrorCategory] = frozenset(
    {
        ErrorCategory.CLIENT_ERROR,
        ErrorCategory.AUTH,
        ErrorCategory.NOT_FOUND,
        ErrorCategory.FORMAT,
        ErrorCategory.UNSUPPORTED_MEDIA,
        ErrorCategory.CIRCUIT_OPEN,
    }
)


@dataclass
class ChannelSendResult:
    """Uniform send result returned by every ``OutboundCapability.send``."""

    ok: bool
    status: SendStatus
    channel_id: str
    error_category: ErrorCategory = ErrorCategory.NONE
    provider_receipt: str | None = None
    message: str | None = None
    attempts: int = 1
    raw: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.status, SendStatus):
            raise TypeError("ChannelSendResult.status must be a SendStatus")
        if not isinstance(self.error_category, ErrorCategory):
            raise TypeError("ChannelSendResult.error_category must be an ErrorCategory")
        if self.attempts < 1:
            raise ValueError("ChannelSendResult.attempts must be >= 1")

    @property
    def retryable(self) -> bool:
        """Whether a failed result may be retried per its category."""
        if self.ok:
            return False
        if self.status is SendStatus.RATE_LIMITED:
            return True
        return self.error_category in RETRYABLE_CATEGORIES

    @classmethod
    def success(
        cls,
        channel_id: str,
        *,
        provider_receipt: str | None = None,
        attempts: int = 1,
        raw: dict[str, Any] | None = None,
    ) -> ChannelSendResult:
        return cls(
            ok=True,
            status=SendStatus.SUCCESS,
            channel_id=channel_id,
            provider_receipt=provider_receipt,
            attempts=attempts,
            raw=raw,
        )

    @classmethod
    def enqueued(
        cls,
        channel_id: str,
        *,
        message: str | None = None,
        attempts: int = 1,
        raw: dict[str, Any] | None = None,
    ) -> ChannelSendResult:
        return cls(
            ok=True,
            status=SendStatus.ENQUEUED,
            channel_id=channel_id,
            message=message,
            attempts=attempts,
            raw=raw,
        )

    @classmethod
    def retryable_error(
        cls,
        channel_id: str,
        *,
        message: str,
        category: ErrorCategory,
        attempts: int = 1,
        raw: dict[str, Any] | None = None,
    ) -> ChannelSendResult:
        if category not in RETRYABLE_CATEGORIES:
            raise ValueError(f"{category!r} is not a retryable category; use nonretryable_error")
        return cls(
            ok=False,
            status=SendStatus.RETRYABLE_ERROR,
            channel_id=channel_id,
            error_category=category,
            message=message,
            attempts=attempts,
            raw=raw,
        )

    @classmethod
    def rate_limited(
        cls,
        channel_id: str,
        *,
        message: str,
        attempts: int = 1,
        raw: dict[str, Any] | None = None,
    ) -> ChannelSendResult:
        return cls(
            ok=False,
            status=SendStatus.RATE_LIMITED,
            channel_id=channel_id,
            error_category=ErrorCategory.RATE_LIMIT,
            message=message,
            attempts=attempts,
            raw=raw,
        )

    @classmethod
    def nonretryable_error(
        cls,
        channel_id: str,
        *,
        message: str,
        category: ErrorCategory = ErrorCategory.UNKNOWN,
        status: SendStatus = SendStatus.NONRETRYABLE_ERROR,
        attempts: int = 1,
        raw: dict[str, Any] | None = None,
    ) -> ChannelSendResult:
        return cls(
            ok=False,
            status=status,
            channel_id=channel_id,
            error_category=category,
            message=message,
            attempts=attempts,
            raw=raw,
        )

    @classmethod
    def unsupported(
        cls,
        channel_id: str,
        *,
        message: str,
        attempts: int = 1,
    ) -> ChannelSendResult:
        return cls(
            ok=False,
            status=SendStatus.UNSUPPORTED,
            channel_id=channel_id,
            error_category=ErrorCategory.NONE,
            message=message,
            attempts=attempts,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "status": self.status.value,
            "channel_id": self.channel_id,
            "error_category": self.error_category.value,
            "provider_receipt": self.provider_receipt,
            "message": self.message,
            "attempts": self.attempts,
            "raw": dict(self.raw) if self.raw is not None else None,
        }


@dataclass
class ChannelHealth:
    """Health-check result exposed by every channel adapter."""

    healthy: bool
    channel_id: str
    circuit_state: str = "closed"  # closed | circuit_open
    last_error: str | None = None
    last_poll_at: float | None = None
    last_inbound_at: float | None = None
    last_outbound_at: float | None = None
    consecutive_failures: int = 0
    account_status: str | None = None
    queue_depth: int | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "healthy": self.healthy,
            "channel_id": self.channel_id,
            "circuit_state": self.circuit_state,
            "last_error": self.last_error,
            "last_poll_at": self.last_poll_at,
            "last_inbound_at": self.last_inbound_at,
            "last_outbound_at": self.last_outbound_at,
            "consecutive_failures": self.consecutive_failures,
            "account_status": self.account_status,
            "queue_depth": self.queue_depth,
            "extra": dict(self.extra) if self.extra else {},
        }


@dataclass
class ValidationResult:
    """Result of ``ChannelAdapter.validate_config``."""

    ok: bool
    errors: list[str] = field(default_factory=list)

    @classmethod
    def ok_result(cls) -> ValidationResult:
        return cls(ok=True, errors=[])

    @classmethod
    def fail(cls, errors: list[str] | str) -> ValidationResult:
        if isinstance(errors, str):
            errors = [errors]
        return cls(ok=False, errors=list(errors))


__all__ = [
    "ChannelHealth",
    "ChannelSendResult",
    "CircuitState",
    "ErrorCategory",
    "NONRETRYABLE_CATEGORIES",
    "RETRYABLE_CATEGORIES",
    "SendStatus",
    "ValidationResult",
]
