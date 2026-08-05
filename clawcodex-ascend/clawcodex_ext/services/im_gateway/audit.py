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

"""Audit redaction helpers.

PII / secret fields are masked before writing to ``audit.ndjson`` so the
audit log is safe to share. Sensitive keys: ``bot_token``,
``context_token``, ``Authorization``, ``webhook_url`` (token segment),
``from_user_id`` (hashed), ``user_id`` (hashed), ``bot_token_enc``.
"""

from __future__ import annotations

# pylint: disable=no-name-in-module  # Split migration branches provide channel transport.

import hashlib
from collections.abc import Mapping, Sequence
from typing import Any

from clawcodex_ext.services.channels.transport import redact_webhook_url

_SENSITIVE_EXACT = frozenset(
    {"bot_token", "bot_token_enc", "context_token", "authorization", "secret", "password", "token"}
)
_HASH_KEYS = frozenset({"from_user_id", "user_id", "to_user_id"})
_REDACTED_URL_KEYS = frozenset({"webhook_url"})


def hash_user(user_id: str) -> str:
    if not user_id:
        return ""
    return hashlib.sha256(str(user_id).encode("utf-8")).hexdigest()[:16]


def redact(fields: dict[str, Any]) -> dict[str, Any]:
    """Return a copy of ``fields`` with sensitive values masked/hashed."""
    out: dict[str, Any] = {}
    for k, v in fields.items():
        lk = k.lower()
        if lk in _SENSITIVE_EXACT:
            out[k] = "***" if v else v
        elif lk in _HASH_KEYS:
            out[k] = hash_user(str(v)) if v else v
        elif lk in _REDACTED_URL_KEYS:
            out[k] = redact_webhook_url(str(v)) if v else v
        elif isinstance(v, Mapping):
            out[k] = redact(v)
        elif isinstance(v, Sequence) and not isinstance(v, (str, bytes, bytearray)):
            out[k] = [_redact_sequence_item(item) for item in v]
        else:
            out[k] = v
    return out


def _redact_sequence_item(value: Any) -> Any:
    if isinstance(value, Mapping):
        return redact(value)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_redact_sequence_item(item) for item in value]
    return value


__all__ = ["hash_user", "redact"]
