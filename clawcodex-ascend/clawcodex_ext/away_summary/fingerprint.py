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

"""Conversation fingerprint helpers for Away Summary."""

from __future__ import annotations

import hashlib
import json
from typing import Any


def conversation_fingerprint(conversation: Any) -> str:
    """Return a stable hash for user/assistant-visible conversation content."""

    messages = list(getattr(conversation, "messages", []) or [])
    payload: list[dict[str, Any]] = []
    for msg in messages:
        if is_away_summary_message(msg):
            continue
        if getattr(msg, "isVirtual", False):
            continue
        role = getattr(msg, "role", "")
        if role not in {"user", "assistant"}:
            continue
        payload.append(
            {
                "role": role,
                "content": _jsonable(getattr(msg, "content", "")),
                "uuid": getattr(msg, "uuid", None),
            }
        )
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=True, default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def session_turn_count(conversation: Any) -> int:
    """Count completed user/assistant pairs."""

    turns = 0
    pending_user = False
    for msg in getattr(conversation, "messages", []) or []:
        if is_away_summary_message(msg):
            continue
        role = getattr(msg, "role", "")
        if role == "user":
            pending_user = True
        elif role == "assistant" and pending_user:
            turns += 1
            pending_user = False
    return turns


def last_away_summary_fingerprint(
    conversation: Any,
    *,
    trigger: str | None = None,
) -> str | None:
    for msg in reversed(list(getattr(conversation, "messages", []) or [])):
        if not is_away_summary_message(msg):
            continue
        data = getattr(msg, "data", None)
        meta = data.get("away_summary") if isinstance(data, dict) else None
        if not isinstance(meta, dict):
            # Backward compatibility for messages persisted before the
            # structured ``SystemMessage.data`` field was adopted.
            meta = getattr(msg, "_away_summary_meta", None)
        if isinstance(meta, dict):
            if trigger is not None and meta.get("trigger") != trigger:
                continue
            value = meta.get("fingerprint")
            return str(value) if value else None
        content = _flatten_content(getattr(msg, "content", ""))
        if trigger is not None:
            marker = "trigger="
            if marker not in content:
                continue
            found = content.split(marker, 1)[1].split()[0].strip()
            if found != trigger:
                continue
        marker = "fingerprint="
        if marker in content:
            return content.split(marker, 1)[1].split()[0].strip()
    return None


def is_away_summary_message(msg: Any) -> bool:
    if getattr(msg, "role", "") != "system":
        return False
    if getattr(msg, "subtype", None) == "away_summary":
        return True
    return "[AWAY SUMMARY]" in _flatten_content(getattr(msg, "content", ""))


def _jsonable(value: Any, seen: set[int] | None = None) -> Any:
    if not isinstance(value, (list, dict)) and not hasattr(value, "__dict__"):
        return value
    seen = set() if seen is None else seen
    marker = id(value)
    if marker in seen:
        return "<recursive>"
    seen.add(marker)
    try:
        if isinstance(value, list):
            return [_jsonable(item, seen) for item in value]
        if isinstance(value, dict):
            return {str(key): _jsonable(item, seen) for key, item in value.items()}
        return _jsonable(value.__dict__, seen)
    finally:
        seen.remove(marker)


def _flatten_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            text = getattr(item, "text", None)
            if text is not None:
                parts.append(str(text))
            elif isinstance(item, dict):
                parts.append(str(item.get("text") or item.get("content") or ""))
            else:
                parts.append(str(item))
        return "\n".join(p for p in parts if p)
    return str(content)
