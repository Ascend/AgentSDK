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

"""Persisted message helpers for Away Summary."""

from __future__ import annotations

from datetime import datetime
from importlib import import_module
from typing import Any, Protocol, cast


class AwaySummaryMessage(Protocol):
    """Message fields populated by the Away Summary persistence helper."""

    content: Any
    data: Any
    role: str
    timestamp: str


def _system_message_class() -> type[Any]:
    """Load the shared message type only when the dependency is available."""

    try:
        module = import_module("clawcodex_ext.types.messages")
    except ModuleNotFoundError as exc:
        if exc.name not in {
            "clawcodex_ext.types",
            "clawcodex_ext.types.messages",
        }:
            raise
        raise RuntimeError("Away Summary requires clawcodex_ext.types.messages.SystemMessage") from exc
    try:
        return getattr(module, "SystemMessage")
    except AttributeError as exc:
        raise RuntimeError("Away Summary requires clawcodex_ext.types.messages.SystemMessage") from exc


def create_away_summary_message(
    summary: str,
    *,
    trigger: str,
    fingerprint: str,
    message_count: int,
    model: str | None = None,
) -> AwaySummaryMessage:
    text = f"[AWAY SUMMARY]\ntrigger={trigger} fingerprint={fingerprint} model={model or ''}\n\n{summary.strip()}"
    metadata = {
        "trigger": trigger,
        "fingerprint": fingerprint,
        "message_count": message_count,
        "model": model,
        "created_at": datetime.now().isoformat(),
    }
    msg = cast(
        AwaySummaryMessage,
        _system_message_class()(
            content=text,
            timestamp=datetime.now().isoformat(),
            subtype="away_summary",
            level="info",
            isMeta=False,
            data={"away_summary": metadata},
        ),
    )
    metadata["created_at"] = msg.timestamp
    return msg


def format_away_summary_for_display(message_or_text: Any) -> str:
    if isinstance(message_or_text, str):
        text = message_or_text
    else:
        text = str(getattr(message_or_text, "content", "") or "")
    if text.startswith("[AWAY SUMMARY]"):
        parts = text.split("\n\n", 1)
        text = parts[1] if len(parts) > 1 else text
    # Use a blank line after the prefix so Rich Markdown renders
    # "Recapitulate" and the recap body as two distinct paragraphs.
    # A single newline is folded into a space by Markdown, which is
    # why the prefix and body were appearing on the same line.
    return "Recapitulate\n\n" + text.strip()
