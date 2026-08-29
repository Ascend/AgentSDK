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

"""Shared builders for parser contract tests."""

from __future__ import annotations

import json
from pathlib import Path


def _iso(ts: float) -> str:
    """Render a Unix epoch as an ISO 8601 UTC string, preserving microseconds.

    Stripping microseconds (``.replace(microsecond=0)``) silently rounds
    fractional timestamps like ``1717500001.5`` down to ``1717500001.0``,
    which then skews every downstream duration assertion by up to a
    second. Keep the resolution — the parser accepts sub-second ISO
    timestamps.
    """
    from datetime import datetime, timezone

    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()


def _write_jsonl(path: Path, entries: list[dict]) -> Path:
    """Write one JSON object per line at ``path``."""
    with open(path, "w", encoding="utf-8") as f:
        for e in entries:
            f.write(json.dumps(e) + "\n")
    return path


def _assistant_tool_use(
    ts: float,
    name: str,
    tool_use_id: str,
    input: dict | None = None,
    text: str | None = None,
) -> dict:
    """Build an assistant entry containing a text block (optional) and a
    single ``tool_use`` block.
    """
    content: list[dict] = []
    if text is not None:
        content.append({"type": "text", "text": text})
    content.append(
        {
            "type": "tool_use",
            "name": name,
            "id": tool_use_id,
            "tool_use_id": tool_use_id,
            "input": input or {},
        }
    )
    return {
        "role": "assistant",
        "type": "message",
        "timestamp": _iso(ts),
        "isMeta": False,
        "isVirtual": False,
        "isCompactSummary": False,
        "content": content,
    }


def _tool_result_entry(
    ts: float,
    tool_use_id: str,
    text: str,
    *,
    is_error: bool = False,
) -> dict:
    """Build a user-role entry that carries a single ``tool_result`` block.

    The new envelope embeds tool results inside user messages — the
    legacy top-level ``role: tool`` envelope is gone.
    """
    block: dict = {
        "type": "tool_result",
        "tool_use_id": tool_use_id,
        "content": [{"type": "text", "text": text}],
    }
    if is_error:
        block["is_error"] = True
    return {
        "role": "user",
        "type": "message",
        "timestamp": _iso(ts),
        "isMeta": False,
        "isVirtual": False,
        "isCompactSummary": False,
        "toolUseID": tool_use_id,
        "content": [block],
    }


def _user_text(ts: float, text: str) -> dict:
    return {
        "role": "user",
        "type": "message",
        "timestamp": _iso(ts),
        "isMeta": False,
        "isVirtual": False,
        "isCompactSummary": False,
        "content": [{"type": "text", "text": text}],
    }


def _assistant_text(
    ts: float,
    text: str,
    *,
    model: str | None = None,
) -> dict:
    entry: dict = {
        "role": "assistant",
        "type": "message",
        "timestamp": _iso(ts),
        "isMeta": False,
        "isVirtual": False,
        "isCompactSummary": False,
        "content": [{"type": "text", "text": text}],
    }
    if model is not None:
        entry["model"] = model
    return entry


def _assistant_entry(ts: float, text: str, **overrides) -> dict:
    """Build an assistant text entry with explicit envelope overrides."""
    entry = _assistant_text(ts, text)
    entry.update(overrides)
    return entry
