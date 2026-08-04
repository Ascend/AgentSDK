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

"""Session-level memory helpers for Away Summary.

Mirrors the optional broader-context injection used by the upstream
``src/services/awaySummary.ts`` (which calls ``getSessionMemoryContent()``).
The clawcodex equivalent reads the session-summary sidecar emitted by
:mod:`clawcodex_ext.session_intelligence` and projects it into a short,
human-readable block the recap prompt can prepend to its instructions.

The default behaviour is *off* — only flip on via
``AwaySummaryConfig.include_session_memory = True`` — so the auto recap
remains byte-identical to its prior behaviour unless the user opts in.
"""

from __future__ import annotations

from importlib import import_module
from pathlib import Path
from typing import Any, Callable

# Test seam and compatibility override. The real optional dependency is
# resolved lazily so importing Away Summary never requires session intelligence.
load_summary: Callable[..., Any] | None = None

MEMORY_SECTIONS = (
    ("Goals", "goals"),
    ("Completed", "completed"),
    ("Open threads", "open_threads"),
    ("Next candidates", "next_action_candidates"),
    ("User preferences", "user_preferences"),
)


def _resolve_load_summary() -> Callable[..., Any] | None:
    if load_summary is not None:
        return load_summary
    try:
        module = import_module("clawcodex_ext.session_intelligence.index")
    except ModuleNotFoundError as exc:
        if str(exc.name or "") in {
            "clawcodex_ext.session_intelligence",
            "clawcodex_ext.session_intelligence.index",
        }:
            return None
        raise
    return getattr(module, "load_summary", None)


def get_session_memory_content(
    *,
    session_id: str | None = None,
    max_chars: int = 4000,
    sessions_dir: Path | None = None,
) -> str | None:
    """Return a short session-memory block for prompt injection, or None.

    Reads :func:`clawcodex_ext.session_intelligence.index.load_summary`
    which returns the ``summary.json`` sidecar produced by the lazy
    sidecar pipeline. The function is intentionally tolerant: any IO /
    parse / missing-file error yields ``None`` so the recap never crashes
    because the sidecar is unavailable.
    """
    if not session_id:
        return None
    loader = _resolve_load_summary()
    if loader is None:
        return None
    try:
        data = loader(session_id, sessions_dir=sessions_dir)
    except Exception:
        return None
    if not data:
        return None
    return _format_memory(data, max_chars=max_chars)


def _format_memory(data: dict[str, Any], *, max_chars: int) -> str | None:
    """Project a session summary dict into a recap-friendly block."""
    lines: list[str] = []

    title = str(data.get("title") or "").strip()
    if title:
        lines.append(f"Title: {title}")

    cwd = str(data.get("cwd") or "").strip()
    if cwd:
        lines.append(f"Working directory: {cwd}")

    for label, field in MEMORY_SECTIONS:
        items = list(data.get(field) or [])
        clean = [str(x).strip() for x in items if x is not None and str(x).strip()]
        if not clean:
            continue
        lines.append(f"{label}:")
        # Tail-truncate each list so the recap stays short even when the
        # sidecar accumulated many entries across long sessions.
        for item in clean[-5:]:
            lines.append(f"- {item}")

    text = "\n".join(lines).strip()
    if not text or max_chars <= 0:
        return None
    if len(text) > max_chars:
        return text[: max_chars - 1].rstrip() + "…"
    return text
