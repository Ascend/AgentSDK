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

"""P88-D — disk-backed Explore/Plan report store.

The Agent tool's one-shot agents (Explore, Plan) emit a high-leverage
markdown report. Without this store, that report lives only in the
in-memory transcript and is lost when the session ends.

This module writes each report to disk in two sibling formats:

  * ``<name>.md``   — human-readable, identical to the agent's
                      output (with a small YAML-ish header added).
  * ``<name>.json`` — machine-parseable, derived from the same
                      structured fields (``title``, ``summary``,
                      ``findings``/``steps``, ``critical_files``).

The two files share a common stem so a consumer can pick the format
it needs by extension.

File layout
-----------
``~/.clawcodex/reports/<kind>/<session_id>/<agent_id>.{md,json}``
where ``kind`` is ``"explore"`` or ``"plan"``. The
``CLAWCODEX_HOME`` environment variable overrides the parent dir
when set (matching the project's existing convention).

Thread safety
-------------
A single :class:`threading.RLock` guards the dir-creation and
write. Multiple agents in the same process can save concurrently
without trampling each other; cross-process safety is not a goal
(the agent tool runs in a single process per session).
"""

from __future__ import annotations

import json
import os
import re
import tempfile
import threading
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


# --- Public dataclasses ---


@dataclass(frozen=True)
class ExploreReport:
    """Structured form of an Explore agent's final report.

    ``findings`` is a tuple (frozen dataclass requires hashable
    members); each entry is one bullet / line of the agent's
    findings. ``critical_files`` is parsed from the optional
    ``### Critical Files for Implementation`` trailing section.
    """

    agent_id: str
    session_id: str
    title: str
    summary: str
    findings: tuple[str, ...] = field(default_factory=tuple)
    critical_files: tuple[str, ...] = field(default_factory=tuple)
    raw_markdown: str = ""
    created_at: str = ""  # ISO-8601 UTC, e.g. "2026-06-21T10:31:00Z"


@dataclass(frozen=True)
class PlanDocument:
    """Structured form of a Plan agent's final report.

    ``steps`` is the ordered list of implementation steps (number
    prefix stripped, e.g. ``"1. ..."`` becomes ``"..."``).
    ``critical_files`` is the same field as ExploreReport.
    """

    agent_id: str
    session_id: str
    title: str
    summary: str
    steps: tuple[str, ...] = field(default_factory=tuple)
    critical_files: tuple[str, ...] = field(default_factory=tuple)
    raw_markdown: str = ""
    created_at: str = ""


# --- Path resolution ---


def _resolve_base_dir(base_dir: Path | None) -> Path:
    """Return the reports root. Honors ``CLAWCODEX_HOME`` when
    ``base_dir`` is not supplied.
    """
    if base_dir is not None:
        return base_dir
    home_override = os.environ.get("CLAWCODEX_HOME")
    if home_override:
        return Path(home_override) / "reports"
    return Path.home() / ".clawcodex" / "reports"


def _safe_segment(value: str) -> str:
    """Make ``value`` filesystem-safe (alnum / ``._-``)."""
    cleaned = "".join(ch if ch.isalnum() or ch in "._-" else "-" for ch in value).strip("-._")
    return cleaned or "unknown"


# --- Atomic write primitive ---


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write ``payload`` as JSON to ``path`` atomically.

    The temp file lives in ``path.parent`` so the final
    ``os.replace`` is a single-directory rename (POSIX atomic;
    Windows atomic when the destination is on the same volume).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False, indent=2, sort_keys=True)
            fh.write("\n")
        os.replace(tmp_path, path)
    except Exception:  # nosec
        try:
            os.unlink(tmp_path)
        except OSError:
            pass  # Best-effort operation failed; keep the surrounding fallback.
        raise


def _atomic_write_text(path: Path, text: str) -> None:
    """Write ``text`` to ``path`` atomically. UTF-8."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(text)
        os.replace(tmp_path, path)
    except Exception:  # nosec
        try:
            os.unlink(tmp_path)
        except OSError:
            pass  # Best-effort operation failed; keep the surrounding fallback.
        raise


# --- Markdown renderers ---


def _render_explore_markdown(report: ExploreReport) -> str:
    lines: list[str] = [
        f"# {report.title or 'Explore Report'}",
        "",
        f"- Agent: `{report.agent_id}`",
        f"- Session: `{report.session_id}`",
        f"- Created: `{report.created_at or '(unknown)'}`",
        "",
    ]
    if report.summary:
        lines.extend(["## Summary", "", report.summary, ""])
    if report.findings:
        lines.append("## Findings")
        lines.append("")
        for finding in report.findings:
            lines.append(f"- {finding}")
        lines.append("")
    if report.critical_files:
        lines.append("## Critical Files")
        lines.append("")
        for path in report.critical_files:
            lines.append(f"- `{path}`")
        lines.append("")
    if report.raw_markdown:
        lines.extend(["## Full Report", "", report.raw_markdown, ""])
    return "\n".join(lines).rstrip() + "\n"


def _render_plan_markdown(plan: PlanDocument) -> str:
    lines: list[str] = [
        f"# {plan.title or 'Implementation Plan'}",
        "",
        f"- Agent: `{plan.agent_id}`",
        f"- Session: `{plan.session_id}`",
        f"- Created: `{plan.created_at or '(unknown)'}`",
        "",
    ]
    if plan.summary:
        lines.extend(["## Summary", "", plan.summary, ""])
    if plan.steps:
        lines.append("## Steps")
        lines.append("")
        for idx, step in enumerate(plan.steps, start=1):
            lines.append(f"{idx}. {step}")
        lines.append("")
    if plan.critical_files:
        lines.append("## Critical Files for Implementation")
        lines.append("")
        for path in plan.critical_files:
            lines.append(f"- `{path}`")
        lines.append("")
    if plan.raw_markdown:
        lines.extend(["## Full Report", "", plan.raw_markdown, ""])
    return "\n".join(lines).rstrip() + "\n"


# --- Critical-files parser ---


_CRITICAL_HEADING_RE = re.compile(
    r"^#{1,6}\s*Critical Files[^\n]*\n+(?P<body>.*?)(?=\n#{1,6}\s|\Z)",
    re.MULTILINE | re.IGNORECASE | re.DOTALL,
)
# Use `[ \t]+` (NOT `\s+`) for the post-marker whitespace: `\s` would
# also match the newline between consecutive bullets, causing the
# next bullet's `-` to be captured as the start of `text` (e.g.
# `- a.py\n- \n- b.py` would yield ``('a.py', '- b.py')``).
_BULLET_RE = re.compile(
    r"^[ \t]*[-*][ \t]+(?P<text>\S.*)$",
    re.MULTILINE,
)


def parse_critical_files(markdown: str) -> tuple[str, ...]:
    """Pull paths from the optional ``### Critical Files ...`` section.

    Returns an empty tuple if the section is missing. The parser is
    deliberately conservative — a path is any non-empty line that
    looks like a markdown bullet. We do not validate that the path
    exists; the agent's word is the contract.
    """
    if not markdown:
        return ()
    match = _CRITICAL_HEADING_RE.search(markdown)
    if not match:
        return ()
    body = match.group("body")
    files: list[str] = []
    for line_match in _BULLET_RE.finditer(body):
        text = line_match.group("text").strip()
        # Strip inline backticks / link syntax that LLMs sometimes add.
        text = text.strip("`")
        if text:
            files.append(text)
    return tuple(files)


# --- Timestamp helper ---


def now_iso_utc() -> str:
    """Current UTC time as ISO-8601 with a ``Z`` suffix."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# --- Public store class ---


class ReportStore:
    """Thread-safe writer for Explore and Plan reports.

    Construct with ``base_dir=None`` to default to
    ``~/.clawcodex/reports`` (or ``$CLAWCODEX_HOME/reports``).
    Construct with an explicit path in tests.
    """

    def __init__(self, base_dir: Path | None = None) -> None:
        self._base_dir = _resolve_base_dir(base_dir)
        self._lock = threading.RLock()

    @property
    def base_dir(self) -> Path:
        return self._base_dir

    def _session_dir(self, kind: str, session_id: str) -> Path:
        return self._base_dir / _safe_segment(kind) / _safe_segment(session_id)

    def save_explore(self, report: ExploreReport) -> Path:
        """Persist an Explore report. Returns the ``.md`` path."""
        with self._lock:
            session_dir = self._session_dir("explore", report.session_id)
            stem = _safe_segment(report.agent_id) or "agent"
            md_path = session_dir / f"{stem}.md"
            json_path = md_path.with_suffix(".json")
            markdown = _render_explore_markdown(report)
            payload = asdict(report)
            payload["findings"] = list(report.findings)
            payload["critical_files"] = list(report.critical_files)
            payload["kind"] = "explore"
            _atomic_write_text(md_path, markdown)
            _atomic_write_json(json_path, payload)
            return md_path

    def save_plan(self, plan: PlanDocument) -> Path:
        """Persist a Plan report. Returns the ``.md`` path."""
        with self._lock:
            session_dir = self._session_dir("plan", plan.session_id)
            stem = _safe_segment(plan.agent_id) or "agent"
            md_path = session_dir / f"{stem}.md"
            json_path = md_path.with_suffix(".json")
            markdown = _render_plan_markdown(plan)
            payload = asdict(plan)
            payload["steps"] = list(plan.steps)
            payload["critical_files"] = list(plan.critical_files)
            payload["kind"] = "plan"
            _atomic_write_text(md_path, markdown)
            _atomic_write_json(json_path, payload)
            return md_path


__all__ = [
    "ExploreReport",
    "PlanDocument",
    "ReportStore",
    "parse_critical_files",
    "now_iso_utc",
]
