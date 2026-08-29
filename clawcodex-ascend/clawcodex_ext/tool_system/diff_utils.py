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

from __future__ import annotations

import re
from typing import Iterable


_HUNK_RE = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")

# One line terminator at end-of-line (difflib is fed splitlines(keepends=True),
# so hunk lines arrive as a mix of "\n" / "\r\n" / bare-final-line).
_LINE_TERM_RE = re.compile(r"(?:\r\n|\r|\n)$")

_LEADING_TABS_RE = re.compile(r"^\t+", re.MULTILINE)


def convert_leading_tabs_to_spaces(content: str) -> str:
    """Leading tabs → 2 spaces each, for display patches only.

    Mirrors the TS ``convertLeadingTabsToSpaces`` (utils/diff.ts) that the
    original applies to both sides before computing ``structuredPatch`` — a
    raw ``\\t`` reaching the TUI renderer has terminal-dependent width and
    breaks the diff gutter/padding math.
    """
    if "\t" not in content:
        return content
    return _LEADING_TABS_RE.sub(lambda m: "  " * len(m.group(0)), content)


def unified_diff_hunks(diff_lines: Iterable[str]) -> list[dict]:
    """Parse ``difflib.unified_diff`` output into jsdiff StructuredPatchHunk dicts.

    Hunk lines keep their ``+``/``-``/`` `` marker but are stripped of the
    single trailing line terminator so the shape matches jsdiff's
    ``structuredPatch`` (whose lines carry no terminators) — consumers concat
    them with ``\\n`` and index into them for word-diff ranges.
    """
    hunks: list[dict] = []
    current: dict | None = None
    for line in diff_lines:
        m = _HUNK_RE.match(line)
        if m:
            if current is not None:
                hunks.append(current)
            old_start = int(m.group(1))
            old_lines = int(m.group(2) or "1")
            new_start = int(m.group(3))
            new_lines = int(m.group(4) or "1")
            current = {
                "oldStart": old_start,
                "oldLines": old_lines,
                "newStart": new_start,
                "newLines": new_lines,
                "lines": [],
            }
            continue
        if current is None:
            # difflib's ---/+++ file headers precede the first @@, so this
            # guard is what skips them. Inside a hunk every line starts with
            # +/-/space; a removed "-- sql comment" emits "--- sql comment"
            # and MUST be kept (an explicit header skip here used to eat it).
            continue
        current["lines"].append(_LINE_TERM_RE.sub("", line))
    if current is not None:
        hunks.append(current)
    return hunks


def record_patch_line_totals(hunks: list[dict], new_file_content: str | None = None) -> None:
    """Accumulate this patch's +/- line counts into the session totals.

    Mirrors TS ``utils/diff.ts:50-69``: counts hunk lines by their marker,
    with the new-file special case (empty patch + content → every line is
    an addition). Feeds /cost's "Total code changes". Best-effort — an
    accounting failure must never fail the edit itself.
    """
    try:
        if not hunks and new_file_content:
            # TS: newFileContent.split(/\r?\n/).length — trailing newline
            # yields a final empty segment that IS counted; keep that.
            added = len(re.split(r"\r?\n", new_file_content))
            removed = 0
        else:
            added = sum(1 for h in hunks for ln in h.get("lines", []) if ln.startswith("+"))
            removed = sum(1 for h in hunks for ln in h.get("lines", []) if ln.startswith("-"))
        if added or removed:
            from src.bootstrap.state import add_to_total_lines_changed

            add_to_total_lines_changed(added, removed)
    except Exception:  # noqa: BLE001  # nosec — cost accounting is best-effort
        pass  # Intentional best-effort path; the surrounding fallback remains valid.
