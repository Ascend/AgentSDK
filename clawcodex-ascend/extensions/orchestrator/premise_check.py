#!/usr/bin/env python3
# coding=utf-8

# -------------------------------------------------------------------------
# This file is part of the AgentSDK project.
#
# Originally from the clawcodex project:
#   https://github.com/agentforce314/clawcodex
#   Copyright (c) 2026 Clawd Codex Team
#   Licensed under the MIT License. See LICENSE-MIT-clawcodex in this directory.
#
# Portions copyright (c) 2026 Huawei Technologies Co.,Ltd.
# Licensed under Mulan PSL v2. You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
#
# This file is redistributed as a verbatim copy of the upstream source
# (minor whitespace / quoting normalization only); the original copyright
# notice and license terms above apply to the corresponding portions of
# this file. Local additions, if any, are licensed under Mulan PSL v2
# by Huawei Technologies Co.,Ltd.
# -------------------------------------------------------------------------

"""Issue premise validation + honest-exit channel.

Two closed loops that were missing from the fix pipeline:

1. **Premise check** — issues frequently claim "file X crashes at line
   N"; when X does not exist in the repository the claim is unverifiable
   and the correct behavior is to say so, not to invent X. This module
   extracts path-like references from the issue text and reports which
   ones are absent from the workspace so the prompt can warn the agent
   *before* it starts working.

2. **Honest exit** — the only terminal outcomes an agent used to have
   were "produce changes" or "burn retries into abandoned". When the
   premise is false the agent needs a first-class way to stop: it writes
   ``CANNOT_PROCEED_MARKER`` (inside ``.orchestrator_control/``, which is
   already excluded from commits by the git-sync artifact guards) and the
   orchestrator turns that into a failed-with-reason issue plus a tracker
   comment instead of a merge request.

A missing path is deliberately **not** a hard block: an issue may
legitimately ask for a new file to be created. The premise warning only
changes what the agent is told, never whether it runs.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Relative to the workspace root. Lives under .orchestrator_control so
# both the .git/info/exclude patterns and _unstage_orchestrator_artifacts
# already keep it out of commits.
CANNOT_PROCEED_MARKER = ".orchestrator_control/cannot_proceed.json"

# Extensions that make a bare token (no slash) count as a file reference.
_SOURCE_EXTENSIONS = (
    "py",
    "pyi",
    "js",
    "jsx",
    "ts",
    "tsx",
    "mjs",
    "cjs",
    "go",
    "rs",
    "java",
    "kt",
    "rb",
    "php",
    "c",
    "h",
    "cc",
    "cpp",
    "hpp",
    "cs",
    "swift",
    "scala",
    "sh",
    "bash",
    "ps1",
    "sql",
    "html",
    "css",
    "scss",
    "vue",
    "svelte",
    "yaml",
    "yml",
    "toml",
    "ini",
    "cfg",
    "json",
    "md",
)

# Optional directory prefix + a filename ending in a known source
# extension. Requiring the extension keeps prose like "and/or" or
# "TCP/IP" from being treated as repository paths.
_PATH_TOKEN_RE = re.compile(r"(?:[A-Za-z0-9_.\-]+[/\\])*[A-Za-z0-9_.\-]+\.(?:%s)\b" % "|".join(_SOURCE_EXTENSIONS))

_MAX_REFERENCES = 20
# cannot_proceed 详情截断上限；检查清单展示列表上限。
_DETAILS_MAX_CHARS = 2000
_MAX_CHECKED_ITEMS = 10


def extract_referenced_paths(text: str | None) -> list[str]:
    """Pull plausible repository paths out of free-form issue text.

    Returns de-duplicated, forward-slash-normalized candidates in
    first-seen order, capped at ``_MAX_REFERENCES``. URLs, wildcard
    patterns and version-like tokens (``3.11.5``) are skipped.
    """
    if not text:
        return []
    seen: set[str] = set()
    results: list[str] = []
    for raw_line in text.splitlines():
        # Drop URLs before tokenizing so domain/path fragments never match.
        line = re.sub(r"\w+://\S+", " ", raw_line)
        for match in _PATH_TOKEN_RE.finditer(line):
            token = match.group(0).replace("\\", "/").strip(".,;:")
            if not token or "*" in token or "?" in token:
                continue
            # Version numbers ("3.11.5") and dotted identifiers whose
            # "extension" segment is numeric are not paths.
            if re.fullmatch(r"[\d.]+", token):
                continue
            # Dotted module refs like click.utils only count when they
            # end in a known source extension (the regex guarantees the
            # no-slash branch does; slash branch needs no extension).
            if token in seen:
                continue
            seen.add(token)
            results.append(token)
            if len(results) >= _MAX_REFERENCES:
                return results
    return results


def find_missing_paths(workspace_root: Path | str, paths: list[str]) -> list[str]:
    """Return the subset of ``paths`` that do not exist under the root.

    Tokens that escape the workspace (``..``) or are absolute are
    skipped rather than reported — we can only vouch for repository
    contents.

    A candidate whose literal path misses is only reported when its
    basename cannot be found anywhere in the tree either: issues
    routinely cite files by bare name (``base_profiling_parser.py``)
    or through a stale directory, and both are resolvable premises the
    agent can locate itself. Reporting them as absent injects a false
    "this file does not exist" warning that stalls obedient models.
    Only a basename with zero hits — a plausibly fabricated file —
    stays in the missing list.
    """
    root = Path(workspace_root)
    missing: list[str] = []
    for candidate in paths:
        rel = candidate.lstrip("/")
        if rel.startswith("~") or ".." in Path(rel).parts:
            continue
        try:
            exists = (root / rel).exists()
        except OSError:
            continue
        if not exists:
            exists = _basename_exists(root, Path(rel).name)
        if not exists:
            missing.append(candidate)
    return missing


def _basename_exists(root: Path, name: str) -> bool:
    """Return True when any file with this basename exists under
    ``root`` (ignoring ``.git``). Short-circuits on the first hit so
    the scan stays cheap on the common resolvable-reference case.
    """
    if not name:
        return False
    try:
        for hit in root.rglob(name):
            if ".git" in hit.parts:
                continue
            return True
    except OSError:
        pass
    return False


def check_issue_premise(issue: Any, workspace_root: Path | str | None) -> list[str]:
    """Convenience wrapper: extract path references from an issue and
    report the ones missing from the workspace.

    Args:
        issue: the issue object or dict to inspect
        workspace_root: repository root to check against; None skips

    Returns:
        The missing paths, or ``[]`` when nothing is suspicious (or no
        workspace to check against).
    """
    if workspace_root is None:
        return []
    title = getattr(issue, "title", None) or (issue.get("title") if isinstance(issue, dict) else None)
    description = getattr(issue, "description", None) or (issue.get("description") if isinstance(issue, dict) else None)
    text = "\n".join(part for part in (title, description) if part)
    references = extract_referenced_paths(text)
    if not references:
        return []
    return find_missing_paths(workspace_root, references)


def build_premise_block(missing: list[str]) -> str:
    """Render the prompt section injected when referenced paths are absent."""
    listed = "\n".join(f"- `{path}`" for path in missing)
    return (
        "---\n"
        "## Premise Check (IMPORTANT)\n"
        "\n"
        "The issue references the following paths, but they do **not** exist\n"
        "in this repository:\n"
        "\n"
        f"{listed}\n"
        "\n"
        "Before making any change, verify the issue's premise:\n"
        "\n"
        "1. Search the repository for the described file/symbol/behavior\n"
        "   (it may have moved or been renamed).\n"
        "2. If the issue describes a bug in code that does not exist, the\n"
        "   bug is unverifiable. **Do NOT create the missing file or invent\n"
        "   code to make the fix look plausible** — fabricated fixes are\n"
        "   worse than no fix.\n"
        "3. If you conclude the task cannot honestly be completed, write a\n"
        f"   JSON file at `{CANNOT_PROCEED_MARKER}` shaped like:\n"
        "\n"
        "   ```json\n"
        '   {"reason": "premise_not_met",\n'
        '    "details": "src/foo.py referenced by the issue does not exist; '
        'searched for <symbol> with no results",\n'
        '    "checked": ["src/foo.py", "git log", "grep _RetryStrategy"]}\n'
        "   ```\n"
        "\n"
        "   then stop without modifying source files. The orchestrator will\n"
        "   report your findings back to the issue author instead of opening\n"
        "   a merge request.\n"
        "\n"
        "Only proceed with code changes if you can locate the real code the\n"
        "issue is talking about.\n"
        "---"
    )


def read_cannot_proceed(workspace_root: Path | str | None) -> dict[str, Any] | None:
    """Return the parsed honest-exit marker, or ``None`` when absent.

    A marker that exists but fails to parse still counts as an exit
    request (the agent's intent was clear); its raw text is preserved
    under ``details`` so nothing the agent wrote is lost.
    """
    if workspace_root is None:
        return None
    marker = Path(workspace_root) / CANNOT_PROCEED_MARKER
    try:
        if not marker.is_file():
            return None
        raw = marker.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        logger.warning("cannot_proceed marker is not valid JSON — honoring it anyway")
        return {"reason": "cannot_proceed", "details": raw.strip()[:_DETAILS_MAX_CHARS]}
    if not isinstance(payload, dict):
        return {"reason": "cannot_proceed", "details": str(payload)[:_DETAILS_MAX_CHARS]}
    payload.setdefault("reason", "cannot_proceed")
    return payload


def format_cannot_proceed_comment(issue: Any, payload: dict[str, Any]) -> str:
    """Build the tracker comment posted when the agent honestly exits."""
    identifier = getattr(issue, "identifier", None) or getattr(issue, "id", "")
    reason = str(payload.get("reason", "cannot_proceed"))
    details = str(payload.get("details", "")).strip()
    checked = payload.get("checked")
    lines = [
        f"## ClawCodex could not proceed with {identifier}".rstrip(),
        "",
        f"**Reason**: `{reason}`",
    ]
    if details:
        lines += ["", details]
    if isinstance(checked, list) and checked:
        lines += ["", "**Checked**:"] + [f"- {item}" for item in checked[:_MAX_CHECKED_ITEMS]]
    lines += [
        "",
        "_No merge request was opened. If the premise is actually valid,"
        " please correct the issue description (file paths, reproduction"
        " steps) and re-trigger with the `agent:retry` label._",
    ]
    return "\n".join(lines)
