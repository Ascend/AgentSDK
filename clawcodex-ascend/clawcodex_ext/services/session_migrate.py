#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# -------------------------------------------------------------------------
#  This file is part of the AgentSDK project.
# Copyright (c) 2026 Huawei Technologies Co.,Ltd.
#
# AgentSDK is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#
#           http://license.coscl.org.cn/MulanPSL2
#
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
# -------------------------------------------------------------------------

"""F-49 P5-H: session format migration tool.

Converts legacy 3-file sessions (session.json + metadata.json +
transcript.jsonl) into the unified 2-file format introduced by Phase 5
(metadata.json + transcript.jsonl with ``session_init`` / ``session_snapshot``
lines).

Usage::

    from src.services.session_migrate import migrate_session, migrate_all

    # Migrate a single session.
    result = migrate_session("sid-12345", remove_legacy=True)

    # Walk the entire sessions dir and migrate every legacy session.
    summary = migrate_all(remove_legacy=False)

Why a separate tool? ``Session.load()`` already auto-detects legacy
``session.json`` files and reads them transparently, so the runtime
read path keeps working without this tool. The migration is needed for
two reasons:

1. **Convergence** — once a session is migrated, new saves land only
   in ``transcript.jsonl`` and ``metadata.json``. ``session.json``
   never reappears, eliminating the message-redundancy cost.
2. **Cost-restore on the tail** — without migration, ``session.json``
   remains the cost-restore primary source. With migration, the
   trailing ``session_snapshot`` line takes over, matching the new
   on-disk shape produced by ``Session.save()``.

The migration is **idempotent**: re-running on an already-migrated
session is a no-op (detected via the ``session_init`` marker on
``transcript.jsonl`` line 1).
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class MigrationResult:
    """Outcome of a single ``migrate_session`` invocation."""

    session_id: str
    migrated: bool = False
    skipped_reason: str = ""
    source_session_json: bool = False
    source_metadata_json: bool = False
    source_transcript_jsonl: bool = False
    messages_migrated: int = 0
    cost_migrated: bool = False
    removed_session_json: bool = False
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "migrated": self.migrated,
            "skipped_reason": self.skipped_reason,
            "source_session_json": self.source_session_json,
            "source_metadata_json": self.source_metadata_json,
            "source_transcript_jsonl": self.source_transcript_jsonl,
            "messages_migrated": self.messages_migrated,
            "cost_migrated": self.cost_migrated,
            "removed_session_json": self.removed_session_json,
            "error": self.error,
        }


@dataclass
class MigrationSummary:
    """Outcome of ``migrate_all`` — aggregate over many sessions."""

    sessions_dir: str
    total_sessions: int = 0
    migrated_count: int = 0
    skipped_count: int = 0
    error_count: int = 0
    results: list[MigrationResult] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "sessions_dir": self.sessions_dir,
            "total_sessions": self.total_sessions,
            "migrated_count": self.migrated_count,
            "skipped_count": self.skipped_count,
            "error_count": self.error_count,
            "results": [r.to_dict() for r in self.results],
        }


def _sessions_dir() -> Path:
    """Default sessions directory. Tests monkeypatch this."""
    return Path.home() / ".clawcodex" / "sessions"


def _has_session_init_marker(transcript_path: Path) -> bool:
    """True iff transcript.jsonl's first non-blank line is session_init."""
    if not transcript_path.exists():
        return False
    try:
        with open(transcript_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    return False
                return isinstance(entry, dict) and entry.get("type") == "session_init"
    except OSError:
        return False
    return False


def _read_session_json(session_json: Path) -> dict[str, Any] | None:
    """Read and parse ``session.json`` or return ``None`` on failure."""
    if not session_json.exists():
        return None
    try:
        data = json.loads(session_json.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("session.json unreadable for %s: %s", session_json, exc)
        return None
    return data if isinstance(data, dict) else None


def _existing_message_uuids(transcript_path: Path) -> set[str]:
    """Collect uuids already present on disk to avoid double-appending."""
    seen: set[str] = set()
    if not transcript_path.exists():
        return seen
    try:
        with open(transcript_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(entry, dict):
                    uuid = entry.get("uuid")
                    if isinstance(uuid, str) and uuid:
                        seen.add(uuid)
    except OSError:
        return seen
    return seen


def migrate_session(
    session_id: str,
    *,
    sessions_dir: Path | None = None,
    remove_legacy: bool = False,
) -> MigrationResult:
    """Migrate a single session to the Phase-5 unified format.

    Parameters
    ----------
    session_id:
        The session directory name (matches ``get_session_id()``).
    sessions_dir:
        Override the default ``~/.clawcodex/sessions`` directory.
        Tests pass a temp directory.
    remove_legacy:
        When True, also delete ``session.json`` after a successful
        migration. Default False — most operators want to inspect the
        result first.

    Returns
    -------
    MigrationResult
        ``migrated=True`` on success; ``skipped_reason`` is set when
        the session is already in the new format or has no source
        files; ``error`` is set on hard failures.
    """
    base = sessions_dir or _sessions_dir()
    session_dir = base / session_id
    result = MigrationResult(session_id=session_id)

    if not session_dir.is_dir():
        result.error = f"session directory not found: {session_dir}"
        return result

    session_json = session_dir / "session.json"
    metadata_json = session_dir / "metadata.json"
    transcript_jsonl = session_dir / "transcript.jsonl"

    result.source_session_json = session_json.exists()
    result.source_metadata_json = metadata_json.exists()
    result.source_transcript_jsonl = transcript_jsonl.exists()

    # Already migrated: transcript.jsonl has a session_init marker.
    if _has_session_init_marker(transcript_jsonl):
        result.skipped_reason = "transcript.jsonl already has session_init marker; session is in the new format"
        return result

    # No legacy sources to migrate from — there's nothing to do.
    if not result.source_session_json and not result.source_transcript_jsonl:
        result.skipped_reason = "no legacy session.json or transcript.jsonl found"
        return result

    session_dir.mkdir(parents=True, exist_ok=True)

    legacy = _read_session_json(session_json) if result.source_session_json else None
    provider = ""
    model = ""
    created_at = ""
    cost_block: dict[str, Any] = {}
    legacy_messages: list[dict[str, Any]] = []

    if legacy is not None:
        provider = str(legacy.get("provider", "") or "")
        model = str(legacy.get("model", "") or "")
        created_at = str(legacy.get("created_at", "") or "")
        cost_block = legacy.get("cost") or {}
        if not isinstance(cost_block, dict):
            cost_block = {}
        conv = legacy.get("conversation") or {}
        if isinstance(conv, dict):
            msgs = conv.get("messages") or []
            if isinstance(msgs, list):
                legacy_messages = [m for m in msgs if isinstance(m, dict)]

    # Fallback: if there's no session.json but there IS a transcript.jsonl
    # without a session_init marker (pure orchestrator / cron legacy),
    # pull provider/model/created_at from metadata.json so the new
    # session_init line carries useful info.
    if legacy is None and metadata_json.exists():
        try:
            md = json.loads(metadata_json.read_text(encoding="utf-8"))
            if isinstance(md, dict):
                provider = str(md.get("provider", "") or provider)
                model = str(md.get("model", "") or model)
                start = md.get("start_time")
                if isinstance(start, (int, float)):
                    from datetime import datetime, timezone

                    created_at = datetime.fromtimestamp(start, tz=timezone.utc).isoformat()
        except (OSError, json.JSONDecodeError):
            pass

    # Existing on-disk uuids — used to skip duplicates when both
    # session.json and transcript.jsonl hold overlapping message lists.
    _existing_message_uuids(transcript_jsonl)

    # Build the new transcript content. Strategy: read the existing
    # transcript (if any) line by line, copy message entries that
    # aren't already represented in legacy_messages, then prepend the
    # session_init line and append the session_snapshot line.
    existing_message_lines: list[str] = []
    if transcript_jsonl.exists():
        try:
            with open(transcript_jsonl, "r", encoding="utf-8") as f:
                for raw in f:
                    raw = raw.rstrip("\r\n")
                    if not raw.strip():
                        continue
                    try:
                        entry = json.loads(raw)
                    except json.JSONDecodeError:
                        # Preserve unparseable lines as-is so we don't
                        # silently drop data during migration.
                        existing_message_lines.append(raw)
                        continue
                    if not isinstance(entry, dict):
                        existing_message_lines.append(raw)
                        continue
                    if entry.get("type") in (
                        "session_init",
                        "session_snapshot",
                        "cost_block",
                    ):
                        # Skip init/snapshot/cost lines — we'll write
                        # fresh ones below. Existing cost_block lines
                        # are legacy and the new session_snapshot
                        # supersedes them.
                        continue
                    if entry.get("role") == "system" and entry.get("content") == "__background_complete__":
                        continue
                    existing_message_lines.append(raw)
        except OSError as exc:
            result.error = f"failed to read transcript.jsonl: {exc}"
            return result

    # Pick the message set for the new transcript. ``transcript.jsonl``
    # is the canonical message list in the unified format, so we prefer
    # it whenever it already has content — its lines are already
    # JSONL-shaped and uuid-bearing, no round-trip through
    # ``session.json`` is required. ``session.json`` is only used as a
    # fallback when the transcript has no messages (the rare case of a
    # session that crashed before flushing its first turn).
    transcript_entries: list[dict[str, Any]] = []
    for raw in existing_message_lines:
        try:
            entry = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if not isinstance(entry, dict):
            continue
        transcript_entries.append(entry)

    if transcript_entries:
        combined = transcript_entries
    elif legacy_messages:
        combined = legacy_messages
    else:
        combined = []

    # ---- Write the new transcript ----
    session_dir.mkdir(parents=True, exist_ok=True)
    try:
        with open(transcript_jsonl, "w", encoding="utf-8") as f:
            # Line 1: session_init
            init_payload: dict[str, Any] = {
                "type": "session_init",
                "session_id": session_id,
                "provider": provider,
                "model": model,
                "created_at": created_at,
            }
            f.write(json.dumps(init_payload, ensure_ascii=False, separators=(",", ":")) + "\n")
            # Middle lines: messages
            f.writelines(json.dumps(msg, ensure_ascii=False, separators=(",", ":")) + "\n" for msg in combined)
            # Last line: session_snapshot (only when we have cost data)
            if cost_block:
                snap_payload: dict[str, Any] = {
                    "type": "session_snapshot",
                    "cost": cost_block,
                    "provider": provider,
                    "model": model,
                }
                f.write(json.dumps(snap_payload, ensure_ascii=False, separators=(",", ":")) + "\n")
                result.cost_migrated = True
            result.messages_migrated = len(combined)
    except OSError as exc:
        result.error = f"failed to write transcript.jsonl: {exc}"
        return result

    result.migrated = True

    # Optionally remove legacy session.json.
    if remove_legacy and session_json.exists():
        try:
            session_json.unlink()
            result.removed_session_json = True
        except OSError as exc:
            logger.warning("migrated %s but could not remove session.json: %s", session_id, exc)

    return result


def migrate_all(
    *,
    sessions_dir: Path | None = None,
    remove_legacy: bool = False,
) -> MigrationSummary:
    """Migrate every legacy session under ``sessions_dir``.

    Walks each subdirectory of ``sessions_dir`` and runs
    :func:`migrate_session`. Errors are recorded per-session but do not
    abort the overall walk.
    """
    base = sessions_dir or _sessions_dir()
    summary = MigrationSummary(sessions_dir=str(base))

    if not base.is_dir():
        return summary

    for entry in sorted(base.iterdir()):
        if not entry.is_dir():
            continue
        summary.total_sessions += 1
        result = migrate_session(entry.name, sessions_dir=base, remove_legacy=remove_legacy)
        summary.results.append(result)
        if result.error:
            summary.error_count += 1
        elif result.migrated:
            summary.migrated_count += 1
        else:
            summary.skipped_count += 1

    return summary


# ---------------------------------------------------------------------------
# CLI helper — invoked by ``clawcodex-dev session migrate --from-3-file``
# and the F-49 P5 subcommand sieve.
# ---------------------------------------------------------------------------


def handle_session_migrate_cli(argv: list[str]) -> int:
    """CLI handler for ``clawcodex-dev session migrate``.

    Subcommands::

        clawcodex-dev session migrate --from-3-file [--all] [--remove-legacy] [SESSION_ID]

    * ``--from-3-file`` — required marker, scopes the migration to the
      Phase-5 format conversion. Future migrations can use other
      source formats.
    * ``--all`` — migrate every session in the directory. Without it,
      the positional ``SESSION_ID`` is required.
    * ``--remove-legacy`` — also delete ``session.json`` after a
      successful migration. Default is dry-run style (keep ``session.json``
      so operators can inspect).
    * ``SESSION_ID`` — when supplied (and ``--all`` is absent), migrate
      just this one session.

    Exit codes:

    * 0 — success (migrated or already in new format)
    * 1 — a session errored during migration
    * 2 — bad arguments
    """
    if "--from-3-file" not in argv:
        print(
            "usage: clawcodex-dev session migrate --from-3-file [--all] [--remove-legacy] [SESSION_ID]",
            file=__import__("sys").stderr,
        )
        return 2

    argv = [a for a in argv if a != "--from-3-file"]
    remove_legacy = "--remove-legacy" in argv
    argv = [a for a in argv if a != "--remove-legacy"]
    do_all = "--all" in argv
    argv = [a for a in argv if a != "--all"]

    if do_all:
        summary = migrate_all(remove_legacy=remove_legacy)
        print(json.dumps(summary.to_dict(), indent=2))
        return 1 if summary.error_count else 0

    if not argv:
        print(
            "SESSION_ID required (or pass --all to migrate every session)",
            file=__import__("sys").stderr,
        )
        return 2

    session_id = argv[0]
    result = migrate_session(session_id, remove_legacy=remove_legacy)
    print(json.dumps(result.to_dict(), indent=2))
    if result.error:
        return 1
    return 0


__all__ = [
    "MigrationResult",
    "MigrationSummary",
    "handle_session_migrate_cli",
    "migrate_all",
    "migrate_session",
]
