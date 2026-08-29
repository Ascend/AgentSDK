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

"""Sidechain transcript for /btw side questions.

Records each ``/btw`` invocation (question + response + usage + provider)
to a dedicated JSONL file under ``$CLAWCODEX_DATA_DIR/sidechains/`` so the
side question has an independent paper trail — and *never* pollutes the
main session's ``transcript.jsonl``. This preserves the side-question
design's "transcript isolation" boundary.

Storage shape
-------------

One file per session: ``btw-<session_id>.jsonl``. Each ``/btw`` call
appends one JSON line; ``O_APPEND`` mode on POSIX guarantees atomic
writes for line sizes up to ``PIPE_BUF`` (≥ 4096 bytes on every modern
Unix) so concurrent ``/btw`` calls from the same session never interleave.
A single side-question JSON line is far below 4 KiB.

Record schema (one JSON line per call)::

 {
 "ts": "2026-07-02T12:34:56", // ISO 8601, second precision
 "epoch": 1751475296.123, // wall-clock for tooling
 "type": "btw", // future-proof for /watz/etc
 "session_id": "<active session uuid>",
 "question": "what is X?",
 "response": "X is...", // null on failure
 "usage": { // token counts; empty on failure
 "input_tokens": 123,
 "output_tokens": 45
 },
 "provider": "anthropic", // optional
 "model": "claude...", // optional
 "error": "..."// optional, present iff /btw failed
 }

Disable / reconfigure
---------------------

* ``CLAWCODEX_DISABLE_SIDECHAIN_TRANSCRIPT=1`` (or any truthy value)
 disables recording — the function becomes a no-op.
* ``CLAWCODEX_DATA_DIR=/path/to/root`` overrides the base directory
 (default: ``~/.clawcodex``).

Failure semantics
-----------------

Sidechain writes are *fire-and-forget*. Any IO failure
(``PermissionError``, ``OSError``, full disk, missing parent dir that
``mkdir`` cannot create, etc.) is logged at WARNING level and swallowed —
the main ``/btw`` user flow must never be blocked or visibly affected by
sidechain bookkeeping. This matches the isolation invariant that
"the side question never observes any failure from the recording path".
"""

from __future__ import annotations

import json
import logging
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


_SIDECHAINS_DIRNAME = "sidechains"
_SIDECHAIN_FILE_PREFIX = "btw-"
_SIDECHAIN_FILE_SUFFIX = ".jsonl"

_DISABLE_ENV_VAR = "CLAWCODEX_DISABLE_SIDECHAIN_TRANSCRIPT"
_DATA_DIR_ENV_VAR = "CLAWCODEX_DATA_DIR"


def _is_env_truthy(value: str | None) -> bool:
    """Standard env-truthy test (matches ``memdir/paths.py`` convention)."""
    if value is None:
        return False
    return value.strip().lower() in ("1", "true", "yes", "on")


def is_sidechain_transcript_enabled() -> bool:
    """Whether sidechain transcript recording is currently enabled.

    Default is *enabled*. Set ``CLAWCODEX_DISABLE_SIDECHAIN_TRANSCRIPT=1``
    (or any truthy value: ``true`` / ``yes`` / ``on``) to opt out.
    """
    return not _is_env_truthy(os.environ.get(_DISABLE_ENV_VAR))


def get_sidechain_dir() -> Path:
    """Return the directory where sidechain JSONL files are written.

    Resolution order:
      1. ``$CLAWCODEX_DATA_DIR/sidechains/`` if the env var is set.
      2. ``~/.clawcodex/sidechains/`` as fallback.

    Note: the parent directory is **not** created here — that happens on
    the first record write, so a session that never calls ``/btw`` leaves
    no trace on disk.
    """
    override = os.environ.get(_DATA_DIR_ENV_VAR)
    if override:
        root = Path(override).expanduser()
    else:
        root = Path.home() / ".clawcodex"
    return root / _SIDECHAINS_DIRNAME


def _sanitize_session_id(session_id: str | None) -> str | None:
    """Reduce *session_id* to a filesystem-safe filename component.

    Session ids are normally ``uuid4().hex`` (lowercase hex), so this is
    a defensive no-op in practice. We replace anything outside
    ``[A-Za-z0-9_-]`` with ``_`` and reject empty / fully-stripped
    results so we never write a file named ``btw-.jsonl``.
    """
    if not session_id:
        return None
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in session_id)
    return safe or None


def get_sidechain_path(session_id: str | None) -> Path | None:
    """Return the sidechain JSONL path for *session_id*.

    Returns ``None`` when *session_id* is missing or unprintable — the
    caller should treat that as "do not record" rather than as an error.
    """
    safe = _sanitize_session_id(session_id)
    if not safe:
        return None
    return get_sidechain_dir() / f"{_SIDECHAIN_FILE_PREFIX}{safe}{_SIDECHAIN_FILE_SUFFIX}"


def _opener_0600(path: str, flags: int) -> int:
    """File opener that enforces ``0o600`` on create.

    Transcripts can carry sensitive prompt content — readable by the
    current user only. Matches ``clawcodex_ext/agent/transcript.py``
    ``TranscriptWriter``'s permission posture.
    """
    return os.open(path, flags, 0o600)


def record_btw_invocation(
    *,
    session_id: str | None,
    question: str,
    response: str | None,
    usage: dict[str, Any] | None,
    provider: str | None = None,
    model: str | None = None,
    error: str | None = None,
) -> Path | None:
    """Append a single JSONL record for one ``/btw`` invocation.

    Returns the file path that was appended to, or ``None`` if recording
    was skipped (disabled, missing session id, or unrecoverable IO
    error). **Never raises** — sidechain bookkeeping must never block
    the user-visible ``/btw`` flow.
    """
    if not is_sidechain_transcript_enabled():
        return None

    path = get_sidechain_path(session_id)
    if path is None:
        return None

    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        record: dict[str, Any] = {
            "ts": datetime.now().isoformat(timespec="seconds"),
            "epoch": time.time(),
            "type": "btw",
            "session_id": session_id,
            "question": question,
            "response": response,
            "usage": dict(usage) if usage else {},
        }
        if provider:
            record["provider"] = provider
        if model:
            record["model"] = model
        if error:
            record["error"] = error

        # ``ensure_ascii=False`` preserves Chinese / unicode question text
        # verbatim, matching the project's interaction-language rule.
        # ``O_APPEND | O_CREAT`` keeps multi-call sessions append-only;
        # ``O_CLOEXEC`` prevents the fd from leaking into bash subprocesses
        # spawned by other tools.
        flags = os.O_WRONLY | os.O_APPEND | os.O_CREAT
        if hasattr(os, "O_CLOEXEC"):
            flags |= os.O_CLOEXEC
        fd = os.open(str(path), flags, 0o600)
        try:
            line = json.dumps(record, ensure_ascii=False) + "\n"
            os.write(fd, line.encode("utf-8"))
        finally:
            os.close(fd)
        return path
    except Exception:  # nosec
        logger.warning(
            "failed to record /btw sidechain transcript (session=%s, question=%r)",
            session_id,
            (question or "")[:80],
            exc_info=True,
        )
        return None


def list_sidechain_files(session_id: str | None = None) -> list[Path]:
    """Return sidechain transcript files on disk.

    When *session_id* is given, returns the single expected path if it
    exists (or an empty list). When omitted, returns every
    ``btw-*.jsonl`` file under the sidechain directory sorted by name.
    Useful for tooling (cleanup, inspection) and tests.
    """
    root = get_sidechain_dir()
    if not root.exists():
        return []
    if session_id:
        target = get_sidechain_path(session_id)
        return [target] if target is not None and target.exists() else []
    return sorted(root.glob(f"{_SIDECHAIN_FILE_PREFIX}*{_SIDECHAIN_FILE_SUFFIX}"))


def read_sidechain_file(path: Path) -> list[dict[str, Any]]:
    """Read a sidechain JSONL file into a list of record dicts.

    Tolerant of malformed lines — they are skipped with a WARNING so a
    half-written file from a crashed previous run does not break tooling.
    Intended for tests and admin scripts; production code paths should
    use the writer-side helpers instead.
    """
    records: list[dict[str, Any]] = []
    if not path.exists():
        return records
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                logger.warning(
                    "malformed JSON at %s:%d (skipping)",
                    path,
                    line_no,
                )
    return records


__all__ = [
    "get_sidechain_dir",
    "get_sidechain_path",
    "is_sidechain_transcript_enabled",
    "list_sidechain_files",
    "read_sidechain_file",
    "record_btw_invocation",
]
