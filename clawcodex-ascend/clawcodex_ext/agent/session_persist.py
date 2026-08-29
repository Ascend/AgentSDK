#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# -------------------------------------------------------------------------
# This file is part of the AgentSDK project.
# Copyright (c) 2026 Clawd Codex Team
# Copyright (c) 2026 Huawei Technologies Co.,Ltd.
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

"""Downstream agent session persistence and session-storage hooks.

Extracted from ``src/agent/session.py`` so the upstream Session class
remains free of orchestrator-specific SessionStorage / TailFollower
concerns.

Architecture::

    src/agent/session.py                   ← upstream Session (calls hooks below)
        ↑ import
    clawcodex_ext/agent/session_persist.py    ← this downstream module

Two public hooks:

* ``save_to_session_storage(session)`` — persist conversation messages
  via SessionStorage (JSONL transcript) so ``--resume`` can attach a
  TailFollower to watch for lines written by a backgrounded agent.
* ``load_from_session_storage(session_id)`` — construct a Session-like
  object from a SessionStorage directory if one exists.

Session-init line changes:

* The very first call per session writes a ``session_init`` line as
  line 1 of ``transcript.jsonl`` carrying ``session_id``, ``provider``,
  ``model``, ``cwd``, and ``created_at``. ``Session.load()`` reads this
  line to reconstruct provider/model without needing ``session.json``.
* The duplicate ``cost_block`` write (to both ``metadata.json`` and
  ``transcript.jsonl``) has been removed. ``Session.save()`` now writes
  a ``session_snapshot`` line containing the cost block; cost_restore
  reads ``tail -1`` from the transcript instead of looking at
  ``session.json``.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Optional
import os


_SESSION_INIT_MARKER = "session_init"


def _resolve_sessions_dir() -> Path:
    """Return the active sessions directory, matching ``Session.load``.

    Mirrors ``clawcodex_ext.agent.session._get_sessions_dir`` without
    introducing a module-level import. Respects ``CLAWCODEX_SESSIONS_DIR``
    and evaluates ``Path.home()`` at call time so patches take effect.
    """
    override = str(os.environ.get("CLAWCODEX_SESSIONS_DIR", "")).strip()
    if override:
        return Path(override).expanduser()
    return Path.home() / ".clawcodex" / "sessions"


def save_to_session_storage(session: Any) -> None:
    """Persist conversation messages via SessionStorage (JSONL transcript).

    Best-effort: errors are swallowed so the upstream agent loop never
    fails on a persistence hiccup. The JSONL transcript is the file
    that :class:`TailFollower` watches during ``--resume``, so it must
    exist and contain all current messages for the resume path to work.

    On the **first** call for a given session, write a
    ``session_init`` line as the very first entry in ``transcript.jsonl``.
    Subsequent calls are idempotent — they detect the existing init line
    and skip writing a duplicate.

    The cost block is NOT written here anymore; ``Session.save()`` writes
    a ``session_snapshot`` line at exit time, and ``cost_restore`` reads
    that line via ``tail -1``. Writing a ``cost_block`` here would
    duplicate the cost on every save and confuse cost_restore (which
    keys on the LAST line of the transcript).
    """
    try:
        from clawcodex_ext.services.session_storage import SessionStorage

        # Use the same directory resolution as ``Session.load`` / ``Session.save``
        # (``clawcodex_ext.agent.session._get_sessions_dir``) so tests that patch
        # ``Path.home`` see writes and reads land in the same place. The module-level
        # ``SESSIONS_DIR`` constant is evaluated at import time, which breaks when
        # ``Path.home`` is patched after the constant has been materialised.
        sessions_dir = _resolve_sessions_dir()
        storage = SessionStorage(
            session_id=session.session_id,
            sessions_dir=sessions_dir,
        )
        storage.init_metadata(
            model=session.model,
            cwd=str(Path.cwd()),
            title=_derive_title(session),
        )

        # Write the session_init line once, at the very start
        # of the transcript, before any messages. Subsequent calls skip
        # this branch because ``_has_session_init`` returns True.
        if not _has_session_init(storage):
            _write_session_init_line(storage, session)

        # Write each message from the conversation.  Use ``write_raw``
        # with the serialised dict so we don't re-encode via
        # ``message_to_dict`` (which may not match the shape stored
        # in ``Conversation.to_dict``).
        conv_dict = session.conversation.to_dict()
        messages_list = conv_dict.get("messages", []) if isinstance(conv_dict, dict) else []

        # Compute ``parentUuid`` for each message and
        # stamp it onto the dict before writing. ``parentUuid``
        # encodes the chain topology so ``walkChainBeforeParse`` can
        # prune dead branches (from /rewind / fork) on read.
        #
        # The chain is rebuilt from scratch each save rather than
        # patched onto existing entries: this naturally reflects
        # rewinds (truncation breaks the previous chain, new
        # messages start a fresh branch pointing at the rewind
        # target) and stays robust against uuid regeneration on
        # resume. We only stamp messages whose uuid differs from
        # what's already on disk, so dedup in ``flush()`` remains
        # untouched and the on-disk history is never rewritten.
        messages_with_parent = _inject_parent_uuids(messages_list)

        for msg_data in messages_with_parent:
            if isinstance(msg_data, dict):
                storage.write_raw(msg_data)
        storage.flush()
    except Exception:  # nosec B110
        pass  # Best-effort; not critical if this fails.


def _inject_parent_uuids(
    messages_list: list,
) -> list[dict[str, Any]]:
    """Return a copy of ``messages_list`` with ``parentUuid`` populated.

    Each message's ``parentUuid`` is set to the previous
    message's ``uuid`` in the conversation list. Root message
    (index 0) gets ``parentUuid = None``.

    Behavioural notes:

    * **Only walks the in-memory conversation, not the on-disk
      transcript.** This is intentional: the chain is rebuilt from
      whatever the model is currently looking at, so rewinds (which
      truncate ``conversation.messages``) automatically produce a new
      chain pointing at the rewind target. The old messages remain
      on disk as a dead branch until ``walkChainBeforeParse``
      prunes them on read.
    * **Defensive against missing uuids.** If a message dict lacks
      a string uuid, its parentUuid is left untouched (we don't
      invent one); subsequent messages still chain off whatever
      previous uuid was last seen. This avoids breaking messages
      that originated from non-standard writers (e.g. legacy
      session_init / session_snapshot stubs).
    * **Idempotent.** Re-running this on the same list yields the
      same chain; we only overwrite an explicit ``None`` if a
      previous uuid exists, so the function is safe to call on
      dicts that already carry a ``parentUuid`` (legacy entries
      or test fixtures).
    * **Pure function.** No I/O. Returns a new list of dict copies
      so the caller's ``messages_list`` is not mutated.

    Args:
        messages_list: list of message dicts (typically from
            ``Conversation.to_dict()``). Non-dict entries are
            passed through unchanged.

    Returns:
        A new list of message dicts with ``parentUuid`` stamped.
    """
    out: list[dict[str, Any]] = []
    prev_uuid: Optional[str] = None
    for entry in messages_list:
        if not isinstance(entry, dict):
            out.append(entry)
            continue
        new_entry = dict(entry)
        uuid = entry.get("uuid")
        if isinstance(uuid, str) and uuid:
            # Always recompute parentUuid from the previous message
            # in the conversation list. The design doc
            # explicitly mandates computing the value on write;
            # we never trust a pre-existing value because rewinds
            # truncate the in-memory conversation and a stale
            # parentUuid on a "new" message would point at the
            # dead branch instead of the rewind target.
            new_entry["parentUuid"] = prev_uuid
            prev_uuid = uuid
        else:
            # No usable uuid — fall back to whatever the caller
            # already wrote (preserves legacy behaviour for
            # metadata stubs) or set None.
            new_entry["parentUuid"] = entry.get("parentUuid")
        out.append(new_entry)
    return out


def _has_session_init(storage: Any) -> bool:
    """Return True if the transcript already has a ``session_init`` line.

    We scan the on-disk transcript rather than maintaining a sentinel in
    memory so this stays correct across process restarts and across the
    ``save_to_session_storage`` <-> ``save_transcript`` <-> ``Session.save``
    dance. The scan is cheap: we stop at the first non-blank line and
    inspect its ``type`` field.
    """
    transcript_path = storage.session_dir / "transcript.jsonl"
    if not transcript_path.exists():
        return False
    try:
        import json

        with open(transcript_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except Exception:
                    return False
                if isinstance(entry, dict):
                    return entry.get("type") == _SESSION_INIT_MARKER
                return False
    except OSError:
        return False
    return False


def _write_session_init_line(storage: Any, session: Any) -> None:
    """Write a ``session_init`` line as the FIRST entry of the transcript.

    This line carries ``session_id``, ``provider``, ``model``,
    ``cwd``, and ``created_at`` — the information :meth:`Session.load`
    needs to reconstruct provider/model without a separate ``session.json``.
    Subsequent ``Session.save()`` calls do NOT touch this line; new
    provider/model values land in the trailing ``session_snapshot`` line.

    When the transcript already exists with legacy message lines (the
    common upgrade path — orchestrator / cron sessions that wrote
    messages before ``save_to_session_storage`` was wired up), we
    rewrite the file: ``session_init`` first, then the existing
    message lines in their original order. JSONL is append-only and
    cannot cheaply prepend, so this rewrite is necessary on the
    first call that runs after the upgrade.
    """
    import json

    payload: dict[str, Any] = {
        "type": _SESSION_INIT_MARKER,
        "session_id": session.session_id,
        "provider": getattr(session, "provider", "") or "",
        "model": getattr(session, "model", "") or "",
        "cwd": str(Path.cwd()),
        "created_at": datetime.now().isoformat(),
    }
    init_line = json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n"

    transcript_path = storage.session_dir / "transcript.jsonl"
    storage.session_dir.mkdir(parents=True, exist_ok=True)

    # Read the existing transcript (if any). We rewrite the file with
    # ``session_init`` first, then the previous lines in order.
    existing_lines: list[str] = []
    if transcript_path.exists():
        try:
            with open(transcript_path, "r", encoding="utf-8") as f:
                for raw in f:
                    raw = raw.rstrip("\r\n")
                    if not raw.strip():
                        continue
                    existing_lines.append(raw + "\n")
        except OSError:
            existing_lines = []

    import tempfile

    tmp_fd, tmp_path = tempfile.mkstemp(dir=str(storage.session_dir), suffix=".tmp")
    try:
        with os.fdopen(tmp_fd, "w", encoding="utf-8") as f:
            f.write(init_line)
            for line in existing_lines:
                f.write(line)
        os.replace(tmp_path, transcript_path)
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass  # Best-effort cleanup must not hide the original write error.
        raise

    # Reset the SessionStorage's de-dup baseline so the follow-up
    # ``flush()`` re-scans the now-prepended transcript and skips any
    # message uuids already on disk.
    storage._flushed_uuids = storage._scan_flushed_uuids()


def _derive_title(session: Any) -> str:
    """Derive a display title for the session."""
    base = f"session-{session.session_id[:8]}"
    # Try to get a title from the session object itself
    title = getattr(session, "title", None) or ""
    return title if title else base


def _extract_last_user_input(messages_list: list) -> str:
    """Extract the most recent user message text from the conversation."""
    for msg in reversed(messages_list):
        if not isinstance(msg, dict):
            continue
        if msg.get("role") != "user":
            continue
        content = msg.get("content", "") or ""
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: list[str] = []
            for item in content:
                if isinstance(item, str):
                    parts.append(item)
                elif isinstance(item, dict):
                    if item.get("type") in (None, "text"):
                        parts.append(str(item.get("text") or ""))
            if parts:
                return " ".join(parts)
    return ""


def load_from_session_storage(session_id: str) -> Optional[dict[str, Any]]:
    """Construct session data from a SessionStorage directory if one exists.

    Supports sessions stored in the SessionStorage
    directory format (``~/.clawcodex/sessions/<sid>/transcript.jsonl`` +
    ``metadata.json``). This is the on-disk shape the orchestrator's
    AgentRunner writes.

    Returns a dict with keys (session_id, model, start_time, last_updated)
    or ``None`` when no SessionStorage directory exists for ``session_id``.
    """
    try:
        from clawcodex_ext.services.session_resume import resume_session
        from clawcodex_ext.services.session_storage import SESSIONS_DIR
    except ImportError:
        return None

    result = resume_session(session_id, sessions_dir=SESSIONS_DIR)
    if not result.success or result.metadata is None:
        return None

    md = result.metadata
    return {
        "session_id": md.session_id,
        "model": md.model,
        "start_time": str(md.start_time),
        "last_updated": str(md.last_updated),
    }


__all__ = [
    "save_to_session_storage",
    "load_from_session_storage",
]
