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

"""Resume-time checks for headless mode.

Warn when a resumed session uses a different system prompt, provider, or model
than the recorded history, and restore stable session metadata so downstream
consumers retain continuity.

All checks are **best-effort and non-fatal**: a missing metadata file,
an unreadable transcript, or a malformed history is silently skipped.
Warnings go to the caller-supplied ``stderr`` stream so they don't
pollute structured stdout output.

Design notes
------------
* Lives in ``clawcodex_ext`` (Layer 1) — augments upstream headless
 behaviour without touching ``src/``.
* Pure functions, no global state, easy to unit-test in isolation.
* The original session's metadata is read via :class:`SessionStorage` so the same on-disk shape that
 ``Session.save`` writes is consulted (no second source of truth).
"""

from __future__ import annotations

import logging
from typing import Any, IO

logger = logging.getLogger(__name__)

__all__ = [
    "warn_system_prompt_drift",
    "warn_provider_model_mismatch",
    "restore_metadata_from_session",
]


def warn_system_prompt_drift(
    history_messages: Any,
    current_append_prompt: str,
    stderr: IO[str],
) -> None:
    """Warn when ``append_system_prompt`` differs across runs.

    The resumed conversation's history was generated under a system
    prompt that included ``old_append``; the current run uses
    ``current_append_prompt``. When they differ, the model's prior
    turns reflect a different instruction set than the one guiding the
    new turn — a subtle source of inconsistency (e.g. the history
    shows terse answers because P_old said "be brief", but P_new says
    "be verbose").

    The original ``append_system_prompt`` is recovered from the
    transcript's first ``session_init`` line when present; otherwise
    the check is skipped (no false positive on legacy transcripts).

    Parameters
    ----------
    history_messages
    The resumed conversation's message iterable. Only the first
    ``session_init``-shaped entry (a system message whose content
    carries the system prompt body) is consulted; anything else is
    ignored.
    current_append_prompt
    The current run's ``append_system_prompt`` value (may be empty).
    stderr
    Destination for the warning. Silently ignored if ``None``.
    """
    if stderr is None:
        return
    old_append = _extract_history_system_prompt_append(history_messages)
    if old_append is None:
        # No recoverable prior prompt — nothing to compare. Avoid
        # false positives on fresh-ish transcripts.
        return
    if _normalize_prompt(old_append) == _normalize_prompt(current_append_prompt):
        return
    try:
        print(
            "warning: --append-system-prompt differs from the resumed "
            "session's system prompt. History was generated under a "
            "different instruction set; the model may produce "
            "inconsistent style/tool usage.",
            file=stderr,
        )
    except Exception:  # nosec
        pass  # Warning output is best-effort and must not block session resume.


def warn_provider_model_mismatch(
    original_provider: str,
    original_model: str,
    current_provider: str,
    current_model: str,
    stderr: IO[str],
) -> None:
    """Warn when provider/model differs on resume.

    A resumed conversation's history carries assistant turns produced
    by ``(original_provider, original_model)``. Running the new turn
    under a different backend can change tokenisation, tool-call
    formatting, and answer style — leading to subtle drift or
    malformed tool_use blocks the new provider can't parse.

    Comparison is case-insensitive and empty original values are
    treated as "unknown" (skip the check to avoid false positives on
    legacy sessions that didn't record provider/model).

    Parameters
    ----------
    original_provider, original_model
    Provider / model recorded for the resumed session (from
    ``Session.provider`` / ``Session.model`` or the transcript's
    ``session_init`` line).
    current_provider, current_model
    The current run's resolved provider / model.
    stderr
    Destination for the warning.
    """
    if stderr is None:
        return
    orig_p = (original_provider or "").strip().lower()
    orig_m = (original_model or "").strip().lower()
    cur_p = (current_provider or "").strip().lower()
    cur_m = (current_model or "").strip().lower()
    # Skip when the original session didn't record provider/model
    # (legacy / orchestrator sessions) — can't compare to unknown.
    if not orig_p and not orig_m:
        return
    mismatches: list[str] = []
    if orig_p and cur_p and orig_p != cur_p:
        mismatches.append(f"provider ({original_provider!r} -> {current_provider!r})")
    if orig_m and cur_m and orig_m != cur_m:
        mismatches.append(f"model ({original_model!r} -> {current_model!r})")
    if not mismatches:
        return
    try:
        print(
            "warning: --resume session was created with a different "
            + " and ".join(mismatches)
            + ". History turns reflect the old backend; answer style "
            "and tool-call shape may diverge.",
            file=stderr,
        )
    except Exception:  # nosec
        pass  # Warning output is best-effort and must not block session resume.


def restore_metadata_from_session(
    target_session_id: str,
    source_session_id: str,
    *,
    agent_name: str = "",
) -> bool:
    """Copy title/tags/agent_name from source to target session.

    When a session is resumed (or forked), the new run should inherit
    the original's ``title`` / ``tags`` so the ``/resume`` browser and
    downstream consumers (visualizer, orchestrator issue registry)
    show continuity rather than a fresh untitled session.

    Reads the source session's ``metadata.json`` via :class:`SessionStorage` and writes the recovered fields onto the
    target session's metadata. The target's ``start_time`` /
    ``message_count`` are preserved (init_metadata is idempotent on
    existing files).

    Parameters
    ----------
    target_session_id
    The session ID whose metadata should receive the inherited
    fields (typically == source for ``--resume``, or the new fork
    ID for ``--fork-session``).
    source_session_id
    The session ID to inherit metadata from.
    agent_name
    Optional agent-name override. When non-empty, written to the
    in-memory metadata (not persisted — matches P5-F policy that
    ``agent_name`` is a transient field). Pass the resolved
    startup agent's name when available.

    Returns
    -------
    bool
    ``True`` if any field was inherited, ``False`` otherwise
    (source missing, unreadable, or had nothing to copy).
    """
    if not target_session_id or not source_session_id:
        return False
    try:
        from src.services.session_storage import SessionStorage

        source_storage = SessionStorage(session_id=source_session_id)
        source_meta = source_storage.get_metadata()
        if source_meta is None:
            return False
        title = getattr(source_meta, "title", "") or ""
        tags = list(getattr(source_meta, "tags", []) or [])
        if not title and not tags and not agent_name:
            return False
        target_storage = SessionStorage(session_id=target_session_id)
        # init_metadata is idempotent: if the target already has a
        # metadata.json (e.g. headless already wrote session_init),
        # only the caller-supplied fields are updated; wall-clock
        # anchors are preserved.
        target_storage.init_metadata(
            title=title,
            tags=tags,
            agent_name=agent_name,
        )
        return True
    except Exception:  # nosec
        logger.debug(
            "metadata inheritance from %s -> %s failed",
            source_session_id,
            target_session_id,
            exc_info=True,
        )
        return False


# ---- internals --------------------------------------------------------


def _extract_history_system_prompt_append(messages: Any) -> str | None:
    """Best-effort recovery of the prior run's append_system_prompt.

    Looks for the transcript's first ``session_init`` line (a system
    message whose dict form carries an ``append_system_prompt`` field)
    OR a system message whose content includes a sentinel marker we
    write when persisting. Returns ``None`` when nothing recoverable
    is found — callers treat ``None`` as "skip the check".
    """
    if messages is None:
        return None
    try:
        iterator = iter(messages)
    except TypeError:
        return None
    for message in iterator:
        append = _system_append_from_message(message)
        if append is not None:
            return append
    return None


def _system_append_from_message(message: Any) -> str | None:
    """Return the ``append_system_prompt`` recorded on a system/init message.

    Tolerates dataclass messages (``.role`` / ``.content`` / ``.extra``)
    and dict-shaped messages. Only system-role messages carrying an
    ``append_system_prompt`` key (in their dict form or an ``extra``
    attribute) are recognised; anything else returns ``None``.
    """
    if message is None:
        return None
    role = _get(message, "role")
    if role is not None and str(role).lower() != "system":
        return None
    # Dict-shaped message (transcript round-trip).
    if isinstance(message, dict):
        extra = message.get("extra")
        if isinstance(extra, dict) and "append_system_prompt" in extra:
            val = extra["append_system_prompt"]
            return str(val) if val is not None else ""
        if "append_system_prompt" in message:
            val = message["append_system_prompt"]
            return str(val) if val is not None else ""
        return None
    # Dataclass message.
    extra = getattr(message, "extra", None)
    if isinstance(extra, dict) and "append_system_prompt" in extra:
        val = extra["append_system_prompt"]
        return str(val) if val is not None else ""
    if hasattr(message, "append_system_prompt"):
        val = getattr(message, "append_system_prompt")
        return str(val) if val is not None else ""
    return None


def _normalize_prompt(text: str) -> str:
    """Normalise a prompt body for comparison (strip + collapse ws)."""
    if not text:
        return ""
    return " ".join(str(text).split())


def _get(obj: Any, key: str) -> Any:
    if obj is None:
        return None
    if isinstance(obj, dict):
        return obj.get(key)
    return getattr(obj, key, None)
