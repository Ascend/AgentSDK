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
# This file is derived from Clawd Codex (https://github.com/agentforce314/clawcodex),
# which is licensed under the MIT License.
# Copyright (c) 2026 Clawd Codex Team
# -------------------------------------------------------------------------
# -------------------------------------------------------------------------

"""Cost-state restore orchestrator.

Mirrors TS ``cost-tracker.ts:149`` (``restoreCostStateForSession``). Reads
the persisted cost snapshot for a given session ID and dispatches
``set_cost_state_for_restore`` into the bootstrap singleton so the
``/resume`` path picks up where the last session left off, rather than
silently starting from zero.

F-49 P5-C: primary source is the **trailing** ``session_snapshot`` line
in ``transcript.jsonl`` (written by :meth:`Session.save`). The reader
walks the transcript once and remembers the latest line whose ``type``
is ``session_snapshot`` (new format) or ``cost_block`` (legacy format
written by pre-P5-E ``session_persist``). Both shapes carry the same
``cost`` dict shape, so the restore code is identical.

When ``transcript.jsonl`` has no snapshot/cost_block line (e.g. very
new sessions that haven't been ``save()``d yet, or pure orchestrator
sessions), falls back to the legacy ``session.json`` snapshot written
by pre-Phase-5 ``Session.save()``.

The TS file ``cost-tracker.ts`` does two things: defines the
``CostTracker`` class (which Python's port has consolidated onto the
bootstrap singleton) and the restore orchestrator. The orchestrator is
the only piece that needs its own file in Python — pricing is at
``src/services/pricing.py`` and accounting is at
``src/bootstrap/state.py``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.bootstrap.state import (
    ModelUsage,
    SessionId,
    set_cost_state_for_restore,
)
from src.utils.clawcodex_dirs import get_sessions_dir


def _sessions_dir() -> Path:
    """Persistence directory — extracted so tests can monkeypatch.

    Delegates to ``get_sessions_dir()`` so it honors ``$CLAWCODEX_CONFIG_DIR``
    (default ``~/.clawcodex/sessions``).
    """
    return get_sessions_dir()


def build_cost_block() -> dict[str, Any]:
    """Snapshot the bootstrap cost counters as the persisted ``cost`` block.

    ch03 round-4 GAP B — single schema owner, colocated with the reader
    below so writer/reader can't drift. Writers: the live agent-server
    persister (``_save_session``) and the legacy ``Session.save`` (which
    delegates here).
    """
    import time

    from src.bootstrap.state import (
        get_model_usage,
        get_start_time,
        get_total_api_duration,
        get_total_api_duration_without_retries,
        get_total_cost_usd,
        get_total_lines_added,
        get_total_lines_removed,
        get_total_tool_duration,
    )

    model_usage = get_model_usage()
    # List-price cost estimate, ALWAYS computed (even under a subscription,
    # where ``cost_usd`` is $0 because plan allowance is consumed rather than
    # metered credits — see cost_tracker.record_api_usage's billing_mode
    # gate). This is an OBSERVABILITY figure — what the tokens would have
    # cost at metered API rates — mirroring Claude Code, which reports a
    # non-zero cost on subscription runs. It never feeds the live ``/cost``
    # display or budget gate (those keep reading the billed ``total_cost_usd``
    # / ``cost_usd``); it exists so downstream trajectory/leaderboard tooling
    # has a comparable cost column.
    from src.services.pricing import compute_cost

    estimated_cost_usd = 0.0
    for model, u in model_usage.items():
        estimated_cost_usd += compute_cost(
            model,
            {
                "input_tokens": u.input_tokens,
                "output_tokens": u.output_tokens,
                "cache_creation_input_tokens": u.cache_creation_input_tokens,
                "cache_read_input_tokens": u.cache_read_input_tokens,
            },
        )

    return {
        "total_cost_usd": get_total_cost_usd(),
        "estimated_cost_usd": estimated_cost_usd,
        "total_api_duration": get_total_api_duration(),
        "total_api_duration_without_retries": get_total_api_duration_without_retries(),
        "total_tool_duration": get_total_tool_duration(),
        "total_lines_added": get_total_lines_added(),
        "total_lines_removed": get_total_lines_removed(),
        # last_duration = elapsed since start_time. The restore reader
        # back-dates the new session's start_time so post-resume duration
        # accumulators continue from where they left off.
        "last_duration": time.time() - get_start_time(),
        "model_usage": {
            model: {
                "input_tokens": u.input_tokens,
                "output_tokens": u.output_tokens,
                "cache_creation_input_tokens": u.cache_creation_input_tokens,
                "cache_read_input_tokens": u.cache_read_input_tokens,
                "cost_usd": u.cost_usd,
            }
            for model, u in model_usage.items()
        },
    }


def _restore_from_cost_block(cost_block: dict[str, Any]) -> None:
    """Dispatch a ``cost_block`` dict into the bootstrap singleton.

    Returns True if the snapshot was found and applied, False otherwise.

    Mirrors TS ``restoreCostStateForSession`` semantics: the gate is the
    **persisted file's session_id**, not the bootstrap singleton's
    runtime session_id. This means the function works regardless of
    whether ``switch_session(sid)`` was called first — the resume path
    can call restore-then-switch or switch-then-restore.

    The on-disk location is ``~/.clawcodex/sessions/<sid>.json`` —
    the same place ``Session.save`` writes. ``Session.save`` persists a
    ``cost`` block since ch03 round-2 R2.1 (``agent/session.py:50-73``);
    the missing-field tolerance below remains for snapshots written by
    pre-R2.1 builds.
    """
    model_usage_raw: dict[str, Any] = cost_block.get("model_usage", {}) or {}
    model_usage: dict[str, ModelUsage] = {}
    for model, entry in model_usage_raw.items():
        if not isinstance(entry, dict):
            continue
        model_usage[model] = ModelUsage(
            input_tokens=int(entry.get("input_tokens", 0)),
            output_tokens=int(entry.get("output_tokens", 0)),
            cache_creation_input_tokens=int(entry.get("cache_creation_input_tokens", 0)),
            cache_read_input_tokens=int(entry.get("cache_read_input_tokens", 0)),
            cost_usd=float(entry.get("cost_usd", 0.0)),
        )

    set_cost_state_for_restore(
        total_cost_usd=float(cost_block.get("total_cost_usd", 0.0)),
        total_api_duration=int(cost_block.get("total_api_duration", 0)),
        total_api_duration_without_retries=int(cost_block.get("total_api_duration_without_retries", 0)),
        total_tool_duration=int(cost_block.get("total_tool_duration", 0)),
        total_lines_added=int(cost_block.get("total_lines_added", 0)),
        total_lines_removed=int(cost_block.get("total_lines_removed", 0)),
        last_duration=cost_block.get("last_duration"),
        model_usage=model_usage if model_usage else None,
    )


def _restore_from_jsonl_tail(session_id: str) -> bool:
    """F-49 P5-C: read the trailing cost line from ``transcript.jsonl``.

    Walks ``~/.clawcodex/sessions/<sid>/transcript.jsonl`` once and
    remembers the LAST line whose ``type`` is either ``session_snapshot``
    (new P5-A format, written by :meth:`Session.save`) or ``cost_block``
    (legacy format, written by pre-P5-E ``session_persist``). The
    trailing cost line is the snapshot of record — successive saves
    append additional lines, but ``cost_restore`` keys on the latest
    one because that is what reflects the current cost counters.

    Returns True if a cost line was found and applied, False otherwise.
    """
    transcript_path = _sessions_dir() / session_id / "transcript.jsonl"
    if not transcript_path.exists():
        return False

    last_cost_block: dict[str, Any] | None = None
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
                if not isinstance(entry, dict):
                    continue
                entry_type = entry.get("type")
                if entry_type not in ("session_snapshot", "cost_block"):
                    continue
                cost = entry.get("cost")
                if isinstance(cost, dict):
                    last_cost_block = cost
    except OSError:
        return False

    if last_cost_block is None:
        return False

    _restore_from_cost_block(last_cost_block)
    return True


def restore_cost_state_for_session(session_id: SessionId | str) -> bool:
    """Restore cost accumulators from the persisted snapshot for
    ``session_id``.

    Returns True if the snapshot was found and applied, False otherwise.

    Mirrors TS ``restoreCostStateForSession`` semantics: the gate is the
    **persisted file's session_id**, not the bootstrap singleton's
    runtime session_id. This means the function works regardless of
    whether ``switch_session(sid)`` was called first — the resume path
    can call restore-then-switch or switch-then-restore.

    F-49 P5-C: primary source is the trailing ``session_snapshot`` /
    ``cost_block`` line in ``transcript.jsonl``. Falls back to
    ``session.json`` when no snapshot line is present (e.g. very new
    sessions or pre-Phase-5 saves that have not yet been migrated).
    """
    target = str(session_id)
    transcript_path = _sessions_dir() / target / "transcript.jsonl"

    # F-49 P5-C: prefer the transcript tail. This is the path taken
    # by ``Session.save()`` after Phase 5 — every save appends a
    # ``session_snapshot`` line, and cost_restore picks the latest.
    if transcript_path.exists() and _restore_from_jsonl_tail(target):
        return True

    # Legacy fallback: pre-Phase-5 ``session.json`` snapshot.
    session_file = _sessions_dir() / target / "session.json"
    if session_file.exists():
        try:
            data = json.loads(session_file.read_text())
        except (OSError, json.JSONDecodeError):
            return False

        if not isinstance(data, dict):
            return False

        # Gate on the *persisted* session_id matching the target — mirrors
        # the TS pattern. Refuses to restore from a file whose session_id
        # header doesn't agree with the filename (defends against a renamed
        # or hand-edited file).
        persisted_sid = data.get("session_id")
        if persisted_sid != target:
            return False

        # Extract cost fields with defaults — tolerate snapshots that
        # don't yet persist them.
        cost_block: dict[str, Any] = data.get("cost", {}) if isinstance(data, dict) else {}
        _restore_from_cost_block(cost_block)
        return True

    return False


__all__ = ["restore_cost_state_for_session"]
