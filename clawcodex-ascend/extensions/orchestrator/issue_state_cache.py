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

"""Per-session cache for ``AgentRunner._should_continue`` polling.

Decouples the skip decision from the agent runner so the orchestrator
can call :class:`IssueStateCache` independently and so unit tests can
exercise the cache without spinning up the full agent loop.

The cache is *per-session*: each :class:`extensions.orchestrator.agent_runner.AgentSession`
gets its own instance at construction time. Two concurrent sessions
never share state. The cache is also *per-issue-id*: a single session
running against one issue maintains a single history list.

Skip policy
-----------
``should_skip_poll(issue_id, turn)`` returns ``True`` iff ALL of:

- ``stable_skip_turns > 0`` (configurable knob — set to 0 disables the
  cache entirely and the runner always polls);
- the last ``stable_skip_turns`` recorded snapshots are all
  ``is_active=True``;
- the last ``stable_skip_turns`` recorded snapshots all share the same
  ``state`` value;
- those snapshots span ``stable_skip_turns`` *consecutive* turns
  ending at ``turn - 1`` (gaps force a re-poll — defensive against
  silent polling skips caused by upstream 429 backoff or other turn
  pauses);
- :meth:`has_recent_inactive` is ``False`` for ``turn - 1`` (an
  inactive issue must always be re-confirmed before declaring the
  loop done).

Invalidation
------------
:meth:`invalidate` clears entries. Callers should invoke it from:

- agent_runner user-interrupt handlers;
- external state mutation callbacks (operator label, manual issue
  close, etc.);
- whenever a freshly-fetched snapshot reports ``is_active=False``
  (the next call must re-confirm anyway).

Thread safety
-------------
The cache is mutated only from the orchestrator's async task driving
the session. ``asyncio`` serialises that single task, so no lock is
needed. Concurrent reads from logging paths are safe because the data
structures are simple dict/list operations.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _IssueSnapshot:
    """One observed poll result for an issue."""

    issue_id: str
    is_active: bool
    state: Optional[str]
    observed_at_turn: int


@dataclass
class IssueStateCache:
    """Skip the tracker poll when issue state has been stable for N turns.

    Default ``stable_skip_turns=3``: three consecutive identical active
    polls are enough evidence that the issue is stable, so subsequent
    ``_should_continue`` calls return the cached active state without
    issuing the tracker HTTP request.
    """

    stable_skip_turns: int = 3
    # Cap history length to bound memory; the skip decision only ever
    # looks at the tail, so trimming the head is safe.
    _max_history: int = 64
    _snapshots: dict[str, list[_IssueSnapshot]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Normalise ``stable_skip_turns`` to a non-negative integer."""
        # Normalise the knob to a non-negative integer. Negative values
        # would otherwise silently disable the cache, which is a foot-gun.
        self.stable_skip_turns = max(0, int(self.stable_skip_turns))

    # ------------------------------------------------------------------
    # Skip decision
    # ------------------------------------------------------------------

    def should_skip_poll(self, issue_id: str, turn: int) -> bool:
        """Return ``True`` if the tracker poll can be skipped at ``turn``."""
        if self.stable_skip_turns <= 0:
            return False

        history = self._snapshots.get(issue_id)
        if history is None or len(history) < self.stable_skip_turns:
            return False

        recent = history[-self.stable_skip_turns :]

        if not all(s.is_active for s in recent):
            return False

        # All recent snapshots must report the same state value. The set
        # collapses duplicates; size != 1 means we saw at least two distinct
        # states, which is precisely what "unstable" looks like.
        states = {s.state for s in recent}
        if len(states) != 1:
            return False

        # Defensive: the polling cadence might have skipped a turn (e.g.
        # a 429 backoff delayed _should_continue by a turn). If the
        # window does not cover ``stable_skip_turns`` consecutive turns
        # ending at ``turn - 1``, force a re-poll.
        last_turn = recent[-1].observed_at_turn
        if last_turn != turn - 1:
            return False
        first_turn = recent[0].observed_at_turn
        if first_turn != last_turn - self.stable_skip_turns + 1:
            return False

        return True

    def has_recent_inactive(self, issue_id: str, turn: int) -> bool:
        """Return ``True`` if the snapshot at ``turn`` reported inactive.

        Used by ``_should_continue`` as a forced-poll condition: an
        issue that was inactive on the previous poll must always be
        re-confirmed before continuing.
        """
        history = self._snapshots.get(issue_id)
        if not history:
            return False
        last = history[-1]
        return (not last.is_active) and last.observed_at_turn == turn

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def record(
        self,
        *,
        issue_id: str,
        is_active: bool,
        state: Optional[str],
        observed_at_turn: int,
    ) -> None:
        """Append a freshly-fetched snapshot to the issue's history.

        Trims the head when the history exceeds ``_max_history``. The
        trim is best-effort — even if it dropped a snapshot the skip
        decision would have used, the next call recomputes from the
        current tail so the result remains correct.
        """
        history = self._snapshots.setdefault(issue_id, [])
        history.append(
            _IssueSnapshot(
                issue_id=issue_id,
                is_active=is_active,
                state=state,
                observed_at_turn=int(observed_at_turn),
            )
        )
        if len(history) > self._max_history:
            del history[: -self._max_history]

    def invalidate(self, issue_id: Optional[str] = None) -> None:
        """Drop cached snapshots.

        With ``issue_id=None`` the entire cache is cleared (used on
        session reset). With a specific id, only that id's history is
        cleared (used on external mutation callbacks).
        """
        if issue_id is None:
            self._snapshots.clear()
        else:
            self._snapshots.pop(issue_id, None)

    # ------------------------------------------------------------------
    # Observability
    # ------------------------------------------------------------------

    def stats(self) -> dict[str, int]:
        """Return a snapshot of cache sizes for debugging and logs."""
        return {
            "tracked_issues": len(self._snapshots),
            "total_snapshots": sum(len(h) for h in self._snapshots.values()),
        }


__all__ = ["IssueStateCache"]
