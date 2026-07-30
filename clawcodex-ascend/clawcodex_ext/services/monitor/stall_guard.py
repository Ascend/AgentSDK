"""Stall-watchdog exemption for monitor tasks.

CCB ``LocalShellTask.tsx:64`` skips the stall watchdog when ``kind === 'monitor'``
because a ``tail -f`` or ``watch`` task is intentionally long-running.  This
module provides the predicate used by that exemption hook.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


class StallWatchdogExemptor:
    """Predicate that decides whether a task should skip stall detection."""

    @classmethod
    def should_skip_stall_check(cls, state: Any) -> bool:
        """Return True for ``kind='monitor'`` tasks.

        The function is tolerant of non-``LocalShellTaskState`` objects and
        of states that pre-date the ``kind`` field (treat them as ``'shell'``).
        """
        if isinstance(state, Mapping):
            kind = state.get("kind", "shell")
        else:
            kind = getattr(state, "kind", "shell")
        return kind == "monitor"


__all__ = ["StallWatchdogExemptor"]
