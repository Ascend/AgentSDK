"""Typed dispatch bridge for cron fire events (F-22-G-2).

Replaces ad-hoc outbox-drain logic in REPL, headless, and TUI
frontends with a unified :class:`CronDispatchBridge`.

The bridge maps directly to ``CronPromptEvent`` (fire) and
``CronMissedEvent`` (missed one-shot) from
``clawcodex_ext/query/outbox_types.py``.  The ``drain()`` method
pops ``CronPromptEvent`` entries from the outbox; callers are
responsible for reading ``CronMissedEvent`` separately via
``drain_missed()``.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from clawcodex_ext.cron_system.runs import claim_cron_run, finalize_cron_run
from clawcodex_ext.query.outbox_types import CronMissedEvent, CronPromptEvent  # pylint: disable=no-name-in-module


@dataclass
class CronDispatchEvent:
    """A single cron fire event, ready for execution.

    Maps directly from a ``CronPromptEvent`` in the outbox.
    """

    prompt: str
    task_id: str
    run_id: str
    wrapped_prompt: str


@dataclass
class CronMissedDispatch:
    """A missed-task notification, ready for delivery.

    Maps from a ``CronMissedEvent`` in the outbox.
    """

    task_ids: list[str]
    notification: str


def _default_wrap_prompt(prompt: str, task_id: str, run_id: str) -> str:
    """Default prompt wrapper (mirrors headless ``_wrap_cron_prompt``)."""
    _ = run_id  # unused in default wrapper
    now = datetime.now()
    time_str = now.strftime("%b %d %-I:%M%p").lower()
    header = f"✻ Running scheduled task ({time_str})"
    if task_id:
        header += f" · {task_id}"
    return f"{header}\n\nThis prompt was generated automatically from a scheduled task.\n\n{prompt}"


def _entry_type(entry: Any) -> str:
    """Return the ``type`` discriminator for any outbox entry.

    Accepts typed :class:`CronPromptEvent` / :class:`CronMissedEvent`
    and legacy dict-style entries (``{"type": "cron_prompt", ...}``).
    Returns "" for entries that don't advertise a type.
    """
    if isinstance(entry, CronPromptEvent):
        return "cron_prompt"
    if isinstance(entry, CronMissedEvent):
        return "cron_missed"
    if isinstance(entry, dict):
        return entry.get("type", "")
    getter = getattr(entry, "get", None)
    if callable(getter):
        try:
            return getter("type", "")
        except Exception:
            return ""
    return ""


def _entry_field(entry: Any, key: str, default: Any = None) -> Any:
    """Read a field from typed or dict-style outbox entry."""
    if isinstance(entry, dict):
        return entry.get(key, default)
    if hasattr(entry, key):
        return getattr(entry, key)
    getter = getattr(entry, "get", None)
    if callable(getter):
        try:
            return getter(key, default)
        except Exception:
            return default
    return default


class CronDispatchBridge:
    """Typed dispatch bridge for cron fire events.

    Replaces ad-hoc outbox-drain logic in REPL, headless, and TUI
    frontends with a unified dispatcher.

    The ``drain()`` method pops ``CronPromptEvent`` entries from the
    outbox, filtering out any non-cron events that remain in the list.
    Callers are responsible for handling ``CronMissedEvent`` entries
    separately via :meth:`drain_missed`.

    Both typed events (``CronPromptEvent`` / ``CronMissedEvent``) and
    legacy dict-style entries are supported — the legacy code path
    (``{"type": "cron_prompt", ...}``) is preserved so existing
    pre-typed outbox producers keep working during migration.
    """

    def __init__(
        self,
        workspace_root: Path,
        wrap_prompt: Callable[[str, str, str], str] | None = None,
    ) -> None:
        self._workspace_root = workspace_root
        self._wrap_prompt = wrap_prompt or _default_wrap_prompt

    def drain(self, outbox: list) -> list[CronDispatchEvent]:
        """Pop ``cron_prompt`` entries from the outbox.

        Accepts typed ``CronPromptEvent`` and dict-style
        ``{"type": "cron_prompt", ...}``. Other entries
        (``CronMissedEvent`` / ``ProactivePromptEvent`` /
        ``GenericOutboxEvent`` / unknown dicts) are left in place for
        the caller to handle.

        Returns a list of :class:`CronDispatchEvent` ready for execution.
        """
        events: list[CronDispatchEvent] = []
        remaining: list = []
        for entry in outbox:
            if _entry_type(entry) == "cron_prompt":
                prompt = (_entry_field(entry, "prompt", "") or "").strip()
                task_id = str(_entry_field(entry, "task_id", "") or "")
                run_id = str(_entry_field(entry, "run_id", "") or "")
                if prompt:
                    events.append(
                        CronDispatchEvent(
                            prompt=prompt,
                            task_id=task_id,
                            run_id=run_id,
                            wrapped_prompt=self._wrap_prompt(prompt, task_id, run_id),
                        )
                    )
                    continue
                # Blank prompt — keep the entry in the outbox so the
                # caller can decide what to do (drop, log, surface).
                remaining.append(entry)
            else:
                remaining.append(entry)
        outbox[:] = remaining
        return events

    def drain_missed(self, outbox: list) -> list[CronMissedDispatch]:
        """Pop ``cron_missed`` entries from the outbox.

        Mirrors :meth:`drain` for missed-task notifications. Other
        entries are left in place. Returns a list of
        :class:`CronMissedDispatch` ready for delivery to the user.
        """
        events: list[CronMissedDispatch] = []
        remaining: list = []
        for entry in outbox:
            if _entry_type(entry) == "cron_missed":
                notification = (_entry_field(entry, "notification", "") or "").strip()
                tasks_field = _entry_field(entry, "tasks", []) or []
                task_ids = [str(t) for t in tasks_field]
                if notification:
                    events.append(
                        CronMissedDispatch(
                            task_ids=task_ids,
                            notification=notification,
                        )
                    )
                    continue
                # Blank notification — keep in outbox for caller decision.
                remaining.append(entry)
            else:
                remaining.append(entry)
        outbox[:] = remaining
        return events

    def claim(self, task_id: str, run_id: str) -> str | None:
        """Mark a run as started (queued → running).

        Returns ``task_id`` on success (the run was queued and is now
        running), or ``None`` if the run was already claimed or
        finalized. ``task_id`` is only used as the return value
        marker; ``runs.claim_cron_run`` keys on ``run_id`` alone.
        """
        claimed = claim_cron_run(self._workspace_root, run_id)
        return task_id if claimed is not None else None

    def finalize(
        self,
        task_id: str,  # noqa: ARG002
        run_id: str,
        status: str,
        error: str | None = None,
    ) -> None:
        """Mark a run as completed/failed/cancelled."""
        finalize_cron_run(self._workspace_root, run_id, status, error=error)
