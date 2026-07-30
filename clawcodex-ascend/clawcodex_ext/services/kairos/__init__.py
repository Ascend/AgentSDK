"""Kairos / Brief scheduling service layer (F-86 first iteration).

This package provides the core primitives for the F-86 feature slice:

* :mod:`models` — :class:`TickConfig`, :class:`TickEvent`,
  :class:`BriefSummarySnapshot`, and :class:`DailyLogEntry` dataclasses
  with strict validation.
* :mod:`scheduler` — :class:`TickScheduler`, a background thread that
  fires :class:`TickEvent`s at a configured interval with optional
  jitter, pause / resume, and clean shutdown. The daemon thread
  lifecycle is inherited from :class:`PeriodicDaemon` so it is shared
  with :mod:`src.services.swarm.mailbox_poller`.
* :mod:`brief` — :class:`BriefSummaryBuilder`, a deterministic
  Markdown summary builder from a :class:`BriefSummarySnapshot`.
  No LLM dependency. Renamed from the original ``BriefGenerator`` to
  disambiguate from the user-facing :class:`BriefTool` in
  :mod:`src.tool_system.tools.brief`.
* :mod:`daily_log` — :class:`DailyLogWriter` for append-only writes
  to ``logs/YYYY/MM/YYYY-MM-DD.md``. The canonical path comes from
  :func:`src.memdir.paths.get_auto_mem_daily_log_path`; this module
  no longer duplicates that helper.

The package deliberately has **no** upstream dependency on
``src.brridge``, ``src.memdir``, or any agent runtime code. Higher-level
layers (CLI ``/tick on|off|status``, ``/brief``, ``SleepTool``, tick
message injection) wrap these primitives and ship in follow-up rounds.
"""

from __future__ import annotations

from ..periodic import PeriodicDaemon
from .brief import BriefSummaryBuilder
from .daily_log import DailyLogWriter
from .exceptions import (
    BriefGenerationError,
    DailyLogError,
    KairosError,
    SchedulerStateError,
    TickConfigError,
)
from .models import (
    BriefSummarySnapshot,
    DailyLogEntry,
    TickConfig,
    TickEvent,
    format_local_timestamp,
)
from .scheduler import TickCallback, TickScheduler

__all__ = [
    "BriefGenerationError",
    "BriefSummaryBuilder",
    "BriefSummarySnapshot",
    "DailyLogEntry",
    "DailyLogError",
    "DailyLogWriter",
    "KairosError",
    "PeriodicDaemon",
    "SchedulerStateError",
    "TickCallback",
    "TickConfig",
    "TickConfigError",
    "TickEvent",
    "TickScheduler",
    "format_local_timestamp",
]
