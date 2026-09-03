"""Reporter Protocol and the fan-out composite.

Concrete reporters include :class:`LocalFileReporter`,
:class:`DryRunReporter`, and :class:`IssueReporter` implementations.

The Protocol intentionally matches the shape used by
``extensions/orchestrator/progress_sink.py:CompositeProgressSink`` —
fan-out with per-reporter exception isolation, never raise out of
``emit`` / ``render``.
"""

from __future__ import annotations

import logging
from typing import Any, Iterable, Protocol, runtime_checkable

logger = logging.getLogger(__name__)


@runtime_checkable
class Reporter(Protocol):
    """A consumer of daily summaries.

    Implementations MUST NOT raise out of ``emit``; a failure is
    reported via the return value and logged. ``render`` is a pure
    function over the summary dict and the date string.
    """

    def render(self, summary: dict[str, Any], date: str) -> str: ...

    def emit(self, rendered: str, *, date: str) -> bool: ...


class CompositeReporter:
    """Synchronous fan-out over a list of :class:`Reporter` consumers.

    Each reporter runs in registration order. An exception raised by
    one reporter is logged and the remaining reporters still receive
    the event; ``emit`` always returns ``True`` as long as at least one
    reporter succeeded.
    """

    def __init__(self, reporters: Iterable[Reporter] = ()) -> None:
        self._reporters: list[Reporter] = list(reporters)

    def add(self, reporter: Reporter) -> None:
        self._reporters.append(reporter)

    def __len__(self) -> int:
        return len(self._reporters)

    def __iter__(self):
        return iter(self._reporters)

    def render(self, summary: dict[str, Any], date: str) -> str:
        """Render once, then fan-out the rendered text.

        If a reporter overrides ``render`` with a custom format, the
        composite still calls each reporter's own ``render`` so each
        consumer can format independently. The aggregated text
        returned by the composite is the first non-empty render to
        keep callers (CLI ``preview``) deterministic.
        """
        first_render = ""
        for reporter in list(self._reporters):
            method = getattr(reporter, "render", None)
            if method is None:
                continue
            try:
                rendered = method(summary, date)
            except Exception as exc:  # noqa: BLE001
                logger.exception("reporter %r render failed: %s", reporter, exc)
                continue
            if rendered and not first_render:
                first_render = rendered
        return first_render

    def emit(self, rendered: str, *, date: str) -> bool:
        any_ok = False
        for reporter in list(self._reporters):
            method = getattr(reporter, "emit", None)
            if method is None:
                continue
            try:
                ok = method(rendered, date=date)
            except Exception as exc:  # noqa: BLE001
                logger.exception("reporter %r emit failed: %s", reporter, exc)
                continue
            if ok:
                any_ok = True
        return any_ok
