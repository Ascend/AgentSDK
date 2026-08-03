"""Cron adapter for asciicast recording (F-REC).

Wraps the four cron scheduler callbacks
(``on_fire_task`` / ``on_missed`` / ``on_fire_event`` /
``on_expired_event`` from ``clawcodex_ext/cron_system/runtime.py:113-126``)
to mirror cron lifecycle into one asciicast capture.

Per the F-REC decision, **each cron fire produces its own .cast file**
(per-fire granularity), so this observer is short-lived: the recording
CLI opens a capture, fires the scheduler, closes the capture, and the
per-run ``.cast`` lives next to the existing ``CronRun`` JSON record.
"""

from __future__ import annotations

import logging
from typing import Any

from extensions.capabilities.recorder import AsciicastCapture
from extensions.recording.renderers import format_cron_event

logger = logging.getLogger(__name__)


class AsciicastCronObserver:
    """Adapts :class:`CronScheduler` callbacks to an asciicast capture.

    The observer is constructed with a single :class:`AsciicastCapture`
    and exposes four callback methods that match the
    ``CronScheduler.__init__`` parameter names. The recording CLI
    constructs one observer per capture and passes the four methods
    into the scheduler.
    """

    def __init__(self, capture: AsciicastCapture) -> None:
        self._capture = capture

    # -- callbacks -------------------------------------------------------

    def on_fire_task(self, task: Any, run: Any) -> None:
        self._emit(
            label=f"cron:fire:{getattr(task, 'id', '?')}",
            payload={
                "task_id": getattr(task, "id", "?"),
                "cron": getattr(task, "cron", "?"),
                "status": "fired",
                "run_id": getattr(run, "id", None),
            },
        )

    def on_missed(self, tasks: Any, notification: str) -> None:
        ids = [getattr(t, "id", "?") for t in (tasks or [])]
        self._emit(
            label=f"cron:missed:{','.join(ids)}",
            payload={
                "task_id": ",".join(ids) or "?",
                "cron": "?",
                "status": "missed",
                "notification": notification,
            },
        )

    def on_fire_event(self, payload: dict[str, Any]) -> None:
        # ``on_fire_event`` is the generic event sink used by the
        # existing ``_log_event`` debug logger — we replace it with a
        # recording-aware sink when capture is wired.
        if not isinstance(payload, dict):
            return
        self._emit(
            label=f"cron:event:{payload.get('status', '?')}",
            payload=payload,
        )

    def on_expired_event(self, payload: dict[str, Any]) -> None:
        if not isinstance(payload, dict):
            return
        self._emit(
            label=f"cron:expired:{payload.get('task_id', '?')}",
            payload=payload,
        )

    # -- internal --------------------------------------------------------

    def _emit(self, *, label: str, payload: dict[str, Any]) -> None:
        text = format_cron_event(payload)
        try:
            self._capture.marker(label, text=text)
        except Exception as exc:  # noqa: BLE001
            logger.warning("AsciicastCronObserver marker failed (label=%s): %s", label, exc)


# A factory shape compatible with the source-registry
# ``register_source(source_id, factory)`` pattern in
# ``extensions/recording/registry.py`` — so the CLI can look up the
# observer factory by source_id.
def make_cron_observer(capture: AsciicastCapture) -> AsciicastCronObserver:
    """Factory: build a fresh observer bound to ``capture``."""
    return AsciicastCronObserver(capture)


# Convenience: a no-op observer usable when capture is disabled but
# ``install_cron_hooks`` still expects a callback arg.
class _NullObserver:
    def on_fire_task(self, task: Any, run: Any) -> None:  # noqa: ARG002
        return

    def on_missed(self, tasks: Any, notification: str) -> None:  # noqa: ARG002
        return

    def on_fire_event(self, payload: dict[str, Any]) -> None:  # noqa: ARG002
        return

    def on_expired_event(self, payload: dict[str, Any]) -> None:  # noqa: ARG002
        return


def null_observer() -> _NullObserver:
    """Return a no-op observer for the recording-disabled path."""
    return _NullObserver()


__all__ = [
    "AsciicastCronObserver",
    "make_cron_observer",
    "null_observer",
]
