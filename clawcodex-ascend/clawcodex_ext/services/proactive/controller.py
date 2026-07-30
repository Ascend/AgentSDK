from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable

from .constants import (
    CONTEXT_BLOCKED_TTL_SEC,
    DEFAULT_FOCUS_LEVEL,
    FOCUS_LEVELS,
    MAX_LAST_TICK_SUMMARY_CHARS,
)
from .state import AutomationState, FocusLevel

logger = logging.getLogger(__name__)

StateListener = Callable[[AutomationState], None]


def _now_ms() -> float:
    return time.time() * 1000


class ProactiveController:
    def __init__(self, *, clock_ms: Callable[[], float] | None = None) -> None:
        self._lock = threading.RLock()
        self._clock_ms = clock_ms or _now_ms
        self._state = AutomationState(phase="inactive")
        self._listeners: list[StateListener] = []

    @property
    def state(self) -> AutomationState:
        with self._lock:
            self._refresh_blocked_locked()
            self._refresh_sleep_locked()
            return self._state

    def is_active(self) -> bool:
        return self.state.is_active

    def is_paused(self) -> bool:
        return self.state.phase == "paused"

    def is_context_blocked(self) -> bool:
        return self.state.phase == "blocked"

    def should_tick(self) -> bool:
        return self.state.phase == "active"

    def activate(self, source: str = "unknown", *, focus: FocusLevel | None = None) -> None:
        with self._lock:
            if self._state.phase != "inactive":
                if focus is not None:
                    self._replace_locked(focus=self._validate_focus(focus))
                return
            self._state = AutomationState(
                phase="active",
                activation_source=source,
                focus=self._validate_focus(focus or DEFAULT_FOCUS_LEVEL),
            )
        self._notify()

    def deactivate(self) -> None:
        with self._lock:
            self._state = AutomationState(phase="inactive", focus=self._state.focus)
        self._notify()

    def pause(self) -> None:
        with self._lock:
            if self._state.phase != "active":
                return
            self._replace_locked(phase="paused", next_tick_at=None)
        self._notify()

    def resume(self) -> None:
        with self._lock:
            if self._state.phase not in ("paused", "blocked"):
                return
            self._replace_locked(phase="active", blocked_until=None)
        self._notify()

    def set_context_blocked(self, blocked: bool) -> None:
        with self._lock:
            if blocked:
                self._replace_locked(
                    phase="blocked",
                    blocked_until=self._clock_ms() + CONTEXT_BLOCKED_TTL_SEC * 1000,
                    next_tick_at=None,
                )
            elif self._state.phase == "blocked":
                self._replace_locked(phase="active", blocked_until=None)
            else:
                return
        self._notify()

    def set_next_tick_at(self, ts_ms: float | None) -> None:
        with self._lock:
            self._replace_locked(next_tick_at=ts_ms)
        self._notify()

    def set_focus(self, focus: FocusLevel) -> None:
        with self._lock:
            self._replace_locked(focus=self._validate_focus(focus))
        self._notify()

    def enter_sleep(self, until_ms: float) -> None:
        with self._lock:
            if self._state.phase == "inactive":
                return
            self._replace_locked(
                phase="sleeping",
                last_sleep_until=until_ms,
                next_tick_at=until_ms,
            )
        self._notify()

    def wake_from_sleep(self) -> None:
        with self._lock:
            if self._state.phase != "sleeping":
                return
            self._replace_locked(phase="active", last_sleep_until=None)
        self._notify()

    def record_tick(
        self,
        *,
        at_ms: float | None = None,
        summary: str | None = None,
    ) -> AutomationState:
        cleaned_summary = self._clean_summary(summary)
        with self._lock:
            self._replace_locked(
                tick_count=self._state.tick_count + 1,
                last_tick_at_ms=at_ms if at_ms is not None else self._clock_ms(),
                last_tick_summary=cleaned_summary,
            )
            state = self._state
        self._notify()
        return state

    def subscribe(self, listener: StateListener) -> Callable[[], None]:
        if not callable(listener):
            raise TypeError("listener must be callable")
        with self._lock:
            self._listeners.append(listener)

        def unsubscribe() -> None:
            self.unsubscribe(listener)

        return unsubscribe

    def unsubscribe(self, listener: StateListener) -> None:
        with self._lock:
            try:
                self._listeners.remove(listener)
            except ValueError:
                return

    def _refresh_blocked_locked(self) -> None:
        if (
            self._state.phase == "blocked"
            and self._state.blocked_until is not None
            and self._clock_ms() >= self._state.blocked_until
        ):
            self._replace_locked(phase="active", blocked_until=None)

    def _refresh_sleep_locked(self) -> None:
        if (
            self._state.phase == "sleeping"
            and self._state.last_sleep_until is not None
            and self._clock_ms() >= self._state.last_sleep_until
        ):
            self._replace_locked(phase="active", last_sleep_until=None)

    def _replace_locked(self, **updates: object) -> None:
        data = self._state.to_dict()
        data.update(updates)
        self._state = AutomationState(**data)  # type: ignore[arg-type]

    def _notify(self) -> None:
        with self._lock:
            listeners = list(self._listeners)
            state = self._state
        for listener in listeners:
            try:
                listener(state)
            except Exception:
                logger.exception("proactive state listener failed")

    @staticmethod
    def _validate_focus(focus: str) -> FocusLevel:
        if focus not in FOCUS_LEVELS:
            raise ValueError(f"focus must be one of: {', '.join(FOCUS_LEVELS)}")
        return focus  # type: ignore[return-value]

    @staticmethod
    def _clean_summary(summary: str | None) -> str | None:
        if not summary:
            return None
        text = " ".join(str(summary).split())
        if len(text) > MAX_LAST_TICK_SUMMARY_CHARS:
            text = text[: MAX_LAST_TICK_SUMMARY_CHARS - 3].rstrip() + "..."
        return text


_DEFAULT_CONTROLLER: ProactiveController | None = None
_DEFAULT_LOCK = threading.Lock()


def get_default_controller() -> ProactiveController:
    global _DEFAULT_CONTROLLER
    if _DEFAULT_CONTROLLER is None:
        with _DEFAULT_LOCK:
            if _DEFAULT_CONTROLLER is None:
                _DEFAULT_CONTROLLER = ProactiveController()
    return _DEFAULT_CONTROLLER


def reset_default_controller_for_tests() -> ProactiveController:
    global _DEFAULT_CONTROLLER
    with _DEFAULT_LOCK:
        _DEFAULT_CONTROLLER = ProactiveController()
        return _DEFAULT_CONTROLLER
