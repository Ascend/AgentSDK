"""Dry-run recorder for Computer Use actions.

The recorder never executes anything; it accumulates structured ``InputAction``
records (and a list of bytes for any generated screenshots) so that callers can
inspect what *would* have happened. This is the default safety mode for the
Linux backend in tests and in any environment that has not explicitly opted in
to real input simulation.
"""

from __future__ import annotations

import threading
from collections.abc import Iterable
from typing import Any

from .models import InputAction, ScreenRegion, WindowRef


class DryRunRecorder:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._actions: list[InputAction] = []
        self._screenshots: list[tuple[str, bytes | None, dict[str, Any]]] = []

    def record_action(self, kind: str, **args: Any) -> None:
        action = InputAction(kind=kind, args=dict(args))
        with self._lock:
            self._actions.append(action)

    def record_screenshot(
        self,
        kind: str,
        payload: bytes | None,
        **meta: Any,
    ) -> None:
        with self._lock:
            self._screenshots.append((kind, payload, dict(meta)))

    def actions(self) -> list[InputAction]:
        with self._lock:
            return list(self._actions)

    def screenshots(self) -> list[tuple[str, bytes | None, dict[str, Any]]]:
        with self._lock:
            return list(self._screenshots)

    def clear(self) -> None:
        with self._lock:
            self._actions.clear()
            self._screenshots.clear()

    @property
    def action_count(self) -> int:
        with self._lock:
            return len(self._actions)

    def __len__(self) -> int:
        return self.action_count

    def __bool__(self) -> bool:
        # A recorder is a real object even when empty. Without this override,
        # Python falls back to ``__len__`` and an empty recorder would be
        # falsy, which breaks the ``recorder or DryRunRecorder()`` idiom
        # used by the provider constructors.
        return True

    def filter(self, kind: str) -> Iterable[InputAction]:
        with self._lock:
            return [action for action in self._actions if action.kind == kind]


def region_to_args(region: ScreenRegion) -> dict[str, Any]:
    return region.to_dict()


def window_to_args(window: WindowRef) -> dict[str, Any]:
    return window.to_dict()
