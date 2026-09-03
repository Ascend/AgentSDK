"""RecordableSource registry — mirrors the dashboard source pattern.

A :class:`RecordableSourceRegistry` maps ``source_id`` strings to
factories that produce a :class:`~extensions.capabilities.recorder.RecordableSource`
instance bound to one :class:`AsciicastCapture`.

The :mod:`extensions.recording.cli` module reads the registry to know
which subsystems to enable when ``clawcodex record --sources a,b,c``
is invoked.

There is one process-wide default registry; tests can build their own
private registry to avoid cross-test contamination.
"""

from __future__ import annotations

import threading
from typing import Callable, Iterator, Optional

from extensions.capabilities.recorder import (
    AsciicastCapture,
    RecordableSource,
)

__all__ = [
    "RecordableSourceRegistry",
    "get_default_registry",
    "register_source",
    "reset_default_registry",
]


# A factory is a callable that takes the capture handle and returns a
# configured source. Factories defer source construction so we can
# build the capture first (and decide the file path) before any
# subsystem starts emitting events.
SourceFactory = Callable[[AsciicastCapture], RecordableSource]


class RecordableSourceRegistry:
    """Thread-safe mapping of ``source_id`` -> :class:`SourceFactory`."""

    def __init__(self) -> None:
        self._factories: dict[str, SourceFactory] = {}
        self._lock = threading.RLock()

    def register(self, source_id: str, factory: SourceFactory) -> None:
        """Register or replace a factory under ``source_id``."""
        norm = source_id.strip().lower()
        if not norm:
            raise ValueError("source_id must be non-empty")
        with self._lock:
            self._factories[norm] = factory

    def unregister(self, source_id: str) -> bool:
        norm = source_id.strip().lower()
        with self._lock:
            return self._factories.pop(norm, None) is not None

    def get(self, source_id: str) -> Optional[SourceFactory]:
        norm = source_id.strip().lower()
        with self._lock:
            return self._factories.get(norm)

    def has(self, source_id: str) -> bool:
        norm = source_id.strip().lower()
        with self._lock:
            return norm in self._factories

    def names(self) -> list[str]:
        with self._lock:
            return sorted(self._factories.keys())

    def __iter__(self) -> Iterator[tuple[str, SourceFactory]]:
        with self._lock:
            return iter(list(self._factories.items()))

    def __len__(self) -> int:
        with self._lock:
            return len(self._factories)

    def clear(self) -> None:
        with self._lock:
            self._factories.clear()


_DEFAULT_REGISTRY: Optional[RecordableSourceRegistry] = None
_DEFAULT_REGISTRY_LOCK = threading.Lock()


def get_default_registry() -> RecordableSourceRegistry:
    """Return the process-wide :class:`RecordableSourceRegistry`."""
    global _DEFAULT_REGISTRY
    with _DEFAULT_REGISTRY_LOCK:
        if _DEFAULT_REGISTRY is None:
            _DEFAULT_REGISTRY = RecordableSourceRegistry()
        return _DEFAULT_REGISTRY


def register_source(source_id: str, factory: SourceFactory) -> None:
    """Register ``factory`` under ``source_id`` on the default registry."""
    get_default_registry().register(source_id, factory)


def reset_default_registry() -> None:
    """Drop the cached default registry. Test-only helper."""
    global _DEFAULT_REGISTRY
    with _DEFAULT_REGISTRY_LOCK:
        _DEFAULT_REGISTRY = None
