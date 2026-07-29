"""Session-scoped in-memory buckets for pos-converter SDK wrapper instances.

Each pos-converter wrapper script maintains a module-level ``_instances`` dict
that maps constructor cache keys to SDK class instances.  When tools run in
separate bash subprocesses that dict is recreated empty on every call; this
registry keeps one shared dict per (session, agent, wrapper script) inside the
clawcodex main process so stateful SDK objects survive across tool invocations.
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

BucketKey = tuple[str, str, str]


class SdkInstanceRegistry:
    """In-memory ``bucket_key -> _instances`` map for SDK wrapper scripts."""

    def __init__(self) -> None:
        self._buckets: dict[BucketKey, dict[Any, Any]] = {}
        self._locks: dict[BucketKey, threading.RLock] = {}
        self._meta_lock = threading.RLock()

    def bucket_key(
        self,
        *,
        session_id: str | None,
        agent_id: str | None,
        script_path: str | Path,
    ) -> BucketKey:
        return (
            session_id or "__default__",
            agent_id or "",
            str(Path(script_path).resolve()),
        )

    def get_bucket(self, key: BucketKey) -> dict[Any, Any]:
        with self._meta_lock:
            bucket = self._buckets.get(key)
            if bucket is None:
                bucket = {}
                self._buckets[key] = bucket
            return bucket

    def lock_for(self, key: BucketKey) -> threading.RLock:
        with self._meta_lock:
            lock = self._locks.get(key)
            if lock is None:
                lock = threading.RLock()
                self._locks[key] = lock
            return lock

    def clear_session(self, session_id: str) -> None:
        """Drop all buckets belonging to *session_id*."""
        prefix = (session_id, "", "")
        with self._meta_lock:
            drop = [k for k in self._buckets if k[0] == prefix[0]]
            for key in drop:
                self._buckets.pop(key, None)
                self._locks.pop(key, None)

    def reset(self) -> None:
        """Clear all buckets (primarily for tests)."""
        with self._meta_lock:
            self._buckets.clear()
            self._locks.clear()


_registry: SdkInstanceRegistry | None = None
_registry_lock = threading.Lock()


def get_sdk_instance_registry() -> SdkInstanceRegistry:
    global _registry
    if _registry is None:
        with _registry_lock:
            if _registry is None:
                _registry = SdkInstanceRegistry()
    return _registry


def reset_sdk_instance_registry() -> SdkInstanceRegistry:
    """Replace the process singleton (tests)."""
    global _registry
    with _registry_lock:
        _registry = SdkInstanceRegistry()
        return _registry
