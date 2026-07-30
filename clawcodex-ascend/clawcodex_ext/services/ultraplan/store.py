"""JSON-on-disk plan store with atomic writes.

The store keeps one JSON file per plan under ``data_dir``. Writes are
atomic: data is written to a sibling temporary file first, then renamed
over the destination, so a crash mid-write cannot leave a half-written
plan on disk. The store is thread-safe; all public methods take an
``RLock`` and reload the in-memory copy on every load to surface writes
from other threads.

This module is intentionally dependency-free — no SQLite, no orjson —
so it works in restricted environments and in unit tests.
"""

from __future__ import annotations

import json
import os
import tempfile
import threading
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from .exceptions import PlanCorruptError, PlanNotFoundError
from .models import Plan, PlanStatus


def _default_plan_id_filename(plan_id: str) -> str:
    # Plan ids are validated to match ``[A-Za-z0-9._-]{1,64}`` so the
    # filename is always safe to use directly. We add a ``.json`` suffix.
    return f"{plan_id}.json"


class PlanStore:
    """Persist and retrieve :class:`Plan` instances.

    The store does not cache plans in memory: every ``load`` reads from
    disk and every ``save`` writes through to disk. This keeps the
    on-disk state authoritative and avoids stale-cache bugs at the cost
    of an extra disk read per operation.
    """

    def __init__(self, data_dir: Path | str) -> None:
        self._data_dir = Path(data_dir)
        self._lock = threading.RLock()
        self._data_dir.mkdir(parents=True, exist_ok=True)

    @property
    def data_dir(self) -> Path:
        return self._data_dir

    def _plan_path(self, plan_id: str) -> Path:
        return self._data_dir / _default_plan_id_filename(plan_id)

    def save(self, plan: Plan) -> Path:
        if not isinstance(plan, Plan):
            raise TypeError("PlanStore.save expects a Plan instance")
        with self._lock:
            target = self._plan_path(plan.id)
            payload = plan.to_dict()
            self._atomic_write(target, payload)
            return target

    def load(self, plan_id: str) -> Plan:
        if not isinstance(plan_id, str) or not plan_id:
            raise ValueError("plan_id must be a non-empty string")
        with self._lock:
            target = self._plan_path(plan_id)
            if not target.exists():
                raise PlanNotFoundError(f"plan {plan_id!r} not found at {target}")
            try:
                raw = target.read_text(encoding="utf-8")
                data: Any = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise PlanCorruptError(f"plan {plan_id!r} contains invalid JSON: {exc}") from exc
            except OSError as exc:
                raise PlanCorruptError(f"plan {plan_id!r} could not be read: {exc}") from exc
        try:
            return Plan.from_dict(data)
        except (ValueError, TypeError) as exc:
            raise PlanCorruptError(f"plan {plan_id!r} failed validation: {exc}") from exc

    def exists(self, plan_id: str) -> bool:
        with self._lock:
            return self._plan_path(plan_id).exists()

    def delete(self, plan_id: str) -> None:
        with self._lock:
            target = self._plan_path(plan_id)
            try:
                target.unlink()
            except FileNotFoundError:
                # Deleting a non-existent plan is a no-op, mirroring the
                # ``dict.pop``-style semantics used by the manager.
                return

    def list_plans(self) -> list[tuple[str, PlanStatus, str]]:
        with self._lock:
            out: list[tuple[str, PlanStatus, str]] = []
            for path in sorted(self._data_dir.glob("*.json")):
                try:
                    raw = path.read_text(encoding="utf-8")
                    data: Any = json.loads(raw)
                    plan = Plan.from_dict(data)
                except (json.JSONDecodeError, ValueError, TypeError, OSError):
                    # Skip corrupted files but do not raise; the caller can
                    # detect and clean them up if needed.
                    continue
                out.append((plan.id, plan.status, plan.title))
            return out

    def _atomic_write(self, target: Path, payload: dict[str, Any]) -> None:
        # Use a per-call temp file in the same directory so the final
        # ``os.replace`` is atomic on POSIX and Windows.
        fd, tmp_path = tempfile.mkstemp(
            prefix=f".{target.name}.",
            suffix=".tmp",
            dir=str(self._data_dir),
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(payload, fh, ensure_ascii=False, indent=2)
                fh.write("\n")
            os.replace(tmp_path, target)
        except Exception:
            # If anything fails, try to clean up the temp file.
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise

    @contextmanager
    def transaction(self) -> Iterator[None]:
        """Acquire the store's lock for a multi-call critical section.

        This is exposed for callers that want to make several load/save
        calls atomically w.r.t. other threads. The store is in-process,
        so this does not protect against cross-process writes.
        """
        with self._lock:
            yield


def safe_filename_suffix() -> str:
    """Return a short, low-collision suffix used in temp file names."""
    return uuid.uuid4().hex[:8]
