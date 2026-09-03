"""Local JSONL storage for telemetry.

The store is fire-and-forget: every write is wrapped in a
``try/except OSError`` and any failure is logged but never raised. This
mirrors the precedent set by
``extensions/orchestrator/state_journal.py:StateJournalWriter.write_event``.

Three directories are managed by default:

* ``events/``     — one JSONL per UTC date, holds ``session_start`` /
  ``session_end`` / ``command_run`` / ``tool_summary`` / ``error`` events.
* ``crashes/``    — one JSONL per UTC date, holds ``crash`` events with
  fingerprint and truncated stacktrace.
* ``summaries/``  — one JSON per UTC date, the aggregator output.
* ``reports/``    — markdown files emitted by ``LocalFileReporter``
  (created on demand).
* ``reporter_blocked/`` — JSONL rows describing reporter refusals
  (created on demand).
* ``reporter_errors/`` — JSONL rows describing remote reporter failures
  without storing rendered report bodies.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Final

logger = logging.getLogger(__name__)

_KINDS: Final[tuple[str, ...]] = (
    "events",
    "crashes",
    "summaries",
    "reports",
    "reporter_blocked",
    "reporter_errors",
    "reporter_cursors",
)


def utc_date(ts: float) -> str:
    """Return the ``YYYY-MM-DD`` UTC date for *ts*."""
    return time.strftime("%Y-%m-%d", time.gmtime(ts))


def utc_now() -> float:
    return time.time()


class LocalJsonlStorage:
    """Append-only fire-and-forget store with date rotation and retention."""

    def __init__(self, base_dir: Path, retention_days: int = 30) -> None:
        self._base_dir = Path(base_dir).expanduser()
        self._retention_days = max(1, int(retention_days))
        self._ensure_base()

    @property
    def base_dir(self) -> Path:
        return self._base_dir

    # -- layout ---------------------------------------------------------

    def _ensure_base(self) -> None:
        try:
            self._base_dir.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            logger.warning(
                "telemetry: cannot create storage dir %s: %s",
                self._base_dir,
                exc,
            )

    def _dir_for(self, kind: str) -> Path:
        if kind not in _KINDS:
            raise ValueError(f"unknown storage kind: {kind!r}")
        return self._base_dir / kind

    def _path_for(self, kind: str, date: str) -> Path:
        if kind == "summaries":
            return self._dir_for(kind) / f"{date}.json"
        if kind == "reports":
            return self._dir_for(kind) / f"{date}.md"
        if kind in {"reporter_blocked", "reporter_errors"}:
            return self._dir_for(kind) / f"{date}.jsonl"
        return self._dir_for(kind) / f"{date}.jsonl"

    # -- public API -----------------------------------------------------

    def append(
        self,
        kind: str,
        payload: dict[str, Any],
        *,
        date: str | None = None,
    ) -> bool:
        """Append *payload* (a single JSONL row) to ``<kind>/YYYY-MM-DD.jsonl``.

        When *date* is provided, the row is written under that date's
        file. When omitted, the current UTC date is used (legacy
        behavior for real-time event appends from the recorder).
        Returns ``True`` on success, ``False`` on any error. The
        surrounding caller MUST NOT treat a ``False`` as fatal.
        """
        if kind not in ("events", "crashes", "reporter_blocked", "reporter_errors"):
            raise ValueError(f"append() only writes JSONL kinds; got {kind!r}")
        effective_date = date or utc_date(utc_now())
        path = self._path_for(kind, effective_date)
        try:
            self._dir_for(kind).mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            logger.warning("telemetry: mkdir failed for %s: %s", kind, exc)
            return False
        try:
            line = json.dumps(payload, ensure_ascii=False, default=str) + "\n"
        except (TypeError, ValueError) as exc:
            logger.warning("telemetry: payload not JSON-encodable: %s", exc)
            return False
        try:
            with open(path, "a", encoding="utf-8") as f:
                f.write(line)
        except OSError as exc:
            logger.warning("telemetry: write failed for %s: %s", path, exc)
            return False
        return True

    def write_summary(self, date: str, summary: dict[str, Any]) -> bool:
        """Atomically write a daily summary JSON for *date*.

        Uses ``tmp + os.replace`` to avoid torn writes. Returns
        ``True`` on success.
        """
        path = self._path_for("summaries", date)
        try:
            self._dir_for("summaries").mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            logger.warning("telemetry: mkdir failed for summaries: %s", exc)
            return False
        tmp = path.with_suffix(".json.tmp")
        try:
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(summary, f, ensure_ascii=False, default=str, indent=2)
                f.write("\n")
            import os

            os.replace(tmp, path)
        except OSError as exc:
            logger.warning("telemetry: summary write failed: %s", exc)
            return False
        return True

    def read_day(self, kind: str, date: str) -> list[dict[str, Any]]:
        """Yield parsed JSONL rows for *kind* / *date*.

        Returns an empty list if the file does not exist or cannot be
        read. The aggregator is the only intended caller.
        """
        path = self._path_for(kind, date)
        if not path.exists():
            return []
        rows: list[dict[str, Any]] = []
        try:
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rows.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        except OSError as exc:
            logger.warning("telemetry: read failed for %s: %s", path, exc)
            return rows
        return rows

    def read_latest_summary(self, date: str) -> dict[str, Any] | None:
        path = self._path_for("summaries", date)
        if not path.exists():
            return None
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (OSError, json.JSONDecodeError):
            return None

    def list_dates(self, kind: str) -> list[str]:
        d = self._dir_for(kind)
        if not d.exists():
            return []
        out: list[str] = []
        for entry in d.iterdir():
            name = entry.name
            for suffix in (".jsonl", ".json", ".md"):
                if name.endswith(suffix):
                    out.append(name[: -len(suffix)])
                    break
        return sorted(out)

    def read_reporter_cursor(self, reporter_name: str) -> dict[str, Any]:
        path = self._cursor_path(reporter_name)
        if not path.exists():
            return {}
        try:
            with open(path, "r", encoding="utf-8") as f:
                payload = json.load(f)
        except (OSError, json.JSONDecodeError):
            return {}
        return payload if isinstance(payload, dict) else {}

    def write_reporter_cursor(self, reporter_name: str, cursor: dict[str, Any]) -> bool:
        path = self._cursor_path(reporter_name)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            logger.warning("telemetry: mkdir failed for reporter cursor: %s", exc)
            return False
        tmp = path.with_suffix(".json.tmp")
        try:
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(cursor, f, ensure_ascii=False, default=str, indent=2)
                f.write("\n")
            import os

            os.replace(tmp, path)
        except OSError as exc:
            logger.warning("telemetry: cursor write failed: %s", exc)
            return False
        return True

    def _cursor_path(self, reporter_name: str) -> Path:
        safe = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in reporter_name)
        return self._dir_for("reporter_cursors") / f"{safe or 'default'}.json"

    # -- retention ------------------------------------------------------

    def retention_sweep(self, now_ts: float | None = None) -> int:
        """Delete files older than ``self._retention_days``.

        Returns the number of files removed. Errors are logged and
        skipped.
        """
        now = now_ts if now_ts is not None else utc_now()
        cutoff_date = utc_date(now - self._retention_days * 86400)
        removed = 0
        for kind in _KINDS:
            d = self._dir_for(kind)
            if not d.exists():
                continue
            try:
                entries = list(d.iterdir())
            except OSError:
                continue
            for entry in entries:
                name = entry.stem
                if len(name) == 10 and name[:4].isdigit() and name < cutoff_date:
                    try:
                        entry.unlink()
                        removed += 1
                    except OSError as exc:
                        logger.debug("telemetry: retention remove failed: %s", exc)
        return removed
