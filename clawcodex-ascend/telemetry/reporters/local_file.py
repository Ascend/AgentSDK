"""LocalFileReporter — writes daily summaries to ``reports/YYYY-MM-DD.md``.

The reporter is intentionally write-only and never blocks the main
flow:

* I/O failures are logged and reported via the return value, not
  raised.
* A secret-scan pass runs against the rendered markdown before
  writing. If the scan hits, the file is **not** written and a row is
  appended to ``reporter_blocked/YYYY-MM-DD.jsonl`` so the operator
  can review.
"""

from __future__ import annotations

import logging
from typing import Any

from ..redaction import Redactor
from ..storage import LocalJsonlStorage, utc_date, utc_now
from .dry_run import _render_markdown

logger = logging.getLogger(__name__)


class LocalFileReporter:
    """Render summary to markdown, scan secrets, write to ``reports/``."""

    def __init__(
        self,
        storage: LocalJsonlStorage,
        redactor: Redactor,
    ) -> None:
        self._storage = storage
        self._redactor = redactor

    # -- Reporter protocol ---------------------------------------------

    def render(self, summary: dict[str, Any], date: str) -> str:
        return _render_markdown(summary, date)

    def emit(self, rendered: str, *, date: str) -> bool:
        # Secret scan gate — refuse the report if any pattern matches.
        hits = self._redactor.scan_secrets(rendered)
        if hits:
            logger.warning(
                "telemetry: report for %s blocked by secret scan: %s",
                date,
                hits,
            )
            self._storage.append(
                "reporter_blocked",
                {
                    "timestamp": utc_now(),
                    "date": date,
                    "kind": "local_file",
                    "reason": "secret_scan",
                    "patterns": hits,
                    "length": len(rendered),
                },
            )
            return False

        path = self._storage.base_dir / "reports" / f"{date}.md"
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            logger.warning("telemetry: cannot create reports dir: %s", exc)
            self._storage.append(
                "reporter_blocked",
                {
                    "timestamp": utc_now(),
                    "date": date,
                    "kind": "local_file",
                    "reason": "mkdir_failed",
                    "error": str(exc),
                },
            )
            return False

        tmp = path.with_suffix(".md.tmp")
        try:
            with open(tmp, "w", encoding="utf-8") as f:
                f.write(rendered)
                f.flush()
            import os

            os.replace(tmp, path)
        except OSError as exc:
            logger.warning("telemetry: report write failed: %s", exc)
            self._storage.append(
                "reporter_blocked",
                {
                    "timestamp": utc_now(),
                    "date": date,
                    "kind": "local_file",
                    "reason": "write_failed",
                    "error": str(exc),
                },
            )
            return False
        return True

    # -- helpers --------------------------------------------------------

    def render_for_today(self, summary: dict[str, Any]) -> str:
        """Convenience for CLI ``preview``."""
        return self.render(summary, utc_date(utc_now()))
