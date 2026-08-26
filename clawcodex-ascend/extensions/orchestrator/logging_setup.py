#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# -------------------------------------------------------------------------
#  This file is part of the AgentSDK project.
# Copyright (c) 2026 Huawei Technologies Co.,Ltd.
#
# AgentSDK is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#
#           http://license.coscl.org.cn/MulanPSL2
#
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
# -------------------------------------------------------------------------

"""Centralized logging configuration for the orchestrator daemon.

Provides three complementary outputs via a single ``configure_orchestrator_logging()``
call — the one function that replaces every ad-hoc ``basicConfig`` across entry points:

1. **Console** — human-friendly, colored by level, ISO8601+TZ, process/thread,
   shortened logger name, trailing ``key=value`` pairs from ``LogContext``.

2. **JSON file** (optional) — one NDJSON object per line under ``<workspace>/.reports/``
   for ingestion by ELK / Loki / Datadog / ``journalctl -o json``.

3. **Structured key=value** — the message itself uses ``key=value`` syntax for
   grep-friendly diagnostics, inspired by Go's zerolog/zap.

Design influences:
  • **Uvicorn/Gunicorn** — ``[timestamp] LEVEL     component  message``
  • **Spring Boot / Logstash** — JSON structured with ``@timestamp``, ``level``,
    ``logger_name``, ``thread_name``, ``process_id``, ``message``
  • **Go zerolog / zap** — trailing ``key=value`` pairs, level colours
  • **Django / MDC** — thread-local context injected automatically into every
    record without polluting call sites
"""

from __future__ import annotations

import json
import logging
import sys
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# ─────────────────────────────────────────────────────────────────────────────
# Thread-local MDC (Mapped Diagnostic Context)
# ─────────────────────────────────────────────────────────────────────────────

_MDC = threading.local()
_MDC_DEFAULTS: dict[str, str] = {}


def set_log_context(**kwargs: str | None) -> None:
    """Set thread-local log context fields (e.g. ``issue_id``, ``run_id``).

    All active fields are appended to every subsequent log record emitted
    from the current thread as ``key=value`` tokens.  Pass ``None`` to
    clear a key::

        set_log_context(issue_id="42", run_id="abc-def")
        logger.info("Agent started")    # → … issue_id=42 run_id=abc-def
        set_log_context(issue_id=None)  # clear issue_id
    """
    store = getattr(_MDC, "fields", {})
    for k, v in kwargs.items():
        if v is None:
            store.pop(k, None)
        else:
            store[k] = v
    _MDC.fields = store


def get_log_context() -> dict[str, str]:
    """Return a snapshot of the current thread-local log context."""
    return dict(getattr(_MDC, "fields", _MDC_DEFAULTS))


def clear_log_context() -> None:
    """Remove all thread-local log context fields."""
    _MDC.fields = {}  # type: ignore[attr-defined]


def _mdc_record_factory(*args: Any, **kwargs: Any) -> logging.LogRecord:
    """Inject MDC fields into the record so the formatter can see them."""
    record = logging.LogRecord(*args, **kwargs)
    record.mdc_fields = get_log_context()  # type: ignore[attr-defined]
    return record


# ---------------------------------------------------------------------------
# Console formatter — human-friendly, coloured, timezone-aware
# ---------------------------------------------------------------------------

_LEVEL_COLOUR: dict[int, str] = {
    logging.CRITICAL: "\033[1;41m",  # white-on-red
    logging.ERROR: "\033[1;31m",  # bold red
    logging.WARNING: "\033[0;33m",  # yellow
    logging.INFO: "\033[0;32m",  # green
    logging.DEBUG: "\033[0;36m",  # cyan
}
_RESET = "\033[0m"


def _shorten_logger(name: str, max_parts: int = 2) -> str:
    """Shorten a fully-qualified logger name to its last *max_parts* segments.

    ``extensions.orchestrator.agent_runner`` → ``…orchestrator.agent_runner``
    """
    parts = name.split(".")
    if len(parts) <= max_parts:
        return name
    return "…" + ".".join(parts[-max_parts:])


def _format_timestamp(ts: float) -> str:
    """ISO-8601 with timezone, millisecond precision, e.g. ``2026-07-01T17:17:00.123Z``."""
    dt = datetime.fromtimestamp(ts, tz=timezone.utc)
    return dt.strftime("%Y-%m-%dT%H:%M:%S.") + f"{dt.microsecond // 1000:03d}Z"


class OrchestratorConsoleFormatter(logging.Formatter):
    """Coloured console formatter with MDC context.

    Output pattern::

        [2026-07-01T17:17:00.123Z] INFO      orchestrator   Agent started  issue_id=42 run_id=abc
        [2026-07-01T17:17:00.456Z] ERROR     agent_runner   Write failed   tool=Write …
    """

    def format(self, record: logging.LogRecord) -> str:
        ts = _format_timestamp(record.created)
        level_name = record.levelname.ljust(8)
        colour = _LEVEL_COLOUR.get(record.levelno, "")
        logger_short = _shorten_logger(record.name)
        msg = super().format(record)

        # Append MDC context fields
        mdc = getattr(record, "mdc_fields", {})
        ctx_str = ""
        if mdc:
            ctx_str = "  " + " ".join(f"{k}={v}" for k, v in mdc.items())

        # Exception info (traceback) – already handled by super() in msg
        if colour:
            return f"\033[2m[{ts}]\033[0m {colour}{level_name}{_RESET} {logger_short:<20s} {msg}{ctx_str}"
        return f"[{ts}] {level_name} {logger_short:<20s} {msg}{ctx_str}"


# ---------------------------------------------------------------------------
# JSON formatter — one NDJSON line per record, for log aggregators
# ---------------------------------------------------------------------------


class OrchestratorJsonFormatter(logging.Formatter):
    """Structured JSON formatter — one line per record.

    Produces logstash-compatible JSON for ELK / Loki / Datadog::

        {"@timestamp":"2026-07-01T17:17:00.123Z","level":"INFO", …}
    """

    def format(self, record: logging.LogRecord) -> str:
        mdc = getattr(record, "mdc_fields", {})
        payload: dict[str, Any] = {
            "@timestamp": _format_timestamp(record.created),
            "level": record.levelname,
            "logger": record.name,
            "thread": record.threadName,
            "process": record.process,
            "message": record.getMessage(),
            "context": mdc,
        }
        if record.exc_info and record.exc_info[0] is not None:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False, default=str)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

_LOG_CONFIGURED = False
_LOG_CONFIG_LOCK = threading.Lock()


def configure_orchestrator_logging(
    *,
    level: int = logging.INFO,
    json_path: str | Path | None = None,
    console: bool = True,
    root_level: int | None = None,
) -> None:
    """One-stop logging setup for the orchestrator daemon.

    Parameters
    ----------
    level:
        Minimum log level for both console and JSON handlers.
    json_path:
        Optional path to a JSONL file for structured log ingestion.
        Typically ``<workspace>/.reports/orchestrator.ndjson``.
    console:
        Whether to install a coloured stderr handler (default ``True``).
    root_level:
        Explicit root logger level override.  If ``None``, the root level
        is raised to ``level`` (but never raised above ``DEBUG``, so that
        ``CLAWCODEX_DEBUG=1`` from upstream ``run_cli`` takes precedence).

    Idempotent — can be called multiple times; only the first call applies.
    """
    global _LOG_CONFIGURED
    with _LOG_CONFIG_LOCK:
        if _LOG_CONFIGURED:
            return
        _LOG_CONFIGURED = True

    # Replace the default LogRecord factory so MDC fields are injected
    # into every record without call-site changes.
    logging.setLogRecordFactory(_mdc_record_factory)

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)  # let handler-level filtering decide

    # Remove pre-existing handlers (installed by upstream run_cli etc.)
    root.handlers.clear()

    if console:
        h = logging.StreamHandler(sys.stderr)
        h.setLevel(level)
        h.setFormatter(OrchestratorConsoleFormatter())
        root.addHandler(h)

    if json_path:
        p = Path(json_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        try:
            fh = logging.FileHandler(str(p), mode="a", encoding="utf-8")
            fh.setLevel(level)
            fh.setFormatter(OrchestratorJsonFormatter())
            root.addHandler(fh)
        except OSError:
            # Best-effort: warn to console but don't crash
            root.warning("Failed to open JSON log path=%s — skipping", json_path)

    # Raise root level if the runner's default (ERROR) is too high;
    # but never override DEBUG when CLAWCODEX_DEBUG=1 is active.
    if root_level is not None:
        root.setLevel(root_level)
    elif root.level > logging.DEBUG:
        root.setLevel(level)

    # Capture Python's ``warnings`` module into the logging system.
    logging.captureWarnings(True)

    root.info(
        "Logging configured level=%s json_path=%s",
        logging.getLevelName(level),
        json_path or "none",
    )


__all__ = [
    "OrchestratorConsoleFormatter",
    "OrchestratorJsonFormatter",
    "clear_log_context",
    "configure_orchestrator_logging",
    "get_log_context",
    "set_log_context",
]
