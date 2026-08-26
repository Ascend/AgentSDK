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

"""State Journal Writer for Orchestrator.

Writes NDJSON events to ``{workspace}/.reports/run_{run_id}/state_journal.ndjson``
so the Visualizer can consume orchestrator runtime state without importing
any orchestrator code.

Design principle: the writer is a thin, fire-and-forget append-only logger.
Write failures are logged but never propagated — a broken journal must not
crash the agent run.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class StateJournalWriter:
    """Append-only NDJSON writer for orchestrator state events.

    Each call to :meth:`write_event` appends one JSON line to the journal
    file.  The file is created on first write; the directory is created
    if it does not exist.

    Usage::

        writer = StateJournalWriter(run_dir)
        writer.write_event({"type": "phase", "phase": "agent_run", ...})
    """

    def __init__(self, run_dir: Path, run_id: str) -> None:
        self._run_dir = run_dir
        self._run_id = run_id
        self._path = run_dir / "state_journal.ndjson"
        self._closed = False
        # Ensure directory exists
        try:
            self._run_dir.mkdir(parents=True, exist_ok=True)
        except OSError:
            logger.warning("Failed to create state journal dir: %s", self._run_dir)

    @property
    def path(self) -> Path:
        return self._path

    @property
    def run_id(self) -> str:
        return self._run_id

    def write_event(self, event: dict[str, Any]) -> None:
        """Append one NDJSON line to the journal file.

        Automatically injects ``timestamp`` (ISO 8601) and ``run_id`` if
        absent.  Exceptions are caught and logged — never propagated.
        """
        if self._closed:
            return
        try:
            if "timestamp" not in event:
                event["timestamp"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            if "run_id" not in event:
                event["run_id"] = self._run_id
            line = json.dumps(event, ensure_ascii=False, default=str) + "\n"
            with open(self._path, "a", encoding="utf-8") as f:
                f.write(line)
        except Exception as exc:
            logger.debug("state_journal write failed: %s", exc)

    def close(self) -> None:
        """Mark the writer as closed.  Subsequent writes are no-ops."""
        self._closed = True

    # ------------------------------------------------------------------
    # Convenience helpers for common event types
    # ------------------------------------------------------------------

    def write_phase(
        self,
        phase: str,
        progress: float | None = None,
        message: str = "",
        issue_id: str = "",
    ) -> None:
        """Write a ``type: "phase"`` event."""
        event: dict[str, Any] = {
            "type": "phase",
            "phase": phase,
            "message": message,
        }
        if progress is not None:
            event["progress"] = round(progress, 3)
        if issue_id:
            event["issue_id"] = issue_id
        self.write_event(event)

    def write_issue_status(
        self,
        issue_id: str,
        status: str,
        message: str = "",
    ) -> None:
        """Write a ``type: "issue_status"`` event."""
        self.write_event(
            {
                "type": "issue_status",
                "issue_id": issue_id,
                "status": status,
                "message": message,
            }
        )

    def write_verification(
        self,
        issue_id: str,
        verification_status: str,
        result: str = "",
    ) -> None:
        """Write a ``type: "verification"`` event."""
        self.write_event(
            {
                "type": "verification",
                "issue_id": issue_id,
                "verification_status": verification_status,
                "result": result,
            }
        )

    def write_pr_status(
        self,
        issue_id: str,
        pr_url: str,
        pr_status: str = "open",
        pr_number: str | None = None,
    ) -> None:
        """Write a ``type: "pr_status"`` event."""
        event: dict[str, Any] = {
            "type": "pr_status",
            "issue_id": issue_id,
            "pr_url": pr_url,
            "pr_status": pr_status,
        }
        if pr_number is not None:
            event["pr_number"] = pr_number
        self.write_event(event)

    def write_session_ref(
        self,
        issue_id: str,
        session_id: str,
        session_path: str = "",
    ) -> None:
        """Write a ``type: "session_ref"`` event."""
        self.write_event(
            {
                "type": "session_ref",
                "issue_id": issue_id,
                "session_id": session_id,
                "session_path": session_path,
            }
        )

    def write_error(
        self,
        issue_id: str,
        error: str,
    ) -> None:
        """Write a ``type: "error"`` event."""
        self.write_event(
            {
                "type": "error",
                "issue_id": issue_id,
                "error": error,
            }
        )

    def write_complete(
        self,
        issue_id: str,
        overall_status: str,
        message: str = "",
    ) -> None:
        """Write a ``type: "complete"`` event."""
        self.write_event(
            {
                "type": "complete",
                "issue_id": issue_id,
                "overall_status": overall_status,
                "message": message,
            }
        )
