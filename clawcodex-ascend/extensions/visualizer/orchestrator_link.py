#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# -------------------------------------------------------------------------
# This file is part of the AgentSDK project.
# Copyright (c) 2026 Clawd Codex Team
# Copyright (c) 2026 Huawei Technologies Co.,Ltd.
#
# AgentSDK is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#
#          http://license.coscl.org.cn/MulanPSL2
#
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
# -------------------------------------------------------------------------

"""Generate links to session artifacts produced by orchestrated runs.

The generated links cover verification reports, tool-event audit logs,
and debug timeline logs in the current ClawCodeX on-disk layout:

- Tool-event audit logs live at ``~/.clawcodex/tool-events/<run_id>/events.ndjson``.
- Debug timeline logs live at ``~/.clawcodex/tool-events/<run_id>/debug.ndjson``.
- Verification reports live alongside the session at ``<session_dir>/report.{md,json}``.

The old fallback path ``<session_dir>/.orchestrator_control/runs/<run_id>/``
is no longer consulted — that layout was retired when the orchestrator
moved to the canonical tool-events root.
"""

from __future__ import annotations

import logging
from pathlib import Path
from urllib.parse import quote
from typing import Any

logger = logging.getLogger(__name__)

# Canonical tool-events root.
TOOL_EVENTS_ROOT = Path.home() / ".clawcodex" / "tool-events"


class OrchestratorLink:
    """Generate orchestrator-related links for a session."""

    def __init__(self, base_url: str = "http://localhost:8765") -> None:
        self.base_url = base_url.rstrip("/")

    def generate_links(self, session_id: str, session_dir: Path | None = None) -> dict[str, Any]:
        """Generate all orchestrator links for a session."""
        sid_url = quote(session_id, safe="")
        links: dict[str, Any] = {
            "session_id": session_id,
            "api_base": f"{self.base_url}/api/viz",
            "frontend": f"{self.base_url}/session/{sid_url}",
        }

        if session_dir is None:
            session_dir = Path.home() / ".clawcodex" / "sessions" / session_id

        if not session_dir.is_dir():
            links["available"] = False
            return links

        links["available"] = True

        # Verification report
        report_md = session_dir / "report.md"
        report_json = session_dir / "report.json"
        if report_md.exists() or report_json.exists():
            links["f38_report"] = {
                "type": "verification_report",
                "api_url": f"{self.base_url}/api/viz/sessions/{sid_url}/report/f38",
                "file_path": str(report_md if report_md.exists() else report_json),
                "available": True,
            }

        # Tool-event audit log
        # Canonical path: ``~/.clawcodex/tool-events/<run_id>/events.ndjson``.
        # The session_id is the run_id (the new orchestrator key is the
        # session id from bootstrap, not a separate ``run_*``).
        events_file = TOOL_EVENTS_ROOT / session_id / "events.ndjson"
        if not events_file.exists():
            # Legacy fallback (transitional periods only) — same file
            # under the session directory. The old
            # ``.orchestrator_control/runs/<run_id>/`` layout is gone.
            fallback = session_dir / "events.ndjson"
            if fallback.exists():
                events_file = fallback

        if events_file.exists():
            links["f45_events"] = {
                "type": "tool_events",
                "api_url": f"{self.base_url}/api/viz/sessions/{sid_url}/report/f45",
                "file_path": str(events_file),
                "available": True,
                "event_count": self._count_ndjson_lines(events_file),
            }

        # Debug timeline log
        debug_file = TOOL_EVENTS_ROOT / session_id / "debug.ndjson"
        if not debug_file.exists():
            fallback = session_dir / "debug.ndjson"
            if fallback.exists():
                debug_file = fallback

        if debug_file.exists():
            links["f54_debug"] = {
                "type": "debug_timeline",
                "api_url": f"{self.base_url}/api/viz/sessions/{sid_url}/report/f54",
                "file_path": str(debug_file),
                "available": True,
                "entry_count": self._count_ndjson_lines(debug_file),
            }

        return links

    @staticmethod
    def _count_ndjson_lines(path: Path) -> int:
        """Count non-empty lines in an NDJSON file."""
        try:
            count = 0
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        count += 1
            return count
        except Exception:
            return 0

    def generate_share_link(self, session_id: str) -> dict[str, str]:
        """Generate a single-session share payload for the API."""
        return {
            "session_id": session_id,
            "format": "json",
        }
