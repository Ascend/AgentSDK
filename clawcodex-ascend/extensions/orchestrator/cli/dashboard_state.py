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

from __future__ import annotations

import atexit
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any

from ..event_tailer import EventTailerManager
from .dashboard_state_token import DashboardStateTokenActivityMixin


class DashboardState(DashboardStateTokenActivityMixin):
    """Per-process shared state for the dashboard HTTP server."""

    def __init__(self, workspace: Path) -> None:
        from .dashboard import _gather_metadata, _gather_issue_metadata

        self.workspace = workspace
        self.snapshot: dict[str, Any] = {
            "type": "snapshot",
            "ts": time.time(),
            "workspace": str(workspace),
            "workspace_exists": workspace.exists(),
            "metadata": _gather_metadata(workspace),
            "issues": _gather_issue_metadata(workspace),
            "events": {"total": 0, "by_type": {}, "recent": []},
            "token_activity": {
                "active_sessions": 0,
                "total_turns": 0,
                "total_tools": 0,
            },
        }
        self.last_snapshot_at: float = time.time()
        self.snapshot_interval: float = 0.5
        self._lock = threading.Lock()
        # Event tailer for live per-session events
        self.tailer_manager = EventTailerManager(workspace)
        atexit.register(self.tailer_manager.stop_all)
        # Rolling event buffers
        self._recent_events: deque[dict[str, Any]] = deque(maxlen=200)
        self._event_by_type: dict[str, int] = {}
        # Track completed issues whose historical events have been loaded
        self._loaded_historical: set[str] = set()
        # Track run_ids that were ever actively tailed, to prevent
        # load_historical() from re-reading files the tailer already consumed
        self._ever_tailed: set[str] = set()

    def refresh_snapshot(self, force: bool = False) -> dict[str, Any]:
        now = time.time()
        with self._lock:
            if force or (now - self.last_snapshot_at) >= self.snapshot_interval:
                from .dashboard import ACTIVE_STATUSES, _gather_issue_metadata, _gather_metadata

                # 1. Read registry + metadata
                issues = _gather_issue_metadata(self.workspace)
                meta = _gather_metadata(self.workspace)

                # 2. Sync tailers: extract active run_id → (issue_id, workspace_path) mapping
                run_id_map: dict[str, tuple[str, Path]] = {}
                for issue in issues.get("issues", []):
                    rid = issue.get("run_id")
                    if rid and issue.get("status") in ACTIVE_STATUSES:
                        ws = issue.get("workspace_path")
                        if ws:
                            run_id_map[rid] = (issue["issue_id"], Path(ws))
                            self._ever_tailed.add(rid)
                self.tailer_manager.sync_active_run_ids(run_id_map)

                # 2b. One-shot historical load for completed issues with run_id
                # that haven't been loaded yet AND were never actively tailed.
                for issue in issues.get("issues", []):
                    rid = issue.get("run_id")
                    if (
                        rid
                        and issue.get("status") not in ACTIVE_STATUSES
                        and rid not in self._loaded_historical
                        and rid not in self._ever_tailed
                    ):
                        ws = issue.get("workspace_path")
                        if ws:
                            self.tailer_manager.load_historical(rid, issue["issue_id"], Path(ws))
                            self._loaded_historical.add(rid)

                # 3. Drain events, update rolling buffers
                for evt in self.tailer_manager.drain_events():
                    self._recent_events.appendleft(evt)
                    et = evt.get("event_type", "unknown")
                    self._event_by_type[et] = self._event_by_type.get(et, 0) + 1

                # 4. Assemble snapshot with events + token_activity
                self.snapshot = {
                    "type": "snapshot",
                    "ts": time.time(),
                    "workspace": str(self.workspace),
                    "workspace_exists": self.workspace.exists(),
                    "metadata": meta,
                    "issues": issues,
                    "events": {
                        "total": sum(self._event_by_type.values()),
                        "by_type": dict(self._event_by_type),
                        "recent": list(self._recent_events)[:200],
                    },
                    "token_activity": self._build_token_activity(issues),
                }
                self.last_snapshot_at = now
            return self.snapshot
