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

"""orchestrator dashboard — standalone LiveView UI.

Usage:
  clawcodex orchestrator dashboard --port 8080 [--workspace PATH]

Launches a standalone HTTP server that streams real-time orchestrator events
to a web-based dashboard. Agents push events to a local event log, and the
dashboard server reads these logs to render a web UI.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import signal
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

# Local import for the on-demand Visualizer session viewer.
from ..session_viewer import SessionViewerManager  # noqa: E402
from .dashboard_state import DashboardState  # noqa: E402
from .dashboard_state_token import DASHBOARD_HTML  # noqa: E402


# ---------------------------------------------------------------------------
# Issue status taxonomy
# ---------------------------------------------------------------------------
# These constants MUST stay in sync with extensions.orchestrator.issue_registry
# .IssueStatus. We re-declare them here so the dashboard server can be loaded
# even when the orchestrator is not installed in the same interpreter, and so
# the strings are stable for the frontend.

ISSUE_STATUSES: tuple[str, ...] = (
    "queued",
    "pending",
    "running",
    "synced",
    "pending_review",
    "completed",
    "failed",
    "abandoned",
    "verification_failed",
)

STATUS_META: dict[str, dict[str, str]] = {
    "queued": {"label": "Queued", "color": "#6e7681", "icon": "◷", "group": "active"},
    "pending": {"label": "Pending", "color": "#d29922", "icon": "○", "group": "active"},
    "running": {"label": "Running", "color": "#58a6ff", "icon": "◉", "group": "active"},
    "synced": {"label": "Synced", "color": "#a371f7", "icon": "⇄", "group": "active"},
    "pending_review": {"label": "Review", "color": "#79c0ff", "icon": "◎", "group": "active"},
    "completed": {"label": "Completed", "color": "#3fb950", "icon": "✓", "group": "terminal"},
    "failed": {"label": "Failed", "color": "#f85149", "icon": "✗", "group": "terminal"},
    "abandoned": {"label": "Abandoned", "color": "#8b949e", "icon": "⊘", "group": "terminal"},
    "verification_failed": {
        "label": "Verify Failed",
        "color": "#db6d28",
        "icon": "⚠",
        "group": "terminal",
    },
}

ACTIVE_STATUSES = {s for s, m in STATUS_META.items() if m["group"] == "active"}
TERMINAL_STATUSES = {s for s, m in STATUS_META.items() if m["group"] == "terminal"}


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------


def add_dashboard_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "dashboard",
        help="Launch standalone LiveView dashboard UI",
        description="Start an HTTP server with a web dashboard for real-time "
        "orchestrator monitoring. Streams running sessions, tool calls, "
        "and LLM responses.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8080,
        help="Port to listen on (default: 8080)",
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Host to bind to (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--workspace",
        type=str,
        default=None,
        metavar="PATH",
        help="Workspace root to read registry/event logs from. "
        "If omitted, uses $CLAWCODEX_WORKSPACE_ROOT, falls back to the "
        "latest metadata under ~/.clawcodex/orchestrator/*/metadata.json, "
        "and finally ~/.clawcodex/workspace.",
    )
    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="Do not auto-open the dashboard URL in a browser.",
    )


# ---------------------------------------------------------------------------
# Workspace resolution
# ---------------------------------------------------------------------------


def _resolve_workspace_root(explicit: str | None = None) -> Path:
    """Resolve the workspace root in priority order.

    1. Explicit --workspace argument.
    2. $CLAWCODEX_WORKSPACE_ROOT environment variable.
    3. Latest metadata.json under ~/.clawcodex/orchestrator/*/metadata.json.
    4. ~/.clawcodex/workspace (last-resort default).
    """
    if explicit:
        return Path(explicit).expanduser().resolve()

    env_ws = os.environ.get("CLAWCODEX_WORKSPACE_ROOT")
    if env_ws:
        return Path(env_ws).expanduser().resolve()

    metadata_dir = Path.home() / ".clawcodex" / "orchestrator"
    if metadata_dir.exists():
        candidates = []
        for md_dir in metadata_dir.iterdir():
            mf = md_dir / "metadata.json"
            if mf.exists():
                try:
                    data = json.loads(mf.read_text(encoding="utf-8"))
                    ws = data.get("workspace_root")
                    started = data.get("started_at") or 0
                    if ws:
                        candidates.append((started, Path(ws)))
                except Exception:
                    continue
        if candidates:
            candidates.sort(key=lambda c: c[0], reverse=True)
            return candidates[0][1]

    return Path.home() / ".clawcodex" / "workspace"


# ---------------------------------------------------------------------------
# State aggregation
# ---------------------------------------------------------------------------


def _safe_read_json(path: Path) -> dict[str, Any] | None:
    try:
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _classify_status(value: Any) -> str:
    """Normalize a status value to one of ISSUE_STATUSES, defaulting to 'pending'."""
    if not isinstance(value, str):
        return "pending"
    v = value.strip().lower()
    return v if v in ISSUE_STATUSES else "pending"


def _gather_issue_metadata(workspace: Path) -> dict[str, Any]:
    """Read the IssueRegistry JSON and produce a structured per-issue view.

    Returns a dict shaped as:
      {
        "issues": [ {issue_id, identifier, status, ...}, ... ],
        "by_status": { "pending": n, "running": n, ... },
        "totals": { "total": n, "active": n, "terminal": n, "prs": n }
      }
    """
    registry_path = workspace / ".clawcodex_issue_registry.json"
    raw = _safe_read_json(registry_path) or {}

    issues: list[dict[str, Any]] = []
    by_status: dict[str, int] = {s: 0 for s in ISSUE_STATUSES}

    now = time.time()
    for issue_id, record in raw.items():
        if not isinstance(record, dict):
            continue
        status = _classify_status(record.get("status"))
        by_status[status] += 1

        created_at = float(record.get("created_at") or 0)
        updated_at = float(record.get("updated_at") or 0)
        age_seconds = max(0, int(now - created_at)) if created_at else 0
        idle_seconds = max(0, int(now - updated_at)) if updated_at else 0

        workspace_path = record.get("workspace_path") or ""
        workspace_short = ""
        if workspace_path:
            try:
                workspace_short = (
                    "/" + str(Path(workspace_path).relative_to(workspace))
                    if workspace in Path(workspace_path).parents
                    else workspace_path
                )
            except Exception:
                workspace_short = workspace_path

        issues.append(
            {
                "issue_id": issue_id,
                "identifier": record.get("issue_identifier") or issue_id,
                "status": status,
                "branch_name": record.get("branch_name"),
                "commit_sha": record.get("commit_sha"),
                "pr_number": record.get("pr_number"),
                "pr_url": record.get("pr_url"),
                "base_branch": record.get("base_branch") or "main",
                "workspace_path": workspace_path,
                "workspace_short": workspace_short,
                "workspace_strategy": record.get("workspace_strategy"),
                "attempt_count": int(record.get("attempt_count") or 0),
                "retry_count": int(record.get("retry_count") or 0),
                "sequence_index": record.get("sequence_index"),
                "intent": record.get("intent") or "none",
                "report_path": record.get("report_path"),
                "verification_status": record.get("verification_status"),
                "clarification_status": record.get("clarification_status"),
                "created_at": created_at,
                "updated_at": updated_at,
                "age_seconds": age_seconds,
                "idle_seconds": idle_seconds,
                "run_id": record.get("run_id"),
                "run_turn_count": record.get("run_turn_count", 0),
                "run_tool_count": record.get("run_tool_count", 0),
            }
        )

    # Sort: active statuses first, then by most recent activity.
    issues.sort(
        key=lambda i: (
            0 if i["status"] in ACTIVE_STATUSES else 1,
            -int(i["updated_at"] or 0),
        )
    )

    return {
        "issues": issues,
        "by_status": by_status,
        "totals": {
            "total": len(issues),
            "active": sum(1 for i in issues if i["status"] in ACTIVE_STATUSES),
            "terminal": sum(1 for i in issues if i["status"] in TERMINAL_STATUSES),
            "prs": sum(1 for i in issues if i.get("pr_number")),
        },
    }


def _gather_metadata(workspace: Path) -> dict[str, Any]:
    """Read the orchestrator daemon metadata.json (PID, started_at, project)."""
    metadata_dir = Path.home() / ".clawcodex" / "orchestrator"
    if not metadata_dir.exists():
        return {"found": False}

    best: tuple[float, dict[str, Any], Path] | None = None
    for md_dir in metadata_dir.iterdir():
        mf = md_dir / "metadata.json"
        if not mf.exists():
            continue
        try:
            data = json.loads(mf.read_text(encoding="utf-8"))
        except Exception:
            continue
        if data.get("workspace_root") and str(data.get("workspace_root")) != str(workspace):
            continue
        started = float(data.get("started_at") or 0)
        if best is None or started > best[0]:  # pylint: disable=unsubscriptable-object
            best = (started, data, mf)

    if best is None:
        return {"found": False}

    _, data, mf = best
    pid = data.get("pid")
    alive = False
    if isinstance(pid, int):
        try:
            os.kill(pid, 0)
            alive = True
        except (OSError, ProcessLookupError):
            alive = False
    started_at = float(data.get("started_at") or 0)
    return {
        "found": True,
        "pid": pid,
        "alive": alive,
        "started_at": started_at,
        "uptime_seconds": max(0, int(time.time() - started_at)) if started_at else 0,
        "project_slug": data.get("project_slug") or "",
        "workflow_path": data.get("workflow_path") or "",
        "metadata_path": str(mf),
    }


# ---------------------------------------------------------------------------
# HTTP handler
# ---------------------------------------------------------------------------


def _build_dashboard_html() -> str:
    """Inject the STATUS_META JSON into the HTML template."""
    return DASHBOARD_HTML.replace(
        "__STATUS_META__",
        json.dumps(STATUS_META, ensure_ascii=False),
    )


# ---------------------------------------------------------------------------
# HTTP server
# ---------------------------------------------------------------------------


class DashboardHandler(BaseHTTPRequestHandler):
    """HTTP handler serving the dashboard UI, JSON snapshots, and SSE events."""

    server_version = "ClawCodexDashboard/1.0"
    state: DashboardState  # set on the class by run()

    # Quieter logs — one line per request is too noisy for a polling UI.
    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002  # pylint: disable=redefined-builtin
        return

    def _send_json(self, payload: Any, status: int = 200) -> None:
        body = json.dumps(payload, ensure_ascii=False, default=str).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def _send_text(self, text: str, content_type: str, status: int = 200) -> None:
        body = text.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        path = parsed.path or "/"

        if path == "/":
            self._send_text(_build_dashboard_html(), "text/html; charset=utf-8")
            return

        if path == "/api/state":
            snap = self.state.refresh_snapshot(force=True)
            self._send_json(snap)
            return

        if path.startswith("/api/issue/"):
            issue_id = path[len("/api/issue/") :]
            snap = self.state.refresh_snapshot(force=True)
            for issue in snap["issues"]["issues"]:
                if issue["issue_id"] == issue_id:
                    self._send_json({"issue": issue})
                    return
            self._send_json({"error": "not found", "issue_id": issue_id}, status=404)
            return

        if path == "/api/health":
            self._send_json(
                {
                    "ok": True,
                    "workspace": str(self.state.workspace),
                    "ts": time.time(),
                }
            )
            return

        if path == "/events":
            self._stream_events()
            return

        self.send_error(404, "Not Found")

    def do_POST(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        path = parsed.path or "/"

        if path == "/api/session-viewer/start":
            try:
                viewer = getattr(self.__class__, "viewer_manager", None)
                if viewer is None:
                    self._send_json({"error": "viewer manager not available"}, 503)
                    return
                port = viewer.ensure_running()
                self._send_json({"port": port, "pid": viewer.pid})
            except Exception as exc:
                logger.error("Failed to start session viewer: %s", exc)
                self._send_json({"error": str(exc)}, 500)
            return

        if path == "/api/session-viewer/stop":
            try:
                viewer = getattr(self.__class__, "viewer_manager", None)
                if viewer is not None:
                    viewer.stop()
                self._send_json({"ok": True})
            except Exception as exc:
                self._send_json({"error": str(exc)}, 500)
            return

        if path == "/api/session-viewer/status":
            viewer = getattr(self.__class__, "viewer_manager", None)
            if viewer is None:
                self._send_json({"running": False, "port": 0})
            else:
                self._send_json(
                    {
                        "running": viewer.is_running,
                        "port": viewer.port,
                        "uptime_s": viewer.uptime_s,
                    }
                )
            return

        self.send_error(404, "Not Found")

    # ----- SSE streaming ---------------------------------------------------

    def _stream_events(self) -> None:
        try:
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream; charset=utf-8")
            self.send_header("Cache-Control", "no-cache, no-transform")
            self.send_header("Connection", "keep-alive")
            self.send_header("X-Accel-Buffering", "no")
            self.end_headers()
        except Exception:
            return

        snapshot_interval = self.state.snapshot_interval
        last_snapshot_at = 0.0

        try:
            # Unified storage: live event stream is sourced from
            # the session transcript JSONL.  We only push periodic
            # snapshot updates here — per-event streaming is handled
            # out-of-band by callers (e.g. ``issue takeover``) since the
            # transcript is a regular append-only file readable by
            # ``tail -f`` / TailFollower.
            snap = self.state.refresh_snapshot(force=True)
            last_snapshot_at = time.time()
            self._write_sse({"type": "snapshot", **snap})

            while True:
                # Throttle snapshot refreshes to once per snapshot_interval.
                now = time.time()
                if now - last_snapshot_at >= snapshot_interval:
                    snap = self.state.refresh_snapshot(force=True)
                    last_snapshot_at = now
                    self._write_sse({"type": "snapshot", **snap})

                # Heartbeat / keep-alive.
                self.wfile.write(b": ping\n\n")
                self.wfile.flush()

                time.sleep(snapshot_interval)
        except Exception:
            # Client disconnected or stream error — exit cleanly.
            return

    def _write_sse(self, payload: dict[str, Any]) -> None:
        data = json.dumps(payload, ensure_ascii=False, default=str)
        self.wfile.write(f"data: {data}\n\n".encode("utf-8"))
        self.wfile.flush()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def run(args: argparse.Namespace) -> int:
    """Execute the orchestrator dashboard command."""
    port: int = args.port
    host: str = args.host
    no_browser: bool = getattr(args, "no_browser", False)

    workspace = _resolve_workspace_root(getattr(args, "workspace", None))
    state = DashboardState(workspace)

    # Bind the per-process state to the handler class.
    DashboardHandler.state = state

    # Initialize the on-demand session viewer manager.
    viewer_manager = SessionViewerManager(idle_timeout_s=300)
    DashboardHandler.viewer_manager = viewer_manager

    # Convert SIGTERM / SIGHUP into KeyboardInterrupt so the existing
    # except-block + atexit hooks clean up the Visualizer subprocess and
    # tailer threads.  Without this, `kill <pid>` or closing the terminal
    # would orphan the subprocess (it's in a separate process group via
    # os.setsid, so the signal doesn't reach it directly).
    def _graceful_signal(signum: int, frame: Any) -> None:
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, _graceful_signal)
    if hasattr(signal, "SIGHUP"):
        signal.signal(signal.SIGHUP, _graceful_signal)

    print(f"[dashboard] Workspace : {workspace}")
    print(f"[dashboard] Starting LiveView dashboard on http://{host}:{port}")
    if not workspace.exists():
        print("[dashboard] Note: workspace does not exist yet — UI will render empty until it is created.")

    try:
        server = ThreadingHTTPServer((host, port), DashboardHandler)
        threading.Thread(target=server.serve_forever, name="DashboardHTTP", daemon=True).start()

        if not no_browser:
            try:
                import webbrowser

                webbrowser.open(f"http://{host}:{port}")
            except Exception:  # nosec B110
                pass

        print(f"[dashboard] Serving at http://{host}:{port}", file=sys.stderr)
        print("[dashboard] Press Ctrl+C to stop", file=sys.stderr)

        # Park the main thread.
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n[dashboard] stopped")
    except OSError as exc:
        print(f"[dashboard] error: {exc}", file=sys.stderr)
        return 1
    finally:
        # Explicit cleanup (belt-and-suspenders alongside atexit hooks).
        viewer_manager.stop()
        state.tailer_manager.stop_all()

    return 0
