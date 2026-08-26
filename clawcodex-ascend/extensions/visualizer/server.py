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

"""Standalone FastAPI application for local session visualization."""

from __future__ import annotations

import json
import logging
import time
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any
from urllib.parse import quote

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from .builders.anomaly_builder import AnomalyBuilder  # pylint: disable=relative-beyond-top-level
from .builders.stats_builder import StatsBuilder  # pylint: disable=relative-beyond-top-level
from .builders.timeline_builder import TimelineBuilder  # pylint: disable=relative-beyond-top-level
from .import_router import create_import_router  # pylint: disable=relative-beyond-top-level
from .models.viz_models import (  # pylint: disable=relative-beyond-top-level
    AgentTreeNode,
    Anomaly,
    DashboardEntryViz,
    ImportStatus,
    OperationStats,
    SessionVizData,
    ShareLink,
    WorkspaceInfo,
)
from .ws import (  # pylint: disable=relative-beyond-top-level
    create_dashboard_ws_router,
    create_orch_ws_router,
    create_ws_router,
)

logger = logging.getLogger(__name__)


class _AppState:
    """Mutable state shared by the visualizer request handlers."""

    def __init__(
        self,
        sessions_dir: Path | None = None,
        transcripts_dir: Path | None = None,
        workspaces_file: Path | None = None,
        allow_import: bool = False,
    ) -> None:
        self.sessions_dir = sessions_dir or (Path.home() / ".clawcodex" / "sessions")
        self.transcripts_dir = transcripts_dir or (Path.home() / ".clawcodex" / "transcripts")
        self.workspaces_file = workspaces_file or (Path.home() / ".clawcodex" / "workspaces.json")
        self.allow_import = allow_import
        self.reports_dir = Path.home() / ".clawcodex" / "reports"
        self.timeline_builder = TimelineBuilder(
            sessions_dir=self.sessions_dir,
            transcripts_dir=self.transcripts_dir,
            reports_dir=self.reports_dir,
        )
        self.stats_builder = StatsBuilder()
        self.anomaly_builder = AnomalyBuilder()
        self.share_links: dict[str, ShareLink] = {}
        self.import_tasks: dict[str, ImportStatus] = {}
        self._shares_path = Path.home() / ".clawcodex" / "viz_shares.json"
        self._load_share_links()
        # Optional adjacent dashboard integration; tests may replace this store.
        try:
            from extensions.agent_dashboard import get_default_store  # pylint: disable=no-name-in-module

            self.dashboard_store = get_default_store()
        except Exception:
            logger.debug("Failed to load default dashboard store", exc_info=True)
            self.dashboard_store = None

    def _load_share_links(self) -> None:
        if not self._shares_path.exists():
            return
        try:
            data = json.loads(self._shares_path.read_text(encoding="utf-8"))
            now = time.time()
            for item in data:
                if not isinstance(item, dict) or item.get("expires_at", 0) <= now:
                    continue
                if item.get("view_type") not in (None, "session"):
                    continue
                try:
                    share = ShareLink(**item)
                except Exception:
                    continue
                self.share_links[share.id] = share
        except Exception:
            logger.debug("Failed to load share links", exc_info=True)

    def _save_share_links(self) -> None:
        try:
            now = time.time()
            active = [share for share in self.share_links.values() if share.expires_at > now]
            self._shares_path.parent.mkdir(parents=True, exist_ok=True)
            self._shares_path.write_text(
                json.dumps([share.model_dump(mode="json") for share in active], indent=2),
                encoding="utf-8",
            )
        except Exception:
            logger.debug("Failed to save share links", exc_info=True)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    logger.info(
        "Visualizer server starting - sessions_dir=%s, allow_import=%s",
        app.state.viz.sessions_dir,
        app.state.viz.allow_import,
    )
    yield
    app.state.viz._save_share_links()
    logger.info("Visualizer server shutting down")


def create_app(
    sessions_dir: Path | None = None,
    transcripts_dir: Path | None = None,
    workspaces_file: Path | None = None,
    allow_import: bool = False,
    host: str = "0.0.0.0",
    port: int = 8765,
) -> FastAPI:
    """Create the standalone visualizer app.

    ``host`` and ``port`` remain accepted for CLI compatibility; uvicorn owns
    the actual socket binding.
    """
    del host, port
    app = FastAPI(title="ClawCodex Visualizer", version="0.2.0", lifespan=lifespan)
    app.state.viz = _AppState(
        sessions_dir=sessions_dir,
        transcripts_dir=transcripts_dir,
        workspaces_file=workspaces_file,
        allow_import=allow_import,
    )
    app.include_router(create_ws_router(), prefix="/api/viz")
    app.include_router(create_orch_ws_router(), prefix="/api/viz")
    app.include_router(create_dashboard_ws_router(), prefix="/api/viz")
    if allow_import:
        app.include_router(create_import_router(), prefix="/api/viz")

    package_dir = Path(__file__).resolve().parent
    templates_dir = package_dir / "templates"
    static_dir = package_dir / "static"
    app.state.templates = Jinja2Templates(directory=str(templates_dir)) if templates_dir.is_dir() else None
    if static_dir.is_dir():
        app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    _register_dashboard_routes(app)
    _register_health_routes(app)
    _register_workspace_routes(app)
    _register_session_routes(app)
    _register_share_routes(app)
    _register_orchestrator_routes(app)
    _register_frontend_routes(app)
    return app


def _register_dashboard_routes(app: FastAPI) -> None:
    @app.get(
        "/api/dashboard/snapshot",
        response_model=list[DashboardEntryViz],
        tags=["dashboard"],
    )
    async def dashboard_snapshot(
        source: str | None = None,
        status: str | None = None,
    ) -> list[DashboardEntryViz]:
        """Return the current cross-system dashboard snapshot."""
        store = getattr(app.state.viz, "dashboard_store", None)
        if store is None:
            return []
        filters: dict[str, Any] = {}
        if source:
            filters["source"] = source
        if status:
            filters["status"] = status
        entries = store.snapshot(filters=filters) if filters else store.snapshot()
        return [DashboardEntryViz(**entry.to_dict()) for entry in entries]

    @app.get("/viz/dashboard", response_class=HTMLResponse, tags=["frontend"])
    async def agent_dashboard(request: Request) -> HTMLResponse:
        if app.state.templates is None:
            return HTMLResponse("<h1>Templates not found</h1>", status_code=500)
        return app.state.templates.TemplateResponse(
            request,
            "agent_dashboard_tab.html",
            {"request": request, "page_title": "Agent Dashboard"},
        )


def _register_health_routes(app: FastAPI) -> None:
    @app.get("/api/viz/health", tags=["health"])
    async def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "allow_import": app.state.viz.allow_import,
            "sessions_dir": str(app.state.viz.sessions_dir),
        }


def _register_workspace_routes(app: FastAPI) -> None:
    @app.get("/api/viz/workspaces", response_model=list[WorkspaceInfo], tags=["workspace"])
    async def list_workspaces() -> list[WorkspaceInfo]:
        return _list_workspaces(app)

    @app.get("/api/viz/workspaces/{workspace_id}/sessions", tags=["workspace"])
    async def list_workspace_sessions(
        workspace_id: str,
        q: str | None = None,
        status: str | None = None,
    ) -> list[dict[str, Any]]:
        sessions = _list_sessions_in_workspace(app, workspace_id)
        if status and status != "all":
            wanted_status = status.lower()
            sessions = [item for item in sessions if (item.status or "").lower() == wanted_status]
        if q:
            needle = q.lower()
            sessions = [item for item in sessions if _session_matches_query(item, needle)]
        return [_session_summary(item) for item in sessions]


def _register_session_routes(app: FastAPI) -> None:
    @app.get(
        "/api/viz/sessions/{session_id}",
        response_model=SessionVizData,
        tags=["session"],
    )
    async def get_session(session_id: str) -> SessionVizData:
        viz = app.state.viz.timeline_builder.build(session_id)
        if viz is None:
            raise HTTPException(status_code=404, detail=f"Session {session_id} not found")
        return viz

    @app.get("/api/viz/sessions/{session_id}/stats", tags=["session"])
    async def get_session_stats(session_id: str) -> OperationStats:
        return _required_session(app, session_id).stats

    @app.get("/api/viz/sessions/{session_id}/anomalies", tags=["session"])
    async def get_session_anomalies(session_id: str) -> list[Anomaly]:
        return _required_session(app, session_id).anomalies

    @app.get("/api/viz/sessions/{session_id}/tree", tags=["session"])
    async def get_session_tree(session_id: str) -> list[AgentTreeNode]:
        return _required_session(app, session_id).agent_tree

    @app.get("/api/viz/sessions/{session_id}/report", tags=["session"])
    async def get_session_report(session_id: str) -> dict[str, str | None]:
        viz = _required_session(app, session_id)
        sid_url = quote(session_id, safe="")
        links: dict[str, str | None] = {}
        if viz.report_path:
            links["f38_report"] = f"/api/viz/sessions/{sid_url}/report/f38"
        if viz.tool_events_path:
            links["f45_events"] = f"/api/viz/sessions/{sid_url}/report/f45"
        if viz.debug_log_path:
            links["f54_debug"] = f"/api/viz/sessions/{sid_url}/report/f54"
        return links

    @app.get("/api/viz/sessions/{session_id}/report/f38", tags=["session"])
    async def get_session_f38_report(session_id: str) -> Response:
        viz = _required_session(app, session_id)
        return _read_report_file(viz.report_path, "text/markdown; charset=utf-8")

    @app.get("/api/viz/sessions/{session_id}/report/f45", tags=["session"])
    async def get_session_f45_events(session_id: str) -> Response:
        viz = _required_session(app, session_id)
        return _read_report_file(viz.tool_events_path, "application/x-ndjson; charset=utf-8")

    @app.get("/api/viz/sessions/{session_id}/report/f54", tags=["session"])
    async def get_session_f54_debug(session_id: str) -> Response:
        viz = _required_session(app, session_id)
        return _read_report_file(viz.debug_log_path, "application/x-ndjson; charset=utf-8")


def _register_share_routes(app: FastAPI) -> None:
    @app.post("/api/viz/share", tags=["share"])
    async def create_share_link(request: Request) -> ShareLink:
        body = await request.json()
        if body.get("view_type") not in (None, "session"):
            raise HTTPException(status_code=400, detail="Only single-session shares are supported")
        session_id = str(body.get("session_id", "")).strip()
        if not session_id:
            raise HTTPException(status_code=400, detail="session_id is required")
        _required_session(app, session_id)
        now = time.time()
        share = ShareLink(
            id=uuid.uuid4().hex[:12],
            session_id=session_id,
            created_at=now,
            expires_at=now + 7 * 86400,
            payload=body.get("payload", {}),
        )
        app.state.viz.share_links[share.id] = share
        app.state.viz._save_share_links()
        return share

    @app.get("/api/viz/share/{link_id}", tags=["share"])
    async def get_share_link(link_id: str) -> dict[str, Any]:
        share = app.state.viz.share_links.get(link_id)
        if share is None:
            raise HTTPException(status_code=404, detail="Share link not found")
        if time.time() > share.expires_at:
            del app.state.viz.share_links[link_id]
            app.state.viz._save_share_links()
            raise HTTPException(status_code=410, detail="Share link expired")
        return {"share": share, "data": _required_session(app, share.session_id)}

    @app.delete("/api/viz/share/{link_id}", tags=["share"])
    async def delete_share_link(link_id: str) -> dict[str, str]:
        if link_id not in app.state.viz.share_links:
            raise HTTPException(status_code=404, detail="Share link not found")
        del app.state.viz.share_links[link_id]
        app.state.viz._save_share_links()
        return {"status": "deleted"}


def _register_orchestrator_routes(app: FastAPI) -> None:
    # Preserve the existing orchestrator parser and response protocol.
    @app.get("/api/viz/orchestrator/runs", tags=["orchestrator"])
    async def list_orchestrator_runs() -> list[dict[str, Any]]:
        return _orchestrator_parser(app).list_runs()

    @app.get("/api/viz/orchestrator/state", tags=["orchestrator"])
    async def get_orchestrator_state() -> dict[str, Any]:
        return _orchestrator_parser(app).get_current_snapshot()

    @app.get("/api/viz/orchestrator/runs/{run_id}", tags=["orchestrator"])
    async def get_orchestrator_run(run_id: str) -> dict[str, Any]:
        state = _orchestrator_parser(app).parse_run(run_id)
        if state is None:
            raise HTTPException(status_code=404, detail="Run not found")
        return {
            "run_id": state.run_id,
            "workflow": state.workflow,
            "started_at": state.started_at,
            "event_count": state.event_count,
            "issues": {
                issue_id: {
                    "issue_id": issue.issue_id,
                    "status": issue.status,
                    "current_phase": issue.current_phase,
                    "progress": issue.progress,
                    "verification_status": issue.verification_status,
                    "pr_url": issue.pr_url,
                    "pr_number": issue.pr_number,
                    "session_path": issue.session_path,
                    "session_id": issue.session_id,
                    "error": issue.error,
                    "started_at": issue.started_at,
                    "last_updated": issue.last_updated,
                    "event_count": len(issue.events),
                }
                for issue_id, issue in state.issues.items()
            },
        }

    @app.get(
        "/api/viz/orchestrator/runs/{run_id}/issues/{issue_id}/timeline",
        tags=["orchestrator"],
    )
    async def get_issue_timeline(run_id: str, issue_id: str) -> list[dict[str, Any]]:
        return _orchestrator_parser(app).get_issue_timeline(run_id, issue_id)

    @app.get("/viz/orchestrator", response_class=HTMLResponse, tags=["frontend"])
    async def orchestrator_dashboard(request: Request) -> HTMLResponse:
        if app.state.templates is None:
            return HTMLResponse("<h1>Templates not found</h1>", status_code=500)
        return app.state.templates.TemplateResponse(
            request,
            "orchestrator_dashboard.html",
            {"request": request},
        )


def _register_frontend_routes(app: FastAPI) -> None:
    @app.get("/", response_class=HTMLResponse, tags=["frontend"])
    async def index(request: Request) -> HTMLResponse:
        if app.state.templates is None:
            return HTMLResponse("<h1>ClawCodex Visualizer</h1><p>Templates not found.</p>")
        return app.state.templates.TemplateResponse(
            request,
            "index.html",
            {"workspaces": [], "page_title": "ClawCodex Visualizer"},
        )

    @app.get("/session/{session_id}", response_class=HTMLResponse, tags=["frontend"])
    async def session_detail(request: Request, session_id: str) -> HTMLResponse:
        if app.state.templates is None:
            return HTMLResponse(f"<h1>Session {session_id}</h1><p>Templates not found.</p>")
        return app.state.templates.TemplateResponse(
            request,
            "session_row.html",
            {"session_id": session_id, "page_title": f"Session {session_id}"},
        )


def _required_session(app: FastAPI, session_id: str) -> SessionVizData:
    viz = app.state.viz.timeline_builder.build(session_id)
    if viz is None:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")
    return viz


def _orchestrator_parser(app: FastAPI) -> Any:
    from .parsers.orchestrator_state_parser import (  # pylint: disable=import-outside-toplevel,relative-beyond-top-level
        OrchestratorStateParser,
    )

    return OrchestratorStateParser(reports_dir=app.state.viz.reports_dir)


def _read_report_file(path_value: str | None, media_type: str) -> Response:
    if not path_value:
        raise HTTPException(status_code=404, detail="Report artifact not found")
    path = Path(path_value)
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Report artifact not found")
    try:
        content = path.read_bytes()
    except OSError as exc:
        raise HTTPException(status_code=404, detail="Report artifact not readable") from exc
    return Response(content=content, media_type=media_type)


def _list_workspaces(app: FastAPI) -> list[WorkspaceInfo]:
    state: _AppState = app.state.viz
    session_dirs: list[Path] = []
    last_updated = 0.0
    if state.sessions_dir.is_dir():
        session_dirs = [entry for entry in state.sessions_dir.iterdir() if entry.is_dir()]
    for session_dir in session_dirs:
        try:
            last_updated = max(last_updated, session_dir.stat().st_mtime)
        except OSError:
            pass
    workspaces = [
        WorkspaceInfo(
            id="default",
            name="All sessions",
            path=str(state.sessions_dir),
            session_count=len(session_dirs),
            last_updated=last_updated,
        )
    ]
    if state.workspaces_file.exists():
        try:
            registered = json.loads(state.workspaces_file.read_text(encoding="utf-8"))
            for item in registered:
                if not isinstance(item, dict) or item.get("id") == "default":
                    continue
                workspaces.append(
                    WorkspaceInfo(
                        id=str(item.get("id", "")),
                        name=str(item.get("name") or item.get("id") or "Workspace"),
                        path=str(item.get("path", "")),
                        session_count=int(item.get("session_count", 0) or 0),
                        last_updated=float(item.get("last_updated", 0.0) or 0.0),
                    )
                )
        except Exception:
            logger.debug("Failed to parse workspaces.json", exc_info=True)
    return workspaces


def _list_sessions_in_workspace(app: FastAPI, workspace_id: str) -> list[SessionVizData]:
    state: _AppState = app.state.viz
    if not state.sessions_dir.is_dir():
        return []
    ranked: list[tuple[float, SessionVizData]] = []
    for session_dir in state.sessions_dir.iterdir():
        if not session_dir.is_dir():
            continue
        viz = state.timeline_builder.session_parser.parse(session_dir.name)
        if viz is None:
            continue
        if workspace_id != "default" and workspace_id.lower() not in (viz.workspace or "").lower():
            continue
        try:
            directory_mtime = session_dir.stat().st_mtime
        except OSError:
            directory_mtime = 0.0
        rank = max(viz.end_time or 0, viz.start_time or 0, directory_mtime)
        ranked.append((rank, viz))
    ranked.sort(key=lambda item: item[0], reverse=True)
    return [viz for _, viz in ranked]


def _session_matches_query(viz: SessionVizData, needle: str) -> bool:
    haystack = [
        viz.session_id,
        viz.title,
        viz.agent_name,
        viz.model,
        viz.provider,
        viz.workspace,
        viz.status,
        viz.verification_status,
        viz.issue_id,
        *viz.tags,
    ]
    return any(needle in str(value or "").lower() for value in haystack)


def _session_summary(viz: SessionVizData) -> dict[str, Any]:
    return {
        "session_id": viz.session_id,
        "title": viz.title,
        "workspace": viz.workspace,
        "model": viz.model,
        "provider": viz.provider,
        "status": viz.status,
        "agent_name": viz.agent_name,
        "start_time": viz.start_time,
        "end_time": viz.end_time,
        "duration_ms": viz.duration_ms,
        "turn_count": viz.turn_count,
        "tool_count": viz.tool_count,
        "tags": viz.tags,
        "parse_warnings": viz.parse_warnings,
        "issue_id": viz.issue_id or None,
        "verification_status": viz.verification_status or None,
    }


def mount_viz(app: FastAPI, prefix: str = "/viz", **kwargs: Any) -> None:
    """Mount the visualizer as a sub-application."""
    app.mount(prefix, create_app(**kwargs))


__all__ = ["create_app", "mount_viz"]
