#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# -------------------------------------------------------------------------
# This file is part of the AgentSDK project.
#
# Originally from Clawd Codex:
# https://github.com/agentforce314/clawcodex
# Copyright (c) 2026 Clawd Codex Team
# Licensed under the MIT License. See clawcodex-ascend/LICENSE.clawcodex.
#
# Portions copyright (c) 2026 Huawei Technologies Co.,Ltd.
# Licensed under Mulan PSL v2. You may obtain a copy of Mulan PSL v2 at:
#
#          http://license.coscl.org.cn/MulanPSL2
#
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
# -------------------------------------------------------------------------

"""Orchestrator and agent-dashboard templates plus realtime contracts."""

from __future__ import annotations

from jinja2 import Environment, FileSystemLoader, select_autoescape
from pathlib import Path
import pytest


class TestLiveDashboardAssets:
    def test_orchestrator_dashboard_clamps_progress(self, dashboard_client):
        resp = dashboard_client.get("/viz/orchestrator")
        assert resp.status_code == 200
        assert "function progressPercent(value)" in resp.text
        assert "clampPercent(Math.round(Number(value) * 100))" in resp.text
        assert "progress=${progressPercent(ev.progress)}%" in resp.text
        assert "Math.round(ev.progress*100)" not in resp.text
        assert "return esc(s).replace(/\"/g, '&quot;');" in resp.text

    def test_orchestrator_issue_row_partial_is_hardened(self):
        repo_root = Path(__file__).resolve().parents[2]
        templates_dir = repo_root / "extensions" / "visualizer" / "templates"
        env = Environment(
            loader=FileSystemLoader(str(templates_dir)),
            autoescape=select_autoescape(["html"]),
        )
        template = env.get_template("orchestrator_issue_row.html")
        html = template.render(
            issue={
                "issue_id": 'ISS" data-x="1',
                "session_id": "session with spaces/<x>",
                "status": 'bad status" onclick="x',
                "progress": 1.7,
                "pr_url": "javascript:evil()",
                "error": "<boom>",
            }
        )

        assert "loadIssueTimeline(this.dataset.issueId)" in html
        assert "javascript:" not in html
        assert 'onclick="x' not in html
        assert "width:100%" in html
        assert "badge-unknown" in html
        assert "session%20with%20spaces" in html
        assert "&lt;boom&gt;" in html

    def test_websocket_client_urlencodes_session_id(self, dashboard_client):
        resp = dashboard_client.get("/static/js/websocket.js?v=20260623-ws-encode-1")
        assert resp.status_code == 200
        assert "/api/viz/ws/sessions/${encodeURIComponent(sessionId)}" in resp.text
        assert "/api/viz/ws/sessions/${sessionId}" not in resp.text
        assert "const generation = ++this.generation;" in resp.text
        assert "this.ws.onclose = null;" in resp.text


class _Entry:
    def __init__(self, entry_id: str, source: str, status: str) -> None:
        self._data = {
            "id": entry_id,
            "source": source,
            "title": entry_id,
            "status": status,
            "progress_pct": 0.5,
            "detail": "",
            "metadata": {},
        }

    def to_dict(self) -> dict[str, object]:
        return dict(self._data)


class _Store:
    def __init__(self) -> None:
        self.entries = [
            _Entry("goal:t1", "goal", "in_progress"),
            _Entry("goal:t2", "goal", "completed"),
            _Entry("task:1", "task", "pending"),
        ]
        self.sinks = []

    def snapshot(self, filters=None):
        filters = filters or {}
        return [
            entry for entry in self.entries if all(entry.to_dict().get(key) == value for key, value in filters.items())
        ]

    def subscribe(self, sink):
        self.sinks.append(sink)

        def unsubscribe():
            self.sinks.remove(sink)

        return unsubscribe

    def replace(self, entries):
        self.entries = entries
        for sink in list(self.sinks):
            sink(entries)


@pytest.fixture
def dashboard_store():
    return _Store()


@pytest.fixture
def dashboard_client(tmp_path, dashboard_store):
    from fastapi.testclient import TestClient
    from extensions.visualizer.server import create_app

    app = create_app(sessions_dir=tmp_path / "sessions", allow_import=False)
    app.state.viz.dashboard_store = dashboard_store
    return TestClient(app)


def test_dashboard_snapshot_returns_entries(dashboard_client) -> None:
    response = dashboard_client.get("/api/dashboard/snapshot")
    assert response.status_code == 200
    assert {item["id"] for item in response.json()} == {"goal:t1", "goal:t2", "task:1"}


def test_dashboard_snapshot_filters_by_source(dashboard_client) -> None:
    data = dashboard_client.get("/api/dashboard/snapshot?source=goal").json()
    assert {item["id"] for item in data} == {"goal:t1", "goal:t2"}


def test_dashboard_snapshot_filters_by_status(dashboard_client) -> None:
    data = dashboard_client.get("/api/dashboard/snapshot?status=completed").json()
    assert [item["id"] for item in data] == ["goal:t2"]


def test_dashboard_page_renders(dashboard_client) -> None:
    response = dashboard_client.get("/viz/dashboard")
    assert response.status_code == 200
    assert "Agent Dashboard" in response.text
    assert "if (!pollTimer) pollTimer = setInterval(poll, 3000);" in response.text
    assert "stopPolling();" in response.text
    assert "startPolling();" in response.text


def test_dashboard_websocket_sends_initial_snapshot(dashboard_client) -> None:
    with dashboard_client.websocket_connect("/api/viz/ws/dashboard/live") as websocket:
        message = websocket.receive_json()
    assert message["type"] == "dashboard_snapshot"
    assert len(message["entries"]) == 3


def test_dashboard_websocket_receives_push_on_change(dashboard_client, dashboard_store) -> None:
    with dashboard_client.websocket_connect("/api/viz/ws/dashboard/live") as websocket:
        websocket.receive_json()
        dashboard_store.replace([_Entry("extra:1", "extra", "pending")])
        pushed = websocket.receive_json()
    assert [item["id"] for item in pushed["entries"]] == ["extra:1"]


def test_dashboard_websocket_heartbeat(dashboard_client) -> None:
    with dashboard_client.websocket_connect("/api/viz/ws/dashboard/live") as websocket:
        websocket.receive_json()
        websocket.send_text("ping")
        assert websocket.receive_text() == "pong"
