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

"""Typed HTTP route behavior for sessions, workspaces and share links."""

from __future__ import annotations

import json
from unittest.mock import Mock
from pathlib import Path
import pytest
from fastapi.testclient import TestClient
from .conftest import _create_minimal_session

from extensions.visualizer.server import create_app


def _create_share(client) -> str:
    response = client.post("/api/viz/share", json={"session_id": "test-session-001"})
    assert response.status_code == 200
    return response.json()["id"]


class TestHealthEndpoint:
    def test_health(self, client):
        resp = client.get("/api/viz/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["allow_import"] is True

    def test_cross_origin_request_is_not_authorized(self, client):
        resp = client.get(
            "/api/viz/health",
            headers={"Origin": "https://attacker.example"},
        )

        assert resp.status_code == 200
        assert "access-control-allow-origin" not in resp.headers
        assert "access-control-allow-credentials" not in resp.headers


class TestWorkspaces:
    def test_list_workspaces(self, client):
        resp = client.get("/api/viz/workspaces")
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, list)

    def test_missing_sessions_dir_still_lists_default_workspace(self, tmp_path):
        missing_dir = tmp_path / "missing-sessions"
        client = TestClient(create_app(sessions_dir=missing_dir, allow_import=False))

        resp = client.get("/api/viz/workspaces")
        assert resp.status_code == 200
        assert resp.json() == [
            {
                "id": "default",
                "name": "All sessions",
                "path": str(missing_dir),
                "session_count": 0,
                "last_updated": 0.0,
            }
        ]

        sessions_resp = client.get("/api/viz/workspaces/default/sessions")
        assert sessions_resp.status_code == 200
        assert sessions_resp.json() == []


class TestRemovedVisualizerSurfaces:
    @pytest.mark.parametrize(
        "path",
        [
            "/api/viz/compare?sessions=test-session-001",
            "/api/viz/compare/export",
            "/api/viz/sessions/test-session-001/export?format=json",
            "/api/viz/multi-session?sessions=test-session-001",
            "/api/viz/turn/test-session-001__tool-1/llm-io",
            "/compare",
            "/multi",
        ],
    )
    def test_removed_surface_returns_404_or_405(self, client, path):
        assert client.get(path).status_code in (404, 405)


class TestShareLinks:
    def test_comparison_share_is_rejected(self, client):
        resp = client.post(
            "/api/viz/share",
            json={
                "session_id": "test-session-001",
                "view_type": "comparison",
            },
        )
        assert resp.status_code == 400

    def test_create_and_get_share_link(self, client):
        link_id = _create_share(client)

        resp = client.get(f"/api/viz/share/{link_id}")
        assert resp.status_code == 200
        data = resp.json()
        assert "share" in data
        assert "data" in data
        assert "format" not in data["share"]

    def test_delete_share_link(self, client):
        link_id = _create_share(client)

        resp = client.delete(f"/api/viz/share/{link_id}")
        assert resp.status_code == 200

        resp = client.get(f"/api/viz/share/{link_id}")
        assert resp.status_code == 404

    def test_get_nonexistent_share(self, client):
        resp = client.get("/api/viz/share/nonexistent")
        assert resp.status_code == 404


class TestWorkspaceSearch:
    """Verify workspace session search and status filtering."""

    def test_list_workspace_sessions(self, client):
        resp = client.get("/api/viz/workspaces/default/sessions")
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, list)
        assert len(data) > 0
        assert "timeline" not in data[0]
        assert "session_id" in data[0]

    def test_search_sessions_by_query(self, client):
        resp = client.get("/api/viz/workspaces/default/sessions?q=test")
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, list)
        assert len(data) > 0

    def test_search_sessions_by_provider_workspace_status_and_tag(self, sessions_dir, client):
        session_dir = _create_minimal_session(sessions_dir, "search-rich")
        metadata = json.loads((session_dir / "metadata.json").read_text(encoding="utf-8"))
        metadata.update(
            {
                "provider": "needle-provider",
                "workspace": "needle-workspace",
                "status": "failed",
                "tags": ["needle-tag"],
            }
        )
        (session_dir / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")

        for query in ("needle-provider", "needle-workspace", "failed", "needle-tag"):
            resp = client.get(f"/api/viz/workspaces/default/sessions?q={query}")
            assert resp.status_code == 200
            ids = {item["session_id"] for item in resp.json()}
            assert "search-rich" in ids

    def test_filter_sessions_by_status(self, sessions_dir, client):
        session_dir = _create_minimal_session(sessions_dir, "failed-only")
        metadata = json.loads((session_dir / "metadata.json").read_text(encoding="utf-8"))
        metadata["status"] = "failed"
        (session_dir / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")

        failed_resp = client.get("/api/viz/workspaces/default/sessions?status=failed")
        assert failed_resp.status_code == 200
        failed_ids = {item["session_id"] for item in failed_resp.json()}
        assert "failed-only" in failed_ids
        assert "test-session-001" not in failed_ids

        all_resp = client.get("/api/viz/workspaces/default/sessions?status=all")
        assert all_resp.status_code == 200
        all_ids = {item["session_id"] for item in all_resp.json()}
        assert {"failed-only", "test-session-001"}.issubset(all_ids)

    def test_search_sessions_no_match(self, client):
        resp = client.get("/api/viz/workspaces/default/sessions?q=zzzzznonexistent")
        assert resp.status_code == 200
        assert resp.json() == []


class TestShareLinkPersistence:
    """Verify share-link persistence uses the configured local store."""

    def test_persistence_preserves_links_across_app_restart(self, sessions_dir, tmp_path, monkeypatch):
        monkeypatch.setattr(Path, "home", classmethod(lambda _cls: tmp_path))
        app1 = create_app(sessions_dir=sessions_dir, allow_import=False)
        client1 = TestClient(app1)

        resp = client1.post("/api/viz/share", json={"session_id": "test-session-001"})
        assert resp.status_code == 200
        link_id = resp.json()["id"]

        app1.state.viz._save_share_links()
        shares_path = app1.state.viz._shares_path
        assert shares_path.exists()

        app2 = create_app(sessions_dir=sessions_dir, allow_import=False)
        client2 = TestClient(app2)

        resp2 = client2.get(f"/api/viz/share/{link_id}")
        assert resp2.status_code == 200, "Share link should survive across app restarts"
        data = resp2.json()
        assert data["share"]["id"] == link_id


def test_get_session_builds_timeline_once(tmp_path) -> None:
    app = create_app(sessions_dir=tmp_path / "sessions", allow_import=False)
    app.state.viz.timeline_builder.build = Mock(return_value=None)
    client = TestClient(app)

    response = client.get("/api/viz/sessions/missing")

    assert response.status_code == 404
    app.state.viz.timeline_builder.build.assert_called_once_with("missing")


def test_dashboard_snapshot_degrades_without_adjacent_extension(tmp_path) -> None:
    app = create_app(sessions_dir=tmp_path / "sessions", allow_import=False)
    app.state.viz.dashboard_store = None
    client = TestClient(app)

    response = client.get("/api/dashboard/snapshot")

    assert response.status_code == 200
    assert response.json() == []
