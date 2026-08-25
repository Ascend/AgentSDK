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


"""Session API and report artifact route contracts."""

from __future__ import annotations

from .conftest import _create_minimal_session


def _get(client, path: str, status: int = 200):
    response = client.get(path)
    assert response.status_code == status
    return response


class TestSessionAPI:
    def test_get_session(self, client):
        data = _get(client, "/api/viz/sessions/test-session-001").json()
        assert data["session_id"] == "test-session-001"
        assert data["title"] == "Test Session"
        assert "timeline" in data

    def test_get_session_not_found(self, client):
        _get(client, "/api/viz/sessions/nonexistent", 404)

    def test_get_session_stats(self, client):
        data = _get(client, "/api/viz/sessions/test-session-001/stats").json()
        assert "total_ops" in data

    def test_get_session_anomalies(self, client):
        data = _get(client, "/api/viz/sessions/test-session-001/anomalies").json()
        assert isinstance(data, list)

    def test_get_session_tree(self, client):
        _get(client, "/api/viz/sessions/test-session-001/tree")

    def test_gantt_endpoint_is_removed(self, client):
        _get(client, "/api/viz/sessions/test-session-001/gantt", 404)

    def test_get_session_report_links(self, client):
        data = _get(client, "/api/viz/sessions/test-session-001/report").json()
        assert "export" not in data

    def test_session_report_links_urlencode_session_id(self, sessions_dir, client):
        session_id = "session with #hash"
        session_dir = _create_minimal_session(sessions_dir, session_id)
        (session_dir / "report.md").write_text("# Encoded Report\n", encoding="utf-8")
        (session_dir / "events.ndjson").write_text('{"event":"ok"}\n', encoding="utf-8")
        (session_dir / "debug.ndjson").write_text('{"debug":"ok"}\n', encoding="utf-8")

        links = _get(client, "/api/viz/sessions/session%20with%20%23hash/report").json()
        assert links["f38_report"] == "/api/viz/sessions/session%20with%20%23hash/report/f38"
        assert links["f45_events"] == "/api/viz/sessions/session%20with%20%23hash/report/f45"
        assert links["f54_debug"] == "/api/viz/sessions/session%20with%20%23hash/report/f54"
        assert client.get(links["f38_report"]).status_code == 200

    def test_session_report_links_are_resolvable(self, sessions_dir, client):
        session_dir = sessions_dir / "test-session-001"
        (session_dir / "report.md").write_text("# Report\n", encoding="utf-8")
        (session_dir / "events.ndjson").write_text('{"event":"ok"}\n', encoding="utf-8")
        (session_dir / "debug.ndjson").write_text('{"debug":"ok"}\n', encoding="utf-8")

        links = _get(client, "/api/viz/sessions/test-session-001/report").json()
        expected = (
            ("f38_report", "text/markdown", "# Report"),
            ("f45_events", "application/x-ndjson", '"event":"ok"'),
            ("f54_debug", "application/x-ndjson", '"debug":"ok"'),
        )
        for key, content_type, text in expected:
            response = _get(client, links[key])
            assert content_type in response.headers.get("content-type", "")
            assert text in response.text

    def test_missing_session_report_artifact_returns_404(self, client):
        _get(client, "/api/viz/sessions/test-session-001/report/f38", 404)
