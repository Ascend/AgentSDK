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

"""Session list, navigation and base-template asset contracts."""

from __future__ import annotations

from .conftest import _create_minimal_session


class TestSessionIndexUI:
    def test_base_no_longer_references_removed_frontend_assets(self, client):
        resp = client.get("/")
        assert resp.status_code == 200
        removed = (
            "multi_session.css",
            "gantt.js",
            "bezier_waterfall.js",
            "multi_session_view.js",
            "utils.js",
            "echarts",
        )
        for needle in removed:
            assert needle not in resp.text

    def test_frontend_assets_do_not_contain_mojibake_or_placeholders(self, client):
        paths = (
            "/",
            "/session/test-session-001",
            "/static/js/app.js?v=20260623-clean-copy-1",
            "/static/js/session_view.js?v=20260623-clean-copy-1",
        )
        bad = (
            "???",
            "\u951f",
            "\u9225",
            "\u6d93?",
            "\u701b?",
            "\u93c3?",
            "\u5a32?",
            "\u93c8?",
            "\u93b5?",
            "\u9435?",
            "\u7eef?",
        )
        for path in paths:
            resp = client.get(path)
            assert resp.status_code == 200
            for token in bad:
                assert token not in resp.text

    def test_index_page(self, client):
        resp = client.get("/")
        assert resp.status_code == 200
        assert "text/html" in resp.headers.get("content-type", "")
        assert "app.js?v=20260623-clean-copy-1" in resp.text
        assert "style.css?v=20260623-clean-base-1" in resp.text
        assert 'id="layout-toggle-grid" class="layout-toggle-btn active" aria-pressed="true"' in resp.text
        assert 'id="layout-toggle-list" class="layout-toggle-btn" aria-pressed="false"' in resp.text

    def test_index_app_renders_real_session_links(self, client):
        resp = client.get("/static/js/app.js?v=20260623-clean-copy-1")
        assert resp.status_code == 200
        assert 'class="session-card" href="${href}"' in resp.text
        assert 'class="session-row" href="${href}"' in resp.text
        assert 'role="link"' not in resp.text
        assert "data-session-id" not in resp.text
        assert "safeClassPart(session.status)" in resp.text
        assert "status-${escapeHtml(session.status)}" not in resp.text
        assert "const VALID_STATUSES" in resp.text
        assert "readChoice('viz.statusFilter', VALID_STATUSES, 'all')" in resp.text
        assert "readChoice('viz.layoutMode', VALID_LAYOUTS, 'grid')" in resp.text
        assert "localStorage.getItem('viz.statusFilter') || 'all'" not in resp.text

    def test_index_app_ignores_stale_session_responses(self, client):
        resp = client.get("/static/js/app.js?v=20260623-clean-copy-1")
        assert resp.status_code == 200
        assert "sessionRequestSeq: 0" in resp.text
        assert "const requestSeq = ++state.sessionRequestSeq;" in resp.text
        assert "const sessions = await requestJson(sessionsUrl());" in resp.text
        assert "if (requestSeq !== state.sessionRequestSeq) return;" in resp.text
        success_guard = resp.text.index(
            "if (requestSeq !== state.sessionRequestSeq) return;",
            resp.text.index("const sessions = await requestJson"),
        )
        catch_start = resp.text.index("} catch (error) {", success_guard)
        catch_guard = resp.text.index("if (requestSeq !== state.sessionRequestSeq) return;", catch_start)
        assign_start = resp.text.index("state.sessions = sessions;", success_guard)
        assert success_guard < assign_start < catch_start < catch_guard

    def test_index_app_sends_search_and_status_to_workspace_api(self, client):
        resp = client.get("/static/js/app.js?v=20260623-clean-copy-1")
        assert resp.status_code == 200
        assert "function sessionsUrl()" in resp.text
        assert "const params = new URLSearchParams();" in resp.text
        assert "if (query) params.set('q', query);" in resp.text
        assert "if (state.status !== 'all') params.set('status', state.status);" in resp.text
        assert "const sessions = await requestJson(sessionsUrl());" in resp.text
        assert "state.sessions = sessions;" in resp.text
        assert "state.query = event.target.value; loadSessions({ quiet: true });" in resp.text
        status_write = resp.text.index("writeChoice('viz.statusFilter', state.status);")
        status_reload = resp.text.index("loadSessions({ quiet: true });", status_write)
        assert status_write < status_reload

    def test_index_app_guards_storage_writes_and_refresh_order(self, client):
        resp = client.get("/static/js/app.js?v=20260623-clean-copy-1")
        assert resp.status_code == 200
        assert "const writeChoice = (key, value)" in resp.text
        assert "try { localStorage.setItem(key, value); } catch (_)" in resp.text
        assert "localStorage.setItem('viz.layoutMode', layout)" not in resp.text
        assert "localStorage.setItem('viz.statusFilter', state.status)" not in resp.text
        assert "await loadWorkspaces(); await loadSessions();" in resp.text
        assert (
            "loadWorkspaces().then(() => loadSessions({ quiet: true })).catch(() => setLiveState('error'))" in resp.text
        )
        initial_error = resp.text.index(
            "catch (error) {",
            resp.text.index("try { await loadWorkspaces(); await loadSessions(); }"),
        )
        interval_start = resp.text.index("state.timer = window.setInterval")
        assert "Cannot scan sessions" in resp.text[initial_error:interval_start]
        assert "setLiveState('error');" in resp.text[initial_error:interval_start]

    def test_index_app_cleans_up_background_polling(self, client):
        resp = client.get("/static/js/app.js?v=20260623-clean-copy-1")
        assert resp.status_code == 200
        assert "function stopPolling()" in resp.text
        assert "window.clearInterval(state.timer);" in resp.text
        assert "state.timer = null;" in resp.text
        assert "window.addEventListener('pagehide', stopPolling);" in resp.text
        assert "window.addEventListener('beforeunload', stopPolling);" in resp.text

    def test_base_uses_neutral_live_tail_status(self, client):
        resp = client.get("/")
        assert "Live idle" in resp.text
        assert "Offline" not in resp.text
        assert "websocket.js?v=20260623-ws-encode-1" in resp.text

    def test_session_page(self, client):
        resp = client.get("/session/test-session-001")
        assert resp.status_code == 200
        assert "text/html" in resp.headers.get("content-type", "")

    def test_session_page_uses_lane_assets(self, client):
        resp = client.get("/session/test-session-001")
        assert "session_view.css?v=20260624-inline-1" in resp.text
        assert "session_view.js?v=20260702-liveguard-1" in resp.text
        assert "echarts" not in resp.text.lower()

    def test_session_page_exposes_report_without_download_exports(self, client):
        resp = client.get("/session/test-session-001")
        assert 'class="session-page-actions"' in resp.text
        assert 'href="/api/viz/sessions/test-session-001/report"' in resp.text
        assert "/export" not in resp.text

    def test_session_page_urlencodes_report_link(self, sessions_dir, client):
        _create_minimal_session(sessions_dir, "session with #hash")
        resp = client.get("/session/session%20with%20%23hash")
        assert resp.status_code == 200
        assert 'href="/api/viz/sessions/session%20with%20%23hash/report"' in resp.text
        assert "/export" not in resp.text
