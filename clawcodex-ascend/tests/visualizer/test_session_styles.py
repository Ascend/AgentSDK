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


"""Session timeline layout and interaction-style contracts."""

from __future__ import annotations


class TestSessionTimelineStyles:
    def test_session_page_actions_use_neutral_non_export_selector(self, client):
        css_resp = client.get("/static/css/session_view.css?v=20260624-inline-1")
        assert css_resp.status_code == 200
        assert ".session-page-actions" in css_resp.text
        assert ".session-export-actions" not in css_resp.text

    def test_session_view_zoom_reset_has_spacing(self, client):
        css_resp = client.get("/static/css/session_view.css?v=20260624-inline-1")
        assert css_resp.status_code == 200
        assert ".zoom-controls button + button { border-left: 1px solid var(--timeline-border); }" in css_resp.text
        assert "#zoom-reset { padding: 0 12px; }" in css_resp.text

    def test_subagent_toggle_arrow_is_centered_chevron(self, client):
        css_resp = client.get("/static/css/session_view.css?v=20260623-hit-target-1")
        assert css_resp.status_code == 200
        assert ".subagent-toggle-arrow {" in css_resp.text
        assert "position: relative; width: 20px; height: 20px" in css_resp.text
        assert 'content: ""; position: absolute; left: 50%; top: 50%' in css_resp.text
        assert "transform: translate(-50%, -50%) rotate(-45deg)" in css_resp.text
        assert "transform: translate(-50%, -50%) rotate(45deg)" in css_resp.text
