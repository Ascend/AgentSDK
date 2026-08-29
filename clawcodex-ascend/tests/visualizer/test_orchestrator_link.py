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

"""Tests for OrchestratorLink artifact discovery."""

from __future__ import annotations

import json
import time

import pytest


@pytest.fixture
def session_dir(tmp_path):
    """Create a temporary session directory with sample data."""
    sd = tmp_path / "sessions" / "test-orch-001"
    sd.mkdir(parents=True)

    # Create a verification report
    (sd / "report.md").write_text("# Test Report\nAll good.", encoding="utf-8")

    # Create a tool-event audit log
    events = [
        json.dumps({"event": "tool_call", "tool_name": "Read", "timestamp": time.time()}),
        json.dumps({"event": "tool_result", "tool_name": "Read", "timestamp": time.time()}),
    ]
    (sd / "events.ndjson").write_text("\n".join(events), encoding="utf-8")

    # Create a debug timeline log
    debug = [
        json.dumps({"phase": "analysis", "timestamp": time.time()}),
    ]
    (sd / "debug.ndjson").write_text("\n".join(debug), encoding="utf-8")

    return sd


class TestOrchestratorLink:
    def test_generate_links(self, session_dir):
        from extensions.visualizer.orchestrator_link import OrchestratorLink

        link = OrchestratorLink(base_url="http://localhost:8765")
        result = link.generate_links("test-orch-001", session_dir)

        assert result["session_id"] == "test-orch-001"
        assert result["available"] is True
        assert "f38_report" in result
        assert "f45_events" in result
        assert "f54_debug" in result
        assert result["f45_events"]["event_count"] == 2
        assert result["f54_debug"]["entry_count"] == 1
        assert result["f38_report"]["api_url"].endswith("/report/f38")
        assert result["f45_events"]["api_url"].endswith("/report/f45")
        assert result["f54_debug"]["api_url"].endswith("/report/f54")

    def test_generate_links_nonexistent(self, tmp_path):
        from extensions.visualizer.orchestrator_link import OrchestratorLink

        link = OrchestratorLink()
        result = link.generate_links("nonexistent", tmp_path / "nope")

        assert result["available"] is False

    def test_generate_share_payload(self):
        from extensions.visualizer.orchestrator_link import OrchestratorLink

        link = OrchestratorLink()
        payload = link.generate_share_link("test-session")

        assert payload["session_id"] == "test-session"
        assert "view_type" not in payload

    def test_frontend_link(self, session_dir):
        from extensions.visualizer.orchestrator_link import OrchestratorLink

        link = OrchestratorLink(base_url="http://localhost:8765")
        result = link.generate_links("test-orch-001", session_dir)

        assert result["frontend"] == "http://localhost:8765/session/test-orch-001"

    def test_links_urlencode_session_id(self, tmp_path):
        from extensions.visualizer.orchestrator_link import OrchestratorLink

        session_id = "session with #hash"
        session_dir = tmp_path / session_id
        session_dir.mkdir()
        (session_dir / "report.md").write_text("# Report\n", encoding="utf-8")

        link = OrchestratorLink(base_url="http://localhost:8765")
        result = link.generate_links(session_id, session_dir)

        assert result["frontend"] == "http://localhost:8765/session/session%20with%20%23hash"
        assert (
            result["f38_report"]["api_url"]
            == "http://localhost:8765/api/viz/sessions/session%20with%20%23hash/report/f38"
        )
