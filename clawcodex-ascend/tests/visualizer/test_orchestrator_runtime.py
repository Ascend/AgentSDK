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

"""Orchestrator journal parsing, rendering and HTTP route behavior."""

from __future__ import annotations

import json
from extensions.visualizer._rendering import panel
from extensions.visualizer.parsers.orchestrator_state_parser import OrchestratorStateParser


def test_orchestrator_parser_ignores_malformed_issue_event(tmp_path) -> None:
    reports = tmp_path / "reports"
    run_dir = reports / "run_20260804_120000"
    run_dir.mkdir(parents=True)
    events = [
        {
            "type": "orchestrator_start",
            "timestamp": "2026-08-04T12:00:00Z",
            "workflow": "review",
        },
        {
            "type": "issue_status",
            "timestamp": "2026-08-04T12:00:01Z",
            "status": "running",
        },
        {
            "type": "issue_status",
            "timestamp": "2026-08-04T12:00:02Z",
            "issue_id": "VIZ-1",
            "status": "running",
        },
        {
            "type": "session_ref",
            "timestamp": "2026-08-04T12:00:03Z",
            "issue_id": "VIZ-1",
            "session_id": "session-1",
            "session_path": "/tmp/session-1",
        },
    ]
    journal = run_dir / "state_journal.ndjson"
    journal.write_text(
        "\n".join(
            [
                json.dumps(events[0]),
                "not-json",
                *[json.dumps(item) for item in events[1:]],
            ]
        ),
        encoding="utf-8",
    )

    state = OrchestratorStateParser(reports_dir=reports).parse_run(run_dir.name)

    assert state is not None
    assert state.workflow == "review"
    assert state.event_count == 4
    assert state.issues["VIZ-1"].session_id == "session-1"
    assert OrchestratorStateParser(reports_dir=reports).get_current_snapshot()["run_id"] == run_dir.name


def test_panel_renders_fixed_width_rows() -> None:
    rendered = panel("State", ["ready"], width=24)
    lines = rendered.splitlines()

    assert "State" in rendered
    assert "ready" in rendered
    assert len(lines[0]) == len(lines[2]) == len(lines[-1]) == 24


class TestOrchestratorRoutes:
    def test_run_detail_preserves_issue_session_id(self, app, client, tmp_path):
        reports = tmp_path / "reports"
        run_dir = reports / "run_20260623_120000"
        run_dir.mkdir(parents=True)
        events = [
            {
                "type": "orchestrator_start",
                "timestamp": "2026-06-23T12:00:00Z",
                "workflow": "test",
            },
            {
                "type": "issue_status",
                "timestamp": "2026-06-23T12:00:01Z",
                "issue_id": "ISS-1",
                "status": "running",
            },
            {
                "type": "session_ref",
                "timestamp": "2026-06-23T12:00:02Z",
                "issue_id": "ISS-1",
                "session_id": "test-session-001",
                "session_path": "C:/tmp/test-session-001",
            },
        ]
        (run_dir / "state_journal.ndjson").write_text("\n".join(json.dumps(item) for item in events), encoding="utf-8")
        app.state.viz.reports_dir = reports

        resp = client.get("/api/viz/orchestrator/runs/run_20260623_120000")
        assert resp.status_code == 200
        issue = resp.json()["issues"]["ISS-1"]
        assert issue["session_id"] == "test-session-001"
        assert issue["session_path"].endswith("test-session-001")
