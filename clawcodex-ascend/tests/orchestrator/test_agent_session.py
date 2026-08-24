# -------------------------------------------------------------------------
# This file is part of the AgentSDK project.
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
#
# Copyright (c) 2026 Clawd Codex Team
# SPDX-License-Identifier: MIT
# Source: https://github.com/agentforce314/clawcodex
# ClawCodex-derived portions remain licensed under the MIT License.
# See clawcodex-ascend/LICENSE.clawcodex.
"""Focused tests for the Agent Runtime session and retry models."""

from __future__ import annotations

import json
import logging
from types import SimpleNamespace

from extensions.orchestrator.agent_session import (
    AgentSession,
    RetryItem,
    _has_user_visible_status_changes,
    _is_orchestrator_internal_path,
    _megaturn_idle_stop_enabled,
    _READ_ONLY_TOOL_NAMES,
)


def test_agent_session_defaults_are_isolated_per_instance() -> None:
    issue = SimpleNamespace(id="F-2")
    workspace = SimpleNamespace(path="/workspace")

    first = AgentSession(issue=issue, workspace=workspace)
    second = AgentSession(issue=issue, workspace=workspace)
    first.previous_run_ids.append("run-1")
    first._transcript_tool_uses.append("tool")

    assert first.status == "running"
    assert first.turn_count == 0
    assert first.has_made_progress is False
    assert second.previous_run_ids == []
    assert second._transcript_tool_uses == []


def test_retry_item_records_identity_and_defaults() -> None:
    item = RetryItem(
        issue_id="F-2",
        attempt=2,
        delay_seconds=15.0,
        identifier="ISSUE-2",
    )

    assert item.issue_id == "F-2"
    assert item.attempt == 2
    assert item.delay_seconds == 15.0
    assert item.scheduled_at > 0


def test_internal_path_classifier_normalizes_relative_and_windows_paths() -> None:
    assert _is_orchestrator_internal_path("./.reports/events.ndjson")
    assert _is_orchestrator_internal_path(".orchestrator_control\\runs\\state.json")
    assert not _is_orchestrator_internal_path("src/feature.py")
    assert not _is_orchestrator_internal_path(None)


def test_user_visible_status_change_ignores_runtime_artifacts() -> None:
    internal = SimpleNamespace(path=".reports/events.ndjson", original_path=None)
    source = SimpleNamespace(path="src/feature.py", original_path=None)

    assert not _has_user_visible_status_changes([internal])
    assert _has_user_visible_status_changes([internal, source])


def test_swarm_disables_megaturn_idle_stop() -> None:
    assert not _megaturn_idle_stop_enabled(SimpleNamespace(run_kind="swarm"))
    assert _megaturn_idle_stop_enabled(SimpleNamespace(run_kind="issue"))


def test_bash_is_not_classified_as_unconditionally_read_only() -> None:
    assert "Read" in _READ_ONLY_TOOL_NAMES
    assert "Bash" not in _READ_ONLY_TOOL_NAMES


def test_snapshot_without_run_id_is_a_noop(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)
    session = AgentSession(
        issue=SimpleNamespace(id="F-2"),
        workspace=SimpleNamespace(path=tmp_path),
    )

    session._save_json_snapshot()

    assert list(tmp_path.iterdir()) == []


def test_snapshot_uses_available_message_implementation(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)
    session = AgentSession(
        issue=SimpleNamespace(id="issue-2"),
        workspace=SimpleNamespace(path=tmp_path),
        run_id="run-2",
    )
    session._transcript_storage = SimpleNamespace(load_messages=lambda: [])

    session._save_json_snapshot()

    snapshot = tmp_path / ".clawcodex" / "sessions" / "run-2" / "session.json"
    payload = json.loads(snapshot.read_text(encoding="utf-8"))
    assert payload["session_id"] == "run-2"
    assert payload["conversation"]["messages"] == []


def test_snapshot_logs_transcript_load_failure(tmp_path, monkeypatch, caplog) -> None:
    monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)

    def _raise_load_error() -> list[dict]:
        raise RuntimeError("storage unavailable")

    session = AgentSession(
        issue=SimpleNamespace(id="issue-3"),
        workspace=SimpleNamespace(path=tmp_path),
        run_id="run-3",
    )
    session._transcript_storage = SimpleNamespace(load_messages=_raise_load_error)

    with caplog.at_level(logging.DEBUG, logger="extensions.orchestrator.agent_session"):
        session._save_json_snapshot()

    snapshot = tmp_path / ".clawcodex" / "sessions" / "run-3" / "session.json"
    assert snapshot.exists()
    assert "Failed to load transcript messages run_id=run-3" in caplog.text
