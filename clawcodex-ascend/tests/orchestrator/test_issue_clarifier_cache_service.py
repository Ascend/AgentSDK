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

"""Focused tests for issue clarification caching and analysis behavior."""

from types import SimpleNamespace

import logging
from pathlib import Path

from extensions.orchestrator.issue_clarifier.cache import ClarifierCache
from extensions.orchestrator.issue_clarifier.models import ClarifyResult
from extensions.orchestrator.issue_clarifier.service import (
    IssueClarifierService,
    format_clarification_request,
)


def _config(**overrides):
    values = {
        "max_questions": 3,
        "max_input_tokens": 2000,
        "max_output_tokens": 500,
        "min_confidence": 0.7,
        "fail_open": True,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _issue(description: str):
    return SimpleNamespace(
        id="124",
        identifier="ISSUE-124",
        title="Clarify migration",
        description=description,
        labels=[],
    )


def test_explicit_gap_is_detected_without_provider(tmp_path) -> None:
    cache = ClarifierCache(tmp_path / "clarifier.json")
    service = IssueClarifierService(config=_config(), cache=cache)

    result = service.analyze(_issue("The target is TBD; ask the issue author."))

    assert not result.is_clear
    assert result.metadata["deterministic_gate"] == "explicit_gap"
    body, options = format_clarification_request(result)
    assert "please clarify" in body.lower()
    assert options == []
    cached = cache.get(result.fingerprint)
    assert cached is not None
    assert cached.cached
    assert cached.questions == result.questions


def test_provider_failure_obeys_fail_open(tmp_path) -> None:
    cache = ClarifierCache(tmp_path / "clarifier.json")
    service = IssueClarifierService(config=_config(fail_open=True), cache=cache)

    result = service.analyze(_issue("Implement the documented behavior."))

    assert result.is_clear
    assert result.degraded
    assert "unavailable" in result.reason


def test_provider_failure_without_issue_id_preserves_fail_open(tmp_path, caplog) -> None:
    cache = ClarifierCache(tmp_path / "clarifier.json")
    service = IssueClarifierService(config=_config(fail_open=True), cache=cache)
    issue = _issue("Implement the documented behavior.")
    del issue.id

    with caplog.at_level(logging.WARNING):
        result = service.analyze(issue)

    assert result.is_clear
    assert result.degraded
    assert "issue ?:" in caplog.text


def test_cache_load_distinguishes_corruption_from_filesystem_failure(tmp_path, caplog, monkeypatch) -> None:
    path = tmp_path / "clarifier.json"
    path.write_text("{", encoding="utf-8")

    with caplog.at_level(logging.WARNING):
        ClarifierCache(path)

    assert "Ignoring corrupted issue clarifier cache" in caplog.text
    caplog.clear()

    def deny_read(_path: Path, *_args, **_kwargs) -> str:
        raise PermissionError("read denied")

    monkeypatch.setattr(Path, "read_text", deny_read)
    with caplog.at_level(logging.WARNING):
        ClarifierCache(path)

    assert "filesystem error" in caplog.text


def test_cache_save_reports_filesystem_failure(tmp_path, caplog, monkeypatch) -> None:
    cache = ClarifierCache(tmp_path / "clarifier.json")

    def deny_mkdir(_path: Path, *_args, **_kwargs) -> None:
        raise PermissionError("write denied")

    monkeypatch.setattr(Path, "mkdir", deny_mkdir)
    with caplog.at_level(logging.WARNING):
        cache.put(ClarifyResult(is_clear=True, fingerprint="fingerprint"))

    assert "filesystem error" in caplog.text


def test_provider_failure_in_fail_closed_mode_returns_question(tmp_path) -> None:
    cache = ClarifierCache(tmp_path / "clarifier.json")
    service = IssueClarifierService(config=_config(fail_open=False), cache=cache)

    result = service.analyze(_issue("Implement the documented behavior."))

    assert not result.is_clear
    assert result.degraded
    assert result.questions
    body, _ = format_clarification_request(result)
    assert "acceptance criteria" in body


def test_workspace_focus_changes_cache_fingerprint(tmp_path) -> None:
    service = IssueClarifierService(
        config=_config(),
        cache=ClarifierCache(tmp_path / "clarifier.json"),
    )
    issue = _issue("Implement the documented behavior.")

    first = service.fingerprint(
        issue,
        workspace_focuses=[{"path": "src/first.py"}],
    )
    second = service.fingerprint(
        issue,
        workspace_focuses=[{"path": "src/second.py"}],
    )

    assert first != second


def test_provider_type_error_is_not_replayed(tmp_path) -> None:
    class Provider:
        def __init__(self) -> None:
            self.calls = 0

        def chat(self, **_kwargs):
            self.calls += 1
            raise TypeError("response parser failed after request")

    provider = Provider()
    service = IssueClarifierService(
        config=_config(fail_open=True),
        cache=ClarifierCache(tmp_path / "clarifier.json"),
        provider=provider,
    )

    result = service.analyze(_issue("Implement the documented behavior."))

    assert provider.calls == 1
    assert result.is_clear
    assert result.degraded
