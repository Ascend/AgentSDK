#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# -------------------------------------------------------------------------
#  This file is part of the AgentSDK project.
# Copyright (c) 2026 Huawei Technologies Co.,Ltd.
# Copyright (c) 2026 Clawd Codex Team
#
# AgentSDK is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#
#           http://license.coscl.org.cn/MulanPSL2
#
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
# -------------------------------------------------------------------------

"""Tests for opt-in telemetry IssueReporter."""

from __future__ import annotations

from typing import Any

from telemetry.config import ReportingConfig
from telemetry.redaction import RedactionConfig, Redactor
from telemetry.reporters.issue import (
    IssueReporter,
    _replace_or_append_date_block,
    _wrap_date_block,
)
from telemetry.storage import LocalJsonlStorage


class _FakePlatform:
    open_state = "open"


class _FakeClient:
    platform = _FakePlatform()

    def __init__(self, existing: dict[str, Any] | None = None, fail: bool = False) -> None:
        self.existing = existing
        self.fail = fail
        self.created: list[dict[str, Any]] = []
        self.updated: list[dict[str, Any]] = []
        self.find_titles: list[str] = []

    async def find_issue_by_title(self, title: str, *, state: str = "open") -> dict[str, Any] | None:
        if self.fail:
            raise RuntimeError("network down")
        self.find_titles.append(title)
        return self.existing

    async def create_issue(self, *, title: str, body: str, labels: list[str] | None = None) -> dict[str, Any]:
        if self.fail:
            raise RuntimeError("network down")
        payload = {"number": 101, "title": title, "body": body}
        self.created.append(payload)
        return payload

    async def update_issue_body(
        self,
        issue_id: str,
        *,
        title: str | None = None,
        body: str | None = None,
        labels: list[str] | None = None,
    ) -> dict[str, Any]:
        if self.fail:
            raise RuntimeError("network down")
        payload = {"number": issue_id, "title": title, "body": body}
        self.updated.append(payload)
        return payload


def _reporter(tmp_path, client: _FakeClient, **overrides: Any) -> tuple[IssueReporter, LocalJsonlStorage]:
    storage = LocalJsonlStorage(tmp_path / "telemetry", 7)
    config = {
        "reporting_enabled": True,
        "kind": "issue",
        "platform": "github",
        "owner": "acme",
        "repo": "widget",
        "api_key": "token",
    }
    config.update(overrides)
    cfg = ReportingConfig(**config)
    redactor = Redactor(RedactionConfig(), (str(tmp_path),))
    return IssueReporter(storage=storage, redactor=redactor, config=cfg, client=client), storage


def test_wrap_date_block_uses_stable_markers() -> None:
    block = _wrap_date_block("# Report\n", "2026-06-15")

    assert "<!-- clawcodex-telemetry:2026-06-15:begin -->" in block
    assert "# Report" in block
    assert "<!-- clawcodex-telemetry:2026-06-15:end -->" in block


def test_replace_or_append_date_block_replaces_existing_block() -> None:
    old = _wrap_date_block("old body\n", "2026-06-15")
    body = f"intro\n\n{old}\nfooter\n"
    new = _wrap_date_block("new body\n", "2026-06-15")

    updated = _replace_or_append_date_block(body, new, "2026-06-15")

    assert "new body" in updated
    assert "old body" not in updated
    assert updated.count("clawcodex-telemetry:2026-06-15:begin") == 1


def test_update_or_create_creates_inbox_issue_when_missing(tmp_path) -> None:
    client = _FakeClient(existing=None)
    reporter, storage = _reporter(tmp_path, client)

    assert reporter.emit("# Summary\n", date="2026-06-15") is True

    assert client.find_titles == ["ClawCodex Telemetry Inbox"]
    assert len(client.created) == 1
    assert "# Summary" in client.created[0]["body"]
    assert storage.read_reporter_cursor("issue")["issue_id"] == "101"


def test_update_or_create_updates_existing_date_block(tmp_path) -> None:
    existing_body = "intro\n\n" + _wrap_date_block("old\n", "2026-06-15")
    client = _FakeClient(existing={"number": 7, "title": "ClawCodex Telemetry Inbox", "body": existing_body})
    reporter, _storage = _reporter(tmp_path, client)

    assert reporter.emit("new\n", date="2026-06-15") is True

    assert not client.created
    assert len(client.updated) == 1
    assert "new" in client.updated[0]["body"]
    assert "old" not in client.updated[0]["body"]
    assert client.updated[0]["body"].count("clawcodex-telemetry:2026-06-15:begin") == 1


def test_create_daily_creates_daily_issue(tmp_path) -> None:
    client = _FakeClient(existing=None)
    reporter, _storage = _reporter(tmp_path, client, mode="create_daily", issue_title="Telemetry")

    assert reporter.emit("daily\n", date="2026-06-15") is True

    assert client.find_titles == ["Telemetry — 2026-06-15"]
    assert client.created[0]["title"] == "Telemetry — 2026-06-15"


def test_cursor_skip_avoids_http_call(tmp_path) -> None:
    client = _FakeClient(existing=None)
    reporter, storage = _reporter(tmp_path, client)

    assert reporter.emit("same\n", date="2026-06-15") is True
    assert reporter.emit("same\n", date="2026-06-15") is True

    assert len(client.find_titles) == 1
    assert len(client.created) == 1
    assert storage.read_reporter_cursor("issue")["date"] == "2026-06-15"


def test_secret_scan_blocks_upload_and_records_error(tmp_path) -> None:
    client = _FakeClient(existing=None)
    reporter, storage = _reporter(tmp_path, client)

    assert reporter.emit("leaked AKIAIOSFODNN7EXAMPLE\n", date="2026-06-15") is False

    assert not client.find_titles
    rows = storage.read_day("reporter_errors", "2026-06-15")
    assert rows and rows[0]["reason"] == "secret_scan"
    assert "rendered" not in rows[0]
    assert "AKIAIOSFODNN7EXAMPLE" not in str(rows[0])


def test_network_failure_returns_false_and_records_error(tmp_path) -> None:
    client = _FakeClient(fail=True)
    reporter, storage = _reporter(tmp_path, client)

    assert reporter.emit("# Summary\n", date="2026-06-15") is False

    rows = storage.read_day("reporter_errors", "2026-06-15")
    assert rows and rows[0]["reason"] == "request_failed"
    assert "# Summary" not in str(rows[0])


def test_missing_config_returns_false_and_records_error(tmp_path) -> None:
    client = _FakeClient(existing=None)
    reporter, storage = _reporter(tmp_path, client, owner="", api_key="")

    assert reporter.emit("# Summary\n", date="2026-06-15") is False

    assert not client.find_titles
    rows = storage.read_day("reporter_errors", "2026-06-15")
    assert rows and rows[0]["reason"] == "missing_config"
