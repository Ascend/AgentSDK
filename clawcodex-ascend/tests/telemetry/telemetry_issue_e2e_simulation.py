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

"""Tests for telemetry issue e2e simulation."""

from __future__ import annotations

import os
import sys
import time
import tempfile
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve().parent.parent.parent
os.chdir(str(_HERE))
_tests_dir = str((_HERE / "tests").resolve())
sys.path = [str(_HERE)] + [p for p in sys.path if p and p != _tests_dir and os.path.realpath(p) != _tests_dir]


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
class _MockPlatform:
    """Tests for _MockPlatform."""

    open_state = "open"


class MockIssueClient:
    """Tests for MockIssueClient."""

    def __init__(self) -> None:
        self.platform = _MockPlatform()
        self.created: list[dict[str, Any]] = []
        self.updated: list[dict[str, Any]] = []
        self.find_titles: list[str] = []
        self._existing: dict[str, Any] | None = None

    def set_existing(self, issue: dict[str, Any] | None) -> None:
        """Test helper for set existing."""
        self._existing = issue

    async def find_issue_by_title(self, title: str, *, state: str = "open") -> dict[str, Any] | None:
        self.find_titles.append(title)
        return self._existing

    async def create_issue(self, *, title: str, body: str, labels: list[str] | None = None) -> dict[str, Any]:
        payload: dict[str, Any] = {"number": len(self.created) + 1, "title": title, "body": body}
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
        payload: dict[str, Any] = {"number": issue_id, "title": title, "body": body}
        self.updated.append(payload)
        return payload

    def summary(self) -> str:
        """Test helper for summary."""
        lines = [
            f"  find_issue_by_title 被调用 {len(self.find_titles)} 次",
            f"  create_issue        被调用 {len(self.created)} 次",
            f"  update_issue_body   被调用 {len(self.updated)} 次",
        ]
        if self.created:
            for c in self.created:
                lines.append(f"    → 创建 Issue #{c['number']}: {c['title']}")
        if self.updated:
            for u in self.updated:
                lines.append(f"    → 更新 Issue #{u['number']}: body 长度 {len(u['body'])}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
def build_events_for_day(storage: Any, date: str) -> None:
    """Test helper for build events for day."""

    from telemetry.events import TelemetryEvent, EventType

    # 2a. SESSION_START × 2
    for i, (platform, provider, model) in enumerate(
        [
            ("Linux", "anthropic", "claude-sonnet-4"),
            ("macOS", "openai", "gpt-4o"),
        ]
    ):
        storage.append(
            "events",
            TelemetryEvent(
                type=EventType.SESSION_START,
                session_id=f"sess-{i}",
                timestamp=time.time() - (i * 300),
                fields={
                    "entrypoint": "cli",
                    "platform": platform,
                    "provider": provider,
                    "model": model,
                    "client_type": "cli",
                    "is_non_interactive": False,
                    "app_version": "0.1.0",
                },
            ).to_dict(),
            date=date,
        )

    # 2b. COMMAND_RUN × 3
    for cmd, success, dur in [
        ("print", True, 2.5),
        ("agent", True, 45.0),
        ("print", False, 0.8),
    ]:
        storage.append(
            "events",
            TelemetryEvent(
                type=EventType.COMMAND_RUN,
                session_id="sess-0",
                timestamp=time.time(),
                fields={
                    "command_name": cmd,
                    "success": success,
                    "duration_s": dur,
                    "exit_status": 0 if success else 1,
                },
            ).to_dict(),
            date=date,
        )

    # 2c. TOOL_SUMMARY × 2
    for tool, success, dur in [
        ("Bash", True, 3.0),
        ("ReadFile", True, 0.5),
    ]:
        storage.append(
            "events",
            TelemetryEvent(
                type=EventType.TOOL_SUMMARY,
                session_id="sess-0",
                timestamp=time.time(),
                fields={
                    "tool_name": tool,
                    "success": success,
                    "duration_s": dur,
                },
            ).to_dict(),
            date=date,
        )

    # 2d. ERROR × 1
    storage.append(
        "crashes",
        TelemetryEvent(
            type=EventType.ERROR,
            session_id="sess-0",
            timestamp=time.time(),
            fields={
                "error_class": "ValueError",
                "fingerprint": "abc1234567890def",
                "stacktrace": ["ValueError: invalid input", "  File main.py:42"],
            },
        ).to_dict(),
        date=date,
    )


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
def main() -> int:
    print("=" * 72)
    print("Telemetry IssueReporter 端到端模拟")
    print("=" * 72)

    from telemetry.storage import LocalJsonlStorage
    from telemetry.config import ReportingConfig
    from telemetry.redaction import RedactionConfig, Redactor
    from telemetry.aggregator import DailyAggregator
    from telemetry.reporters.issue import IssueReporter

    with tempfile.TemporaryDirectory(prefix="telemetry-e2e-") as tmpdir:
        storage = LocalJsonlStorage(Path(tmpdir) / "telemetry", retention_days=7)
        date = "2026-07-15"
        print(f"\n📦 Storage root : {storage.base_dir}")
        print(f"📅 模拟日期     : {date}")

        print("\n--- 步骤 1: 写入模拟事件 ---")
        build_events_for_day(storage, date)
        print("  写入: 2 SESSION_START, 3 COMMAND_RUN, 2 TOOL_SUMMARY, 1 ERROR")

        print("\n--- 步骤 2: 运行 DailyAggregator ---")
        agg = DailyAggregator(storage)
        summary = agg.aggregate(date)
        assert summary, "aggregate() 返回了空 summary"
        print("  聚合结果:")
        print(f"    sessions      : {summary['sessions']}")
        print(f"    commands      : {summary['commands']}")
        print(f"    platforms     : {summary.get('platforms', {})}")
        print(f"    tools (top)   : {[t['name'] for t in summary.get('tools', {}).get('top', [])]}")
        print(f"    crashes total : {summary.get('crashes', {}).get('total', 0)}")
        assert summary["sessions"] == 2
        assert summary["commands"] == 3

        print("\n--- 步骤 3: 构造 IssueReporter (mode=update_or_create) ---")
        client = MockIssueClient()
        redactor = Redactor(RedactionConfig(), (str(Path(tmpdir)),))
        config = ReportingConfig(
            reporting_enabled=True,
            kind="issue",
            platform="github",
            owner="chadwweng",
            repo="clawcodex-telemetry",
            api_key="ghp_simulated_token_12345",
            mode="update_or_create",
            issue_title="ClawCodex Telemetry Inbox",
        )
        reporter = IssueReporter(
            storage=storage,
            redactor=redactor,
            config=config,
            client=client,
        )
        print(f"  config valid: {reporter._valid_config()}")

        print("\n--- 步骤 4: 首次 emit (期望创建 Issue) ---")
        rendered = reporter.render(summary, date)
        print(f"  渲染后 Markdown 长度: {len(rendered)} 字符")
        print("  渲染内容预览:")
        for line in rendered.splitlines()[:6]:
            print(f"    {line}")
        print("    ...")

        ok = reporter.emit(rendered, date=date)
        assert ok, "首次 emit 失败"
        print("  结果: emit 成功")
        print("  MockClient 状态:")
        print(client.summary())

        assert len(client.created) == 1, "预期创建 1 个 Issue"
        assert client.created[0]["title"] == "ClawCodex Telemetry Inbox"
        assert "ClawCodex Telemetry" in client.created[0]["body"]

        print("\n--- 步骤 5: 验证 reporter cursor ---")
        cursor = storage.read_reporter_cursor("issue")
        print(f"  cursor keys: {list(cursor.keys())}")
        assert cursor.get("issue_id") == "1", f"预期 issue_id=1, 实际={cursor.get('issue_id')}"
        assert cursor.get("date") == date
        assert cursor.get("reporter") == "issue"
        print(f"  ✅ cursor 已写入, issue_id={cursor['issue_id']}")

        print("\n--- 步骤 6: 去重测试 (相同内容再次 emit) ---")
        prev_find_count = len(client.find_titles)
        prev_create_count = len(client.created)

        ok2 = reporter.emit(rendered, date=date)
        assert ok2, "去重 emit 应返回 True"
        assert len(client.find_titles) == prev_find_count, "去重不应再调用 find_issue_by_title"
        assert len(client.created) == prev_create_count, "去重不应再创建 Issue"

        cursor2 = storage.read_reporter_cursor("issue")
        assert cursor2.get("issue_id") == "1"
        print("  ✅ 去重生效: 无额外 HTTP 调用")

        print("\n--- 步骤 7: 更新测试 (Issue 已存在, 追加新日期) ---")
        client2 = MockIssueClient()
        date2 = "2026-07-16"

        from telemetry.reporters.issue import _wrap_date_block

        existing_body = f"Intro\n\n{_wrap_date_block(rendered, date)}"
        client2.set_existing(
            {
                "number": "1",
                "title": "ClawCodex Telemetry Inbox",
                "body": existing_body,
            }
        )

        build_events_for_day(storage, date2)
        summary2 = agg.aggregate(date2)
        rendered2 = reporter.render(summary2, date2)

        reporter2 = IssueReporter(
            storage=storage,
            redactor=redactor,
            config=config,
            client=client2,
        )
        ok3 = reporter2.emit(rendered2, date=date2)
        assert ok3, "更新 emit 失败"

        print("  MockClient 状态:")
        print(client2.summary())
        assert len(client2.updated) == 1, "预期 1 次 update_issue_body 调用"
        assert client2.updated[0]["number"] == "1"
        updated_body = client2.updated[0]["body"]
        assert "clawcodex-telemetry:2026-07-15" in updated_body, "旧日期数据块应保留"
        assert "clawcodex-telemetry:2026-07-16" in updated_body, "新日期数据块应追加"
        print("  ✅ 更新成功: 新日期追加到已有 Issue body")

        print("\n--- 步骤 8: Secret scan 阻断测试 ---")
        client3 = MockIssueClient()
        reporter3 = IssueReporter(
            storage=storage,
            redactor=redactor,
            config=config,
            client=client3,
        )
        leaked = "This contains AKIAIOSFODNN7EXAMPLE key\n"
        ok4 = reporter3.emit(leaked, date=date)
        assert not ok4, "含 secret 的 emit 应返回 False"
        errors = storage.read_day("reporter_errors", date)
        secret_hit = any(r.get("reason") == "secret_scan" for r in errors)
        assert secret_hit, "reporter_errors 应记录 secret_scan"
        assert len(client3.find_titles) == 0, "含 secret 时不应调用任何 HTTP API"
        print("  ✅ Secret scan 正确阻断, reporter_errors 已记录")

        print("\n" + "=" * 72)
        print("✅ 所有场景通过")
        print("=" * 72)
        print("""
  场景清单:
  ┌─────────────────────────────────────────────┬────────┐
  │ 首次 emit 创建 Issue                       │ ✅     │
  │ reporter cursor 持久化                     │ ✅     │
  │ 相同内容去重 (skip HTTP)                   │ ✅     │
  │ 已有 Issue 追加新日期块                     │ ✅     │
  │ Secret scan 阻断 + 错误日志                 │ ✅     │
  │ aggregation 正确 (2 sessions, 3 commands)   │ ✅     │
  └─────────────────────────────────────────────┴────────┘
""")

    return 0


if __name__ == "__main__":
    sys.exit(main())
