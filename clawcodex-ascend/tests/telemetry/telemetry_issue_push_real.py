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

"""Tests for telemetry issue push real."""

from __future__ import annotations

import argparse
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
PLATFORM = "gitcode"
OWNER = "chadwweng"
REPO = "clawcodex"
ISSUE_TITLE = "Telemetry E2E Test — do not modify"
DATE = time.strftime("%Y-%m-%d", time.gmtime())


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
def build_events(storage: Any) -> None:
    """Test helper for build events."""
    from telemetry.events import TelemetryEvent, EventType

    now = time.time()
    # SESSION_START × 2
    for i, (pf, prov, mdl) in enumerate(
        [
            ("Linux", "anthropic", "claude-sonnet-4-20250514"),
            ("macOS", "openai", "gpt-4o"),
        ]
    ):
        storage.append(
            "events",
            TelemetryEvent(
                type=EventType.SESSION_START,
                session_id=f"push-test-sess-{i}",
                timestamp=now - i * 120,
                fields={
                    "entrypoint": "cli",
                    "platform": pf,
                    "provider": prov,
                    "model": mdl,
                    "client_type": "cli",
                    "is_non_interactive": False,
                    "app_version": "2026.6.24-e2e-test",
                },
            ).to_dict(),
            date=DATE,
        )

    # COMMAND_RUN × 4
    for cmd, ok, dur in [
        ("print", True, 1.2),
        ("agent", True, 32.0),
        ("print", False, 0.6),
        ("Bash", True, 5.0),
    ]:
        storage.append(
            "events",
            TelemetryEvent(
                type=EventType.COMMAND_RUN,
                session_id="push-test-sess-0",
                timestamp=now,
                fields={
                    "command_name": cmd,
                    "success": ok,
                    "duration_s": dur,
                    "exit_status": 0 if ok else 1,
                },
            ).to_dict(),
            date=DATE,
        )

    # TOOL_SUMMARY
    for tool, ok, dur in [("Bash", True, 3.0), ("ReadFile", True, 0.5), ("Edit", True, 1.2)]:
        storage.append(
            "events",
            TelemetryEvent(
                type=EventType.TOOL_SUMMARY,
                session_id="push-test-sess-0",
                timestamp=now,
                fields={"tool_name": tool, "success": ok, "duration_s": dur},
            ).to_dict(),
            date=DATE,
        )

    # ERROR (crashes)
    storage.append(
        "crashes",
        TelemetryEvent(
            type=EventType.ERROR,
            session_id="push-test-sess-0",
            timestamp=now,
            fields={
                "error_class": "ValueError",
                "fingerprint": "e2e-test-fingerprint-001",
                "stacktrace": ["ValueError: e2e test error", "  File test_cli.py:99"],
            },
        ).to_dict(),
        date=DATE,
    )


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser(description="Telemetry 真实 Issue 推送测试")
    parser.add_argument("--preview", action="store_true", help="仅预览，不推送")
    parser.add_argument("--close", action="store_true", help="推送后关闭 Issue")
    args = parser.parse_args()

    token = os.environ.get("CLAW_TELEMETRY_REPORTING_TOKEN") or os.environ.get("GITCODE_TOKEN") or ""
    if not token and not args.preview:
        print("❌ 需要设置 CLAW_TELEMETRY_REPORTING_TOKEN 或 GITCODE_TOKEN")
        print("   获取方式: https://gitcode.com/settings/tokens (需要 issue 权限)")
        return 1

    print("=" * 72)
    print("Telemetry IssueReporter 真实推送测试")
    print(f"  目标: {PLATFORM}.com/{OWNER}/{REPO}")
    print(f"  日期: {DATE}")
    print(f"  模式: {'预览 (不推送)' if args.preview else '真实推送'}")
    print("=" * 72)

    from telemetry.storage import LocalJsonlStorage
    from telemetry.config import ReportingConfig
    from telemetry.redaction import RedactionConfig, Redactor
    from telemetry.aggregator import DailyAggregator
    from telemetry.reporters.issue import IssueReporter
    from extensions.orchestrator.repo_tracker.client import RepositoryIssueClient

    tmpdir_obj = tempfile.TemporaryDirectory(prefix="telemetry-push-")
    try:
        storage = LocalJsonlStorage(Path(tmpdir_obj.name) / "telemetry", retention_days=7)

        print("\n--- 步骤 1: 写入模拟事件 ---")
        build_events(storage)
        print("  写入: 2 SESSION_START, 4 COMMAND_RUN, 3 TOOL_SUMMARY, 1 ERROR (crashes)")

        print("\n--- 步骤 2: DailyAggregator ---")
        agg = DailyAggregator(storage)
        summary = agg.aggregate(DATE)
        assert summary, "聚合失败"
        print(f"  sessions  : {summary['sessions']}")
        print(f"  commands  : {summary['commands']}")
        print(f"  platforms : {summary.get('platforms', {})}")
        print(f"  crashes   : {summary.get('crashes', {}).get('total', 0)}")
        assert summary["sessions"] == 2
        assert summary["commands"] == 4

        print("\n--- 步骤 3: 渲染 Markdown (DryRunReporter) ---")
        from telemetry.reporters.dry_run import DryRunReporter

        rendered = DryRunReporter().render(summary, DATE)
        print(f"  渲染长度: {len(rendered)} 字符")
        print("  --- 渲染内容 ---")
        for line in rendered.splitlines():
            print(f"  {line}")
        print("  ---")

        # Step 4: Secret scan
        print("\n--- 步骤 4: Secret scan 检查 ---")
        redactor = Redactor(RedactionConfig(), (str(Path(tmpdir_obj.name)),))
        hits = redactor.scan_secrets(rendered)
        if hits:
            print(f"  ⚠ Secret scan 命中: {hits}")
            print("  请检查事件数据中是否包含敏感信息")
            return 1
        print("  ✅ 无敏感内容")

        if args.preview:
            print("\n🔍 预览模式 — 未执行推送")
            print("   运行以下命令执行推送:")
            print("   CLAW_TELEMETRY_REPORTING_TOKEN=xxx python3 tests/telemetry/telemetry_issue_push_real.py")
            return 0

        print("\n--- 步骤 5: 构造 IssueReporter (真实 HTTP) ---")
        config = ReportingConfig(
            reporting_enabled=True,
            kind="issue",
            platform=PLATFORM,
            owner=OWNER,
            repo=REPO,
            api_key=token,
            mode="update_or_create",
            issue_title=ISSUE_TITLE,
        )
        reporter = IssueReporter(storage=storage, redactor=redactor, config=config)
        assert reporter._valid_config(), "配置无效"
        print(f"  配置有效: platform={PLATFORM} owner={OWNER} repo={REPO}")

        print("\n--- 步骤 6: 推送 Issue ---")
        timestamp = time.strftime("%H:%M:%S UTC", time.gmtime())
        body_with_tag = (
            f"{rendered.rstrip()}\n\n"
            f"_Test pushed at {timestamp} — "
            f"this issue was created by `tests/telemetry/telemetry_issue_push_real.py`._\n"
        )
        ok = reporter.emit(body_with_tag, date=DATE)
        assert ok, "emit 失败"
        cursor = storage.read_reporter_cursor("issue")
        issue_id = cursor.get("issue_id", "?")
        issue_url = f"https://{PLATFORM}.com/{OWNER}/{REPO}/issues/{issue_id}"
        print("  ✅ Issue 创建/更新成功")
        print(f"  Issue ID: #{issue_id}")
        print(f"  URL:      {issue_url}")
        print("  请访问以上 URL 查看渲染结果")

        cursor2 = storage.read_reporter_cursor("issue")
        print("\n--- 步骤 7: reporter cursor 验证 ---")
        print(f"  cursor issue_id = {cursor2.get('issue_id')}")
        print(f"  cursor date     = {cursor2.get('date')}")
        assert cursor2.get("issue_id") == issue_id
        assert cursor2.get("date") == DATE
        print("  ✅ cursor 已持久化")

        if args.close:
            import asyncio

            print(f"\n--- 步骤 8: 关闭 Issue #{issue_id} ---")
            client = RepositoryIssueClient(
                platform=PLATFORM,
                owner=OWNER,
                repo=REPO,
                api_key=token,
            )
            asyncio.run(client.update_issue(issue_id, state="closed"))
            print(f"  ✅ Issue #{issue_id} 已关闭")

        print("\n" + "=" * 72)
        print("✅ 测试完成")
        print(f"  查看 Issue: {issue_url}")
        if not args.close:
            print("  使用 --close 参数可在推送后自动关闭此 Issue")
        print("=" * 72)

    finally:
        tmpdir_obj.cleanup()

    return 0


if __name__ == "__main__":
    sys.exit(main())
