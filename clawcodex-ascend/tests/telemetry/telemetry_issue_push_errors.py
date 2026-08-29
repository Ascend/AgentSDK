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

"""Tests for telemetry issue push errors."""

from __future__ import annotations

import os
import sys
import time
import tempfile
import uuid
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve().parent.parent.parent
os.chdir(str(_HERE))
_tests_dir = str((_HERE / "tests").resolve())
sys.path = [str(_HERE)] + [p for p in sys.path if p and p != _tests_dir and os.path.realpath(p) != _tests_dir]

PLATFORM = "gitcode"
OWNER = "chadwweng"
REPO = "clawcodex"
ISSUE_TITLE = "Telemetry Error E2E Test"

PASS = 0
FAIL = 0
ERRORS: list[str] = []


def check(cond: bool, msg: str) -> None:
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  ✅ {msg}")
    else:
        FAIL += 1
        ERRORS.append(msg)
        print(f"  ❌ {msg}")


# ---------------------------------------------------------------------------
# Mock / real setup
# ---------------------------------------------------------------------------
def build_recorder(tmpdir: str, token: str, client: Any = None):
    """Build a real _TelemetryRecorderImpl wired to GitCode IssueReporter."""
    from telemetry import (
        TelemetryConfig,
        ReportingConfig,
        RedactionConfig,
        LocalJsonlStorage,
        DailyAggregator,
        Redactor,
        CompositeReporter,
    )
    from telemetry.recorder import _TelemetryRecorderImpl
    from telemetry.reporters.issue import IssueReporter

    storage = LocalJsonlStorage(Path(tmpdir) / "telemetry", retention_days=7)
    cfg = TelemetryConfig(
        enabled=True,
        storage_dir=Path(tmpdir) / "telemetry",
        retention_days=7,
        redaction=RedactionConfig(),
        reporting=ReportingConfig(
            reporting_enabled=True,
            kind="issue",
            platform=PLATFORM,
            owner=OWNER,
            repo=REPO,
            api_key=token,
            mode="update_or_create",
            issue_title=ISSUE_TITLE,
        ),
    )
    agg = DailyAggregator(storage)
    redactor = Redactor(cfg.redaction, (str(tmpdir),))
    reporters = CompositeReporter()

    if client is not None:
        reporters.add(client)
    else:
        reporters.add(
            IssueReporter(
                storage=storage,
                redactor=redactor,
                config=cfg.reporting,
            )
        )

    return (
        _TelemetryRecorderImpl(
            cfg=cfg,
            storage=storage,
            aggregator=agg,
            redactor=redactor,
            reporters=reporters,
        ),
        storage,
        cfg,
    )


def close_issue(token: str, issue_id: str) -> None:
    """Close the test Issue via GitCode API."""
    import httpx

    r = httpx.patch(
        f"https://api.gitcode.com/api/v5/repos/{OWNER}/{REPO}/issues/{issue_id}",
        params={"access_token": token},
        data={"state_event": "close", "title": ISSUE_TITLE},
        headers={"Accept": "application/json"},
    )
    # GitCode accepts the close but doesn't actually close — that's a known
    # platform limitation. We at least try.
    print(f"    close attempt: HTTP {r.status_code}")


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
def test_cli_error_record(tmpdir: str, token: str) -> dict[str, Any] | None:
    print("\n" + "=" * 72)
    print("场景 1: CLI 命令报错退出")
    print("  模拟: record_error() + flush()")
    print("=" * 72)

    from telemetry.recorder import override_recorder, reset_recorder_for_tests

    reset_recorder_for_tests()

    emitted: list[tuple[str, str]] = []

    class _TrackingReporter:
        def render(self, summary: dict, date: str) -> str:
            return ""

        def emit(self, rendered: str, *, date: str) -> bool:
            emitted.append((rendered, date))
            return True

    recorder, storage, cfg = build_recorder(tmpdir, token, client=_TrackingReporter())
    override_recorder(recorder)

    sid = uuid.uuid4().hex
    recorder.record_session_start(session_id=sid, entrypoint="cli")
    try:
        raise ValueError("test error: invalid input data")
    except ValueError as exc:
        recorder.record_error(session_id=sid, exc=exc)

    check(len(emitted) == 0, "record_error 后不应自动 emit（需等 flush）")

    recorder.flush()
    check(len(emitted) >= 1, "flush 后应有 emit")
    check(emitted[0][1] == time.strftime("%Y-%m-%d"), f"emit date = today ({emitted[0][1]})")

    reset_recorder_for_tests()
    print(f"  → 共 {len(emitted)} 次 emit")
    return None


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
def test_unhandled_exception_via_hooks(tmpdir: str, token: str) -> dict[str, Any] | None:
    print("\n" + "=" * 72)
    print("场景 2: 未捕获异常 → sys.excepthook")
    print("  模拟: install_exception_hooks → raise → hooks._emit()")
    print("  hooks._emit 仅 record_error，flush 由退出时 cleanup 触发")
    print("=" * 72)

    from telemetry.recorder import override_recorder, reset_recorder_for_tests
    from telemetry.hooks import install_exception_hooks, uninstall_exception_hooks

    reset_recorder_for_tests()

    emitted: list[tuple[str, str]] = []

    class _TrackingReporter:
        last_rendered = ""

        def render(self, summary: dict, date: str) -> str:
            return ""

        def emit(self, rendered: str, *, date: str) -> bool:
            emitted.append((rendered, date))
            self.last_rendered = rendered
            return True

    recorder, storage, cfg = build_recorder(tmpdir, token, client=_TrackingReporter())
    override_recorder(recorder)
    install_exception_hooks()

    try:
        raise RuntimeError("simulated unhandled exception for telemetry test")
    except RuntimeError as exc:
        from telemetry.hooks import _emit as hook_emit

        hook_emit(exc)

    check(len(emitted) == 0, "hooks._emit() 不应立即 emit（由 cleanup 触发）")

    today = time.strftime("%Y-%m-%d")
    crashes = storage.read_day("crashes", today)
    check(len(crashes) >= 1, f"crashes 文件中有 {len(crashes)} 条 error 事件")

    recorder.flush()
    check(len(emitted) >= 1, "flush() 后应有 emit")

    uninstall_exception_hooks()
    reset_recorder_for_tests()
    return None


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
def test_shutdown_cleanup_only_on_error(tmpdir: str, token: str) -> dict[str, Any] | None:
    print("\n" + "=" * 72)
    print("场景 3: shutdown cleanup — 仅在有 error 时 flush")
    print("  子场景 3a: 正常退出（无 error）→ 不应 flush")
    print("  子场景 3b: 有 error → 应 flush")
    print("=" * 72)

    from telemetry.recorder import override_recorder, reset_recorder_for_tests
    from clawcodex_ext.telemetry_lifecycle import _telemetry_shutdown_flush

    reset_recorder_for_tests()

    emitted: list[tuple[str, str]] = []

    class _TrackingReporter:
        def render(self, summary: dict, date: str) -> str:
            return ""

        def emit(self, rendered: str, *, date: str) -> bool:
            emitted.append((rendered, date))
            return True

    recorder_ok, storage_ok, _ = build_recorder(tmpdir + "/ok", token, client=_TrackingReporter())
    override_recorder(recorder_ok)
    emitted.clear()

    sid = uuid.uuid4().hex
    recorder_ok.record_session_start(session_id=sid, entrypoint="cli")
    recorder_ok.record_session_end(session_id=sid, duration_s=1.0, exit_status=0)
    recorder_ok.record_command_run(session_id=sid, command_name="print", mode="test")

    _telemetry_shutdown_flush()
    check(len(emitted) == 0, "3a: 正常退出（无 error）→ 不 emit")

    reset_recorder_for_tests()

    recorder_err, storage_err, _ = build_recorder(tmpdir + "/err", token, client=_TrackingReporter())
    override_recorder(recorder_err)
    emitted.clear()

    sid2 = uuid.uuid4().hex
    recorder_err.record_session_start(session_id=sid2, entrypoint="cli")
    try:
        raise ConnectionError("network timeout in e2e test")
    except ConnectionError as exc:
        recorder_err.record_error(session_id=sid2, exc=exc)

    _telemetry_shutdown_flush()
    import time

    for _ in range(25):
        if emitted:
            break
        time.sleep(0.2)
    check(len(emitted) >= 1, "3b: 有 error → 应 emit")

    reset_recorder_for_tests()
    return None


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
def test_real_issue_push(tmpdir: str, token: str) -> None:
    print("\n" + "=" * 72)
    print("场景 4: 真实 Issue 推送到 GitCode")
    print(f"  目标: {PLATFORM}.com/{OWNER}/{REPO}")
    print("=" * 72)

    from telemetry.recorder import override_recorder, reset_recorder_for_tests

    reset_recorder_for_tests()

    recorder, storage, _ = build_recorder(tmpdir, token)
    override_recorder(recorder)

    sid = uuid.uuid4().hex
    recorder.record_session_start(session_id=sid, entrypoint="cli")
    try:
        raise RuntimeError("E2E test error: something went wrong in telemetry push")
    except RuntimeError as exc:
        recorder.record_error(session_id=sid, exc=exc)

    recorder.flush()

    cursor = storage.read_reporter_cursor("issue")
    issue_id = cursor.get("issue_id", "")
    check(bool(issue_id), f"Issue 已创建，id={issue_id}")
    print(f"  Issue URL: https://{PLATFORM}.com/{OWNER}/{REPO}/issues/{issue_id}")

    if issue_id:
        import httpx

        r = httpx.get(
            f"https://api.gitcode.com/api/v5/repos/{OWNER}/{REPO}/issues/{issue_id}",
            params={"access_token": token},
            headers={"Accept": "application/json"},
        )
        check(r.status_code == 200, f"API 可读取 Issue (HTTP {r.status_code})")
        if r.status_code == 200:
            body = r.json().get("body", "")
            check("RuntimeError" in body, "Issue body 包含 error_class=RuntimeError")
            check("Exit status counts: error=1" in body, "Issue body 包含 error 计数")
            check("clawcodex-telemetry:" in str(body), "Issue body 含 marker 标签")
            check("## Error report" in body, "Issue body 含 Error report 章节")
            check("Fingerprint" in body, "Issue body 含 fingerprint hash（privacy-safe）")
            print(f"  body 长度: {len(body)} 字符")
            print(f"  body 摘录:\n{body[:400]}\n  ...")

            close_issue(token, issue_id)

    reset_recorder_for_tests()


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main() -> int:
    token = os.environ.get("CLAW_TELEMETRY_REPORTING_TOKEN") or os.environ.get("GITCODE_TOKEN") or ""
    if not token:
        print("❌ 需要设置 CLAW_TELEMETRY_REPORTING_TOKEN 或 GITCODE_TOKEN")
        return 1

    with tempfile.TemporaryDirectory(prefix="telemetry-error-e2e-") as tmpdir:
        test_cli_error_record(tmpdir, token)

        test_unhandled_exception_via_hooks(tmpdir, token)

        test_shutdown_cleanup_only_on_error(tmpdir, token)

        test_real_issue_push(tmpdir + "/real", token)

    print("\n" + "=" * 72)
    total = PASS + FAIL
    print(f"结果: {PASS}/{total} 通过", end="")
    if FAIL:
        print(f", {FAIL} 失败:")
        for e in ERRORS:
            print(f"  - {e}")
    else:
        print(" ✅")
    print("=" * 72)
    return 0 if FAIL == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
