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

"""Unit tests for :mod:`extensions.orchestrator.report_writer`.

Covers the structured per-run report writer:

* :class:`RunReport` and :class:`ReportResult` dataclasses.
* :func:`write` dual-write (workspace ``.reports/`` + persistent
  ``~/.clawcodex/reports/{tracker}/{owner}/{repo}/{issue}/``) of both
  Markdown and JSON artefacts.
* tool-event dual-write (per-tool ``events.ndjson`` mirrored into
  the persistent layer).
* :func:`_safe_segment` path-segment sanitisation.
* :func:`_excerpt` truncation behaviour.
* :func:`_copy_with_fallback` primary vs fallback copy path.
* :func:`_render_markdown` content (run id, status, branch, PR,
  verification, output excerpt, tool events line).
"""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from extensions.orchestrator.report_writer import (
    ReportResult,
    RunReport,
    _copy_with_fallback,
    _excerpt,
    _render_markdown,
    _safe_segment,
    write,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_issue(
    issue_id: str = "42",
    identifier: str = "owner/repo#42",
    title: str = "Test issue",
) -> SimpleNamespace:
    return SimpleNamespace(id=issue_id, identifier=identifier, title=title)


def _patch_home(monkeypatch, home: Path) -> None:
    monkeypatch.setattr(Path, "home", lambda: home)


# ---------------------------------------------------------------------------
# Dataclass sanity
# ---------------------------------------------------------------------------


class TestRunReport(unittest.TestCase):
    def test_default_tool_events_path(self) -> None:
        report = RunReport(
            run_id="r1",
            tracker="t",
            owner=None,
            repo=None,
            issue_id="1",
            issue_identifier=None,
            issue_title=None,
            status="completed",
            branch_name=None,
            base_branch=None,
            commit_sha=None,
            pr_number=None,
            pr_url=None,
            turn_count=0,
            tool_count=0,
            verification_status=None,
            verification_output=None,
            output_excerpt="",
        )
        self.assertIsNone(report.tool_events_path)

    def test_tool_events_path_override(self) -> None:
        report = RunReport(
            run_id="r1",
            tracker="t",
            owner=None,
            repo=None,
            issue_id="1",
            issue_identifier=None,
            issue_title=None,
            status="completed",
            branch_name=None,
            base_branch=None,
            commit_sha=None,
            pr_number=None,
            pr_url=None,
            turn_count=0,
            tool_count=0,
            verification_status=None,
            verification_output=None,
            output_excerpt="",
            tool_events_path="/tmp/events.ndjson",
        )
        self.assertEqual(report.tool_events_path, "/tmp/events.ndjson")

    def test_report_result_is_frozen(self) -> None:
        result = ReportResult(
            run_id="r1",
            workspace_markdown_path="a",
            persistent_markdown_path="b",
            workspace_json_path="c",
            persistent_json_path="d",
        )
        with self.assertRaises(Exception):
            result.run_id = "changed"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# _safe_segment
# ---------------------------------------------------------------------------


class TestSafeSegment(unittest.TestCase):
    def test_alphanumeric_passes_through(self) -> None:
        self.assertEqual(_safe_segment("abc123"), "abc123")

    def test_dots_underscores_hyphens_kept(self) -> None:
        self.assertEqual(_safe_segment("a.b_c-d"), "a.b_c-d")

    def test_special_chars_replaced(self) -> None:
        self.assertEqual(_safe_segment("foo/bar:baz"), "foo-bar-baz")

    def test_stripped_leading_trailing_separators(self) -> None:
        self.assertEqual(_safe_segment("--foo--"), "foo")

    def test_empty_falls_back_to_unknown(self) -> None:
        self.assertEqual(_safe_segment(""), "unknown")
        self.assertEqual(_safe_segment("///"), "unknown")
        self.assertEqual(_safe_segment("..."), "unknown")

    def test_all_special_falls_back_to_unknown(self) -> None:
        self.assertEqual(_safe_segment("@#$%"), "unknown")


# ---------------------------------------------------------------------------
# _excerpt
# ---------------------------------------------------------------------------


class TestExcerpt(unittest.TestCase):
    def test_short_text_unchanged(self) -> None:
        self.assertEqual(_excerpt("hello", limit=10), "hello")

    def test_text_at_limit_unchanged(self) -> None:
        text = "x" * 100
        self.assertEqual(_excerpt(text, limit=100), text)

    def test_long_text_truncated(self) -> None:
        text = "x" * 5000
        result = _excerpt(text, limit=100)
        # Marked as truncated.
        self.assertIn("truncated from 5000 chars", result)
        # Tail preserved.
        self.assertIn("x" * 100, result)

    def test_excerpt_uses_tail_not_head(self) -> None:
        # Head is dropped, tail is preserved.
        text = "HEAD" + "x" * 200 + "TAIL"
        result = _excerpt(text, limit=50)
        self.assertNotIn("HEAD", result)
        self.assertIn("TAIL", result)


# ---------------------------------------------------------------------------
# _copy_with_fallback
# ---------------------------------------------------------------------------


class TestCopyWithFallback(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.src = Path(self.tmp.name) / "src.txt"
        self.src.write_text("hello", encoding="utf-8")
        self.dst = Path(self.tmp.name) / "dst.txt"

    def test_primary_copy_succeeds(self) -> None:
        _copy_with_fallback(self.src, self.dst)
        self.assertEqual(self.dst.read_text(), "hello")

    def test_fallback_path_uses_tmp_replace(self) -> None:
        # Force shutil.copy2 to fail → fallback should write a tmp + os.replace.
        import shutil as _shutil

        with patch.object(_shutil, "copy2", side_effect=OSError("boom")):
            _copy_with_fallback(self.src, self.dst)
        self.assertEqual(self.dst.read_text(), "hello")
        # No leftover .tmp files.
        leftovers = list(Path(self.tmp.name).glob("*.tmp"))
        self.assertEqual(leftovers, [])


# ---------------------------------------------------------------------------
# _render_markdown
# ---------------------------------------------------------------------------


class TestRenderMarkdown(unittest.TestCase):
    def _report(self, **overrides) -> RunReport:
        base = dict(
            run_id="r1",
            tracker="github",
            owner="octo",
            repo="hello",
            issue_id="42",
            issue_identifier="octo/hello#42",
            issue_title="Fix the bug",
            status="completed",
            branch_name="feat/fix",
            base_branch="main",
            commit_sha="abc123",
            pr_number="7",
            pr_url="https://x/y/pull/7",
            turn_count=3,
            tool_count=9,
            verification_status="passed",
            verification_output="ok",
            output_excerpt="...",
        )
        base.update(overrides)
        return RunReport(**base)

    def test_contains_core_fields(self) -> None:
        md = _render_markdown(self._report())
        self.assertIn("# ClawCodex Run Report", md)
        self.assertIn("`r1`", md)
        self.assertIn("octo/hello#42", md)
        self.assertIn("`completed`", md)
        self.assertIn("`feat/fix`", md)
        self.assertIn("`abc123`", md)
        self.assertIn("https://x/y/pull/7", md)
        self.assertIn("`passed`", md)

    def test_skipped_verification_marker(self) -> None:
        md = _render_markdown(self._report(verification_status=None))
        self.assertIn("`skipped`", md)

    def test_tool_events_line_included_when_set(self) -> None:
        md = _render_markdown(self._report(tool_events_path="/tmp/events.ndjson"))
        self.assertIn("Tool events:", md)
        self.assertIn("/tmp/events.ndjson", md)

    def test_tool_events_line_absent_when_none(self) -> None:
        md = _render_markdown(self._report())
        self.assertNotIn("Tool events:", md)

    def test_missing_branch_renders_n_a(self) -> None:
        md = _render_markdown(self._report(branch_name=None, base_branch=None))
        self.assertIn("`n/a`", md)


# ---------------------------------------------------------------------------
# write() — happy path
# ---------------------------------------------------------------------------


class TestWriteHappyPath(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.home = Path(self.tmp.name) / "home"
        self.home.mkdir()
        self.workspace = Path(self.tmp.name) / "workspace"
        self.workspace.mkdir()
        self._patcher = patch.object(Path, "home", lambda: self.home)
        self._patcher.start()
        self.addCleanup(self._patcher.stop)

    def _issue(self) -> SimpleNamespace:
        return _make_issue()

    def test_writes_workspace_md_and_json(self) -> None:
        result = write(
            run_id="r1",
            workspace_path=self.workspace,
            tracker="github",
            owner="octo",
            repo="hello",
            issue=self._issue(),
            status="completed",
        )
        # Workspace paths exist.
        self.assertTrue(os.path.exists(result.workspace_markdown_path))
        self.assertTrue(os.path.exists(result.workspace_json_path))
        # Workspace .reports dir was created.
        self.assertTrue((self.workspace / ".reports").exists())

    def test_writes_persistent_md_and_json(self) -> None:
        result = write(
            run_id="r1",
            workspace_path=self.workspace,
            tracker="github",
            owner="octo",
            repo="hello",
            issue=self._issue(),
            status="completed",
        )
        self.assertTrue(os.path.exists(result.persistent_markdown_path))
        self.assertTrue(os.path.exists(result.persistent_json_path))
        # Persistent path uses safe segments under ~/.clawcodex/reports/.
        self.assertIn(str(self.home), result.persistent_markdown_path)
        self.assertIn("github", result.persistent_markdown_path)

    def test_workspace_and_persistent_content_match(self) -> None:
        write(
            run_id="r1",
            workspace_path=self.workspace,
            tracker="github",
            owner="octo",
            repo="hello",
            issue=self._issue(),
            status="completed",
        )
        workspace_md = (self.workspace / ".reports" / "r1.md").read_text()
        persistent_md = (
            self.home / ".clawcodex" / "reports" / "github" / "octo" / "hello" / "42" / "r1.md"
        ).read_text()
        self.assertEqual(workspace_md, persistent_md)

    def test_json_payload_round_trip(self) -> None:
        write(
            run_id="r1",
            workspace_path=self.workspace,
            tracker="github",
            owner="octo",
            repo="hello",
            issue=_make_issue(identifier="octo/hello#42"),
            status="completed",
            commit_sha="abc",
            pr_number=99,
            pr_url="https://x/y/pull/99",
        )
        workspace_json = (self.workspace / ".reports" / "r1.json").read_text()
        payload = json.loads(workspace_json)
        self.assertEqual(payload["run_id"], "r1")
        self.assertEqual(payload["commit_sha"], "abc")
        self.assertEqual(payload["pr_number"], 99)
        self.assertEqual(payload["pr_url"], "https://x/y/pull/99")
        self.assertEqual(payload["issue_identifier"], "octo/hello#42")

    def test_report_result_paths(self) -> None:
        result = write(
            run_id="r1",
            workspace_path=self.workspace,
            tracker="github",
            owner="octo",
            repo="hello",
            issue=self._issue(),
            status="completed",
        )
        self.assertEqual(result.run_id, "r1")
        self.assertTrue(result.workspace_markdown_path.endswith("r1.md"))
        self.assertTrue(result.workspace_json_path.endswith("r1.json"))
        self.assertTrue(result.persistent_markdown_path.endswith("r1.md"))
        self.assertTrue(result.persistent_json_path.endswith("r1.json"))


# ---------------------------------------------------------------------------
# write() — edge cases
# ---------------------------------------------------------------------------


class TestWriteEdgeCases(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.home = Path(self.tmp.name) / "home"
        self.home.mkdir()
        self.workspace = Path(self.tmp.name) / "workspace"
        self.workspace.mkdir()
        self._patcher = patch.object(Path, "home", lambda: self.home)
        self._patcher.start()
        self.addCleanup(self._patcher.stop)

    def test_missing_issue_id_uses_unknown(self) -> None:
        issue = SimpleNamespace(id=None, identifier=None, title=None)
        result = write(
            run_id="r1",
            workspace_path=self.workspace,
            tracker="t",
            owner="o",
            repo="r",
            issue=issue,
            status="completed",
        )
        # safe_segment falls back to "unknown" for empty strings.
        self.assertIn("unknown", result.persistent_markdown_path)

    def test_unsafe_path_segments_are_sanitized(self) -> None:
        issue = SimpleNamespace(id="42/with spaces", identifier="x", title="t")
        result = write(
            run_id="r1",
            workspace_path=self.workspace,
            tracker="t",
            owner="o",
            repo="r",
            issue=issue,
            status="completed",
        )
        # The persistent path must not contain raw slashes from the issue id.
        self.assertNotIn("with spaces", result.persistent_markdown_path)
        self.assertIn("42-with-spaces", result.persistent_markdown_path)

    def test_missing_tracker_owner_repo_use_safe_defaults(self) -> None:
        result = write(
            run_id="r1",
            workspace_path=self.workspace,
            tracker="",
            owner="",
            repo="",
            issue=SimpleNamespace(id="1", identifier="x", title="t"),
            status="completed",
        )
        # All defaults should map to "unknown" / "local".
        self.assertIn("unknown", result.persistent_markdown_path)
        # "local" appears for both owner and repo segments.
        self.assertIn("local", result.persistent_markdown_path)

    def test_output_excerpt_is_truncated(self) -> None:
        # 10,000 chars > 4000 default limit.
        long_output = "x" * 10_000
        write(
            run_id="r1",
            workspace_path=self.workspace,
            tracker="t",
            owner="o",
            repo="r",
            issue=SimpleNamespace(id="1", identifier="x", title="t"),
            status="completed",
            output_text=long_output,
        )
        workspace_json = (self.workspace / ".reports" / "r1.json").read_text()
        payload = json.loads(workspace_json)
        self.assertIn("truncated from 10000 chars", payload["output_excerpt"])


# ---------------------------------------------------------------------------
# write() — tool events dual-write
# ---------------------------------------------------------------------------


class TestWriteToolEvents(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.home = Path(self.tmp.name) / "home"
        self.home.mkdir()
        self.workspace = Path(self.tmp.name) / "workspace"
        self.workspace.mkdir()
        self._patcher = patch.object(Path, "home", lambda: self.home)
        self._patcher.start()
        self.addCleanup(self._patcher.stop)

    def test_tool_events_copied_to_persistent(self) -> None:
        # Pre-create the source events file.
        events_src = self.home / "events.ndjson"
        events_src.write_text('{"tool": "bash"}\n', encoding="utf-8")
        result = write(
            run_id="r1",
            workspace_path=self.workspace,
            tracker="t",
            owner="o",
            repo="r",
            issue=SimpleNamespace(id="1", identifier="x", title="t"),
            status="completed",
            tool_events_path=str(events_src),
        )
        # Persistent events file should exist with same content.
        persistent_events = self.home / ".clawcodex" / "reports" / "t" / "o" / "r" / "1" / "r1.events.ndjson"
        self.assertTrue(persistent_events.exists())
        self.assertEqual(persistent_events.read_text(), '{"tool": "bash"}\n')
        # And the markdown report must mention the path.
        md = Path(result.workspace_markdown_path).read_text(encoding="utf-8")
        self.assertIn("Tool events:", md)

    def test_missing_tool_events_file_is_silently_skipped(self) -> None:
        # No file at the provided path — must not raise, and the persistent
        # events file must not be created. (The markdown report still
        # records the *path* on the report so the operator can investigate
        # why the audit log is missing; only the actual file copy is gated.)
        write(
            run_id="r1",
            workspace_path=self.workspace,
            tracker="t",
            owner="o",
            repo="r",
            issue=SimpleNamespace(id="1", identifier="x", title="t"),
            status="completed",
            tool_events_path="/does/not/exist.ndjson",
        )
        # Persistent events file must not exist.
        persistent_events = self.home / ".clawcodex" / "reports" / "t" / "o" / "r" / "1" / "r1.events.ndjson"
        self.assertFalse(persistent_events.exists())

    def test_none_tool_events_path_skips_copy_and_markdown_line(self) -> None:
        # when audit_log is "none" the runner never sets
        # tool_events_path; report_writer must skip both copy and mention.
        result = write(
            run_id="r1",
            workspace_path=self.workspace,
            tracker="t",
            owner="o",
            repo="r",
            issue=SimpleNamespace(id="1", identifier="x", title="t"),
            status="completed",
            tool_events_path=None,
        )
        persistent_events = self.home / ".clawcodex" / "reports" / "t" / "o" / "r" / "1" / "r1.events.ndjson"
        self.assertFalse(persistent_events.exists())
        md = Path(result.workspace_markdown_path).read_text(encoding="utf-8")
        self.assertNotIn("Tool events:", md)


# ---------------------------------------------------------------------------
# Idempotency / overwrite
# ---------------------------------------------------------------------------


class TestWriteOverwrite(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.home = Path(self.tmp.name) / "home"
        self.home.mkdir()
        self.workspace = Path(self.tmp.name) / "workspace"
        self.workspace.mkdir()
        self._patcher = patch.object(Path, "home", lambda: self.home)
        self._patcher.start()
        self.addCleanup(self._patcher.stop)

    def test_overwrite_existing_report(self) -> None:
        issue = SimpleNamespace(id="1", identifier="x", title="t")
        # First write.
        write(
            run_id="r1",
            workspace_path=self.workspace,
            tracker="t",
            owner="o",
            repo="r",
            issue=issue,
            status="running",
        )
        # Second write with different status — must overwrite without error.
        result = write(
            run_id="r1",
            workspace_path=self.workspace,
            tracker="t",
            owner="o",
            repo="r",
            issue=issue,
            status="completed",
        )
        payload = json.loads(Path(result.workspace_json_path).read_text(encoding="utf-8"))
        self.assertEqual(payload["status"], "completed")


if __name__ == "__main__":
    unittest.main()
