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

"""Unit tests for :mod:`extensions.orchestrator.state_journal` and
:mod:`extensions.orchestrator.state_journal_sink`.

Covers:

* :class:`StateJournalWriter` — directory creation, NDJSON append
  semantics, automatic ``timestamp`` / ``run_id`` injection, ``close()``
  no-op behaviour, write-failure suppression, and the seven convenience
  helpers (``write_phase`` / ``write_issue_status`` / ``write_verification``
  / ``write_pr_status`` / ``write_session_ref`` / ``write_error`` /
  ``write_complete``).
* :class:`StateJournalSink` — translation of the three
  :class:`ProgressSink` callbacks into NDJSON events.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from extensions.orchestrator.state_journal import StateJournalWriter  # pylint: disable=import-error
from extensions.orchestrator.state_journal_sink import StateJournalSink


# ---------------------------------------------------------------------------
# StateJournalWriter
# ---------------------------------------------------------------------------


def _read_ndjson(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


class TestStateJournalWriterBasic(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.run_dir = Path(self.tmp.name) / "run_r1"
        self.writer = StateJournalWriter(self.run_dir, run_id="r1")

    def test_directory_is_created(self) -> None:
        self.assertTrue(self.run_dir.exists())

    def test_path_property(self) -> None:
        self.assertEqual(self.writer.path, self.run_dir / "state_journal.ndjson")

    def test_run_id_property(self) -> None:
        self.assertEqual(self.writer.run_id, "r1")

    def test_write_event_creates_file(self) -> None:
        self.writer.write_event({"type": "custom", "foo": "bar"})
        self.assertTrue(self.writer.path.exists())
        events = _read_ndjson(self.writer.path)
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["type"], "custom")
        self.assertEqual(events[0]["foo"], "bar")
        self.assertEqual(events[0]["run_id"], "r1")
        self.assertIn("timestamp", events[0])

    def test_write_event_injects_timestamp(self) -> None:
        self.writer.write_event({"type": "x"})
        events = _read_ndjson(self.writer.path)
        # ISO 8601 with Z suffix.
        self.assertRegex(events[0]["timestamp"], r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")

    def test_write_event_preserves_explicit_timestamp(self) -> None:
        self.writer.write_event({"type": "x", "timestamp": "2026-01-01T00:00:00Z"})
        events = _read_ndjson(self.writer.path)
        self.assertEqual(events[0]["timestamp"], "2026-01-01T00:00:00Z")

    def test_write_event_preserves_explicit_run_id(self) -> None:
        self.writer.write_event({"type": "x", "run_id": "other"})
        events = _read_ndjson(self.writer.path)
        self.assertEqual(events[0]["run_id"], "other")

    def test_write_event_appends(self) -> None:
        for i in range(5):
            self.writer.write_event({"type": "x", "i": i})
        events = _read_ndjson(self.writer.path)
        self.assertEqual([e["i"] for e in events], [0, 1, 2, 3, 4])

    def test_close_blocks_subsequent_writes(self) -> None:
        self.writer.write_event({"type": "x", "i": 0})
        self.writer.close()
        self.writer.write_event({"type": "x", "i": 1})
        events = _read_ndjson(self.writer.path)
        self.assertEqual(len(events), 1)

    def test_write_failure_does_not_propagate(self) -> None:
        # Force the open() call inside write_event to fail.
        with patch("builtins.open", side_effect=OSError("disk full")):
            with self.assertLogs("extensions.orchestrator.state_journal", level="DEBUG"):
                # Must not raise.
                self.writer.write_event({"type": "x"})

    def test_close_after_failure_still_blocks(self) -> None:
        self.writer.close()
        # Writes are no-ops even after errors.
        self.writer.write_event({"type": "x"})
        self.assertFalse(self.writer.path.exists())


class TestStateJournalWriterHelpers(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.run_dir = Path(self.tmp.name) / "run_r1"
        self.writer = StateJournalWriter(self.run_dir, run_id="r1")

    def test_write_phase_with_progress(self) -> None:
        self.writer.write_phase(phase="agent_run", progress=0.5, message="running", issue_id="i1")
        events = _read_ndjson(self.writer.path)
        self.assertEqual(events[0]["type"], "phase")
        self.assertEqual(events[0]["phase"], "agent_run")
        self.assertEqual(events[0]["progress"], 0.5)
        self.assertEqual(events[0]["message"], "running")
        self.assertEqual(events[0]["issue_id"], "i1")

    def test_write_phase_progress_rounds_to_three_dp(self) -> None:
        self.writer.write_phase(phase="x", progress=0.3456789)
        events = _read_ndjson(self.writer.path)
        self.assertEqual(events[0]["progress"], 0.346)

    def test_write_phase_omits_progress_when_none(self) -> None:
        self.writer.write_phase(phase="x", progress=None)
        events = _read_ndjson(self.writer.path)
        self.assertNotIn("progress", events[0])

    def test_write_phase_omits_issue_id_when_empty(self) -> None:
        self.writer.write_phase(phase="x", issue_id="")
        events = _read_ndjson(self.writer.path)
        self.assertNotIn("issue_id", events[0])

    def test_write_issue_status(self) -> None:
        self.writer.write_issue_status(issue_id="i1", status="running", message="ok")
        events = _read_ndjson(self.writer.path)
        self.assertEqual(events[0]["type"], "issue_status")
        self.assertEqual(events[0]["issue_id"], "i1")
        self.assertEqual(events[0]["status"], "running")
        self.assertEqual(events[0]["message"], "ok")

    def test_write_verification(self) -> None:
        self.writer.write_verification(issue_id="i1", verification_status="passed", result="ok")
        events = _read_ndjson(self.writer.path)
        self.assertEqual(events[0]["type"], "verification")
        self.assertEqual(events[0]["issue_id"], "i1")
        self.assertEqual(events[0]["verification_status"], "passed")
        self.assertEqual(events[0]["result"], "ok")

    def test_write_pr_status_default_open(self) -> None:
        self.writer.write_pr_status(issue_id="i1", pr_url="https://x")
        events = _read_ndjson(self.writer.path)
        self.assertEqual(events[0]["pr_status"], "open")

    def test_write_pr_status_with_number(self) -> None:
        self.writer.write_pr_status(issue_id="i1", pr_url="https://x", pr_status="merged", pr_number="42")
        events = _read_ndjson(self.writer.path)
        self.assertEqual(events[0]["pr_number"], "42")

    def test_write_pr_status_omits_number_when_none(self) -> None:
        self.writer.write_pr_status(issue_id="i1", pr_url="https://x", pr_number=None)
        events = _read_ndjson(self.writer.path)
        self.assertNotIn("pr_number", events[0])

    def test_write_session_ref(self) -> None:
        self.writer.write_session_ref(issue_id="i1", session_id="s1", session_path="/tmp/s1")
        events = _read_ndjson(self.writer.path)
        self.assertEqual(events[0]["type"], "session_ref")
        self.assertEqual(events[0]["issue_id"], "i1")
        self.assertEqual(events[0]["session_id"], "s1")
        self.assertEqual(events[0]["session_path"], "/tmp/s1")

    def test_write_error(self) -> None:
        self.writer.write_error(issue_id="i1", error="boom")
        events = _read_ndjson(self.writer.path)
        self.assertEqual(events[0]["type"], "error")
        self.assertEqual(events[0]["issue_id"], "i1")
        self.assertEqual(events[0]["error"], "boom")

    def test_write_complete(self) -> None:
        self.writer.write_complete(issue_id="i1", overall_status="success", message="all done")
        events = _read_ndjson(self.writer.path)
        self.assertEqual(events[0]["type"], "complete")
        self.assertEqual(events[0]["issue_id"], "i1")
        self.assertEqual(events[0]["overall_status"], "success")
        self.assertEqual(events[0]["message"], "all done")


class TestStateJournalWriterFailure(unittest.TestCase):
    def test_constructor_handles_uncreatable_dir(self) -> None:
        # Path under a file — mkdir will fail. Constructor must not raise.
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        # Make a file where we then try to make a subdir.
        blocker = Path(tmp.name) / "blocker"
        blocker.write_text("x", encoding="utf-8")
        # The run_dir would be blocker/sub — but blocker is a file.
        run_dir = blocker / "sub"
        with self.assertLogs("extensions.orchestrator.state_journal", level="WARNING"):
            # Should not raise.
            writer = StateJournalWriter(run_dir, run_id="r1")
        # Even without the dir, the writer object is constructed.
        self.assertEqual(writer.run_id, "r1")


# ---------------------------------------------------------------------------
# StateJournalSink
# ---------------------------------------------------------------------------


class TestStateJournalSink(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.run_dir = Path(self.tmp.name) / "run_r1"
        self.writer = StateJournalWriter(self.run_dir, run_id="r1")
        self.sink = StateJournalSink(writer=self.writer, task_id="task-1")

    def _read(self) -> list[dict]:
        return _read_ndjson(self.writer.path)

    def test_on_phase_complete_uses_event_phase(self) -> None:
        event = SimpleNamespace(phase="agent_run", progress=0.25, message="hi")
        self.sink.on_phase_complete(event, session=None)
        events = self._read()
        self.assertEqual(events[0]["type"], "phase")
        self.assertEqual(events[0]["phase"], "agent_run")
        self.assertEqual(events[0]["progress"], 0.25)
        self.assertEqual(events[0]["message"], "hi")
        self.assertEqual(events[0]["issue_id"], "task-1")

    def test_on_phase_complete_falls_back_to_counter(self) -> None:
        # No `phase` attribute → use counter (1-based). The writer
        # always stringifies the phase name, so the JSON value is "1".
        event = SimpleNamespace(progress=None, message="")
        self.sink.on_phase_complete(event, session=None)
        events = self._read()
        self.assertEqual(events[0]["phase"], "1")
        # Counter incremented.
        self.assertEqual(self.sink._phase_count, 1)

    def test_on_phase_complete_default_message(self) -> None:
        event = SimpleNamespace(phase="x", progress=None, message=None)
        self.sink.on_phase_complete(event, session=None)
        events = self._read()
        self.assertEqual(events[0]["message"], "Phase 1 completed")

    def test_on_phase_complete_counter_increments(self) -> None:
        for _ in range(3):
            self.sink.on_phase_complete(
                SimpleNamespace(progress=None, message=None),
                session=None,
            )
        events = self._read()
        self.assertEqual(len(events), 3)

    def test_on_turn_complete(self) -> None:
        event = SimpleNamespace(turn=5)
        self.sink.on_turn_complete(event, session=None)
        events = self._read()
        self.assertEqual(events[0]["type"], "phase")
        self.assertEqual(events[0]["phase"], "agent_turn")
        self.assertEqual(events[0]["message"], "Turn 5 completed")
        self.assertEqual(events[0]["issue_id"], "task-1")

    def test_on_turn_complete_missing_turn(self) -> None:
        event = SimpleNamespace()  # no turn attr
        self.sink.on_turn_complete(event, session=None)
        events = self._read()
        self.assertEqual(events[0]["message"], "Turn 0 completed")

    def test_on_session_complete(self) -> None:
        event = SimpleNamespace()
        session = SimpleNamespace(
            status="completed",
            session_end_reason="task_complete",
            session_end_summary="all green",
        )
        self.sink.on_session_complete(event, session=session)
        events = self._read()
        self.assertEqual(events[0]["type"], "complete")
        self.assertEqual(events[0]["issue_id"], "task-1")
        self.assertEqual(events[0]["overall_status"], "completed")
        self.assertIn("task_complete", events[0]["message"])
        self.assertIn("all green", events[0]["message"])

    def test_on_session_complete_default_status(self) -> None:
        event = SimpleNamespace()
        session = SimpleNamespace()  # no status
        self.sink.on_session_complete(event, session=session)
        events = self._read()
        self.assertEqual(events[0]["overall_status"], "completed")

    def test_on_session_complete_no_reason_no_summary(self) -> None:
        event = SimpleNamespace()
        session = SimpleNamespace(status="failed")
        self.sink.on_session_complete(event, session=session)
        events = self._read()
        self.assertEqual(events[0]["overall_status"], "failed")
        # With no reason/summary the formatted message is empty (the
        # `": "` separator is stripped).
        self.assertEqual(events[0]["message"], "")


if __name__ == "__main__":
    unittest.main()
