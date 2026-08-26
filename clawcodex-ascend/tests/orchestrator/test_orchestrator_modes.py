#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# -------------------------------------------------------------------------
#  This file is part of the AgentSDK project.
# Copyright (c) 2026 Huawei Technologies Co.,Ltd.
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

"""Phase-1 unit tests for the collaboration-mode framework.

These cover the new abstractions added in (multi-agent modes):

* ``extensions.orchestrator.modes`` — registry (register / get / available)
* ``extensions.orchestrator.modes.base`` — Protocol + ``ModeDecision``
* ``extensions.orchestrator.modes.single`` — pass-through runner
* ``extensions.orchestrator.mode_selector`` — label-driven decision
* ``IssueRecord`` — new ``collaboration_mode`` /
  ``mode_decision_reason`` fields + back-compat load

The tests intentionally do NOT spin up a full Orchestrator: that's the
job of the existing 270+ integration tests (which keep passing because
Single mode is a literal pass-through). Here we verify the building
blocks themselves.
"""

from __future__ import annotations

import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

from extensions.orchestrator import modes as mode_registry
from extensions.orchestrator.issue_registry import (
    IssueRecord,
    IssueRegistry,
    IssueStatus,
)
from extensions.orchestrator.mode_selector import (
    KNOWN_MODES,
    MODE_LABEL_PREFIX,
    ModeSelector,
)
from extensions.orchestrator.modes.base import (
    DEFAULT_MODE,
    ModeDecision,
    ModeRunner,
)
from extensions.orchestrator.modes.single import SingleModeRunner


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _FakeIssue:
    """Minimal stand-in for ``Issue`` — ModeSelector only reads ``labels``."""

    def __init__(self, labels: list[str] | None = None) -> None:
        self.labels = labels or []


def _make_fake_agent_runner() -> MagicMock:
    runner = MagicMock(name="AgentRunner")

    # ``SingleModeRunner.run`` awaits ``self._agent_runner.run(...)``, so the
    # mock's ``run`` must be an async method that returns a sentinel we can
    # assert on.
    async def _fake_run(*args: Any, **kwargs: Any) -> str:
        return "agent-runner-return"

    runner.run = _fake_run
    return runner


# ---------------------------------------------------------------------------
# ModeDecision dataclass
# ---------------------------------------------------------------------------


class TestModeDecision(unittest.TestCase):
    def test_defaults(self) -> None:
        d = ModeDecision()
        self.assertEqual(d.mode, DEFAULT_MODE)
        self.assertEqual(d.mode, "single")
        self.assertEqual(d.reason, "")
        self.assertEqual(d.source, "fallback")
        self.assertEqual(d.agents, [])
        self.assertEqual(d.confidence, 1.0)

    def test_construct_full(self) -> None:
        d = ModeDecision(
            mode="pipeline",
            reason="issue mentions analyzer→implementer→tester",
            source="router",
            agents=["analyzer", "implementer", "tester"],
            confidence=0.82,
        )
        self.assertEqual(d.mode, "pipeline")
        self.assertEqual(d.agents, ["analyzer", "implementer", "tester"])
        self.assertEqual(d.confidence, 0.82)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


class TestModeRegistry(unittest.TestCase):
    def setUp(self) -> None:
        # Reset module-level registry between tests so order-independence
        # holds; we intentionally touch the private dict here.
        mode_registry._registry.clear()

    def test_register_and_get(self) -> None:
        runner = SingleModeRunner(_make_fake_agent_runner())
        mode_registry.register("single", runner)
        self.assertIs(mode_registry.get("single"), runner)
        self.assertEqual(mode_registry.available(), ["single"])

    def test_get_unknown_raises(self) -> None:
        with self.assertRaises(KeyError):
            mode_registry.get("nonexistent")

    def test_re_registration_overwrites(self) -> None:
        r1 = SingleModeRunner(_make_fake_agent_runner())
        r2 = SingleModeRunner(_make_fake_agent_runner())
        mode_registry.register("single", r1)
        mode_registry.register("single", r2)
        self.assertIs(mode_registry.get("single"), r2)


# ---------------------------------------------------------------------------
# SingleModeRunner
# ---------------------------------------------------------------------------


class TestSingleModeRunner(unittest.TestCase):
    def test_satisfies_protocol(self) -> None:
        runner = SingleModeRunner(_make_fake_agent_runner())
        # ``ModeRunner`` is ``@runtime_checkable`` so ``isinstance`` works.
        self.assertIsInstance(runner, ModeRunner)

    def test_run_delegates_to_agent_runner(self) -> None:
        agent = _make_fake_agent_runner()
        runner = SingleModeRunner(agent)
        result = asyncio.run(runner.run(MagicMock(), MagicMock(), foo="bar"))
        self.assertEqual(result, "agent-runner-return")


# ---------------------------------------------------------------------------
# ModeSelector
# ---------------------------------------------------------------------------


class TestModeSelector(unittest.TestCase):
    def test_default_when_no_labels(self) -> None:
        selector = ModeSelector()
        decision = selector.choose(_FakeIssue())
        self.assertEqual(decision.mode, DEFAULT_MODE)
        self.assertEqual(decision.source, "fallback")

    def test_default_when_labels_is_none(self) -> None:
        issue = MagicMock()
        issue.labels = None  # some trackers return None instead of []
        decision = ModeSelector().choose(issue)
        self.assertEqual(decision.mode, DEFAULT_MODE)

    def test_label_picks_mode(self) -> None:
        decision = ModeSelector().choose(_FakeIssue(labels=["mode:pipeline"]))
        self.assertEqual(decision.mode, "pipeline")
        self.assertEqual(decision.source, "label")
        self.assertIn("mode:pipeline", decision.reason)

    def test_label_case_insensitive(self) -> None:
        decision = ModeSelector().choose(_FakeIssue(labels=["MODE:Coordinator"]))
        self.assertEqual(decision.mode, "coordinator")
        self.assertEqual(decision.source, "label")

    def test_unknown_label_falls_through(self) -> None:
        decision = ModeSelector().choose(_FakeIssue(labels=["mode:bogus"]))
        self.assertEqual(decision.mode, DEFAULT_MODE)
        self.assertEqual(decision.source, "fallback")

    def test_auto_label_invokes_router_stub(self) -> None:
        # Phase-1 router stub returns the default with source="fallback".
        decision = ModeSelector().choose(_FakeIssue(labels=["mode:auto"]))
        self.assertEqual(decision.mode, DEFAULT_MODE)
        self.assertEqual(decision.source, "fallback")
        self.assertIn("router", decision.reason)

    def test_first_mode_label_wins(self) -> None:
        decision = ModeSelector().choose(_FakeIssue(labels=["bug", "mode:debate", "mode:pipeline"]))
        self.assertEqual(decision.mode, "debate")

    def test_non_mode_labels_ignored(self) -> None:
        decision = ModeSelector().choose(_FakeIssue(labels=["bug", "p1", "needs-design"]))
        self.assertEqual(decision.mode, DEFAULT_MODE)

    def test_invalid_default_rejected(self) -> None:
        with self.assertRaises(ValueError):
            ModeSelector(default_mode="not-a-mode")

    def test_known_modes_includes_label_constants(self) -> None:
        self.assertIn("single", KNOWN_MODES)
        self.assertIn("pipeline", KNOWN_MODES)
        self.assertIn("coordinator", KNOWN_MODES)
        self.assertIn("debate", KNOWN_MODES)
        self.assertIn("auto", KNOWN_MODES)
        self.assertEqual(MODE_LABEL_PREFIX, "mode:")


# ---------------------------------------------------------------------------
# IssueRecord new fields + back-compat
# ---------------------------------------------------------------------------


class TestIssueRecordModeFields(unittest.TestCase):
    def test_defaults(self) -> None:
        rec = IssueRecord(issue_id="i1", issue_identifier="owner/repo#1")
        self.assertEqual(rec.collaboration_mode, "single")
        self.assertIsNone(rec.mode_decision_reason)

    def test_explicit_values(self) -> None:
        rec = IssueRecord(
            issue_id="i1",
            issue_identifier="owner/repo#1",
            collaboration_mode="pipeline",
            mode_decision_reason="explicit label 'mode:pipeline'",
        )
        self.assertEqual(rec.collaboration_mode, "pipeline")
        self.assertEqual(rec.mode_decision_reason, "explicit label 'mode:pipeline'")


class TestIssueRegistryBackCompat(unittest.TestCase):
    """Pre-mode registry files must still load — known_fields filter handles it."""

    def test_load_old_record_without_mode_fields(self) -> None:
        # Simulate a registry.json written before collaboration_mode was added.
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "registry.json"
            payload = {
                "i1": {
                    "issue_id": "i1",
                    "issue_identifier": "owner/repo#1",
                    "status": "completed",
                    # NOTE: no collaboration_mode, no mode_decision_reason.
                }
            }
            path.write_text(json.dumps(payload), encoding="utf-8")

            reg = IssueRegistry(storage_path=path)
            record = reg.get("i1")
            self.assertIsNotNone(record)
            assert record is not None  # mypy/pyright
            # Defaults must kick in.
            self.assertEqual(record.collaboration_mode, "single")
            self.assertIsNone(record.mode_decision_reason)
            # And status was still loaded correctly.
            self.assertEqual(record.status, IssueStatus.COMPLETED)

    def test_round_trip_with_mode_fields(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "registry.json"
            reg = IssueRegistry(storage_path=path)
            reg.register(
                issue_id="i1",
                issue_identifier="owner/repo#1",
                branch_name="feature/x",
                base_branch="main",
                workspace_strategy="isolated",
                workspace_path="/tmp/ws",
                base_commit_sha=None,
                start_commit_sha="abc",
                previous_issue_id=None,
                sequence_index=None,
            )
            record = reg.get("i1")
            assert record is not None
            record.collaboration_mode = "coordinator"
            record.mode_decision_reason = "router: cross-module refactor"
            reg._save()

            # Reload from disk and verify the fields persisted.
            reg2 = IssueRegistry(storage_path=path)
            r2 = reg2.get("i1")
            assert r2 is not None
            self.assertEqual(r2.collaboration_mode, "coordinator")
            self.assertEqual(r2.mode_decision_reason, "router: cross-module refactor")


if __name__ == "__main__":
    unittest.main()
