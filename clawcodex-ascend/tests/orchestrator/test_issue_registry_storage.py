"""Unit tests for the IssueRegistry storage persistence (PR 742 fix).

Covers the review finding that a failed disk write was silently swallowed:
``_save`` now re-raises so durable state mutations surface the failure to
callers, while the throttled diagnostics path keeps its best-effort
behavior.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from extensions.orchestrator.issue_registry.models import IssueRecord
from extensions.orchestrator.issue_registry.state_machine import StateMachineMixin
from extensions.orchestrator.issue_registry.storage import StorageMixin


class _FakeRegistry(StorageMixin, StateMachineMixin):
    """Minimal host composing the storage + state-machine mixins."""

    def __init__(self, path: Path) -> None:
        self._path = path
        self._records = {}
        self._diagnostics_min_save_interval_s = 0.0
        self._last_diagnostics_save_monotonic = 0.0
        self._pending_diagnostics_save = False


class TestStoragePersistenceFailure(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / "registry.json"

    def _registry_with_record(self) -> _FakeRegistry:
        registry = _FakeRegistry(self.path)
        registry._records = {"i1": IssueRecord(issue_id="i1", issue_identifier="x#1")}
        return registry

    def test_save_raises_on_write_failure_and_cleans_temp(self) -> None:
        registry = self._registry_with_record()
        with patch("pathlib.Path.replace", side_effect=OSError("disk full")):
            with self.assertLogs("extensions.orchestrator.issue_registry.storage", level="WARNING"):
                with self.assertRaises(OSError):
                    registry._save()
        self.assertEqual(list(Path(self.tmp.name).glob("*.tmp")), [])

    def test_save_diagnostics_swallows_write_failure(self) -> None:
        registry = self._registry_with_record()
        with patch("pathlib.Path.replace", side_effect=OSError("disk full")):
            with self.assertLogs("extensions.orchestrator.issue_registry.storage", level="WARNING"):
                registry._save_diagnostics()
        # The throttled diagnostics path must not crash the run; the
        # pending flag stays set so a later durable flush retries.
        self.assertTrue(registry._pending_diagnostics_save)

    def test_durable_transition_propagates_write_failure(self) -> None:
        registry = _FakeRegistry(self.path)
        with patch("pathlib.Path.replace", side_effect=OSError("disk full")):
            with self.assertLogs("extensions.orchestrator.issue_registry.storage", level="WARNING"):
                with self.assertRaises(OSError):
                    registry.register(
                        issue_id="i1",
                        issue_identifier="x#1",
                        branch_name="feat/x",
                    )

    def test_save_success_resets_throttle_bookkeeping(self) -> None:
        registry = self._registry_with_record()
        registry._pending_diagnostics_save = True
        registry._save()
        self.assertFalse(registry._pending_diagnostics_save)
        self.assertGreater(registry._last_diagnostics_save_monotonic, 0)
        self.assertTrue(self.path.exists())
