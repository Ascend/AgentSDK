from __future__ import annotations

import unittest
import tempfile

from extensions.orchestrator.local_tracker.adapter import LocalTrackerAdapter
from extensions.orchestrator.tracker import Intent


class TestLocalTrackerAdapterIntent(unittest.IsolatedAsyncioTestCase):
    async def test_default_labels(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            adapter = LocalTrackerAdapter(issues_path=tmp)
            self.assertIs(
                await adapter.extract_intent_from_labels(["agent:retry"]),
                Intent.RETRY,
            )
            self.assertIs(
                await adapter.extract_intent_from_labels(["agent:blocked", "agent:follow-up"]),
                Intent.BLOCKED,
            )
            self.assertIs(
                await adapter.extract_intent_from_labels(["agent:retry", "agent:follow-up"]),
                Intent.FOLLOWUP,
            )

    async def test_custom_labels(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            adapter = LocalTrackerAdapter(
                issues_path=tmp,
                intent_labels={
                    "retry": "rerun",
                    "followup": "followup",
                    "blocked": "blocked",
                },
            )
            self.assertIs(
                await adapter.extract_intent_from_labels(["rerun"]),
                Intent.RETRY,
            )
            self.assertIs(
                await adapter.extract_intent_from_labels(["blocked"]),
                Intent.BLOCKED,
            )


# ---------------------------------------------------------------------------
# IssueRecord new fields
# ---------------------------------------------------------------------------
