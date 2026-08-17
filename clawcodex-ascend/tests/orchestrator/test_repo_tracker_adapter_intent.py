from __future__ import annotations

import unittest
from extensions.orchestrator.repo_tracker.adapter import RepositoryTrackerAdapter
from extensions.orchestrator.tracker import Intent


class TestRepositoryTrackerAdapterIntent(unittest.IsolatedAsyncioTestCase):
    def _make(self, intent_labels: dict[str, str] | None = None) -> RepositoryTrackerAdapter:
        return RepositoryTrackerAdapter(
            platform="github",
            owner="o",
            repo="r",
            api_key="dummy",
            intent_labels=intent_labels,
        )

    async def test_default_labels(self) -> None:
        adapter = self._make()
        self.assertIs(
            await adapter.extract_intent_from_labels(["agent:retry"]),
            Intent.RETRY,
        )
        self.assertIs(
            await adapter.extract_intent_from_labels(["agent:follow-up"]),
            Intent.FOLLOWUP,
        )
        self.assertIs(
            await adapter.extract_intent_from_labels(["agent:blocked"]),
            Intent.BLOCKED,
        )

    async def test_custom_labels(self) -> None:
        adapter = self._make(
            intent_labels={
                "retry": "rerun",
                "followup": "followup",
                "blocked": "blocked",
            }
        )
        self.assertIs(
            await adapter.extract_intent_from_labels(["rerun"]),
            Intent.RETRY,
        )
        self.assertIs(
            await adapter.extract_intent_from_labels(["blocked"]),
            Intent.BLOCKED,
        )

    async def test_empty_labels(self) -> None:
        adapter = self._make()
        self.assertIs(
            await adapter.extract_intent_from_labels([]),
            Intent.NONE,
        )

    async def test_intent_labels_isolated_per_instance(self) -> None:
        a = self._make(
            intent_labels={
                "retry": "a:retry",
                "followup": "a:followup",
                "blocked": "a:blocked",
            }
        )
        b = self._make()
        # Mutating one must not affect the other.
        a.intent_labels["retry"] = "mutated"
        self.assertEqual(b.intent_labels["retry"], "agent:retry")
