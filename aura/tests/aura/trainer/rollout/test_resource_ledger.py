#!/usr/bin/env python3
# coding=utf-8
# -------------------------------------------------------------------------
# This file is part of the AgentSDK project.
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

"""Unit tests for the rollout resource ledger (phase 4)."""

import sys
import time
import unittest
from unittest import mock

_torch_stub = mock.MagicMock()
_added_torch_modules = []
for _name, _module in (
    ("torch", _torch_stub),
    ("torch.distributed", _torch_stub.distributed),
):
    if _name not in sys.modules:
        sys.modules[_name] = _module
        _added_torch_modules.append(_name)

from aura.trainer.rollout.resource_ledger import (
    DEFAULT_RELEASE_TIMEOUT,
    STATUS_FAILED,
    STATUS_RELEASED,
    STATUS_SKIPPED,
    ResourceEntry,
    RolloutResourceLedger,
)


def teardown_module():
    """Drop the torch stub injected by this module so it cannot leak to others."""
    for name in _added_torch_modules:
        sys.modules.pop(name, None)


class TestRolloutResourceLedger(unittest.TestCase):
    """Release order, isolation, timeout and idempotency of the ledger."""

    def test_release_order_follows_registration_order(self):
        """Registration order is the release order."""
        ledger = RolloutResourceLedger()
        order = []
        ledger.acquire(ResourceEntry("actor", "RolloutWorker", lambda: order.append("worker")))
        ledger.acquire(ResourceEntry("http", "rollout_server", lambda: order.append("server")))
        ledger.acquire(ResourceEntry("http", "session_status", lambda: order.append("status")))

        summary = ledger.release_all()

        self.assertEqual(order, ["worker", "server", "status"])
        self.assertEqual(summary["owned"], 3)
        self.assertEqual(summary["released"], 3)
        self.assertEqual(summary["failed"], 0)
        self.assertEqual(
            [item for item in summary["items"]],
            [("actor", "RolloutWorker", STATUS_RELEASED),
             ("http", "rollout_server", STATUS_RELEASED),
             ("http", "session_status", STATUS_RELEASED)],
        )

    def test_failed_item_does_not_block_others(self):
        """One raising disposer is reported failed, the rest still release."""
        ledger = RolloutResourceLedger()
        order = []

        def _boom():
            raise RuntimeError("boom")

        ledger.acquire(ResourceEntry("http", "session_status", lambda: order.append("status")))
        ledger.acquire(ResourceEntry("actor", "RolloutWorker", _boom))

        summary = ledger.release_all()

        self.assertEqual(order, ["status"])
        self.assertEqual(summary["owned"], 2)
        self.assertEqual(summary["released"], 1)
        self.assertEqual(summary["failed"], 1)
        self.assertIn(("actor", "RolloutWorker", STATUS_FAILED), summary["items"])

    def test_item_timeout_is_reported_failed(self):
        """A hanging disposer is cut off after its per-item timeout."""
        ledger = RolloutResourceLedger()
        ledger.acquire(ResourceEntry("actor", "slow", lambda: time.sleep(1.0), timeout=0.05))

        started = time.time()
        summary = ledger.release_all()

        self.assertLess(time.time() - started, 0.8)
        self.assertEqual(summary["failed"], 1)
        self.assertIn(("actor", "slow", STATUS_FAILED), summary["items"])

    def test_entry_without_disposer_is_skipped(self):
        """Delegated/external entries are recorded, never released."""
        ledger = RolloutResourceLedger()
        ledger.acquire(ResourceEntry("delegated", "actor"))

        summary = ledger.release_all()

        self.assertEqual(summary["owned"], 0)
        self.assertEqual(summary["released"], 0)
        self.assertEqual(summary["failed"], 0)
        self.assertEqual(summary["items"], [("delegated", "actor", STATUS_SKIPPED)])

    def test_release_all_is_idempotent(self):
        """A second release reports the already released items as skipped."""
        ledger = RolloutResourceLedger()
        calls = []
        ledger.acquire(ResourceEntry("actor", "RolloutWorker", lambda: calls.append(1)))

        first = ledger.release_all()
        second = ledger.release_all()

        self.assertEqual(len(calls), 1)
        self.assertEqual(first["released"], 1)
        self.assertEqual(second["owned"], 0)
        self.assertEqual(second["items"], [("actor", "RolloutWorker", STATUS_SKIPPED)])

    def test_default_release_timeout_is_a_constant(self):
        """The per-item budget is an implementation constant, not a config knob."""
        self.assertEqual(DEFAULT_RELEASE_TIMEOUT, 5.0)
        self.assertEqual(ResourceEntry("actor", "x").timeout, DEFAULT_RELEASE_TIMEOUT)


if __name__ == "__main__":
    unittest.main()
