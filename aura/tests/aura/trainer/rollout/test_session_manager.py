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

"""Unit tests for the rollout session manager and status endpoint."""

import json
import sys
import unittest
import urllib.request
from unittest import mock

_torch_stub = mock.MagicMock()
sys.modules.setdefault("torch", _torch_stub)
sys.modules.setdefault("torch.distributed", _torch_stub.distributed)

from aura.trainer.rollout.session_manager import NullSessionManager, SessionManager
from aura.trainer.rollout.session_status_server import STATUS_PATH, SessionStatusServer


class TestSessionManager(unittest.TestCase):
    def test_create_finalize_keeps_active_zero(self):
        manager = SessionManager(enabled=False)
        manager.create_session("a", prompt_id=0)
        self.assertEqual(manager.get_status()["active"], 1)
        manager.finalize_session("a")
        status = manager.get_status()
        self.assertEqual(status["active"], 0)
        self.assertEqual(status["finished"], 1)

    def test_abort_counts_and_unknown_session_is_safe(self):
        manager = SessionManager(enabled=False)
        manager.create_session("a", prompt_id=0)
        manager.abort_session("a", reason="timeout")
        manager.abort_session("missing", reason="timeout")
        status = manager.get_status()
        self.assertEqual(status["aborted"], 1)
        self.assertEqual(status["active"], 0)

    def test_shutdown_aborts_remaining_sessions(self):
        manager = SessionManager(enabled=False)
        manager.create_session("a", prompt_id=0)
        manager.create_session("b", prompt_id=1)
        aborted = manager.shutdown(timeout=0)
        self.assertEqual(aborted, 2)
        self.assertEqual(manager.get_status()["active"], 0)

    def test_placeholder_is_counted(self):
        manager = SessionManager(enabled=False)
        manager.record_placeholder(0, index=1, session_id="dummy-0")
        self.assertEqual(manager.get_status()["placeholder"], 1)

    def test_late_terminal_calls_do_not_double_count(self):
        manager = SessionManager(enabled=False)
        manager.create_session("a", prompt_id=0)
        manager.finalize_session("a")
        manager.finalize_session("a")
        manager.abort_session("a", reason="shutdown")
        status = manager.get_status()
        self.assertEqual(status["active"], 0)
        self.assertEqual(status["finished"], 1)
        self.assertEqual(status["aborted"], 0)


class TestNullSessionManager(unittest.TestCase):
    def test_disabled_manager_is_pass_through(self):
        manager = NullSessionManager()
        self.assertFalse(manager.enabled)
        self.assertEqual(manager.create_session("a"), "a")
        self.assertIsNone(manager.finalize_session("a"))
        self.assertIsNone(manager.abort_session("a", reason="timeout"))
        self.assertEqual(manager.shutdown(timeout=1), 0)
        self.assertEqual(manager.get_status()["active"], 0)


class TestSessionStatusServer(unittest.TestCase):
    def test_serves_status_payload(self):
        payload = {"mode": "hybrid", "sessions": {"active": 0, "finished": 2, "aborted": 0}}
        server = SessionStatusServer(port=0, status_cb=lambda: payload, host="127.0.0.1")
        # port=0 binds an ephemeral port, the actual one is exposed by addr.
        self.assertIsNotNone(server.start())
        self.addCleanup(server.stop)
        actual_port = int(server.addr.rsplit(":", 1)[1])
        with urllib.request.urlopen(f"http://127.0.0.1:{actual_port}{STATUS_PATH}", timeout=5) as response:
            self.assertEqual(response.status, 200)
            self.assertEqual(json.loads(response.read()), payload)


if __name__ == "__main__":
    unittest.main()
