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

"""parentUuid chain write-side tests.

Extracted from the upstream ``tests/test_session_chain.py`` — covers only
the ``_inject_parent_uuids`` write-side logic that now lives in
``clawcodex_ext/agent/session_persist.py``.
"""

from __future__ import annotations

import unittest

from clawcodex_ext.agent.session_persist import _inject_parent_uuids


def _msg(uuid: str, parent_uuid, role: str, content: str) -> dict:
    """Build a chat message dict with the ``parentUuid`` field."""
    return {
        "uuid": uuid,
        "parentUuid": parent_uuid,
        "role": role,
        "content": content,
    }


class TestInjectParentUuids(unittest.TestCase):
    """P103-E — write-side parentUuid stamping."""

    def test_root_message_has_null_parent(self):
        out = _inject_parent_uuids([_msg("u1", None, "user", "hi")])
        self.assertIsNone(out[0]["parentUuid"])

    def test_chain_topology(self):
        msgs = [
            _msg("u1", None, "user", "hi"),
            _msg("u2", "u1", "assistant", "hello"),
            _msg("u3", "u2", "user", "again"),
            _msg("u4", "u3", "assistant", "ok"),
        ]
        out = _inject_parent_uuids(msgs)
        self.assertIsNone(out[0]["parentUuid"])
        self.assertEqual(out[1]["parentUuid"], "u1")
        self.assertEqual(out[2]["parentUuid"], "u2")
        self.assertEqual(out[3]["parentUuid"], "u3")

    def test_rewind_creates_fork_topology(self):
        """After /rewind the new messages form a branch pointing at the rewind target."""
        post_rewind = [
            _msg("u1", None, "user", "hi"),
            _msg("u2", "u1", "assistant", "hello"),
            {"uuid": "u5", "role": "user", "content": "after rewind"},
        ]
        out = _inject_parent_uuids(post_rewind)
        self.assertEqual([m["uuid"] for m in out], ["u1", "u2", "u5"])
        self.assertEqual(out[2]["parentUuid"], "u2")

    def test_always_recomputes_existing_parentUuid(self):
        """Design mandates recomputation on every write."""
        msgs = [
            {"uuid": "u1", "role": "user", "content": "x"},
        ]
        out = _inject_parent_uuids(msgs)
        self.assertIsNone(out[0]["parentUuid"])

    def test_missing_uuid_does_not_break_chain(self):
        """Defensive: messages without uuid get parentUuid=None but don't advance cursor."""
        msgs = [
            {"role": "system", "content": "hi"},
            _msg("u1", None, "user", "hi"),
        ]
        out = _inject_parent_uuids(msgs)
        self.assertIsNone(out[0]["parentUuid"])
        self.assertIsNone(out[1]["parentUuid"])

    def test_does_not_mutate_caller_list(self):
        """Pure function: input list and its dicts are not mutated."""
        original_entry = {"uuid": "u1", "role": "user", "content": "hi"}
        original = [original_entry]
        _inject_parent_uuids(original)
        self.assertNotIn("parentUuid", original[0])
        self.assertIs(original[0], original_entry)


if __name__ == "__main__":
    unittest.main()
