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

from __future__ import annotations

import unittest
from extensions.orchestrator.tracker import MergeableStatus


class TestMergeableStatusDataclass(unittest.TestCase):
    def test_default_values(self) -> None:
        s = MergeableStatus()
        self.assertIsNone(s.mergeable)
        self.assertIsNone(s.mergeable_state)
        self.assertIsNone(s.behind_by)
        self.assertIsNone(s.ahead_by)
        self.assertFalse(s.has_conflicts)
        self.assertEqual(s.raw, {})

    def test_has_conflicts_false_when_mergeable_true(self) -> None:
        # Constructor: has_conflicts is set explicitly by the caller;
        # the dataclass does not derive it.
        s = MergeableStatus(mergeable=True, mergeable_state="clean")
        self.assertFalse(s.has_conflicts)

    def test_has_conflicts_explicitly_true(self) -> None:
        # When the normalizer sees mergeable=False or state="dirty"
        # it sets has_conflicts=True explicitly.
        s = MergeableStatus(
            mergeable=False,
            mergeable_state="dirty",
            has_conflicts=True,
        )
        self.assertTrue(s.has_conflicts)

    def test_has_conflicts_false_when_state_behind(self) -> None:
        # "behind" just means ahead/behind, not a conflict.
        s = MergeableStatus(mergeable=None, mergeable_state="behind", behind_by=2)
        self.assertFalse(s.has_conflicts)

    def test_hashable_when_raw_is_empty(self) -> None:
        # Regression: ``raw`` is a dict, which is unhashable.  With the
        # default dataclass ``frozen=True`` semantics this used to raise
        # ``TypeError: unhashable type: 'dict'`` whenever the instance
        # was used as a ``set`` element or ``dict`` key.
        s = MergeableStatus()
        # Must not raise.
        hash(s)
        hash(MergeableStatus(raw={"platform": "github", "x": 1}))

    def test_hashable_in_set_and_as_dict_key(self) -> None:
        s = MergeableStatus(mergeable=True, mergeable_state="clean")
        # Set membership: no TypeError.
        self.assertIn(s, {s})
        # Dict key: no TypeError.
        lookup = {s: "ok"}
        self.assertEqual(lookup[s], "ok")

    def test_equality_ignores_raw(self) -> None:
        # ``raw`` is debug payload that must not influence equality.
        a = MergeableStatus(
            mergeable=False,
            mergeable_state="dirty",
            has_conflicts=True,
            raw={"platform": "github", "trace": "abc"},
        )
        b = MergeableStatus(
            mergeable=False,
            mergeable_state="dirty",
            has_conflicts=True,
            raw={"platform": "gitcode", "trace": "xyz"},
        )
        self.assertEqual(a, b)
        # Hash must also agree so both can coexist in a set.
        self.assertEqual(hash(a), hash(b))
        self.assertEqual(len({a, b}), 1)
