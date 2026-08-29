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

"""
Tests for post-compact cleanup.
"""

from __future__ import annotations

import unittest

from clawcodex_ext.services.compact.post_compact_cleanup import (
    PostCompactContext,
    run_post_compact_cleanup,
)


class TestRunPostCompactCleanup(unittest.TestCase):
    """Tests for run_post_compact_cleanup()."""

    def test_none_context(self):
        """No-op when context is None."""
        result = run_post_compact_cleanup(None)
        self.assertEqual(result, [])

    def test_clears_registered_caches(self):
        """Clears all registered caches."""
        cleared_caches = []
        ctx = PostCompactContext(
            caches={
                "cache_a": lambda: cleared_caches.append("a"),
                "cache_b": lambda: cleared_caches.append("b"),
            }
        )
        result = run_post_compact_cleanup(ctx)
        self.assertIn("cache_a", result)
        self.assertIn("cache_b", result)
        self.assertIn("a", cleared_caches)
        self.assertIn("b", cleared_caches)

    def test_clears_read_file_state(self):
        """Clears read_file_state dict."""
        state = {"file1.py": "content"}
        ctx = PostCompactContext(read_file_state=state)
        result = run_post_compact_cleanup(ctx)
        self.assertIn("read_file_state", result)
        self.assertEqual(state, {})

    def test_clears_nested_memory_paths(self):
        """Clears loaded_nested_memory_paths set."""
        paths = {"/path/a", "/path/b"}
        ctx = PostCompactContext(loaded_nested_memory_paths=paths)
        result = run_post_compact_cleanup(ctx)
        self.assertIn("loaded_nested_memory_paths", result)
        self.assertEqual(paths, set())

    def test_handles_failing_cache(self):
        """Failing cache doesn't break cleanup of others."""

        def bad_clear():
            raise RuntimeError("boom")

        ctx = PostCompactContext(
            caches={
                "bad_cache": bad_clear,
                "good_cache": lambda: None,
            }
        )
        result = run_post_compact_cleanup(ctx)
        # good_cache should still be cleared
        self.assertIn("good_cache", result)

    def test_empty_context(self):
        """Empty context returns empty list."""
        ctx = PostCompactContext()
        result = run_post_compact_cleanup(ctx)
        self.assertEqual(result, [])


if __name__ == "__main__":
    unittest.main()
