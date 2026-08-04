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

"""Smoke test for the deprecated `src.context_system.memory_prefetch` shim.

The full recall pipeline lives in ``src.memdir`` now; coverage is in
``tests/test_memdir_scan_recall.py``. This file exists only to verify the
shim re-exports the public surface so existing imports keep working for
one release.
"""

from __future__ import annotations

import asyncio
import unittest


class CompatShimTest(unittest.TestCase):
    def test_public_surface_reexported(self):
        from src.context_system.memory_prefetch import (
            MAX_RELEVANT_MEMORIES,
            MemoryHeader,
            RelevantMemory,
            find_relevant_memories,
            format_memory_manifest,
            scan_memory_files,
        )

        self.assertEqual(MAX_RELEVANT_MEMORIES, 5)
        self.assertTrue(callable(find_relevant_memories))
        self.assertTrue(callable(format_memory_manifest))
        self.assertTrue(callable(scan_memory_files))
        self.assertTrue(hasattr(MemoryHeader, "__init__"))
        self.assertTrue(hasattr(RelevantMemory, "__init__"))

    def test_shim_returns_empty_without_provider(self):
        # The keyword fallback was removed (chapter rejects it). With no
        # provider, the shim should return an empty list rather than
        # retrieving via keyword match.
        from src.context_system.memory_prefetch import find_relevant_memories

        result = asyncio.new_event_loop().run_until_complete(
            find_relevant_memories(
                "anything",
                "/nonexistent/dir",
                provider=None,
            )
        )
        self.assertEqual(result, [])


if __name__ == "__main__":
    unittest.main()
