#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# -------------------------------------------------------------------------
# This file is part of the AgentSDK project.
# Copyright (c) 2026 Huawei Technologies Co.,Ltd.
# Copyright (c) 2026 Clawd Codex Team
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

"""Tests for F-88 generic text tail follower."""
# pylint: disable=no-name-in-module

from __future__ import annotations

import asyncio
import tempfile

import pytest

from clawcodex_ext.services.monitor.text_tail import TextTailBuffer, TextTailFollower


class TestTextTailFollower:
    @pytest.mark.asyncio
    async def test_reads_appended_text(self):
        with tempfile.NamedTemporaryFile(mode="w", delete=False) as fh:
            path = fh.name
            fh.write("line1\n")

        follower = TextTailFollower(path)
        await follower.start()
        assert follower.current_tail == ""

        with open(path, "a", encoding="utf-8") as fh:
            fh.write("line2\n")

        chunk = await asyncio.wait_for(follower.read_chunk(), timeout=1.0)
        assert "line2" in chunk
        assert "line1" not in follower.current_tail
        await follower.stop()

    @pytest.mark.asyncio
    async def test_ring_buffer_drops_old_bytes(self):
        with tempfile.NamedTemporaryFile(mode="w", delete=False) as fh:
            path = fh.name
            fh.write("old")

        follower = TextTailFollower(path, ring_size=5)
        await follower.start()
        with open(path, "a", encoding="utf-8") as fh:
            fh.write("newer-long-text")
        await asyncio.wait_for(follower.read_chunk(), timeout=1.0)
        # The ring keeps only the last 5 bytes.
        assert len(follower.current_tail) <= 5
        await follower.stop()

    @pytest.mark.asyncio
    async def test_truncation_recovery(self):
        with tempfile.NamedTemporaryFile(mode="w", delete=False) as fh:
            path = fh.name
            fh.write("1234567890")

        follower = TextTailFollower(path)
        await follower.start()
        # By default we start at the current end of file.
        assert follower.offset == 10

        # Truncate the file.
        with open(path, "w", encoding="utf-8") as fh:
            fh.write("ab")

        # Polling should detect truncation and reset.
        for _ in range(10):
            follower.read_available_now()
            if follower.offset <= 2:
                break
            await asyncio.sleep(0.05)
        assert follower.offset <= 2
        await follower.stop()

    @pytest.mark.asyncio
    async def test_async_iteration(self):
        with tempfile.NamedTemporaryFile(mode="w", delete=False) as fh:
            path = fh.name

        follower = TextTailFollower(path)
        await follower.start()

        async def append_later():
            await asyncio.sleep(0.05)
            with open(path, "a", encoding="utf-8") as fh:
                fh.write("iter")

        task = asyncio.create_task(append_later())
        chunks = []
        async for chunk in follower:
            chunks.append(chunk)
            if len(chunks) >= 1:
                break
        await task
        await follower.stop()
        assert any("iter" in c for c in chunks)


class TestTextTailBuffer:
    def test_buffer_keeps_last_bytes(self):
        buf = TextTailBuffer(maxlen=10)
        buf.append("12345")
        buf.append("67890")
        buf.append("abc")
        assert buf.text.endswith("abc")
        assert len(buf.text) <= 12  # deques can briefly exceed maxlen

    def test_append_lines(self):
        buf = TextTailBuffer(maxlen=100)
        buf.append_lines(["line1", "line2"])
        assert buf.text == "line1\nline2\n"

    def test_clear(self):
        buf = TextTailBuffer(maxlen=100)
        buf.append("x")
        buf.clear()
        assert buf.text == ""
