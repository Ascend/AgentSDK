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

"""Tests for outbound text helpers."""

from __future__ import annotations

import pytest

from clawcodex_ext.services.im_gateway.text import (
    maybe_truncate_with_liveview,
    split_text,
    strip_markdown,
)


def test_strip_markdown_removes_code_fence_and_bold() -> None:
    src = "## Title\n\n**bold** and _italic_ and `code`.\n\n```python\nprint(1)\n```"
    out = strip_markdown(src)
    assert "##" not in out
    assert "**" not in out
    assert "_" not in out
    assert "`" not in out
    assert "print(1)" in out
    assert "Title" in out


def test_strip_markdown_links_to_text() -> None:
    src = "see [docs](https://example.com/x) and ![alt](https://example.com/i.png)"
    out = strip_markdown(src)
    assert "docs" in out
    assert "alt" in out
    assert "https://example.com" not in out


def test_strip_markdown_list_markers() -> None:
    src = "- one\n- two\n1. first\n2. second\n> quote"
    out = strip_markdown(src)
    assert out.startswith("one")
    assert "first" in out
    assert "quote" in out
    assert "- " not in out


def test_split_text_short() -> None:
    assert split_text("hello", 4000) == ["hello"]


def test_split_text_long_on_boundary() -> None:
    text = "\n".join(f"line {i}" for i in range(1000))  # ~7000 chars
    chunks = split_text(text, 1000)
    assert len(chunks) > 1
    assert all(len(c) <= 1000 for c in chunks)
    # reconstruct loses only whitespace
    rejoined = " ".join(c.replace("\n", " ") for c in chunks)
    for i in range(1000):
        assert f"line {i}" in rejoined


def test_maybe_truncate_with_liveview_keeps_short() -> None:
    text = "x" * 100
    out = maybe_truncate_with_liveview(text, chunk_size=4000, max_chunks=4)
    assert out == [text]


def test_maybe_truncate_with_liveview_truncates_with_link() -> None:
    text = "\n".join(["word" * 200] * 40)  # very long
    out = maybe_truncate_with_liveview(text, chunk_size=4000, max_chunks=2, liveview_url="https://lv/x")
    assert len(out) == 2
    assert "Content truncated" in out[-1]
    assert "https://lv/x" in out[-1]
    assert all(len(chunk) <= 4000 for chunk in out)


@pytest.mark.parametrize("chunk_size", [0, -1])
def test_split_text_rejects_non_positive_chunk_size(chunk_size: int) -> None:
    with pytest.raises(ValueError, match="greater than zero"):
        split_text("text", chunk_size=chunk_size)
