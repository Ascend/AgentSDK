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

"""Tests for clawcodex_ext.services.lodestone.parser.

Covers all five anchor kinds + ensures span tracking matches substrings.
"""
# pylint: disable=no-name-in-module

from __future__ import annotations


from clawcodex_ext.services.lodestone.models import LodestoneAnchor
from clawcodex_ext.services.lodestone.parser import (
    AnchorParser,
    detect_anchor_kind,
    parse_anchors,
)


def test_file_path_with_line_and_column():
    anchors = parse_anchors("trace src/foo.py:42:13")
    assert len(anchors) == 1
    a = anchors[0]
    assert a.kind == "file_path"
    assert a.file_path == "src/foo.py"
    assert a.line == 42
    assert a.column == 13
    assert a.span == (6, 22)


def test_file_path_with_range():
    a = parse_anchors("src/foo.py:42-50:5")[0]
    assert a.file_path == "src/foo.py"
    assert a.line == 42
    assert a.end_line == 50
    assert a.end_column == 5


def test_file_path_extension_required():
    # Bare ``src/foo:42`` does NOT look like a file_path because ``foo``
    # has no extension.  Should yield no anchors.
    assert parse_anchors("src/foo:42") == []


def test_function_ref_basic():
    anchors = parse_anchors("calling clawcodex_ext.utils.git.run() and clawcodex_ext::cli::main()")
    syms = [a.symbol for a in anchors if a.kind == "function_ref"]
    assert "clawcodex_ext.utils.git.run" in syms
    assert "clawcodex_ext::cli::main" in syms


def test_tracker_issue_variants():
    text = "look at #123 and ORG-456 and [BUG-789]"
    anchors = parse_anchors(text)
    keys = {a.tracker_key for a in anchors if a.kind == "tracker_issue"}
    assert ("gitcode", "123") in keys
    assert ("org", "456") in keys
    assert ("bug", "789") in keys


def test_git_blob():
    anchors = parse_anchors("see @abc1234def:src/foo.py:42")
    blobs = [a for a in anchors if a.kind == "git_blob"]
    assert blobs and blobs[0].git_sha == "abc1234def" and blobs[0].file_path == "src/foo.py"


def test_git_commit_not_preceded_by_at():
    anchors = parse_anchors("commit abc1234def is good")
    commits = [a for a in anchors if a.kind == "git_commit"]
    assert any(a.git_sha == "abc1234def" for a in commits)
    # ``abc1234def`` as bare is NOT classified as a git_blob.
    assert not [a for a in anchors if a.kind == "git_blob"]


def test_url_strips_trailing_punctuation():
    anchors = parse_anchors("see https://example.com/foo),")
    urls = [a for a in anchors if a.kind == "url"]
    assert urls and urls[0].url == "https://example.com/foo"


def test_various_url_schemes():
    raw = (
        "vscode://file/x.py:1 cursor://file/y idea://open?file=z subl://x file:///etc/hosts git@github.com:foo/bar.git"
    )
    anchors = parse_anchors(raw)
    urls = [a.url for a in anchors if a.kind == "url"]
    # ``git@github.com:...`` is not a URL the parser handles — it stays
    # out of anchors; the rest are explicitly recognised.
    assert any(u.startswith("vscode://") for u in urls)
    assert any(u.startswith("cursor://") for u in urls)
    assert any(u.startswith("idea://") for u in urls)
    assert any(u.startswith("subl://") for u in urls)
    assert any(u.startswith("file://") for u in urls)


def test_overlapping_spans_keep_longest():
    """``src/foo.py:42`` is a substring of ``src/foo.py:42:13`` — the
    longer match should win.
    """
    anchors = parse_anchors("src/foo.py:42:13")
    assert len(anchors) == 1
    assert anchors[0].raw == "src/foo.py:42:13"


def test_parse_first_helper():
    assert detect_anchor_kind("hello #123 world") == "tracker_issue"
    assert detect_anchor_kind("plain text") is None
    assert detect_anchor_kind("") is None


def test_custom_file_globs_extra_extension():
    parser = AnchorParser(file_globs=(".proto",))
    a = parser.parse("check api.proto:7")[0]
    assert a.kind == "file_path"
    assert a.file_path == "api.proto"


def test_anchor_link_text_for_files():
    a = LodestoneAnchor(kind="file_path", raw="", file_path="src/foo.py", line=42, column=13)
    assert a.link_text() == "src/foo.py:42:13"


def test_anchor_link_text_with_range():
    a = LodestoneAnchor(
        kind="file_path",
        raw="",
        file_path="src/foo.py",
        line=1,
        end_line=5,
        end_column=9,
    )
    assert a.link_text() == "src/foo.py:1-5:9"


def test_anchor_link_text_for_tracker_gitcode():
    a = LodestoneAnchor(
        kind="tracker_issue",
        raw="",
        tracker_key=("gitcode", "42"),
    )
    assert a.link_text() == "#42"


def test_anchor_link_text_for_tracker_linear():
    a = LodestoneAnchor(
        kind="tracker_issue",
        raw="",
        tracker_key=("linear", "LIN-42"),
    )
    assert a.link_text() == "linear-LIN-42"


def test_anchor_link_text_for_commit_sha():
    a = LodestoneAnchor(kind="git_commit", raw="abcdef0123456789", git_sha="abcdef0123456789")
    assert a.link_text() == "abcdef0"
