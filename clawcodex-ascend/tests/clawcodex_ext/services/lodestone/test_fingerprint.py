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

"""Tests for clawcodex_ext.services.lodestone.fingerprint."""
# pylint: disable=no-name-in-module

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from clawcodex_ext.services.lodestone.config import default_config
from clawcodex_ext.services.lodestone.fingerprint import (
    build_anchor_context,
    detect_workspace_fingerprint,
    invalidate_cache,
    is_known_tracking_host,
    parse_remote_url,
)


# ---------------------------------------------------------------------------
# Remote URL parsing
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "url,expected",
    [
        (
            "https://github.com/anthropics/claude-code.git",
            ("github.com", "anthropics", "claude-code"),
        ),
        ("https://gitcode.com/chadwweng/clawcodex.git", ("gitcode.com", "chadwweng", "clawcodex")),
        ("https://gitee.com/foo/bar.git", ("gitee.com", "foo", "bar")),
        ("git@github.com:anthropics/claude-code.git", ("github.com", "anthropics", "claude-code")),
        (
            "ssh://git@github.com/anthropics/claude-code",
            ("github.com", "anthropics", "claude-code"),
        ),
    ],
)
def test_parse_remote_url_common_flavours(url, expected):
    assert parse_remote_url(url) == expected


def test_parse_remote_url_returns_none_for_garbage():
    assert parse_remote_url("not a url") is None
    assert parse_remote_url("") is None


def test_is_known_tracking_host():
    assert is_known_tracking_host("gitcode.com")
    assert is_known_tracking_host("github.com")
    assert is_known_tracking_host("github.com".upper())
    assert not is_known_tracking_host("evilcorp.com")


# ---------------------------------------------------------------------------
# Fingerprint detection
# ---------------------------------------------------------------------------


def _init_git_repo(root: Path, *, remote: str | None, default_branch: str = "main") -> None:
    subprocess.run(
        ["git", "init", "--initial-branch=" + default_branch],
        cwd=str(root),
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "config", "user.email", "t@example.com"],
        cwd=str(root),
        check=True,
        capture_output=True,
    )
    subprocess.run(["git", "config", "user.name", "Test"], cwd=str(root), check=True, capture_output=True)
    (root / "README.md").write_text("hello", encoding="utf-8")
    subprocess.run(["git", "add", "README.md"], cwd=str(root), check=True, capture_output=True)
    subprocess.run(["git", "commit", "-m", "init"], cwd=str(root), check=True, capture_output=True)
    if remote is not None:
        subprocess.run(
            ["git", "remote", "add", "origin", remote],
            cwd=str(root),
            check=True,
            capture_output=True,
        )


def test_detect_fingerprint_no_git(tmp_path: Path):
    fp = detect_workspace_fingerprint(tmp_path, use_cache=False)
    assert fp.has_git is False
    assert fp.primary_remote_url is None
    assert fp.primary_remote_host is None


def test_detect_fingerprint_with_origin(tmp_path: Path):
    _init_git_repo(tmp_path, remote="https://gitcode.com/foo/bar.git")
    fp = detect_workspace_fingerprint(tmp_path, use_cache=False)
    assert fp.has_git is True
    assert fp.primary_remote_url == "https://gitcode.com/foo/bar.git"
    assert fp.primary_remote_host == "gitcode.com"
    assert fp.default_branch == "main"


def test_detect_fingerprint_caches_to_disk(tmp_path: Path):
    _init_git_repo(tmp_path, remote="https://gitcode.com/foo/bar.git")
    fp1 = detect_workspace_fingerprint(tmp_path)
    # Second call should read cache.
    fp2 = detect_workspace_fingerprint(tmp_path)
    assert fp1.workspace_root == fp2.workspace_root


def test_invalidate_cache_removes_json(tmp_path: Path):
    _init_git_repo(tmp_path, remote="https://gitcode.com/foo/bar.git")
    detect_workspace_fingerprint(tmp_path)
    cache = tmp_path / ".clawcodex" / "lodestone.json"
    assert cache.exists()
    assert invalidate_cache(tmp_path)
    assert not cache.exists()
    # Second invalidate returns False.
    assert not invalidate_cache(tmp_path)


def test_build_anchor_context_with_workspace(tmp_path: Path):
    _init_git_repo(tmp_path, remote="https://gitcode.com/foo/bar.git", default_branch="main")
    ctx = build_anchor_context(tmp_path, default_config(), session_id="sess-1")
    assert ctx.workspace_root == tmp_path
    assert ctx.remote_url == "https://gitcode.com/foo/bar.git"
    assert ctx.branch == "main"
    assert ctx.session_id == "sess-1"


def test_build_anchor_context_without_workspace():
    # ``build_anchor_context`` requires a Path-like; ``None`` is rejected.
    with pytest.raises(TypeError):
        build_anchor_context(None, default_config(), session_id="sess-2")  # type: ignore[arg-type]  # noqa: E501
