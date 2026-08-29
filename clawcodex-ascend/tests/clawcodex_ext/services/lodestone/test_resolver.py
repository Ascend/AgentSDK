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

"""Tests for clawcodex_ext.services.lodestone.resolver."""
# pylint: disable=no-name-in-module

from __future__ import annotations

from pathlib import Path


from clawcodex_ext.services.lodestone.models import (
    AnchorContext,
    LodestoneAnchor,
    LodestoneConfig,
)
from clawcodex_ext.services.lodestone.resolver import (
    _guard_path,
)


# ---------------------------------------------------------------------------
# path-traversal guard
# ---------------------------------------------------------------------------


def test_guard_path_rejects_traversal(tmp_path: Path):
    cfg = LodestoneConfig()
    ctx = AnchorContext(workspace_root=tmp_path, session_id=None, config=cfg)
    bad = LodestoneAnchor(kind="file_path", raw="", file_path="../../etc/passwd")
    ok, reason = _guard_path(bad, ctx)
    assert not ok
    assert "workspace_root" in (reason or "")


def test_guard_path_allows_inside(tmp_path: Path):
    cfg = LodestoneConfig()
    ctx = AnchorContext(workspace_root=tmp_path, session_id=None, config=cfg)
    good = LodestoneAnchor(kind="file_path", raw="", file_path="src/foo.py")
    ok, _ = _guard_path(good, ctx)
    assert ok


def test_guard_path_no_workspace_root():
    cfg = LodestoneConfig()
    ctx = AnchorContext(workspace_root=None, session_id=None, config=cfg)
    a = LodestoneAnchor(kind="file_path", raw="", file_path="src/foo.py")
    ok, reason = _guard_path(a, ctx)
    assert ok
    # Reason is advisory; caller may surface it.
    assert reason == "no workspace_root; path not verified"
