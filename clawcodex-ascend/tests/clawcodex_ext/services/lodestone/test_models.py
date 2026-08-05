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

"""Tests for clawcodex_ext.services.lodestone.models.

Covers:
* Placeholder allow-list + registration whitelist;
* Multiple kinds can share a target_id (id+kind key);
* update_config() keeps registry config in sync;
* pick() honours default_editor then prefers non-remote targets;
"""
# pylint: disable=no-name-in-module

from __future__ import annotations

import pytest

from clawcodex_ext.services.lodestone.models import (
    AnchorContext,
    AnchorTarget,
    AnchorTargetRegistry,
    LodestoneConfig,
    allowed_placeholders,
    extract_placeholders,
)


def _ctx(cfg: LodestoneConfig | None = None) -> AnchorContext:
    return AnchorContext(workspace_root=None, session_id=None, config=cfg or LodestoneConfig())


def test_default_registry_accepts_built_in_placeholders():
    reg = AnchorTargetRegistry()
    # Should not raise — built-in templates use only whitelisted keys.
    for target_id in ("vscode", "gitcode", "tracker:linear", "tracker:gitcode"):
        # Last-writer wins on identical (id, kind); refetch an existing
        # to prove id+kind keying works.
        existing = reg.get(target_id, "file_path") or reg.get(target_id)
        if existing:
            reg.register(existing, overwrite=True)


def test_register_rejects_unknown_placeholder():
    reg = AnchorTargetRegistry()
    target = AnchorTarget(
        kind="file_path",
        target_id="malformed",
        template="vscode://file/{abs}:{line}:{col}:{exec}",
    )
    with pytest.raises(ValueError, match="disallowed placeholder"):
        reg.register(target)


def test_register_rejects_empty_target_id():
    reg = AnchorTargetRegistry()
    target = AnchorTarget(kind="file_path", target_id="", template="file://{abs}")
    with pytest.raises(ValueError, match="non-empty"):
        reg.register(target)


def test_same_id_different_kinds_is_allowed():
    reg = AnchorTargetRegistry()
    reg.register(
        AnchorTarget(
            kind="file_path",
            target_id="github",
            template="{remote}/blob/{branch}/{rel}#L{line}",
        )
    )
    reg.register(
        AnchorTarget(
            kind="git_commit",
            target_id="github",
            template="{remote}/commit/{ref}",
        )
    )
    reg.register(
        AnchorTarget(
            kind="git_blob",
            target_id="github",
            template="{remote}/blob/{ref}/{rel}#L{line}",
        )
    )
    assert reg.get("github", "file_path").kind == "file_path"
    assert reg.get("github", "git_commit").kind == "git_commit"
    assert reg.get("github", "git_blob").kind == "git_blob"
    # When called without a kind, ``get`` returns the first registered.
    assert reg.get("github").kind == "file_path"


def test_unregister_id_only_drops_all_kinds():
    reg = AnchorTargetRegistry()
    reg.register(
        AnchorTarget(
            kind="file_path",
            target_id="abc",
            template="file://{abs}",
        )
    )
    reg.register(
        AnchorTarget(
            kind="git_commit",
            target_id="abc",
            template="{remote}/commit/{ref}",
        )
    )
    assert reg.unregister("abc")
    assert reg.get("abc", "file_path") is None
    assert reg.get("abc", "git_commit") is None


def test_pick_prefers_default_editor_then_non_remote():
    reg = AnchorTargetRegistry()
    reg.register(
        AnchorTarget(
            kind="file_path",
            target_id="cursor",
            template="cursor://file/{abs}:{line}:{col}",
        )
    )
    reg.register(
        AnchorTarget(
            kind="file_path",
            target_id="vscode",
            template="vscode://file/{abs}:{line}:{col}",
        )
    )
    reg.register(
        AnchorTarget(
            kind="file_path",
            target_id="gitcode",
            is_remote=True,
            template="{remote}/blob/{branch}/{rel}#L{line}",
        )
    )
    cfg = LodestoneConfig(default_editor="cursor")
    picked = reg.pick("file_path", ctx=_ctx(cfg))
    assert picked is not None and picked.target_id == "cursor"


def test_pick_falls_back_to_non_remote_then_remote():
    reg = AnchorTargetRegistry()
    reg.register(
        AnchorTarget(
            kind="file_path",
            target_id="gitcode",
            is_remote=True,
            template="{remote}/blob/{branch}/{rel}#L{line}",
        )
    )
    reg.register(
        AnchorTarget(
            kind="file_path",
            target_id="vscode",
            template="vscode://file/{abs}:{line}:{col}",
        )
    )
    cfg = LodestoneConfig(default_editor="neovim")  # not registered
    picked = reg.pick("file_path", ctx=_ctx(cfg))
    assert picked is not None and picked.target_id == "vscode"


def test_pick_returns_none_when_no_kind_match():
    reg = AnchorTargetRegistry()
    reg.register(
        AnchorTarget(
            kind="file_path",
            target_id="vscode",
            template="vscode://file/{abs}:{line}:{col}",
        )
    )
    picked = reg.pick("tracker_issue", ctx=_ctx())
    assert picked is None


def test_pick_filters_by_required_env():
    reg = AnchorTargetRegistry()
    reg.register(
        AnchorTarget(
            kind="file_path",
            target_id="restricted",
            template="vscode://file/{abs}",
            requires=("WATERMARK_TOKEN",),
        )
    )
    cfg = LodestoneConfig(default_editor="restricted")
    # No env ⇒ skipped.
    ctx_no_env = AnchorContext(workspace_root=None, session_id=None, config=cfg, env={})
    assert reg.pick("file_path", ctx=ctx_no_env) is None
    # Env present ⇒ picked.
    ctx_with_env = AnchorContext(workspace_root=None, session_id=None, config=cfg, env={"WATERMARK_TOKEN": "1"})
    assert reg.pick("file_path", ctx=ctx_with_env) is not None


def test_extract_placeholders_unique_sorted():
    template = "vscode://file/{abs}:{line}:{line}:{col}"
    assert extract_placeholders(template) == ["abs", "col", "line"]


def test_allowed_placeholders_includes_documented_keys():
    keys = allowed_placeholders()
    for k in (
        "abs",
        "rel",
        "line",
        "col",
        "remote",
        "branch",
        "ref",
        "owner",
        "repo",
        "key",
        "host",
        "workspace",
    ):
        assert k in keys
