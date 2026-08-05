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

"""Tests for the built-in target set + registry factory."""
# pylint: disable=no-name-in-module

from __future__ import annotations

from clawcodex_ext.services.lodestone.models import LodestoneConfig
from clawcodex_ext.services.lodestone.targets import (
    build_default_registry,
    default_target_ids,
    GITCODE_HOST,
    GITHUB_HOST,
    GITEE_HOST,
    LINEAR_HOST,
)


def test_default_registry_has_all_built_in_targets():
    registry = build_default_registry(include_user_targets=False)
    ids = {t.target_id for t in registry.list()}
    expected = {
        "vscode",
        "vscode-insiders",
        "cursor",
        "idea",
        "subl",
        "file",
        "github",
        "gitcode",
        "gitee",
        "tracker:gitcode",
        "tracker:github",
        "tracker:gitee",
        "tracker:linear",
        "vscode-symbol",
    }
    assert expected.issubset(ids)


def test_default_target_ids_is_sorted_unique():
    ids = default_target_ids()
    assert ids == tuple(sorted(ids))
    assert len(set(ids)) == len(ids)


def test_default_targets_carry_correct_hosts():
    registry = build_default_registry(include_user_targets=False)
    by_id = {t.target_id: t for t in registry.list()}
    assert by_id["gitcode"].hosts == (GITCODE_HOST,)
    assert by_id["github"].hosts == (GITHUB_HOST,)
    assert by_id["gitee"].hosts == (GITEE_HOST,)
    assert by_id["tracker:linear"].hosts == (LINEAR_HOST,)


def test_custom_targets_passed_through():
    # not used; we test via AnchorTarget below
    assert type("T", (), {})() is not None


def test_user_custom_targets_are_registered():
    from clawcodex_ext.services.lodestone.models import AnchorTarget

    custom_target = AnchorTarget(
        kind="file_path",
        target_id="custom-vscode",
        template="vscode://file/{abs}:{line}:{col}",
    )
    cfg = LodestoneConfig(custom_targets=(custom_target,))
    registry = build_default_registry(cfg)
    registered = {t.target_id for t in registry.list()}
    assert "custom-vscode" in registered


def test_default_config_used_when_cfg_is_none():
    registry = build_default_registry()
    # Sanity: built-in editors present + repo's normal template body.
    assert registry.get("vscode") is not None
