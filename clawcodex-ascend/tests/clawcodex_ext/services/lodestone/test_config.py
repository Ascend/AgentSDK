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

"""Tests for clawcodex_ext.services.lodestone.config."""
# pylint: disable=no-name-in-module

from __future__ import annotations

import json


from clawcodex_ext.services.lodestone.config import (
    config_dir,
    default_config,
    load_config,
    save_config,
)
from clawcodex_ext.services.lodestone.models import LodestoneConfig


def test_default_config_has_safe_defaults():
    cfg = default_config()
    assert cfg.enabled is True
    assert cfg.default_editor in {"vscode", "cursor", "idea", "subl"}
    assert cfg.fallback_editor == "file"
    assert cfg.default_tracker_host == "gitcode.com"


def test_load_config_missing_returns_default(tmp_path, monkeypatch):
    monkeypatch.setenv("CLAWCODEX_CONFIG_DIR", str(tmp_path))
    cfg = load_config()
    assert cfg == default_config()


def test_load_config_off_env_disables(monkeypatch, tmp_path):
    monkeypatch.setenv("CLAWCODEX_CONFIG_DIR", str(tmp_path))
    monkeypatch.setenv("LODESTONE", "off")
    cfg = load_config()
    assert cfg.enabled is False


def test_save_load_round_trip(tmp_path):
    cfg_path = tmp_path / "lodestone.json"
    payload = LodestoneConfig(
        enabled=True,
        default_editor="cursor",
        fallback_editor="file",
        default_tracker_host="gitcode.com",
        default_tracker_repo=("chadwweng", "clawcodex"),
    )
    save_config(payload, path=cfg_path)
    # Read it back
    loaded = load_config(path=cfg_path)
    assert loaded.default_editor == "cursor"
    assert loaded.default_tracker_repo == ("chadwweng", "clawcodex")


def test_save_load_handles_tuple_fields(tmp_path):
    cfg_path = tmp_path / "lodestone.json"
    raw = {
        "enabled": True,
        "default_editor": "vscode",
        "default_tracker_host": "gitcode.com",
        "disabled_kinds": ["git_commit", "url"],
        "extra_hosts": ["extra.example.com"],
    }
    cfg_path.write_text(json.dumps(raw), encoding="utf-8")
    loaded = load_config(path=cfg_path)
    assert loaded.disabled_kinds == ("git_commit", "url")
    assert "extra.example.com" in loaded.extra_hosts


def test_save_config_overwrites_atomically(tmp_path):
    cfg_path = tmp_path / "lodestone.json"
    save_config(LodestoneConfig(default_editor="vscode"), path=cfg_path)
    save_config(LodestoneConfig(default_editor="cursor"), path=cfg_path)
    loaded = load_config(path=cfg_path)
    assert loaded.default_editor == "cursor"


def test_config_dir_creates_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("CLAWCODEX_CONFIG_DIR", str(tmp_path / "fresh"))
    out = config_dir()
    assert out.exists()
    assert out.is_dir()
