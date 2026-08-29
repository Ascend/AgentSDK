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

"""Tests for gateway config load/save (atomic, ChannelConfig roundtrip)."""

from __future__ import annotations

import pytest

from clawcodex_ext.services.channels.models import ChannelConfig, ChannelType
from clawcodex_ext.services.im_gateway.config import (
    CommandAllowlistConfig,
    GatewayConfig,
    ReliabilityConfig,
    load_config,
    save_config,
)


def _cfg() -> GatewayConfig:
    return GatewayConfig(
        enabled=True,
        default_targets=["wechat-main"],
        state_dir="~/.clawcodex/gateway",
        reliability=ReliabilityConfig(),
        channels=[
            ChannelConfig(
                type=ChannelType.WECHAT,
                webhook_url="https://ilinkai.weixin.qq.com/dummy",
                name="wechat-main",
                enabled=True,
                extra={"account_id": "default", "base_url": "https://ilinkai.weixin.qq.com"},
            ),
            ChannelConfig(
                type=ChannelType.SLACK,
                webhook_url="https://hooks.example.com/services/T/B/abcdef0123456789",
                name="slack-ops",
                enabled=False,
            ),
        ],
    )


def test_config_roundtrip(tmp_path) -> None:
    p = tmp_path / "channels.yaml"
    save_config(_cfg(), p)
    loaded = load_config(p)
    assert loaded.enabled is True
    assert loaded.default_targets == ["wechat"]
    assert len(loaded.channels) == 2
    wechat = loaded.get_channel("wechat")
    assert wechat is not None
    assert wechat.type is ChannelType.WECHAT
    assert wechat.extra["account_id"] == "default"
    slack = loaded.get_channel("slack-ops")
    assert slack is not None
    assert slack.enabled is False
    assert loaded.command_allowlists == CommandAllowlistConfig()
    raw = p.read_text(encoding="utf-8")
    assert "command_allowlists:" in raw
    assert "repl:" in raw
    assert "orchestrator:" in raw


def test_config_roundtrip_preserves_explicit_command_allowlists(tmp_path) -> None:
    p = tmp_path / "channels.yaml"
    cfg = _cfg()
    cfg.command_allowlists = CommandAllowlistConfig(
        repl=("/stop", "/model"),
        orchestrator=("/server status", "/issue takeover"),
    )

    save_config(cfg, p)
    loaded = load_config(p)

    assert loaded.command_allowlists.repl == ("/stop", "/model")
    assert loaded.command_allowlists.orchestrator == (
        "/server status",
        "/issue takeover",
    )


def test_config_roundtrip_preserves_push_timeout(tmp_path) -> None:
    p = tmp_path / "channels.yaml"
    cfg = _cfg()
    cfg.push_timeout_seconds = 0.25

    save_config(cfg, p)

    assert load_config(p).push_timeout_seconds == 0.25


def test_config_rejects_non_positive_push_timeout() -> None:
    with pytest.raises(ValueError, match="push_timeout_seconds"):
        GatewayConfig(push_timeout_seconds=0)


def test_config_explicit_empty_command_allowlists_block_all_slash_commands(tmp_path) -> None:
    p = tmp_path / "channels.yaml"
    p.write_text(
        "command_allowlists:\n  repl: []\n  orchestrator: []\n",
        encoding="utf-8",
    )

    loaded = load_config(p)

    assert loaded.command_allowlists.repl == ()
    assert loaded.command_allowlists.orchestrator == ()


def test_config_omitted_runtime_allowlist_keeps_its_default() -> None:
    loaded = GatewayConfig.from_dict({"command_allowlists": {"repl": []}})

    assert loaded.command_allowlists.repl == ()
    assert loaded.command_allowlists.orchestrator == CommandAllowlistConfig().orchestrator


def test_config_rejects_malformed_command_allowlist() -> None:
    with pytest.raises(ValueError, match="command_allowlists.repl"):
        GatewayConfig.from_dict({"command_allowlists": {"repl": "/stop"}})


def test_load_config_normalizes_legacy_wechat_name_to_single_instance(tmp_path) -> None:
    p = tmp_path / "channels.yaml"
    p.write_text(
        """
default_targets:
  - wechat-main
channels:
  - type: wechat
    webhook_url: https://ilinkai.weixin.qq.com/dummy
    name: wechat-main
    enabled: true
    extra:
      account_id: default
      base_url: https://ilinkai.weixin.qq.com
""",
        encoding="utf-8",
    )

    loaded = load_config(p)

    assert loaded.default_targets == ["wechat"]
    assert loaded.get_channel("wechat") is not None
    assert loaded.get_channel("wechat-main") is None


def test_config_channel_replace_and_remove(tmp_path) -> None:
    cfg = _cfg()
    new = ChannelConfig(
        type=ChannelType.WECHAT,
        webhook_url="https://ilinkai.weixin.qq.com/dummy",
        name="wechat-main",
        enabled=False,
    )
    cfg.replace_channel(new)
    assert cfg.get_channel("wechat").enabled is False
    assert cfg.remove_channel("slack-ops") is True
    assert cfg.get_channel("slack-ops") is None
    assert cfg.remove_channel("nope") is False


def test_config_replace_channel_keeps_one_entry_per_type() -> None:
    cfg = _cfg()
    replacement = ChannelConfig(
        type=ChannelType.SLACK,
        webhook_url="https://hooks.example.com/services/T/B/newabcdef012345",
        name="slack-alerts",
        enabled=True,
    )
    cfg.replace_channel(replacement)

    assert cfg.get_channel("slack-ops") is None
    assert cfg.get_channel("slack-alerts") == replacement
    assert [c.type for c in cfg.channels].count(ChannelType.SLACK) == 1


def test_replace_channel_by_type_updates_default_targets_when_name_changes() -> None:
    cfg = _cfg()
    cfg.default_targets = ["slack-ops", "wechat-main"]
    replacement = ChannelConfig(
        type=ChannelType.SLACK,
        webhook_url="https://hooks.example.com/services/T/B/newabcdef012345",
        name="slack-alerts",
        enabled=True,
    )

    cfg.replace_channel(replacement)

    assert cfg.default_targets == ["slack-alerts", "wechat"]


def test_load_config_collapses_duplicate_channel_types(tmp_path) -> None:
    p = tmp_path / "channels.yaml"
    p.write_text(
        """
channels:
  - type: slack
    webhook_url: https://hooks.example.com/services/T/B/oldabcdef012345
    name: slack-old
    enabled: true
    extra: null
  - type: slack
    webhook_url: https://hooks.example.com/services/T/B/newabcdef012345
    name: slack-new
    enabled: false
    extra: null
""",
        encoding="utf-8",
    )

    loaded = load_config(p)

    assert loaded.get_channel("slack-old") is None
    assert loaded.get_channel("slack-new") is not None
    assert len(loaded.channels) == 1


def test_save_config_collapses_duplicate_channel_types(tmp_path) -> None:
    p = tmp_path / "channels.yaml"
    cfg = GatewayConfig(
        channels=[
            ChannelConfig(
                type=ChannelType.SLACK,
                webhook_url="https://hooks.example.com/services/T/B/oldabcdef012345",
                name="slack-old",
                enabled=True,
            ),
            ChannelConfig(
                type=ChannelType.SLACK,
                webhook_url="https://hooks.example.com/services/T/B/newabcdef012345",
                name="slack-new",
                enabled=False,
            ),
        ],
    )

    save_config(cfg, p)

    raw = p.read_text(encoding="utf-8")
    assert "slack-old" not in raw
    assert raw.count("type: slack") == 1
    loaded = load_config(p)
    assert loaded.get_channel("slack-old") is None
    assert loaded.get_channel("slack-new") is not None
    assert len(loaded.channels) == 1


def test_config_atomic_save_uses_replace(tmp_path) -> None:
    p = tmp_path / "channels.yaml"
    save_config(_cfg(), p)
    # no .tmp left behind
    assert not (tmp_path / "channels.yaml.tmp").exists()
    assert p.exists()


def test_channel_config_to_dict_roundtrip_preserves_extra() -> None:
    c = ChannelConfig(
        type=ChannelType.WECHAT,
        webhook_url="https://x.example.com/dummy",
        name="w1",
        extra={"allowed_users": ["u1", "u2"], "max_consecutive_failures": 10},
    )
    d = c.to_dict()
    back = ChannelConfig.from_dict(d)
    assert back == c
    assert back.extra == {"allowed_users": ["u1", "u2"], "max_consecutive_failures": 10}


def test_load_config_missing_file_returns_default(tmp_path) -> None:
    cfg = load_config(tmp_path / "nope.yaml")
    assert cfg.enabled is True
    assert cfg.channels == []


def test_migrate_legacy_state_dir_moves_legacy_and_is_idempotent(tmp_path, monkeypatch) -> None:
    from clawcodex_ext.services.im_gateway import config as cfg_mod

    legacy = tmp_path / "im-gateway"
    legacy.mkdir()
    (legacy / "channels.yaml").write_text("enabled: true\n", encoding="utf-8")
    (legacy / "gateway.pid").write_text("123\n", encoding="utf-8")
    target = tmp_path / "gateway"

    monkeypatch.setattr(cfg_mod, "LEGACY_STATE_DIR", str(legacy))

    # First call: target absent, legacy present → move.
    result = cfg_mod.migrate_legacy_state_dir(str(target))
    assert result == target
    assert (target / "channels.yaml").exists()
    assert (target / "gateway.pid").read_text(encoding="utf-8") == "123\n"
    assert not legacy.exists()

    # Second call: target now exists → idempotent no-op.
    assert cfg_mod.migrate_legacy_state_dir(str(target)) == target
    assert (target / "channels.yaml").exists()


def test_migrate_legacy_state_dir_no_legacy_returns_target_uncreated(tmp_path, monkeypatch) -> None:
    from clawcodex_ext.services.im_gateway import config as cfg_mod

    monkeypatch.setattr(cfg_mod, "LEGACY_STATE_DIR", str(tmp_path / "absent-im-gateway"))
    target = tmp_path / "gateway"
    result = cfg_mod.migrate_legacy_state_dir(str(target))
    assert result == target
    assert not target.exists()  # caller owns mkdir


def test_migration_copy_failure_does_not_publish_partial_target(tmp_path, monkeypatch) -> None:
    from clawcodex_ext.services.im_gateway import config as cfg_mod

    legacy = tmp_path / "im-gateway"
    legacy.mkdir()
    (legacy / "channels.yaml").write_text("enabled: true\n", encoding="utf-8")
    target = tmp_path / "gateway"
    monkeypatch.setattr(cfg_mod, "LEGACY_STATE_DIR", str(legacy))
    original_rename = cfg_mod.Path.rename
    original_copytree = cfg_mod.shutil.copytree

    def fail_rename(self, destination):
        raise OSError("cross-device")

    def fail_copy(source, destination):
        destination.mkdir()
        (destination / "partial").write_text("partial", encoding="utf-8")
        raise OSError("disk full")

    monkeypatch.setattr(cfg_mod.Path, "rename", fail_rename)
    monkeypatch.setattr(cfg_mod.shutil, "copytree", fail_copy)
    with pytest.raises(OSError, match="disk full"):
        cfg_mod.migrate_legacy_state_dir(target)

    assert legacy.exists()
    assert not target.exists()
    assert not list(tmp_path.glob(".gateway.migrate-*"))

    monkeypatch.setattr(cfg_mod.Path, "rename", original_rename)
    monkeypatch.setattr(cfg_mod.shutil, "copytree", original_copytree)
    assert cfg_mod.migrate_legacy_state_dir(target) == target
    assert (target / "channels.yaml").exists()


def test_channel_type_lookup_is_case_insensitive() -> None:
    config = GatewayConfig(
        channels=[
            ChannelConfig(
                type=ChannelType.WECHAT,
                webhook_url="https://example.com/hook",
                name="wechat",
            )
        ]
    )

    assert config.get_channel_by_type("WECHAT").name == "wechat"
