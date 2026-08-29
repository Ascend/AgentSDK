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

"""Tests for the `clawcodex-dev channels` CLI command (config ops + wizard)."""
# pylint: disable=no-name-in-module, too-many-lines

from __future__ import annotations

import json

from clawcodex_ext.cli.channels_cmd import commands as ch
from clawcodex_ext.services.channels.models import ChannelConfig, ChannelType
from clawcodex_ext.services.im_gateway.models import IM_DIRECT_ALL_ORIGIN, WECHAT_DIRECT_ALL_ORIGIN


def _slack(name: str = "slack-ops", enabled: bool = False) -> ChannelConfig:
    return ChannelConfig(
        type=ChannelType.SLACK,
        webhook_url="https://hooks.example.com/services/T/B/abcdef0123456789",
        name=name,
        enabled=enabled,
    )


def _write_json(path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")


def test_list_channels_empty(tmp_path) -> None:
    p = tmp_path / "channels.yaml"
    assert ch.list_channels(str(p)) == []


def test_add_list_remove_channel(tmp_path) -> None:
    p = tmp_path / "channels.yaml"
    ch.add_channel(str(p), _slack("s1", enabled=True))
    listed = ch.list_channels(str(p))
    assert listed == [{"name": "s1", "type": "slack", "enabled": True}]
    assert ch.remove_channel(str(p), "s1") is True
    assert ch.list_channels(str(p)) == []


def test_remove_wechat_channel_deletes_owned_state(tmp_path) -> None:
    p = tmp_path / "channels.yaml"
    ch.add_channel(str(p), ch.build_default_channel("wechat"))
    wechat_dir = tmp_path / "wechat"
    wechat_dir.mkdir()
    owned_files = [
        wechat_dir / "wechat_auth.json",
        wechat_dir / "wechat_auth.key",
        wechat_dir / "wechat_auth.json.tmp",
        wechat_dir / "wechat_pairing.json",
        wechat_dir / "wechat_pairing.json.lock",
        wechat_dir / "wechat_pairing.json.tmp",
        wechat_dir / "wechat-main_auth.json",
        wechat_dir / "wechat-main_auth.key",
        wechat_dir / "wechat-main_auth.json.tmp",
        wechat_dir / "wechat-main_pairing.json",
        wechat_dir / "wechat-main_pairing.json.lock",
        wechat_dir / "wechat-main_pairing.json.tmp",
        tmp_path / "wechat_context_tokens.json",
        tmp_path / "wechat_accounts.json",
    ]
    for path in owned_files:
        path.write_text("owned", encoding="utf-8")
    preserved_files = [
        tmp_path / "outbox.ndjson",
        tmp_path / "processed_inbound.ndjson",
        tmp_path / "dead_letter.ndjson",
        tmp_path / "audit.ndjson",
    ]
    for path in preserved_files:
        path.write_text("preserve", encoding="utf-8")

    assert ch.remove_channel(str(p), "wechat") is True

    assert ch.list_channels(str(p)) == []
    assert all(not path.exists() for path in owned_files)
    assert all(path.exists() for path in preserved_files)
    assert p.exists()


def test_remove_feishu_channel_prunes_last_sender_key(tmp_path) -> None:
    p = tmp_path / "channels.yaml"
    ch.add_channel(str(p), ch.build_channel_from_inputs("feishu", "feishu", {}))
    _write_json(tmp_path / "feishu_last_senders.json", {"feishu": "ou_old", "other": "ou_keep"})

    assert ch.remove_channel(str(p), "feishu") is True

    assert json.loads((tmp_path / "feishu_last_senders.json").read_text(encoding="utf-8")) == {"other": "ou_keep"}


def test_remove_feishu_channel_deletes_empty_last_sender_file(tmp_path) -> None:
    p = tmp_path / "channels.yaml"
    ch.add_channel(str(p), ch.build_channel_from_inputs("feishu", "feishu", {}))
    _write_json(tmp_path / "feishu_last_senders.json", {"feishu": "ou_old"})

    assert ch.remove_channel(str(p), "feishu") is True

    assert not (tmp_path / "feishu_last_senders.json").exists()


def test_remove_generic_channel_preserves_gateway_state(tmp_path) -> None:
    p = tmp_path / "channels.yaml"
    ch.add_channel(str(p), _slack("s1", enabled=True))
    preserved = tmp_path / "outbox.ndjson"
    preserved.write_text("preserve", encoding="utf-8")

    assert ch.remove_channel(str(p), "s1") is True

    assert preserved.exists()


def test_add_channel_replaces_existing_channel_of_same_type(tmp_path) -> None:
    p = tmp_path / "channels.yaml"
    ch.add_channel(str(p), _slack("slack-old", enabled=False))
    ch.add_channel(str(p), _slack("slack-new", enabled=True))

    assert ch.list_channels(str(p)) == [{"name": "slack-new", "type": "slack", "enabled": True}]


def test_update_channel_replaces(tmp_path) -> None:
    p = tmp_path / "channels.yaml"
    ch.add_channel(str(p), _slack("s1", enabled=True))
    ch.update_channel(str(p), _slack("s1", enabled=False))
    listed = ch.list_channels(str(p))
    assert listed[0]["enabled"] is False
    assert ch.update_channel(str(p), _slack("renamed")) is True
    assert ch.list_channels(str(p)) == [{"name": "renamed", "type": "slack", "enabled": False}]
    missing_type = ChannelConfig(
        type=ChannelType.DISCORD,
        webhook_url="https://discord.com/api/webhooks/1/token",
        name="discord-main",
    )
    assert ch.update_channel(str(p), missing_type) is False


def test_format_status(tmp_path) -> None:
    p = tmp_path / "channels.yaml"
    ch.add_channel(str(p), _slack("s1", enabled=True))
    out = ch.format_status(str(p))
    assert "s1" in out and "enabled" in out
    assert "no channels" in ch.format_status(str(p), "nope")


def test_format_status_shows_connected_clients(tmp_path, monkeypatch) -> None:
    p = tmp_path / "channels.yaml"
    ch.add_channel(str(p), _slack("s1", enabled=True))
    monkeypatch.setattr(
        ch,
        "_read_gateway_runtime_status",
        lambda state_dir=None: {
            "gateway_running": True,
            "peers": [
                {"session_id": "repl-12345", "host_type": "repl", "online": True},
                {"session_id": "orch-67890", "host_type": "orchestrator", "online": True},
            ],
        },
        raising=False,
    )

    out = ch.format_status(str(p), state_dir=str(tmp_path))

    assert "connected clients: repl (session=repl-12345, pid=12345, online)" in out
    assert "orchestrator (session=orch-67890, online)" in out


def test_format_status_shows_connected_client_pid(tmp_path, monkeypatch) -> None:
    p = tmp_path / "channels.yaml"
    ch.add_channel(str(p), _slack("s1", enabled=True))
    monkeypatch.setattr(
        ch,
        "_read_gateway_runtime_status",
        lambda state_dir=None: {
            "gateway_running": True,
            "peers": [
                {
                    "session_id": "repl-4321-1",
                    "host_type": "repl",
                    "online": True,
                    "pid": 4321,
                },
            ],
        },
        raising=False,
    )

    out = ch.format_status(str(p), state_dir=str(tmp_path))

    assert "repl (session=repl-4321-1, pid=4321, online)" in out


def test_format_status_shows_no_clients_when_peers_empty(tmp_path, monkeypatch) -> None:
    p = tmp_path / "channels.yaml"
    ch.add_channel(str(p), _slack("s1", enabled=True))
    monkeypatch.setattr(
        ch,
        "_read_gateway_runtime_status",
        lambda state_dir=None: {"gateway_running": True, "peers": []},
        raising=False,
    )

    out = ch.format_status(str(p), state_dir=str(tmp_path))

    assert "connected clients: none" in out


def test_format_status_omits_clients_line_when_daemon_down(tmp_path, monkeypatch) -> None:
    p = tmp_path / "channels.yaml"
    ch.add_channel(str(p), _slack("s1", enabled=True))
    monkeypatch.setattr(
        ch,
        "_read_gateway_runtime_status",
        lambda state_dir=None: {"gateway_running": False},
        raising=False,
    )

    out = ch.format_status(str(p), state_dir=str(tmp_path))

    assert "connected clients" not in out


def test_format_status_wechat_includes_login_and_conversation_connection(
    tmp_path,
    monkeypatch,
) -> None:
    p = tmp_path / "channels.yaml"
    ch.add_channel(str(p), ch.build_default_channel("wechat"))
    monkeypatch.setattr(
        ch,
        "wechat_login_status",
        lambda name, state_dir=None: "logged_in (account_id=acct, user_id=bot_user)",
    )
    monkeypatch.setattr(
        ch,
        "_read_gateway_runtime_status",
        lambda state_dir=None: {
            "bindings": [
                {
                    "origin": WECHAT_DIRECT_ALL_ORIGIN,
                    "session_id": "repl-main",
                    "host_type": "repl",
                    "connection_state": "active",
                }
            ],
            "peers": [
                {
                    "session_id": "repl-main",
                    "origin": WECHAT_DIRECT_ALL_ORIGIN,
                    "host_type": "repl",
                    "online": True,
                }
            ],
            "channel_health": [
                {
                    "channel_id": "wechat",
                    "healthy": True,
                    "account_status": "logged_in",
                }
            ],
        },
        raising=False,
    )

    out = ch.format_status(str(p), "wechat", state_dir=str(tmp_path))

    assert "login: logged_in" in out
    assert "conversation: connected to repl" in out
    assert "session=repl-main" in out


def test_format_status_wechat_reports_disconnected_conversation(tmp_path, monkeypatch) -> None:
    p = tmp_path / "channels.yaml"
    ch.add_channel(str(p), ch.build_default_channel("wechat"))
    monkeypatch.setattr(ch, "wechat_login_status", lambda name, state_dir=None: "unconfigured")
    monkeypatch.setattr(ch, "_read_gateway_runtime_status", lambda state_dir=None: {}, raising=False)

    out = ch.format_status(str(p), "wechat", state_dir=str(tmp_path))

    assert "conversation: disconnected" in out


def test_format_status_feishu_includes_generic_conversation_connection(
    tmp_path,
    monkeypatch,
) -> None:
    p = tmp_path / "channels.yaml"
    ch.add_channel(str(p), ch.build_channel_from_inputs("feishu", "feishu", {}))
    monkeypatch.setattr(
        ch,
        "_read_gateway_runtime_status",
        lambda state_dir=None: {
            "bindings": [
                {
                    "origin": IM_DIRECT_ALL_ORIGIN,
                    "session_id": "repl-main",
                    "host_type": "repl",
                    "connection_state": "active",
                }
            ],
            "peers": [
                {
                    "session_id": "repl-main",
                    "origin": IM_DIRECT_ALL_ORIGIN,
                    "host_type": "repl",
                    "online": True,
                }
            ],
            "channel_health": [
                {
                    "channel_id": "feishu",
                    "healthy": True,
                    "account_status": "websocket:connected",
                }
            ],
        },
        raising=False,
    )

    out = ch.format_status(str(p), "feishu", state_dir=str(tmp_path))

    assert "conversation: connected to repl" in out
    assert "session=repl-main" in out


def test_format_status_wechat_reports_offline_binding_as_disconnected(tmp_path, monkeypatch) -> None:
    p = tmp_path / "channels.yaml"
    ch.add_channel(str(p), ch.build_default_channel("wechat"))
    monkeypatch.setattr(
        ch,
        "wechat_login_status",
        lambda name, state_dir=None: "logged_in (account_id=acct, user_id=bot_user)",
    )
    monkeypatch.setattr(
        ch,
        "_read_gateway_runtime_status",
        lambda state_dir=None: {
            "bindings": [
                {
                    "origin": WECHAT_DIRECT_ALL_ORIGIN,
                    "session_id": "repl-main",
                    "host_type": "repl",
                    "connection_state": "offline",
                }
            ],
            "peers": [
                {
                    "session_id": "repl-main",
                    "origin": WECHAT_DIRECT_ALL_ORIGIN,
                    "host_type": "repl",
                    "online": False,
                }
            ],
        },
        raising=False,
    )

    out = ch.format_status(str(p), "wechat", state_dir=str(tmp_path))

    assert "conversation: disconnected" in out
    assert "connected to repl" not in out


def test_restart_channel_reports_connected_client_pid_on_exception(tmp_path, monkeypatch, capsys) -> None:
    from extensions.im_gateway.server import DaemonPaths

    paths = DaemonPaths.for_state_dir(tmp_path)
    ch.add_channel(str(paths.state_dir / "channels.yaml"), ch.build_default_channel("wechat"))
    monkeypatch.setattr(ch, "_daemon_alive", lambda daemon: True)

    class _FakeGatewayIpcClient:
        def __init__(self, *_args, **_kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_exc):
            return None

        async def status(self):
            return {
                "gateway_running": True,
                "peers": [
                    {
                        "session_id": "orchestrator-4321",
                        "host_type": "orchestrator",
                        "online": True,
                        "pid": 4321,
                    }
                ],
            }

        async def reload_channel(self, name):
            raise RuntimeError("adapter busy")

    monkeypatch.setattr(
        "clawcodex_ext.services.im_gateway.ipc_client.GatewayIpcClient",
        _FakeGatewayIpcClient,
    )

    rc = ch.restart_channel("wechat", state_dir=str(tmp_path))
    captured = capsys.readouterr()

    assert rc == 1
    assert "restart failed" in captured.err
    assert "REPL/Orchestrator" in captured.err
    assert "pid=4321" in captured.err


def test_build_channel_from_inputs_wechat(tmp_path) -> None:
    p = tmp_path / "channels.yaml"
    channel = ch.build_channel_from_inputs(
        "wechat",
        "wechat",
        {
            "base_url": "https://ilinkai.weixin.qq.com",
            "account_id": "default",
            "enabled": "true",
        },
    )
    assert channel.type is ChannelType.WECHAT
    assert channel.enabled is True
    assert channel.extra["base_url"] == "https://ilinkai.weixin.qq.com"
    assert "allowed_users" not in channel.extra
    # round-trip via save/load
    ch.add_channel(str(p), channel)
    loaded = ch.list_channels(str(p))
    assert loaded == [{"name": "wechat", "type": "wechat", "enabled": True}]


def test_build_channel_from_inputs_feishu_websocket() -> None:
    channel = ch.build_channel_from_inputs(
        "feishu",
        "feishu",
        {
            "connection_mode": "websocket",
            "app_id": "cli_app",
            "app_secret": "secret",
            "encrypt_key": "encrypt-key",
            "verification_token": "verification-token",
            "domain": "lark",
            "allowed_user_open_id": "ou_allowed",
            "bot_open_id": "ou_bot",
            "bot_name": "ClawCodex",
            "ws_reconnect_interval": "180",
            "ws_ping_interval": "",
            "ws_ping_timeout": "12",
            "enabled": "true",
        },
    )

    assert channel.type is ChannelType.FEISHU
    assert channel.webhook_url == ""
    assert channel.name == "feishu"
    assert channel.enabled is True
    assert channel.extra["connection_mode"] == "websocket"
    assert channel.extra["app_id"] == "cli_app"
    assert channel.extra["app_secret"] == "secret"
    assert channel.extra["encrypt_key"] == "encrypt-key"
    assert channel.extra["verification_token"] == "verification-token"
    assert channel.extra["domain"] == "lark"
    assert channel.extra["allowed_user_open_id"] == "ou_allowed"
    assert channel.extra["bot_open_id"] == "ou_bot"
    assert channel.extra["bot_name"] == "ClawCodex"
    assert "websocket" not in channel.extra
    assert channel.extra["send"] == {
        "sdk_send_attempts": 3,
        "sdk_send_backoff_base_seconds": 1.0,
    }


def test_build_channel_from_inputs_feishu_webhook_compat() -> None:
    channel = ch.build_channel_from_inputs(
        "feishu",
        "feishu",
        {
            "connection_mode": "webhook",
            "webhook_url": "https://open.feishu.cn/open-apis/bot/v2/hook/abcdef",
            "secret": "sign_secret",
            "enabled": "true",
        },
    )

    assert channel.webhook_url == "https://open.feishu.cn/open-apis/bot/v2/hook/abcdef"
    assert channel.extra == {"connection_mode": "webhook", "secret": "sign_secret"}


def test_wizard_add_then_edit_then_remove(tmp_path, monkeypatch) -> None:
    p = tmp_path / "channels.yaml"
    monkeypatch.setattr(ch, "wechat_login", lambda name, state_dir=None: 0)
    monkeypatch.setattr(ch, "wechat_login_status", lambda name, state_dir=None: "logged_in")
    # WeChat is the second option; unauthenticated users enter the scan-and-edit flow.
    inputs = iter(
        [
            "2",  # select wechat (not logged in → add flow: scan + edit menu)
            "",  # ESC wechat edit menu → back to channel select
            "",  # ESC exit wizard
        ]
    )
    rc = ch.run_wizard(str(p), input_fn=lambda _prompt: next(inputs))
    assert rc == 0
    listed = ch.list_channels(str(p))
    assert listed == [{"name": "wechat", "type": "wechat", "enabled": True}]

    # Edit the logged-in channel: option 4 toggles it and option 5 removes it.
    inputs2 = iter(
        [
            "2",  # select wechat (logged in → edit menu)
            "4",  # toggle enable/disable
            "",  # ESC wechat edit → back to channel select
            "2",  # select wechat again
            "5",  # remove channel
            "y",  # confirm
            "",  # ESC exit wizard
        ]
    )
    ch.run_wizard(str(p), input_fn=lambda _prompt: next(inputs2))
    assert ch.list_channels(str(p)) == []


def test_wechat_wizard_creates_default_config_and_runs_scan_before_options(tmp_path, monkeypatch, capsys) -> None:
    p = tmp_path / "channels.yaml"
    login_calls: list[str] = []

    def _fake_login(name: str, *, state_dir: str | None = None) -> int:
        login_calls.append(name)
        return 0

    monkeypatch.setattr(ch, "wechat_login", _fake_login)
    monkeypatch.setattr(ch, "wechat_login_status", lambda name, state_dir=None: "unconfigured")
    inputs = iter(
        [
            "2",  # select wechat (not logged in → add flow: scan + edit menu)
            "",  # ESC post-scan edit menu → back to channel select
            "",  # ESC exit wizard
        ]
    )
    prompts: list[str] = []

    def _input(prompt: str) -> str:
        prompts.append(prompt)
        return next(inputs)

    assert ch.run_wizard(str(p), input_fn=_input) == 0

    assert login_calls == ["wechat"]
    assert ch.list_channels(str(p)) == [{"name": "wechat", "type": "wechat", "enabled": True}]
    text = (capsys.readouterr().out + "\n".join(prompts)).lower()
    assert "account id" not in text
    assert "allowed users" not in text
    assert "base url" not in text
    assert "编辑字段" not in text


def test_wizard_remove_wechat_channel_deletes_owned_state(tmp_path, monkeypatch) -> None:
    p = tmp_path / "channels.yaml"
    ch.add_channel(str(p), ch.build_default_channel("wechat"))
    monkeypatch.setattr(ch, "wechat_login_status", lambda name, state_dir=None: "logged_in")
    (tmp_path / "wechat_context_tokens.json").write_text('{"acct:user": "ctx"}', encoding="utf-8")
    (tmp_path / "wechat").mkdir()
    (tmp_path / "wechat" / "wechat_auth.json").write_text("owned", encoding="utf-8")
    inputs = iter(
        [
            "2",  # select wechat (logged in → edit menu)
            "5",  # remove channel
            "y",  # confirm
            "",  # ESC exit wizard
        ]
    )

    assert ch.run_wizard(str(p), input_fn=lambda _prompt: next(inputs)) == 0

    assert ch.list_channels(str(p)) == []
    assert not (tmp_path / "wechat_context_tokens.json").exists()
    assert not (tmp_path / "wechat" / "wechat_auth.json").exists()


def test_existing_wechat_wizard_options_do_not_prompt_for_internal_fields(tmp_path, monkeypatch, capsys) -> None:
    p = tmp_path / "channels.yaml"
    ch.add_channel(str(p), ch.build_default_channel("wechat"))
    monkeypatch.setattr(ch, "wechat_login_status", lambda name, state_dir=None: "logged_in")
    inputs = iter(
        [
            "2",  # select existing wechat (logged in → edit menu)
            "2",  # status
            "",  # ESC wechat edit → back to channel select
            "",  # ESC exit wizard
        ]
    )
    prompts: list[str] = []

    def _input(prompt: str) -> str:
        prompts.append(prompt)
        return next(inputs)

    assert ch.run_wizard(str(p), input_fn=_input) == 0

    text = capsys.readouterr().out + "\n".join(prompts)
    assert "编辑字段" not in text
    assert "account id" not in text.lower()
    assert "allowed users" not in text.lower()
    assert "base url" not in text.lower()
    assert "logged_in" in text


def test_wechat_login_status_unconfigured(tmp_path) -> None:
    from extensions.im_gateway.server import DaemonPaths

    paths = DaemonPaths.for_state_dir(tmp_path)
    ch.add_channel(str(paths.state_dir / "channels.yaml"), ch.build_default_channel("wechat"))
    out = ch.wechat_login_status("wechat", state_dir=str(tmp_path))
    assert "unconfigured" in out


def test_wechat_login_status_reuses_legacy_wechat_main_auth(tmp_path) -> None:
    from extensions.im_gateway.server import DaemonPaths

    from clawcodex_ext.services.channels.wechat_ilink import WeChatAuthRecord, WeChatIlinkAuthStore

    paths = DaemonPaths.for_state_dir(tmp_path)
    ch.add_channel(str(paths.state_dir / "channels.yaml"), ch.build_default_channel("wechat"))
    legacy_auth = paths.state_dir / "wechat" / "wechat-main_auth.json"
    WeChatIlinkAuthStore(legacy_auth).save(
        WeChatAuthRecord(
            bot_token="bot_tok_123",
            account_id="acct",
            base_url="https://ilinkai.weixin.qq.com",
            user_id="bot_user",
        )
    )

    out = ch.wechat_login_status("wechat", state_dir=str(tmp_path))

    assert "logged_in" in out
    assert "account_id=acct" in out


def test_wechat_login_reports_ilink_http_error_without_traceback(tmp_path, monkeypatch, capsys) -> None:
    from extensions.im_gateway.server import DaemonPaths

    from clawcodex_ext.services.channels.wechat_ilink import (
        WeChatIlinkChannelAdapter,
        _IlinkHttpError,
    )

    paths = DaemonPaths.for_state_dir(tmp_path)
    ch.add_channel(str(paths.state_dir / "channels.yaml"), ch.build_default_channel("wechat"))

    async def _raise_404(self, **_kwargs) -> dict:
        raise _IlinkHttpError(404, b"not found")

    monkeypatch.setattr(WeChatIlinkChannelAdapter, "qr_login", _raise_404)

    rc = ch.wechat_login("wechat", state_dir=str(tmp_path))

    captured = capsys.readouterr()
    assert rc == 1
    assert "iLink 登录失败" in captured.err
    assert "404" in captured.err
    assert "/getlogincode" not in captured.err


def test_wechat_login_prints_qr_before_waiting_and_reports_success(tmp_path, monkeypatch, capsys) -> None:
    from extensions.im_gateway.server import DaemonPaths

    from clawcodex_ext.services.channels.wechat_ilink import WeChatIlinkChannelAdapter

    paths = DaemonPaths.for_state_dir(tmp_path)
    ch.add_channel(str(paths.state_dir / "channels.yaml"), ch.build_default_channel("wechat"))
    monkeypatch.setattr(ch, "_print_terminal_qr", lambda scan_data: None)

    async def _fake_qr_login(self, *, on_code=None, on_status=None, **_kwargs) -> dict:
        if on_code is not None:
            on_code("https://ilinkai.weixin.qq.com/qr/abc")
        if on_status is not None:
            on_status("wait")
            on_status("scaned")
        return {
            "status": "confirmed",
            "code_url": "https://ilinkai.weixin.qq.com/qr/abc",
            "bot_token": "bot_tok_123",
            "account_id": "bot_account_1",
        }

    monkeypatch.setattr(WeChatIlinkChannelAdapter, "qr_login", _fake_qr_login)
    # daemon is "running" so login takes the live-reload path (no subprocess)
    monkeypatch.setattr(ch, "_daemon_alive", lambda daemon: True)
    monkeypatch.setattr(ch, "restart_channel", lambda name, state_dir=None: 0)

    rc = ch.wechat_login("wechat", state_dir=str(tmp_path))

    captured = capsys.readouterr()
    assert rc == 0
    assert "https://ilinkai.weixin.qq.com/qr/abc" in captured.out
    assert "已扫码" in captured.out
    assert "WeChat 登录成功 (account_id=bot_account_1)" in captured.out


def test_wechat_login_reloads_channel_after_success(tmp_path, monkeypatch, capsys) -> None:
    from extensions.im_gateway.server import DaemonPaths

    from clawcodex_ext.services.channels.wechat_ilink import WeChatIlinkChannelAdapter

    paths = DaemonPaths.for_state_dir(tmp_path)
    ch.add_channel(str(paths.state_dir / "channels.yaml"), ch.build_default_channel("wechat"))
    monkeypatch.setattr(ch, "_print_terminal_qr", lambda scan_data: None)
    reload_calls: list[tuple[str, str | None]] = []

    async def _fake_qr_login(self, *, on_code=None, on_status=None, **_kwargs) -> dict:
        if on_code is not None:
            on_code("https://ilinkai.weixin.qq.com/qr/abc")
        if on_status is not None:
            on_status("confirmed")
        return {
            "status": "confirmed",
            "bot_token": "bot_tok_123",
            "account_id": "bot_account_1",
        }

    def _fake_restart(name: str, *, state_dir: str | None = None) -> int:
        reload_calls.append((name, state_dir))
        return 0

    monkeypatch.setattr(WeChatIlinkChannelAdapter, "qr_login", _fake_qr_login)
    monkeypatch.setattr(ch, "restart_channel", _fake_restart)
    # daemon is "running" so login takes the live-reload path
    monkeypatch.setattr(ch, "_daemon_alive", lambda daemon: True)

    rc = ch.wechat_login("wechat", state_dir=str(tmp_path))

    assert rc == 0
    assert reload_calls == [("wechat", str(tmp_path))]
    assert "WeChat 登录成功" in capsys.readouterr().out


def test_wechat_login_auto_starts_daemon_when_down(tmp_path, monkeypatch, capsys) -> None:
    """When the daemon is DOWN, login must start it (not silently succeed)."""
    from extensions.im_gateway.server import DaemonPaths

    from clawcodex_ext.services.channels.wechat_ilink import WeChatIlinkChannelAdapter

    paths = DaemonPaths.for_state_dir(tmp_path)
    ch.add_channel(str(paths.state_dir / "channels.yaml"), ch.build_default_channel("wechat"))
    monkeypatch.setattr(ch, "_print_terminal_qr", lambda scan_data: None)

    async def _fake_qr_login(self, *, on_code=None, on_status=None, **_kwargs) -> dict:
        return {"status": "confirmed", "bot_token": "bot_tok_123", "account_id": "bot_account_1"}

    start_calls: list[bool] = []

    class _FakeDaemon:
        def __init__(self, *a, **kw):
            pass

        def start(self):
            start_calls.append(True)
            return 0

    monkeypatch.setattr(WeChatIlinkChannelAdapter, "qr_login", _fake_qr_login)
    monkeypatch.setattr(ch, "_daemon_alive", lambda daemon: False)
    monkeypatch.setattr("extensions.im_gateway.server.GatewayDaemon", _FakeDaemon)
    # restart_channel must NOT be called when daemon was down
    monkeypatch.setattr(ch, "restart_channel", lambda *a, **k: 1)

    rc = ch.wechat_login("wechat", state_dir=str(tmp_path))
    assert rc == 0
    assert start_calls == [True]
    assert "守护进程已启动" in capsys.readouterr().out


def test_wechat_login_daemon_start_failure_reports_error(tmp_path, monkeypatch, capsys) -> None:
    """When daemon.start() fails, login must surface the error (not silent success)."""
    from extensions.im_gateway.server import DaemonPaths

    from clawcodex_ext.services.channels.wechat_ilink import WeChatIlinkChannelAdapter

    paths = DaemonPaths.for_state_dir(tmp_path)
    ch.add_channel(str(paths.state_dir / "channels.yaml"), ch.build_default_channel("wechat"))
    monkeypatch.setattr(ch, "_print_terminal_qr", lambda scan_data: None)

    async def _fake_qr_login(self, *, on_code=None, on_status=None, **_kwargs) -> dict:
        return {"status": "confirmed", "bot_token": "bot_tok_123", "account_id": "bot_account_1"}

    class _FakeDaemon:
        def __init__(self, *a, **kw):
            pass

        def start(self):
            return 1  # start failed

    monkeypatch.setattr(WeChatIlinkChannelAdapter, "qr_login", _fake_qr_login)
    monkeypatch.setattr(ch, "_daemon_alive", lambda daemon: False)
    monkeypatch.setattr("extensions.im_gateway.server.GatewayDaemon", _FakeDaemon)

    rc = ch.wechat_login("wechat", state_dir=str(tmp_path))
    assert rc == 1
    assert "启动失败" in capsys.readouterr().err


# -- Feishu wizard -------------------------------------------------------


def _feishu_ws_channel(name: str = "feishu") -> ChannelConfig:
    return ch.build_channel_from_inputs(
        "feishu",
        name,
        {
            "connection_mode": "websocket",
            "app_id": "cli_app",
            "app_secret": "orig_plaintext_secret_xyz",
            "domain": "feishu",
            "bot_open_id": "ou_bot",
            "enabled": "true",
        },
    )


def test_wizard_add_feishu_interrupts_when_lark_oapi_missing(tmp_path, monkeypatch, capsys) -> None:
    p = tmp_path / "channels.yaml"
    monkeypatch.setattr(ch, "_feishu_dependencies_available", lambda: False)
    inputs = iter(["1", ""])  # select feishu (not logged in → add → dep_check fails), then exit

    assert ch.run_wizard(str(p), input_fn=lambda _p: next(inputs)) == 0

    captured = capsys.readouterr()
    assert "lark-oapi not installed" in captured.err
    assert "pip install lark-oapi" in captured.err
    assert ch.list_channels(str(p)) == []


def test_wizard_add_feishu_websocket_defaults_to_scan_and_skips_webhook_fields(tmp_path, monkeypatch, capsys) -> None:
    p = tmp_path / "channels.yaml"
    monkeypatch.setattr(ch, "_feishu_dependencies_available", lambda: True)
    monkeypatch.setattr(
        ch,
        "_feishu_scan_login",
        lambda input_fn: {
            "connection_mode": "websocket",
            "app_id": "cli_app",
            "app_secret": "secret",
            "domain": "feishu",
            "allowed_user_open_id": "ou_scanner",
        },
    )
    inputs = iter(["1", "1", ""])  # select feishu (add), websocket mode (idx 0), ESC exit
    prompts: list[str] = []

    def _input(prompt: str) -> str:
        prompts.append(prompt)
        return next(inputs)

    assert ch.run_wizard(str(p), input_fn=_input) == 0

    listed = ch.list_channels(str(p))
    assert listed == [{"name": "feishu", "type": "feishu", "enabled": True}]
    from clawcodex_ext.services.im_gateway.config import load_config

    channel = load_config(str(p)).get_channel("feishu")
    assert channel.extra["connection_mode"] == "websocket"
    assert channel.extra["allowed_user_open_id"] == "ou_scanner"
    assert channel.webhook_url == ""
    assert "bot_open_id" not in channel.extra
    assert "websocket" not in channel.extra

    text = (capsys.readouterr().out + "\n".join(prompts)).lower()
    assert "webhook url" not in text
    assert "allowed user" not in text
    assert "ws reconnect" not in text
    assert "渠道名称" not in text
    assert "gateway 将自动重启" in text
    assert text.count("gateway 将自动重启") == 1


def test_wizard_add_feishu_webhook_mode_prompts_webhook_url(tmp_path, monkeypatch) -> None:
    p = tmp_path / "channels.yaml"
    monkeypatch.setattr(ch, "_feishu_dependencies_available", lambda: True)
    inputs = iter(
        [
            "1",  # select feishu (not logged in → add)
            "2",  # webhook mode (idx 1)
            "https://open.feishu.cn/open-apis/bot/v2/hook/abcdef",  # webhook_url
            "sign_secret",  # secret
            "",  # ESC exit wizard
        ]
    )

    assert ch.run_wizard(str(p), input_fn=lambda _p: next(inputs)) == 0

    from clawcodex_ext.services.im_gateway.config import load_config

    channel = load_config(str(p)).get_channel("feishu")
    assert channel.extra["connection_mode"] == "webhook"
    assert channel.webhook_url == "https://open.feishu.cn/open-apis/bot/v2/hook/abcdef"
    assert channel.extra["secret"] == "sign_secret"


def test_wizard_edit_feishu_toggle_enabled(tmp_path, monkeypatch) -> None:
    p = tmp_path / "channels.yaml"
    ch.add_channel(str(p), _feishu_ws_channel())
    monkeypatch.setattr(ch, "_feishu_dependencies_available", lambda: True)
    inputs = iter(
        [
            "1",  # select feishu (logged in → edit menu)
            "2",  # toggle enable/disable (idx 1)
            "",  # ESC feishu edit → back to channel select
            "",  # ESC exit wizard
        ]
    )

    assert ch.run_wizard(str(p), input_fn=lambda _p: next(inputs)) == 0

    listed = ch.list_channels(str(p))
    assert listed[0]["enabled"] is False


def test_wizard_add_feishu_websocket_keeps_wechat_enabled(tmp_path, monkeypatch) -> None:
    p = tmp_path / "channels.yaml"
    ch.add_channel(str(p), ch.build_default_channel("wechat"))
    monkeypatch.setattr(ch, "_feishu_dependencies_available", lambda: True)
    monkeypatch.setattr(
        ch,
        "_feishu_scan_login",
        lambda input_fn: {
            "connection_mode": "websocket",
            "app_id": "cli_app",
            "app_secret": "secret",
            "domain": "feishu",
        },
    )
    inputs = iter(["1", "1", ""])  # select feishu (add), websocket mode (idx 0), ESC exit

    assert ch.run_wizard(str(p), input_fn=lambda _prompt: next(inputs)) == 0

    from clawcodex_ext.services.im_gateway.config import load_config

    cfg = load_config(str(p))
    assert cfg.get_channel("wechat").enabled is True
    assert cfg.get_channel("feishu").enabled is True


def test_format_status_feishu_reports_runtime_health_and_approval_support(tmp_path, monkeypatch) -> None:
    p = tmp_path / "channels.yaml"
    ch.add_channel(str(p), _feishu_ws_channel())
    monkeypatch.setattr(
        ch,
        "_read_gateway_runtime_status",
        lambda _state_dir: {
            "gateway_running": True,
            "channel_health": [
                {
                    "channel_id": "feishu",
                    "account_status": "websocket:connected",
                    "extra": {
                        "connection_mode": "websocket",
                        "domain": "feishu",
                        "approval_cards": "supported",
                    },
                }
            ],
        },
    )

    out = ch.format_status(str(p), "feishu")

    assert "mode: websocket" in out
    assert "health: websocket:connected" in out
    assert "approval_cards" not in out


def test_feishu_scan_login_uses_real_qr_registration_without_manual_prompts(monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        ch,
        "_feishu_qr_register",
        lambda: {
            "app_id": "cli_app",
            "app_secret": "secret",
            "domain": "feishu",
            "open_id": "ou_scanner",
        },
    )

    def _unexpected_input(_prompt: str) -> str:
        raise AssertionError("successful scan login must not ask for manual credentials")

    result = ch._feishu_scan_login(_unexpected_input)

    captured = capsys.readouterr().out
    assert "占位" not in captured
    assert "clawcodex-dev gateway restart" not in captured
    assert "第一条消息" not in captured
    assert result == {
        "connection_mode": "websocket",
        "app_id": "cli_app",
        "app_secret": "secret",
        "domain": "feishu",
        "allowed_user_open_id": "ou_scanner",
    }


def test_feishu_scan_login_falls_back_to_manual_when_scan_fails(monkeypatch, capsys) -> None:
    monkeypatch.setattr(ch, "_feishu_qr_register", lambda: None)
    inputs = iter(["cli_app", "secret", "", "", "lark"])

    result = ch._feishu_scan_login(lambda _p: next(inputs))

    captured = capsys.readouterr().out
    assert "占位" not in captured
    assert "手动填写" in captured
    assert result == {
        "connection_mode": "websocket",
        "app_id": "cli_app",
        "app_secret": "secret",
        "domain": "lark",
    }


def test_wizard_edit_feishu_login_manual_masks_secret_and_keeps_values(tmp_path, monkeypatch, capsys) -> None:
    p = tmp_path / "channels.yaml"
    ch.add_channel(str(p), _feishu_ws_channel())
    monkeypatch.setattr(ch, "_feishu_dependencies_available", lambda: True)
    # In the input_fn path, an empty prompt response means ESC; re-enter values to keep them.
    inputs = iter(
        [
            "1",  # select feishu (logged in → edit menu)
            "1",  # reset (idx 0)
            "2",  # manual (idx 1)
            "cli_app",  # app_id: re-enter existing to "keep"
            "new_secret",  # app_secret: new value
            "enc_key",  # encrypt_key
            "ver_tok",  # verification_token
            "feishu",  # domain: re-enter existing to "keep"
            "ou_bot",  # bot_open_id: re-enter existing to "keep"
            "",  # ESC feishu edit → back to channel select
            "",  # ESC exit wizard
        ]
    )
    prompts: list[str] = []

    def _input(prompt: str) -> str:
        prompts.append(prompt)
        return next(inputs)

    assert ch.run_wizard(str(p), input_fn=_input) == 0

    from clawcodex_ext.services.im_gateway.config import load_config

    channel = load_config(str(p)).get_channel("feishu")
    assert channel.extra["app_id"] == "cli_app"
    assert channel.extra["app_secret"] == "new_secret"
    assert channel.extra["domain"] == "feishu"
    assert channel.extra["encrypt_key"] == "enc_key"
    assert channel.extra["verification_token"] == "ver_tok"

    text = capsys.readouterr().out + "\n".join(prompts)
    assert "已配置" in text
    # never echo the existing plaintext secret value
    assert "orig_plaintext_secret_xyz" not in text


def test_wizard_edit_feishu_scan_login_updates_allowed_user(tmp_path, monkeypatch, capsys) -> None:
    p = tmp_path / "channels.yaml"
    existing = _feishu_ws_channel()
    existing.extra["allowed_user_open_id"] = "ou_old_scanner"
    ch.add_channel(str(p), existing)
    sender_state = tmp_path / "feishu_last_senders.json"
    sender_state.write_text('{"feishu":"oc_old_chat"}', encoding="utf-8")
    monkeypatch.setattr(ch, "_feishu_dependencies_available", lambda: True)
    monkeypatch.setattr(
        ch,
        "_feishu_scan_login",
        lambda input_fn: {
            "connection_mode": "websocket",
            "app_id": "new_app",
            "app_secret": "new_secret",
            "domain": "lark",
            "allowed_user_open_id": "ou_new_scanner",
        },
    )
    inputs = iter(
        [
            "1",  # select feishu (logged in → edit menu)
            "1",  # reset (idx 0)
            "1",  # scan (idx 0)
            "",  # ESC feishu edit → back to channel select
            "",  # ESC exit wizard
        ]
    )

    assert ch.run_wizard(str(p), input_fn=lambda _prompt: next(inputs)) == 0

    from clawcodex_ext.services.im_gateway.config import load_config

    channel = load_config(str(p)).get_channel("feishu")
    assert channel.extra["app_id"] == "new_app"
    assert channel.extra["app_secret"] == "new_secret"
    assert channel.extra["domain"] == "lark"
    assert channel.extra["allowed_user_open_id"] == "ou_new_scanner"
    assert "bot_open_id" not in channel.extra
    assert not sender_state.exists()
    output = capsys.readouterr().out
    assert output.count("Gateway 将自动重启") == 1


# -- new arrow-key wizard flow (plan: gateway-setup-arrow-key-menu) ---------


def test_run_wizard_esc_at_channel_select_exits(tmp_path) -> None:
    """Verify that ESC at channel selection exits setup."""
    p = tmp_path / "channels.yaml"
    inputs = iter([""])  # ESC at channel select
    assert ch.run_wizard(str(p), input_fn=lambda _p: next(inputs)) == 0


def test_run_wizard_feishu_not_logged_in_runs_add_flow(tmp_path, monkeypatch) -> None:
    """Verify that selecting an unconfigured Feishu channel starts the add flow."""
    p = tmp_path / "channels.yaml"
    monkeypatch.setattr(ch, "_feishu_dependencies_available", lambda: True)
    monkeypatch.setattr(
        ch,
        "_feishu_scan_login",
        lambda input_fn: {
            "connection_mode": "websocket",
            "app_id": "cli_app",
            "app_secret": "secret",
            "domain": "feishu",
            "allowed_user_open_id": "ou_scanner",
        },
    )
    inputs = iter(
        [
            "1",  # select feishu (not configured -> add flow)
            "1",  # websocket mode (default)
            "",  # ESC to exit setup after add
        ]
    )

    assert ch.run_wizard(str(p), input_fn=lambda _p: next(inputs)) == 0

    listed = ch.list_channels(str(p))
    assert listed == [{"name": "feishu", "type": "feishu", "enabled": True}]


def test_run_wizard_wechat_not_logged_in_runs_add_flow(tmp_path, monkeypatch) -> None:
    """Verify that selecting an unconfigured WeChat channel starts the scan flow."""
    p = tmp_path / "channels.yaml"
    login_calls: list[str] = []

    def _fake_login(name: str, *, state_dir: str | None = None) -> int:
        login_calls.append(name)
        return 0

    monkeypatch.setattr(ch, "wechat_login", _fake_login)
    monkeypatch.setattr(ch, "wechat_login_status", lambda name, state_dir=None: "unconfigured")
    inputs = iter(
        [
            "2",  # select wechat (not configured -> add flow)
            "",  # ESC to exit edit menu after scan
            "",  # ESC to exit setup
        ]
    )

    assert ch.run_wizard(str(p), input_fn=lambda _p: next(inputs)) == 0

    assert login_calls == ["wechat"]
    assert ch.list_channels(str(p)) == [{"name": "wechat", "type": "wechat", "enabled": True}]


def test_run_wizard_feishu_logged_in_goes_to_edit_menu(tmp_path, monkeypatch, capsys) -> None:
    """Verify that a logged-in Feishu channel opens its edit menu."""
    p = tmp_path / "channels.yaml"
    ch.add_channel(str(p), _feishu_ws_channel())  # Configured with app_id and app_secret.
    monkeypatch.setattr(ch, "_feishu_dependencies_available", lambda: True)
    inputs = iter(
        [
            "1",  # select feishu (logged in -> edit menu)
            "",  # ESC back to channel select
            "",  # ESC exit setup
        ]
    )

    assert ch.run_wizard(str(p), input_fn=lambda _p: next(inputs)) == 0

    text = capsys.readouterr().out
    # The new menu uses reset instead of login.
    assert "重置" in text
    # Limit the negative assertion to the Feishu edit-menu section so the
    # channel-selection menu and its rerender after ESC cannot affect it.
    feishu_edit_section = ""
    if "编辑 feishu [feishu]" in text:
        after = text.split("编辑 feishu [feishu]", 1)[-1]
        # Stop before the channel-selection title is rendered again.
        feishu_edit_section = after.split("ClawCodex 消息渠道配置", 1)[0]
    assert "登录" not in feishu_edit_section


def test_run_wizard_wechat_logged_in_goes_to_edit_menu(tmp_path, monkeypatch, capsys) -> None:
    """Verify that a logged-in WeChat channel opens its edit menu."""
    p = tmp_path / "channels.yaml"
    ch.add_channel(str(p), ch.build_default_channel("wechat"))
    monkeypatch.setattr(ch, "wechat_login_status", lambda name, state_dir=None: "logged_in")
    inputs = iter(
        [
            "2",  # select wechat (logged in -> edit menu)
            "",  # ESC back to channel select
            "",  # ESC exit setup
        ]
    )

    assert ch.run_wizard(str(p), input_fn=lambda _p: next(inputs)) == 0

    text = capsys.readouterr().out
    assert "重置" in text


def test_run_wizard_esc_at_feishu_edit_returns_to_channel_select(tmp_path, monkeypatch) -> None:
    """Verify that ESC returns from Feishu editing and a second ESC exits setup."""
    p = tmp_path / "channels.yaml"
    ch.add_channel(str(p), _feishu_ws_channel())
    monkeypatch.setattr(ch, "_feishu_dependencies_available", lambda: True)
    inputs = iter(
        [
            "1",  # select feishu -> edit menu
            "",  # ESC at feishu edit -> back to channel select
            "",  # ESC at channel select -> exit
        ]
    )

    assert ch.run_wizard(str(p), input_fn=lambda _p: next(inputs)) == 0


def test_feishu_edit_remove_then_reenter_runs_add_flow(tmp_path, monkeypatch) -> None:
    """Verify that removing Feishu and selecting it again starts the add flow."""
    p = tmp_path / "channels.yaml"
    ch.add_channel(str(p), _feishu_ws_channel())
    monkeypatch.setattr(ch, "_feishu_dependencies_available", lambda: True)
    add_calls: list[int] = []
    orig_add = ch._wizard_add_feishu

    def _spy_add(cfg, path, ui):
        add_calls.append(1)
        return orig_add(cfg, path, ui)

    monkeypatch.setattr(ch, "_wizard_add_feishu", _spy_add)
    monkeypatch.setattr(
        ch,
        "_feishu_scan_login",
        lambda input_fn: {
            "connection_mode": "websocket",
            "app_id": "new_app",
            "app_secret": "new_secret",
            "domain": "feishu",
            "allowed_user_open_id": "ou_new_scanner",
        },
    )
    inputs = iter(
        [
            "1",  # select feishu (logged in -> edit menu)
            "3",  # remove channel
            "y",  # confirm remove
            "1",  # select feishu again (now removed -> add flow)
            "1",  # websocket mode
            "",  # ESC exit setup
        ]
    )

    assert ch.run_wizard(str(p), input_fn=lambda _p: next(inputs)) == 0

    assert add_calls == [1]  # The add flow ran once.
    listed = ch.list_channels(str(p))
    assert listed == [{"name": "feishu", "type": "feishu", "enabled": True}]


def test_existing_slack_channel_remains_editable_but_not_addable(tmp_path, monkeypatch) -> None:
    """Verify that an existing Slack channel remains editable but cannot be added again."""
    p = tmp_path / "channels.yaml"
    ch.add_channel(str(p), _slack("slack-ops", enabled=True))
    inputs = iter(
        [
            "3",  # Select the existing Slack item, which is third.
            "3",  # remove
            "y",  # confirm
            "",  # ESC exit setup
        ]
    )
    prompts: list[str] = []

    def _input(prompt: str) -> str:
        prompts.append(prompt)
        return next(inputs)

    assert ch.run_wizard(str(p), input_fn=_input) == 0

    assert ch.list_channels(str(p)) == []
    # The menu must not offer to add Slack again.
    text = "\n".join(prompts)
    assert "新增" not in text or "slack" not in text.lower().rsplit("新增", maxsplit=1)[-1]


def test_feishu_manual_login_esc_aborts_without_saving(tmp_path, monkeypatch) -> None:
    """Verify that ESC during manual Feishu login aborts without saving."""
    p = tmp_path / "channels.yaml"
    original = _feishu_ws_channel()
    ch.add_channel(str(p), original)
    monkeypatch.setattr(ch, "_feishu_dependencies_available", lambda: True)
    inputs = iter(
        [
            "1",  # select feishu -> edit menu
            "1",  # reset
            "2",  # manual
            "",  # app_id: ESC -> abort manual
            "",  # ESC feishu edit -> back to channel select
            "",  # ESC exit setup
        ]
    )

    assert ch.run_wizard(str(p), input_fn=lambda _p: next(inputs)) == 0

    from clawcodex_ext.services.im_gateway.config import load_config

    channel = load_config(str(p)).get_channel("feishu")
    # The channel is unchanged.
    assert channel.extra["app_id"] == "cli_app"
    assert channel.extra["app_secret"] == "orig_plaintext_secret_xyz"


def test_edit_fields_esc_aborts_without_saving(tmp_path, monkeypatch) -> None:
    """Verify that ESC while editing Slack fields leaves the channel unchanged."""
    p = tmp_path / "channels.yaml"
    ch.add_channel(str(p), _slack("slack-ops", enabled=True))
    inputs = iter(
        [
            "3",  # select slack
            "1",  # edit fields
            "",  # webhook_url: ESC -> abort
            "",  # ESC slack edit -> back to channel select
            "",  # ESC exit
        ]
    )

    assert ch.run_wizard(str(p), input_fn=lambda _p: next(inputs)) == 0

    from clawcodex_ext.services.im_gateway.config import load_config

    channel = load_config(str(p)).get_channel("slack-ops")
    assert channel.webhook_url == "https://hooks.example.com/services/T/B/abcdef0123456789"


def test_feishu_is_logged_in_detects_configured_credentials() -> None:
    """Return True when Feishu app_id and app_secret are configured."""
    channel = _feishu_ws_channel()
    assert ch._feishu_is_logged_in(channel) is True

    bare = ch.build_channel_from_inputs("feishu", "feishu", {"connection_mode": "websocket"})
    assert ch._feishu_is_logged_in(bare) is False


def test_wechat_is_logged_in_reflects_login_status(monkeypatch) -> None:
    """Delegate WeChat login detection to wechat_login_status."""
    monkeypatch.setattr(ch, "wechat_login_status", lambda name, state_dir=None: "logged_in (x)")
    assert ch._wechat_is_logged_in("wechat") is True

    monkeypatch.setattr(ch, "wechat_login_status", lambda name, state_dir=None: "unconfigured")
    assert ch._wechat_is_logged_in("wechat") is False
