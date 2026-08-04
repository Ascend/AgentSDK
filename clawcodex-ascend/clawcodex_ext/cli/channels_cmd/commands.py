#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# -------------------------------------------------------------------------
#  This file is part of the AgentSDK project.
# Copyright (c) 2026 Huawei Technologies Co.,Ltd.
# Copyright (c) 2026 Clawd Codex Team
#
# AgentSDK is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#
#           http://license.coscl.org.cn/MulanPSL2
#
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
# -------------------------------------------------------------------------

# pylint: disable=no-name-in-module,too-many-lines
"""IM channel configuration helpers invoked via ``clawcodex-dev gateway``.

These functions are called from ``gateway_cmd.commands.run_gateway_command``
using the flattened verb surface::

  clawcodex-dev gateway setup                  Guided wizard (interactive menu)
  clawcodex-dev gateway status [name]          Show channel health/status
  clawcodex-dev gateway disconnect <name>      Remove REPL/orchestrator connection
  clawcodex-dev gateway restart <name>         Rebuild a channel adapter
  clawcodex-dev gateway login <name>           WeChat iLink QR login

The wizard reads/writes ``~/.clawcodex/gateway/channels.yaml``
atomically (single-writer lock + tmp+replace). WeChat-specific QR
login / login-state operations are wired through the same gateway state.

Editable fields are driven by a per-type map in P1; P2/P5 refine this
to be descriptor-driven from ``ChannelCapabilityDescriptor``.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Callable

from clawcodex_ext.cli._interactive import InteractiveInput
from clawcodex_ext.services.channels.models import ChannelConfig, ChannelType
from clawcodex_ext.services.im_gateway.config import load_config, save_config

InputFn = Callable[[str], str]
logger = logging.getLogger(__name__)

_USAGE = (
    "usage: clawcodex-dev gateway {setup|status|disconnect|restart|login} [name]\n\n"
    "  setup            Guided configuration wizard.\n"
    "  status [name]    Show channel status (all if name omitted).\n"
    "  disconnect <name> Remove the active REPL/orchestrator connection.\n"
    "  restart <name>   Rebuild a channel adapter and reload its config.\n"
    "  login <name>     WeChat iLink QR login.\n"
)

# Per-type editable fields. Webhook types share webhook_url; WeChat stores
# platform fields in ChannelConfig.extra. Feishu has its own dedicated
# wizard branch (scan login + connection-mode split), so it carries no
# generic fields here — the empty entry only keeps it in the "available
# types" listing.
_FIELD_MAP: dict[str, list[tuple[str, str, Any]]] = {
    "feishu": [],
    "slack": [("webhook_url", "webhook URL", ""), ("enabled", "enabled (true/false)", True)],
    "discord": [("webhook_url", "webhook URL", ""), ("enabled", "enabled (true/false)", True)],
    "wechat": [
        ("base_url", "iLink base URL", "https://ilinkai.weixin.qq.com"),
        ("account_id", "account id", "default"),
        ("enabled", "enabled (true/false)", True),
    ],
}

_DEFAULT_CHANNEL_NAMES = {
    "discord": "discord-main",
    "feishu": "feishu",
    "slack": "slack-main",
    "wechat": "wechat",
}


# -- pure config ops (testable) -----------------------------------------


def list_channels(path: str | None = None) -> list[dict[str, Any]]:
    cfg = load_config(path)
    return [{"name": c.name, "type": c.type.value, "enabled": c.enabled} for c in cfg.channels]


def add_channel(path: str | None, channel: ChannelConfig) -> None:
    cfg = load_config(path)
    cfg.replace_channel(channel)
    save_config(cfg, path)


def update_channel(path: str | None, channel: ChannelConfig) -> bool:
    cfg = load_config(path)
    if cfg.get_channel(channel.name) is None and cfg.get_channel_by_type(channel.type) is None:
        return False
    cfg.replace_channel(channel)
    save_config(cfg, path)
    return True


def remove_channel(path: str | None, name: str) -> bool:
    cfg = load_config(path)
    return _remove_channel_config_and_state(cfg, path, name)


def _remove_channel_config_and_state(cfg, path: str | None, name: str) -> bool:
    channel = _resolve_channel_for_removal(cfg, name)
    if channel is None:
        return False
    if not cfg.remove_channel(channel.name):
        return False
    save_config(cfg, path)
    _cleanup_removed_channel_state(path, channel)
    return True


def _resolve_channel_for_removal(cfg, name: str) -> ChannelConfig | None:
    channel = cfg.get_channel(name)
    if channel is not None:
        return channel
    if name == "wechat-main":
        return cfg.get_channel("wechat")
    with contextlib.suppress(ValueError):
        return cfg.get_channel_by_type(ChannelType(name))
    return None


def _cleanup_removed_channel_state(path: str | None, channel: ChannelConfig) -> None:
    state_dir = _state_dir_for_config_path(path)
    if channel.type is ChannelType.WECHAT:
        _cleanup_wechat_state(state_dir, channel.name)
    elif channel.type is ChannelType.FEISHU:
        _cleanup_feishu_state(state_dir, channel.name)


def _state_dir_for_config_path(path: str | None) -> Path:
    if path is not None:
        return Path(path).expanduser().parent
    from extensions.im_gateway.server import DaemonPaths

    return DaemonPaths.for_state_dir(None).state_dir


def _cleanup_wechat_state(state_dir: Path, channel_name: str) -> None:
    wechat_dir = state_dir / "wechat"
    names = {channel_name}
    if channel_name == "wechat":
        names.add("wechat-main")
    elif channel_name == "wechat-main":
        names.add("wechat")

    for name in names:
        auth_path = wechat_dir / f"{name}_auth.json"
        pairing_path = wechat_dir / f"{name}_pairing.json"
        for candidate in (
            auth_path,
            auth_path.with_suffix(".key"),
            auth_path.with_suffix(".key.tmp"),
            auth_path.with_suffix(auth_path.suffix + ".tmp"),
            pairing_path,
            pairing_path.with_suffix(pairing_path.suffix + ".lock"),
            pairing_path.with_suffix(pairing_path.suffix + ".lock.tmp"),
            pairing_path.with_suffix(pairing_path.suffix + ".tmp"),
        ):
            _unlink_if_exists(candidate)
    _unlink_if_exists(state_dir / "wechat_context_tokens.json")
    _unlink_if_exists(state_dir / "wechat_accounts.json")
    with contextlib.suppress(OSError):
        wechat_dir.rmdir()


def _cleanup_feishu_state(state_dir: Path, channel_name: str) -> None:
    path = state_dir / "feishu_last_senders.json"
    if not path.exists():
        return
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("failed to read feishu sender state during removal: %s", exc)
        return
    if not isinstance(data, dict) or channel_name not in data:
        return
    data.pop(channel_name, None)
    if data:
        _atomic_write_json(path, data)
    else:
        _unlink_if_exists(path)


def _atomic_write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)


def _unlink_if_exists(path: Path) -> None:
    try:
        path.unlink()
    except FileNotFoundError:
        return
    except OSError as exc:
        logger.warning("failed to remove gateway state file %s: %s", path, exc)


def _coerce(value: str, default: Any) -> Any:
    if isinstance(default, bool):
        return value.strip().lower() in ("1", "true", "yes", "y", "on")
    if isinstance(default, int):
        try:
            return int(value)
        except ValueError:
            return default
    return value


def build_channel_from_inputs(ctype: str, name: str, inputs: dict[str, str]) -> ChannelConfig:
    """Build a ChannelConfig from wizard inputs for the given type."""
    if ctype == "feishu":
        return _build_feishu_channel_from_inputs(name, inputs)
    fields = _FIELD_MAP.get(ctype, [])
    enabled = True
    webhook_url = "https://placeholder.invalid/"
    extra: dict[str, Any] = {}
    for field_name, _label, default in fields:
        raw = inputs.get(field_name, "")
        value = _coerce(raw if raw != "" else str(default), default)
        if field_name == "webhook_url":
            webhook_url = str(value) if value else "https://placeholder.invalid/"
        elif field_name == "enabled":
            enabled = bool(value)
        else:
            extra[field_name] = value
    return ChannelConfig(
        type=ChannelType(ctype),
        webhook_url=webhook_url,
        name=name,
        enabled=enabled,
        extra=extra or None,
    )


def _build_feishu_channel_from_inputs(name: str, inputs: dict[str, str]) -> ChannelConfig:
    raw_mode = str(inputs.get("connection_mode") or "").strip().lower()
    webhook_url = str(inputs.get("webhook_url") or "").strip()
    mode = raw_mode or ("webhook" if webhook_url else "websocket")
    enabled = _coerce(inputs.get("enabled", "") or "true", True)
    if mode == "webhook":
        extra = {"connection_mode": "webhook"}
        secret = str(inputs.get("secret") or "").strip()
        if secret:
            extra["secret"] = secret
        return ChannelConfig(
            type=ChannelType.FEISHU,
            webhook_url=webhook_url or "https://placeholder.invalid/",
            name=name,
            enabled=bool(enabled),
            extra=extra,
        )
    extra: dict[str, Any] = {
        "connection_mode": "websocket",
        "app_id": str(inputs.get("app_id") or "").strip(),
        "app_secret": str(inputs.get("app_secret") or "").strip(),
        "domain": str(inputs.get("domain") or "feishu").strip().lower(),
        "batching": {
            "dedup_cache_size": 2048,
            "dedup_ttl_seconds": 86400,
            "text_batch_delay_seconds": 0.6,
            "text_batch_split_delay_seconds": 1.2,
            "text_batch_max_messages": 8,
            "text_batch_max_chars": 4000,
        },
        "send": {
            "sdk_send_attempts": 3,
            "sdk_send_backoff_base_seconds": 1.0,
        },
        "approval_cards": {
            "enabled": True,
            "action_token_ttl_seconds": 900,  # nosec B105
            "decision_ttl_seconds": 600,
        },
    }
    encrypt_key = str(inputs.get("encrypt_key") or "").strip()
    if encrypt_key:
        extra["encrypt_key"] = encrypt_key
    verification_token = str(inputs.get("verification_token") or "").strip()
    if verification_token:
        extra["verification_token"] = verification_token
    allowed_user_open_id = str(inputs.get("allowed_user_open_id") or "").strip()
    if allowed_user_open_id:
        extra["allowed_user_open_id"] = allowed_user_open_id
    bot_open_id = str(inputs.get("bot_open_id") or "").strip()
    if bot_open_id:
        extra["bot_open_id"] = bot_open_id
    bot_name = str(inputs.get("bot_name") or "").strip()
    if bot_name:
        extra["bot_name"] = bot_name
    return ChannelConfig(
        type=ChannelType.FEISHU,
        webhook_url="",
        name=name,
        enabled=bool(enabled),
        extra=extra,
    )


def build_default_channel(ctype: str) -> ChannelConfig:
    """Build the lowest-friction default config for a channel type."""
    name = _DEFAULT_CHANNEL_NAMES.get(ctype, f"{ctype}-main")
    if ctype == "wechat":
        return ChannelConfig(
            type=ChannelType.WECHAT,
            webhook_url="https://ilinkai.weixin.qq.com/dummy",
            name=name,
            enabled=True,
            extra={
                "base_url": "https://ilinkai.weixin.qq.com",
                "account_id": "default",
            },
        )
    return build_channel_from_inputs(ctype, name, {})


# -- status / restart ---------------------------------------------------


def format_status(path: str | None = None, name: str | None = None, *, state_dir: str | None = None) -> str:
    cfg = load_config(path)
    lines: list[str] = []
    channels = [c for c in cfg.channels if name is None or c.name == name]
    if not channels:
        return f"no channels configured{f' matching {name!r}' if name else ''}"
    resolved_state_dir = _resolve_status_state_dir(path, state_dir)
    runtime_status = _read_gateway_runtime_status(resolved_state_dir)
    clients_line = _format_connected_clients(runtime_status)
    for c in channels:
        dot = "●" if c.enabled else "○"
        lines.append(f"{dot} {c.name} [{c.type.value}] enabled={c.enabled}")
        if c.type is ChannelType.WECHAT:
            lines.append(f"  login: {wechat_login_status(c.name, state_dir=resolved_state_dir)}")
            lines.append(f"  {_format_wechat_conversation(runtime_status)}")
        elif c.type is ChannelType.FEISHU:
            lines.extend(_format_feishu_status(c, runtime_status))
        if clients_line:
            lines.append(f"  {clients_line}")
    return "\n".join(lines)


def _format_connected_clients(runtime_status: dict[str, Any]) -> str:
    """Summarize REPL/Orchestrator clients bound to the gateway.

    Draws from the runtime ``peers`` snapshot (registered opt-in clients). Each
    peer carries a ``host_type`` of ``repl`` / ``orchestrator`` / ``opt_in`` and
    an ``online`` flag. Returns an empty string when the daemon is down or no
    clients are registered so the channel status view stays uncluttered.
    """
    if runtime_status.get("gateway_error") or not runtime_status.get("gateway_running", True):
        return ""
    peers = runtime_status.get("peers") or []
    if not peers:
        return "connected clients: none"
    parts: list[str] = []
    for peer in peers:
        host_type = str(peer.get("host_type") or "opt_in")
        session_id = str(peer.get("session_id") or "")
        online = bool(peer.get("online"))
        state = "online" if online else "offline"
        pid = _peer_pid(peer)
        pid_part = f", pid={pid}" if pid is not None else ""
        parts.append(f"{host_type} (session={session_id}{pid_part}, {state})")
    return "connected clients: " + ", ".join(parts)


def _peer_pid(peer: dict[str, Any]) -> int | None:
    pid = peer.get("pid")
    try:
        parsed = int(pid)
    except (TypeError, ValueError):
        parsed = _peer_pid_from_session(str(peer.get("session_id") or ""))
    return parsed if parsed and parsed > 0 else None


def _peer_pid_from_session(session_id: str) -> int | None:
    parts = session_id.split("-")
    if len(parts) < 2 or parts[0] not in {"repl", "orchestrator"}:
        return None
    try:
        return int(parts[1])
    except ValueError:
        return None


def _format_blocking_gateway_peers(runtime_status: dict[str, Any]) -> list[str]:
    if runtime_status.get("gateway_error") or runtime_status.get("gateway_running") is False:
        return []
    formatted: list[str] = []
    for peer in runtime_status.get("peers") or []:
        if not isinstance(peer, dict):
            continue
        host_type = str(peer.get("host_type") or "opt_in")
        if host_type not in {"repl", "orchestrator"}:
            continue
        if peer.get("online") is False:
            continue
        session_id = str(peer.get("session_id") or "")
        pid = _peer_pid(peer)
        pid_part = f"pid={pid}, " if pid is not None else ""
        formatted.append(f"{host_type} ({pid_part}session={session_id})")
    return formatted


def _print_restart_peer_hint(runtime_status: dict[str, Any]) -> None:
    peers = _format_blocking_gateway_peers(runtime_status)
    if not peers:
        return
    print(
        "提示：当前通道仍连接到 REPL/Orchestrator；请先关闭对应的 "
        "REPL/Orchestrator 或执行 `/gateway disconnect`，然后再重试 restart。",
        file=sys.stderr,
    )
    print("已连接进程：", file=sys.stderr)
    for peer in peers:
        print(f"  - {peer}", file=sys.stderr)


def _resolve_status_state_dir(path: str | None, state_dir: str | None) -> str | None:
    if state_dir is not None:
        return state_dir
    if path is None:
        return None

    return str(Path(path).expanduser().parent)


def _read_gateway_runtime_status(state_dir: str | None = None) -> dict[str, Any]:
    import asyncio

    from extensions.im_gateway.server import DaemonPaths, GatewayDaemon

    paths = DaemonPaths.for_state_dir(state_dir)
    daemon = GatewayDaemon(paths)
    if not _daemon_alive(daemon):
        return {"gateway_running": False}

    async def _status() -> dict[str, Any]:
        from clawcodex_ext.services.im_gateway.ipc_client import GatewayIpcClient

        async with GatewayIpcClient(paths.sock_file) as client:
            return await client.status() or {}

    try:
        data = asyncio.run(_status())
    except (ConnectionError, FileNotFoundError, OSError, RuntimeError) as exc:
        return {"gateway_running": True, "gateway_error": str(exc)}
    data["gateway_running"] = True
    return data


def _format_wechat_conversation(runtime_status: dict[str, Any]) -> str:
    return _format_im_conversation(runtime_status, channel_type="wechat")


def _format_im_conversation(runtime_status: dict[str, Any], *, channel_type: str) -> str:
    if runtime_status.get("gateway_error"):
        return f"conversation: unknown (gateway error: {runtime_status['gateway_error']})"
    bindings = [
        b
        for b in runtime_status.get("bindings", [])
        if _binding_applies_to_channel(str(b.get("origin", "")), channel_type)
    ]
    if not bindings:
        return "conversation: disconnected"
    binding = bindings[0]
    session_id = str(binding.get("session_id") or "")
    peers = {str(p.get("session_id") or ""): p for p in runtime_status.get("peers", []) if p.get("session_id")}
    peer = peers.get(session_id, {})
    host_type = str(binding.get("host_type") or peer.get("host_type") or "unknown")
    connection_state = str(binding.get("connection_state") or "unknown")
    online = bool(peer.get("online", connection_state == "active"))
    if connection_state != "active" or not online:
        state = "offline" if not online else connection_state
        return f"conversation: disconnected (last {host_type}, session={session_id}, state={state})"
    state = "online" if online and connection_state == "active" else connection_state
    return f"conversation: connected to {host_type} (session={session_id}, state={state})"


def _binding_applies_to_channel(origin: str, channel_type: str) -> bool:
    if origin == "im:direct:*:*":
        return channel_type in {"wechat", "feishu"}
    if channel_type == "wechat":
        return origin.startswith("wechat:direct:")
    if channel_type == "feishu":
        return origin.startswith("feishu:dm:")
    return False


def _format_feishu_status(channel: ChannelConfig, runtime_status: dict[str, Any]) -> list[str]:
    extra = dict(channel.extra or {})
    mode = str(extra.get("connection_mode") or ("webhook" if channel.webhook_url else "websocket"))
    health = _find_runtime_channel_health(runtime_status, channel.name)
    health_status = "disabled"
    if channel.enabled:
        if health:
            health_status = str(
                health.get("account_status")
                or ("healthy" if health.get("healthy") else health.get("last_error") or "unhealthy")
            )
        elif runtime_status.get("gateway_error"):
            health_status = f"unknown (gateway error: {runtime_status['gateway_error']})"
        elif runtime_status.get("gateway_running") is False:
            health_status = "gateway_down"
        else:
            health_status = "unknown"
    return [
        f"  mode: {mode}",
        f"  health: {health_status}",
        f"  {_format_im_conversation(runtime_status, channel_type='feishu')}",
    ]


def _find_runtime_channel_health(runtime_status: dict[str, Any], channel_name: str) -> dict[str, Any] | None:
    for entry in runtime_status.get("channel_health", []):
        if isinstance(entry, dict) and entry.get("channel_id") == channel_name:
            return entry
    return None


def _disconnect_gateway_connection(name: str, *, state_dir: str | None = None) -> int:
    import asyncio

    from clawcodex_ext.services.im_gateway.ipc_client import GatewayIpcClient
    from clawcodex_ext.services.im_gateway.models import IM_DIRECT_ALL_ORIGIN
    from extensions.im_gateway.server import DaemonPaths, GatewayDaemon

    cfg = load_config(_wechat_config_path(state_dir=state_dir))
    channel = cfg.get_channel(name)
    if channel is None:
        print(f"error: no configured channel named {name!r}", file=sys.stderr)
        return 2
    if channel.type not in {ChannelType.WECHAT, ChannelType.FEISHU}:
        print(
            f"error: disconnect is only supported for IM app channels, got {name!r}",
            file=sys.stderr,
        )
        return 2
    paths = DaemonPaths.for_state_dir(state_dir)
    daemon = GatewayDaemon(paths)
    if not _daemon_alive(daemon):
        print(f"{name}: conversation already disconnected (gateway daemon not running).")
        return 0

    async def _unbind() -> int:
        try:
            async with GatewayIpcClient(paths.sock_file) as client:
                resp = await client.unbind_origin(IM_DIRECT_ALL_ORIGIN)
        except (ConnectionError, FileNotFoundError, OSError) as exc:
            print(f"error: could not reach gateway daemon: {exc}", file=sys.stderr)
            return 1
        if resp is not None and resp.ack_layer == "accepted":
            print(f"{name}: conversation connection removed.")
            return 0
        print(f"error: disconnect failed ({resp.reason if resp else 'no response'})", file=sys.stderr)
        return 1

    return asyncio.run(_unbind())


def restart_channel(name: str, *, state_dir: str | None = None) -> int:
    """Rebuild a channel adapter live via the running daemon (P4).

    If the daemon is running, send a ``control.reload`` IPC frame so the
    adapter is rebuilt in-process (即时生效). Otherwise validate config and
    advise starting the daemon.
    """
    import asyncio

    from extensions.im_gateway.server import DaemonPaths, GatewayDaemon

    paths = DaemonPaths.for_state_dir(state_dir)
    cfg = load_config(paths.state_dir / "channels.yaml" if state_dir else None)
    if cfg.get_channel(name) is None:
        print(f"error: no channel named {name!r} in config", file=sys.stderr)
        return 2
    daemon = GatewayDaemon(paths)
    if _daemon_alive(daemon):
        from clawcodex_ext.services.im_gateway.ipc_client import GatewayIpcClient

        async def _reload() -> int:
            runtime_status: dict[str, Any] = {}
            try:
                async with GatewayIpcClient(paths.sock_file) as client:
                    with contextlib.suppress(Exception):
                        runtime_status = await client.status() or {}
                    resp = await client.reload_channel(name)
            except (ConnectionError, FileNotFoundError, OSError) as exc:
                print(f"error: could not reach gateway daemon: {exc}", file=sys.stderr)
                _print_restart_peer_hint(runtime_status)
                return 1
            except Exception as exc:  # noqa: BLE001
                print(f"error: channel {name!r} restart failed: {exc}", file=sys.stderr)
                _print_restart_peer_hint(runtime_status)
                return 1
            if resp is not None and resp.ack_layer == "accepted":
                print(f"channel {name!r} reloaded live (gateway daemon PID {read_pid_value(paths)}).")
                return 0
            if resp is not None and isinstance(resp.payload, dict):
                runtime_status.update(resp.payload)
            print(
                f"channel {name!r} reload returned: {resp.ack_layer if resp else 'no response'} "
                f"({resp.reason if resp else ''})",
                file=sys.stderr,
            )
            _print_restart_peer_hint(runtime_status)
            return 1

        return asyncio.run(_reload())
    print(f"channel {name!r} config validated (gateway daemon not running).")
    print("Start the daemon with `clawcodex-dev gateway start` for live reload.")
    return 0


def read_pid_value(paths) -> int | None:
    from extensions.im_gateway.server import read_pid

    return read_pid(paths)


def _daemon_alive(daemon) -> bool:
    from extensions.im_gateway.server import is_pid_alive, read_pid

    pid = read_pid(daemon.paths)
    return pid is not None and is_pid_alive(pid)


# -- Feishu-specific ops ------------------------------------------------


_FEISHU_SCAN_URL = "https://open.feishu.cn/app"
_FEISHU_DEP_MISSING_MSG = (
    "lark-oapi not installed; install with `uv sync --locked --extra feishu` or `pip install lark-oapi`"
)


def _feishu_dependencies_available() -> bool:
    """Lazy check for the ``lark_oapi`` SDK required by the Feishu app adapter."""
    from clawcodex_ext.services.channels.feishu_sdk import feishu_dependencies_available

    return feishu_dependencies_available()


def _feishu_dep_check() -> bool:
    """Pre-flight dependency gate for the Feishu wizard.

    Prints the install hint and returns ``False`` when ``lark_oapi`` is not
    installed so the caller can abort setup without partial writes.
    """
    if _feishu_dependencies_available():
        return True
    print(f"error: {_FEISHU_DEP_MISSING_MSG}", file=sys.stderr)
    return False


def _feishu_qrcode_available() -> bool:
    import importlib

    return importlib.util.find_spec("qrcode") is not None


def _feishu_print_scan_qr() -> None:
    """Render the scan-to-create QR (or URL fallback when qrcode is missing)."""
    if _feishu_qrcode_available():
        _print_terminal_qr(_FEISHU_SCAN_URL)
        return
    print(f"扫码链接：{_FEISHU_SCAN_URL}")
    print("提示：完整二维码服务需要安装 qrcode（`pip install qrcode`）。")


def _feishu_qr_register() -> dict[str, str] | None:
    from clawcodex_ext.services.channels.feishu_onboarding import qr_register

    result = qr_register(initial_domain="feishu")
    return dict(result) if isinstance(result, dict) else None


def _feishu_scan_login(input_fn: InputFn) -> dict[str, str]:
    """Scan-to-create login entry, with manual credential fallback."""
    print("\n飞书扫码登录（推荐）")
    result = _feishu_qr_register()
    if result:
        print("飞书扫码登录成功。")
        payload = {
            "connection_mode": "websocket",
            "app_id": str(result.get("app_id") or ""),
            "app_secret": str(result.get("app_secret") or ""),
            "domain": str(result.get("domain") or "feishu"),
            # The registration SDK requests and returns the scanning user's
            # open_id. Persist it as the initial outbound/allowlist target so
            # REPL and Orchestrator can send before the first inbound message.
            "allowed_user_open_id": str(result.get("open_id") or ""),
        }
        encrypt_key = str(result.get("encrypt_key") or "")
        if encrypt_key:
            payload["encrypt_key"] = encrypt_key
        verification_token = str(result.get("verification_token") or "")
        if verification_token:
            payload["verification_token"] = verification_token
        return payload
    print("扫码未完成或注册失败，将改为手动填写应用凭证。")
    app_id = input_fn("Feishu app_id: ").strip()
    app_secret = input_fn("Feishu app_secret: ").strip()
    encrypt_key = input_fn("Feishu encrypt_key (可选): ").strip()
    verification_token = input_fn("Feishu verification_token (可选): ").strip()
    domain = input_fn("domain (feishu/lark) [feishu]: ").strip() or "feishu"
    payload = {
        "connection_mode": "websocket",
        "app_id": app_id,
        "app_secret": app_secret,
        "domain": domain,
    }
    if encrypt_key:
        payload["encrypt_key"] = encrypt_key
    if verification_token:
        payload["verification_token"] = verification_token
    return payload


def _feishu_manual_login(channel: ChannelConfig, ui: InteractiveInput) -> ChannelConfig | None:
    """Manual edit of login fields; masks app_secret, keeps existing values.

    Returns ``None`` when the user ESCs out mid-way (channel未修改)。
    """
    extra = dict(channel.extra or {})
    current_app_id = str(extra.get("app_id") or "")
    current_domain = str(extra.get("domain") or "feishu")
    current_bot = str(extra.get("bot_open_id") or "")
    secret_state = "已配置" if extra.get("app_secret") else "未配置"
    encrypt_key_state = "已配置" if extra.get("encrypt_key") else "未配置"
    verification_token_state = "已配置" if extra.get("verification_token") else "未配置"

    app_id = ui.prompt(f"新 app_id [{current_app_id}] (回车保留，ESC 中断): ")
    if app_id is None:
        return None
    app_secret = ui.prompt(f"新 app_secret [{secret_state}] (回车保留，输入新值覆盖，ESC 中断): ")
    if app_secret is None:
        return None
    if app_id:
        extra["app_id"] = app_id
    if app_secret:
        extra["app_secret"] = app_secret
    encrypt_key = ui.prompt(f"新 encrypt_key [{encrypt_key_state}] (可选，ESC 中断): ")
    if encrypt_key is None:
        return None
    if encrypt_key:
        extra["encrypt_key"] = encrypt_key
    verification_token = ui.prompt(f"新 verification_token [{verification_token_state}] (可选，ESC 中断): ")
    if verification_token is None:
        return None
    if verification_token:
        extra["verification_token"] = verification_token
    domain = ui.prompt(f"新 domain [{current_domain}] (feishu/lark，ESC 中断): ")
    if domain is None:
        return None
    if domain:
        extra["domain"] = domain.lower()
    bot_open_id = ui.prompt(f"新 bot_open_id [{current_bot}] (可选，ESC 中断): ")
    if bot_open_id is None:
        return None
    if bot_open_id:
        extra["bot_open_id"] = bot_open_id
    return ChannelConfig(
        type=channel.type,
        webhook_url=channel.webhook_url,
        name=channel.name,
        enabled=channel.enabled,
        extra=extra or None,
    )


def _wizard_add_feishu(cfg, path, ui: InteractiveInput) -> None:
    """First-time Feishu channel creation: dep check → connection mode → scan login."""
    if not _feishu_dep_check():
        return
    name = _DEFAULT_CHANNEL_NAMES["feishu"]
    mode_idx = ui.select(
        [("websocket（推荐）", ""), ("webhook（兼容旧链路）", "")],
        title="连接方式",
    )
    if mode_idx is None:
        return
    if mode_idx == 1:
        webhook_url = ui.prompt("legacy webhook URL: ")
        if webhook_url is None:
            return
        secret = ui.prompt("legacy webhook secret (可选): ")
        if secret is None:
            secret = ""  # nosec B105
        inputs: dict[str, str] = {
            "connection_mode": "webhook",
            "webhook_url": webhook_url,
            "secret": secret,
            "enabled": "true",
        }
    else:
        inputs = _feishu_scan_login(ui.input_fn)
        inputs["enabled"] = "true"

    channel = build_channel_from_inputs("feishu", name, inputs)
    cfg.replace_channel(channel)
    save_config(cfg, path)
    print(f"已保存渠道 {name!r}。退出 setup 后 Gateway 将自动重启生效。")


def _wizard_edit_feishu(cfg, path, channel: ChannelConfig, ui: InteractiveInput) -> None:
    """Edit menu for an existing Feishu channel."""
    while True:
        login_label = "重置" if _feishu_is_logged_in(channel) else "登录"
        idx = ui.select(
            [
                (login_label, ""),
                (f"启用/停用 (当前: {'enabled' if channel.enabled else 'disabled'})", ""),
                ("移除该渠道", ""),
            ],
            title=f"编辑 {channel.name} [feishu]",
        )
        if idx is None:
            return
        if idx == 0:
            sub = ui.select(
                [("扫码登录", ""), ("手动填写", "")],
                title=f"{login_label} 方式",
            )
            if sub is None:
                continue
            if sub == 0:
                if not _feishu_dep_check():
                    continue
                login_inputs = _feishu_scan_login(ui.input_fn)
                scanned_channel = build_channel_from_inputs(
                    "feishu",
                    channel.name,
                    {**login_inputs, "enabled": "true" if channel.enabled else "false"},
                )
                extra = dict(scanned_channel.extra or {})
                previous_extra = dict(channel.extra or {})
                # Re-login replaces credentials but preserves non-login tuning
                # (batching / send / approval_cards) the user may have customized.
                for key in ("batching", "send", "approval_cards", "reactions"):
                    if key in previous_extra:
                        extra[key] = previous_extra[key]
                if previous_extra.get("allowed_user_open_id") and not extra.get("allowed_user_open_id"):
                    extra["allowed_user_open_id"] = previous_extra["allowed_user_open_id"]
                channel = ChannelConfig(
                    type=channel.type,
                    webhook_url=scanned_channel.webhook_url,
                    name=channel.name,
                    enabled=channel.enabled,
                    extra=extra or None,
                )
                cfg.replace_channel(channel)
                save_config(cfg, path)
                # A re-scan may switch the app or operator. Do not let a
                # persisted chat_id from the previous login override the new
                # scanner open_id in ``last_known_sender``.
                _cleanup_feishu_state(_state_dir_for_config_path(path), channel.name)
                print("登录配置已更新。退出 setup 后 Gateway 将自动重启生效。")
            elif sub == 1:
                updated = _feishu_manual_login(channel, ui)
                if updated is None:  # ESC 中断：保留原 channel，回 feishu 编辑菜单
                    continue
                channel = updated
                cfg.replace_channel(channel)
                save_config(cfg, path)
                print("登录配置已保存。")
        elif idx == 1:
            channel = ChannelConfig(
                type=channel.type,
                webhook_url=channel.webhook_url,
                name=channel.name,
                enabled=not channel.enabled,
                extra=channel.extra,
            )
            cfg.replace_channel(channel)
            save_config(cfg, path)
            print(f"{channel.name} → {'enabled' if channel.enabled else 'disabled'}")
        elif idx == 2:
            confirm = ui.confirm(f"确认移除 {channel.name}?")
            if confirm is None:
                continue
            if confirm:
                _remove_channel_config_and_state(cfg, path, channel.name)
                print(f"已移除 {channel.name}")
                return


# -- WeChat-specific ops (P2) ------------------------------------------


def _wechat_paths(name: str, *, state_dir: str | None = None):
    from extensions.im_gateway.server import DaemonPaths

    base = DaemonPaths.for_state_dir(state_dir).state_dir
    wechat_dir = Path(base) / "wechat"
    wechat_dir.mkdir(parents=True, exist_ok=True)
    auth_path = wechat_dir / f"{name}_auth.json"
    if name == "wechat" and not auth_path.exists():
        legacy_auth = wechat_dir / "wechat-main_auth.json"
        if legacy_auth.exists():
            auth_path = legacy_auth
    return auth_path, wechat_dir / f"{name}_pairing.json"


def _wechat_config_path(*, state_dir: str | None = None):
    from extensions.im_gateway.server import DaemonPaths

    return DaemonPaths.for_state_dir(state_dir).state_dir / "channels.yaml" if state_dir else None


def _require_wechat_channel(name: str, *, state_dir: str | None = None) -> ChannelConfig:
    cfg = load_config(_wechat_config_path(state_dir=state_dir))
    channel = cfg.get_channel(name)
    if channel is None or channel.type is not ChannelType.WECHAT:
        raise ValueError(f"no configured wechat channel named {name!r}")
    return channel


def wechat_login_status(name: str, *, state_dir: str | None = None) -> str:
    from clawcodex_ext.services.channels.wechat_ilink import WeChatIlinkAuthStore

    _require_wechat_channel(name, state_dir=state_dir)
    auth_path, _ = _wechat_paths(name, state_dir=state_dir)
    record = WeChatIlinkAuthStore(auth_path).load()
    if record is None:
        return "unconfigured (not logged in; run 扫码登录)"
    return f"logged_in (account_id={record.account_id}, user_id={record.user_id})"


def _feishu_is_logged_in(channel: ChannelConfig) -> bool:
    """Feishu 视为已登录当 app_id + app_secret 齐全（兼容纯手动配置，bot_open_id 可缺）。"""
    extra = dict(channel.extra or {})
    return bool(extra.get("app_id") and extra.get("app_secret"))


def _wechat_is_logged_in(name: str, *, state_dir: str | None = None) -> bool:
    return "logged_in" in wechat_login_status(name, state_dir=state_dir)


def _print_terminal_qr(scan_data: str) -> None:
    """Print a terminal QR when the optional qrcode package is available."""
    import importlib

    try:
        qrcode = importlib.import_module("qrcode")
    except ImportError:
        print("（当前环境未安装 qrcode，已改为显示扫码链接。）")
        return

    try:
        qr = qrcode.QRCode(border=1)
        qr.add_data(scan_data)
        qr.make(fit=True)
        qr.print_ascii(invert=True)
    except Exception as exc:  # noqa: BLE001
        print(f"（终端二维码渲染失败: {exc}，请直接打开上面的二维码链接。）")


def wechat_login(name: str, *, state_dir: str | None = None) -> int:
    """Perform iLink QR login; persists encrypted bot_token. Async/real endpoint."""
    import asyncio

    from clawcodex_ext.services.channels.transport import UrllibChannelTransport
    from clawcodex_ext.services.channels.wechat_ilink import (
        WeChatIlinkAuthStore,
        WeChatIlinkChannelAdapter,
        _IlinkHttpError,
        _IlinkPlatformError,
    )
    from clawcodex_ext.services.channels.exceptions import TransportError
    from extensions.im_gateway.server import DaemonPaths

    try:
        channel_cfg = _require_wechat_channel(name, state_dir=state_dir)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    extra = channel_cfg.extra or {}
    auth_path, _ = _wechat_paths(name, state_dir=state_dir)
    adapter = WeChatIlinkChannelAdapter(
        channel_cfg,
        auth_store=WeChatIlinkAuthStore(auth_path),
        store=None,  # type: ignore[arg-type]  # not needed for login
        transport=UrllibChannelTransport(),
        account_id=extra.get("account_id", "default"),
        base_url=extra.get("base_url", "https://ilinkai.weixin.qq.com"),
    )

    last_status: str | None = None

    def _show_code(scan_data: str) -> None:
        print("\n请使用微信扫描以下二维码/链接：")
        print(scan_data)
        _print_terminal_qr(scan_data)
        print("等待扫码确认", end="", flush=True)

    def _show_status(status: str) -> None:
        nonlocal last_status
        if status == "wait":
            print(".", end="", flush=True)
        elif status != last_status:
            if status == "scaned":
                print("\n已扫码，请在微信里确认。")
            elif status == "scaned_but_redirect":
                print("\n已扫码，正在切换 iLink 确认节点。")
            elif status == "expired":
                print("\n二维码已过期，正在刷新。")
            else:
                print(f"\n扫码状态: {status}")
        last_status = status

    async def _do() -> dict:
        return await adapter.qr_login(on_code=_show_code, on_status=_show_status)

    try:
        data = asyncio.run(_do())
    except _IlinkHttpError as exc:
        print(
            f"iLink 登录失败: HTTP {exc.status} (base_url={extra.get('base_url', 'https://ilinkai.weixin.qq.com')})",
            file=sys.stderr,
        )
        if exc.status == 404:
            print(
                "提示: 当前 iLink 地址未提供 QR 登录接口；请确认 base_url 可访问 "
                "ilink/bot/get_bot_qrcode 与 ilink/bot/get_qrcode_status。",
                file=sys.stderr,
            )
        return 1
    except _IlinkPlatformError as exc:
        print(f"iLink 登录失败: 平台错误 {exc.code}: {exc.msg}", file=sys.stderr)
        return 1
    except TransportError as exc:
        print(f"iLink 登录失败: 无法连接 iLink 服务: {exc}", file=sys.stderr)
        return 1
    if data.get("bot_token"):
        print(f"\nWeChat 登录成功 (account_id={data.get('account_id', 'default')})。")
        # The bot only receives WeChat messages while the gateway daemon is
        # running its getupdates long-poll. restart_channel reloads a running
        # daemon, but if the daemon is DOWN it just prints a hint and returns
        # success — leaving the user "logged in" but with no live connection
        # (the exact "clawcodex OK, WeChat side silent" symptom). Auto-start
        # the daemon here so the poll loop actually begins.
        from extensions.im_gateway.server import GatewayDaemon

        paths = DaemonPaths.for_state_dir(state_dir)
        daemon = GatewayDaemon(paths)
        if _daemon_alive(daemon):
            return restart_channel(name, state_dir=state_dir)
        print("Gateway 守护进程未运行，正在启动以建立微信消息连接……")
        start_rc = daemon.start()
        if start_rc != 0:
            print(
                "警告: Gateway 守护进程启动失败；微信消息将无法接收。\n"
                "  请手动执行 `clawcodex-dev gateway start` 后重试 `gateway restart "
                f"{name}`。",
                file=sys.stderr,
            )
            return start_rc
        print("Gateway 守护进程已启动，微信消息轮询已就绪。\n  在微信里向 bot 发消息即可触发对话。")
        return 0
    if data.get("status") == "timeout":
        print("\n微信登录超时，请重新执行扫码登录。", file=sys.stderr)
        return 1
    if data.get("status") == "expired":
        print("\n二维码多次过期，请重新执行扫码登录。", file=sys.stderr)
        return 1
    if data.get("code_url"):
        print(f"\n登录未完成: {data}", file=sys.stderr)
        return 1
    print(f"登录未完成: {data}")
    return 1


# -- wizard -------------------------------------------------------------


def run_wizard(path: str | None = None, *, input_fn: InputFn | None = None) -> int:
    """箭头键菜单驱动的 setup 向导。

    频道选择菜单列出 feishu / wechat 为主选项（带 [已登录]/[未登录] 标注），
    已配置的 slack/discord 作为"存量"项追加可编辑/移除。选择 feishu/wechat 时
    按登录态路由到新增或编辑流程。频道选择层 ESC 退出 setup；所有子菜单 ESC 返回上一步。
    """
    ui = InteractiveInput(input_fn)
    cfg = load_config(path)
    while True:
        options: list[tuple[str, str]] = []
        feishu_ch = cfg.get_channel_by_type("feishu")
        feishu_desc = "已登录" if (feishu_ch and _feishu_is_logged_in(feishu_ch)) else "未登录"
        options.append(("feishu", f"[{feishu_desc}]"))
        wechat_ch = cfg.get_channel_by_type("wechat")
        wechat_desc = (
            "已登录"
            if (wechat_ch and _wechat_is_logged_in(wechat_ch.name, state_dir=_resolve_status_state_dir(path, None)))
            else "未登录"
        )
        options.append(("wechat", f"[{wechat_desc}]"))
        legacy = [c for c in cfg.channels if c.type.value in ("slack", "discord")]
        for c in legacy:
            dot = "●" if c.enabled else "○"
            options.append((f"{c.name} [{c.type.value}] [存量] {dot}", ""))
        idx = ui.select(
            options,
            title="ClawCodex 消息渠道配置（↑↓ 选择 · Enter 确认 · ESC 退出）",
        )
        if idx is None:
            return 0
        if idx == 0:  # feishu
            if feishu_ch and _feishu_is_logged_in(feishu_ch):
                _wizard_edit_feishu(cfg, path, feishu_ch, ui)
            else:
                _wizard_add_feishu(cfg, path, ui)
            cfg = load_config(path)
        elif idx == 1:  # wechat
            if wechat_ch and _wechat_is_logged_in(wechat_ch.name, state_dir=_resolve_status_state_dir(path, None)):
                _wizard_edit_wechat(cfg, path, wechat_ch, ui)
            else:
                _wizard_add_wechat(cfg, path, ui)
            cfg = load_config(path)
        else:  # legacy slack/discord
            legacy_idx = idx - 2
            if 0 <= legacy_idx < len(legacy):
                _wizard_edit(cfg, path, legacy[legacy_idx], ui)
                cfg = load_config(path)
    return 0


def _wizard_add_wechat(cfg, path, ui: InteractiveInput) -> None:
    channel = build_default_channel("wechat")
    cfg.replace_channel(channel)
    save_config(cfg, path)
    print(f"已创建 WeChat 渠道 {channel.name!r}。")
    print("接下来进行微信扫码登录；扫码完成后可继续配置授权/启停等选项。")
    wechat_login(channel.name)
    refreshed = cfg.get_channel(channel.name) or channel
    _wizard_edit(cfg, path, refreshed, ui)


def _wizard_edit(cfg, path, channel: ChannelConfig, ui: InteractiveInput) -> None:
    if channel.type.value == "feishu":
        _wizard_edit_feishu(cfg, path, channel, ui)
        return
    if channel.type.value == "wechat":
        _wizard_edit_wechat(cfg, path, channel, ui)
        return
    while True:
        idx = ui.select(
            [
                ("编辑字段", ""),
                (f"启用/停用 (当前: {'enabled' if channel.enabled else 'disabled'})", ""),
                ("移除该渠道", ""),
            ],
            title=f"编辑 {channel.name} [{channel.type.value}]",
        )
        if idx is None:
            return
        if idx == 0:
            _edit_fields(cfg, path, channel, ui)
        elif idx == 1:
            channel = ChannelConfig(
                type=channel.type,
                webhook_url=channel.webhook_url,
                name=channel.name,
                enabled=not channel.enabled,
                extra=channel.extra,
            )
            cfg.replace_channel(channel)
            save_config(cfg, path)
            print(f"{channel.name} → {'enabled' if channel.enabled else 'disabled'}")
        elif idx == 2:
            confirm = ui.confirm(f"确认移除 {channel.name}?")
            if confirm is None:
                continue
            if confirm:
                _remove_channel_config_and_state(cfg, path, channel.name)
                print(f"已移除 {channel.name}")
                return


def _wizard_edit_wechat(cfg, path, channel: ChannelConfig, ui: InteractiveInput) -> None:
    while True:
        login_label = (
            "重置"
            if _wechat_is_logged_in(channel.name, state_dir=_resolve_status_state_dir(path, None))
            else "扫码登录"
        )
        idx = ui.select(
            [
                (login_label, ""),
                ("查看登录态 / conversation 连接", ""),
                ("移除 REPL/orchestrator 连接", ""),
                (f"启用/停用 (当前: {'enabled' if channel.enabled else 'disabled'})", ""),
                ("移除该渠道", ""),
            ],
            title=f"编辑 {channel.name} [wechat]",
        )
        if idx is None:
            return
        if idx == 0:
            wechat_login(channel.name)
        elif idx == 1:
            print(format_status(path, channel.name))
        elif idx == 2:
            _disconnect_gateway_connection(channel.name, state_dir=_resolve_status_state_dir(path, None))
        elif idx == 3:
            channel = ChannelConfig(
                type=channel.type,
                webhook_url=channel.webhook_url,
                name=channel.name,
                enabled=not channel.enabled,
                extra=channel.extra,
            )
            cfg.replace_channel(channel)
            save_config(cfg, path)
            print(f"{channel.name} → {'enabled' if channel.enabled else 'disabled'}")
        elif idx == 4:
            confirm = ui.confirm(f"确认移除 {channel.name}?")
            if confirm is None:
                continue
            if confirm:
                _remove_channel_config_and_state(cfg, path, channel.name)
                print(f"已移除 {channel.name}")
                return


def _edit_fields(cfg, path, channel: ChannelConfig, ui: InteractiveInput) -> None:
    fields = _FIELD_MAP.get(channel.type.value, [])
    extra = dict(channel.extra or {})
    webhook_url = channel.webhook_url
    enabled = channel.enabled
    for field_name, label, _default in fields:
        current = (
            extra.get(field_name)
            if field_name in extra
            else (
                webhook_url
                if field_name == "webhook_url"
                else ("true" if field_name == "enabled" and enabled else "false")
            )
        )
        raw = ui.prompt(f"{label} [{current}] (回车保留，ESC 中断): ")
        if raw is None:
            return  # ESC: 中断，已改字段不保存
        if raw == "":
            continue
        if field_name == "webhook_url":
            webhook_url = raw
        elif field_name == "enabled":
            enabled = raw.lower() in ("1", "true", "yes", "y", "on")
        else:
            extra[field_name] = raw
    updated = ChannelConfig(
        type=channel.type,
        webhook_url=webhook_url,
        name=channel.name,
        enabled=enabled,
        extra=extra or None,
    )
    cfg.replace_channel(updated)
    save_config(cfg, path)
    print("字段已保存。")


__all__ = [
    "add_channel",
    "build_channel_from_inputs",
    "format_status",
    "list_channels",
    "remove_channel",
    "restart_channel",
    "run_wizard",
    "update_channel",
    "wechat_login",
    "wechat_login_status",
]
