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

"""MessageGateway — the unified IM entry point.

Facade over the inbound dispatcher, outbound dispatcher, session router,
binding policy, capability gate, and reliability store. External callers
(Community Radar, Orchestrator event sink, REPL wrapper) use this API;
it does not reverse-import Orchestrator.

P1 ships the skeleton with working outbound send/broadcast, capability
fail-closed, session routing/binding, and a ``reload_channel`` hook for
``clawcodex-dev channels restart``. Inbound adapter lifecycle (WeChat)
lands in P2; full reliability hardening in P4; six-semantics in P5.
"""

from __future__ import annotations

# pylint: disable=no-name-in-module  # Split migration branches provide these modules.

import asyncio
import logging
from dataclasses import replace

from clawcodex_ext.services.channels.capabilities import ChannelAdapter, ChannelCapability
from clawcodex_ext.services.channels.registry import (
    ChannelAdapterRegistry,
    build_default_registry,
)
from clawcodex_ext.services.channels.results import SendStatus

from .binding import BindingEntry, BindingPolicy
from .capability_gate import CapabilityGate
from .config import GatewayConfig
from .dispatcher import InboundDispatcher, InboundHandler
from .models import AckReceipt, InboundMessage, OutboundMessage
from .outbound import OutboundDispatcher
from .processing_status import ProcessingStatusManager
from .router import SessionRouter
from .store import ReliabilityStore

logger = logging.getLogger(__name__)


class MessageGateway:
    def __init__(
        self,
        config: GatewayConfig | None = None,
        *,
        registry: ChannelAdapterRegistry | None = None,
        store: ReliabilityStore | None = None,
    ) -> None:
        self.config = _normalize_config_channels(config or GatewayConfig())
        self.registry = registry or build_default_registry()
        self.store = store or ReliabilityStore(self.config.state_dir, self.config.reliability)
        self.binding = BindingPolicy(auditor=self._audit_binding)
        self.router = SessionRouter(self.binding)
        self.gate = CapabilityGate(self.registry)
        self.outbound = OutboundDispatcher(self.registry, self.gate, self.store, self.config)
        self.processing_status = ProcessingStatusManager(self.registry)
        self.inbound = InboundDispatcher(
            self.store,
            self.router,
            command_allowlists=self.config.command_allowlists,
            processing_status=self.processing_status,
            push_timeout_seconds=self.config.push_timeout_seconds,
        )
        self._inbound_adapters: list = []
        self._running = False
        self._stop_lock = asyncio.Lock()
        self._load_channels()

    def _build_adapter(self, cfg) -> ChannelAdapter | None:
        """Build one adapter from a ChannelConfig (gateway-owned deps for WeChat)."""
        from pathlib import Path

        from clawcodex_ext.services.channels.feishu_app import FeishuAppChannelAdapter
        from clawcodex_ext.services.channels.models import ChannelType
        from clawcodex_ext.services.channels.transport import UrllibChannelTransport
        from clawcodex_ext.services.channels.wechat_ilink import (
            WeChatIlinkAuthStore,
            WeChatIlinkChannelAdapter,
        )

        if cfg.type is ChannelType.WECHAT:
            extra = cfg.extra or {}
            state_dir = Path(self.config.state_dir).expanduser()
            auth_path = _wechat_state_file(state_dir, cfg.name, "auth")
            adapter = WeChatIlinkChannelAdapter(
                cfg,
                auth_store=WeChatIlinkAuthStore(
                    auth_path,
                    secret_env=self.config.reliability.secret_encryption_env,
                ),
                store=self.store,
                transport=UrllibChannelTransport(),
                allowed_users=extra.get("allowed_users"),
                account_id=extra.get("account_id", "default"),
                base_url=extra.get("base_url", "https://ilinkai.weixin.qq.com"),
                long_poll_timeout_ms=extra.get("long_poll_timeout_ms", 35000),
                max_consecutive_failures=extra.get("max_consecutive_failures", 10),
            )
            adapter.load_credentials()
            return adapter
        if cfg.type is ChannelType.FEISHU:
            extra = cfg.extra or {}
            mode = str(extra.get("connection_mode") or "").strip().lower()
            if mode == "websocket":
                return FeishuAppChannelAdapter(cfg, sender_store=self.store)
        try:
            return self.registry.create(cfg)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "gateway: failed to build adapter for channel %s (%s): %s",
                cfg.name,
                cfg.type.value,
                exc,
            )
            return None

    def _attach_inbound(self, adapter) -> None:
        if adapter is not None and adapter.capabilities.has(ChannelCapability.INBOUND_POLLING):
            adapter.set_inbound_handler(self._on_inbound)
            self._inbound_adapters.append(adapter)

    def _load_channels(self) -> None:
        """Build adapters from config; WeChat needs gateway-owned deps."""
        for cfg in self.config.channels:
            if not cfg.enabled:
                continue
            adapter = self._build_adapter(cfg)
            if adapter is not None:
                self.registry.register(adapter)
                self._attach_inbound(adapter)
                logger.info("gateway loaded channel: %s (%s)", cfg.name, cfg.type.value)
            else:
                logger.warning("gateway failed to build adapter for channel: %s", cfg.name)

    async def _on_inbound(self, message) -> AckReceipt:
        """Inbound hook called by channel adapters (WeChat poller)."""
        ack = await self.inbound.process(message)
        await self._notify_sender_for_ack(message, ack)
        return ack

    async def _notify_sender_for_ack(self, message: InboundMessage, ack: AckReceipt) -> None:
        if not ack.notify_user or not ack.message:
            return
        channel = message.channel
        target = message.from_user_id or message.context_token
        if not channel or not target:
            logger.warning(
                "gateway ack notify skipped: channel=%r target=%r origin=%s",
                channel,
                target,
                message.origin[:32],
            )
            return
        try:
            result = await self.outbound.send(
                OutboundMessage(
                    text=ack.message,
                    channel=channel,
                    target=target,
                    context_token=message.context_token,
                    markdown=False,
                    idempotency_key=f"inbound-ack:{ack.delivery_id}",
                    semantic_tags=["gateway_notice"],
                    metadata={"source": "inbound_ack", "ack_layer": ack.layer.value},
                )
            )
        except Exception:  # noqa: BLE001
            logger.exception("gateway ack notify failed origin=%s", message.origin[:32])
            return
        if result is not None and getattr(result, "ok", True) is False:
            logger.warning(
                "gateway ack notify send rejected channel=%s origin=%s message=%s",
                channel,
                message.origin[:32],
                getattr(result, "message", "") or "",
            )

    # -- lifecycle -------------------------------------------------------
    async def start(self) -> None:
        if self._running:
            logger.debug("gateway already running")
            return
        self._running = True
        logger.info(
            "gateway starting: %d inbound adapter(s), %d registered channel(s)",
            len(self._inbound_adapters),
            len(self.registry.names()),
        )
        # Start only adapters that declare inbound_polling (P2 wires WeChat).
        for adapter in self._inbound_adapters:
            await adapter.start()
        logger.info("gateway started")

    async def stop(self) -> None:
        async with self._stop_lock:
            if not self._running:
                return
            self._running = False
            logger.info("gateway stopping")
            for adapter in self._inbound_adapters:
                try:
                    await adapter.stop()
                except Exception as exc:  # noqa: BLE001
                    logger.warning(
                        "gateway: adapter stop failed for %s: %s",
                        getattr(adapter, "channel_id", "?"),
                        exc,
                    )
            logger.info("gateway stopped")

    @property
    def running(self) -> bool:
        return self._running

    async def wait_channels_ready(self, timeout: float = 15.0) -> dict[str, str]:
        """Wait for inbound adapters to report a connected/healthy state.

        Polls each inbound adapter's ``health_check()`` until its
        ``account_status`` indicates the channel is live (or degraded-but-up),
        or ``timeout`` seconds elapse. Returns a ``{channel_id: status}`` map.
        Adapters that start non-blocking (e.g. Feishu WS connect loop) flip to
        ``connected`` asynchronously; this gives ``serve()`` a bounded window to
        confirm connectivity before declaring startup done, so a stuck connect
        surfaces as a degraded status rather than a silent invisible daemon.
        """
        deadline = asyncio.get_running_loop().time() + timeout
        result: dict[str, str] = {}
        while True:
            result = {}
            all_ready = True
            for adapter in self._inbound_adapters:
                cid = getattr(adapter, "channel_id", "?")
                try:
                    health = await adapter.health_check()
                    status = str(getattr(health, "account_status", "") or "")
                except Exception as exc:  # noqa: BLE001
                    status = f"health_error:{exc!s}"
                result[cid] = status
                # "connected"/"logged_in" = ready; anything else (reconnecting,
                # credentials_missing, etc.) keeps waiting.
                normalized_status = status.strip().lower()
                if normalized_status not in {"connected", "logged_in", "websocket:connected"}:
                    all_ready = False
            if all_ready or asyncio.get_running_loop().time() >= deadline:
                return result
            await asyncio.sleep(0.5)

    # -- inbound ---------------------------------------------------------
    def register_inbound(self, adapter) -> None:
        self._inbound_adapters.append(adapter)

    def set_handler(self, handler: InboundHandler) -> None:
        self.inbound.set_handler(handler)

    def set_push_handler(self, handler) -> None:
        """Register the IPC push callback for opt-in origins (REPL/orchestrator)."""
        self.inbound.set_push_handler(handler)

    async def receive(self, message: InboundMessage):
        """Convenience: publish to the bus and process synchronously."""
        return await self.inbound.process(message)

    # -- outbound --------------------------------------------------------
    async def send(self, message: OutboundMessage):
        return await self.outbound.send(message)

    async def broadcast(self, message: OutboundMessage, *, channels: list[str] | None = None):
        return await self.outbound.broadcast(message, channels=channels)

    # -- channel management ---------------------------------------------
    def reload_channel(self, name: str) -> bool:
        """Rebuild a single adapter from config and hot-swap it (P4 live reload).

        Stops the old adapter's inbound loop, rebuilds via ``_build_adapter``,
        re-registers, and re-attaches the inbound handler. In-flight message
        safety is best-effort in v1 (the outbox preserves pending sends).
        """
        channel_cfg = self.config.get_channel(name)
        if channel_cfg is None:
            logger.warning("gateway reload: channel %r not found in config", name)
            return False
        old = self.registry.get(name)
        if old is not None:
            _schedule_adapter_stop(old)
            self.registry.remove(name)
            if old in self._inbound_adapters:
                self._inbound_adapters.remove(old)
        adapter = self._build_adapter(channel_cfg)
        if adapter is None:
            return False
        self.registry.register(adapter)
        self._attach_inbound(adapter)
        if self._running and hasattr(adapter, "start"):
            _schedule_adapter_start(adapter)
        self.store.audit("channel_reload", channel=name)
        logger.info("gateway channel reloaded: %s", name)
        return True

    # -- audit -----------------------------------------------------------
    def _audit_binding(self, action: str, entry: BindingEntry, previous: BindingEntry | None) -> None:
        self.store.audit(
            action,
            origin=entry.origin,
            session_id=entry.target.session_id,
            host_type=entry.target.host_type,
        )
        # Schedule best-effort connection notification (async, never blocks
        # the binding transition). The auditor is called from the IPC server's
        # async handlers, so a running event loop is expected.
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                loop.create_task(self._notify_connection_change(action, entry, previous))
        except RuntimeError:
            pass

    async def _notify_connection_change(self, action: str, entry: BindingEntry, previous: BindingEntry | None) -> None:
        """Send a best-effort connection notification to all connected IM channels.

        When REPL or orchestrator connects/disconnects, every operator who has
        messaged the gateway is notified via the outbound dispatcher (e.g.
        ``"REPL connected"`` / ``"orchestrator disconnected"``). The notification is
        broadcast to every outbound channel that knows a recent sender — so
        operators on WeChat, Feishu, or any other connected channel all hear
        the connection event. All failures are logged and swallowed —
        notifications must never block binding transitions or crash the gateway.
        """
        # Broadcast to every outbound channel with a known recent sender.
        # This supersedes the origin-only routing (which hard-coded WeChat and
        # silently skipped when the WeChat adapter had no last_known_sender).
        targets = _collect_broadcast_targets(self.registry)
        if not targets:
            logger.debug(
                "connection notify: no broadcast targets for origin %s — skipping",
                (entry.origin or "")[:32],
            )
            return

        host_label = _host_label(entry.target.host_type)
        # Orchestrator currently supports command interaction only, so make
        # that limitation explicit in the connection notice.
        connected_note = (
            "; Orchestrator currently supports commands only" if entry.target.host_type == "orchestrator" else ""
        )
        messages: list[str] = []
        if action == "binding_created":
            messages.append(f"{host_label} connected{connected_note}")
        elif action == "binding_override" and previous is not None:
            if previous.connection_state == "active":
                messages.append(f"{_host_label(previous.target.host_type)} disconnected")
            messages.append(f"{host_label} connected{connected_note}")
        elif action in ("binding_offline", "binding_terminated"):
            messages.append(f"{host_label} disconnected")

        for out_channel, out_target in targets:
            for text in messages:
                try:
                    result = await self.outbound.send(
                        OutboundMessage(text=text, channel=out_channel, target=out_target, markdown=False)
                    )
                    if result is not None and getattr(result, "ok", True) is False:
                        error_category = getattr(result, "error_category", "")
                        category_value = getattr(error_category, "value", "") or str(error_category or "")
                        logger.error(
                            "connection notify: send failed for %r channel=%s category=%s message=%s",
                            text,
                            out_channel,
                            category_value,
                            getattr(result, "message", "") or "",
                        )
                        continue
                    if result is not None and getattr(result, "status", None) is SendStatus.ENQUEUED:
                        raw = getattr(result, "raw", None) or {}
                        logger.info(
                            "connection notify: enqueued %r channel=%s target=%s message=%s retry_after=%s",
                            text,
                            out_channel,
                            (out_target or "")[:16] + "…" if len(out_target or "") > 16 else out_target,
                            getattr(result, "message", "") or "",
                            raw.get("retry_after_seconds") or raw.get("retry_after") or "",
                        )
                        continue
                    logger.info(
                        "connection notify: sent %r channel=%s target=%s",
                        text,
                        out_channel,
                        (out_target or "")[:16] + "…" if len(out_target or "") > 16 else out_target,
                    )
                except Exception:  # noqa: BLE001
                    logger.error(
                        "connection notify: send failed for %r channel=%s — best-effort",
                        text,
                        out_channel,
                    )

    # -- health ----------------------------------------------------------
    async def health(self) -> dict:
        channel_health: list[dict] = []
        for adapter in self.registry.all_adapters():
            health_check = getattr(adapter, "health_check", None)
            if not callable(health_check):
                continue
            try:
                health = await health_check()
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "gateway: health check failed for %s: %s",
                    getattr(adapter, "channel_id", "?"),
                    exc,
                )
                channel_health.append(
                    {
                        "channel_id": getattr(adapter, "channel_id", ""),
                        "healthy": False,
                        "last_error": str(exc),
                    }
                )
                continue
            to_dict = getattr(health, "to_dict", None)
            channel_health.append(to_dict() if callable(to_dict) else dict(health))
        return {
            "running": self._running,
            "channels": self.registry.names(),
            "channel_health": channel_health,
            "inbound_adapters": len(self._inbound_adapters),
            "outbox_pending": len(self.store.outbox_pending()),
            "dead_letter": len(self.store.dead_letter_entries()),
            "bindings": [
                {
                    "origin": entry.origin,
                    "session_id": entry.target.session_id,
                    "host_type": entry.target.host_type,
                    "connection_state": entry.connection_state,
                }
                for entry in self.binding.all_bindings()
            ],
        }


def _schedule_adapter_stop(adapter) -> None:
    """Stop an adapter's inbound loop on the running loop (best-effort)."""
    if not hasattr(adapter, "stop"):
        return
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        return
    if loop.is_running():
        loop.create_task(adapter.stop())


def _schedule_adapter_start(adapter) -> None:
    """Start an adapter's inbound loop on the running loop (best-effort)."""
    if not hasattr(adapter, "start"):
        return
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        return
    if loop.is_running():
        loop.create_task(adapter.start())


def _normalize_config_channels(config: GatewayConfig) -> GatewayConfig:
    """Ensure direct GatewayConfig construction cannot bypass type uniqueness."""
    normalized = replace(config, channels=[])
    for channel in config.channels:
        normalized.replace_channel(channel)
    return normalized


def _wechat_state_file(state_dir, channel_name: str, suffix: str):
    path = state_dir / "wechat" / f"{channel_name}_{suffix}.json"
    if path.exists() or channel_name != "wechat":
        return path
    legacy = state_dir / "wechat" / f"wechat-main_{suffix}.json"
    if legacy.exists():
        return legacy
    matches = sorted((state_dir / "wechat").glob(f"*_{suffix}.json"))
    return matches[0] if matches else path


__all__ = ["MessageGateway"]


_HOST_LABELS = {
    "repl": "clawcodex-REPL",
    "orchestrator": "clawcodex-orchestrator",
    "opt_in": "opt-in",
}


def _host_label(host_type: str) -> str:
    """Map a binding host_type to a user-facing label for notifications."""
    return _HOST_LABELS.get(host_type, host_type)


def _collect_broadcast_targets(registry: ChannelAdapterRegistry) -> list[tuple[str, str]]:
    """Collect (channel_id, target) pairs for every outbound channel with a known sender.

    Used as the fallback when an origin (e.g. ``wechat:direct:*:*``) cannot be
    resolved to a concrete target. Each adapter that declares OUTBOUND_TEXT and
    exposes a non-None ``last_known_sender()`` contributes one target, so the
    connection notification reaches operators on every channel that has seen
    inbound traffic this gateway lifetime.
    """
    targets: list[tuple[str, str]] = []
    for adapter in registry.all_adapters():
        caps = getattr(adapter, "capabilities", None)
        if caps is None or not caps.has(ChannelCapability.OUTBOUND_TEXT):
            continue
        last_known = getattr(adapter, "last_known_sender", None)
        if not callable(last_known):
            continue
        sender = last_known()
        if not sender:
            continue
        channel_id = getattr(adapter, "channel_id", None)
        if channel_id:
            targets.append((channel_id, sender))
    return targets
