#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# -------------------------------------------------------------------------
# This file is part of the AgentSDK project.
#
# Originally from Clawd Codex:
# https://github.com/agentforce314/clawcodex
# Copyright (c) 2026 Clawd Codex Team
# Licensed under the MIT License. See clawcodex-ascend/LICENSES/Clawd-Codex-MIT.txt.
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

"""Extension hook for the REPL frontend.

This module owns the downstream side of the ``/provider`` and
``/model`` slash command wiring for :class:`src.repl.core.ClawcodexREPL`.
The goal is to keep all knowledge in ``clawcodex_ext/`` so the
upstream-shaped REPL core (``src/repl/core.py``) only sees a thin seam
(``runtime_context`` field + observer notification on swap).

Responsibilities
----------------
1. Register the ``/provider`` and ``/model`` ``LocalCommand``
   objects on the REPL's command registry.
2. Install a :class:`RuntimeObserver` that syncs the REPL's private
   ``provider`` / ``tool_registry`` / ``tool_context`` references after
   a :meth:`RuntimeContext.swap_provider` rebuild.

The frontend plugin (:class:`clawcodex_ext.frontend.repl.REPLFrontend`)
calls :func:`install_repl_extensions` immediately after
``ClawcodexREPL(...)`` construction but before ``repl.run()``.
"""

from __future__ import annotations
# pylint: disable=C0302,E0611

import logging
import shlex
import threading
from pathlib import Path
from typing import TYPE_CHECKING, Any
from clawcodex_ext.away_summary.controller import AwaySummaryController
from clawcodex_ext.away_summary.registration import register_away_summary_commands
from clawcodex_ext.cli.runtime_commands import register_runtime_commands
from clawcodex_ext.multimodel.runtime_command import register_multimodel_runtime_command
from clawcodex_ext.intent_forecast.config import load_intent_forecast_config
from clawcodex_ext.intent_forecast.controller import IntentForecastController
from clawcodex_ext.intent_forecast.messages import (
    create_forecast_system_message,
    format_forecast_for_display,
)
from clawcodex_ext.intent_forecast.registration import register_intent_forecast_commands
from clawcodex_ext.runtime.observer import attach_observer
from clawcodex_ext.providers.model_catalog_cache import prewarm_model_catalog

if TYPE_CHECKING:  # pragma: no cover
    from src.repl.core import ClawcodexREPL

_log = logging.getLogger(__name__)

# Chinese rendering of the (English) option descriptions built in
# ``_handle_permission_request``. Keys are the exact desc strings; the
# ``Enable {setting} and allow`` form is matched by prefix/suffix so any
# setting name is covered. Unknown descs pass through unchanged.
_DESC_ZH = {
    "Yes, allow this action": "是，允许此操作",
    "No, deny this action": "否，拒绝此操作",
}


def _zh_option_desc(desc: str) -> str:
    if desc in _DESC_ZH:
        return _DESC_ZH[desc]
    if desc.startswith("Enable ") and desc.endswith(" and allow"):
        setting = desc[len("Enable ") : -len(" and allow")]
        return f"启用 {setting} 并允许"
    return desc


async def _send_client_outbound(
    client,
    *,
    origin: str,
    text: str,
    context_token: str | None = None,
    metadata: dict | None = None,
    semantic_tags: list[str] | None = None,
    in_reply_to: str | None = None,
):
    kwargs = {
        "origin": origin,
        "text": text,
        "context_token": context_token,
        "metadata": metadata,
        "semantic_tags": semantic_tags,
        "in_reply_to": in_reply_to,
    }
    kwargs = {key: value for key, value in kwargs.items() if value is not None}
    fallback_order = ("context_token", "metadata", "semantic_tags", "in_reply_to")
    while True:
        try:
            return await client.send_outbound(**kwargs)
        except TypeError as exc:
            message = str(exc)
            remove_key = next(
                (key for key in fallback_order if key in kwargs and key in message),
                None,
            )
            if remove_key is None:
                remove_key = next((key for key in fallback_order if key in kwargs), None)
            if remove_key is None:
                raise
            kwargs.pop(remove_key, None)


class _ImReplyController:
    """Sends the REPL's final assistant reply back to the IM origin.

    Mirrors the ``AwaySummaryController``/``GoalController`` lifecycle hooks
    that ``ClawCodexExtREPL.chat()`` invokes in its ``finally`` block. On
    ``on_assistant_turn_complete`` it pulls the last assistant text from the
    conversation and ships it to the gateway via an OUTBOUND IPC frame; the
    gateway's OutboundDispatcher then delivers it to WeChat.
    """

    def __init__(self, repl, client, origin: str) -> None:
        self._repl = repl
        self._client = client
        self._origin = origin
        self._last_sent: str | None = None
        self._run_outcome = "success"
        self._assistant_count_at_start: int | None = None

    def on_run_start(self) -> None:
        # reset per-turn so a repeated identical reply across turns still sends
        self._last_sent = None
        self._run_outcome = "success"
        self._assistant_count_at_start = self._assistant_message_count()

    def on_run_finish(self, outcome: str = "success") -> None:
        self._run_outcome = outcome if outcome in {"success", "failure", "cancelled"} else "failure"

    def on_assistant_turn_complete(self) -> None:
        import asyncio

        loop = getattr(self._repl, "_cron_loop", None)
        if loop is None or loop.is_closed():
            return
        text_fn = getattr(self._repl, "_get_last_assistant_text", None)
        pending = self._pop_reply_delivery()
        if pending is None:
            return
        im_origin, context_token, delivery_id = pending
        if self._run_outcome != "success":
            self._schedule_processing_complete(delivery_id, self._run_outcome)
            return
        assistant_count = self._assistant_message_count()
        if (
            self._assistant_count_at_start is not None
            and assistant_count is not None
            and assistant_count <= self._assistant_count_at_start
        ):
            self._schedule_processing_complete(delivery_id, "success")
            return
        if not callable(text_fn):
            self._schedule_processing_complete(delivery_id, "success")
            return
        try:
            text = text_fn()
        except Exception:  # noqa: BLE001
            self._schedule_processing_complete(delivery_id, "failure")
            return
        if not text or text == self._last_sent:
            self._schedule_processing_complete(delivery_id, "success")
            return
        self._last_sent = text
        client = self._client._client  # GatewayIpcClient
        try:
            # run_coroutine_threadsafe works even when the target loop is
            # not running (the normal state after chat() returns). The
            # coroutine executes on the next run_until_complete call.
            asyncio.run_coroutine_threadsafe(
                _send_client_outbound(
                    client,
                    origin=im_origin,
                    text=text,
                    context_token=context_token,
                    in_reply_to=delivery_id,
                ),
                loop,
            )
            _log.info("repl IM reply sent: origin=%s len=%d", im_origin[:24], len(text))
        except Exception:  # noqa: BLE001
            _log.warning("repl IM reply send failed", exc_info=True)
            self._schedule_processing_complete(delivery_id, "failure")

    def _assistant_message_count(self) -> int | None:
        session = getattr(self._repl, "session", None)
        conversation = getattr(session, "conversation", None)
        messages = getattr(conversation, "messages", None)
        if messages is None:
            return None
        try:
            return sum(1 for message in messages if getattr(message, "role", None) == "assistant")
        except Exception:  # noqa: BLE001
            return None

    def send_command_feedback(
        self,
        command: str,
        *,
        success: bool = True,
        message: str | None = None,
    ) -> bool:
        """Send a visible completion notice for an IM-driven local command."""
        pending = self._pop_reply_delivery()
        if pending is None:
            return False
        im_origin, context_token, delivery_id = pending

        normalized = self._normalize_command_name(command)
        text = message or self._format_command_feedback(normalized, success=success)
        sent = self._send_outbound_text(
            im_origin,
            text,
            context_token=context_token,
            metadata={
                "intent": "command_feedback",
                "command": normalized,
                "success": success,
            },
            semantic_tags=["command_feedback"],
            in_reply_to=delivery_id if success else None,
        )
        if not success or not sent:
            self._schedule_processing_complete(delivery_id, "failure")
        return sent

    def _pop_reply_delivery(self) -> tuple[str, str | None, str | None] | None:
        pop_delivery = getattr(self._client, "pop_reply_delivery", None)
        if callable(pop_delivery):
            pending = pop_delivery()
            if pending is None:
                return None
            return pending.origin, pending.context_token, pending.delivery_id
        delivery_fn = getattr(self._client, "peek_reply_delivery_id", None)
        delivery_id = delivery_fn() if callable(delivery_fn) else None
        pop_context = getattr(self._client, "pop_reply_context", None)
        if callable(pop_context):
            im_origin, context_token = pop_context()
        else:
            context_fn = getattr(self._client, "peek_reply_context_token", None)
            context_token = context_fn() if callable(context_fn) else None
            pop_origin = getattr(self._client, "pop_reply_origin", None)
            if not callable(pop_origin):
                return None
            im_origin = pop_origin()
        if not im_origin:
            return None
        return im_origin, context_token, delivery_id

    def _schedule_processing_complete(self, message_id: str | None, outcome: str) -> None:
        if not message_id:
            return
        import asyncio

        client = getattr(self._client, "_client", None)
        complete = getattr(client, "complete_processing", None)
        loop = getattr(self._repl, "_cron_loop", None)
        if not callable(complete) or loop is None or loop.is_closed():
            return
        try:
            asyncio.run_coroutine_threadsafe(
                complete(message_id=message_id, outcome=outcome),
                loop,
            )
        except Exception:  # noqa: BLE001
            _log.warning("repl IM processing completion failed", exc_info=True)

    @staticmethod
    def _normalize_command_name(command: str) -> str:
        raw = str(command or "").strip()
        if not raw:
            return "/"
        name = raw.split(maxsplit=1)[0]
        if not name.startswith("/"):
            name = f"/{name}"
        return name

    @staticmethod
    def _format_command_feedback(command: str, *, success: bool = True) -> str:
        status = "已执行" if success else "执行失败"
        return f"命令{status}：{command}"

    def send_permission_prompt(
        self,
        *,
        message: str,
        options: list[tuple[str, str]],
        suggestion: str | None = None,
        interactive: bool = False,
        allow_choices: set[str] | None = None,
    ) -> bool:
        """Forward an in-REPL permission menu to the active IM origin.

        ``interactive=True`` is used when the WeChat user is expected to
        reply with a choice (IM-driven permission wait): the footer invites
        a reply instead of directing the user back to the REPL.

        ``allow_choices`` carries stable decision semantics for rich channels;
        consumers must not infer approval from REPL-specific keys such as ``s``.
        """
        peek_origin = getattr(self._client, "peek_reply_origin", None)
        if not callable(peek_origin):
            return False
        im_origin = peek_origin()
        if not im_origin:
            return False
        context_fn = getattr(self._client, "peek_reply_context_token", None)
        context_token = context_fn() if callable(context_fn) else None
        text = self._format_permission_prompt(message, options, suggestion=suggestion, interactive=interactive)
        metadata = None
        semantic_tags = None
        if interactive:
            metadata = {
                "intent": "permission_approval",
                "permission": {
                    "message": str(message).strip().replace("Claude", "ClawCodex"),
                    "suggestion": suggestion,
                    "options": [
                        {
                            "value": key,
                            "label": _zh_option_desc(desc),
                            **(
                                {"decision": "allow" if key in allow_choices else "deny"}
                                if allow_choices is not None
                                else {}
                            ),
                        }
                        for key, desc in options
                    ],
                    "expires_in_seconds": 600,
                },
            }
            semantic_tags = ["approval"]
        send_kwargs = {"metadata": metadata, "semantic_tags": semantic_tags}
        if context_token is not None:
            send_kwargs["context_token"] = context_token
        return self._send_outbound_text(im_origin, text, **send_kwargs)

    @staticmethod
    def _format_permission_prompt(
        message: str,
        options: list[tuple[str, str]],
        *,
        suggestion: str | None = None,
        interactive: bool = False,
    ) -> str:
        # WeChat-facing prompt is Chinese; the upstream ``message`` and
        # option descriptions are English (and the message may say "Claude"),
        # so translate the wrapper + option descriptions and rebrand Claude
        # → ClawCodex here only (the REPL console keeps its own rendering).
        msg = str(message).strip().replace("Claude", "ClawCodex")
        if msg.startswith("ClawCodex wants to use ") and msg.endswith(". Allow?"):
            tool = msg[len("ClawCodex wants to use ") : -len(". Allow?")]
            msg = f"ClawCodex 想使用 {tool}，是否允许？"
        lines = ["需要权限", "", msg]
        if suggestion:
            lines.extend(["", f"建议：{suggestion}"])
        lines.extend(["", "选项："])
        for idx, (key, desc) in enumerate(options, start=1):
            lines.append(f"{idx}. [{key}] {_zh_option_desc(desc)}")
        if interactive:
            lines.extend(["", "请回复选项编号或字母（如 1 或 y）进行选择。"])
        else:
            lines.extend(["", "请在 REPL 中选择对应的选项以继续。"])
        return "\n".join(lines)

    def _send_outbound_text(
        self,
        im_origin: str,
        text: str,
        *,
        context_token: str | None = None,
        metadata: dict | None = None,
        semantic_tags: list[str] | None = None,
        in_reply_to: str | None = None,
    ) -> bool:
        import asyncio

        loop = getattr(self._repl, "_cron_loop", None)
        if loop is None or loop.is_closed():
            return False
        client = self._client._client
        try:
            try:
                running = asyncio.get_running_loop()
            except RuntimeError:
                running = None
            if running is loop:
                # Inside _cron_loop and it is turning — schedule cooperatively.
                asyncio.run_coroutine_threadsafe(
                    _send_client_outbound(
                        client,
                        origin=im_origin,
                        text=text,
                        context_token=context_token,
                        metadata=metadata,
                        semantic_tags=semantic_tags,
                        in_reply_to=in_reply_to,
                    ),
                    loop,
                )
                return True
            if in_reply_to is not None:
                # Correlated final/command replies must use the registered
                # REPL IPC peer so the gateway can authorize completion of
                # the matching pending message. The loop will drain as soon
                # as the synchronous command returns.
                asyncio.run_coroutine_threadsafe(
                    _send_client_outbound(
                        client,
                        origin=im_origin,
                        text=text,
                        context_token=context_token,
                        metadata=metadata,
                        semantic_tags=semantic_tags,
                        in_reply_to=in_reply_to,
                    ),
                    loop,
                )
                return True
            # Not inside _cron_loop: the caller is sync code that is about to
            # block the main thread (e.g. the permission handler blocking on
            # ``_safe_input``). ``run_coroutine_threadsafe`` would queue on
            # ``_cron_loop`` which won't drain until ``chat()`` returns — far
            # too late (the user would already have approved in the REPL).
            # Send from a one-shot thread so the message leaves immediately.
            return self._send_outbound_text_from_thread(
                im_origin,
                text,
                context_token=context_token,
                metadata=metadata,
                semantic_tags=semantic_tags,
                in_reply_to=in_reply_to,
            )
        except Exception:  # noqa: BLE001
            _log.warning("repl IM outbound send failed", exc_info=True)
            return False

    def _send_outbound_text_from_thread(
        self,
        im_origin: str,
        text: str,
        *,
        context_token: str | None = None,
        metadata: dict | None = None,
        semantic_tags: list[str] | None = None,
        in_reply_to: str | None = None,
    ) -> bool:
        """Send while the REPL event loop is about to block on terminal input."""
        socket_path = getattr(self._client, "_socket_path", None)
        if not socket_path:
            return False

        def _runner() -> None:
            import asyncio

            async def _send_once() -> None:
                from clawcodex_ext.services.im_gateway.ipc_client import GatewayIpcClient

                client = GatewayIpcClient(str(socket_path), instance_id="repl-im-permission")
                await client.connect()
                try:
                    await _send_client_outbound(
                        client,
                        origin=im_origin,
                        text=text,
                        context_token=context_token,
                        metadata=metadata,
                        semantic_tags=semantic_tags,
                        in_reply_to=in_reply_to,
                    )
                finally:
                    await client.close()

            try:
                asyncio.run(_send_once())
            except Exception:  # noqa: BLE001
                _log.warning("repl IM one-shot outbound failed", exc_info=True)

        threading.Thread(target=_runner, name="repl-im-outbound", daemon=True).start()
        return True


class _ReplMultiModelRenderer:
    """Render per-slot outcomes emitted by a :class:`MultiModelRouter`.

    The router runs provider calls in worker threads, while the regular REPL
    only receives the selected aggregate response.  Listening at the router
    boundary lets the interactive transcript show the otherwise-audited
    candidate responses without changing the query loop.
    """

    def __init__(self, repl: "ClawcodexREPL") -> None:
        self._repl = repl
        self._provider: Any | None = None

    def bind(self, provider: Any) -> None:
        """Subscribe to the active router, detaching from a prior provider."""
        if self._provider is provider:
            return
        if self._provider is not None:
            remove = getattr(self._provider, "remove_event_listener", None)
            if callable(remove):
                remove(self._on_router_event)
        self._provider = provider
        add = getattr(provider, "add_event_listener", None)
        if callable(add):
            add(self._on_router_event)

    def _on_router_event(self, event: str, payload: dict[str, Any]) -> None:
        if event != "aggregated":
            return
        results = payload.get("results") or []
        if not results:
            return

        console = getattr(self._repl, "console", None)
        if console is None:
            return
        console.print("\n[bold]多模型候选结果[/bold]")
        for result in results:
            response = getattr(result, "response", None)
            slot = getattr(result, "slot_name", "unknown")
            model = getattr(response, "model", "") or "unknown"
            elapsed_s = getattr(result, "duration_ms", 0) / 1000
            console.print(f"  [cyan]{slot}[/cyan] · {model} · {elapsed_s:.2f}s")
            error = getattr(result, "error", None)
            if error:
                console.print(f"    错误: {error}", markup=False)
                continue
            content = (getattr(response, "content", "") or "").strip()
            if content:
                console.print("    " + content.replace("\n", "\n    "), markup=False)

        output = payload.get("output")
        chosen = getattr(getattr(output, "chosen", None), "model", None)
        if chosen:
            console.print(f"[dim]最终采用: {chosen}[/dim]")


class _ReplRuntimeObserver:
    """Sync REPL private state when the runtime swaps provider.

    Implements :class:`RuntimeObserver`. The REPL holds cached
    references to ``provider`` / ``tool_registry`` / ``tool_context`` and
    a command context that mirrors them; all four must be refreshed
    after a provider swap so the next prompt uses the new model.
    """

    def __init__(self, repl: "ClawcodexREPL", renderer: _ReplMultiModelRenderer) -> None:
        self._repl = repl
        self._renderer = renderer

    def on_runtime_swap(self, runtime) -> None:
        repl = self._repl
        repl.provider = runtime.provider
        repl.provider_name = runtime.provider_name
        repl.tool_registry = runtime.tool_registry
        repl.tool_context = runtime.tool_context
        self._renderer.bind(runtime.provider)
        if hasattr(repl, "command_context") and repl.command_context is not None:
            repl.command_context.provider = runtime.provider
            repl.command_context.tool_registry = runtime.tool_registry
            repl.command_context.tool_context = runtime.tool_context
        prewarm_model_catalog(runtime.provider_name, runtime.provider)


def install_repl_extensions(repl: "ClawcodexREPL", ctx) -> None:
    """Wire slash commands + observer into the REPL.

    Args:
        repl: A fully-constructed :class:`ClawcodexREPL`. The function
            reads ``repl.command_registry`` and ``repl.runtime_context``;
            it does not mutate the REPL's public surface beyond
            registering commands and attaching an observer.
        ctx: The downstream :class:`RuntimeContext` (or any object
            exposing the runtime protocol). Used to attach the observer
            that fires on ``swap_provider``.
    """
    # Register /provider and /model into the REPL's local command
    # registry so the slash-command dispatcher can find them.
    if getattr(repl, "command_registry", None) is not None:
        register_runtime_commands(repl.command_registry)
        register_multimodel_runtime_command(repl.command_registry)
        register_away_summary_commands(repl.command_registry)
        register_intent_forecast_commands(repl.command_registry)
        register_intent_forecast_commands(None)
        update_commands = getattr(
            repl,
            "_update_built_in_commands_with_command_system",
            None,
        )
        if callable(update_commands):
            update_commands()

    _install_away_summary_controller(repl)
    _install_intent_forecast_controller(repl)
    _install_goal_controller(repl)
    _install_gateway_client(repl, ctx)

    runtime = getattr(repl, "runtime_context", None)
    if runtime is None:
        runtime = ctx
    if runtime is None:
        return

    renderer = _ReplMultiModelRenderer(repl)
    renderer.bind(runtime.provider)
    repl._multimodel_renderer = renderer
    attach_observer(runtime, _ReplRuntimeObserver(repl, renderer))
    prewarm_model_catalog(runtime.provider_name, runtime.provider)

    # ---- SIGTERM / SIGINT: save session + print resume hint (S-R1) ----
    _register_signal_session_save(repl)


def _install_gateway_client(repl: "ClawcodexREPL", ctx=None) -> None:
    """Install REPL IM gateway command hooks and optional startup opt-in.

    Startup connection remains opt-in via ``--gateway`` / ``--gateway-origin``.
    The ``/gateway`` command is always installed so a normally-started REPL can
    connect to an already-running gateway daemon later.
    """
    import os

    _install_gateway_command(repl)
    repl._handle_gateway_command = lambda args="": _handle_gateway_command(repl, args)
    if getattr(repl, "command_context", None) is not None:
        repl.command_context.repl = repl

    options = getattr(ctx, "options", None)
    sock = (
        getattr(options, "gateway_sock", None)
        or os.environ.get("CLAWCODEX_GATEWAY_SOCK")
        or os.environ.get("CLAWCODEX_IM_GATEWAY_SOCK")
    )
    origin = (
        getattr(options, "gateway_origin", None)
        or os.environ.get("CLAWCODEX_GATEWAY_ORIGIN")
        or os.environ.get("CLAWCODEX_IM_ORIGIN")
    )
    enabled = bool(getattr(options, "gateway", False))
    if not origin and enabled:
        from clawcodex_ext.services.im_gateway.models import IM_DIRECT_ALL_ORIGIN

        origin = IM_DIRECT_ALL_ORIGIN
    if not origin:
        return  # opt-in not requested
    if not sock:
        sock = str(os.path.expanduser("~/.clawcodex/gateway/gateway.sock"))

    client = _new_repl_gateway_client(repl, origin=origin, sock=sock)
    repl._gateway_client = client
    repl._im_reply_controller = _ImReplyController(repl, client, origin)
    _log.info("repl IM gateway startup opt-in staged: origin=%s sock=%s", origin[:32], sock)

    async def _im_init(loop):
        """Connect + register + start heartbeat on the REPL's loop."""
        message = await _connect_repl_gateway(repl, origin=origin, sock=sock)
        if "connected" in message:
            _log.info("repl IM gateway opt-in connected: origin=%s sock=%s", origin[:32], sock)
        elif "IM gateway daemon is not running" in message:
            _log.error(
                "IM gateway daemon is not running. Start it first:\n"
                "    clawcodex-dev gateway start\n"
                "Then relaunch with --gateway."
            )
        else:
            _log.warning("repl IM gateway opt-in connect failed: %s", message)

    repl._gateway_init = _im_init


def _install_gateway_command(repl: "ClawcodexREPL") -> None:
    try:
        from src.command_system.registry import get_command_registry
        from src.command_system.types import LocalCommand
    except Exception:  # noqa: BLE001
        return

    def _make_command() -> LocalCommand:
        command = LocalCommand(
            name="gateway",
            description="Connect, show status, or disconnect the IM gateway",
            argument_hint="connect|status|disconnect [--origin ORIGIN] [--sock PATH]",
        )
        command.set_call(_gateway_command_call)
        return command

    for registry in (get_command_registry(), getattr(repl, "command_registry", None)):
        if registry is None:
            continue
        try:
            registry.register(_make_command())
        except Exception:  # noqa: BLE001
            _log.debug("failed to register /gateway command", exc_info=True)
    update_commands = getattr(repl, "_update_built_in_commands_with_command_system", None)
    if callable(update_commands):
        update_commands()
    if hasattr(repl, "_slash_suggestions_cache"):
        repl._slash_suggestions_cache = None
        repl._slash_suggestions_cache_at = 0.0


def _gateway_command_call(args: str, context) -> object:
    from src.command_system.types import LocalCommandResult

    repl = getattr(context, "repl", None)
    if repl is None:
        return LocalCommandResult(type="text", value="IM gateway: unavailable")
    handler = getattr(repl, "_handle_gateway_command", None)
    if not callable(handler):
        return LocalCommandResult(type="text", value="IM gateway: unavailable")
    return LocalCommandResult(type="text", value=str(handler(args)))


def _handle_gateway_command(repl: "ClawcodexREPL", args: str = "") -> str:
    try:
        tokens = shlex.split(args or "")
    except ValueError as exc:
        return f"Usage: /gateway connect|status|disconnect [--origin ORIGIN] [--sock PATH]\n{exc}"
    action = (tokens[0].lower() if tokens else "status").strip()
    rest = tokens[1:] if tokens else []

    if action == "status":
        return _format_repl_gateway_status(repl)
    if action == "connect":
        origin, sock, error = _parse_gateway_options(rest)
        if error:
            return error
        return _run_repl_gateway_coro(
            repl,
            _connect_repl_gateway(repl, origin=origin, sock=sock),
        )
    if action == "disconnect":
        return _run_repl_gateway_coro(repl, _disconnect_repl_gateway(repl))
    return "Usage: /gateway connect|status|disconnect [--origin ORIGIN] [--sock PATH]"


def _parse_gateway_options(tokens: list[str]) -> tuple[str | None, str | None, str | None]:
    origin = None
    sock = None
    idx = 0
    while idx < len(tokens):
        token = tokens[idx]
        if token in {"--origin", "--gateway-origin", "--im-gateway-origin"}:
            if idx + 1 >= len(tokens):
                return None, None, f"error: {token} requires a value"
            origin = tokens[idx + 1]
            idx += 2
            continue
        if token in {"--sock", "--gateway-sock", "--im-gateway-sock"}:
            if idx + 1 >= len(tokens):
                return None, None, f"error: {token} requires a value"
            sock = tokens[idx + 1]
            idx += 2
            continue
        if token in {"--gateway", "--im-gateway"}:
            from clawcodex_ext.services.im_gateway.models import IM_DIRECT_ALL_ORIGIN

            origin = IM_DIRECT_ALL_ORIGIN
            idx += 1
            continue
        return None, None, f"error: unknown /gateway option {token!r}"
    return origin, sock, None


def _default_gateway_origin() -> str:
    from clawcodex_ext.services.im_gateway.models import IM_DIRECT_ALL_ORIGIN

    return IM_DIRECT_ALL_ORIGIN


def _default_gateway_sock() -> str:
    import os

    return (
        os.environ.get("CLAWCODEX_GATEWAY_SOCK")
        or os.environ.get("CLAWCODEX_IM_GATEWAY_SOCK")
        or str(os.path.expanduser("~/.clawcodex/gateway/gateway.sock"))
    )


def _next_repl_gateway_session_id(repl: "ClawcodexREPL") -> str:
    import os

    counter = int(getattr(repl, "_gateway_session_counter", 0)) + 1
    repl._gateway_session_counter = counter
    return f"repl-{os.getpid()}-{counter}"


def _new_repl_gateway_client(repl: "ClawcodexREPL", *, origin: str, sock: str):
    from clawcodex_ext.frontend.repl_gateway import ReplGatewayClient

    session_id = _next_repl_gateway_session_id(repl)
    client = ReplGatewayClient(
        sock,
        session_id=session_id,
        origin=origin,
        enqueue=repl._enqueue_prompt,
        queue_size=lambda: len(getattr(repl, "_queued_prompts", [])),
        wake=repl._wake_prompt_for_im,
        control_handler=lambda text, inbound_origin=None: _handle_im_control(repl, text, inbound_origin),
        permission_probe=lambda text: _handle_im_permission_reply(repl, text),
    )
    return client


async def _connect_repl_gateway(
    repl: "ClawcodexREPL",
    *,
    origin: str | None = None,
    sock: str | None = None,
) -> str:
    origin = origin or _default_gateway_origin()
    sock = sock or _default_gateway_sock()
    current = getattr(repl, "_gateway_client", None)
    current_origin = _client_origin(current)
    current_sock = _client_sock(current)
    if current is not None and getattr(current, "is_connected", False):
        if str(current_origin) == origin and str(current_sock) == sock:
            return f"IM gateway connected: origin={origin} sock={sock}"

    reuse_current = current is not None and str(current_origin) == origin and str(current_sock) == sock
    client = current if reuse_current else _new_repl_gateway_client(repl, origin=origin, sock=sock)
    try:
        await client.connect()
        await _start_repl_gateway_heartbeat(client)
    except FileNotFoundError:
        if not reuse_current:
            await _close_repl_gateway_client(client)
        return "IM gateway daemon is not running"
    except Exception as exc:  # noqa: BLE001
        if not reuse_current:
            await _close_repl_gateway_client(client)
        _log.warning("repl IM gateway connect failed", exc_info=True)
        return f"IM gateway connect failed: {exc}"

    old = getattr(repl, "_gateway_client", None)
    repl._gateway_client = client
    repl._im_reply_controller = _ImReplyController(repl, client, origin)
    if old is not None and old is not client:
        try:
            await _close_repl_gateway_client(old)
        except Exception:  # noqa: BLE001
            _log.debug("failed to close previous REPL gateway client", exc_info=True)
    return f"IM gateway connected: origin={origin} sock={sock}"


async def _disconnect_repl_gateway(repl: "ClawcodexREPL") -> str:
    client = getattr(repl, "_gateway_client", None)
    if client is None:
        repl._im_reply_controller = None
        return "IM gateway disconnected"
    try:
        await _close_repl_gateway_client(client)
    finally:
        repl._gateway_client = None
        repl._im_reply_controller = None
    return "IM gateway disconnected"


def _client_origin(client) -> str | None:
    if client is None:
        return None
    value = getattr(client, "origin", getattr(client, "_origin", None))
    if value is None:
        value = getattr(client, "kwargs", {}).get("origin")
    return value


def _client_sock(client) -> str | None:
    if client is None:
        return None
    return getattr(client, "socket_path", getattr(client, "_socket_path", None))


async def _start_repl_gateway_heartbeat(client) -> None:
    start = getattr(client, "start_heartbeat", None)
    if callable(start):
        await start(30.0)
        return
    loop = getattr(client, "_heartbeat_loop", None)
    if callable(loop):
        import asyncio

        client._heartbeat_task = asyncio.create_task(loop(30.0))


async def _close_repl_gateway_client(client) -> None:
    close = getattr(client, "close", None)
    if callable(close):
        await close()


def _format_repl_gateway_status(repl: "ClawcodexREPL") -> str:
    client = getattr(repl, "_gateway_client", None)
    if client is None or not getattr(client, "is_connected", False):
        return "IM gateway disconnected"
    origin = _client_origin(client) or "unknown"
    sock = _client_sock(client) or "unknown"
    return f"IM gateway connected: origin={origin} sock={sock}"


def _run_repl_gateway_coro(repl: "ClawcodexREPL", coro) -> str:
    import asyncio

    loop = getattr(repl, "_cron_loop", None)
    try:
        if loop is not None and not loop.is_closed():
            if loop.is_running():
                future = asyncio.run_coroutine_threadsafe(coro, loop)
                return str(future.result(timeout=10.0))
            return str(loop.run_until_complete(coro))
        return str(asyncio.run(coro))
    except Exception as exc:  # noqa: BLE001
        return f"IM gateway command failed: {exc}"


def _handle_im_control(repl: "ClawcodexREPL", text: str, origin: str | None = None) -> bool:
    command = (text or "").strip().split(maxsplit=1)[0].lower()
    if command.startswith("/") and command != "/stop":
        return False
    interrupt = getattr(repl, "_interrupt_active_chat_from_im", None)
    if not callable(interrupt):
        return False
    return bool(interrupt())


def _handle_im_permission_reply(repl: "ClawcodexREPL", text: str) -> bool:
    """Consume a WeChat reply as a pending permission decision.

    Active only while the REPL is blocked in ``_wait_im_permission_choice``
    (i.e. an IM-driven turn is waiting for the user to approve a tool). A
    reply matching one of the menu's valid choices resolves the wait;
    ``/stop`` resolves it as deny so the user can abort from WeChat instead
    of hanging. Anything else returns False and falls through to the normal
    control/enqueue path.

    Runs on the IPC reader thread; ``_wait_im_permission_choice`` waits on
    the main thread — ``repl._im_permission_lock`` serializes them.
    """
    lock = getattr(repl, "_im_permission_lock", None)
    if lock is None:
        return False
    with lock:
        state = getattr(repl, "_im_permission_wait", None)
        if not state:
            return False
        norm = (text or "").strip().lower()
        if not norm:
            return False
        if norm == "/stop":
            choice = "n"
        elif norm in state["valid"]:
            choice = norm
        else:
            return False
        state["choice"] = choice
        state["event"].set()
    return True


def _install_away_summary_controller(repl: "ClawcodexREPL") -> None:
    if getattr(repl, "_away_summary_controller", None) is not None:
        return

    session = getattr(repl, "session", None)
    conversation = getattr(session, "conversation", None)
    if conversation is None:
        return

    def _display(text: str) -> None:
        print_recap = getattr(repl, "_print_local_command_text", None)
        if callable(print_recap):
            print_recap(text, command="recap")
            return
        console = getattr(repl, "console", None)
        if console is not None:
            console.print(text)

    repl._away_summary_controller = AwaySummaryController(
        conversation=conversation,
        provider_getter=lambda: getattr(repl, "provider", None),
        model_getter=lambda: getattr(getattr(repl, "provider", None), "model", None),
        session_getter=lambda: getattr(repl, "session", None),
        display=_display,
    )


def _install_intent_forecast_controller(repl: "ClawcodexREPL") -> None:
    old = getattr(repl, "_intent_forecast_controller", None)
    if old is not None:
        try:
            old.close()
        except Exception:  # nosec B110
            pass  # Intentional best-effort path; the surrounding fallback remains valid.

    session = getattr(repl, "session", None)
    conversation = getattr(session, "conversation", None)
    if conversation is None:
        return

    workspace_root = getattr(repl, "workspace_root", None)
    if workspace_root is None:
        tool_context = getattr(repl, "tool_context", None)
        workspace_root = getattr(tool_context, "workspace_root", None) or getattr(tool_context, "cwd", None) or "."

    def _display(result) -> None:
        text = format_forecast_for_display(result)
        print_text = getattr(repl, "_print_local_command_text", None)
        if callable(print_text):
            print_text(text, command="forecast")
        else:
            console = getattr(repl, "console", None)
            if console is not None:
                console.print(text)

        # Persist forecast to conversation so it survives REPL→TUI transition.
        try:
            _session = getattr(repl, "session", None)
            _conv = getattr(_session, "conversation", None) if _session else None
            if _conv is not None and result.generated:
                msg = create_forecast_system_message(result, trigger="auto")
                _conv.messages.append(msg)
                if _session is not None:
                    _session.save()
        except Exception:  # nosec B110
            pass  # Intentional best-effort path; the surrounding fallback remains valid.

    def _submit(prompt: str) -> None:
        chat = getattr(repl, "chat", None)
        if callable(chat):
            chat(prompt)

    repl._intent_forecast_controller = IntentForecastController(
        provider_getter=lambda: getattr(repl, "provider", None),
        model_getter=lambda: getattr(getattr(repl, "provider", None), "model", None),
        session_getter=lambda: getattr(repl, "session", None),
        workspace_root=Path(workspace_root),
        display=_display,
        submit=_submit,
        config_loader=lambda: load_intent_forecast_config(cwd=Path(workspace_root)),
        conversation_getter=lambda: getattr(getattr(repl, "session", None), "conversation", None),
    )
    try:
        repl._intent_forecast_controller.on_mount()
    except Exception:  # nosec B110
        pass  # Intentional best-effort path; the surrounding fallback remains valid.
    if getattr(repl, "command_context", None) is not None:
        repl.command_context.intent_forecast_controller = repl._intent_forecast_controller


def _install_goal_controller(repl: "ClawcodexREPL") -> None:
    """Legacy GoalController is removed; goal runtime is tool-context based."""
    repl._goal_controller = None


def _register_signal_session_save(repl: "ClawcodexREPL") -> None:
    """Register a graceful-shutdown cleanup that saves the session and
    prints a resume hint when the process receives SIGTERM/SIGINT.

    Uses the upstream ``register_cleanup`` from ``src.utils.graceful_shutdown``
    which is already installed by ``init()``.

    The print is delegated to :func:`clawcodex_ext.utils.resume_hint.print_resume_hint`
    so the hint is centralised — and that helper's process-wide latch
    keeps the hint to a single emission even if the inline REPL ``/exit``
    path has already printed one. The ``session.save()`` call is
    unconditional because persistence must run regardless of whether the
    user has already seen the hint.
    """
    try:
        from src.utils.graceful_shutdown import register_cleanup
    except ImportError:
        return

    # Capture session reference once at registration time and again
    # just before the cleanup runs (the REPL may swap sessions mid-run).
    sid_ref = {"session": None}

    def _capture_ref() -> None:
        sid_ref["session"] = getattr(repl, "session", None)

    _capture_ref()

    def _cleanup() -> None:
        _capture_ref()
        session = sid_ref["session"]
        if session is None:
            return
        # Always persist — independent of whether the hint gets printed.
        try:
            session.save()
        except Exception:  # nosec B110
            pass  # Intentional best-effort path; the surrounding fallback remains valid.
        # Print via the canonical helper. Its process-wide latch
        # suppresses the duplicate if ``/exit`` already printed.
        try:
            from clawcodex_ext.utils.resume_hint import print_resume_hint

            print_resume_hint(getattr(session, "session_id", None))
        except Exception:  # nosec B110
            pass  # Intentional best-effort path; the surrounding fallback remains valid.

    register_cleanup(_cleanup)
