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

"""Tests for dispatcher."""

from __future__ import annotations

# pylint: disable=no-name-in-module  # Split migration branches provide gateway modules.

import asyncio

import pytest

from clawcodex_ext.services.im_gateway.binding import BindingPolicy
from clawcodex_ext.services.im_gateway.config import CommandAllowlistConfig
from clawcodex_ext.services.im_gateway.dispatcher import InboundDispatcher
from clawcodex_ext.services.im_gateway.models import (
    AckLayer,
    AckReceipt,
    InboundMessage,
    MessageSemantics,
    SessionTarget,
)
from clawcodex_ext.services.im_gateway.router import SessionRouter
from clawcodex_ext.services.im_gateway.store import ReliabilityStore


def _make_message(origin: str, text: str, *, message_id: str | None = None) -> InboundMessage:
    return InboundMessage(
        origin=origin,
        text=text,
        message_id=message_id or f"mid-{origin}-{abs(hash(text))}",
        channel="wechat",
    )


def _make_dispatcher(
    tmp_path,
    *,
    repl_origin: str = "wechat:acct:user1",
    orchestrator_origin: str = "wechat:acct:user2",
    command_allowlists: CommandAllowlistConfig | None = None,
) -> tuple[InboundDispatcher, SessionRouter, list[InboundMessage]]:
    """Test helper for make dispatcher."""
    store = ReliabilityStore(tmp_path)
    binding = BindingPolicy()
    binding.bind(repl_origin, SessionTarget(session_id="repl-sess", host_type="repl"))
    binding.bind(
        orchestrator_origin,
        SessionTarget(session_id="orch-sess", host_type="orchestrator"),
    )
    router = SessionRouter(binding)

    pushed: list[InboundMessage] = []

    async def push_handler(message: InboundMessage) -> bool:
        pushed.append(message)
        return True

    dispatcher = InboundDispatcher(store, router, command_allowlists=command_allowlists)
    dispatcher.set_push_handler(push_handler)
    return dispatcher, router, pushed


@pytest.mark.asyncio
async def test_repl_blocked_command_not_pushed(tmp_path) -> None:
    """Verify repl blocked command not pushed."""
    dispatcher, router, pushed = _make_dispatcher(tmp_path)
    msg = _make_message("wechat:acct:user1", "/exit")

    receipt = await dispatcher.process(msg)

    assert len(pushed) == 0, "blocked command must not be pushed to REPL"
    assert receipt.layer == AckLayer.ACCEPTED
    assert "/exit" in (receipt.message or "")


@pytest.mark.asyncio
async def test_repl_blocked_command_with_args_not_pushed(tmp_path) -> None:
    """Verify repl blocked command with args not pushed."""
    dispatcher, router, pushed = _make_dispatcher(tmp_path)
    msg = _make_message("wechat:acct:user1", "/model gpt-4")

    receipt = await dispatcher.process(msg)

    assert len(pushed) == 0
    assert receipt.layer == AckLayer.ACCEPTED
    assert "/model" in (receipt.message or "")


@pytest.mark.asyncio
async def test_repl_allowed_command_pushed(tmp_path) -> None:
    """Verify repl allowed command pushed."""
    dispatcher, router, pushed = _make_dispatcher(tmp_path)
    msg = _make_message("wechat:acct:user1", "/clear")

    receipt = await dispatcher.process(msg)

    assert len(pushed) == 1
    assert pushed[0].text == "/clear"
    assert receipt.layer == AckLayer.ENQUEUED


@pytest.mark.asyncio
async def test_repl_allowed_command_with_args_pushed(tmp_path) -> None:
    """Verify repl allowed command with args pushed."""
    dispatcher, router, pushed = _make_dispatcher(tmp_path)
    msg = _make_message("wechat:acct:user1", "/goal finish the task")

    receipt = await dispatcher.process(msg)

    assert len(pushed) == 1
    assert receipt.layer == AckLayer.ENQUEUED


@pytest.mark.asyncio
async def test_repl_stop_command_pushed(tmp_path) -> None:
    """Verify repl stop command pushed."""
    dispatcher, router, pushed = _make_dispatcher(tmp_path)
    msg = _make_message("wechat:acct:user1", "/stop")

    receipt = await dispatcher.process(msg)

    assert len(pushed) == 1
    assert receipt.layer == AckLayer.ENQUEUED


@pytest.mark.asyncio
async def test_repl_plain_text_pushed(tmp_path) -> None:
    """Verify repl plain text pushed."""
    dispatcher, router, pushed = _make_dispatcher(tmp_path)
    msg = _make_message("wechat:acct:user1", "你好，请帮我写个函数")

    receipt = await dispatcher.process(msg)

    assert len(pushed) == 1
    assert receipt.layer == AckLayer.ENQUEUED


@pytest.mark.asyncio
async def test_repl_uses_channels_yaml_command_allowlist(tmp_path) -> None:
    dispatcher, router, pushed = _make_dispatcher(
        tmp_path,
        command_allowlists=CommandAllowlistConfig(
            repl=("/model",),
            orchestrator=(),
        ),
    )

    allowed_receipt = await dispatcher.process(
        _make_message("wechat:acct:user1", "/model gpt-5", message_id="repl-custom-allowed")
    )
    blocked_receipt = await dispatcher.process(
        _make_message("wechat:acct:user1", "/clear", message_id="repl-default-blocked")
    )

    assert [message.text for message in pushed] == ["/model gpt-5"]
    assert allowed_receipt.layer is AckLayer.ENQUEUED
    assert blocked_receipt.notify_user is True


@pytest.mark.asyncio
async def test_orchestrator_blocked_command_not_pushed(tmp_path) -> None:
    """Verify orchestrator blocked command not pushed."""
    dispatcher, router, pushed = _make_dispatcher(tmp_path)
    msg = _make_message("wechat:acct:user2", "/dashboard")

    receipt = await dispatcher.process(msg)

    assert len(pushed) == 0, "blocked command must not be pushed to orchestrator"
    assert receipt.layer == AckLayer.ACCEPTED
    assert receipt.notify_user is True
    assert receipt.message == "Command /dashboard is not supported."


@pytest.mark.asyncio
async def test_orchestrator_issue_takeover_not_pushed(tmp_path) -> None:
    """Verify orchestrator issue takeover not pushed."""
    dispatcher, router, pushed = _make_dispatcher(tmp_path)
    msg = _make_message("wechat:acct:user2", "/issue takeover --id AGENTSDK-15")

    receipt = await dispatcher.process(msg)

    assert len(pushed) == 0
    assert receipt.layer == AckLayer.ACCEPTED
    assert receipt.notify_user is True
    assert receipt.message == "Command /issue takeover is not supported."


@pytest.mark.asyncio
async def test_orchestrator_allowed_issue_command_pushed(tmp_path) -> None:
    """Verify orchestrator allowed issue command pushed."""
    dispatcher, router, pushed = _make_dispatcher(tmp_path)
    msg = _make_message("wechat:acct:user2", "/issue list")

    receipt = await dispatcher.process(msg)

    assert len(pushed) == 1
    assert pushed[0].text == "/issue list"
    assert pushed[0].semantic is MessageSemantics.COMMAND
    assert receipt.layer == AckLayer.ENQUEUED


@pytest.mark.asyncio
async def test_orchestrator_allowed_server_status_pushed(tmp_path) -> None:
    """Verify orchestrator allowed server status pushed."""
    dispatcher, router, pushed = _make_dispatcher(tmp_path)
    msg = _make_message("wechat:acct:user2", "/server status")

    receipt = await dispatcher.process(msg)

    assert len(pushed) == 1
    assert pushed[0].text == "/server status"
    assert pushed[0].semantic is MessageSemantics.COMMAND
    assert receipt.layer == AckLayer.ENQUEUED


@pytest.mark.asyncio
async def test_orchestrator_uses_channels_yaml_command_allowlist(tmp_path) -> None:
    dispatcher, router, pushed = _make_dispatcher(
        tmp_path,
        command_allowlists=CommandAllowlistConfig(
            repl=(),
            orchestrator=("/issue takeover",),
        ),
    )

    allowed_receipt = await dispatcher.process(
        _make_message(
            "wechat:acct:user2",
            "/issue takeover --id AGENTSDK-15",
            message_id="orch-custom-allowed",
        )
    )
    blocked_receipt = await dispatcher.process(
        _make_message(
            "wechat:acct:user2",
            "/server status",
            message_id="orch-default-blocked",
        )
    )

    assert [message.text for message in pushed] == ["/issue takeover --id AGENTSDK-15"]
    assert allowed_receipt.layer is AckLayer.ENQUEUED
    assert blocked_receipt.notify_user is True


@pytest.mark.asyncio
async def test_repl_blocked_command_records_audit(tmp_path) -> None:
    """Verify repl blocked command records audit."""
    dispatcher, router, pushed = _make_dispatcher(tmp_path)
    msg = _make_message("wechat:acct:user1", "/exit")

    await dispatcher.process(msg)

    audit_entries = dispatcher._store.audit_entries()
    blocked_entries = [e for e in audit_entries if e.get("event_type") == "repl_command_blocked"]
    assert len(blocked_entries) == 1
    assert "/exit" in (blocked_entries[0].get("command") or "")


@pytest.mark.asyncio
async def test_orchestrator_blocked_command_records_audit(tmp_path) -> None:
    """Verify orchestrator blocked command records audit."""
    dispatcher, router, pushed = _make_dispatcher(tmp_path)
    msg = _make_message("wechat:acct:user2", "/server stop")

    await dispatcher.process(msg)

    audit_entries = dispatcher._store.audit_entries()
    blocked_entries = [e for e in audit_entries if e.get("event_type") == "orchestrator_command_blocked"]
    assert len(blocked_entries) == 1
    assert "/server stop" in (blocked_entries[0].get("command") or "")


@pytest.mark.asyncio
async def test_opt_in_plain_text_defers_idle_classification_to_peer(tmp_path) -> None:
    dispatcher, _router, pushed = _make_dispatcher(tmp_path)

    await dispatcher.process(_make_message("wechat:acct:user1", "plain text", message_id="m-idle"))

    assert pushed[0].semantic is None


@pytest.mark.asyncio
async def test_busy_origin_plain_text_is_pushed_as_follow_up(tmp_path) -> None:
    class _BusyStatus:
        def is_busy(self, origin: str) -> bool:
            return origin == "wechat:acct:user1"

        async def start(self, _message: InboundMessage) -> bool:
            return True

        async def complete(self, *_args, **_kwargs) -> bool:
            return True

    dispatcher, router, pushed = _make_dispatcher(tmp_path)
    dispatcher._processing_status = _BusyStatus()

    await dispatcher.process(_make_message("wechat:acct:user1", "steer this", message_id="m-busy"))

    assert pushed[0].semantic is MessageSemantics.FOLLOW_UP


@pytest.mark.asyncio
async def test_explicit_new_prompt_is_not_deferred_to_peer(tmp_path) -> None:
    dispatcher, _router, pushed = _make_dispatcher(tmp_path)
    message = _make_message("wechat:acct:user1", "start separately", message_id="m-explicit")
    message.raw = {"deliverAs": "newPrompt"}

    await dispatcher.process(message)

    assert pushed[0].semantic is MessageSemantics.NEW_PROMPT


@pytest.mark.asyncio
async def test_failed_delivery_releases_dedupe_reservation_for_retry(tmp_path) -> None:
    store = ReliabilityStore(tmp_path)
    binding = BindingPolicy()
    origin = "wechat:acct:user1"
    binding.bind(origin, SessionTarget(session_id="repl-sess", host_type="repl"))
    dispatcher = InboundDispatcher(store, SessionRouter(binding))
    attempts = 0

    async def push_handler(_message: InboundMessage) -> bool:
        nonlocal attempts
        attempts += 1
        return False

    dispatcher.set_push_handler(push_handler)
    message = _make_message(origin, "retry me", message_id="m-retry")

    await dispatcher.process(message)
    await dispatcher.process(message)

    assert attempts == 2
    assert store.is_duplicate("m-retry") is False


@pytest.mark.asyncio
async def test_handler_exception_releases_dedupe_reservation(tmp_path) -> None:
    store = ReliabilityStore(tmp_path)
    dispatcher = InboundDispatcher(store, SessionRouter(BindingPolicy()))
    attempts = 0

    async def handler(message: InboundMessage) -> AckReceipt:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise RuntimeError("temporary failure")
        return AckReceipt(message.message_id, AckLayer.PROCESSED, "done")

    dispatcher.set_handler(handler)
    message = _make_message("wechat:acct:user", "retry handler", message_id="m-handler")

    with pytest.raises(RuntimeError, match="temporary failure"):
        await dispatcher.process(message)
    receipt = await dispatcher.process(message)

    assert attempts == 2
    assert receipt.layer is AckLayer.PROCESSED
    assert store.is_duplicate("m-handler") is True


@pytest.mark.asyncio
async def test_concurrent_duplicate_is_rejected_while_first_is_in_flight(tmp_path) -> None:
    store = ReliabilityStore(tmp_path)
    binding = BindingPolicy()
    origin = "wechat:acct:user1"
    binding.bind(origin, SessionTarget(session_id="repl-sess", host_type="repl"))
    dispatcher = InboundDispatcher(store, SessionRouter(binding))
    entered = asyncio.Event()
    release = asyncio.Event()
    pushes = 0

    async def push_handler(_message: InboundMessage) -> bool:
        nonlocal pushes
        pushes += 1
        entered.set()
        await release.wait()
        return True

    dispatcher.set_push_handler(push_handler)
    message = _make_message(origin, "one delivery", message_id="m-concurrent")
    first = asyncio.create_task(dispatcher.process(message))
    await entered.wait()

    duplicate = await dispatcher.process(message)
    release.set()
    first_receipt = await first

    assert duplicate.message == "duplicate; skipped"
    assert first_receipt.layer is AckLayer.ENQUEUED
    assert pushes == 1


@pytest.mark.asyncio
async def test_push_timeout_cancels_handler_and_falls_back(tmp_path) -> None:
    store = ReliabilityStore(tmp_path)
    binding = BindingPolicy()
    origin = "wechat:acct:user1"
    binding.bind(origin, SessionTarget(session_id="repl-sess", host_type="repl"))
    dispatcher = InboundDispatcher(
        store,
        SessionRouter(binding),
        push_timeout_seconds=0.01,
    )
    cancelled = asyncio.Event()
    fallback_messages: list[InboundMessage] = []

    async def push_handler(_message: InboundMessage) -> bool:
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()

    async def fallback(message: InboundMessage) -> AckReceipt:
        fallback_messages.append(message)
        return AckReceipt(message.message_id, AckLayer.PROCESSED, "fallback")

    dispatcher.set_push_handler(push_handler)
    dispatcher.set_handler(fallback)

    receipt = await dispatcher.process(_make_message(origin, "timeout", message_id="m-timeout"))

    assert cancelled.is_set()
    assert fallback_messages
    assert receipt.layer is AckLayer.PROCESSED
