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

"""Tests for the outbound dispatcher (fail-closed, markdown fallback, outbox, dead-letter)."""

from __future__ import annotations

import pytest

from clawcodex_ext.services.channels.capabilities import (
    CapabilityDescriptor,
    ChannelAdapter,
    ChannelCapability,
    ChannelCapabilitySet,
)
from clawcodex_ext.services.channels.models import ChannelMessage
from clawcodex_ext.services.channels.registry import ChannelAdapterRegistry
from clawcodex_ext.services.channels.results import (
    ChannelHealth,
    ChannelSendResult,
    ErrorCategory,
    SendStatus,
    ValidationResult,
)
from clawcodex_ext.services.im_gateway.capability_gate import CapabilityGate
from clawcodex_ext.services.im_gateway.config import GatewayConfig, ReliabilityConfig
from clawcodex_ext.services.im_gateway.models import OutboundMessage
from clawcodex_ext.services.im_gateway.outbound import OutboundDispatcher
from clawcodex_ext.services.im_gateway.store import ReliabilityStore


class _FakeOutAdapter(ChannelAdapter):
    def __init__(
        self,
        name: str = "fake",
        *,
        supports_markdown: bool = False,
        send_result: ChannelSendResult | None = None,
        results_seq: list[ChannelSendResult] | None = None,
        caps: ChannelCapabilitySet | None = None,
    ) -> None:
        self._name = name
        if caps is not None:
            self._caps = caps
        else:
            self._caps = ChannelCapabilitySet.of(
                ChannelCapability.OUTBOUND_TEXT,
                descriptors={
                    ChannelCapability.OUTBOUND_TEXT: CapabilityDescriptor(
                        ChannelCapability.OUTBOUND_TEXT,
                        supports_markdown=supports_markdown,
                    )
                },
            )
        self._send_result = send_result
        self._results_seq = list(results_seq) if results_seq else None
        self.sends: list[ChannelMessage] = []

    @property
    def channel_id(self) -> str:
        return self._name

    @property
    def capabilities(self) -> ChannelCapabilitySet:
        return self._caps

    def validate_config(self) -> ValidationResult:
        return ValidationResult.ok_result()

    async def health_check(self) -> ChannelHealth:
        return ChannelHealth(healthy=True, channel_id=self._name)

    async def send(self, message, *, target=None, context_token=None) -> ChannelSendResult:
        self.sends.append(message)
        if self._results_seq is not None:
            idx = len(self.sends) - 1
            if idx < len(self._results_seq):
                return self._results_seq[idx]
            return self._results_seq[-1]
        if self._send_result is not None:
            return self._send_result
        return ChannelSendResult.success(self._name, provider_receipt=f"mid_{len(self.sends)}")


async def _noop_sleep(_delay: float) -> None:
    return None


def _make_dispatcher(
    tmp_path, adapter: _FakeOutAdapter
) -> tuple[OutboundDispatcher, ReliabilityStore, ChannelAdapterRegistry]:
    reg = ChannelAdapterRegistry()
    reg.register(adapter)
    store = ReliabilityStore(tmp_path, ReliabilityConfig(markdown_fallback=True, long_message_threshold_chunks=4))
    config = GatewayConfig(state_dir=str(tmp_path), reliability=store._reliability)
    gate = CapabilityGate(reg)
    return OutboundDispatcher(reg, gate, store, config, sleep=_noop_sleep), store, reg


@pytest.mark.asyncio
async def test_send_success_records_outbox_and_receipt(tmp_path) -> None:
    adapter = _FakeOutAdapter("wechat-main", supports_markdown=False)
    disp, store, _ = _make_dispatcher(tmp_path, adapter)
    result = await disp.send(OutboundMessage(text="hello", channel="wechat-main"))
    assert result.ok is True
    assert result.provider_receipt is not None
    entries = store.outbox_entries()
    statuses = [e["status"] for e in entries]
    assert "pending" in statuses and "delivered" in statuses
    delivered = [e for e in entries if e["status"] == "delivered"]
    assert delivered[-1]["provider_receipt"] == result.provider_receipt


@pytest.mark.asyncio
async def test_send_fail_closed_when_no_outbound_capability(tmp_path) -> None:
    caps = ChannelCapabilitySet.of(ChannelCapability.MEDIA_IMAGE)
    adapter = _FakeOutAdapter("media-only", caps=caps)
    disp, store, _ = _make_dispatcher(tmp_path, adapter)
    result = await disp.send(OutboundMessage(text="hi", channel="media-only"))
    assert result.ok is False
    assert result.status is SendStatus.UNSUPPORTED
    assert any(e["status"] == "dead" for e in store.outbox_entries())


@pytest.mark.asyncio
async def test_send_strips_markdown_for_non_markdown_channel(tmp_path) -> None:
    adapter = _FakeOutAdapter("wechat-main", supports_markdown=False)
    disp, _, _ = _make_dispatcher(tmp_path, adapter)
    await disp.send(OutboundMessage(text="**bold** and `code`", channel="wechat-main", markdown=True))
    assert len(adapter.sends) == 1
    sent = adapter.sends[0]
    assert "**" not in sent.text
    assert "`" not in sent.text
    assert sent.markdown is False  # stripped -> markdown False


@pytest.mark.asyncio
async def test_send_keeps_markdown_for_markdown_channel(tmp_path) -> None:
    adapter = _FakeOutAdapter("slack-ops", supports_markdown=True)
    disp, _, _ = _make_dispatcher(tmp_path, adapter)
    await disp.send(OutboundMessage(text="**bold**", channel="slack-ops", markdown=True))
    assert adapter.sends[0].text == "**bold**"
    assert adapter.sends[0].markdown is True


@pytest.mark.asyncio
async def test_send_nonretryable_failure_goes_to_dead_letter(tmp_path) -> None:
    adapter = _FakeOutAdapter(
        "wechat-main",
        supports_markdown=False,
        send_result=ChannelSendResult.nonretryable_error("wechat-main", message="bad", category=ErrorCategory.AUTH),
    )
    disp, store, _ = _make_dispatcher(tmp_path, adapter)
    result = await disp.send(OutboundMessage(text="hi", channel="wechat-main"))
    assert result.ok is False
    assert len(store.dead_letter_entries()) == 1
    assert store.dead_letter_entries()[0]["error_category"] == "auth"


@pytest.mark.asyncio
async def test_send_retryable_exhausts_then_dead_letters(tmp_path) -> None:
    adapter = _FakeOutAdapter(
        "wechat-main",
        send_result=ChannelSendResult.retryable_error(
            "wechat-main", message="5xx", category=ErrorCategory.SERVER_ERROR
        ),
    )
    disp, store, _ = _make_dispatcher(tmp_path, adapter)
    result = await disp.send(OutboundMessage(text="hi", channel="wechat-main"))
    assert result.ok is False
    # default policy max_attempts=5 → 5 send calls then dead-letter
    assert len(adapter.sends) == 5
    assert len(store.dead_letter_entries()) == 1
    # retry_pending outbox entries recorded across attempts
    statuses = [e["status"] for e in store.outbox_entries()]
    assert "retry_pending" in statuses
    assert store.outbox_pending() == []


@pytest.mark.asyncio
async def test_wechat_rate_limit_uses_common_retry_policy(tmp_path) -> None:
    adapter = _FakeOutAdapter(
        "wechat",
        send_result=ChannelSendResult.rate_limited(
            "wechat",
            message="rate limited",
            raw={"retry_after_seconds": 10},
        ),
    )
    disp, store, _ = _make_dispatcher(tmp_path, adapter)

    result = await disp.send(OutboundMessage(text="hi", channel="wechat", target="u1"))

    assert result.ok is False
    assert result.error_category is ErrorCategory.RATE_LIMIT
    assert result.status is SendStatus.RATE_LIMITED
    assert len(adapter.sends) == 5
    assert len(store.dead_letter_entries()) == 1
    assert not any(e["status"] == "deferred" for e in store.outbox_entries())
    assert any(e["status"] == "failed" for e in store.outbox_entries())


@pytest.mark.asyncio
async def test_rate_limited_status_uses_channel_agnostic_retry(tmp_path) -> None:
    adapter = _FakeOutAdapter(
        "line-direct",
        send_result=ChannelSendResult(
            ok=False,
            status=SendStatus.RATE_LIMITED,
            channel_id="line-direct",
            error_category=ErrorCategory.RATE_LIMIT,
            message="platform rate limited",
            raw={"retry_after_seconds": 10},
        ),
    )
    disp, store, _ = _make_dispatcher(tmp_path, adapter)

    result = await disp.send(OutboundMessage(text="hi", channel="line-direct", target="u1"))

    assert result.ok is False
    assert result.status is SendStatus.RATE_LIMITED
    assert result.error_category is ErrorCategory.RATE_LIMIT
    assert len(adapter.sends) == 5
    assert len(store.dead_letter_entries()) == 1
    assert not any(e["status"] == "deferred" for e in store.outbox_entries())
    assert any(e["status"] == "failed" for e in store.outbox_entries())


@pytest.mark.asyncio
async def test_wechat_rate_limit_without_retry_after_still_retries(tmp_path) -> None:
    adapter = _FakeOutAdapter(
        "wechat",
        send_result=ChannelSendResult.rate_limited(
            "wechat",
            message="rate limited",
        ),
    )
    disp, _, _ = _make_dispatcher(tmp_path, adapter)

    result = await disp.send(OutboundMessage(text="hi", channel="wechat", target="u1"))

    assert result.ok is False
    assert result.status is SendStatus.RATE_LIMITED
    assert len(adapter.sends) == 5


@pytest.mark.asyncio
async def test_send_retryable_then_success_recovers(tmp_path) -> None:
    adapter = _FakeOutAdapter(
        "wechat-main",
        results_seq=[
            ChannelSendResult.retryable_error("wechat-main", message="5xx", category=ErrorCategory.SERVER_ERROR),
            ChannelSendResult.retryable_error("wechat-main", message="5xx", category=ErrorCategory.SERVER_ERROR),
            ChannelSendResult.success("wechat-main", provider_receipt="ok_mid"),
        ],
    )
    disp, store, _ = _make_dispatcher(tmp_path, adapter)
    result = await disp.send(OutboundMessage(text="hi", channel="wechat-main"))
    assert result.ok is True
    assert result.provider_receipt == "ok_mid"
    assert len(adapter.sends) == 3
    assert store.dead_letter_entries() == []


@pytest.mark.asyncio
async def test_broadcast_partial_failure_does_not_crash(tmp_path) -> None:
    a_ok = _FakeOutAdapter("a")
    a_bad = _FakeOutAdapter(
        "b",
        send_result=ChannelSendResult.nonretryable_error("b", message="x", category=ErrorCategory.AUTH),
    )
    reg = ChannelAdapterRegistry()
    reg.register(a_ok)
    reg.register(a_bad)
    store = ReliabilityStore(tmp_path)
    config = GatewayConfig(state_dir=str(tmp_path), reliability=ReliabilityConfig())
    disp = OutboundDispatcher(reg, CapabilityGate(reg), store, config)
    results = await disp.broadcast(OutboundMessage(text="hi", channel="a"))
    assert results["a"].ok is True
    assert results["b"].ok is False


@pytest.mark.asyncio
async def test_empty_text_is_rejected_without_reporting_success(tmp_path) -> None:
    adapter = _FakeOutAdapter("wechat")
    dispatcher, store, _ = _make_dispatcher(tmp_path, adapter)

    result = await dispatcher.send(OutboundMessage(text="", channel="wechat"))

    assert result.ok is False
    assert result.error_category is ErrorCategory.CLIENT_ERROR
    assert not adapter.sends
    assert any(entry.get("error") == "empty_message" for entry in store.outbox_entries())


@pytest.mark.asyncio
async def test_idempotency_key_survives_dispatcher_restart(tmp_path) -> None:
    first_adapter = _FakeOutAdapter("wechat")
    first, _, _ = _make_dispatcher(tmp_path, first_adapter)
    message = OutboundMessage(
        text="hello",
        channel="wechat",
        idempotency_key="stable-key",
    )
    first_result = await first.send(message)

    second_adapter = _FakeOutAdapter("wechat")
    second, _, _ = _make_dispatcher(tmp_path, second_adapter)
    second_result = await second.send(message)

    assert first_result.ok and second_result.ok
    assert len(first_adapter.sends) == 1
    assert not second_adapter.sends


@pytest.mark.asyncio
async def test_plain_text_fallback_preserves_title_and_metadata(tmp_path) -> None:
    adapter = _FakeOutAdapter(
        "wechat",
        results_seq=[
            ChannelSendResult.nonretryable_error("wechat", message="format", category=ErrorCategory.FORMAT),
            ChannelSendResult.success("wechat"),
        ],
    )
    dispatcher, _, _ = _make_dispatcher(tmp_path, adapter)

    result = await dispatcher.send(
        OutboundMessage(
            text="**hello**",
            channel="wechat",
            title="Important",
            metadata={"trace": "1"},
        )
    )

    assert result.ok
    assert len(adapter.sends) == 2
    assert adapter.sends[1].title == "Important"
    assert adapter.sends[1].metadata == {"trace": "1"}


@pytest.mark.asyncio
async def test_broadcast_derives_stable_per_channel_idempotency_keys(tmp_path) -> None:
    first = _FakeOutAdapter("a")
    second = _FakeOutAdapter("b")
    dispatcher, _, registry = _make_dispatcher(tmp_path, first)
    registry.register(second)
    message = OutboundMessage(text="hello", channel="a", idempotency_key="broadcast-key")

    await dispatcher.broadcast(message)
    await dispatcher.broadcast(message)

    assert len(first.sends) == 1
    assert len(second.sends) == 1


@pytest.mark.asyncio
async def test_send_unknown_channel_returns_not_found(tmp_path) -> None:
    adapter = _FakeOutAdapter("a")
    disp, store, _ = _make_dispatcher(tmp_path, adapter)
    result = await disp.send(OutboundMessage(text="hi", channel="missing"))
    assert result.ok is False
    assert result.error_category is ErrorCategory.NOT_FOUND
