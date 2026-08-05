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

"""Tests for gateway core components: store, binding, router, gate."""

from __future__ import annotations

import logging
from types import SimpleNamespace

import pytest

from clawcodex_ext.services.channels.capabilities import (
    CapabilityDescriptor,
    CapabilityNotDeclaredError,
    ChannelAdapter,
    ChannelCapability,
    ChannelCapabilitySet,
)
from clawcodex_ext.services.channels.models import ChannelMessage
from clawcodex_ext.services.channels.registry import ChannelAdapterRegistry
from clawcodex_ext.services.channels.results import (
    ChannelHealth,
    ChannelSendResult,
    ValidationResult,
)
from clawcodex_ext.services.im_gateway.binding import BindingPolicy
from clawcodex_ext.services.im_gateway.capability_gate import CapabilityGate
from clawcodex_ext.services.im_gateway.config import ReliabilityConfig
from clawcodex_ext.services.im_gateway.models import (
    IM_DIRECT_ALL_ORIGIN,
    OriginKey,
    SessionTarget,
)
from clawcodex_ext.services.im_gateway.origin_utils import configured_channel_by_type
from clawcodex_ext.services.im_gateway.router import SessionRouter
from clawcodex_ext.services.im_gateway.store import ReliabilityStore


def test_configured_channel_warning_redacts_exception_text(caplog) -> None:
    class SecretConfig:
        @staticmethod
        def get_channel_by_type(_channel_type: str):
            raise RuntimeError("token=origin-secret")

    gateway = SimpleNamespace(config=SecretConfig())
    with caplog.at_level(logging.WARNING, logger="clawcodex_ext.services.im_gateway.origin_utils"):
        assert configured_channel_by_type(gateway, "wechat") is None

    assert "error_type=RuntimeError" in caplog.text
    assert "error_fingerprint=" in caplog.text
    assert "origin-secret" not in caplog.text


# -- fake adapter --------------------------------------------------------


class _FakeAdapter(ChannelAdapter):
    def __init__(
        self,
        name: str = "fake",
        caps: ChannelCapabilitySet | None = None,
        *,
        send_result: ChannelSendResult | None = None,
        supports_markdown: bool = False,
    ) -> None:
        self._name = name
        self._caps = caps or ChannelCapabilitySet.of(
            ChannelCapability.OUTBOUND_TEXT,
            descriptors={
                ChannelCapability.OUTBOUND_TEXT: CapabilityDescriptor(
                    ChannelCapability.OUTBOUND_TEXT,
                    supports_markdown=supports_markdown,
                )
            },
        )
        self._send_result = send_result
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
        if self._send_result is not None:
            return self._send_result
        return ChannelSendResult.success(self._name, provider_receipt=f"mid_{len(self.sends)}")


def _registry_with(*adapters: _FakeAdapter) -> ChannelAdapterRegistry:
    reg = ChannelAdapterRegistry()
    for a in adapters:
        reg.register(a)
    return reg


# -- store ---------------------------------------------------------------


def test_store_dedupe(tmp_path) -> None:
    s = ReliabilityStore(tmp_path, ReliabilityConfig(inbound_dedupe_ttl_seconds=600))
    assert s.check_and_record("k1", message_id="m1") is True
    assert s.check_and_record("k1", message_id="m1") is False
    assert s.is_duplicate("k1") is True


def test_store_check_and_record_honors_dedupe_ttl(tmp_path, monkeypatch) -> None:
    from clawcodex_ext.services.im_gateway import store as store_mod

    now = [1000.0]
    monkeypatch.setattr(store_mod.time, "time", lambda: now[0])
    s = ReliabilityStore(tmp_path, ReliabilityConfig(inbound_dedupe_ttl_seconds=1))
    assert s.check_and_record("k1", message_id="m1") is True
    assert s.check_and_record("k1", message_id="m1") is False
    now[0] += 2.0
    assert s.check_and_record("k1", message_id="m1") is True


def test_store_outbox_and_dead_letter(tmp_path) -> None:
    s = ReliabilityStore(tmp_path)
    s.append_outbox({"idempotency_key": "o1", "channel": "c", "status": "pending"})
    s.append_outbox({"idempotency_key": "o1", "channel": "c", "status": "delivered"})
    s.append_outbox({"idempotency_key": "o2", "channel": "c", "status": "pending"})
    pending = s.outbox_pending()
    assert {e["idempotency_key"] for e in pending} == {"o2"}
    s.append_dead_letter({"idempotency_key": "o2", "reason": "boom"})
    assert len(s.dead_letter_entries()) == 1
    s.append_outbox({"idempotency_key": "o2", "channel": "c", "status": "failed"})
    assert s.outbox_pending() == []


def test_store_context_tokens(tmp_path) -> None:
    s = ReliabilityStore(tmp_path)
    s.set_context_token("acct", "user1", "tok_abc")
    assert s.get_context_token("acct", "user1") == "tok_abc"
    s.set_context_token("acct", "user1", None)
    assert s.get_context_token("acct", "user1") is None


def test_store_audit(tmp_path) -> None:
    s = ReliabilityStore(tmp_path)
    s.audit("binding_override", origin="o1", session_id="s1")
    entries = s.audit_entries()
    assert len(entries) == 1
    assert entries[0]["event_type"] == "binding_override"


# -- binding + router ----------------------------------------------------


def test_binding_unique_target_and_override_audit() -> None:
    audits: list[tuple] = []
    bp = BindingPolicy(auditor=lambda action, entry, prev: audits.append((action, entry.origin)))
    o = OriginKey.wechat("default", "user_gz")
    bp.bind(o, SessionTarget("repl_main", "repl"))
    assert bp.is_opt_in(o)
    # override
    bp.bind(o, SessionTarget("run_xyz", "orchestrator"))
    assert bp.get(o).target.session_id == "run_xyz"
    actions = [a for a, _ in audits]
    assert "binding_created" in actions
    assert "binding_override" in actions


def test_binding_keeps_wechat_and_feishu_direct_targets_independent() -> None:
    policy = BindingPolicy()
    policy.bind("wechat:direct:acct:user", SessionTarget("wechat-session", "repl"))
    policy.bind("feishu:dm:app:user", SessionTarget("feishu-session", "orchestrator"))

    assert policy.get("wechat:direct:acct:user").target.session_id == "wechat-session"
    assert policy.get("feishu:dm:app:user").target.session_id == "feishu-session"


def test_binding_offline_and_terminate_mutate_matched_wildcard_key() -> None:
    policy = BindingPolicy()
    policy.bind("wechat:direct:*:*", SessionTarget("wildcard-session", "repl"))

    policy.mark_offline("wechat:direct:acct:user", session_id="wildcard-session")
    assert policy.get("wechat:direct:acct:user").connection_state == "offline"

    policy.terminate("wechat:direct:acct:user", session_id="wildcard-session")
    assert policy.get("wechat:direct:acct:user") is None


def test_binding_terminate_restores_default_route() -> None:
    bp = BindingPolicy()
    o = OriginKey.wechat("default", "user_gz")
    bp.bind(o, SessionTarget("repl_main", "repl"))
    assert bp.is_opt_in(o)
    bp.terminate(o)
    assert not bp.is_opt_in(o)


def test_router_default_when_no_binding() -> None:
    bp = BindingPolicy()
    router = SessionRouter(bp)
    o = OriginKey.wechat("default", "user_gz")
    target = router.route(o)
    assert target.host_type == "default"
    assert "im:default:" in target.session_id


def test_router_ignores_legacy_session_map(tmp_path) -> None:
    (tmp_path / "im_session_map.json").write_text(
        '{"wechat:direct:default:user_gz": {"session_id": "stale", "host_type": "repl"}}',
        encoding="utf-8",
    )
    router = SessionRouter(BindingPolicy())

    target = router.route(OriginKey.wechat("default", "user_gz"))

    assert target.host_type == "default"
    assert target.session_id == "im:default:wechat:direct:default:user_gz"


def test_router_opt_in_overrides_default() -> None:
    bp = BindingPolicy()
    o = OriginKey.wechat("default", "user_gz")
    bp.bind(o, SessionTarget("repl_main", "repl"))
    router = SessionRouter(bp)
    assert router.route(o).session_id == "repl_main"
    assert router.is_opt_in(o)


def test_router_wechat_direct_wildcard_binding_matches_any_private_sender() -> None:
    bp = BindingPolicy()
    bp.bind("wechat:direct:*:*", SessionTarget("repl_all_private", "repl"))
    router = SessionRouter(bp)

    assert router.route("wechat:direct:acct_a:user_1").session_id == "repl_all_private"
    assert router.route("wechat:direct:acct_b:user_2").session_id == "repl_all_private"
    assert router.is_opt_in("wechat:direct:acct_a:user_1")


def test_router_generic_direct_wildcard_binding_matches_wechat_and_feishu() -> None:
    bp = BindingPolicy()
    bp.bind(IM_DIRECT_ALL_ORIGIN, SessionTarget("repl_all_im", "repl"))
    router = SessionRouter(bp)

    assert router.route("wechat:direct:acct_a:user_1").session_id == "repl_all_im"
    assert router.route("feishu:dm:cli_app:ou_user").session_id == "repl_all_im"
    assert router.is_opt_in("feishu:dm:cli_app:ou_user")


def test_router_wechat_direct_wildcard_offline_applies_to_matching_private_sender() -> None:
    bp = BindingPolicy()
    bp.bind("wechat:direct:*:*", SessionTarget("repl_all_private", "repl"))
    bp.mark_offline("wechat:direct:*:*")
    router = SessionRouter(bp)

    assert router.is_offline("wechat:direct:acct_a:user_1")


# -- capability gate -----------------------------------------------------


def test_capability_gate_fail_closed_for_undeclared() -> None:
    adapter = _FakeAdapter(caps=ChannelCapabilitySet.of(ChannelCapability.OUTBOUND_TEXT))
    reg = _registry_with(adapter)
    gate = CapabilityGate(reg)
    gate.require_outbound("fake")  # ok
    with pytest.raises(CapabilityNotDeclaredError):
        gate.require_media("fake", ChannelCapability.MEDIA_IMAGE)
    with pytest.raises(CapabilityNotDeclaredError):
        gate.require_context_reply("fake")


def test_capability_gate_rejects_non_media_capability() -> None:
    reg = _registry_with(_FakeAdapter())
    gate = CapabilityGate(reg)
    with pytest.raises(ValueError):
        gate.require_media("fake", ChannelCapability.OUTBOUND_TEXT)
