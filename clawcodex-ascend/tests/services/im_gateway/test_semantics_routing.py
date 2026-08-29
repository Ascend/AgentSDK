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

"""End-to-end semantic routing acceptance tests (P5)."""

from __future__ import annotations

import pytest

from clawcodex_ext.services.im_gateway.binding import BindingPolicy
from clawcodex_ext.services.im_gateway.dispatcher import InboundDispatcher
from clawcodex_ext.services.im_gateway.models import (
    AckLayer,
    InboundMessage,
    MessageSemantics,
    OriginKey,
    SessionTarget,
)
from clawcodex_ext.services.im_gateway.router import SessionRouter
from clawcodex_ext.services.im_gateway.store import ReliabilityStore


def _make_dispatcher(tmp_path):
    store = ReliabilityStore(tmp_path)
    bp = BindingPolicy()
    router = SessionRouter(bp)
    return InboundDispatcher(store, router), store, bp, router


def _msg(text, raw=None, tags=None, mid="m1") -> InboundMessage:
    return InboundMessage(
        origin="wechat:direct:default:u",
        text=text,
        message_id=mid,
        channel="wechat-main",
        raw=raw,
        semantic_tags=tags or [],
    )


# NL plain text must NOT trigger interrupt/contextOnly ----------------


@pytest.mark.asyncio
async def test_plain_text_does_not_classify_as_interrupt_or_context_only(tmp_path) -> None:
    disp, _, _, _ = _make_dispatcher(tmp_path)
    message = _msg("停下当前任务", mid="m1")
    await disp.process(message)
    assert message.semantic is MessageSemantics.NEW_PROMPT


@pytest.mark.asyncio
async def test_natural_language_interrupt_intent_is_newprompt(tmp_path) -> None:
    disp, _, _, _ = _make_dispatcher(tmp_path)
    m = _msg("中断这个任务", mid="m1")
    await disp.process(m)
    assert m.semantic is MessageSemantics.NEW_PROMPT


# control verbs + bridge interrupt route to control -------------------


@pytest.mark.asyncio
async def test_control_verb_classified_as_command(tmp_path) -> None:
    disp, _, _, _ = _make_dispatcher(tmp_path)
    m = _msg("/pause AGENTSDK-15", mid="m1")
    await disp.process(m)
    assert m.semantic is MessageSemantics.COMMAND


@pytest.mark.asyncio
async def test_structured_interrupt_classified(tmp_path) -> None:
    disp, _, _, _ = _make_dispatcher(tmp_path)
    m = _msg("x", raw={"deliverAs": "interrupt"}, mid="m1")
    await disp.process(m)
    assert m.semantic is MessageSemantics.INTERRUPT


# busy ordinary text → follow-up classification + accepted ack --------


@pytest.mark.asyncio
async def test_busy_plain_text_classifies_as_followup(tmp_path) -> None:
    disp, _, _, router = _make_dispatcher(tmp_path)
    # bind an opt-in target so routing is deterministic
    o = OriginKey.wechat("default", "u")
    router._binding.bind(o, SessionTarget("repl_main", "repl"))
    # simulate busy: classify explicitly then process
    m = _msg("顺便更新注释", mid="m1")
    m.semantic = disp.classify(m, is_busy=True)
    assert m.semantic is MessageSemantics.FOLLOW_UP
    ack = await disp.process(m)
    assert ack.layer is AckLayer.ACCEPTED


# explicit follow-up reuses /agent follow-up --------------------------


@pytest.mark.asyncio
async def test_explicit_followup_command_is_command(tmp_path) -> None:
    disp, _, _, _ = _make_dispatcher(tmp_path)
    m = _msg("/agent follow-up note", mid="m1")
    await disp.process(m)
    assert m.semantic is MessageSemantics.COMMAND


# contextOnly via structured metadata --------------------------------


@pytest.mark.asyncio
async def test_context_only_only_via_metadata(tmp_path) -> None:
    disp, _, _, _ = _make_dispatcher(tmp_path)
    m = _msg("any text", raw={"deliverAs": "contextOnly"}, mid="m1")
    await disp.process(m)
    assert m.semantic is MessageSemantics.CONTEXT_ONLY
    # plain "context" wording is NOT contextOnly
    m2 = _msg("add context about X", mid="m2")
    await disp.process(m2)
    assert m2.semantic is MessageSemantics.NEW_PROMPT


# HostAgentManager -----------------------------------------------------


def test_host_agent_contract_and_claim() -> None:
    from extensions.im_gateway.host_agent import HostAgentContract, HostAgentManager

    contract = HostAgentContract()
    assert contract.session_id("wechat:direct:default:u") == "im:default:wechat:direct:default:u"
    mgr = HostAgentManager(contract)
    sid = mgr.claim("wechat:direct:default:u")
    assert mgr.is_hosted("wechat:direct:default:u")
    assert mgr.session_for("wechat:direct:default:u") == sid
    mgr.release("wechat:direct:default:u")
    assert not mgr.is_hosted("wechat:direct:default:u")


@pytest.mark.asyncio
async def test_host_agent_reply_routes_to_outbound(tmp_path) -> None:
    from extensions.im_gateway.host_agent import HostAgentManager
    from clawcodex_ext.services.im_gateway.models import OutboundMessage

    sent: list[OutboundMessage] = []

    class _Out:
        async def send(self, msg):
            sent.append(msg)

    mgr = HostAgentManager()
    await mgr.reply("wechat:direct:default:user_gz", "hello back", outbound=_Out())
    assert len(sent) == 1
    assert sent[0].channel == "wechat"
    assert sent[0].target == "user_gz"
