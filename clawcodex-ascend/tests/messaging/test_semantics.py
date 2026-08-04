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

"""Tests for six-class message semantics (P5)."""

from __future__ import annotations

import pytest

from clawcodex_ext.messaging.semantics import (
    CommandRouter,
    ControlBridge,
    InboundRuntimeRouter,
    MessageClassifier,
)
from clawcodex_ext.services.im_gateway.models import InboundMessage, MessageSemantics


def _msg(text: str, *, raw=None, tags=None, message_id="m1") -> InboundMessage:
    return InboundMessage(
        origin="wechat:direct:default:u",
        text=text,
        message_id=message_id,
        channel="wechat-main",
        raw=raw,
        semantic_tags=tags or [],
    )


# -- classifier --------------------------------------------------------


def test_classify_slash_agent_is_command() -> None:
    c = MessageClassifier()
    assert c.classify(_msg("/agent retry AGENTSDK-15")) is MessageSemantics.COMMAND
    assert c.classify(_msg("/agent follow-up note")) is MessageSemantics.COMMAND
    assert c.classify(_msg("/agent unblock")) is MessageSemantics.COMMAND


def test_classify_control_verb_is_command() -> None:
    c = MessageClassifier()
    for v in (
        "/pause",
        "/resume",
        "/stop",
        "/takeover",
        "/inject",
        "/clarify",
        "/review",
        "/feedback",
    ):
        assert c.classify(_msg(f"{v} something")) is MessageSemantics.COMMAND


def test_classify_orchestrator_cli_is_command() -> None:
    c = MessageClassifier()
    assert c.classify(_msg("/issue list")) is MessageSemantics.COMMAND
    assert c.classify(_msg("/issue show --id AGENTSDK-15")) is MessageSemantics.COMMAND
    assert c.classify(_msg("/issue feedback --id AGENTSDK-15 --approve")) is MessageSemantics.COMMAND
    assert (
        c.classify(_msg('/issue review --id AGENTSDK-15 --reject --feedback "needs tests"')) is MessageSemantics.COMMAND
    )
    assert c.classify(_msg("/issue retry --id AGENTSDK-15 --mode reset")) is MessageSemantics.COMMAND
    assert c.classify(_msg("/server status")) is MessageSemantics.COMMAND


def test_classify_orchestrator_takeover_is_not_supported_issue_cli() -> None:
    c = MessageClassifier()
    assert c.classify(_msg("/issue takeover --id AGENTSDK-15")) is MessageSemantics.NEW_PROMPT


def test_classify_busy_plain_text_is_followup() -> None:
    c = MessageClassifier()
    assert c.classify(_msg("顺便更新下注释"), is_busy=True) is MessageSemantics.FOLLOW_UP


def test_classify_idle_plain_text_is_newprompt() -> None:
    c = MessageClassifier()
    assert c.classify(_msg("写个新功能")) is MessageSemantics.NEW_PROMPT


def test_classify_deliver_as_structured_wins() -> None:
    c = MessageClassifier()
    # contextOnly via structured metadata — NOT natural language
    assert c.classify(_msg("any text", raw={"deliverAs": "contextOnly"})) is MessageSemantics.CONTEXT_ONLY
    assert c.classify(_msg("any text", raw={"deliverAs": "interrupt"})) is MessageSemantics.INTERRUPT
    assert c.classify(_msg("any text", raw={"deliverAs": "approval"})) is MessageSemantics.APPROVAL
    assert c.classify(_msg("any text", tags=["contextOnly"])) is MessageSemantics.CONTEXT_ONLY


def test_classify_does_not_guess_interrupt_from_plain_text() -> None:
    """Plain-language interrupt intent must NOT auto-classify as interrupt."""
    c = MessageClassifier()
    assert c.classify(_msg("停下当前任务")) is MessageSemantics.NEW_PROMPT
    assert c.classify(_msg("中断")) is MessageSemantics.NEW_PROMPT


def test_classify_approval_needs_bound_wait() -> None:
    c = MessageClassifier()
    # bare "yes" with pending wait but no approval tag → newPrompt
    assert c.classify(_msg("yes"), has_pending_wait=True) is MessageSemantics.NEW_PROMPT
    assert c.classify(_msg("yes", tags=["approval"]), has_pending_wait=True) is MessageSemantics.APPROVAL


# -- command router ----------------------------------------------------


def test_command_router_agent_intent_with_issue() -> None:
    r = CommandRouter().route(_msg("/agent retry AGENTSDK-15"))
    assert r is not None
    assert r.kind == "agent_intent"
    assert r.verb == "retry"
    assert r.issue_hint == "AGENTSDK-15"


def test_command_router_control_verb() -> None:
    r = CommandRouter().route(_msg("/pause AGENTSDK-15"))
    assert r is not None
    assert r.kind == "control_verb"
    assert r.verb == "pause"
    assert r.issue_hint == "AGENTSDK-15"


def test_command_router_no_issue_returns_none_hint() -> None:
    r = CommandRouter().route(_msg("/agent unblock"))
    assert r is not None
    assert r.issue_hint is None


def test_command_router_plain_text_returns_none() -> None:
    assert CommandRouter().route(_msg("hello")) is None


def test_command_router_orchestrator_issue_cli() -> None:
    r = CommandRouter().route(_msg("/issue list --status running"))
    assert r is not None
    assert r.kind == "orchestrator_cli"
    assert r.verb == "issue"
    assert r.argv == ("issue", "list", "--status", "running")


def test_command_router_orchestrator_server_status() -> None:
    r = CommandRouter().route(_msg("/server status --workflow ./workflow.md"))
    assert r is not None
    assert r.kind == "orchestrator_cli"
    assert r.verb == "server"
    assert r.argv == ("server", "status", "--workflow", "./workflow.md")


def test_command_router_orchestrator_issue_id_hint() -> None:
    r = CommandRouter().route(_msg("/issue stop --id AGENTSDK-15"))
    assert r is not None
    assert r.issue_hint == "AGENTSDK-15"


@pytest.mark.parametrize(
    ("text", "argv"),
    [
        (
            "/issue feedback --id AGENTSDK-15 --approve",
            ("issue", "feedback", "--id", "AGENTSDK-15", "--approve"),
        ),
        (
            '/issue review --id AGENTSDK-15 --reject --feedback "needs tests"',
            (
                "issue",
                "review",
                "--id",
                "AGENTSDK-15",
                "--reject",
                "--feedback",
                "needs tests",
            ),
        ),
        (
            "/issue retry --id AGENTSDK-15 --mode reset",
            ("issue", "retry", "--id", "AGENTSDK-15", "--mode", "reset"),
        ),
        (
            "/issue rebase --id AGENTSDK-15",
            ("issue", "rebase", "--id", "AGENTSDK-15"),
        ),
    ],
)
def test_command_router_orchestrator_lifecycle_issue_cli(text: str, argv: tuple[str, ...]) -> None:
    r = CommandRouter().route(_msg(text))
    assert r is not None
    assert r.kind == "orchestrator_cli"
    assert r.argv == argv
    assert r.issue_hint == "AGENTSDK-15"


def test_command_router_orchestrator_takeover_returns_none() -> None:
    assert CommandRouter().route(_msg("/issue takeover --id AGENTSDK-15")) is None


# -- control bridge ----------------------------------------------------


def test_control_bridge_pause_to_control_socket() -> None:
    from clawcodex_ext.messaging.semantics.command_router import CommandRoute

    route = CommandRoute(kind="control_verb", verb="pause", issue_hint="AGENTSDK-15")
    target = ControlBridge().resolve(MessageSemantics.COMMAND, route)
    assert target is not None
    assert target.surface == "control_socket"


def test_control_bridge_inject_to_issue_inject_not_control_socket() -> None:
    """inject must NOT route to the control-socket no-op."""
    from clawcodex_ext.messaging.semantics.command_router import CommandRoute

    route = CommandRoute(kind="control_verb", verb="inject", issue_hint="AGENTSDK-15")
    target = ControlBridge().resolve(MessageSemantics.COMMAND, route)
    assert target is not None
    assert target.surface == "issue_inject"


def test_control_bridge_interrupt_semantic_to_bridge() -> None:
    target = ControlBridge().resolve(MessageSemantics.INTERRUPT, None)
    assert target is not None
    assert target.surface == "bridge_interrupt"


def test_control_bridge_context_only_to_operator_hints() -> None:
    target = ControlBridge().context_only_target(None)
    assert target.surface == "operator_hints"
    assert target.verb == "contextOnly"


# -- runtime router ----------------------------------------------------


def test_runtime_router_followup_to_handler() -> None:
    d = InboundRuntimeRouter().decide(_msg("more"), MessageSemantics.FOLLOW_UP)
    assert d.target == "handler"


def test_runtime_router_newprompt_to_host_agent() -> None:
    d = InboundRuntimeRouter().decide(_msg("new task"), MessageSemantics.NEW_PROMPT)
    assert d.target == "host_agent"


def test_runtime_router_command_routes_to_control() -> None:
    d = InboundRuntimeRouter().decide(_msg("/pause AGENTSDK-15"), MessageSemantics.COMMAND)
    assert d.target == "command"
    assert d.route is not None and d.route.verb == "pause"
    assert d.control is not None and d.control.surface == "control_socket"


def test_runtime_router_interrupt_routes_to_control() -> None:
    d = InboundRuntimeRouter().decide(_msg("x", raw={"deliverAs": "interrupt"}), MessageSemantics.INTERRUPT)
    assert d.target == "control"
    assert d.control is not None and d.control.surface == "bridge_interrupt"


def test_runtime_router_context_only_to_context_only() -> None:
    d = InboundRuntimeRouter().decide(_msg("ctx", raw={"deliverAs": "contextOnly"}), MessageSemantics.CONTEXT_ONLY)
    assert d.target == "context_only"
    assert d.control is not None and d.control.surface == "operator_hints"


def test_runtime_router_approval_target() -> None:
    d = InboundRuntimeRouter().decide(_msg("yes", tags=["approval"]), MessageSemantics.APPROVAL)
    assert d.target == "approval"
