#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# -------------------------------------------------------------------------
# This file is part of the AgentSDK project.
# Copyright (c) 2026 Huawei Technologies Co.,Ltd.
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

"""InboundRuntimeRouter — dispatch a classified message to a runtime target (P5).

Decides, given a semantic + the session binding, where the message goes:
  - ``newPrompt`` → default host agent (Gateway-hosted) or REPL/orchestrator opt-in
  - ``followUp`` → default handler (stub agent for unbound; opt-in hosts own queueing)
  - ``command`` → CommandRouter → existing entry (control socket / issue CLI / agent intent)
  - ``interrupt`` → ControlBridge → bridge interrupt / control verbs
  - ``contextOnly`` → operator hints / issue inject (no run trigger)
  - ``approval`` → bound wait-point (clarify/review/feedback)

The router is a pure decision object; execution is performed by the
caller (gateway inbound handler / opt-in client) which owns the handles.
This keeps the semantics layer free of Orchestrator/REPL imports.
"""

from __future__ import annotations

from dataclasses import dataclass

# pylint: disable=no-name-in-module  # services.im_gateway: pending patch migration
from clawcodex_ext.services.im_gateway.models import InboundMessage, MessageSemantics

from .command_router import CommandRoute, CommandRouter
from .control_bridge import ControlBridge, ControlTarget


@dataclass
class RoutingDecision:
    semantic: MessageSemantics
    target: str  # "host_agent" | "handler" | "control" | "context_only" | "approval" | "command"
    route: CommandRoute | None = None
    control: ControlTarget | None = None
    reject_reason: str | None = None


class InboundRuntimeRouter:
    def __init__(
        self,
        command_router: CommandRouter | None = None,
        control_bridge: ControlBridge | None = None,
    ) -> None:
        self._commands = command_router or CommandRouter()
        self._control = control_bridge or ControlBridge()

    def decide(
        self,
        message: InboundMessage,
        semantic: MessageSemantics,
        *,
        is_opt_in: bool = False,
        is_busy: bool = False,
    ) -> RoutingDecision:
        if semantic is MessageSemantics.FOLLOW_UP:
            return RoutingDecision(semantic, "handler")
        if semantic is MessageSemantics.CONTEXT_ONLY:
            return RoutingDecision(semantic, "context_only", control=self._control.context_only_target(None))
        if semantic is MessageSemantics.INTERRUPT:
            ctrl = self._control.resolve(semantic, None)
            return RoutingDecision(semantic, "control", control=ctrl)
        if semantic is MessageSemantics.APPROVAL:
            # approval requires a bound wait-point; without one the dispatcher rejects.
            return RoutingDecision(semantic, "approval")
        if semantic is MessageSemantics.COMMAND:
            route = self._commands.route(message)
            ctrl = self._control.resolve(semantic, route)
            return RoutingDecision(semantic, "command", route=route, control=ctrl)
        # newPrompt → host agent (default) or opt-in target
        return RoutingDecision(semantic, "host_agent")


__all__ = ["InboundRuntimeRouter", "RoutingDecision"]
