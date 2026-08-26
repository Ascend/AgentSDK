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

"""HostAgentManager — Gateway-hosted default auto session contract (P5).

When an origin has no opt-in REPL/orchestrator binding, the gateway
hosts its own agent session (hermes-style, zero user operation). This
module freezes the contract (cwd / provider-model / approval / tool
policy / transcript / stop-resume / reply-flow) and provides a
minimal skeleton. Full default-agent execution is high-risk and is
implemented incrementally; v1 ships the contract + a deterministic
session-id factory + the reply-to-OutboundDispatcher hook shape so
the dispatcher can route ``newPrompt`` to it without a live LLM.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class HostAgentContract:
    """Frozen contract for a default host-agent session.

    A default session is created lazily per origin and reused. v1 does
    not run a live agent loop inside the daemon by default — the
    contract documents what a future full implementation must provide so
    callers (the inbound runtime router) can target it safely.
    """

    cwd: str = "."
    provider: str = "anthropic"
    model: str = "claude-sonnet-4-6"
    approval_policy: str = "auto_deny"  # headless default: deny tools
    tool_bundle: str = "bare"
    transcript_dir: str = "~/.clawcodex/gateway/transcripts"
    max_turns: int = 50
    stop_on_interrupt: bool = True
    reply_channel: str = "im_outbound"  # replies flow to OutboundDispatcher

    def session_id(self, origin: str) -> str:
        return f"im:default:{origin}"


@dataclass
class HostAgentManager:
    """Owns the default-host session index + reply routing shape.

    The actual agent execution is deferred; v1 records that a default
    session was claimed for an origin and exposes the reply hook so the
    inbound handler can send a deterministic acknowledgment through the
    outbound dispatcher.
    """

    contract: HostAgentContract = field(default_factory=HostAgentContract)
    # WeChat channel name to route replies to. Defaults to the wizard's
    # single-instance ``wechat``; inject the configured name so renamed channels still
    # receive replies instead of silently routing to a non-existent channel.
    wechat_channel_name: str = "wechat"
    _sessions: dict[str, str] = field(default_factory=dict)  # origin -> session_id

    def claim(self, origin: str) -> str:
        sid = self._sessions.get(origin)
        if sid is None:
            sid = self.contract.session_id(origin)
            self._sessions[origin] = sid
        return sid

    def release(self, origin: str) -> None:
        self._sessions.pop(origin, None)

    def is_hosted(self, origin: str) -> bool:
        return origin in self._sessions

    def session_for(self, origin: str) -> str | None:
        return self._sessions.get(origin)

    async def reply(self, origin: str, text: str, *, outbound) -> None:
        """Send ``text`` back to the origin's channel via ``outbound``.

        ``outbound`` is the :class:`OutboundDispatcher`. v1 uses this to
        deliver the deterministic ack; a full impl would stream agent
        output here under the path-one "full output" policy.
        """
        from clawcodex_ext.services.im_gateway.models import OutboundMessage

        channel, target = self._route(origin)
        await outbound.send(OutboundMessage(text=text, channel=channel, target=target, markdown=True))

    def _route(self, origin: str) -> tuple[str, str]:
        """Derive (channel, target) from a wechat origin key."""
        # origin = wechat:direct:{account}:{user}
        parts = origin.split(":")
        if len(parts) >= 4 and parts[0] == "wechat":
            target = parts[3]
        else:
            target = ""
        return self.wechat_channel_name, target


__all__ = ["HostAgentContract", "HostAgentManager"]
