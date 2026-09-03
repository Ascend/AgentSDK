#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# -------------------------------------------------------------------------
# This file is part of the AgentSDK project.
# Copyright (c) 2026 Clawd Codex Team
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

"""ClawcodexImChannel — concrete ``ImChannel`` + ``ImCommandRouter`` adapter.

Thin wrapper over ``OrchestratorGatewayClient`` so ``im_gateway_client``
does not lazy-import clawcodex_ext IM models / messaging / entrypoints.

``deliver`` / ``listen`` / ``close`` map to ``send_outbound`` /
``dispatch`` / teardown (``close`` is currently a no-op; orchestrator
owns the client lifetime). ``ImCommandRouter.dispatch`` forwards to
``OrchestratorGatewayClient.dispatch``. Inbound
``InboundMessage`` dataclasses become ``ImInbound``; outbound is the
reverse.
"""

from __future__ import annotations

from typing import Any, AsyncIterator

from extensions.orchestrator_runtime.protocols.im_channel import (
    ImChannel,
    ImCommandRouter,
    ImInbound,
    ImOutbound,
)


def _inbound_from_upstream(msg: Any) -> ImInbound:
    """Convert upstream ``InboundMessage`` → ``ImInbound``.

    Defensive attribute reads; missing fields fall back to empty defaults.
    """
    sem = getattr(msg, "semantic", None)
    sem_kind = getattr(sem, "kind", None) if sem is not None else None
    metadata: dict[str, Any] = {}
    if sem is not None:
        metadata["semantic_kind"] = sem_kind
        # Preserve any additional semantic fields the upstream defines.
        for attr in ("command", "args", "payload"):
            if hasattr(sem, attr):
                metadata[f"semantic_{attr}"] = getattr(sem, attr)
    return ImInbound(
        origin=getattr(msg, "origin", "") or "",
        text=getattr(msg, "text", "") or "",
        issue_id=getattr(msg, "issue_id", None),
        thread_id=getattr(msg, "thread_id", None),
        sender_id=getattr(msg, "sender_id", None),
        metadata=metadata,
    )


def _outbound_to_upstream(out: ImOutbound) -> dict[str, Any]:
    """Convert ``ImOutbound`` → dict compatible with upstream ``send_outbound``."""
    return {
        "origin": out.origin,
        "text": out.text,
        "issue_id": out.issue_id,
        "card": out.card or {},
    }


class ClawcodexImChannel(ImChannel, ImCommandRouter):
    """Adapter over ``OrchestratorGatewayClient``.

    Constructed lazily inside ``OrchestratorGatewayClient.__init__`` via the
    new ``im_channel=`` kw arg; does not own its gateway lifetime.
    """

    def __init__(self, gateway: Any) -> None:
        self._gateway = gateway

    @property
    def channel_id(self) -> str:
        return getattr(self._gateway, "origin", "") or "clawcodex-im"

    async def deliver(self, message: ImOutbound) -> None:
        # ``OrchestratorGatewayClient.send_outbound`` is the canonical writer;
        # it handles queueing + retry + backoff internally.
        send = getattr(self._gateway, "send_outbound", None)
        if send is None:
            return
        send(_outbound_to_upstream(message))

    async def listen(self) -> AsyncIterator[ImInbound]:
        # ``OrchestratorGatewayClient`` is a *server-pushed* model via the
        # ``_on_pushed_deliver`` callback. ``listen()`` here is a no-op async
        # iterator that yields nothing — actual inbound delivery happens via
        # the callback path the gateway registered against ``ipc_client``.
        if False:  # pragma: no cover — explicit no-op  # pylint: disable=W0125
            yield ImInbound(origin="", text="")
        return

    async def close(self) -> None:
        # No explicit close on OrchestratorGatewayClient — orchestrator owns
        # the gateway lifecycle. Keep as no-op for Protocol conformance.
        return None

    async def dispatch(self, inbound: ImInbound) -> ImOutbound | None:
        """ImCommandRouter.dispatch — forward to gateway ``dispatch()``."""
        # Convert ImInbound → upstream InboundMessage-shaped dict; the gateway
        # dispatch() expects an InboundMessage instance, but its internal
        # dispatch path reads attributes (origin / text / semantic). We pass
        # a lightweight shim that satisfies duck-typing.
        sem_kind = inbound.metadata.get("semantic_kind") if inbound.metadata else None
        sem_obj = _SemanticShim(kind=sem_kind) if sem_kind else None
        upstream_msg = _InboundShim(
            origin=inbound.origin,
            text=inbound.text,
            issue_id=inbound.issue_id,
            thread_id=inbound.thread_id,
            sender_id=inbound.sender_id,
            semantic=sem_obj,
        )
        result = self._gateway.dispatch(upstream_msg)
        if result is None:
            return None
        if isinstance(result, ImOutbound):
            return result
        # Convert upstream return → ImOutbound
        return ImOutbound(
            origin=inbound.origin,
            text=getattr(result, "text", "") or "",
            issue_id=inbound.issue_id,
            card=getattr(result, "card", None),
        )


# ─── Lightweight shims (duck-typed to upstream ``InboundMessage`` /
#     ``MessageSemantics``; defined locally to avoid importing upstream).


class _SemanticShim:
    def __init__(self, kind: str | None) -> None:
        self.kind = kind


class _InboundShim:
    def __init__(
        self,
        *,
        origin: str,
        text: str,
        issue_id: str | None,
        thread_id: str | None,
        sender_id: str | None,
        semantic: Any | None,
    ) -> None:
        self.origin = origin
        self.text = text
        self.issue_id = issue_id
        self.thread_id = thread_id
        self.sender_id = sender_id
        self.semantic = semantic


__all__ = ["ClawcodexImChannel"]
