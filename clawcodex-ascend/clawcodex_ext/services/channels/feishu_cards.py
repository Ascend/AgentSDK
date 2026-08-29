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

"""Feishu interactive permission card helpers."""

from __future__ import annotations

# Sibling clawcodex_ext packages (im_gateway) are migrated in separate branches;
# suppress E0611 until the full series lands.
# pylint: disable=no-name-in-module

import secrets
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ApprovalPending:
    approval_id: str
    nonce: str
    origin: str
    chat_id: str
    allowed_user_open_id: str
    choices: frozenset[str]
    allow_choices: frozenset[str]
    expires_at: float


def build_permission_card(
    *,
    message: str,
    suggestion: str | None,
    options: list[dict[str, str]],
    approval_id: str,
    nonce: str,
) -> dict[str, Any]:
    elements: list[dict[str, Any]] = [
        {"tag": "div", "text": {"tag": "plain_text", "content": message}},
    ]
    if suggestion:
        elements.append({"tag": "div", "text": {"tag": "plain_text", "content": f"建议：{suggestion}"}})
    actions = []
    for option in options:
        value = str(option.get("value") or "")
        label = str(option.get("label") or value)
        actions.append(
            {
                "tag": "button",
                "text": {"tag": "plain_text", "content": label},
                "type": "primary" if value.lower() in {"y", "yes", "1"} else "default",
                "value": {
                    "clawcodex_action": "permission_approval",
                    "approval_id": approval_id,
                    "nonce": nonce,
                    "choice": value,
                },
            }
        )
    elements.append({"tag": "action", "actions": actions})
    return {
        "msg_type": "interactive",
        "content": {
            "config": {"wide_screen_mode": True},
            "header": {"title": {"tag": "plain_text", "content": "权限审批"}},
            "elements": elements,
        },
    }


def build_resolved_permission_card(
    *,
    choice: str,
    operator_open_id: str = "",
    allowed: bool | None = None,
) -> dict[str, Any]:
    if allowed is None:
        allowed = _legacy_choice_is_allowed(choice)
    status = "已允许" if allowed else "已拒绝"
    template = "green" if allowed else "red"
    actor = f"\n处理人: {operator_open_id}" if operator_open_id else ""
    return {
        "config": {"wide_screen_mode": True},
        "header": {
            "template": template,
            "title": {"tag": "plain_text", "content": f"权限审批{status}"},
        },
        "elements": [
            {
                "tag": "div",
                "text": {
                    "tag": "plain_text",
                    "content": f"{status}{actor}",
                },
            }
        ],
    }


class ApprovalCardManager:
    def __init__(
        self,
        *,
        clock: Callable[[], float] = time.time,
        token_ttl_seconds: int = 900,
    ) -> None:
        self._clock = clock
        self._token_ttl_seconds = token_ttl_seconds
        self.pending: dict[str, ApprovalPending] = {}
        self._seen_tokens: dict[str, float] = {}

    def create_pending(
        self,
        *,
        approval_id: str | None = None,
        nonce: str | None = None,
        origin: str,
        chat_id: str,
        allowed_user_open_id: str,
        choices: Iterable[str],
        ttl_seconds: int,
        allow_choices: Iterable[str] | None = None,
    ) -> ApprovalPending:
        approval_id = approval_id or secrets.token_urlsafe(12)
        nonce = nonce or secrets.token_urlsafe(8)
        choice_set = frozenset(str(choice) for choice in choices)
        allow_choice_set = (
            frozenset(str(choice) for choice in allow_choices)
            if allow_choices is not None
            else frozenset(choice for choice in choice_set if _legacy_choice_is_allowed(choice))
        )
        state = ApprovalPending(
            approval_id=approval_id,
            nonce=nonce,
            origin=origin,
            chat_id=chat_id,
            allowed_user_open_id=allowed_user_open_id,
            choices=choice_set,
            allow_choices=allow_choice_set,
            expires_at=self._clock() + ttl_seconds,
        )
        self.pending[approval_id] = state
        return state

    def resolve_action(self, payload: Any) -> Any | None:
        event = _get(payload, "event") or payload
        action = _get(event, "action")
        value = _mapping(_get(action, "value"))
        if value.get("clawcodex_action") != "permission_approval":
            return None
        approval_id = str(value.get("approval_id") or "")
        choice = str(value.get("choice") or "")
        nonce = str(value.get("nonce") or "")
        state = self.pending.get(approval_id)
        if state is None or self._clock() > state.expires_at:
            self.pending.pop(approval_id, None)
            return None
        if nonce != state.nonce or choice not in state.choices:
            return None
        operator_open_id = _operator_open_id(event)
        if state.allowed_user_open_id and operator_open_id != state.allowed_user_open_id:
            return None
        chat_id = _chat_id(event)
        if chat_id != state.chat_id:
            return None
        action_token = f"{approval_id}:{nonce}:{operator_open_id}:{choice}"
        self._purge_tokens()
        if action_token in self._seen_tokens:
            return None
        self._seen_tokens[action_token] = self._clock()
        self.pending.pop(approval_id, None)
        decision = "allow" if choice in state.allow_choices else "deny"
        from clawcodex_ext.services.im_gateway.models import InboundMessage, MessageSemantics

        return InboundMessage(
            origin=state.origin,
            text=choice,
            message_id=f"feishu-card:{approval_id}:{action_token}",
            channel="feishu",
            context_token=state.chat_id,
            from_user_id=operator_open_id,
            semantic=MessageSemantics.APPROVAL,
            semantic_tags=["approval"],
            raw={
                "deliverAs": "approval",
                "source": "feishu_card_action",
                "approval_id": approval_id,
                "choice": choice,
                "decision": decision,
            },
        )

    def _purge_tokens(self) -> None:
        now = self._clock()
        expired = [token for token, seen_at in self._seen_tokens.items() if now - seen_at > self._token_ttl_seconds]
        for token in expired:
            self._seen_tokens.pop(token, None)


def _legacy_choice_is_allowed(choice: Any) -> bool:
    """Compatibility fallback for approval payloads without explicit semantics."""
    return str(choice).strip().lower() in {
        "y",
        "yes",
        "1",
        "allow",
        "allowed",
        "e",
        "enable",
        "s",
        "session",
    }


def _operator_open_id(event: Any) -> str:
    operator = _get(event, "operator") or _get(event, "user")
    operator_id = _get(operator, "operator_id") or _get(operator, "user_id")
    return str(_get(operator, "open_id") or _get(operator_id, "open_id") or "")


def _chat_id(event: Any) -> str:
    context = _get(event, "context")
    return str(
        _get(context, "open_chat_id")
        or _get(context, "chat_id")
        or _get(event, "open_chat_id")
        or _get(event, "chat_id")
        or ""
    )


def _get(obj: Any, key: str, default: Any = None) -> Any:
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _mapping(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


__all__ = [
    "ApprovalCardManager",
    "ApprovalPending",
    "build_permission_card",
    "build_resolved_permission_card",
]
