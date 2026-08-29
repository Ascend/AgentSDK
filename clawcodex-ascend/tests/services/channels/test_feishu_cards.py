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

"""Feishu permission approval card tests."""
# pylint: disable=no-name-in-module

from __future__ import annotations

from types import SimpleNamespace

from clawcodex_ext.services.channels.feishu_cards import (
    ApprovalCardManager,
    build_permission_card,
    build_resolved_permission_card,
)
from clawcodex_ext.services.im_gateway.models import MessageSemantics


def _event(*, user="ou_allowed", chat="oc_chat", approval_id="ap1", nonce="n1", choice="y"):
    return {
        "event": {
            "operator": {"open_id": user},
            "context": {"open_chat_id": chat},
            "action": {
                "value": {
                    "clawcodex_action": "permission_approval",
                    "approval_id": approval_id,
                    "nonce": nonce,
                    "choice": choice,
                }
            },
        }
    }


def test_permission_metadata_renders_feishu_card() -> None:
    payload = build_permission_card(
        message="ClawCodex wants to use Bash.",
        suggestion="Check the command first.",
        options=[
            {"value": "y", "label": "允许"},
            {"value": "n", "label": "拒绝"},
        ],
        approval_id="ap1",
        nonce="n1",
    )

    assert payload["msg_type"] == "interactive"
    card = payload["content"]
    assert card["header"]["title"]["content"] == "权限审批"
    assert card["elements"][0]["text"]["content"] == "ClawCodex wants to use Bash."
    actions = card["elements"][-1]["actions"]
    assert actions[0]["text"]["content"] == "允许"
    assert actions[0]["value"]["approval_id"] == "ap1"
    assert actions[0]["value"]["choice"] == "y"


def test_resolved_permission_card_removes_action_buttons() -> None:
    card = build_resolved_permission_card(choice="y", operator_open_id="ou_allowed")

    assert card["header"]["template"] == "green"
    assert "已允许" in card["header"]["title"]["content"]
    assert all(element.get("tag") != "action" for element in card["elements"])


def test_resolved_permission_card_treats_session_choice_as_allowed() -> None:
    card = build_resolved_permission_card(choice="s", allowed=True)

    assert card["header"]["template"] == "green"
    assert "已允许" in card["header"]["title"]["content"]


def test_card_click_from_allowed_user_emits_approval_inbound() -> None:
    manager = ApprovalCardManager(clock=lambda: 100.0)
    manager.create_pending(
        approval_id="ap1",
        nonce="n1",
        origin="feishu:dm:cli_app:ou_allowed",
        chat_id="oc_chat",
        allowed_user_open_id="ou_allowed",
        choices={"y", "n"},
        ttl_seconds=60,
    )

    inbound = manager.resolve_action(_event())

    assert inbound is not None
    assert inbound.origin == "feishu:dm:cli_app:ou_allowed"
    assert inbound.text == "y"
    assert inbound.context_token == "oc_chat"
    assert inbound.from_user_id == "ou_allowed"
    assert inbound.semantic is MessageSemantics.APPROVAL
    assert inbound.semantic_tags == ["approval"]
    assert inbound.raw["source"] == "feishu_card_action"
    assert "ap1" not in manager.pending


def test_card_click_session_choice_carries_explicit_allow_decision() -> None:
    manager = ApprovalCardManager(clock=lambda: 100.0)
    manager.create_pending(
        approval_id="ap1",
        nonce="n1",
        origin="feishu:dm:cli_app:ou_allowed",
        chat_id="oc_chat",
        allowed_user_open_id="ou_allowed",
        choices={"y", "s", "n"},
        allow_choices={"y", "s"},
        ttl_seconds=60,
    )

    inbound = manager.resolve_action(_event(choice="s"))

    assert inbound is not None
    assert inbound.text == "s"
    assert inbound.raw["decision"] == "allow"


def test_card_click_from_sdk_model_object_emits_approval_inbound() -> None:
    """The official SDK passes P2CardActionTrigger-style objects, not dicts."""
    manager = ApprovalCardManager(clock=lambda: 100.0)
    manager.create_pending(
        approval_id="ap1",
        nonce="n1",
        origin="feishu:dm:cli_app:ou_allowed",
        chat_id="oc_chat",
        allowed_user_open_id="ou_allowed",
        choices={"y", "n"},
        ttl_seconds=60,
    )
    payload = SimpleNamespace(
        event=SimpleNamespace(
            operator=SimpleNamespace(open_id="ou_allowed"),
            context=SimpleNamespace(open_chat_id="oc_chat"),
            action=SimpleNamespace(
                value={
                    "clawcodex_action": "permission_approval",
                    "approval_id": "ap1",
                    "nonce": "n1",
                    "choice": "y",
                }
            ),
        )
    )

    inbound = manager.resolve_action(payload)

    assert inbound is not None
    assert inbound.text == "y"
    assert inbound.context_token == "oc_chat"
    assert inbound.from_user_id == "ou_allowed"


def test_card_click_from_other_user_is_rejected() -> None:
    manager = ApprovalCardManager(clock=lambda: 100.0)
    manager.create_pending(
        approval_id="ap1",
        nonce="n1",
        origin="feishu:dm:cli_app:ou_allowed",
        chat_id="oc_chat",
        allowed_user_open_id="ou_allowed",
        choices={"y", "n"},
        ttl_seconds=60,
    )

    assert manager.resolve_action(_event(user="ou_other")) is None


def test_card_click_wrong_chat_is_rejected() -> None:
    manager = ApprovalCardManager(clock=lambda: 100.0)
    manager.create_pending(
        approval_id="ap1",
        nonce="n1",
        origin="feishu:dm:cli_app:ou_allowed",
        chat_id="oc_chat",
        allowed_user_open_id="ou_allowed",
        choices={"y", "n"},
        ttl_seconds=60,
    )

    assert manager.resolve_action(_event(chat="oc_other")) is None


def test_card_click_duplicate_token_is_ignored() -> None:
    manager = ApprovalCardManager(clock=lambda: 100.0)
    manager.create_pending(
        approval_id="ap1",
        nonce="n1",
        origin="feishu:dm:cli_app:ou_allowed",
        chat_id="oc_chat",
        allowed_user_open_id="ou_allowed",
        choices={"y", "n"},
        ttl_seconds=60,
    )

    assert manager.resolve_action(_event()) is not None
    assert manager.resolve_action(_event()) is None


def test_card_click_expired_approval_is_ignored() -> None:
    now = [100.0]
    manager = ApprovalCardManager(clock=lambda: now[0])
    manager.create_pending(
        approval_id="ap1",
        nonce="n1",
        origin="feishu:dm:cli_app:ou_allowed",
        chat_id="oc_chat",
        allowed_user_open_id="ou_allowed",
        choices={"y", "n"},
        ttl_seconds=10,
    )
    now[0] = 111.0

    assert manager.resolve_action(_event()) is None


def test_card_click_empty_allowlist_accepts_any_user_in_same_chat() -> None:
    """V1 opens to all p2p users: empty allowlist relies on chat_id match."""
    manager = ApprovalCardManager(clock=lambda: 100.0)
    manager.create_pending(
        approval_id="ap1",
        nonce="n1",
        origin="feishu:dm:cli_app:ou_anyone",
        chat_id="oc_chat",
        allowed_user_open_id="",
        choices={"y", "n"},
        ttl_seconds=60,
    )

    inbound = manager.resolve_action(_event(user="ou_anyone"))

    assert inbound is not None
    assert inbound.from_user_id == "ou_anyone"


def test_card_click_empty_allowlist_still_rejects_wrong_chat() -> None:
    manager = ApprovalCardManager(clock=lambda: 100.0)
    manager.create_pending(
        approval_id="ap1",
        nonce="n1",
        origin="feishu:dm:cli_app:ou_anyone",
        chat_id="oc_chat",
        allowed_user_open_id="",
        choices={"y", "n"},
        ttl_seconds=60,
    )

    assert manager.resolve_action(_event(user="ou_anyone", chat="oc_other")) is None
