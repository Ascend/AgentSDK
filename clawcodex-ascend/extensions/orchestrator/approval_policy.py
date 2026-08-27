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
#
# Copyright (c) 2026 Clawd Codex Team
# SPDX-License-Identifier: MIT
# Source: https://github.com/agentforce314/clawcodex
# ClawCodex-derived portions remain licensed under the MIT License.
# See clawcodex-ascend/LICENSE.clawcodex.
"""Tool call approval policy system.

Port of INTEGRATION.md Section 5.2 — direct Python API for tool call
interception and policy evaluation, replacing Symphony's Codex JSON-RPC.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import logging
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# ToolCallEvent — the event object passed to policy.evaluate()
# ---------------------------------------------------------------------------


@dataclass
class ToolCallEvent:
    """Tool call event passed to the agent loop.

    Attaches approval state so the policy can mutate it in-place
    (no need to return a separate result object).
    """

    tool_name: str
    params: dict[str, Any] = field(default_factory=dict)
    tool_use_id: str | None = None
    _approved: bool | None = None
    _deny_reason: str | None = None

    def allow(self, reason: str = "") -> None:
        """Mark this tool call as approved and clear any denial state.

        ``reason`` is accepted for compatibility with existing policy
        implementations, but it must not be stored in ``_deny_reason``:
        downstream audit consumers treat every non-empty denial reason as a
        rejected tool call.
        """
        self._approved = True
        self._deny_reason = None

    def deny(self, reason: str) -> None:
        """Mark this tool call as denied with a reason."""
        self._approved = False
        self._deny_reason = reason

    @property
    def is_approved(self) -> bool | None:
        """True if allowed, False if denied, None if not yet evaluated."""
        return self._approved


# ---------------------------------------------------------------------------
# ApprovalPolicy interface
# ---------------------------------------------------------------------------


class ApprovalPolicy(ABC):
    """Abstract base for tool call approval policies."""

    @abstractmethod
    def evaluate(
        self,
        event: ToolCallEvent,
        session_context: dict[str, Any],
    ) -> bool:
        """Return True to approve the tool call, False to deny.

        Implementations should call event.allow() or event.deny() to
        record the decision on the event object itself.
        """
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Built-in policies
# ---------------------------------------------------------------------------


class NeverApprovalPolicy(ApprovalPolicy):
    """Auto-approve all tool calls — mirrors Symphony's approval_policy: never."""

    def evaluate(
        self,
        event: ToolCallEvent,
        session_context: dict[str, Any],
    ) -> bool:
        event.allow("policy=never")
        return True


class AskApprovalPolicy(ApprovalPolicy):
    """Deny all tool calls — user decision required.

    In autonomous mode this is typically not used, but provided
    for parity with Symphony's approval_policy: ask.
    """

    def evaluate(
        self,
        event: ToolCallEvent,
        session_context: dict[str, Any],
    ) -> bool:
        event.deny(reason="policy=ask (not supported in autonomous mode)")
        return False


class ApproveSafeOnlyPolicy(ApprovalPolicy):
    """Approve read-only tools (glob, grep, read, web_search, web_fetch).

    All other tools require explicit approval.
    """

    _SAFE_TOOLS: frozenset[str] = frozenset(
        {
            "glob",
            "grep",
            "read",
            "read_multiple_files",
            "web_search",
            "web_fetch",
            "toolsearch",
            "ask_user_question",
        }
    )

    def evaluate(
        self,
        event: ToolCallEvent,
        session_context: dict[str, Any],
    ) -> bool:
        if event.tool_name.lower() in self._SAFE_TOOLS:
            event.allow("policy=approve-safe-only")
            return True
        event.deny(reason=f"policy=approve-safe-only ({event.tool_name} not in safe list)")
        return False


# ---------------------------------------------------------------------------
# Policy registry
# ---------------------------------------------------------------------------


_APPROVAL_POLICY_MAP: dict[str | int, type[ApprovalPolicy]] = {
    "never": NeverApprovalPolicy,
    "ask": AskApprovalPolicy,
    "approve-safe-only": ApproveSafeOnlyPolicy,
}


def get_approval_policy(policy_name: str | dict[str, Any]) -> ApprovalPolicy:
    """Resolve policy name (or inline dict) to an ApprovalPolicy instance."""
    if isinstance(policy_name, dict):
        # Parse legacy inline configuration instead of silently turning
        # structured input into unrestricted auto-approval.
        policy_name = str(
            policy_name.get("approval_policy") or policy_name.get("name") or policy_name.get("policy") or "ask"
        )

    name = str(policy_name).strip().lower()
    policy_cls = _APPROVAL_POLICY_MAP.get(name)
    if policy_cls is None:
        logger.warning("Unknown approval policy %r; falling back to fail-closed 'ask'", policy_name)
        return AskApprovalPolicy()
    return policy_cls()


def build_approval_policy_map(
    sandbox_config: Any,
) -> dict[str | int, ApprovalPolicy]:
    """Build a map from policy names to instantiated policies.

    Mirrors INTEGRATION.md section 3.5 AgentRunner._approval_policy_map.
    """
    raw = getattr(sandbox_config, "approval_policy", "never") or "never"
    if isinstance(raw, dict):
        key = str(raw.get("approval_policy") or raw.get("name") or raw.get("policy") or "ask").strip().lower()
    else:
        key = raw
    return {key: get_approval_policy(raw)}
