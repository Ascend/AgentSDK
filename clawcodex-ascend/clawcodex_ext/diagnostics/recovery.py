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

"""Layer-3 auto-recovery policy (F-108 §十八 P108-G).

Catalogues the five recovery paths the freeze-detection plan
promises, and routes them to the existing AbortController
harness. Each path is named in the plan's recovery table::

    Permission 弹窗超时 → auto-deny → 继续 agent loop | 无感知
    AskUser      超时    → 空 dict  → 继续 agent loop | 模型可能重试
    单 LLM turn 超时    → abort    → 下一 turn        | 短暂提示
    工具执行   超时    → abort    → agent 继续       | 工具超时提示
    Agent loop 总超时  → abort    → SessionComplete  | 完整的结果输出

The actual mechanisms are wired by:

* :mod:`clawcodex_ext.tui.agent_bridge` — Permission / AskUser
  auto-deny (``done.wait(timeout=…)`` → fall back).
* :mod:`clawcodex_ext.query.agent_loop_compat` — per-turn
  ``asyncio.wait_for(_drain_turn, timeout=turn_timeout_s)`` that
  trips ``AbortController`` on expiry.
* :mod:`extensions.api.query` — ``ToolGapWatchdog`` + outer
  ``agent_loop_timeout_s`` budget.
* :mod:`clawcodex_ext.diagnostics.freeze_detector` — Layer-1
  watchdog that only DUMPS (never aborts), so a frozen run gets a
  stack trace even if the other layers don't fire.

This module exists to:
1. Provide a single ``recovery_actions()`` helper that callers
   integrate with (so tests can assert coverage), and
2. Document the policy so future contributions don't drift.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass


class RecoveryAction(str, enum.Enum):
    """The five auto-recovery paths the plan commits to."""

    PERMISSION_AUTO_DENY = "permission_auto_deny"
    ASK_USER_EMPTY = "ask_user_empty_answers"
    LLM_TURN_TIMEOUT = "llm_turn_timeout_abort"
    TOOL_TIMEOUT = "tool_timeout_abort"
    AGENT_LOOP_TIMEOUT = "agent_loop_timeout_abort"


@dataclass(frozen=True)
class RecoverySpec:
    """Documentation record for one recovery path.

    Used by tests (``test_recovery_strategies.py``) to assert that
    the plan's commitments remain intact after refactors.
    """

    action: RecoveryAction
    user_perception: str
    mechanism: str
    integration_point: str


_RECOVERY_TABLE: tuple[RecoverySpec, ...] = (
    RecoverySpec(
        action=RecoveryAction.PERMISSION_AUTO_DENY,
        user_perception="无",
        mechanism="bound done.wait(timeout=permission_timeout_s); outcome=deny",
        integration_point="clawcodex_ext/tui/agent_bridge.py:_permission_handler",
    ),
    RecoverySpec(
        action=RecoveryAction.ASK_USER_EMPTY,
        user_perception="模型可能重试",
        mechanism="bound done.wait(timeout=permission_timeout_s); answers={}",
        integration_point="clawcodex_ext/tui/agent_bridge.py:_ask_user_handler",
    ),
    RecoverySpec(
        action=RecoveryAction.LLM_TURN_TIMEOUT,
        user_perception="短暂提示",
        mechanism="asyncio.wait_for(_drain_turn, timeout=turn_timeout_s); abort_controller.abort",
        integration_point="clawcodex_ext/query/agent_loop_compat.py:run_query_as_agent_loop",
    ),
    RecoverySpec(
        action=RecoveryAction.TOOL_TIMEOUT,
        user_perception="工具超时提示",
        mechanism="ToolGapWatchdog.observe_tool_use/tick; AbortController.abort",
        integration_point="extensions/api/query.py:QueryRunner.stream",
    ),
    RecoverySpec(
        action=RecoveryAction.AGENT_LOOP_TIMEOUT,
        user_perception="完整的结果输出",
        mechanism="polling loop checks agent_loop_timeout_s; SessionComplete(exit_code=124)",
        integration_point="extensions/api/query.py:QueryRunner.stream",
    ),
)


def recovery_actions() -> tuple[RecoverySpec, ...]:
    """Return the documented recovery table.

    Used by tests to assert the policy is intact after refactors.
    Production code generally doesn't need this — the mechanisms
    are invoked directly via the AbortController.
    """
    return _RECOVERY_TABLE


def describe(action: RecoveryAction) -> RecoverySpec:
    """Look up one spec by enum. Raises ``KeyError`` on miss."""
    for spec in _RECOVERY_TABLE:
        if spec.action is action:
            return spec
    raise KeyError(action)


# Public re-exports for callers that only want policy-level types.
__all__ = [
    "RecoveryAction",
    "RecoverySpec",
    "describe",
    "recovery_actions",
]
