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

"""Bridge between an ``ask`` decision and an interactive surface.

C1 (components-folder parity) replaced the legacy 3-arg callback
``(tool_name, message, suggestion_str) -> (allowed, enable)`` with the
request/reply shape mirroring TS ``PermissionResult``: the surface
receives the full :class:`PermissionAskRequest` (tool input drives
per-tool previews; suggestions drive the "always allow" option) and
answers with a :class:`PermissionAskReply` (which may carry the chosen
"don't ask again" updates and deny feedback).

HOOKS-1 (hooks-folder parity) adds two behaviors at this — the single —
ask choke point (both live seams funnel here: the query-loop adapter,
``can_use_tool_adapter.py``, and ``registry.dispatch``):

* **PermissionRequest hooks** run BEFORE any interactive prompt — and
  before the no-handler fail-closed branch, so hook decisions work
  headless (half their point). Port of ``PermissionContext.runHooks``
  (typescript/src/hooks/toolPermission/PermissionContext.ts:216-263):
  first decisive hook wins — ``allow`` (optional updatedInput +
  updatedPermissions → chosen_updates) or ``deny`` (message; optional
  ``interrupt`` aborts the turn via the context's abort controller);
  no decision → the normal flow, unchanged. Hook failures are contained
  (logged; flow continues) — a broken hook must not brick every prompt.
* **Rejection texts the model can act on** — the TS constants verbatim
  (utils/messages.ts:214-221) with the main-agent vs subagent split from
  ``cancelAndAbort`` (PermissionContext.ts:154-173), keyed on
  ``ToolContext.agent_id`` (the ``toolUseContext.agentId`` analog, set by
  ``subagent_context``). ``withMemoryCorrectionHint`` is NOT ported
  (GrowthBook ``tengu_amber_prism``, default false — a no-op upstream too).
"""

from __future__ import annotations

import logging
from typing import Any

from .types import (
    HookDecisionReason,
    PermissionAllowDecision,
    PermissionAskDecision,
    PermissionAskHandler,
    PermissionAskRequest,
    PermissionDecision,
    PermissionDenyDecision,
    PermissionUpdate,
)

logger = logging.getLogger(__name__)


# Verbatim TS rejection texts (typescript/src/utils/messages.ts:214-221).
# These are MODEL-FACING: the instructive wording ("the new_string was NOT
# written", "STOP…" / "Try a different approach…") is what lets the model
# change course correctly after a denial.
REJECT_MESSAGE = (
    "The user doesn't want to proceed with this tool use. The tool use was "
    "rejected (eg. if it was a file edit, the new_string was NOT written to "
    "the file). STOP what you are doing and wait for the user to tell you "
    "how to proceed."
)
REJECT_MESSAGE_WITH_REASON_PREFIX = (
    "The user doesn't want to proceed with this tool use. The tool use was "
    "rejected (eg. if it was a file edit, the new_string was NOT written to "
    "the file). To tell you how to proceed, the user said:\n"
)
SUBAGENT_REJECT_MESSAGE = (
    "Permission for this tool use was denied. The tool use was rejected "
    "(eg. if it was a file edit, the new_string was NOT written to the "
    "file). Try a different approach or report the limitation to complete "
    "your task."
)
SUBAGENT_REJECT_MESSAGE_WITH_REASON_PREFIX = (
    "Permission for this tool use was denied. The tool use was rejected "
    "(eg. if it was a file edit, the new_string was NOT written to the "
    "file). The user said:\n"
)


class PermissionAskOutcome(tuple):
    """Two-tuple result with the legacy direct-decision attribute surface."""

    def __new__(
        cls,
        decision: PermissionDecision,
        updates: tuple[PermissionUpdate, ...],
    ) -> PermissionAskOutcome:
        return tuple.__new__(cls, (decision, updates))

    @property
    def behavior(self) -> str:
        return self[0].behavior

    def __getattr__(self, name: str) -> Any:
        return getattr(self[0], name)


def _rejection_message(feedback: str, *, is_subagent: bool) -> str:
    """The cancelAndAbort message matrix (PermissionContext.ts:154-167)."""
    if feedback:
        prefix = SUBAGENT_REJECT_MESSAGE_WITH_REASON_PREFIX if is_subagent else REJECT_MESSAGE_WITH_REASON_PREFIX
        return f"{prefix}{feedback}"
    return SUBAGENT_REJECT_MESSAGE if is_subagent else REJECT_MESSAGE


def _run_permission_request_hooks(
    tool_name: str,
    decision: PermissionAskDecision,
    tool_input: dict[str, Any] | None,
    context: Any,
    tool_use_id: str | None,
) -> tuple[PermissionDecision, tuple[PermissionUpdate, ...]] | None:
    """Consult PermissionRequest hooks; a decisive result resolves the ask.

    Returns ``None`` when no hook made a decision (or hooks are not
    configured / failed) — the caller continues the normal flow.
    """
    try:
        from src.hooks.hook_executor import (
            execute_permission_request_hooks,
            has_hook_for_event,
        )

        if not has_hook_for_event("PermissionRequest", context):
            return None

        from src.utils.async_bridge import run_coroutine_blocking

        from .updates import serialize_permission_update

        # Canonical wire JSON for hook stdin (critic MAJOR: __dict__ left
        # nested PermissionRuleValue objects to be repr-stringified —
        # unusable for hook authors and divergent from TS's
        # PermissionUpdate JSON).
        suggestions_json = [serialize_permission_update(s) for s in (decision.suggestions or ())] or None

        async def _collect() -> list[dict[str, Any]]:
            results: list[dict[str, Any]] = []
            async for item in execute_permission_request_hooks(
                tool_name,
                tool_use_id or "",
                tool_input or {},
                context,
                permission_suggestions=suggestions_json,
            ):
                results.append(item)
                # First decisive hook wins AND later hooks never execute —
                # exact TS runHooks parity (returns mid-generator,
                # abandoning the rest). Progress/attachment items carry no
                # permission_behavior and keep streaming until then.
                if item.get("permission_behavior") in ("allow", "deny"):
                    break
            return results

        results = run_coroutine_blocking(_collect(), thread_name=f"permission-request-hooks:{tool_name}")
    except Exception:
        logger.exception(
            "PermissionRequest hooks failed for %s; continuing to the normal flow",
            tool_name,
        )
        return None

    # First decisive result wins (runHooks loop semantics).
    for item in results:
        behavior = item.get("permission_behavior")
        if behavior == "allow":
            chosen: tuple[PermissionUpdate, ...] = ()
            raw_updates = item.get("updated_permissions")
            if isinstance(raw_updates, list):
                from .updates import deserialize_permission_update

                chosen = tuple(
                    u
                    for u in (deserialize_permission_update(d) for d in raw_updates if isinstance(d, dict))
                    if u is not None
                )
            return (
                PermissionAllowDecision(
                    behavior="allow",
                    updated_input=item.get("updated_input") or decision.updated_input,
                    decision_reason=HookDecisionReason(
                        hook_name="PermissionRequest",
                    ),
                ),
                chosen,
            )
        if behavior == "deny":
            message = item.get("hook_permission_decision_reason") or "Permission denied by hook"
            if item.get("interrupt"):
                abort_ctrl = getattr(context, "abort_controller", None)
                if abort_ctrl is not None:
                    try:
                        abort_ctrl.abort("permission_request_hook_interrupt")
                    except Exception:
                        logger.exception("PermissionRequest hook interrupt abort failed")
            return (
                PermissionDenyDecision(
                    behavior="deny",
                    message=message,
                    decision_reason=HookDecisionReason(
                        hook_name="PermissionRequest",
                        reason=message,
                    ),
                ),
                (),
            )
    return None


def handle_permission_ask(
    tool_name: str,
    decision: PermissionAskDecision,
    handler: PermissionAskHandler | None = None,
    tool_input: dict[str, Any] | None = None,
    context: Any = None,
    tool_use_id: str | None = None,
) -> tuple[PermissionDecision, tuple[PermissionUpdate, ...]]:
    """Resolve an ``ask`` decision through hooks, then ``handler``.

    Returns ``(final_decision, chosen_updates)``. ``chosen_updates`` are
    the "don't ask again" rules the user accepted (empty on deny / plain
    allow); the caller applies them to the live context and persists the
    persistable destinations — mirroring how TS applies
    ``updatedPermissions`` from the dialog's ``PermissionResult``.
    ``context``/``tool_use_id`` are optional for backward compatibility;
    call sites that pass them enable PermissionRequest hooks and the
    subagent-aware rejection texts.
    """

    if context is not None:
        hook_resolution = _run_permission_request_hooks(tool_name, decision, tool_input, context, tool_use_id)
        if hook_resolution is not None:
            return PermissionAskOutcome(*hook_resolution)

    if handler is None:
        return PermissionAskOutcome(
            PermissionDenyDecision(
                behavior="deny",
                message=decision.message or f"Permission required but no handler available for {tool_name}",
                decision_reason=decision.decision_reason,
            ),
            (),
        )

    request = PermissionAskRequest(
        tool_name=tool_name,
        message=decision.message or f"Tool '{tool_name}' requires permission",
        tool_input=tool_input,
        suggestions=tuple(decision.suggestions or ()),
        decision_reason=decision.decision_reason,
    )
    try:
        reply = handler(request)
    except TypeError:
        # Preserve direct callers using the original three-argument callback;
        # registry callers are already normalized by its adapter.
        allowed, _enable = handler(  # type: ignore[call-arg]
            tool_name,
            request.message,
            request.suggestions,
        )
        reply = type(
            "_LegacyPermissionReply",
            (),
            {
                "behavior": "allow" if allowed else "deny",
                "updated_input": None,
                "chosen_updates": (),
                "message": None,
            },
        )()

    if reply.behavior == "allow":
        return PermissionAskOutcome(
            PermissionAllowDecision(
                behavior="allow",
                updated_input=reply.updated_input or decision.updated_input,
                decision_reason=decision.decision_reason,
            ),
            tuple(reply.chosen_updates or ()),
        )

    feedback = (reply.message or "").strip()
    is_subagent = bool(getattr(context, "agent_id", None))
    message = _rejection_message(feedback, is_subagent=is_subagent)
    return PermissionAskOutcome(
        PermissionDenyDecision(
            behavior="deny",
            message=message,
            decision_reason=decision.decision_reason,
        ),
        (),
    )


__all__ = [
    "REJECT_MESSAGE",
    "REJECT_MESSAGE_WITH_REASON_PREFIX",
    "SUBAGENT_REJECT_MESSAGE",
    "SUBAGENT_REJECT_MESSAGE_WITH_REASON_PREFIX",
    "PermissionAskOutcome",
    "handle_permission_ask",
]
