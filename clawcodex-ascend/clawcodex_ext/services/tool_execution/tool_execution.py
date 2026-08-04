#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# -------------------------------------------------------------------------
#  This file is part of the AgentSDK project.
# Copyright (c) 2026 Huawei Technologies Co.,Ltd.
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
# This file is derived from Clawd Codex (https://github.com/agentforce314/clawcodex),
# which is licensed under the MIT License.
# Copyright (c) 2026 Clawd Codex Team
# -------------------------------------------------------------------------
# -------------------------------------------------------------------------

"""Tool execution — mirrors TypeScript toolExecution.ts.

Core function: run_tool_use() — async generator yielding MessageUpdateLazy.
Handles tool lookup, input validation, permission checks, pre/post hooks,
tool execution with progress, error handling, and result mapping.
"""

from __future__ import annotations

import logging
import math
import time
from collections.abc import AsyncGenerator, Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from src.types.messages import (
    CANCEL_MESSAGE,
    INTERRUPT_MESSAGE_FOR_TOOL_USE,
    AssistantMessage,
    Message,
    create_user_message,
)
from src.utils.abort_controller import AbortError

if TYPE_CHECKING:
    from src.tool_system.build_tool import Tool
    from src.tool_system.context import ToolContext

logger = logging.getLogger(__name__)

HOOK_TIMING_DISPLAY_THRESHOLD_MS = 500


@dataclass
class ContextModifier:
    tool_use_id: str
    modify_context: Callable[[Any], Any]


@dataclass
class MessageUpdateLazy:
    message: Message | None = None
    context_modifier: ContextModifier | None = None


async def run_tool_use(
    tool_use: Any,
    assistant_message: AssistantMessage,
    can_use_tool: Any,
    tool_use_context: ToolContext,
) -> AsyncGenerator[MessageUpdateLazy, None]:
    from extensions.sop_converter.runtime.macros.resolve_tool import resolve_tool_for_context

    tool_name = tool_use.name
    tool = resolve_tool_for_context(
        tool_use_context,
        tool_name,
        base_registry=getattr(tool_use_context, "tool_registry", None),
    )

    if tool is None:
        try:
            from src.tool_system.build_tool import find_tool_by_name
            from src.tool_system.defaults import build_default_registry

            tool = find_tool_by_name(
                build_default_registry().list_tools(),
                tool_name,
            )
        except Exception:  # nosec B110
            pass

    if tool is None:
        logger.debug("Unknown tool %s: %s", tool_name, tool_use.id)
        yield MessageUpdateLazy(
            message=create_user_message(
                content=[
                    {
                        "type": "tool_result",
                        "content": f"<tool_use_error>Error: No such tool available: {tool_name}</tool_use_error>",
                        "is_error": True,
                        "tool_use_id": tool_use.id,
                    }
                ],
                toolUseResult=f"Error: No such tool available: {tool_name}",
            ),
        )
        return

    tool_input = tool_use.input if isinstance(tool_use.input, dict) else {}

    try:
        # ``abort_controller`` is non-optional on ``ToolContext`` — the
        # historical ``if abort_ctrl and …`` guard masked the field-is-None
        # hazard class that broke ESC propagation into subagents.
        if tool_use_context.abort_controller.signal.aborted:
            if _is_user_cancelled_abort(tool_use_context):
                yield MessageUpdateLazy(
                    message=_build_user_cancelled_message(tool_use.id),
                )
                return
            content = _create_tool_result_stop(tool_use.id)
            yield MessageUpdateLazy(
                message=create_user_message(
                    content=[content],
                    toolUseResult=CANCEL_MESSAGE,
                ),
            )
            return

        async for update in _streamed_check_permissions_and_call_tool(
            tool,
            tool_use.id,
            tool_input,
            tool_use_context,
            can_use_tool,
            assistant_message,
        ):
            yield update
    except Exception as error:
        logger.error("Error in run_tool_use: %s", error)
        error_msg = str(error)
        tool_info = f" ({tool.name})" if tool else ""
        detailed = f"Error calling tool{tool_info}: {error_msg}"
        yield MessageUpdateLazy(
            message=create_user_message(
                content=[
                    {
                        "type": "tool_result",
                        "content": f"<tool_use_error>{detailed}</tool_use_error>",
                        "is_error": True,
                        "tool_use_id": tool_use.id,
                    }
                ],
                toolUseResult=detailed,
            ),
        )


async def _streamed_check_permissions_and_call_tool(
    tool: Tool,
    tool_use_id: str,
    tool_input: dict[str, Any],
    tool_use_context: ToolContext,
    can_use_tool: Any,
    assistant_message: AssistantMessage,
) -> AsyncGenerator[MessageUpdateLazy, None]:
    results = await _check_permissions_and_call_tool(
        tool,
        tool_use_id,
        tool_input,
        tool_use_context,
        can_use_tool,
        assistant_message,
    )
    for result in results:
        yield result


async def _check_permissions_and_call_tool(  # pylint: disable=too-many-boolean-expressions
    tool: Tool,
    tool_use_id: str,
    tool_input: dict[str, Any],
    tool_use_context: ToolContext,
    can_use_tool: Any,
    assistant_message: AssistantMessage,
) -> list[MessageUpdateLazy]:
    resulting_messages: list[MessageUpdateLazy] = []
    processed_input = tool_input

    # ----- Step 3 — Schema validation with deferred-tool recovery hint.
    # Mirrors typescript/src/services/tools/toolExecution.ts schema check at
    # the head of checkPermissionsAndCallTool. Skipped when the tool has no
    # input_schema (defensive — every Tool should declare one).
    # ``validate_tool_input`` returns the semantically-coerced input (string
    # "true"/"30" → bool/number, the zod preprocess wrappers) and the
    # coerced object is what flows onward — TS carries ``parsedInput.data``
    # forward (toolExecution.ts:669 → 821), not the raw model input.
    if getattr(tool, "input_schema", None):
        from src.tool_system.schema_validation import (
            build_schema_not_sent_hint,
            validate_tool_input,
        )

        try:
            processed_input = validate_tool_input(
                tool.name,
                processed_input,
                tool.input_schema,
            )
        except Exception as schema_err:  # ToolInputError or any subclass
            msg = str(schema_err)
            if getattr(tool, "should_defer", False):
                msg = msg + build_schema_not_sent_hint(tool)
            # ch06 round-4 PR-A GAP C — the ``InputValidationError:`` prefix
            # (TS toolExecution.ts:726). Load-bearing: the tool-failure-loop
            # guard categorizes on ``\bInputValidationError\b`` (ch01
            # round-3 port), so without the prefix repeated schema failures
            # are miscategorized as generic errors and the path-based /
            # signature-based trip logic doesn't recognize them.
            resulting_messages.append(
                MessageUpdateLazy(
                    message=create_user_message(
                        content=[
                            {
                                "type": "tool_result",
                                "content": (f"<tool_use_error>InputValidationError: {msg}</tool_use_error>"),
                                "is_error": True,
                                "tool_use_id": tool_use_id,
                            }
                        ],
                        toolUseResult=f"InputValidationError: {msg}",
                    ),
                )
            )
            return resulting_messages

    # ----- Step 4 — Semantic validation (`validate_input`).
    if tool.validate_input is not None:
        try:
            validation = tool.validate_input(processed_input, tool_use_context)
            if hasattr(validation, "result") and not validation.result:
                msg = getattr(validation, "message", "Validation failed")
                resulting_messages.append(
                    MessageUpdateLazy(
                        message=create_user_message(
                            content=[
                                {
                                    "type": "tool_result",
                                    "content": f"<tool_use_error>{msg}</tool_use_error>",
                                    "is_error": True,
                                    "tool_use_id": tool_use_id,
                                }
                            ],
                            toolUseResult=f"Error: {msg}",
                        ),
                    )
                )
                return resulting_messages
        except Exception as e:
            logger.debug("Validation error for %s: %s", tool.name, e)

    # ----- Step 6 — Input Backfill (clone, not mutate).
    # call() receives the PRE-BACKFILL input (post schema-coercion — TS's
    # callInput is likewise parsedInput.data): tool results embed input
    # fields verbatim (e.g. "File created successfully at: {path}"), and
    # backfill mutations (path expansion) would alter the serialized
    # transcript. The cloned, backfilled input is the hooks/permissions
    # audience only. Mirrors typescript/src/services/tools/toolExecution.ts:838-853.
    call_input = processed_input
    backfilled_clone: dict[str, Any] | None = None
    if tool.backfill_observable_input is not None:
        try:
            backfilled = dict(processed_input)
            tool.backfill_observable_input(backfilled)
            backfilled_clone = backfilled
            processed_input = backfilled
        except Exception as e:
            logger.debug("backfill_observable_input error for %s: %s", tool.name, e)

    should_prevent_continuation = False
    stop_reason: str | None = None
    hook_permission_result = None

    try:
        from src.services.tool_execution.tool_hooks import run_pre_tool_use_hooks

        async for result in run_pre_tool_use_hooks(
            tool_use_context,
            tool,
            processed_input,
            tool_use_id,
        ):
            result_type = result.get("type") if isinstance(result, dict) else getattr(result, "type", None)

            if result_type == "message":
                msg = result.get("message") if isinstance(result, dict) else getattr(result, "message", None)
                if msg:
                    resulting_messages.append(
                        msg if isinstance(msg, MessageUpdateLazy) else MessageUpdateLazy(message=msg)
                    )
            elif result_type == "hookPermissionResult":
                hook_permission_result = (
                    result.get("hookPermissionResult")
                    if isinstance(result, dict)
                    else getattr(result, "hook_permission_result", None)
                )
            elif result_type == "hookUpdatedInput":
                processed_input = (
                    result.get("updatedInput")
                    if isinstance(result, dict)
                    else getattr(result, "updated_input", processed_input)
                )
            elif result_type == "additionalContext":
                # A PreToolUse hook's additionalContext / additionalContexts
                # (the hook_additional_context attachment). run_pre_tool_use_hooks
                # wraps it as {"type":"additionalContext","message":{"message":
                # <attachment>}} (tool_hooks.py:116-128); surface it like the
                # "message" case. Previously DROPPED (critic C1-M2).
                _wrap = result.get("message") if isinstance(result, dict) else None
                _msg = _wrap.get("message") if isinstance(_wrap, dict) else _wrap
                if _msg:
                    resulting_messages.append(
                        _msg if isinstance(_msg, MessageUpdateLazy) else MessageUpdateLazy(message=_msg)
                    )
            elif result_type == "preventContinuation":
                should_prevent_continuation = True
            elif result_type == "stopReason":
                stop_reason = (
                    result.get("stopReason") if isinstance(result, dict) else getattr(result, "stop_reason", None)
                )
            elif result_type == "stop":
                resulting_messages.append(
                    MessageUpdateLazy(
                        message=create_user_message(
                            content=[_create_tool_result_stop(tool_use_id)],
                            toolUseResult=f"Error: {stop_reason or 'stopped'}",
                        ),
                    )
                )
                return resulting_messages
    except Exception as e:
        logger.debug("Pre-tool hook error: %s", e)

    permission_decision = await _resolve_permission(
        hook_permission_result,
        tool,
        processed_input,
        tool_use_context,
        can_use_tool,
        assistant_message,
        tool_use_id,
    )

    if permission_decision.get("behavior") != "allow":
        error_message = permission_decision.get("message", "Permission denied")
        if should_prevent_continuation and not error_message:
            error_message = f"Execution stopped by PreToolUse hook{': ' + stop_reason if stop_reason else ''}"

        resulting_messages.append(
            MessageUpdateLazy(
                message=create_user_message(
                    content=[
                        {
                            "type": "tool_result",
                            "content": error_message,
                            "is_error": True,
                            "tool_use_id": tool_use_id,
                        }
                    ],
                    toolUseResult=f"Error: {error_message}",
                ),
            )
        )
        return resulting_messages

    updated_input = permission_decision.get("updatedInput")
    if updated_input is not None:
        processed_input = updated_input

    start_time = time.monotonic()

    try:
        call_context = tool_use_context
        call_context.tool_use_id = tool_use_id
        call_context.user_modified = permission_decision.get("userModified", False)

        # If processed_input still points at the backfill clone, no
        # hook/permission replaced it — pass the pre-backfill call_input so
        # call() sees the pre-expansion field values (post schema-coercion).
        # Hook/permission flows may return a fresh object derived from the
        # backfilled clone (e.g. via schema re-parse): if its file_path
        # matches the backfill-expanded value, restore the pre-backfill one
        # so the tool result string embeds the path the model emitted.
        # Other modifications flow through unchanged. Mirrors
        # typescript/src/services/tools/toolExecution.ts:1212-1237.
        if (
            backfilled_clone is not None
            and processed_input is not call_input
            and isinstance(processed_input, dict)
            and "file_path" in processed_input
            and isinstance(call_input, dict)
            and "file_path" in call_input
            and processed_input.get("file_path") == backfilled_clone.get("file_path")
        ):
            call_input = {**processed_input, "file_path": call_input["file_path"]}
        elif processed_input is not backfilled_clone:
            call_input = processed_input

        result = await _call_tool(tool, call_input, call_context)

        # Post-tool override: a tool that observed the abort and returned
        # (e.g. bash's interrupted payload) reads as a generic failure;
        # replace it so the resume turn sees an unambiguous "user
        # rejected" signal. Mirrors TS StreamingToolExecutor.ts:332-345;
        # ported from the retired slim lane (ch07 unification).
        if _is_user_cancelled_abort(tool_use_context):
            resulting_messages.append(
                MessageUpdateLazy(
                    message=_build_user_cancelled_message(tool_use_id),
                )
            )
            return resulting_messages

        duration_ms = int((time.monotonic() - start_time) * 1000)

        # Step 11 — Result Budgeting. Per-tool max_result_size_chars is
        # honored here: empty content is replaced with a "(<tool> completed
        # with no output)" marker, oversized content is persisted to disk
        # and replaced with a <persisted-output> wrapper that includes a
        # preview, image content reaches the model intact.
        from src.services.tool_execution.tool_result_persistence import (
            compute_block_chars,
            process_tool_result_block,
            resolve_tool_results_dir,
        )

        tool_results_dir = resolve_tool_results_dir(tool_use_context)
        # WI-5.1: thread the per-message aggregate counter through. The
        # function consults it to decide whether to force-persist a
        # block that would push the running total past the cap. We then
        # update the counter with the post-decision block size via
        # ``compute_block_chars``.
        tool_result_block = process_tool_result_block(
            tool,
            result.data,
            tool_use_id,
            tool_results_dir=tool_results_dir,
            aggregate_chars_so_far=tool_use_context.tool_result_chars_so_far,
            duration_ms=duration_ms,
        )
        # Tools with a non-finite per-result threshold (for example Read)
        # opt out of aggregate replacement and aggregate accounting.
        if math.isfinite(tool.max_result_size_chars):
            tool_use_context.tool_result_chars_so_far += compute_block_chars(
                tool_result_block,
            )

        resulting_messages.append(
            MessageUpdateLazy(
                message=create_user_message(
                    content=[tool_result_block],
                    toolUseResult=result.data if not tool_use_context.agent_id else None,
                ),
                context_modifier=ContextModifier(
                    tool_use_id=tool_use_id,
                    modify_context=result.context_modifier,
                )
                if result.context_modifier
                else None,
            )
        )

        try:
            from src.services.tool_execution.tool_hooks import run_post_tool_use_hooks

            async for hook_result in run_post_tool_use_hooks(
                tool_use_context,
                tool,
                tool_use_id,
                processed_input,
                result.data,
            ):
                if isinstance(hook_result, dict) and "message" in hook_result:
                    resulting_messages.append(MessageUpdateLazy(message=hook_result["message"]))
                elif isinstance(hook_result, MessageUpdateLazy):
                    resulting_messages.append(hook_result)
        except Exception as e:
            logger.debug("Post-tool hook error: %s", e)

        # SERVICES-2 autoFix — a SIBLING of the PostToolUse hooks (run_post_tool_use_hooks
        # early-returns when no user hook is configured; autoFix must run
        # regardless, gated only on the settings.autoFix opt-in).
        try:
            from src.services.autofix.step import run_auto_fix_step

            async for af_result in run_auto_fix_step(
                tool_use_context,
                tool.name,
                tool_use_id,
            ):
                if isinstance(af_result, dict) and "message" in af_result:
                    resulting_messages.append(MessageUpdateLazy(message=af_result["message"]))
        except Exception as e:
            logger.debug("autoFix step error: %s", e)

        if result.new_messages:
            for msg in result.new_messages:
                resulting_messages.append(MessageUpdateLazy(message=msg))

        if should_prevent_continuation:
            from src.types.messages import create_attachment_message

            resulting_messages.append(
                MessageUpdateLazy(
                    message=create_attachment_message(
                        {
                            "type": "hook_stopped_continuation",
                            "message": stop_reason or "Execution stopped by hook",
                            "hook_name": f"PreToolUse:{tool.name}",
                            "tool_use_id": tool_use_id,
                            "hook_event": "PreToolUse",
                        }
                    ),
                )
            )

        return resulting_messages

    except AbortError as abort_error:
        # Keep the tool_use/tool_result pair valid while distinguishing a real
        # user rejection from an internal cancellation.  The latter also trips
        # the loop signal so no follow-up API turn is attempted.
        if _is_user_cancelled_abort(tool_use_context):
            resulting_messages.append(
                MessageUpdateLazy(
                    message=_build_user_cancelled_message(tool_use_id),
                )
            )
            return resulting_messages
        try:
            tool_use_context.abort_controller.abort("tool_raised_abort_error")
        except Exception:  # nosec B110
            pass
        error_text = f"Error: Tool execution aborted ({abort_error})"
        resulting_messages.append(
            MessageUpdateLazy(
                message=create_user_message(
                    content=[
                        {
                            "type": "tool_result",
                            "content": error_text,
                            "is_error": True,
                            "tool_use_id": tool_use_id,
                        }
                    ],
                    toolUseResult=error_text,
                ),
            )
        )
        return resulting_messages

    except Exception as error:
        duration_ms = int((time.monotonic() - start_time) * 1000)
        # Step 14 — Error Handling with telemetry-safe classification.
        # classify_tool_error returns a stable, mangling-safe string so
        # downstream telemetry/log scrapers can group errors by class
        # without leaking unredacted error messages. Mirrors
        # classifyToolError in typescript/src/services/tools/toolExecution.ts:151.
        classified = classify_tool_error(error)
        logger.error(
            "Tool %s failed (%dms) classified=%s: %s",
            tool.name,
            duration_ms,
            classified,
            error,
        )

        error_content = _format_error(error)

        error_block: dict[str, Any] = {
            "type": "tool_result",
            "content": error_content,
            "is_error": True,
            "tool_use_id": tool_use_id,
        }
        if duration_ms is not None and duration_ms >= 0:
            error_block["duration_ms"] = int(duration_ms)

        resulting_messages.append(
            MessageUpdateLazy(
                message=create_user_message(
                    content=[error_block],
                    toolUseResult=f"Error: {error_content}",
                ),
            )
        )

        try:
            from src.services.tool_execution.tool_hooks import (
                run_post_tool_use_failure_hooks,
            )

            async for hook_result in run_post_tool_use_failure_hooks(
                tool_use_context,
                tool,
                tool_use_id,
                processed_input,
                error_content,
            ):
                if isinstance(hook_result, dict) and "message" in hook_result:
                    resulting_messages.append(MessageUpdateLazy(message=hook_result["message"]))
        except Exception:  # nosec B110
            pass

        return resulting_messages


async def _call_tool(tool: Tool, tool_input: dict[str, Any], context: ToolContext) -> Any:
    from clawcodex_ext.tool_system.protocol import ToolResult  # pylint: disable=no-name-in-module

    call_fn = tool.call
    import asyncio
    import inspect

    if inspect.iscoroutinefunction(call_fn):
        result = await call_fn(tool_input, context)
    else:
        # Offload sync tools to a worker thread so they don't block the
        # event loop. Without this, a long Bash command holds the loop
        # for its entire duration — ESC key presses and abort_controller
        # listeners can't fire because no other task gets to run. Mirrors
        # the responsiveness TS gets "for free" from Node's
        # callback-driven I/O.
        result = await asyncio.to_thread(call_fn, tool_input, context)

    if not isinstance(result, ToolResult):
        result = ToolResult(name=tool.name, output=result)

    return result


async def _resolve_permission(
    hook_permission_result: Any,
    tool: Tool,
    tool_input: dict[str, Any],
    tool_use_context: ToolContext,
    can_use_tool: Any,
    assistant_message: AssistantMessage,
    tool_use_id: str,
) -> dict[str, Any]:
    from src.services.tool_execution.tool_hooks import resolve_hook_permission_decision

    return await resolve_hook_permission_decision(
        hook_permission_result,
        tool,
        tool_input,
        tool_use_context,
        can_use_tool,
        assistant_message,
        tool_use_id,
    )


def _is_user_cancelled_abort(tool_use_context: Any) -> bool:
    """Return whether the abort represents a user-initiated rejection."""
    controller = tool_use_context.abort_controller
    if not controller.signal.aborted:
        return False
    return controller.signal.reason not in ("sibling_error", "streaming_fallback")


def _build_user_cancelled_message(tool_use_id: str) -> Any:
    """Build the synthetic rejected tool result used for user aborts."""
    from src.types.content_blocks import ToolResultBlock
    from src.types.messages import REJECT_MESSAGE

    return create_user_message(
        content=[
            ToolResultBlock(
                tool_use_id=tool_use_id,
                content=REJECT_MESSAGE,
                is_error=True,
            )
        ],
        toolUseResult="User rejected tool use",
    )


def _create_tool_result_stop(tool_use_id: str) -> dict[str, Any]:
    return {
        "type": "tool_result",
        "content": CANCEL_MESSAGE,
        "is_error": True,
        "tool_use_id": tool_use_id,
    }


# TS formatError caps thrown-error content at 40KB with middle truncation
# (toolErrors.ts:16) — "enough for most command error logs".
_FORMAT_ERROR_MAX_LENGTH = 40000


def _format_error(error: Exception) -> str:
    """Format a thrown tool error — port of formatError (utils/toolErrors.ts:5).

    Thrown ``call()`` errors go on the wire RAW: ``is_error: True`` on the
    tool_result carries the error-ness (TS toolExecution.ts:1633+1674 sends
    ``content: formatError(error)`` with no ``<tool_use_error>`` wrapping).
    The tags are reserved for pre-execution failures — no-such-tool,
    InputValidationError, validate_input — which TS also tags. Wrapping
    thrown errors too (the old behavior here) leaked the tags into the TUI
    transcript (`Error: <tool_use_error>path is not a file: …`) and diverged
    the model-facing bytes from the original.

    getErrorParts (toolErrors.ts:26) appends ``stderr``/``stdout`` string
    attributes when present (subprocess.CalledProcessError carries both).
    The TS ShellError branch has no Python parallel — our bash tool formats
    shell failures into the result body itself, so only the generic branch
    is ported.
    """
    if isinstance(error, AbortError):
        # TS: error.message || INTERRUPT_MESSAGE_FOR_TOOL_USE. Effectively
        # dead here — AbortError is caught by the dedicated except at the
        # call-tool layer before Step 14 — kept as a belt for direct callers.
        return str(error) or INTERRUPT_MESSAGE_FOR_TOOL_USE
    parts = [str(error)]
    for attr in ("stderr", "stdout"):
        value = getattr(error, attr, None)
        if isinstance(value, str):
            parts.append(value)
    full_message = "\n".join(p for p in parts if p).strip()
    if not full_message:
        full_message = "Command failed with no output"
    if len(full_message) <= _FORMAT_ERROR_MAX_LENGTH:
        return full_message
    half_length = _FORMAT_ERROR_MAX_LENGTH // 2
    start = full_message[:half_length]
    end = full_message[-half_length:]
    return f"{start}\n\n... [{len(full_message) - _FORMAT_ERROR_MAX_LENGTH} characters truncated] ...\n\n{end}"


def classify_tool_error(error: Exception) -> str:
    if hasattr(error, "telemetry_message"):
        return str(error.telemetry_message)[:200]
    if isinstance(error, OSError) and hasattr(error, "errno"):
        import errno as errno_mod

        code = errno_mod.errorcode.get(error.errno, "")
        if code:
            return f"Error:{code}"
    name = getattr(error, "name", None) or type(error).__name__
    if name and name != "Error" and len(name) > 3:
        return name[:60]
    return "Error"
