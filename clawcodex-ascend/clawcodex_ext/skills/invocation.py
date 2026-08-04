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

"""Canonical skill invocation service shared by user and model surfaces.

Discovery/catalogue code owns *which* skills exist.  This module owns the
request-scoped invocation transaction: resolve (including aliases), apply the
origin-specific gate, render the prompt, prepare the inline message and context
overrides, and only then record the invocation.  Keeping those steps together
prevents the user slash-command and model ``Skill`` tool paths from drifting.
"""

from __future__ import annotations

import inspect
import logging
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import TYPE_CHECKING, Any, Callable, Literal, Mapping

from clawcodex_ext.types.messages import create_user_message

if TYPE_CHECKING:
    from clawcodex_ext.tool_system.context import ToolContext


logger = logging.getLogger(__name__)


class SkillInvocationOrigin(str, Enum):
    """The surface that requested a skill invocation."""

    USER = "user"
    MODEL = "model"


# Short public spelling requested by the service contract.
Origin = SkillInvocationOrigin


class SkillInvocationErrorCode(str, Enum):
    INVALID_NAME = "invalid_name"
    NOT_FOUND = "not_found"
    DISABLED = "disabled"
    ENABLEMENT_CHECK_FAILED = "enablement_check_failed"
    MODEL_INVOCATION_DISABLED = "model_invocation_disabled"
    USER_INVOCATION_DISABLED = "user_invocation_disabled"
    NOT_PROMPT = "not_prompt"
    RESOLUTION_FAILED = "resolution_failed"
    PROMPT_BUILD_FAILED = "prompt_build_failed"
    HOOKS_UNSUPPORTED = "hooks_unsupported"
    HOOK_REGISTRATION_FAILED = "hook_registration_failed"
    FORK_UNSUPPORTED = "fork_unsupported"
    FORK_EXECUTION_FAILED = "fork_execution_failed"
    INVOCATION_RECORD_FAILED = "invocation_record_failed"
    RECURSIVE_INVOCATION = "recursive_invocation"


@dataclass(frozen=True)
class SkillInvocationError:
    """Stable, machine-readable invocation failure."""

    code: SkillInvocationErrorCode
    message: str
    model_error_code: int | None = None
    details: Mapping[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "code": self.code.value,
            "message": self.message,
        }
        if self.model_error_code is not None:
            data["modelErrorCode"] = self.model_error_code
        if self.details:
            data["details"] = dict(self.details)
        return data


@dataclass(frozen=True)
class SkillInvocationRequest:
    skill_name: str
    args: str = ""
    origin: SkillInvocationOrigin = SkillInvocationOrigin.MODEL

    @property
    def normalized_name(self) -> str:
        return self.skill_name.strip().lstrip("/")

    @property
    def skill(self) -> str:
        """Compatibility spelling matching the model tool input."""

        return self.skill_name


ContextModifier = Callable[["ToolContext"], "ToolContext"]


@dataclass(frozen=True)
class SkillInvocationResult:
    request: SkillInvocationRequest
    success: bool
    command_name: str | None = None
    requested_name: str | None = None
    skill: Any | None = None
    prompt: str | None = None
    content_blocks: tuple[Any, ...] = ()
    status: Literal["inline", "fork"] | None = None
    new_messages: tuple[Any, ...] = ()
    context_modifier: ContextModifier | None = None
    error: SkillInvocationError | None = None
    fork_result: str | None = None

    diagnostics: tuple[str, ...] = ()

    @classmethod
    def failure(
        cls,
        request: SkillInvocationRequest,
        code: SkillInvocationErrorCode,
        message: str,
        *,
        model_error_code: int | None = None,
        details: Mapping[str, Any] | None = None,
        skill: Any | None = None,
    ) -> "SkillInvocationResult":
        return cls(
            request=request,
            success=False,
            requested_name=request.normalized_name or None,
            skill=skill,
            error=SkillInvocationError(
                code=code,
                message=message,
                model_error_code=model_error_code,
                details=details or {},
            ),
        )


SkillResolver = Callable[[str, "ToolContext"], Any | None]
HookRegistrar = Callable[[Any, SkillInvocationRequest, "ToolContext"], None]
ForkExecutor = Callable[
    [Any, SkillInvocationRequest, "ToolContext", str],
    SkillInvocationResult,
]
InvocationRecorder = Callable[[str, str, str, str | None], None]


def _default_resolver(skill_name: str, context: "ToolContext") -> Any | None:
    """Resolve from the live per-workspace catalogue, including aliases."""

    from clawcodex_ext.skills.catalog import resolve

    return resolve(
        skill_name,
        project_root=context.workspace_root,
        session_id=str(context.session_id) if context.session_id is not None else None,
        include_disabled=True,
    )


def _default_recorder(
    skill_name: str,
    skill_path: str,
    content: str,
    agent_id: str | None,
) -> None:
    from clawcodex_ext.bootstrap.state import add_invoked_skill  # pylint: disable=no-name-in-module

    add_invoked_skill(skill_name, skill_path, content, agent_id)


def _invoke_prompt_builder(builder: Any, args: str, context: "ToolContext") -> Any:
    """Call both legacy one-argument and context-aware prompt builders."""

    try:
        signature = inspect.signature(builder)
    except (TypeError, ValueError):
        return builder(args)

    try:
        signature.bind(args, context)
    except TypeError:
        try:
            signature.bind(args, context=context)
        except TypeError:
            return builder(args)
        return builder(args, context=context)
    return builder(args, context)


def _coerce_prompt_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, (list, tuple)):
        text_blocks = [
            str(block.get("text", "")) for block in value if isinstance(block, Mapping) and block.get("type") == "text"
        ]
        if text_blocks:
            return "\n\n".join(text_blocks)
    if value is None:
        return ""
    return str(value)


def _make_shell_executor(
    context: "ToolContext",
    allowed_tools: list[str] | None,
    *,
    slash_command_name: str,
    shell: str = "auto",
) -> Callable[[str, bool], str]:
    """Create the renderer callback for embedded skill shell blocks."""

    from clawcodex_ext.skills.runtime_substitution import (
        format_shell_error,
        format_shell_output,
    )
    from clawcodex_ext.tool_system.registry import ToolCall

    allowed_tools = list(allowed_tools or [])
    modifier = build_request_context_modifier(
        allowed_tools=allowed_tools,
        model=None,
        effort=None,
    )
    shell_context = modifier(context) if modifier is not None else context

    def _exec(command: str, inline: bool) -> str:
        try:
            registry = shell_context.tool_registry
            if registry is None:
                from clawcodex_ext.tool_system.defaults import build_default_registry

                registry = build_default_registry(provider=getattr(shell_context, "_active_provider", None))
            tool_result = registry.dispatch(
                ToolCall(
                    name="Bash",
                    input={"command": command, "shell": shell},
                ),
                shell_context,
            )
        except Exception as exc:  # noqa: BLE001 - surface renderer failures in prompt
            return format_shell_error(exc, command, inline=inline)

        output = tool_result.output if isinstance(tool_result.output, dict) else {}
        stdout = str(output.get("stdout", ""))
        stderr = str(output.get("stderr", ""))
        exit_code = output.get("exit_code")

        if isinstance(exit_code, int) and exit_code != 0:
            error_text = format_shell_output(stdout, stderr, inline=inline)
            error_text = error_text or f"command failed (exit {exit_code})"
            return format_shell_error(error_text, command, inline=inline)
        if tool_result.is_error:
            error_text = format_shell_output(stdout, stderr, inline=inline) or output.get("error") or "command failed"
            return format_shell_error(str(error_text), command, inline=inline)
        return format_shell_output(stdout, stderr, inline=inline)

    _exec.__name__ = f"execute_shell_for_{slash_command_name.lstrip('/')}"
    return _exec


def _render_prompt(skill: Any, request: SkillInvocationRequest, context: "ToolContext") -> str:
    from clawcodex_ext.skills.runtime_substitution import render_skill_prompt

    builder = getattr(skill, "get_prompt_for_command", None)
    if builder is not None:
        body = _coerce_prompt_text(_invoke_prompt_builder(builder, request.args or "", context))
        base_dir = getattr(builder, "_bundled_resource_root", None)
        if base_dir:
            # The bundled compatibility wrapper adds this header for direct
            # Skill.get_prompt callers. Strip it so the canonical renderer can
            # normalize the body, then add the same header exactly once.
            prefix = f"Base directory for this skill: {base_dir}\n\n"
            if body.startswith(prefix):
                body = body[len(prefix) :]
        render_args = ""
        argument_names: list[str] = []
    else:
        body = getattr(skill, "markdown_content", "") or getattr(skill, "content", "") or ""
        base_dir = getattr(skill, "base_dir", None) or getattr(skill, "skill_root", None)
        render_args = request.args
        argument_names = list(getattr(skill, "argument_names", None) or [])

    allowed_tools = list(getattr(skill, "allowed_tools", None) or [])
    skill_shell = getattr(skill, "shell", "auto") or "auto"
    shell_executor = _make_shell_executor(
        context,
        allowed_tools,
        slash_command_name=f"/{request.normalized_name}",
        shell=skill_shell,
    )
    return render_skill_prompt(
        body=body,
        args=render_args,
        base_dir=base_dir,
        argument_names=argument_names,
        session_id=context.session_id,
        loaded_from=getattr(skill, "loaded_from", "skills"),
        slash_command_name=f"/{request.normalized_name}",
        shell_executor=shell_executor,
    )


def _rendered_resource_roots(skill: Any) -> tuple[str, ...]:
    builder = getattr(skill, "get_prompt_for_command", None)
    root = getattr(builder, "_bundled_resource_root", None) if builder is not None else None
    return (str(root),) if root else ()


def _invocation_diagnostics(skill: Any, context: "ToolContext") -> tuple[str, ...]:
    diagnostics: list[str] = []
    builder = getattr(skill, "get_prompt_for_command", None)
    resource_diagnostic = getattr(builder, "_bundled_resource_diagnostic", None) if builder is not None else None
    if resource_diagnostic:
        diagnostics.append(str(resource_diagnostic))

    effort = getattr(skill, "effort", None)
    provider = context._active_provider
    if effort not in (None, "") and provider is not None:
        module_name = type(provider).__module__.lower()
        class_name = type(provider).__name__.lower()
        supports_effort = (
            ("anthropic_provider" in module_name and "minimax" not in class_name)
            or "kimi_provider" in module_name
            or any(marker in module_name for marker in ("openai", "litellm", "grok"))
            or callable(getattr(provider, "skill_effort_kwargs", None))
        )
        if not supports_effort:
            diagnostics.append(
                f"Provider {type(provider).__name__} does not support skill effort "
                f"override {effort!r}; continuing without it"
            )
    return tuple(diagnostics)


def build_request_context_modifier(
    *,
    allowed_tools: list[str] | tuple[str, ...],
    model: str | None,
    effort: str | int | None,
    resource_roots: list[str] | tuple[str, ...] = (),
) -> ContextModifier | None:
    """Return a modifier that clones, rather than mutates, request context.

    ``allowed_tools`` are installed in the request-private ``command`` source,
    model replaces ``ToolUseOptions.main_loop_model``, and effort is carried in
    the request's ``thinking_config``.  Every nested object touched here is
    copied, so the caller's shared ``ToolContext`` remains unchanged.
    """

    tool_rules = list(allowed_tools)
    roots = tuple(dict.fromkeys(str(root) for root in resource_roots if root))
    if not tool_rules and not model and not effort and not roots:
        return None

    def _modifier(context: "ToolContext") -> "ToolContext":
        permission_context = context.permission_context
        if tool_rules:
            always_allow_rules = {
                source: list(rules) for source, rules in permission_context.always_allow_rules.items()
            }
            command_rules = list(always_allow_rules.get("command", []))
            for rule in tool_rules:
                if rule not in command_rules:
                    command_rules.append(rule)
            always_allow_rules["command"] = command_rules
            permission_context = replace(
                permission_context,
                always_allow_rules=always_allow_rules,
            )

        options = context.options
        option_updates: dict[str, Any] = {}
        if model:
            option_updates["main_loop_model"] = model
        if effort:
            thinking_config = dict(options.thinking_config or {})
            thinking_config["effort"] = effort
            option_updates["thinking_config"] = thinking_config
        if option_updates:
            options = replace(options, **option_updates)

        merged_roots = tuple(dict.fromkeys((*context.skill_resource_roots, *roots)))
        return replace(
            context,
            permission_context=permission_context,
            options=options,
            skill_resource_roots=merged_roots,
        )

    return _modifier


def apply_skill_context_modifier(
    context: "ToolContext",
    modifier: ContextModifier | None,
) -> "ToolContext":
    """Apply a cloned modifier to the live request context and save its base."""
    if modifier is None:
        return context

    modified = modifier(context)
    if context._skill_permission_base is None:
        context._skill_permission_base = context.permission_context
    if context._skill_options_base is None:
        context._skill_options_base = context.options
    if context._skill_resource_roots_base is None:
        context._skill_resource_roots_base = context.skill_resource_roots

    context.permission_context = modified.permission_context
    context.options = modified.options
    context.skill_model_override = modified.options.main_loop_model or None
    context.skill_resource_roots = modified.skill_resource_roots
    thinking_config = modified.options.thinking_config or {}
    context.skill_effort_override = thinking_config.get("effort")
    if not context.skill_scope_active:
        context.skill_scope_pending = True
    return context


def _build_inline_meta_message(command_name: str, args: str, prompt: str) -> Any:
    metadata = [
        f"<command-message>{command_name}</command-message>",
        f"<command-name>/{command_name}</command-name>",
    ]
    metadata.append(f"<command-args>{args}</command-args>")
    metadata_text = "\n".join(metadata)
    content = f"{metadata_text}\n\n{prompt}"
    return create_user_message(content, isMeta=True)


def _effective_skill_root(skill: Any) -> str | None:
    """Return a usable skill root, excluding failed bundled extraction paths."""

    base_dir = getattr(skill, "base_dir", None)
    if base_dir:
        return str(base_dir)

    source = str(getattr(skill, "source", "") or "")
    loaded_from = str(getattr(skill, "loaded_from", "") or "")
    if source == "bundled" or loaded_from == "bundled":
        builder = getattr(skill, "get_prompt_for_command", None)
        extracted = getattr(builder, "_bundled_resource_root", None) if builder is not None else None
        return str(extracted) if extracted else None

    root = getattr(skill, "skill_root", None)
    return str(root) if root else None


def _skill_source_path(skill: Any, command_name: str) -> str:
    """Return the concrete source path used for compaction recovery."""

    root = _effective_skill_root(skill)
    if root:
        return root
    source = str(getattr(skill, "source", "") or "")
    return f"{source}:{command_name}" if source else command_name


def _rollback_skill_hooks_since(
    context: "ToolContext",
    previous_keys: frozenset[str],
) -> None:
    """Remove only hooks added by the current invocation transaction."""

    added_keys = set(context.skill_hook_keys).difference(previous_keys)
    if not added_keys:
        return
    for event, configs in list(context.skill_hooks.items()):
        context.skill_hooks[event] = [
            config for config in configs if getattr(config, "_skill_registration_key", None) not in added_keys
        ]
        if not context.skill_hooks[event]:
            context.skill_hooks.pop(event, None)
    context.skill_hook_keys.difference_update(added_keys)


def _skill_requires_fork(skill: Any) -> bool:
    """An explicit agent is a fork request even when context is omitted."""

    return (getattr(skill, "context", "inline") or "inline") == "fork" or bool(getattr(skill, "agent", None))


class SkillInvocationService:
    """Resolve, gate and invoke skills for every interactive surface.

    Fork execution and skill-hook registration require runtime-owned async and
    snapshot lifecycle state that this synchronous service cannot safely invent.
    Callers can provide explicit handlers.  Without them, such skills fail with
    structured ``fork_unsupported``/``hooks_unsupported`` errors rather than
    claiming a successful invocation that injected nothing.
    """

    def __init__(
        self,
        *,
        resolver: SkillResolver | None = None,
        recorder: InvocationRecorder | None = None,
        hook_registrar: HookRegistrar | None = None,
        fork_executor: ForkExecutor | None = None,
    ) -> None:
        self._resolver = resolver or _default_resolver
        self._recorder = recorder
        self._hook_registrar = hook_registrar
        self._fork_executor = fork_executor

    def _record_invocation(
        self,
        command_name: str,
        skill_path: str,
        prompt: str,
        context: "ToolContext",
    ) -> None:
        if self._recorder is not None:
            self._recorder(command_name, skill_path, prompt, context.agent_id)
            return
        _default_recorder(
            command_name,
            skill_path,
            prompt,
            context.agent_id,
        )

    def validate(
        self,
        request: SkillInvocationRequest,
        context: "ToolContext",
        *,
        resolved_skill: Any | None = None,
    ) -> SkillInvocationResult:
        normalized_name = request.normalized_name
        if not normalized_name:
            return SkillInvocationResult.failure(
                request,
                SkillInvocationErrorCode.INVALID_NAME,
                f"Invalid skill format: {request.skill_name}",
                model_error_code=1,
            )

        skill = resolved_skill
        if skill is None:
            try:
                skill = self._resolver(normalized_name, context)
            except Exception as exc:  # noqa: BLE001 - resolution is an invocation boundary
                logger.exception("Failed to resolve skill %s", normalized_name)
                return SkillInvocationResult.failure(
                    request,
                    SkillInvocationErrorCode.RESOLUTION_FAILED,
                    f"Failed to resolve skill {normalized_name}: {exc}",
                    model_error_code=2,
                    details={"exceptionType": type(exc).__name__},
                )
        if skill is None:
            return SkillInvocationResult.failure(
                request,
                SkillInvocationErrorCode.NOT_FOUND,
                f"Unknown skill: {normalized_name}",
                model_error_code=2,
            )

        try:
            enabled_value = getattr(skill, "is_enabled", None)
            enabled = (
                True if enabled_value is None else bool(enabled_value() if callable(enabled_value) else enabled_value)
            )
        except Exception as exc:  # noqa: BLE001 - dynamic gates must not crash dispatch
            logger.exception("Skill enablement check failed for %s", normalized_name)
            return SkillInvocationResult.failure(
                request,
                SkillInvocationErrorCode.ENABLEMENT_CHECK_FAILED,
                f"Skill {normalized_name} enablement check failed: {exc}",
                model_error_code=2,
                details={"exceptionType": type(exc).__name__},
                skill=skill,
            )
        if not enabled:
            # Disabled commands are absent from the model's available-command
            # view in Claude Code, so model callers observe the existing code 2.
            return SkillInvocationResult.failure(
                request,
                SkillInvocationErrorCode.DISABLED,
                f"Skill {normalized_name} is disabled",
                model_error_code=2,
                skill=skill,
            )

        if request.origin is SkillInvocationOrigin.MODEL and getattr(skill, "disable_model_invocation", False):
            return SkillInvocationResult.failure(
                request,
                SkillInvocationErrorCode.MODEL_INVOCATION_DISABLED,
                f"Skill {normalized_name} cannot be used with Skill tool due to disable-model-invocation",
                model_error_code=4,
                skill=skill,
            )

        if getattr(skill, "type", "prompt") != "prompt":
            return SkillInvocationResult.failure(
                request,
                SkillInvocationErrorCode.NOT_PROMPT,
                f"Skill {normalized_name} is not a prompt-based skill",
                model_error_code=5,
                skill=skill,
            )

        if request.origin is SkillInvocationOrigin.USER and not getattr(skill, "user_invocable", True):
            return SkillInvocationResult.failure(
                request,
                SkillInvocationErrorCode.USER_INVOCATION_DISABLED,
                f"Skill {normalized_name} is not user-invocable",
                skill=skill,
            )

        hooks = getattr(skill, "hooks", None)
        if hooks and self._hook_registrar is None:
            return SkillInvocationResult.failure(
                request,
                SkillInvocationErrorCode.HOOKS_UNSUPPORTED,
                f"Skill {normalized_name} declares hooks, but no skill hook registrar is configured",
                skill=skill,
            )

        if _skill_requires_fork(skill) and self._fork_executor is None:
            return SkillInvocationResult.failure(
                request,
                SkillInvocationErrorCode.FORK_UNSUPPORTED,
                f"Skill {normalized_name} requires fork execution, but no fork executor is configured",
                skill=skill,
            )

        return SkillInvocationResult(
            request=request,
            success=True,
            command_name=str(getattr(skill, "name", normalized_name)),
            requested_name=normalized_name,
            skill=skill,
        )

    def invoke(
        self,
        request: SkillInvocationRequest,
        context: "ToolContext",
        *,
        resolved_skill: Any | None = None,
    ) -> SkillInvocationResult:
        validation = self.validate(request, context, resolved_skill=resolved_skill)
        if not validation.success:
            return validation

        skill = validation.skill
        assert skill is not None
        command_name = validation.command_name or request.normalized_name

        if command_name in context.active_skill_names:
            return SkillInvocationResult.failure(
                request,
                SkillInvocationErrorCode.RECURSIVE_INVOCATION,
                f"Skill {command_name} is already active in this query",
                skill=skill,
            )

        try:
            prompt = _render_prompt(skill, request, context)
        except Exception as exc:  # noqa: BLE001 - prompt builders are extension code
            logger.exception("Prompt builder failed for skill %s", command_name)
            return SkillInvocationResult.failure(
                request,
                SkillInvocationErrorCode.PROMPT_BUILD_FAILED,
                f"Failed to build prompt for skill {command_name}: {exc}",
                details={"exceptionType": type(exc).__name__},
                skill=skill,
            )

        diagnostics = _invocation_diagnostics(skill, context)

        if not prompt.strip():
            return SkillInvocationResult.failure(
                request,
                SkillInvocationErrorCode.PROMPT_BUILD_FAILED,
                f"Skill {command_name} produced an empty prompt",
                skill=skill,
            )

        hook_keys_before = frozenset(context.skill_hook_keys)
        hooks = getattr(skill, "hooks", None)
        if hooks:
            assert self._hook_registrar is not None
            try:
                self._hook_registrar(skill, request, context)
            except Exception as exc:  # noqa: BLE001 - extension boundary
                logger.exception("Hook registration failed for skill %s", command_name)
                _rollback_skill_hooks_since(context, hook_keys_before)
                return SkillInvocationResult.failure(
                    request,
                    SkillInvocationErrorCode.HOOK_REGISTRATION_FAILED,
                    f"Failed to register hooks for skill {command_name}: {exc}",
                    details={"exceptionType": type(exc).__name__},
                    skill=skill,
                )

        if _skill_requires_fork(skill):
            assert self._fork_executor is not None
            try:
                fork_result = self._fork_executor(skill, request, context, prompt)
            except Exception as exc:  # noqa: BLE001 - extension boundary
                logger.exception("Fork execution failed for skill %s", command_name)
                _rollback_skill_hooks_since(context, hook_keys_before)
                return SkillInvocationResult.failure(
                    request,
                    SkillInvocationErrorCode.FORK_EXECUTION_FAILED,
                    f"Fork execution failed for skill {command_name}: {exc}",
                    details={"exceptionType": type(exc).__name__},
                    skill=skill,
                )
            if not fork_result.success:
                _rollback_skill_hooks_since(context, hook_keys_before)
                return fork_result

            skill_path = _skill_source_path(skill, command_name)
            try:
                self._record_invocation(command_name, skill_path, prompt, context)
            except Exception as exc:  # noqa: BLE001 - transactional boundary
                logger.exception("Failed to record invoked skill %s", command_name)
                _rollback_skill_hooks_since(context, hook_keys_before)
                return SkillInvocationResult.failure(
                    request,
                    SkillInvocationErrorCode.INVOCATION_RECORD_FAILED,
                    f"Failed to record invoked skill {command_name}: {exc}",
                    details={"exceptionType": type(exc).__name__},
                    skill=skill,
                )
            return replace(
                fork_result,
                content_blocks=({"type": "text", "text": prompt},),
                diagnostics=diagnostics,
            )

        context_modifier = build_request_context_modifier(
            allowed_tools=list(getattr(skill, "allowed_tools", None) or []),
            model=getattr(skill, "model", None),
            effort=getattr(skill, "effort", None),
            resource_roots=_rendered_resource_roots(skill),
        )
        inline_message = _build_inline_meta_message(command_name, request.args, prompt)

        skill_path = _skill_source_path(skill, command_name)
        try:
            self._record_invocation(command_name, skill_path, prompt, context)
        except Exception as exc:  # noqa: BLE001 - preserve no-partial-injection guarantee
            logger.exception("Failed to record invoked skill %s", command_name)
            _rollback_skill_hooks_since(context, hook_keys_before)
            return SkillInvocationResult.failure(
                request,
                SkillInvocationErrorCode.INVOCATION_RECORD_FAILED,
                f"Failed to record invoked skill {command_name}: {exc}",
                details={"exceptionType": type(exc).__name__},
                skill=skill,
            )

        context.active_skill_names = tuple(context.active_skill_names) + (command_name,)
        if request.origin is SkillInvocationOrigin.USER and not context.skill_scope_active:
            context.skill_scope_pending = True

        return SkillInvocationResult(
            request=request,
            success=True,
            command_name=command_name,
            requested_name=request.normalized_name,
            skill=skill,
            prompt=prompt,
            content_blocks=({"type": "text", "text": prompt},),
            status="inline",
            new_messages=(inline_message,),
            context_modifier=context_modifier,
            diagnostics=diagnostics,
        )


def _register_skill_hooks(
    skill: Any,
    request: SkillInvocationRequest,
    context: "ToolContext",
) -> None:
    """Validate and atomically register skill hooks on the active context."""
    from clawcodex_ext.hooks.config_manager import _parse_hook_config  # pylint: disable=no-name-in-module
    from clawcodex_ext.hooks.hook_types import ALL_HOOK_EVENTS, HookSource  # pylint: disable=no-name-in-module

    raw_hooks = getattr(skill, "hooks", None)
    if not raw_hooks:
        return
    if not isinstance(raw_hooks, Mapping):
        raise ValueError("skill hooks must be an event mapping")

    skill_name = str(getattr(skill, "name", request.normalized_name))
    skill_root = _effective_skill_root(skill)
    agent_id = context.agent_id or ""
    prepared: list[tuple[str, Any, str]] = []
    required_fields = {
        "command": ("command",),
        "http": ("url",),
        "prompt": ("promptText", "prompt_text"),
        "agent": ("agentInstructions", "agent_instructions"),
    }

    for event, matchers in raw_hooks.items():
        if event not in ALL_HOOK_EVENTS or not isinstance(matchers, list):
            raise ValueError(f"invalid skill hook event: {event}")
        for matcher_index, matcher_group in enumerate(matchers):
            if not isinstance(matcher_group, Mapping):
                raise ValueError(f"invalid matcher for skill hook event: {event}")
            matcher_value = matcher_group.get("matcher")
            matcher = matcher_value if isinstance(matcher_value, str) else None
            inner_hooks = matcher_group.get("hooks")
            if not isinstance(inner_hooks, list):
                raise ValueError(f"skill hook {event} is missing hooks list")
            for hook_index, raw_hook in enumerate(inner_hooks):
                if not isinstance(raw_hook, Mapping):
                    raise ValueError(f"invalid hook entry for {event}")
                hook_type = str(raw_hook.get("type", "command"))
                required = required_fields.get(hook_type)
                if required is None or not any(raw_hook.get(field) for field in required):
                    raise ValueError(f"invalid {hook_type} hook for {event}")
                config = _parse_hook_config(dict(raw_hook), HookSource.SKILL)
                config = replace(
                    config,
                    matcher=matcher or config.matcher,
                    source=HookSource.SKILL,
                    skill_root=skill_root,
                )
                key = f"{agent_id}:{skill_name}:{skill_root or ''}:{event}:{matcher_index}:{hook_index}:{raw_hook!r}"
                setattr(config, "_skill_registration_key", key)
                setattr(config, "_skill_name", skill_name)
                prepared.append((event, config, key))

    for event, config, key in prepared:
        if key in context.skill_hook_keys:
            continue
        context.skill_hooks.setdefault(event, []).append(config)
        context.skill_hook_keys.add(key)


def _execute_forked_skill(
    skill: Any,
    request: SkillInvocationRequest,
    context: "ToolContext",
    prompt: str,
) -> SkillInvocationResult:
    """Execute a fork skill through the existing Agent tool runner."""
    from clawcodex_ext.tool_system.tools.agent import make_agent_tool

    command_name = str(getattr(skill, "name", request.normalized_name))
    registry = context.tool_registry
    provider = context._active_provider
    if registry is None or provider is None:
        return SkillInvocationResult.failure(
            request,
            SkillInvocationErrorCode.FORK_UNSUPPORTED,
            f"Skill {command_name} requires an active Agent runtime",
            skill=skill,
        )

    modifier = build_request_context_modifier(
        allowed_tools=list(getattr(skill, "allowed_tools", None) or []),
        model=getattr(skill, "model", None),
        effort=getattr(skill, "effort", None),
        resource_roots=_rendered_resource_roots(skill),
    )
    fork_context = modifier(context) if modifier is not None else replace(context)
    fork_context.active_skill_names = tuple(context.active_skill_names) + (command_name,)
    fork_context.skill_model_override = getattr(skill, "model", None)
    fork_context.skill_effort_override = getattr(skill, "effort", None)
    fork_context.skill_scope_pending = True

    tool_input: dict[str, Any] = {
        "prompt": prompt,
        "description": f"Run /{command_name}",
    }
    agent_name = getattr(skill, "agent", None)
    if agent_name:
        tool_input["subagent_type"] = agent_name
        tool_input["_force_foreground"] = True
        if (getattr(skill, "context", "inline") or "inline") == "fork":
            tool_input["_inherit_context"] = True
    elif (getattr(skill, "context", "inline") or "inline") == "fork":
        # This internal AgentTool control is intentionally absent from its
        # model-visible input schema. An explicit skill fork must not depend
        # on the optional CLAUDE_FORK_SUBAGENT gate.
        tool_input["_force_fork"] = True
    # Rebuild model-visible skills from the child context/catalog instead of
    # inheriting the parent's potentially stale Available Skills prose.
    tool_input["_refresh_skill_listing"] = True
    if getattr(skill, "model", None):
        tool_input["model"] = skill.model

    agent_result = make_agent_tool(registry, provider).call(tool_input, fork_context)
    output = agent_result.output if isinstance(agent_result.output, dict) else {}
    if agent_result.is_error or output.get("status") not in {"completed", "success"}:
        return SkillInvocationResult.failure(
            request,
            SkillInvocationErrorCode.FORK_EXECUTION_FAILED,
            str(output.get("error") or f"Fork execution failed for skill {command_name}"),
            skill=skill,
        )

    content = output.get("content", [])
    if isinstance(content, list):
        result_text = "\n".join(
            str(block.get("text", ""))
            for block in content
            if isinstance(block, Mapping) and block.get("type") == "text"
        ).strip()
    else:
        result_text = str(content or "")

    return SkillInvocationResult(
        request=request,
        success=True,
        command_name=command_name,
        requested_name=request.normalized_name,
        skill=skill,
        prompt=prompt,
        status="fork",
        fork_result=result_text,
    )


DEFAULT_SKILL_INVOCATION_SERVICE = SkillInvocationService(
    hook_registrar=_register_skill_hooks,
    fork_executor=_execute_forked_skill,
)


__all__ = [
    "DEFAULT_SKILL_INVOCATION_SERVICE",
    "ForkExecutor",
    "HookRegistrar",
    "Origin",
    "SkillInvocationError",
    "SkillInvocationErrorCode",
    "SkillInvocationOrigin",
    "SkillInvocationRequest",
    "SkillInvocationResult",
    "SkillInvocationService",
    "apply_skill_context_modifier",
    "build_request_context_modifier",
]
