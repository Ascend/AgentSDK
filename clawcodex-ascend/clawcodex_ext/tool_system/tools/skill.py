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

# pylint: disable=not-callable
from __future__ import annotations

import importlib
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

from clawcodex_ext.permissions.rules import (
    get_allow_rules,
    get_ask_rules,
    get_deny_rules,
)
from clawcodex_ext.permissions.types import (
    PermissionAllowDecision,
    PermissionAskDecision,
    PermissionDenyDecision,
    PermissionPassthroughResult,
    RuleDecisionReason,
)
from clawcodex_ext.skills.invocation import (
    DEFAULT_SKILL_INVOCATION_SERVICE,
    SkillInvocationOrigin,
    SkillInvocationRequest,
    SkillInvocationResult,
    apply_skill_context_modifier,
    effective_skill_root,
)

from ..build_tool import Tool, ValidationResult, build_tool
from ..context import ToolContext
from ..errors import ToolInputError
from ..protocol import ToolResult


# ---------------------------------------------------------------------------
# Prompt (ported from TS SkillTool/prompt.ts getPrompt)
# ---------------------------------------------------------------------------

SKILL_TOOL_PROMPT = """\
Execute a skill within the main conversation

When users ask you to perform tasks, check if any of the available skills match. Skills provide specialized capabilities and domain knowledge.

When users reference a "slash command" or "/<something>" (e.g., "/commit", "/review-pr"), they are referring to a skill. Use this tool to invoke it.

How to invoke:
- Set `skill` to the exact name of an available skill (no leading slash). For plugin-namespaced skills use the fully qualified `plugin:skill` form.
- Set `args` to pass optional arguments.

Important:
- Available skills are listed in system-reminder messages in the conversation
- Only invoke a skill that appears in that list, or one the user explicitly typed as `/<name>` in their message. Never guess or invent a skill name from training data; otherwise do not call this tool
- When a skill matches the user's request, this is a BLOCKING REQUIREMENT: invoke the relevant Skill tool BEFORE generating any other response about the task
- NEVER mention a skill without actually calling this tool
- Do not invoke a skill that is already running
- Do not use this tool for built-in CLI commands (like /help, /clear, etc.)
- If you see a <command-name> tag in the current conversation turn, the skill has ALREADY been loaded - follow the instructions directly instead of calling this tool again
"""


# ---------------------------------------------------------------------------
# Input validation (ported from TS SkillTool/SkillTool.ts validateInput)
# ---------------------------------------------------------------------------


def _validate_skill_input(tool_input: dict[str, Any], context: ToolContext) -> ValidationResult:
    """Validate the model surface through the canonical invocation service."""
    skill = tool_input.get("skill")

    # Legacy Python modules are a local compatibility API, not prompt skills.
    # Production model dispatch rejects them with the established code 5.
    if not skill and tool_input.get("name"):
        return ValidationResult.fail(
            "Legacy Python modules are not prompt-based skills",
            error_code=5,
        )

    if not isinstance(skill, str):
        return ValidationResult.fail(
            "Missing skill name. Pass the slash command name as the skill parameter "
            '(e.g., skill: "commit" for /commit, skill: "review-pr" for /review-pr).',
            error_code=1,
        )

    request = SkillInvocationRequest(
        skill_name=skill,
        args=str(tool_input.get("args", "") or ""),
        origin=SkillInvocationOrigin.MODEL,
    )
    result = DEFAULT_SKILL_INVOCATION_SERVICE.validate(request, context)
    if result.success:
        return ValidationResult.ok()

    error = result.error
    if error is None:  # pragma: no cover - defensive service boundary
        return ValidationResult.fail("Skill validation failed", error_code=2)
    return ValidationResult.fail(
        error.message,
        error_code=error.model_error_code or 2,
    )


# ---------------------------------------------------------------------------
# Permission check (adapted from TS SkillTool/SkillTool.ts checkPermissions)
# ---------------------------------------------------------------------------


def _skill_rule_matches(rule_content: str | None, canonical_name: str) -> bool:
    if rule_content is None:
        return True
    if rule_content.endswith("*"):
        return canonical_name.startswith(rule_content[:-1])
    return canonical_name == rule_content


def _skill_rule_for(
    rules: list[Any],
    canonical_name: str,
) -> Any | None:
    for rule in rules:
        value = rule.rule_value
        if value.tool_name == "Skill" and _skill_rule_matches(
            value.rule_content,
            canonical_name,
        ):
            return rule
    return None


def _skill_check_permissions(
    tool_input: dict[str, Any],
    tool_context: ToolContext | None,
) -> Any:
    """Apply deny > allow > safe-property > ask permission ordering."""
    if tool_context is None:
        return PermissionPassthroughResult()

    raw_name = tool_input.get("skill")
    if not isinstance(raw_name, str) or not raw_name.strip():
        return PermissionPassthroughResult()

    from clawcodex_ext.skills.catalog import resolve

    requested_name = raw_name.strip().lstrip("/")
    skill = resolve(
        requested_name,
        project_root=tool_context.workspace_root,
        session_id=(str(tool_context.session_id) if tool_context.session_id is not None else None),
        include_disabled=True,
    )
    canonical_name = str(getattr(skill, "name", requested_name))
    permission_context = tool_context.permission_context

    deny_rule = _skill_rule_for(get_deny_rules(permission_context), canonical_name)
    if deny_rule is not None:
        return PermissionDenyDecision(
            message=f"Permission to use Skill({canonical_name}) has been denied.",
            decision_reason=RuleDecisionReason(rule=deny_rule),
        )

    allow_rule = _skill_rule_for(get_allow_rules(permission_context), canonical_name)
    if allow_rule is not None:
        return PermissionAllowDecision(
            updated_input=tool_input,
            decision_reason=RuleDecisionReason(rule=allow_rule),
        )

    safe_prompt = bool(
        skill is not None
        and getattr(skill, "type", "prompt") == "prompt"
        and not (getattr(skill, "allowed_tools", None) or [])
        and not getattr(skill, "hooks", None)
        and (getattr(skill, "context", "inline") or "inline") == "inline"
        and not getattr(skill, "agent", None)
        and not getattr(skill, "model", None)
        and not getattr(skill, "effort", None)
    )
    if safe_prompt:
        return PermissionAllowDecision(updated_input=tool_input)

    ask_rule = _skill_rule_for(get_ask_rules(permission_context), canonical_name)
    if ask_rule is not None:
        return PermissionAskDecision(
            message=f"Claude wants to run /{canonical_name}. Allow?",
            updated_input=tool_input,
            decision_reason=RuleDecisionReason(rule=ask_rule),
        )

    return PermissionAskDecision(
        message=f"Claude wants to run /{canonical_name}. Allow?",
        updated_input=tool_input,
    )


# ---------------------------------------------------------------------------
# mapResultToApi (ported from TS SkillTool/SkillTool.ts
#     mapToolResultToToolResultBlockParam)
# ---------------------------------------------------------------------------


def _skill_map_result_to_api(output: Any, tool_use_id: str) -> dict[str, Any]:
    """Format the skill result for the API.

    Inline skills return a short launch message (the full content is injected
    via new_messages or context_modifier). Forked skills include their result
    text.
    """
    if isinstance(output, dict):
        error = output.get("error")
        if error:
            return {
                "type": "tool_result",
                "tool_use_id": tool_use_id,
                "content": f"Skill error: {error}",
            }

        status = output.get("status")
        command_name = output.get("commandName")
        if not command_name:
            return {
                "type": "tool_result",
                "tool_use_id": tool_use_id,
                "content": "Skill error: invalid skill response (missing commandName)",
            }

        if status in {"fork", "forked"}:
            result_text = output.get("result", "")
            return {
                "type": "tool_result",
                "tool_use_id": tool_use_id,
                "content": f'Skill "{command_name}" completed (forked execution).\n\nResult:\n{result_text}',
            }

        # Inline skill (default)
        return {
            "type": "tool_result",
            "tool_use_id": tool_use_id,
            "content": f"Launching skill: {command_name}",
        }

    # Fallback for legacy or unexpected output shapes
    if isinstance(output, str):
        content: str | list[dict[str, Any]] = output
    else:
        content = json.dumps(output) if isinstance(output, dict) else str(output)
    return {
        "type": "tool_result",
        "tool_use_id": tool_use_id,
        "content": content,
    }


# ---------------------------------------------------------------------------
# Call implementation
# ---------------------------------------------------------------------------


def _invocation_result_to_tool_result(
    result: SkillInvocationResult,
    *,
    expose_modifier: bool = True,
    include_user_details: bool = False,
) -> ToolResult:
    """Translate the shared service result without rebuilding skill semantics."""
    if not result.success:
        error = result.error
        if error is None:  # pragma: no cover - defensive service boundary
            output = {"error": "skill invocation failed", "code": "unknown"}
        else:
            output = error.as_dict()
            output["error"] = error.message
        return ToolResult(
            name="Skill",
            output=output,
            is_error=True,
        )

    skill = result.skill
    output: dict[str, Any] = {
        "success": True,
        "status": result.status,
        "commandName": result.command_name,
    }
    if result.status == "fork":
        output["result"] = result.fork_result or ""
    if include_user_details:
        output.update(
            {
                "prompt": result.prompt,
                "loadedFrom": getattr(skill, "loaded_from", None),
                "skillRoot": effective_skill_root(skill),
                "allowedTools": list(getattr(skill, "allowed_tools", None) or []) or None,
                "model": getattr(skill, "model", None),
                "effort": getattr(skill, "effort", None),
            }
        )

    return ToolResult(
        name="Skill",
        output=output,
        new_messages=list(result.new_messages) or None,
        context_modifier=result.context_modifier if expose_modifier else None,
    )


def _skill_call(tool_input: dict[str, Any], context: ToolContext) -> ToolResult:
    skill_name = tool_input.get("skill")
    if isinstance(skill_name, str) and skill_name.strip():
        request = SkillInvocationRequest(
            skill_name=skill_name,
            args=str(tool_input.get("args", "") or ""),
            origin=SkillInvocationOrigin.MODEL,
        )
        return _invocation_result_to_tool_result(DEFAULT_SKILL_INVOCATION_SERVICE.invoke(request, context))

    legacy_name = tool_input.get("name")
    if isinstance(legacy_name, str) and legacy_name.strip():
        return _run_legacy_python_skill(legacy_name.strip(), tool_input.get("input", {}), context)

    raise ToolInputError("either 'skill' (for SKILL.md) or 'name' (for legacy .py) is required")


def _skill_error(message: str) -> ToolResult:
    return ToolResult(name="Skill", output={"error": message}, is_error=True)


def run_markdown_skill(skill_name: str, args: str, context: ToolContext) -> ToolResult:
    """Public expansion entry point shared by typed and model invocations."""
    return _run_markdown_skill(skill_name, args, context)


def run_user_invoked_skill(
    skill_name: str,
    args: str,
    context: ToolContext,
) -> ToolResult:
    """Invoke a slash command with the user-specific gate."""
    request = SkillInvocationRequest(
        skill_name=skill_name,
        args=args or "",
        origin=SkillInvocationOrigin.USER,
    )
    result = DEFAULT_SKILL_INVOCATION_SERVICE.invoke(request, context)
    if result.success and result.context_modifier is not None:
        apply_skill_context_modifier(context, result.context_modifier)
    return _invocation_result_to_tool_result(
        result,
        expose_modifier=False,
        include_user_details=True,
    )


def _run_markdown_skill(
    skill_name: str,
    args: str,
    context: ToolContext,
    *,
    _resolved_skill: Any | None = None,
) -> ToolResult:
    """Compatibility wrapper over the canonical user invocation service."""

    if _resolved_skill is None:
        # The historical helper resolves through the mutable compatibility
        # registry populated by ``get_all_skills``.  Passing that exact object
        # into the canonical service preserves old callers while the normal
        # user/model surfaces continue using immutable workspace catalogues.
        from clawcodex_ext.skills.loader import get_all_skills, get_registered_skill

        get_all_skills(project_root=context.workspace_root)
        _resolved_skill = get_registered_skill(skill_name)

    request = SkillInvocationRequest(
        skill_name=skill_name,
        args=args or "",
        origin=SkillInvocationOrigin.USER,
    )
    # Legacy callers may render the same skill after the command surface has
    # already activated it in this request (the old parity helper did not own
    # recursion state). Temporarily remove only that matching marker.
    active_before = tuple(context.active_skill_names)
    context.active_skill_names = tuple(name for name in active_before if name != skill_name)
    try:
        result = DEFAULT_SKILL_INVOCATION_SERVICE.invoke(
            request,
            context,
            resolved_skill=_resolved_skill,
        )
    finally:
        context.active_skill_names = active_before
    if result.success and result.context_modifier is not None:
        apply_skill_context_modifier(context, result.context_modifier)
    return _invocation_result_to_tool_result(
        result,
        expose_modifier=False,
        include_user_details=True,
    )


def _permission_context_with_skill_bash_rules(base: Any, allowed_tools: list[str] | None) -> Any:
    """Return ``base`` with the skill's ``Bash(...)`` allowed-tools added as
    session allow rules.

    Mirrors TS injecting a skill's ``allowed-tools`` as
    ``alwaysAllowRules.command`` for the duration of the call: commands the skill
    declares auto-allow, while everything else stays gated. Non-Bash entries are
    irrelevant to embedded shell and ignored. A new context is returned; ``base``
    is never mutated.
    """
    from dataclasses import replace

    bash_rules = [t for t in (allowed_tools or []) if t == "Bash" or t.startswith("Bash(")]
    if not bash_rules:
        return base
    # Use the ``command`` source (TS injects allowed-tools as the slash
    # command's own rules), so these never conflate with genuine session grants.
    merged = {src: list(rules) for src, rules in base.always_allow_rules.items()}
    merged["command"] = merged.get("command", []) + bash_rules
    return replace(base, always_allow_rules=merged)


def _make_shell_executor(
    context: ToolContext,
    allowed_tools: list[str] | None,
    *,
    slash_command_name: str,
    shell: str = "auto",
):
    """Return a callable that runs a skill's embedded ``!`` shell command via
    BashTool, **gated through the permission system**.

    The returned executor matches the ``runtime_substitution.ShellExecutor``
    signature ``(command, inline) -> rendered text``. Before running, each
    command is permission-checked exactly like any Bash tool call (deny rules →
    bash safety screen → the skill's declared ``allowed_tools`` Bash rules), with
    ``allowed_tools`` injected as the command's allow rules so declared commands
    run silently. Only a permission ``allow`` runs; an undeclared or
    safety-screened command is hard-denied and rendered inline as an error
    (matching TS ``promptShellExecution``, which fails rather than prompting the
    user mid-expansion). ``bypassPermissions`` mode still runs everything. So
    embedded shell can no longer bypass the gate.
    """
    from .bash import BashTool
    from src.skills.runtime_substitution import (
        format_shell_error,
        format_shell_output,
    )

    # Skill-scoped permission context: the skill's declared Bash commands
    # auto-allow; everything else flows through the normal gate.
    skill_perm_ctx = _permission_context_with_skill_bash_rules(context.permission_context, allowed_tools)

    def _exec(command: str, inline: bool) -> str:
        # Gate the command before running it. Mirrors TS ``promptShellExecution``:
        # only a permission ``allow`` runs the command; anything else (``ask`` or
        # ``deny``) is treated as denied and rendered inline as an error instead
        # of executing — TS hard-denies here rather than prompting the user
        # mid-skill-expansion, and we match that. A skill grants its commands by
        # DECLARING them in ``allowed-tools``; undeclared / safety-screened
        # commands do not run. Fails CLOSED on any gate error.
        try:
            from src.permissions.check import has_permissions_to_use_tool

            decision = has_permissions_to_use_tool(
                BashTool,
                {"command": command},
                skill_perm_ctx,
                tool_use_context=context,
            )
        except Exception as exc:  # noqa: BLE001 — fail closed, never crash render
            return format_shell_error(exc, command, inline=inline)

        if decision.behavior != "allow":
            reason = getattr(decision, "decision_reason", None)
            if reason is not None and getattr(reason, "type", None) == "safetyCheck":
                msg = getattr(decision, "message", None) or "blocked by a safety check"
            else:
                msg = "command not permitted — declare it in the skill's `allowed-tools` (e.g. `Bash(<cmd>:*)`)"
            return format_shell_error(msg, command, inline=inline)

        try:
            tr = BashTool.call({"command": command, "shell": shell}, context)
        except Exception as exc:  # noqa: BLE001 — surface every failure
            return format_shell_error(exc, command, inline=inline)

        output = tr.output if isinstance(tr.output, dict) else {}
        stdout = str(output.get("stdout", ""))
        stderr = str(output.get("stderr", ""))
        exit_code = output.get("exit_code")

        # Treat non-zero exit codes the same way TS' ShellError surfaces
        # — embed the failure text inline so the model sees what went
        # wrong, but keep going so the rest of the prompt still renders.
        if isinstance(exit_code, int) and exit_code != 0:
            err_text = format_shell_output(stdout, stderr, inline=inline)
            err_text = err_text or f"command failed (exit {exit_code})"
            return format_shell_error(err_text, command, inline=inline)

        if tr.is_error:
            err_text = format_shell_output(stdout, stderr, inline=inline) or output.get("error") or "command failed"
            return format_shell_error(str(err_text), command, inline=inline)

        return format_shell_output(stdout, stderr, inline=inline)

    return _exec


def _build_context_modifier(skill: Any) -> Any:
    """Build a context modifier closure from skill frontmatter fields.

    When ``allowed_tools`` is set, filters ``ctx.options.tools`` so the LLM
    only sees base tools plus the skill's domain tools on subsequent turns.
    """
    allowed_tools = getattr(skill, "allowed_tools", None) or []
    model = getattr(skill, "model", None)
    effort = getattr(skill, "effort", None)

    if not allowed_tools and not model and not effort:
        return None

    def _modifier(ctx: ToolContext) -> ToolContext:
        from src.tool_system.build_tool import tool_matches_name

        from clawcodex_ext.agent.constants import SKILL_CONTEXT_BASE_TOOLS

        if allowed_tools:
            current_tools = ctx.options.tools
            if not current_tools:
                current_tools = list(ctx.tool_registry.list_tools()) if ctx.tool_registry else []
            filtered = []
            for tool in current_tools:
                if getattr(tool, "is_mcp", False) or tool.name.startswith("mcp__"):
                    filtered.append(tool)
                    continue
                if tool.name.lower() in SKILL_CONTEXT_BASE_TOOLS:
                    filtered.append(tool)
                    continue
                if any(tool_matches_name(tool, name) for name in allowed_tools):
                    filtered.append(tool)
                    continue
            ctx.options.tools = filtered

        if model:
            ctx.options.main_loop_model = model

        if effort:
            ctx.options.thinking_config = {"effort": effort}

        return ctx

    return _modifier


def _run_legacy_python_skill(name: str, skill_input: dict[str, Any], context: ToolContext) -> ToolResult:
    """Run a legacy local module without allowing path or symlink escape."""

    if re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]{0,63}", name) is None:
        return _skill_error(f"invalid legacy skill name: {name}")

    skills_dir = _get_skills_dir()
    if skills_dir is None:
        return _skill_error("no skills directory found")

    try:
        root = skills_dir.resolve(strict=True)
        py_path = (root / f"{name}.py").resolve(strict=True)
        py_path.relative_to(root)
    except (FileNotFoundError, OSError, RuntimeError, ValueError):
        return _skill_error(f"legacy skill not found: {name}")
    if not py_path.is_file():
        return _skill_error(f"legacy skill not found: {name}")

    module_name = f"_clawcodex_skill_{name}"
    spec = importlib.util.spec_from_file_location(module_name, py_path)
    if spec is None or spec.loader is None:
        return _skill_error(f"cannot load skill: {name}")

    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    run_fn = getattr(mod, "run", None)
    if not callable(run_fn):
        return _skill_error(f"skill has no run() function: {name}")

    result = run_fn(skill_input, context)
    return ToolResult(name="Skill", output={"output": result})


def _get_skills_dir() -> Path | None:
    env = os.environ.get("CLAWCODEX_SKILLS_DIR")
    if env:
        p = Path(env).expanduser().resolve()
        if p.is_dir():
            return p
    d = Path.home() / ".clawcodex" / "skills"
    if d.is_dir():
        return d
    return None


SkillTool: Tool = build_tool(
    name="Skill",
    input_schema={
        "type": "object",
        "properties": {
            "skill": {
                "type": "string",
                "description": 'The skill name. E.g., "commit", "review-pr", or "pdf"',
            },
            "args": {
                "type": "string",
                "description": "Optional arguments for the skill",
            },
        },
    },
    call=_skill_call,
    prompt=SKILL_TOOL_PROMPT,
    description="Execute a skill within the main conversation",
    map_result_to_api=_skill_map_result_to_api,
    validate_input=_validate_skill_input,
    check_permissions=_skill_check_permissions,
    max_result_size_chars=100_000,
    is_read_only=lambda _input: False,
    is_concurrency_safe=lambda _input: False,
    search_hint="skill run execute invoke slash command",
    to_auto_classifier_input=lambda _input: _input.get("skill", ""),
)
