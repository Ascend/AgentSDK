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

# pylint: disable=too-many-lines,ungrouped-imports
"""
Built-in commands for Claw Codex.

Implements core commands like /help, /clear, /exit, /skills, etc.
"""

from __future__ import annotations

import asyncio
import logging
import sys
from typing import Any

from src.context_system.context_analyzer import (
    analyze_context,
    format_context_as_markdown,
)
from clawcodex_ext.context_system.microcompact import (  # pylint: disable=no-name-in-module
    microcompact_messages,
    strip_images_from_messages,
)
from clawcodex_ext.command_system.engine import CommandContext, CommandResult
from clawcodex_ext.command_system.registry import CommandRegistry, get_command_registry
from clawcodex_ext.command_system.types import (
    Command,
    CommandType,
    CompactionResult,
    LocalCommand,
    LocalCommandResult,
    PromptCommand,
)
from clawcodex_ext.command_system.model_command import MODEL_COMMAND
from clawcodex_ext.command_system.effort_command import EFFORT_COMMAND
from clawcodex_ext.command_system.theme_command import THEME_COMMAND
from clawcodex_ext.command_system.export_command import EXPORT_COMMAND
from clawcodex_ext.command_system.output_style_command import OUTPUT_STYLE_COMMAND
from clawcodex_ext.command_system.voice_command import VOICE_COMMAND
from clawcodex_ext.command_system.tts_command import TTS_COMMAND
from clawcodex_ext.command_system.dialogue_command import DIALOGUE_COMMAND
from clawcodex_ext.command_system.statusline import STATUSLINE_COMMAND
from clawcodex_ext.command_system.security_review import SECURITY_REVIEW_COMMAND
from clawcodex_ext.command_system.proactive_command import PROACTIVE_COMMAND
from clawcodex_ext.command_system.btw_command import BTW_COMMAND
from clawcodex_ext.command_system.monitor_command import MONITOR_COMMAND

# Agent Dashboard — ``/dashboard`` cross-system read-only view.
from clawcodex_ext.command_system.dashboard_command import DASHBOARD_COMMAND

# Upstream 0573f4c new slash commands. The implementations live in
# src/command_system/ (real upstream modules, not facades); their Command
# types are runtime-identical to the downstream ones because
# src.command_system.types delegates every attribute to
# clawcodex_ext.command_system.types via __getattr__, so the objects drop
# straight into BUILTIN_COMMANDS. ``resume`` is intentionally excluded —
# the downstream RESUME_COMMAND below already owns that name.
from src.command_system.doctor_command import DOCTOR_COMMAND
from src.command_system.diff_command import DIFF_COMMAND
from src.command_system.tasks_command import TASKS_COMMAND
from src.command_system.permissions_command import PERMISSIONS_COMMAND
from src.command_system.release_notes_command import RELEASE_NOTES_COMMAND
from src.command_system.copy_command import COPY_COMMAND
from src.command_system.vim_command import VIM_COMMAND
from src.command_system.memory_command import MEMORY_COMMAND
from src.command_system.stickers_command import STICKERS_COMMAND
from src.command_system.rename_command import RENAME_COMMAND
from src.command_system.logo_command import LOGO_COMMAND
from src.command_system.mcp_command import MCP_COMMAND
from src.command_system.eco_command import ECO_COMMAND

# Late-binding imports — safe at module level; moved from mid-file deferred sites
# to satisfy ruff E402 (import not at top of file).
from clawcodex_ext.query.outbox_types import CronPromptEvent  # noqa: E402 (used in _append_cron_outbox)
from src.utils.advisor import can_user_configure_advisor as _can_user_configure_advisor  # noqa: E402 (used in ADVISOR_COMMAND)

logger = logging.getLogger(__name__)


# Official Claude Code /init prompts (Simplified)
NEW_INIT_PROMPT = """Set up a CLAWCODEX.md file for this repo. CLAWCODEX.md is loaded into every clawcodex session, so it must be concise — only include what the agent would get wrong without it.

## Step 1: Ask what to set up

Use AskUserQuestion to ask the user:
- "Which CLAWCODEX.md files should /init set up?" with options: "Project CLAWCODEX.md" | "Personal CLAWCODEX.local.md" | "Both project + personal"

Use AskUserQuestion to ask:
- "Also set up skills and hooks?" with options: "Skills + hooks" | "Skills only" | "Hooks only" | "Neither, just CLAWCODEX.md"

## Step 2: Explore the codebase

Use tools to understand the project:
- Read key files: README, package.json, pyproject.toml, Cargo.toml, Makefile, existing CLAWCODEX.md
- Detect: build/test/lint commands, languages, frameworks, project structure
- Detect: code style rules, required env vars, gotchas
- Check for formatter config (ruff, black, prettier, etc.)

## Step 3: Ask follow-up questions (if needed)

Use AskUserQuestion to ask only things you CAN'T figure out from code:
- User's role (e.g., "backend engineer", "new hire")
- Non-obvious workflows or commands
- Communication preferences (terse vs detailed)

## Step 4: Write CLAWCODEX.md

Write a minimal CLAWCODEX.md at the project root.

Include:
- Build/test/lint commands that aren't standard (e.g., "uv run pytest" not just "pytest")
- Code style rules that DIFFER from defaults
- Required env vars or setup steps
- Non-obvious gotchas

Exclude:
- File structure (the agent can discover this)
- Standard conventions the agent already knows
- Generic advice

Prefix with:
```
# CLAWCODEX.md

This file provides guidance to clawcodex when working with code in this repository.
```

If CLAWCODEX.md exists: read it, propose specific improvements.

## Step 5: Write CLAWCODEX.local.md (if user chose personal or both)

Write CLAWCODEX.local.md at project root. Add it to .gitignore.

Include:
- User's role and familiarity with codebase
- Personal sandbox URLs, test accounts
- Communication preferences

## Step 6: Create skills (if user chose skills)

Create skills at `.clawcodex/skills/<name>/SKILL.md`:
```yaml
---
name: <skill-name>
description: <what it does>
---

<Instructions>
```

## Step 7: Create hooks (if user chose hooks)

Before constructing the first hook, invoke the Skill tool once with
`skill: "update-config"` and args beginning with `[hooks-only]`, followed by a
one-line summary of the hook being built. Follow that skill's ClawCodex hook
schema and verification workflow. Reuse the loaded instructions for subsequent
hooks; do not invoke it again for every hook.

Persist hooks only through the Config tool. Do not edit settings JSON directly.

## Step 8: Summary

Tell the user what was set up and suggest any additional optimizations."""

# Fallback prompt for simpler initialization
OLD_INIT_PROMPT = """Please analyze this codebase and create a CLAWCODEX.md file, which will be given to future clawcodex sessions operating in this repository.

What to add:
1. Commands that will be commonly used, such as how to build, lint, and run tests. Include the necessary commands to develop in this codebase, such as how to run a single test.
2. High-level code architecture and structure so that future instances can be productive more quickly. Focus on the "big picture" architecture that requires reading multiple files to understand.

Usage notes:
- If there's already a CLAWCODEX.md, suggest improvements to it.
- When you make the initial CLAWCODEX.md, do not repeat yourself and do not include obvious instructions like "Provide helpful error messages to users", "Write unit tests for all new utilities", "Never include sensitive information (API keys, tokens) in code or commits".
- Avoid listing every component or file structure that can be easily discovered.
- Don't include generic development practices.
- If there are Cursor rules (in .cursor/rules/ or .cursorrules) or Copilot rules (in .github/copilot-instructions.md), make sure to include the important parts.
- If there is a README.md, make sure to include the important parts.
- Do not make up information such as "Common Development Tasks", "Tips for Development", "Support and Documentation" unless this is expressly included in other files that you read.
- Be sure to prefix the file with the following text:

```
# CLAWCODEX.md

This file provides guidance to clawcodex when working with code in this repository.
```"""


def clear_command_call(args: str, context: CommandContext) -> LocalCommandResult:
    """
    Handle /clear command - clear conversation history.

    Args:
        args: Command arguments
        context: Command context

    Returns:
        LocalCommandResult
    """
    from clawcodex_ext.goal.service import clear_goal_for_context  # pylint: disable=no-name-in-module

    # Keep /clear atomic: if an active goal cannot be removed, preserve the
    # conversation instead of hiding a still-running goal behind an empty UI.
    clear_goal_for_context(context)

    if hasattr(context.conversation, "clear"):
        context.conversation.clear()

    if hasattr(context.history, "events"):
        context.history.events.clear()

    # A cleared conversation starts a fresh cache epoch. Keep this a direct
    # reset so /clear is not recorded as a compaction event.
    try:
        from src.context_system.system_prompt_cache import (
            clear_system_prompt_sections,
        )

        clear_system_prompt_sections()
    except Exception:
        logger.debug("Failed to clear system prompt sections", exc_info=True)

    return LocalCommandResult(
        type="text",
        value="Conversation cleared.",
    )


def help_command_call(args: str, context: CommandContext) -> LocalCommandResult:
    """
    Handle /help command - show available commands.

    Args:
        args: Command arguments (optional search query)
        context: Command context

    Returns:
        LocalCommandResult
    """
    registry = get_command_registry()
    query = args.strip()

    if query:
        commands = registry.find_commands(query, limit=50)
        header = f"Commands matching '{query}':"
    else:
        commands = registry.list_commands(include_hidden=False)
        header = "Available commands:"

    lines = [header, ""]

    for cmd in commands:
        alias_str = f" (aliases: {', '.join(cmd.aliases)})" if cmd.aliases else ""
        lines.append(f"  /{cmd.name}{alias_str}")
        lines.append(f"      {cmd.description}")
        if cmd.argument_hint:
            lines.append(f"      Usage: /{cmd.name} {cmd.argument_hint}")
        lines.append("")

    return LocalCommandResult(
        type="text",
        value="\n".join(lines),
    )


def skills_command_call(args: str, context: CommandContext) -> LocalCommandResult:
    """
    Handle /skills command - list available skills or search/rebuild the index.

    Subcommands:
        /skills                    — list all skills
        /skills search <query>     — search skills by TF-IDF
        /skills inspect <name>     — show token breakdown for a skill
        /skills rebuild            — force-rebuild the search index
        /skills stats              — show index statistics
    """
    args = args.strip()

    if args:
        return _skills_subcommand(args, context)

    try:
        from src.skills.loader import get_all_skills

        skills = get_all_skills(project_root=context.cwd or context.workspace_root)
    except Exception:
        skills = []

    if not skills:
        return LocalCommandResult(
            type="text",
            value="No skills available. Add skills to ~/.clawcodex/skills/ or ./.clawcodex/skills/.",
        )

    searcher_section = _try_get_search_stats()

    lines = ["Available skills:", ""]
    for skill in skills:
        lines.append(f"  {skill.name}")
        lines.append(f"      {skill.description}")
        if skill.when_to_use:
            lines.append(f"      When to use: {skill.when_to_use}")
        lines.append("")

    lines.extend(searcher_section)

    return LocalCommandResult(
        type="text",
        value="\n".join(lines),
    )


def _skills_subcommand(args: str, context: CommandContext) -> LocalCommandResult:
    """Dispatch /skills subcommands: search, inspect, rebuild, stats."""
    parts = args.split(maxsplit=1)
    sub = parts[0].lower()
    rest = parts[1] if len(parts) > 1 else ""

    if sub == "search":
        return _skills_search(rest, context)
    if sub == "inspect":
        return _skills_inspect(rest)
    if sub == "rebuild":
        return _skills_rebuild()
    if sub == "stats":
        return _skills_stats()
    return LocalCommandResult(
        type="text",
        value=f"Unknown subcommand: {sub}. Valid: search, inspect, rebuild, stats",
    )


def _skills_search(query: str, context: CommandContext) -> LocalCommandResult:
    if not query:
        return LocalCommandResult(type="text", value="Usage: /skills search <query>")

    searcher = _get_skills_searcher()
    try:
        results = asyncio.run(searcher.search(query))
    except Exception as e:
        return LocalCommandResult(type="text", value=f"Search failed: {e}")

    if not results:
        return LocalCommandResult(type="text", value=f"No matching skills for: {query}")

    lines = [f'Search results for "{query}":', ""]
    for i, r in enumerate(results, 1):
        doc = r.document
        lines.append(f"{i}. {doc.name}  (score: {r.score:.3f}, source: {doc.source})")
        if doc.description:
            lines.append(f"   {doc.description}")
        if r.reason:
            lines.append(f"   {r.reason}")
        lines.append("")

    return LocalCommandResult(type="text", value="\n".join(lines))


def _skills_inspect(name: str) -> LocalCommandResult:
    if not name:
        return LocalCommandResult(type="text", value="Usage: /skills inspect <name>")

    searcher = _get_skills_searcher()
    try:
        asyncio.run(searcher.ensure_index())
    except Exception as e:
        return LocalCommandResult(type="text", value=f"Cannot inspect: {e}")

    result = searcher.inspect(name)
    if result is None:
        return LocalCommandResult(type="text", value=f"Skill not found in index: {name}")

    lines = [
        f"Inspect: {result.name}",
        f"  Source: {result.source}",
        f"  Total tokens: {result.token_count}",
        "",
        "  Per-field breakdown:",
    ]
    for field_name, field_info in result.fields.items():
        lines.append(f"    {field_name}: {field_info.token_count} tokens")
        if field_info.token_sample:
            sample = ", ".join(field_info.token_sample[:10])
            lines.append(f"      sample: {sample}")

    return LocalCommandResult(type="text", value="\n".join(lines))


def _skills_rebuild() -> LocalCommandResult:
    searcher = _get_skills_searcher()
    try:
        asyncio.run(searcher.refresh())
        stats = searcher.stats()
        if stats:
            return LocalCommandResult(
                type="text",
                value=f"Index rebuilt: {stats.total_docs} docs, {stats.total_terms} terms",
            )
        return LocalCommandResult(type="text", value="Index rebuilt successfully.")
    except Exception as e:
        return LocalCommandResult(type="text", value=f"Rebuild failed: {e}")


def _skills_stats() -> LocalCommandResult:
    searcher = _get_skills_searcher()
    try:
        asyncio.run(searcher.ensure_index())
    except Exception:
        return LocalCommandResult(type="text", value="Index not loaded (feature flag may be off).")

    stats = searcher.stats()
    if stats is None:
        return LocalCommandResult(type="text", value="Index not loaded.")

    pinned = searcher.get_pinned()
    lines = [
        "Skill Search Index Stats",
        "=======================",
        f"  Documents:     {stats.total_docs}",
        f"  Unique terms:  {stats.total_terms}",
        f"  Inverted size: {stats.total_inverted_entries} entries",
        f"  Approx memory: {stats.approximate_bytes} bytes",
        f"  Pinned skills: {len(pinned)}",
    ]
    if pinned:
        lines.append(f"    {', '.join(pinned)}")

    return LocalCommandResult(type="text", value="\n".join(lines))


def _try_get_search_stats() -> list[str]:
    """Return a summary line about the search index, or empty list."""
    try:
        searcher = _get_skills_searcher()
        stats = searcher.stats()
        if stats is not None:
            return [
                "",
                f"Search index: {stats.total_docs} docs indexed.",
                "Use /skills search <query> to find relevant skills.",
            ]
    except Exception:
        logger.debug("Failed to build skill search index stats", exc_info=True)
    return []


def _get_skills_searcher():
    """Lazily create and cache a SkillSearcher singleton for the command system."""
    from extensions.skills_ext.registry_ext import get_default_registry

    from clawcodex_ext.services.skill_search.config import SkillSearchConfig
    from clawcodex_ext.services.skill_search.searcher import SkillSearcher
    from clawcodex_ext.services.skill_search.tokenizer import create_default_tokenizer

    searcher: SkillSearcher | None = getattr(_get_skills_searcher, "_instance", None)
    if searcher is None:
        config = SkillSearchConfig.from_feature_gate()
        registry = get_default_registry()
        tokenizer = create_default_tokenizer(cjk_word_tokenizer=None)
        searcher = SkillSearcher(registry, config=config, tokenizer=tokenizer)
        _get_skills_searcher._instance = searcher  # type: ignore[attr-defined]
        # Start watcher for incremental index updates (P92-E).
        if config.enabled:
            searcher.create_watcher().start()
    return searcher


def exit_command_call(args: str, context: CommandContext) -> LocalCommandResult:
    """
    Handle /exit command - exit the application.

    Args:
        args: Command arguments
        context: Command context

    Returns:
        LocalCommandResult
    """
    return LocalCommandResult(
        type="text",
        value="Goodbye!",
    )


def _call_cron_tool(
    context: CommandContext,
    name: str,
    tool_input: dict[str, Any],
) -> Any:
    registry = getattr(context, "tool_registry", None)
    tool_context = getattr(context, "tool_context", None)
    if registry is None or tool_context is None:
        raise ValueError("Cron runtime is not available in this command context")

    from clawcodex_ext.tool_system.protocol import ToolCall

    result = registry.dispatch(ToolCall(name=name, input=tool_input), tool_context)
    if result.is_error:
        output = result.output
        if isinstance(output, dict) and output.get("error"):
            raise ValueError(str(output["error"]))
        raise ValueError(f"{name} failed")
    return result.output


def _has_cron_tool_runtime(context: CommandContext) -> bool:
    return getattr(context, "tool_registry", None) is not None and getattr(context, "tool_context", None) is not None


def _cron_runtime_required_result(action: str) -> LocalCommandResult:
    return LocalCommandResult(
        type="text",
        value=f"Cron runtime is required to {action}; no changes were made.",
    )


def _cron_session_store(context: CommandContext) -> Any:
    tool_context = getattr(context, "tool_context", None)
    return getattr(tool_context, "crons", None)


def _cron_deep_arg(args: str) -> bool:
    return "--deep" in (args or "").split()


def _append_cron_outbox(context: CommandContext, run: dict[str, Any]) -> bool:
    tool_context = getattr(context, "tool_context", None)
    outbox = getattr(tool_context, "outbox", None)
    if not hasattr(outbox, "append"):
        return False
    try:
        outbox.append(
            CronPromptEvent(
                prompt=run["prompt"],
                task_id=run["task_id"],
                run_id=run["id"],
            )
        )
    except Exception:
        return False
    return True


def _format_cron_job(job: dict[str, Any]) -> str:
    kind = "recurring" if job.get("recurring") else "one-shot"
    durable = "durable" if job.get("durable") else "session"
    next_fire = job.get("nextFireAt")
    next_fire_text = str(next_fire) if next_fire is not None else "not scheduled"
    prompt = str(job.get("prompt") or "").replace("\n", " ").strip()
    if len(prompt) > 80:
        prompt = f"{prompt[:77]}..."
    return f"{job.get('id', '')}  {job.get('cron', '')}  {kind}  {durable}  next={next_fire_text}  {prompt}"


def cron_list_command_call(args: str, context: CommandContext) -> LocalCommandResult:
    deep = _cron_deep_arg(args)
    if not _has_cron_tool_runtime(context):
        from clawcodex_ext.cron_system.status import build_schedule_list  # pylint: disable=no-name-in-module

        body = build_schedule_list(context.workspace_root, deep=deep)
        if deep:
            body = f"{body}\n(deep mode)"
        return LocalCommandResult(type="text", value=body)

    output = _call_cron_tool(context, "CronList", {})
    jobs = output.get("jobs", []) if isinstance(output, dict) else []
    if not jobs:
        return LocalCommandResult(type="text", value="No scheduled cron jobs.")

    lines = ["Scheduled cron jobs:", ""]
    for job in jobs:
        if isinstance(job, dict):
            lines.append(_format_cron_job(job))
    if deep:
        lines.append("(deep mode)")
    return LocalCommandResult(type="text", value="\n".join(lines))


def cron_delete_command_call(args: str, context: CommandContext) -> LocalCommandResult:
    cron_id = (args or "").strip()
    if not cron_id:
        return LocalCommandResult(
            type="text",
            value="Usage: /cron-delete <id>",
        )

    if _has_cron_tool_runtime(context):
        output = _call_cron_tool(context, "CronDelete", {"id": cron_id})
        deleted_id = cron_id
        if isinstance(output, dict) and output.get("id"):
            deleted_id = str(output["id"])
        return LocalCommandResult(
            type="text",
            value=f"Deleted scheduled cron job {deleted_id}.",
        )

    return _cron_runtime_required_result("delete scheduled cron jobs")


def cron_status_command_call(args: str, context: CommandContext) -> LocalCommandResult:
    from clawcodex_ext.cron_system.status import build_autonomy_status  # pylint: disable=no-name-in-module

    return LocalCommandResult(
        type="text",
        value=build_autonomy_status(context.workspace_root, deep=_cron_deep_arg(args)),
    )


def cron_runs_command_call(args: str, context: CommandContext) -> LocalCommandResult:
    from clawcodex_ext.cron_system.status import build_autonomy_runs  # pylint: disable=no-name-in-module

    return LocalCommandResult(
        type="text",
        value=build_autonomy_runs(context.workspace_root, deep=_cron_deep_arg(args)),
    )


def cron_run_command_call(args: str, context: CommandContext) -> LocalCommandResult:
    cron_id = (args or "").strip()
    if not cron_id:
        return LocalCommandResult(type="text", value="Usage: /cron-run <id>")
    if not _has_cron_tool_runtime(context):
        return _cron_runtime_required_result("manually fire scheduled cron jobs")

    output = _call_cron_tool(context, "CronRun", {"id": cron_id})
    if isinstance(output, dict) and output.get("disabled"):
        return LocalCommandResult(type="text", value=str(output.get("message") or "Cron is disabled."))
    if isinstance(output, dict) and output.get("not_found"):
        return LocalCommandResult(
            type="text",
            value=f"No scheduled cron job found with id '{cron_id}'.",
        )

    run = output.get("run") if isinstance(output, dict) else None
    if not isinstance(run, dict):
        return LocalCommandResult(
            type="text",
            value=f"Trigger {cron_id} was not fired because a previous run is still queued or running.",
        )

    value = f"Trigger {cron_id} fired.\nRun ID: {run['id']}"
    if _append_cron_outbox(context, run):
        value = f"{value}\nQueued for execution in this session."
    else:
        value = f"{value}\nQueued, but no active cron outbox is available in this context."
    return LocalCommandResult(type="text", value=value)


def cost_command_call(args: str, context: CommandContext) -> LocalCommandResult:
    """
    Handle /cost command - show session cost and usage.

    Reads from the bootstrap singleton's per-model usage accumulators,
    which are populated by ``CostTracker.record_usage()`` (called by the
    query loop, compaction, and advisor after every API response).

    Args:
        args: Command arguments
        context: Command context

    Returns:
        LocalCommandResult
    """
    # Lazy imports to avoid circular dependency at module level.
    from src.bootstrap.state import (
        get_model_usage,
        get_total_cache_creation_input_tokens,
        get_total_cache_read_input_tokens,
        get_total_cost_usd,
        get_total_input_tokens,
        get_total_output_tokens,
        has_unknown_model_cost,
    )

    tracker = context.cost_tracker
    if tracker is None:
        return LocalCommandResult(
            type="text",
            value="Cost tracking not available.",
        )

    total_in = get_total_input_tokens()
    total_out = get_total_output_tokens()
    total_cache_creation = get_total_cache_creation_input_tokens()
    total_cache_read = get_total_cache_read_input_tokens()
    total_cost = get_total_cost_usd()
    unknown_cost = has_unknown_model_cost()
    model_usage = get_model_usage()

    # Nothing recorded yet — show a clear "zero" state.
    if not model_usage:
        return LocalCommandResult(
            type="text",
            value="Session Cost:\n\n  No API usage recorded yet.",
        )

    lines = ["Session Cost:", ""]
    lines.append(f"  Total input tokens:        {total_in:>8,}")
    lines.append(f"  Total output tokens:       {total_out:>8,}")
    if total_cache_creation:
        lines.append(f"  Cache creation tokens:     {total_cache_creation:>8,}")
    if total_cache_read:
        lines.append(f"  Cache read tokens:         {total_cache_read:>8,}")

    if unknown_cost:
        lines.append(f"  Estimated cost USD:        ${total_cost:<7.4f}  (some models have unknown pricing)")
    else:
        lines.append(f"  Estimated cost USD:        ${total_cost:<7.4f}")

    # Per-model breakdown (only models with non-zero usage).
    lines.append("")
    lines.append("  Per model:")
    for model_name, usage in sorted(model_usage.items()):
        cost_str = f"(${usage.cost_usd:.4f})" if usage.cost_usd > 0 else "(cost unknown)"
        lines.append(f"    {model_name}:  {usage.input_tokens:>6,} in / {usage.output_tokens:>6,} out  {cost_str}")

    # Legacy events (costHook path) — if any, append as supplemental info.
    if tracker.events:
        lines.append("")
        lines.append("  Legacy events:")
        for event in tracker.events[-10:]:
            lines.append(f"    - {event}")

    return LocalCommandResult(
        type="text",
        value="\n".join(lines),
    )


def context_command_call(args: str, context: CommandContext) -> LocalCommandResult:
    """
    Handle /context command - show token usage breakdown.

    Args:
        args: Command arguments
        context: Command context

    Returns:
        LocalCommandResult with Markdown table of context usage
    """
    try:
        # Get conversation messages in API format
        conversation_api: list[dict[str, Any]] = []
        if hasattr(context.conversation, "get_messages"):
            conversation_api = context.conversation.get_messages()
        elif hasattr(context.conversation, "messages"):
            # Fall back for simple mock conversations
            for msg in context.conversation.messages:
                role = getattr(msg, "role", "unknown")
                content = getattr(msg, "content", "")
                conversation_api.append({"role": role, "content": content})

        # Get system prompt from config
        system_prompt = context.config.get("system_prompt", "")

        # Get tool schemas from config
        tool_schemas = context.config.get("tool_schemas", [])

        # Get MCP tools info from config
        mcp_tools = context.config.get("mcp_tools", [])

        # Get custom agents info from config
        custom_agents = context.config.get("custom_agents", [])

        # Get CLAWCODEX.md content
        clawcodex_md_content = ""
        try:
            from clawcodex_ext.context_system.clawcodex_md import get_clawcodex_mds, get_memory_files  # pylint: disable=no-name-in-module

            async def _load():
                files = await get_memory_files(cwd=str(context.cwd or context.workspace_root))
                return get_clawcodex_mds(files)

            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = None
            if loop and loop.is_running():
                import concurrent.futures

                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                    clawcodex_md_content = pool.submit(asyncio.run, _load()).result(timeout=10)
            else:
                clawcodex_md_content = asyncio.run(_load())
        except Exception:
            logger.debug("Failed to load CLAWCODEX.md content", exc_info=True)

        # Get model from config
        model = context.config.get("model", "claude-sonnet-4-6")

        # Get skills info from config
        skills_frontmatter_tokens = context.config.get("skills_tokens", 0)
        skills_count = context.config.get("skills_count", 0)

        # Get API usage from cost tracker
        api_usage = None
        if hasattr(context.cost_tracker, "last_usage"):
            api_usage = context.cost_tracker.last_usage

        # Get auto-compact info from config
        auto_compact_threshold = context.config.get("auto_compact_threshold")
        is_auto_compact_enabled = context.config.get("is_auto_compact_enabled", False)

        data = analyze_context(
            conversation_api_messages=conversation_api,
            model=model,
            system_prompt=system_prompt,
            tool_schemas=tool_schemas,
            clawcodex_md_content=clawcodex_md_content,
            skills_frontmatter_tokens=skills_frontmatter_tokens,
            skills_count=skills_count,
            api_usage=api_usage,
            mcp_tools=mcp_tools,
            custom_agents=custom_agents,
            auto_compact_threshold=auto_compact_threshold,
            is_auto_compact_enabled=is_auto_compact_enabled,
        )

        markdown = format_context_as_markdown(data)
        return LocalCommandResult(type="text", value=markdown)
    except Exception as e:
        logger.exception("context analysis failed")
        return LocalCommandResult(type="text", value=f"Context analysis failed: {e}")


async def _compact_async(args: str, context: CommandContext) -> LocalCommandResult:
    """
    Async implementation of compact command.
    """
    if not hasattr(context.conversation, "messages"):
        return LocalCommandResult(
            type="text",
            value="No conversation to compact.",
        )

    messages = context.conversation.messages
    if len(messages) < 2:
        return LocalCommandResult(
            type="text",
            value=f"Nothing to compact: only {len(messages)} messages.",
        )

    # Get provider from config
    provider = context.config.get("provider")
    if provider is None:
        return LocalCommandResult(
            type="text",
            value="Compact requires an LLM provider (not available in this context).",
        )

    model = context.config.get("model", "claude-sonnet-4-6")
    custom_instructions = args.strip() or None

    try:
        # Import here to avoid circular imports
        from src.compact_service.service import compact_conversation

        result = await compact_conversation(
            conversation=context.conversation,
            provider=provider,
            model=model,
            custom_instructions=custom_instructions,
            trigger="manual",
        )
        return LocalCommandResult(
            type="compact",
            value=result.user_display_message or "Conversation compacted.",
            compaction_result=CompactionResult(
                pre_compact_count=result.pre_compact_count,
                post_compact_count=result.post_compact_count,
                tokens_saved=result.tokens_saved,
                trigger=result.trigger,
                summary_preview=result.summary_text[:200] if len(result.summary_text) > 200 else result.summary_text,
            ),
        )
    except ValueError as e:
        return LocalCommandResult(type="text", value=str(e))
    except Exception as e:
        logger.exception("compact failed")
        return LocalCommandResult(
            type="text",
            value=f"Compact failed: {e}",
        )


def _read_current_advisor_model(context: CommandContext) -> str | None:
    """Resolve the user's currently configured advisor model.

    Prefers the reactive AppState store if the caller wired one (so a
    just-issued ``/advisor`` from this session reads its own write
    without a settings-cache roundtrip), and falls back to the
    persisted settings — the source of truth on every restart.
    """
    store = getattr(context, "app_state_store", None)
    if store is not None:
        try:
            state = store.get_state()
            value = getattr(state, "advisor_model", None)
            if value:
                return value
        except Exception:
            logger.debug("Failed to read advisor_model from app state store", exc_info=True)
    try:
        from src.settings.settings import get_settings

        configured = (get_settings().advisor_model or "").strip()
        return configured or None
    except Exception:
        return None


def _write_advisor_model(context: CommandContext, value: str | None) -> None:
    """Persist a new advisor_model and update the reactive store if present.

    Both writes are idempotent. When an AppState store is wired (e.g.
    tests, future TUI wiring), ``replace_state`` fires
    ``_on_advisor_model_change`` which itself writes to settings — so
    we skip the direct settings write to avoid double-saving. When the
    store is absent (the current TUI configuration), we update settings
    directly via the same chokepoint the handler uses.
    """
    store = getattr(context, "app_state_store", None)
    if store is not None:
        from src.state.app_state import replace_state

        store.set_state(lambda s: replace_state(s, advisor_model=value or None))
        return
    # No reactive store — write straight to settings + invalidate cache
    # so the next API call picks up the change. Use the shared default
    # ConfigManager (instead of a fresh one) so the in-process
    # ``_global_cache`` field stays consistent for callers that read
    # via ``load_config()`` / ``_get_default_manager().get_merged()``.
    from src import config as cfg_mod
    from src.settings.settings import invalidate_settings_cache

    mgr = cfg_mod._get_default_manager()
    cfg = mgr.load_global()
    settings_section = cfg.get("settings")
    if not isinstance(settings_section, dict):
        settings_section = {}
    settings_section["advisor_model"] = value or ""
    cfg["settings"] = settings_section
    mgr.save_global(cfg)
    invalidate_settings_cache()


def _read_current_advisor_provider(context: CommandContext) -> str:
    """Resolve the current advisor_provider (store preferred, settings
    fallback). Empty string = unset.
    Mirrors ``_read_current_advisor_model``.
    """
    store = getattr(context, "app_state_store", None)
    if store is not None:
        try:
            v = getattr(store.get_state(), "advisor_provider", None)
            return (v or "").strip()
        except Exception:
            logger.debug("Failed to read advisor_provider from app state store", exc_info=True)
    try:
        from src.settings.settings import get_settings

        return (getattr(get_settings(), "advisor_provider", "") or "").strip()
    except Exception:
        return ""


def _write_advisor_provider(context: CommandContext, value: str | None) -> None:
    """Persist advisor_provider (store preferred, settings fallback).
    Mirrors ``_write_advisor_model`` — same dual-path persistence.
    Empty / None clears the field.
    """
    normalized = (value or "").strip()
    store = getattr(context, "app_state_store", None)
    if store is not None:
        from src.state.app_state import replace_state

        store.set_state(lambda s: replace_state(s, advisor_provider=(normalized or None)))
        return
    from src import config as cfg_mod
    from src.settings.settings import invalidate_settings_cache

    mgr = cfg_mod._get_default_manager()
    cfg = mgr.load_global()
    settings_section = cfg.get("settings")
    if not isinstance(settings_section, dict):
        settings_section = {}
    settings_section["advisor_provider"] = normalized
    cfg["settings"] = settings_section
    mgr.save_global(cfg)
    invalidate_settings_cache()


def _list_configured_providers() -> list[str]:
    """Return the set of provider keys configured in
    ``~/.clawcodex/config.json``. Used by /advisor to validate that the
    user-supplied provider prefix refers to a real entry.
    """
    try:
        from src import config as cfg_mod

        mgr = cfg_mod._get_default_manager()
        cfg = mgr.load_global()
        providers = cfg.get("providers")
        if isinstance(providers, dict):
            return sorted(providers.keys())
    except Exception:
        logger.debug("Failed to load provider configuration", exc_info=True)
    return []


def _read_current_advisor_client_mode(context: CommandContext) -> bool:
    """Resolve the user's current advisor_client_mode flag (reactive
    store preferred, settings fallback). Mirrors
    ``_read_current_advisor_model``.
    """
    store = getattr(context, "app_state_store", None)
    if store is not None:
        try:
            return bool(getattr(store.get_state(), "advisor_client_mode", False))
        except Exception:
            logger.debug("Failed to read advisor_client_mode from app state store", exc_info=True)
    try:
        from src.settings.settings import get_settings

        return bool(getattr(get_settings(), "advisor_client_mode", False))
    except Exception:
        return False


def _write_advisor_client_mode(context: CommandContext, value: bool) -> None:
    """Persist advisor_client_mode (store-preferred, settings fallback).
    Mirrors ``_write_advisor_model`` — same dual-path persistence.
    """
    store = getattr(context, "app_state_store", None)
    if store is not None:
        from src.state.app_state import replace_state

        store.set_state(lambda s: replace_state(s, advisor_client_mode=bool(value)))
        return
    from src import config as cfg_mod
    from src.settings.settings import invalidate_settings_cache

    mgr = cfg_mod._get_default_manager()
    cfg = mgr.load_global()
    settings_section = cfg.get("settings")
    if not isinstance(settings_section, dict):
        settings_section = {}
    settings_section["advisor_client_mode"] = bool(value)
    cfg["settings"] = settings_section
    mgr.save_global(cfg)
    invalidate_settings_cache()


def advisor_command_call(args: str, context: CommandContext) -> LocalCommandResult:
    """Handle /advisor — configure the reviewer model.

    Required argument shape: ``<provider>:<model>`` — both halves
    explicit, separated by the first colon (so model strings
    containing ``/`` or further ``:`` are preserved verbatim).
    Provider must match a key in ``~/.clawcodex/config.json``'s
    ``providers`` map; clawcodex is multi-provider and the same
    model name (e.g. ``claude-opus-4-7``) can sit behind anthropic,
    openai (litellm), openrouter, bedrock, etc. Name-based inference
    was ambiguous and silently routed to the wrong endpoint.

    Branches (after parsing optional ``--client`` / ``--no-client``):
      * no args, no flags → status report (provider/model + mode).
      * ``unset`` | ``off`` → clear advisor_model, advisor_provider,
        and advisor_client_mode.
      * ``--no-client`` alone → keep model+provider, clear client-mode.
      * ``--client`` alone → keep model+provider, set client-mode.
      * ``<provider>:<model>`` → validate provider exists in config,
        persist both fields together. ``--client`` flag (if present)
        also persists advisor_client_mode.

    Examples:
      * ``/advisor anthropic:claude-opus-4-7``  (direct Anthropic API)
      * ``/advisor openai:claude-opus-4-7``     (litellm/proxy via openai provider)
      * ``/advisor openrouter:anthropic/claude-opus-4.1``
      * ``/advisor gemini:gemini-2.5-pro``
    """
    from src.models.model import canonical_model_name, resolve_model
    from src.models.validation import validate_model_name
    from src.utils.advisor import (
        ADVISOR_MODE_CLIENT_SIDE,
        ADVISOR_MODE_INACTIVE,
        ADVISOR_MODE_SERVER_SIDE,
        decide_advisor_mode,
    )

    provider = getattr(context, "provider", None)

    # Hard-reject only when env-disabled (the user would silently
    # configure a value that no request can use).
    if not _can_user_configure_advisor(provider):
        return LocalCommandResult(
            type="text",
            value=("Advisor is disabled by the CLAUDE_CODE_DISABLE_ADVISOR_TOOL env var."),
        )

    # Tokenize raw args so flag handling is order-insensitive. A
    # trailing or leading ``--client`` / ``--no-client`` should peel
    # off cleanly without breaking the model identifier.
    raw_tokens = (args or "").strip().split()
    force_client_flag: bool | None = None  # None = no flag passed
    rest_tokens: list[str] = []
    for tok in raw_tokens:
        if tok == "--client":
            force_client_flag = True
        elif tok == "--no-client":
            force_client_flag = False
        else:
            rest_tokens.append(tok)
    arg = " ".join(rest_tokens).strip()
    arg_lower = arg.lower()

    current_advisor = _read_current_advisor_model(context)
    current_provider = _read_current_advisor_provider(context)
    current_client_mode = _read_current_advisor_client_mode(context)

    main_loop_model = ""
    if provider is not None:
        main_loop_model = getattr(provider, "model", "") or ""
    if not main_loop_model:
        store = getattr(context, "app_state_store", None)
        if store is not None:
            try:
                main_loop_model = getattr(store.get_state(), "main_loop_model", "") or ""
            except Exception:
                logger.debug("Failed to read main_loop_model from app state store", exc_info=True)

    def _render_status() -> str:
        """Format the current state for the no-args branch."""
        if not current_advisor or not current_provider:
            # Either field missing → effectively unset. Show what's
            # there (if anything) so users can fix a partial config.
            partial = ""
            if current_advisor and not current_provider:
                partial = (
                    f"\n(Found advisor_model={current_advisor!r} but no "
                    "advisor_provider — clear with /advisor unset then "
                    "re-run with the explicit syntax.)"
                )
            elif current_provider and not current_advisor:
                partial = (
                    f"\n(Found advisor_provider={current_provider!r} but no advisor_model — clear with /advisor unset.)"
                )
            # Critic C1: surface advisor_client_mode even on partial
            # configs so the user sees stored state that would silently
            # activate as soon as both fields land.
            if current_client_mode:
                partial += (
                    "\n(advisor_client_mode is ON but won't engage "
                    "until both advisor_model and advisor_provider "
                    "are set.)"
                )
            providers = _list_configured_providers()
            providers_hint = f"Configured providers: {', '.join(providers)}.\n" if providers else ""
            return (
                "Advisor: not set\n"
                f"{providers_hint}"
                'Use "/advisor <provider>:<model>" to enable, e.g.:\n'
                "  /advisor anthropic:claude-opus-4-7   (direct Anthropic)\n"
                "  /advisor openai:claude-opus-4-7      (via openai-compat, "
                "e.g. litellm)\n"
                "  /advisor openrouter:anthropic/claude-opus-4.1"
                f"{partial}"
            )
        mode = decide_advisor_mode(
            provider,
            main_loop_model,
            current_advisor,
            force_client_mode=current_client_mode,
            advisor_provider=current_provider,
        )
        mode_label = {
            ADVISOR_MODE_SERVER_SIDE: "active (server-side)",
            ADVISOR_MODE_CLIENT_SIDE: "active (client-side)",
            ADVISOR_MODE_INACTIVE: "inactive",
        }.get(mode, "inactive")
        suffix = ""
        if current_client_mode:
            suffix = " [--client forced]"
        return (
            f"Advisor: {current_provider}:{current_advisor} — {mode_label}{suffix}\n"
            'Use "/advisor unset" to disable or '
            '"/advisor <provider>:<model>" to change.'
        )

    # No model arg, no flags → status only.
    if not arg and force_client_flag is None:
        return LocalCommandResult(type="text", value=_render_status())

    # --no-client alone (no model) → just clear the forced-client flag.
    if not arg and force_client_flag is False:
        if not current_client_mode:
            return LocalCommandResult(
                type="text",
                value="Advisor client mode already off.",
            )
        _write_advisor_client_mode(context, False)
        return LocalCommandResult(
            type="text",
            value=("Advisor client mode disabled. Server-side will be used when applicable."),
        )

    # --client alone (no model) → just turn on the forced-client flag.
    # Critic C2: both fields are required before the flag matters; a
    # partial config + --client would silently fail at request time.
    if not arg and force_client_flag is True:
        if not current_advisor or not current_provider:
            return LocalCommandResult(
                type="text",
                value=(
                    "Cannot force client mode: advisor is not fully "
                    'configured. Use "/advisor <provider>:<model> '
                    '--client" together.'
                ),
            )
        if current_client_mode:
            return LocalCommandResult(
                type="text",
                value="Advisor client mode already on.",
            )
        _write_advisor_client_mode(context, True)
        return LocalCommandResult(
            type="text",
            value=("Advisor client mode enabled. The advisor will run via client-side dispatch on every request."),
        )

    if arg_lower in ("unset", "off"):
        previous_model = current_advisor
        previous_provider = current_provider
        if previous_model:
            _write_advisor_model(context, None)
        if previous_provider:
            _write_advisor_provider(context, None)
        if current_client_mode:
            _write_advisor_client_mode(context, False)
        if previous_model or previous_provider:
            prior = (
                f"{previous_provider}:{previous_model}"
                if previous_provider and previous_model
                else (previous_model or previous_provider or "?")
            )
            return LocalCommandResult(
                type="text",
                value=f"Advisor disabled (was {prior}).",
            )
        return LocalCommandResult(
            type="text",
            value="Advisor already unset.",
        )

    # Parse <provider>:<model> — provider must be a known config key.
    raw = arg
    if ":" not in raw:
        providers = _list_configured_providers()
        providers_hint = f" Configured providers: {', '.join(providers)}." if providers else ""
        return LocalCommandResult(
            type="text",
            value=(
                "Advisor requires explicit <provider>:<model> syntax.\n"
                f"Got: {raw!r}.{providers_hint}\n"
                "Example: /advisor anthropic:claude-opus-4-7"
            ),
        )
    provider_part, model_part = raw.split(":", 1)
    provider_part = provider_part.strip()
    model_part = model_part.strip()
    if not provider_part or not model_part:
        return LocalCommandResult(
            type="text",
            value=(f"Invalid syntax. Expected <provider>:<model> with both halves non-empty. Got: {raw!r}."),
        )
    # Critic S3: validate against the configured providers list
    # unconditionally. ``_list_configured_providers`` is empty only
    # when ``load_global`` crashes (its try/except swallows then
    # returns []) — in that pathological case a clearer signal is
    # better than a friendly silent-skip that lets bad input through.
    configured = _list_configured_providers()
    if provider_part not in configured:
        return LocalCommandResult(
            type="text",
            value=(
                f"Unknown provider {provider_part!r}. Configured: "
                f"{', '.join(configured) or '(none — check ~/.clawcodex/config.json)'}. "
                "Configure new providers in ~/.clawcodex/config.json."
            ),
        )
    try:
        resolved = resolve_model(model_part)
    except Exception as e:
        return LocalCommandResult(
            type="text",
            value=f"Invalid advisor model: {e}",
        )
    if not validate_model_name(resolved):
        return LocalCommandResult(
            type="text",
            value=f"Unknown model: {model_part} ({resolved})",
        )

    normalized = canonical_model_name(resolved)
    _write_advisor_model(context, normalized)
    _write_advisor_provider(context, provider_part)
    if force_client_flag is True:
        _write_advisor_client_mode(context, True)
    elif force_client_flag is False:
        _write_advisor_client_mode(context, False)

    # Report what mode the chosen pair lands in, so the user can spot
    # mismatches immediately (e.g., they expected server-side but the
    # main model doesn't qualify).
    effective_client_mode = force_client_flag if force_client_flag is not None else current_client_mode
    chosen_mode = decide_advisor_mode(
        provider,
        main_loop_model,
        normalized,
        force_client_mode=effective_client_mode,
        advisor_provider=provider_part,
    )
    if chosen_mode == ADVISOR_MODE_SERVER_SIDE:
        mode_msg = "Will run server-side (Anthropic beta path)."
    elif chosen_mode == ADVISOR_MODE_CLIENT_SIDE:
        mode_msg = "Will run client-side (separate API call)."
    else:
        mode_msg = (
            "Note: advisor is currently inactive (no path applies for "
            f"main loop {main_loop_model!r} + advisor {normalized!r})."
        )
    return LocalCommandResult(
        type="text",
        value=f"Advisor set to {provider_part}:{normalized}. {mode_msg}",
    )


def compact_command_call(args: str, context: CommandContext) -> LocalCommandResult:
    """
    Handle /compact command - compact conversation context.

    Args:
        args: Command arguments
        context: Command context

    Returns:
        LocalCommandResult
    """
    # Run the async version in a new event loop
    try:
        asyncio.get_running_loop()  # raise RuntimeError if in async context
        return _sync_compact_fallback(context)
    except RuntimeError:
        # No running event loop — safe to use asyncio.run
        try:
            return asyncio.run(_compact_async(args, context))
        except Exception:
            logger.exception("compact (asyncio.run fallback) failed")
            return _sync_compact_fallback(context)


def _sync_compact_fallback(context: CommandContext) -> LocalCommandResult:
    """Synchronous fallback when async provider is not available."""
    if not hasattr(context.conversation, "messages"):
        return LocalCommandResult(type="text", value="No conversation to compact.")

    messages = context.conversation.messages
    if len(messages) < 2:
        return LocalCommandResult(
            type="text",
            value=f"Nothing to compact: only {len(messages)} messages.",
        )

    # Get messages after last boundary
    try:
        from clawcodex_ext.compact_service.messages import (  # pylint: disable=no-name-in-module
            create_compact_boundary_message,
            create_compact_summary_message,
            get_messages_after_boundary,
            is_compact_boundary_message,
        )
        from clawcodex_ext.utils.token_estimation import count_messages_tokens  # pylint: disable=no-name-in-module

        after_boundary = get_messages_after_boundary(messages)
        if len(after_boundary) < 2:
            return LocalCommandResult(
                type="text",
                value=f"Nothing to compact: only {len(after_boundary)} messages after boundary.",
            )

        # Count tokens
        api_messages = context.conversation.get_messages()
        pre_tokens = count_messages_tokens(api_messages)

        # Strip images and microcompact
        stripped = strip_images_from_messages(api_messages)
        compacted, saved = microcompact_messages(stripped)

        # Find boundary position
        boundary_indices = [i for i, m in enumerate(messages) if is_compact_boundary_message(m)]

        if boundary_indices:
            insert_pos = max(boundary_indices) + 1
        else:
            insert_pos = 0

        # Create simple text summary
        summary_parts = [f"Conversation had {len(after_boundary)} messages ({pre_tokens:,} tokens)."]
        summary_text = "\n".join(summary_parts)

        boundary = create_compact_boundary_message(
            trigger="manual",
            pre_compact_token_count=pre_tokens,
        )
        summary = create_compact_summary_message(summary_text)

        # Rebuild conversation
        if insert_pos == 0:
            context.conversation.messages.clear()
            context.conversation.messages.append(boundary)
            context.conversation.messages.append(summary)
        else:
            context.conversation.messages = list(messages[:insert_pos])
            context.conversation.messages.append(boundary)
            context.conversation.messages.append(summary)

        return LocalCommandResult(
            type="compact",
            value=f"Compacted: removed {len(after_boundary) - 2} messages ({pre_tokens:,} tokens → ~{saved} saved).",
            compaction_result=CompactionResult(
                pre_compact_count=len(messages),
                post_compact_count=len(context.conversation.messages),
                tokens_saved=saved,
                trigger="manual",
                summary_preview=summary_text[:200],
            ),
        )
    except Exception:
        # Last resort: just clear old messages
        original_count = len(messages)
        if original_count > 10:
            context.conversation.messages = list(messages[-10:])
            return LocalCommandResult(
                type="compact",
                value=f"Compacted: removed {original_count - 10} messages (fallback mode).",
                compaction_result=CompactionResult(
                    pre_compact_count=original_count,
                    post_compact_count=10,
                    tokens_saved=0,
                    trigger="manual",
                ),
            )
        return LocalCommandResult(
            type="text",
            value="Nothing to compact.",
        )


def telemetry_command_call(args: str, context: CommandContext) -> LocalCommandResult:
    """Handle /telemetry command — show & manage telemetry configuration.

    Delegates to the telemetry.cli module, capturing its stdout
    output into the command result. Best-effort: failures (missing
    telemetry package, config errors) return a clear text message
    instead of crashing.

    Subcommands (mirror ``telemetry.cli.main``):
    status — show config, recorder kind, today's summary (default)
    preview — dry-run the reporter output for today
    flush — run the aggregator and emit to reporters
    enable — print the config snippet to enable telemetry
    disable — print the config snippet to disable telemetry
    """
    import io

    # Preserve real stdout/stderr so we can restore them
    real_stdout = sys.stdout
    real_stderr = sys.stderr

    buf = io.StringIO()
    try:
        sys.stdout = buf
        sys.stderr = buf

        from telemetry.cli import main as _telemetry_main

        # Split the slash-command args into argv-style tokens
        argv = args.split() if args else []
        exit_code = _telemetry_main(argv)
    except ImportError:
        return LocalCommandResult(
            type="text",
            value=("Telemetry package not available. Ensure ``telemetry`` is installed and on the Python path."),
        )
    except Exception as exc:
        return LocalCommandResult(
            type="text",
            value=f"Telemetry command failed: {exc}",
        )
    finally:
        sys.stdout = real_stdout
        sys.stderr = real_stderr

    output = buf.getvalue().strip()
    if not output:
        output = "(no output)"

    if exit_code != 0:
        output = f"(exit code {exit_code})\n{output}"

    return LocalCommandResult(type="text", value=output)


def resume_command_call(args: str, context: CommandContext) -> LocalCommandResult:
    """List past sessions and resume one interactively or by ID.

    With a session ID argument, validates it exists and returns instructions.
    Without arguments, shows the interactive session browser.
    """
    from src.services.session_storage import SessionStorage

    session_id = args.strip() if args else ""

    if session_id:
        # Validate session exists
        from clawcodex_ext.services.session_storage import SESSIONS_DIR

        session_path = SESSIONS_DIR / session_id
        if not session_path.is_dir():
            return LocalCommandResult(
                type="text",
                value=f"Session not found: {session_id}\n  Use /resume without arguments to browse available sessions.",
            )
        return LocalCommandResult(
            type="text",
            value=f"Session found: {session_id}\n"
            f"  Use ``clawcodex --resume {session_id}`` or\n"
            f"  use /load {session_id} to restore it in this session.",
        )

    # No session ID — show interactive browser
    try:
        from clawcodex_ext.repl.session_browser import browse_sessions_interactive  # pylint: disable=no-name-in-module

        selected_id = browse_sessions_interactive()
        if selected_id:
            return LocalCommandResult(
                type="text",
                value=f"Session selected: {selected_id}\n"
                f"  Use ``clawcodex --resume {selected_id}`` or\n"
                f"  use /load {selected_id} to restore it in this session.",
            )
        return LocalCommandResult(type="text", value="Session selection cancelled.")
    except Exception:
        # Fallback: list sessions as text
        try:
            metas = SessionStorage.list_sessions(limit=50)
            if not metas:
                return LocalCommandResult(type="text", value="No past sessions found.")
            lines = ["Available sessions (use /resume <session_id> to resume):\n"]
            for i, m in enumerate(metas, 1):
                sid = m.session_id
                preview = getattr(m, "title", "") or getattr(m, "last_user_input", "") or ""
                if preview:
                    preview = preview[:60]
                lines.append(f"  {i:>3}. {sid[:12]}…  {preview}")
            lines.append("\n  Use /resume <session_id> or /load <session_id> to restore a session.")
            return LocalCommandResult(type="text", value="\n".join(lines))
        except Exception:
            return LocalCommandResult(
                type="text",
                value="No past sessions found. Interactive browser unavailable.",
            )


# Command definitions (sorted alphabetically by variable name)
ADVISOR_COMMAND = LocalCommand(
    name="advisor",
    description="Configure the advisor model (server-side on 1P Anthropic, client-side on any provider)",
    argument_hint="[<model> [--client] | --no-client | off]",
    supports_non_interactive=True,
    is_enabled=lambda: _can_user_configure_advisor(None),
)

CLEAR_COMMAND = LocalCommand(
    name="clear",
    description="Clear conversation history",
    aliases=["reset", "new"],
    supports_non_interactive=False,
)

COMPACT_COMMAND = LocalCommand(
    name="compact",
    description="Compact conversation to save context space",
    argument_hint="",
    supports_non_interactive=True,
)

CONTEXT_COMMAND = LocalCommand(
    name="context",
    description="Show current workspace context",
    argument_hint="",
    supports_non_interactive=True,
)

COST_COMMAND = LocalCommand(
    name="cost",
    description="Show session cost and usage",
    argument_hint="",
    supports_non_interactive=True,
)

CRON_LIST_COMMAND = LocalCommand(
    name="cron-list",
    description="List scheduled cron jobs",
    argument_hint="[--deep]",
    supports_non_interactive=True,
)

CRON_DELETE_COMMAND = LocalCommand(
    name="cron-delete",
    description="Delete a scheduled cron job",
    argument_hint="<id>",
    supports_non_interactive=True,
)

CRON_STATUS_COMMAND = LocalCommand(
    name="cron-status",
    description="Show cron autonomy status",
    argument_hint="[--deep]",
    supports_non_interactive=True,
)

CRON_RUNS_COMMAND = LocalCommand(
    name="cron-runs",
    description="Show scheduled-task run history",
    argument_hint="[--deep]",
    supports_non_interactive=True,
)

CRON_RUN_COMMAND = LocalCommand(
    name="cron-run",
    description="Manually fire a scheduled cron job",
    aliases=["cron-fire", "cron-trigger"],
    argument_hint="<id>",
    supports_non_interactive=True,
)

EXIT_COMMAND = LocalCommand(
    name="exit",
    description="Exit the application",
    aliases=["quit", "q"],
    supports_non_interactive=True,
)

HELP_COMMAND = LocalCommand(
    name="help",
    description="Show available commands",
    aliases=["?"],
    argument_hint="[search_query]",
    supports_non_interactive=True,
)

INIT_COMMAND = PromptCommand(
    name="init",
    description="Initialize new CLAWCODEX.md file(s) and optional skills/hooks with codebase documentation",
    markdown_content=NEW_INIT_PROMPT,
    progress_message="analyzing your codebase",
    content_length=0,
    source="builtin",
)

# Port of typescript/src/commands/auto-fix.ts (type: 'prompt'). Text is verbatim.
AUTO_FIX_PROMPT = (
    "The user wants to configure auto-fix settings. Auto-fix automatically runs lint "
    "and test commands after AI file edits, feeding errors back for self-repair.\n\n"
    "Current settings location: `.clawcodex/settings.json` or "
    "`.clawcodex/settings.local.json`\n\n"
    "Example configuration:\n```json\n{\n  \"autoFix\": {\n    \"enabled\": true,\n"
    "    \"lint\": \"eslint . --fix\",\n    \"test\": \"bun test\",\n"
    "    \"maxRetries\": 3,\n    \"timeout\": 30000\n  }\n}\n```\n\n"
    "Ask the user what lint and test commands they use, then help them set up the "
    "configuration."
)

REVIEW_PROMPT = """
You are an expert code reviewer. Follow these steps:

1. If no PR number is provided in the args, run `gh pr list` to show open PRs
2. If a PR number is provided, run `gh pr view <number>` to get PR details
3. Run `gh pr diff <number>` to get the diff
4. Analyze correctness, conventions, performance, tests, security, and risks

Give a concise but thorough review with clear sections and bullet points.

PR number: $ARGUMENTS
"""

AUTO_FIX_COMMAND = PromptCommand(
    name="auto-fix",
    description="Configure auto-fix: run lint/test after AI edits",
    markdown_content=AUTO_FIX_PROMPT,
    progress_message="Configuring auto-fix...",
    content_length=0,
    source="builtin",
)

REVIEW_COMMAND = PromptCommand(
    name="review",
    description="Review a pull request",
    markdown_content=REVIEW_PROMPT,
    progress_message="reviewing pull request",
    content_length=0,
    source="builtin",
)

SKILLS_COMMAND = LocalCommand(
    name="skills",
    description="List available skills or search/reload the skill index",
    argument_hint="[search <query> | inspect <name> | rebuild | stats]",
    supports_non_interactive=True,
)

TELEMETRY_COMMAND = LocalCommand(
    name="telemetry",
    description="Show & manage telemetry (stats collection & error reporting)",
    argument_hint="[status|preview|flush|enable|disable]",
    supports_non_interactive=True,
)

RESUME_COMMAND = LocalCommand(
    name="resume",
    description="List past sessions and resume one interactively, or /resume <session_id>",
    argument_hint="[session_id]",
    supports_non_interactive=True,
)
RESUME_COMMAND.set_call(resume_command_call)


# Synchronous versions for REPL integration
def execute_command_sync(cmd_name: str, args: str, context: CommandContext) -> tuple[bool, str | None, str | None]:
    """
    Execute a command synchronously.

    Returns:
        Tuple of (success: bool, result_text: str | None, error: str | None)
    """
    cmd = get_command_registry().get(cmd_name)
    if cmd is None:
        for builtin_cmd in get_builtin_commands():
            if builtin_cmd.name.lower() == cmd_name.lower() or cmd_name.lower() in [
                a.lower() for a in builtin_cmd.aliases
            ]:
                cmd = builtin_cmd
                break

    if cmd is None:
        return False, None, f"Unknown command: {cmd_name}"
    if cmd.command_type != CommandType.LOCAL:
        return False, None, f"Command not implemented for sync execution: {cmd_name}"
    if isinstance(cmd, LocalCommand) and cmd.run_in_thread:
        return False, None, f"Command requires async execution: {cmd_name}"

    try:
        if isinstance(cmd, LocalCommand) and cmd._call_impl is not None:
            result = cmd._call_impl(args, context)
        else:
            return False, None, f"Command not implemented for sync execution: {cmd_name}"

        return True, result.value, None
    except Exception as e:
        return False, None, str(e)


# Set the call implementations
HELP_COMMAND.set_call(help_command_call)
CLEAR_COMMAND.set_call(clear_command_call)
EXIT_COMMAND.set_call(exit_command_call)
SKILLS_COMMAND.set_call(skills_command_call)
COST_COMMAND.set_call(cost_command_call)
CONTEXT_COMMAND.set_call(context_command_call)
COMPACT_COMMAND.set_call(compact_command_call)
CRON_LIST_COMMAND.set_call(cron_list_command_call)
CRON_DELETE_COMMAND.set_call(cron_delete_command_call)
CRON_STATUS_COMMAND.set_call(cron_status_command_call)
CRON_RUNS_COMMAND.set_call(cron_runs_command_call)
CRON_RUN_COMMAND.set_call(cron_run_command_call)
ADVISOR_COMMAND.set_call(advisor_command_call)
TELEMETRY_COMMAND.set_call(telemetry_command_call)


def get_builtin_commands() -> list[Command]:
    """Get all built-in commands."""
    cmds: list[Command] = [
        ADVISOR_COMMAND,
        CLEAR_COMMAND,
        COMPACT_COMMAND,
        CONTEXT_COMMAND,
        COST_COMMAND,
        CRON_LIST_COMMAND,
        CRON_DELETE_COMMAND,
        CRON_STATUS_COMMAND,
        CRON_RUNS_COMMAND,
        CRON_RUN_COMMAND,
        EXIT_COMMAND,
        HELP_COMMAND,
        INIT_COMMAND,
        MONITOR_COMMAND,
        SKILLS_COMMAND,
        TELEMETRY_COMMAND,
        # Upstream interactive/prompt commands (b24b8cb)
        AUTO_FIX_COMMAND,
        REVIEW_COMMAND,
        SECURITY_REVIEW_COMMAND,
        STATUSLINE_COMMAND,
        PERMISSIONS_COMMAND,
        OUTPUT_STYLE_COMMAND,
        EXPORT_COMMAND,
        THEME_COMMAND,
        ECO_COMMAND,
        EFFORT_COMMAND,
        MODEL_COMMAND,
        PROACTIVE_COMMAND,
        RESUME_COMMAND,
        BTW_COMMAND,
        # Agent Dashboard — cross-system read-only view.
        DASHBOARD_COMMAND,
        # Upstream 0573f4c new slash commands
        DOCTOR_COMMAND,
        DIFF_COMMAND,
        TASKS_COMMAND,
        RELEASE_NOTES_COMMAND,
        COPY_COMMAND,
        VIM_COMMAND,
        MEMORY_COMMAND,
        STICKERS_COMMAND,
        RENAME_COMMAND,
        LOGO_COMMAND,
        MCP_COMMAND,
        # Voice Mode — /voice toggle + STT backend selection.
        VOICE_COMMAND,
        # TTS — /tts toggle, backend selection, and preview.
        TTS_COMMAND,
        # Voice Dialogue — /dialogue full-duplex mode.
        DIALOGUE_COMMAND,
    ]
    try:
        from clawcodex_ext.goal.command import GOAL_COMMAND  # pylint: disable=no-name-in-module

        cmds.append(GOAL_COMMAND)
    except Exception:  # nosec B110
        pass  # This optional feature is unavailable; continue with the core command path.
    try:
        from clawcodex_ext.command_system.ultraplan_command import ULTRAPLAN_COMMAND

        cmds.append(ULTRAPLAN_COMMAND)
    except Exception:  # nosec B110
        pass  # This optional feature is unavailable; continue with the core command path.
    try:
        from extensions.skills_ext.bundled.dream import get_dream_command

        cmds.append(get_dream_command())
    except Exception:
        logger.warning("failed to load the /dream command adapter", exc_info=True)

    from src.command_system.buddy_command import is_buddy_command_enabled, BUDDY_COMMAND

    if is_buddy_command_enabled():
        cmds.append(BUDDY_COMMAND)

    # LKB (Logical Kanban) diagnostic command — always registered so
    # it's discoverable; the handler itself gates on the feature flag.
    try:
        from clawcodex_ext.command_system.lkb_command import LKB_COMMAND  # pylint: disable=no-name-in-module

        cmds.append(LKB_COMMAND)
    except Exception:  # nosec B110
        pass  # This optional feature is unavailable; continue with the core command path.

    # Bundled dynamic-workflow slash commands (/workflows list + /deep-research).
    # Gated by is_workflows_enabled() (env CLAUDE_CODE_DISABLE_WORKFLOWS or
    # settings.disable_workflows can turn it off). The Workflow tool is already
    # registered in clawcodex_ext/tool_system/defaults.py, so the model can
    # invoke it once prompted.
    try:
        from src.workflow.gating import is_workflows_enabled

        if is_workflows_enabled():
            from src.command_system.workflows_integration import (
                bundled_workflow_commands,
            )

            cmds.extend(bundled_workflow_commands())
    except Exception:  # noqa: BLE001 — never let a bad file break the command list  # nosec B110
        pass  # This optional feature is unavailable; continue with the core command path.

    return cmds


def register_builtin_commands(registry: CommandRegistry | None = None) -> None:
    """
    Register all built-in commands.

    Args:
        registry: Optional registry to use (uses global if None)
    """
    reg = registry or get_command_registry()
    try:
        from clawcodex_ext.multimodel.runtime_command import register_multimodel_runtime_command  # pylint: disable=no-name-in-module

        register_multimodel_runtime_command(reg)
    except Exception:  # nosec B110
        pass  # This optional feature is unavailable; continue with the core command path.
    for cmd in get_builtin_commands():
        reg.register(cmd)
    try:
        from clawcodex_ext.away_summary.registration import register_away_summary_commands  # pylint: disable=no-name-in-module
        from clawcodex_ext.intent_forecast.registration import register_intent_forecast_commands  # pylint: disable=no-name-in-module

        register_away_summary_commands(reg)
        register_intent_forecast_commands(reg)
    except Exception:  # nosec B110
        pass  # This optional feature is unavailable; continue with the core command path.
    # BG_SESSIONS — /bg command family. Self-gates on
    # CLAWCODEX_BG_SESSIONS (returns disabled message when off), so
    # unconditional registration is safe.
    try:
        from clawcodex_ext.command_system.bg_commands import register_bg_commands

        register_bg_commands(reg)
    except Exception:  # nosec B110
        pass  # This optional feature is unavailable; continue with the core command path.
    # LODESTONE — /link command family. Self-gates on
    # ``LODESTONE=off`` (renderer falls back to plain text), so the
    # command stays harmless when the feature is disabled.
    try:
        from clawcodex_ext.command_system.lodestone_commands import register_lodestone_commands

        register_lodestone_commands(reg)
    except Exception:  # nosec B110
        pass  # This optional feature is unavailable; continue with the core command path.
    # TEMPLATES — productized template catalogue/render/create surface.
    try:
        from clawcodex_ext.command_system.template_commands import register_template_commands

        register_template_commands(reg)
    except Exception:  # nosec B110
        pass  # This optional feature is unavailable; continue with the core command path.


async def execute_command_async(
    cmd_name: str,
    args: str,
    context: CommandContext,
) -> CommandResult:
    """
    Execute a command asynchronously.

    This function handles both LocalCommand and PromptCommand types.
    For PromptCommand, it returns the prompt content that should be sent to the LLM.

    Args:
        cmd_name: Name of the command to execute
        args: Arguments for the command
        context: Command context

    Returns:
        CommandResult with the execution result
    """
    from clawcodex_ext.command_system.engine import CommandEngine

    registry = get_command_registry()
    cmd = registry.get(cmd_name)

    if cmd is None:
        return CommandResult.failure(cmd_name, f"Unknown command: {cmd_name}")

    if not cmd.is_enabled():
        return CommandResult.failure(cmd_name, f"Command {cmd_name} is disabled")

    engine = CommandEngine(
        registry=registry,
        workspace_root=context.workspace_root,
        context=context,
    )

    # Create a fake command input string for the engine
    command_input = f"/{cmd_name}"
    if args:
        command_input += f" {args}"

    return await engine.execute(command_input)
