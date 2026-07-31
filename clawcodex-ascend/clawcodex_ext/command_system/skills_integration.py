"""
Skills system integration with command system.

Bridges the existing skills system to the command system.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from clawcodex_ext.skills.catalog import get_skill_catalog  # pylint: disable=no-name-in-module
from clawcodex_ext.skills.loader import (  # pylint: disable=no-name-in-module
    PromptSkill,
    load_skills_from_dir,
)
from .registry import CommandRegistry, get_command_registry, register_command
from .types import PromptCommand, SkillPromptCommand


def skill_to_prompt_command(skill: PromptSkill) -> PromptCommand:
    """
    Convert a PromptSkill to a ``SkillPromptCommand``.

    Returns a ``SkillPromptCommand`` (a ``PromptCommand`` subclass) so that when
    this command is executed via the registry, its prompt is rendered through the
    same canonical user invocation service the Skill surfaces use — preserving the
    base-dir header, ``${CLAUDE_SKILL_DIR}`` / ``${CLAUDE_SESSION_ID}``
    substitution, and gated shell-exec. The plain base ``PromptCommand`` renderer
    would drop all of that (P0-6 Option B / Phase 3.5).

    Args:
        skill: The PromptSkill to convert

    Returns:
        SkillPromptCommand instance (typed as its ``PromptCommand`` base)
    """
    return SkillPromptCommand(
        name=skill.name,
        description=skill.description,
        aliases=list(skill.aliases),
        is_enabled=skill.is_enabled,
        argument_hint=skill.argument_hint,
        progress_message=f"Executing {skill.name}...",
        content_length=skill.content_length,
        arg_names=list(skill.arg_names),
        allowed_tools=list(skill.allowed_tools),
        model=skill.model,
        source=skill.loaded_from,
        hooks=dict(skill.hooks or {}),
        skill_root=skill.skill_root,
        context=skill.context or "inline",
        agent=skill.agent,
        effort=skill.effort,
        paths=list(skill.paths) if skill.paths else [],
        markdown_content=skill.markdown_content,
        when_to_use=skill.when_to_use,
        version=skill.version,
        disable_model_invocation=skill.disable_model_invocation,
        user_invocable=skill.user_invocable,
        loaded_from=skill.loaded_from,
        is_hidden=skill.is_hidden,
        has_user_specified_description=skill.has_user_specified_description,
    )


def register_skill_as_command(skill: PromptSkill) -> PromptCommand:
    """
    Register a PromptSkill as a PromptCommand.

    Args:
        skill: The PromptSkill to register

    Returns:
        The registered PromptCommand
    """
    command = skill_to_prompt_command(skill)
    register_command(command)
    return command


def load_and_register_skills(
    project_root: str | Path | None = None,
    user_skills_dir: str | Path | None = None,
    registry: CommandRegistry | None = None,
) -> list[PromptCommand]:
    """
    Load all skills and register them as commands.

    Args:
        project_root: Optional project root directory
        user_skills_dir: Optional user skills directory
        registry: Optional command registry (uses global if None)

    Returns:
        List of registered PromptCommands
    """
    skills = get_skill_catalog(
        project_root=project_root,
        user_skills_dir=user_skills_dir,
    ).skills

    registered_commands: list[PromptCommand] = []
    for skill in skills:
        command = skill_to_prompt_command(skill)
        target_registry = registry or get_command_registry()
        # Built-in/core command names are reserved. A project skill must not
        # replace /review, /help, or another command that was registered first.
        if not target_registry.has(command.name):
            target_registry.register(command)
        registered_commands.append(command)

    return registered_commands


def get_skill_command(name: str) -> Optional[PromptCommand]:
    """
    Get a skill-based command by name.

    Args:
        name: Name of the skill/command

    Returns:
        PromptCommand if found, None otherwise
    """
    skill = get_skill_catalog().resolve(name, include_disabled=True)
    if skill:
        return skill_to_prompt_command(skill)
    return None


def get_skill_tool_commands(
    cwd: str | None = None,
    session_id: str | None = None,
) -> tuple[PromptCommand, ...]:
    """ALL prompt-based commands the model may invoke.

    Source for the model-facing "# Available Skills" system-prompt listing,
    wired into ``build_full_system_prompt_blocks(skills=...)`` at
    ``clawcodex_ext/query/engine.py``. The canonical catalog is keyed by
    workspace and session, so this view must not use the process-global
    command registry.
    """
    snapshot = get_skill_catalog(
        project_root=cwd,
        session_id=session_id,
    )
    commands: list[PromptCommand] = []
    skill_buckets = {
        "skills",
        "user",
        "project",
        "bundled",
        "commands_DEPRECATED",
    }
    for skill in snapshot.skills:
        if getattr(skill, "type", "prompt") != "prompt":
            continue
        if skill.disable_model_invocation or skill.is_hidden:
            continue
        try:
            if not skill.is_enabled():
                continue
        except Exception:
            continue
        if (
            skill.loaded_from not in skill_buckets
            and not skill.has_user_specified_description
            and not skill.when_to_use
        ):
            continue
        commands.append(skill_to_prompt_command(skill))
    return tuple(commands)


def load_skill_from_directory(
    directory: str | Path,
    loaded_from: str = "skills",
) -> list[PromptCommand]:
    """
    Load skills from a directory and convert to commands.

    Args:
        directory: Directory to load skills from
        loaded_from: Source label for the skills

    Returns:
        List of PromptCommands
    """
    skills = load_skills_from_dir(directory, loaded_from=loaded_from)
    return [skill_to_prompt_command(skill) for skill in skills]


async def execute_skill_command(
    command: PromptCommand,
    args: str,
    context: Any,
) -> list[dict[str, Any]]:
    """
    Execute a skill-based prompt command.

    Args:
        command: The PromptCommand to execute
        args: Arguments string
        context: Command context

    Returns:
        Prompt content blocks
    """
    return await command.get_prompt_for_command(args, context)
