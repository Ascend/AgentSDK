"""Team-aware tool pool assembly.

Higher-level wrapper around :func:`extensions.tool_system_ext.team_filter.filter_team_only_tools`
that pulls the full tool list from a ``ToolRegistry`` and applies the
team-context filter in one call. This is the single entry point the
caller sites in ``src/query/agent_loop_compat.py`` and
``clawcodex_ext/repl/core.py`` use to materialize their model-facing
tool list.

Decoupling rationale: the upstream tool system (``src/tool_system``)
owns tool definitions, the extension layer (``extensions/`` and
``clawcodex_ext/``) owns the policy of which tools are visible under
which conditions. Putting the policy in extensions means a single
deployment can override the visibility rule without forking the
upstream registry code.
"""

from __future__ import annotations

from typing import Any

from .registry import ToolRegistry
from extensions.tool_system_ext.team_filter import (
    filter_team_only_tools,
    has_team_context,
)


def get_team_aware_tool_list(
    registry: ToolRegistry,
    team: object,
    context: Any | None = None,
) -> list:
    """Return the model-facing tool list for the current context.

    Active-team-only tools are hidden when ``team`` indicates no active
    team context. ``TeamCreate`` remains visible so the model can
    bootstrap a team. Goal model tools are hidden when the current
    context does not have a persisted session or is a review subagent.

    Args:
        registry: The :class:`ToolRegistry` to enumerate.
        team: The value of ``ToolContext.team`` (a dict when active,
            ``None`` or anything else when not).
        context: The current ``ToolContext`` when available.

    Returns:
        A list of tools, in the registry's natural order, with
        ``SendMessage`` / ``TeamDelete`` removed when no team is active,
        ``TeamCreate`` preserved for bootstrap, and goal tools removed
        when upstream's ``tools_visible`` predicate would be false.
    """
    tools = registry.list_tools()
    if _is_coordinator_mode_active():
        from clawcodex_ext.coordinator.mode import filter_coordinator_tools

        tools = filter_coordinator_tools(tools)
    else:
        tools = filter_team_only_tools(tools, has_team_context(team))
    if context is not None:
        from clawcodex_ext.goal.tools import filter_goal_model_tools_for_context

        tools = filter_goal_model_tools_for_context(tools, context)
    return tools


def _is_coordinator_mode_active() -> bool:
    try:
        from clawcodex_ext.coordinator.mode import is_coordinator_mode

        return is_coordinator_mode()
    except Exception:  # nosec
        return False


__all__ = [
    "get_team_aware_tool_list",
]
