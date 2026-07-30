"""Team-membership predicates — Chunk F / WI-6.4."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.tool_system.context import ToolContext


def is_team_lead(context: ToolContext) -> bool:
    """True iff the active agent is the team lead."""
    team = getattr(context, "team", None)
    agent_id = getattr(context, "agent_id", None)
    if not isinstance(team, dict) or not agent_id:
        return False
    lead_agent_id = team.get("lead_agent_id")
    return bool(lead_agent_id) and agent_id == lead_agent_id


__all__ = [
    "is_team_lead",
]
