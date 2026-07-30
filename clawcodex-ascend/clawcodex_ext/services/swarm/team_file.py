"""Team file schema + read/write/mutator helpers — Chunk F / WI-6.4."""

from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Literal

BackendType = Literal["in-process", "tmux", "iterm2"]


@dataclass(frozen=True)
class TeamMember:
    """One team-roster entry. Mirrors TS ``teamHelpers.ts:TeamMember``."""

    agent_id: str
    name: str
    color: str | None = None
    tmux_pane_id: str | None = None
    backend_type: BackendType = "in-process"


@dataclass(frozen=True)
class TeamFile:
    """Typed view of ``.clawcodex/team.json``."""

    team_name: str
    lead_agent_id: str
    description: str | None = None
    members: tuple[TeamMember, ...] = field(default_factory=tuple)


def get_team_file_path(workspace_root: Path) -> Path:
    """Stable path for a workspace's team file."""
    return workspace_root / ".clawcodex" / "team.json"


def read_team_file(workspace_root: Path) -> TeamFile | None:
    """Read and parse the team file. Returns ``None`` if absent."""
    path = get_team_file_path(workspace_root)
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    if not isinstance(raw, dict):
        return None
    members_raw = raw.get("members") or []
    members: list[TeamMember] = []
    if isinstance(members_raw, list):
        for entry in members_raw:
            if not isinstance(entry, dict):
                continue
            members.append(
                TeamMember(
                    agent_id=str(entry.get("agent_id", "")),
                    name=str(entry.get("name", "")),
                    color=entry.get("color"),
                    tmux_pane_id=entry.get("tmux_pane_id"),
                    backend_type=entry.get("backend_type", "in-process"),
                )
            )
    return TeamFile(
        team_name=str(raw.get("team_name", "")),
        lead_agent_id=str(raw.get("lead_agent_id", "")),
        description=raw.get("description"),
        members=tuple(members),
    )


def write_team_file(team: TeamFile, workspace_root: Path) -> None:
    """Serialize ``team`` to ``.clawcodex/team.json``."""
    path = get_team_file_path(workspace_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "team_name": team.team_name,
        "description": team.description,
        "lead_agent_id": team.lead_agent_id,
        "members": [
            {
                "agent_id": m.agent_id,
                "name": m.name,
                "color": m.color,
                "tmux_pane_id": m.tmux_pane_id,
                "backend_type": m.backend_type,
            }
            for m in team.members
        ],
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def add_member(team: TeamFile, member: TeamMember) -> TeamFile:
    """Return a new TeamFile with ``member`` appended."""
    filtered = tuple(m for m in team.members if m.agent_id != member.agent_id)
    return replace(team, members=filtered + (member,))


def remove_member(team: TeamFile, agent_id: str) -> TeamFile:
    return replace(team, members=tuple(m for m in team.members if m.agent_id != agent_id))


def find_member_by_name(team: TeamFile, name: str) -> TeamMember | None:
    """Case-insensitive lookup; returns the first match or None."""
    target = name.lower()
    for m in team.members:
        if m.name.lower() == target:
            return m
    return None


__all__ = [
    "BackendType",
    "TeamFile",
    "TeamMember",
    "add_member",
    "find_member_by_name",
    "get_team_file_path",
    "read_team_file",
    "remove_member",
    "write_team_file",
]
