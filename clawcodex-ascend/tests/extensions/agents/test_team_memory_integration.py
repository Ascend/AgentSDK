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

"""Integration tests for team_memory_integration.

Covers the TeamCreate/TeamDelete/SendMessage hooks and the prompt
section builder. Uses a tmp workspace with a real ``.clawcodex/team.json``
and an explicit ``CLAWCODEX_TEAM_MEMORY=1`` + auto-memory env so the
gating predicates return True.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from clawcodex_ext.services.swarm.team_file import TeamFile, TeamMember, write_team_file

from extensions.agents.team_memory_integration import (
    archive_team_memory,
    build_team_memory_prompt_section,
    initialize_team_memory,
    is_team_memory_active,
    sink_send_message_summary,
)


@pytest.fixture(autouse=True)
def _team_mem_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """Enable team memory + auto-memory and point the auto-mem dir at tmp."""
    monkeypatch.setenv("CLAUDE_CODE_TEAM_MEMORY", "1")
    monkeypatch.setenv("CLAUDE_CODE_DISABLE_AUTO_MEMORY", "0")
    # Override the memory base dir so writes land in tmp, not ~/.claude.
    monkeypatch.setenv("CLAUDE_CODE_REMOTE_MEMORY_DIR", str(tmp_path / "membase"))
    # The auto-mem path derives from the git root; force cwd to tmp so
    # the project-root sanitizer produces a stable path.
    monkeypatch.chdir(tmp_path)
    return tmp_path


def _make_team(workspace: Path, *, lead: str = "lead-1", members: tuple[str, ...] = ("lead-1",)) -> None:
    team = TeamFile(
        team_name="t-int",
        lead_agent_id=lead,
        members=tuple(TeamMember(agent_id=a, name=a) for a in members),
    )
    write_team_file(team, workspace)


def test_is_team_memory_active_respects_env(_team_mem_env: Path) -> None:
    assert is_team_memory_active() is True


def test_initialize_creates_memory_md(_team_mem_env: Path) -> None:
    _make_team(_team_mem_env)
    assert initialize_team_memory(_team_mem_env) is True
    # Find the team MEMORY.md under the auto-mem dir.
    from clawcodex_ext.memdir.team_mem_paths import get_team_mem_entrypoint

    entry = Path(get_team_mem_entrypoint())
    assert entry.exists()
    assert "Team Memory" in entry.read_text(encoding="utf-8")


def test_initialize_noop_without_team_file(_team_mem_env: Path) -> None:
    # No team.json → returns False, no dir created.
    assert initialize_team_memory(_team_mem_env) is False


def test_archive_on_team_delete(_team_mem_env: Path) -> None:
    _make_team(_team_mem_env)
    initialize_team_memory(_team_mem_env)
    # Write an entry via the service so there's something to archive.
    from extensions.agents.team_memory import TeamMemoryConfig, TeamMemoryService

    svc = TeamMemoryService(workspace_root=_team_mem_env, config=TeamMemoryConfig(enabled=True))
    svc.remember("team fact", author_agent_id="lead-1", tags=("fact",))
    dst = archive_team_memory(_team_mem_env, reason="TeamDelete")
    assert dst is not None and dst.exists()
    assert dst.read_text(encoding="utf-8").strip()  # non-empty snapshot


def test_sink_send_message_summary_writes_entry(_team_mem_env: Path) -> None:
    _make_team(_team_mem_env)
    initialize_team_memory(_team_mem_env)
    entry = sink_send_message_summary(
        _team_mem_env,
        sender="lead-1",
        sender_agent_id="lead-1",
        recipients=["agent-2"],
        summary="handoff",
        message="please review the deploy checklist",
    )
    assert entry is not None
    assert "send_message" in entry.tags


def test_sink_summary_noop_when_disabled(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("CLAUDE_CODE_TEAM_MEMORY", "0")
    monkeypatch.setenv("CLAUDE_CODE_DISABLE_AUTO_MEMORY", "0")
    _make_team(tmp_path)
    assert (
        sink_send_message_summary(
            tmp_path,
            sender="x",
            sender_agent_id="x",
            recipients=["y"],
            summary="s",
            message="m",
        )
        is None
    )


def test_prompt_section_empty_when_no_recall(_team_mem_env: Path) -> None:
    _make_team(_team_mem_env)
    initialize_team_memory(_team_mem_env)
    section = build_team_memory_prompt_section(_team_mem_env, requester_agent_id="lead-1", task="nothing matches")
    # No entries → empty section.
    assert section == ""


def test_prompt_section_contains_entries_and_stale_caveat(_team_mem_env: Path) -> None:
    _make_team(_team_mem_env)
    initialize_team_memory(_team_mem_env)
    from extensions.agents.team_memory import TeamMemoryConfig, TeamMemoryService

    svc = TeamMemoryService(workspace_root=_team_mem_env, config=TeamMemoryConfig(enabled=True))
    svc.remember(
        "Run stability gate before commit",
        author_agent_id="lead-1",
        tags=("build",),
        summary="stability gate before commit",
    )
    section = build_team_memory_prompt_section(
        _team_mem_env, requester_agent_id="lead-1", task="commit build stability"
    )
    assert "<team_memory>" in section
    assert "</team_memory>" in section
    assert "stability gate" in section
    assert "trust the files" in section  # stale caveat


def test_prompt_section_disabled_returns_empty(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("CLAUDE_CODE_TEAM_MEMORY", "0")
    monkeypatch.setenv("CLAUDE_CODE_DISABLE_AUTO_MEMORY", "0")
    _make_team(tmp_path)
    assert build_team_memory_prompt_section(tmp_path, requester_agent_id="lead-1", task="x") == ""


def test_team_member_can_recall_other_member_entry(_team_mem_env: Path) -> None:
    """Acceptance #3: member A writes, member B recalls."""
    _make_team(_team_mem_env, members=("lead-1", "agent-2"))
    initialize_team_memory(_team_mem_env)
    from extensions.agents.team_memory import (
        TeamMemoryConfig,
        TeamMemoryQuery,
        TeamMemoryService,
    )

    svc = TeamMemoryService(workspace_root=_team_mem_env, config=TeamMemoryConfig(enabled=True))
    svc.remember("shared fact", author_agent_id="lead-1", tags=("fact",))
    results = svc.recall(
        TeamMemoryQuery(
            team_id=svc.team_id,
            query="shared fact",
            requester_agent_id="agent-2",
            top_k=5,
        )
    )
    assert len(results) == 1
    assert "shared fact" in results[0].entry.content
