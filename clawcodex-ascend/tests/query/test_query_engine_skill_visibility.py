from __future__ import annotations

import asyncio
import logging
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, create_autospec

import pytest

import clawcodex_ext.query.engine as engine_module
from clawcodex_ext.query.engine import QueryEngine, QueryEngineConfig
from clawcodex_ext.tool_system.context import ToolContext
from clawcodex_ext.tool_system.registry import ToolRegistry


async def _build_prompt(
    monkeypatch,
    tmp_path,
    *,
    registered: bool = True,
    model_visible: bool = True,
    enabled: bool = True,
    denied: bool = False,
    denied_skill: bool = False,
):
    skill_tool = MagicMock()
    skill_tool.name = "Skill"
    skill_tool.aliases = ()
    skill_tool.is_enabled.return_value = enabled

    registry = ToolRegistry([skill_tool] if registered else [])
    context = ToolContext(workspace_root=tmp_path)
    if denied:
        context.permission_context.always_deny_rules = {"session": ["Skill"]}
    elif denied_skill:
        context.permission_context.always_deny_rules = {"session": ["Skill(visible-skill)"]}

    provider = MagicMock()
    provider.model = "test-model"
    config = QueryEngineConfig(
        cwd=tmp_path,
        provider=provider,
        tool_registry=registry,
        tools=[skill_tool] if registered and model_visible else [],
        tool_context=context,
    )

    from clawcodex_ext.command_system.skills_integration import (
        get_skill_tool_commands,
    )

    command_module = ModuleType("clawcodex_ext.command_system")
    command_loader = create_autospec(
        get_skill_tool_commands,
        return_value=["visible-skill"],
    )
    command_module.get_skill_tool_commands = command_loader
    monkeypatch.setitem(sys.modules, "clawcodex_ext.command_system", command_module)

    monkeypatch.setattr(
        engine_module,
        "fetch_system_prompt_parts",
        AsyncMock(
            return_value=SimpleNamespace(
                user_context={},
                system_context={},
            )
        ),
    )
    prompt_builder = MagicMock(return_value=[{"type": "text", "text": "base"}])
    monkeypatch.setattr(engine_module, "build_full_system_prompt_blocks", prompt_builder)

    await QueryEngine(config)._build_system_prompt_parts()
    return command_loader, prompt_builder


@pytest.mark.asyncio
async def test_query_includes_skill_listing_when_skill_tool_is_available(
    monkeypatch,
    tmp_path,
):
    command_loader, prompt_builder = await _build_prompt(monkeypatch, tmp_path)

    command_loader.assert_called_once_with(str(tmp_path), None)
    assert prompt_builder.call_args.kwargs["skills"] == ["visible-skill"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("registered", "model_visible", "enabled", "denied"),
    [
        (False, False, True, False),
        (True, False, True, False),
        (True, True, False, False),
        (True, True, True, True),
    ],
)
async def test_query_omits_skill_listing_when_skill_tool_is_unavailable(
    monkeypatch,
    tmp_path,
    registered,
    model_visible,
    enabled,
    denied,
):
    command_loader, prompt_builder = await _build_prompt(
        monkeypatch,
        tmp_path,
        registered=registered,
        model_visible=model_visible,
        enabled=enabled,
        denied=denied,
    )

    command_loader.assert_not_called()
    assert prompt_builder.call_args.kwargs["skills"] == []


@pytest.mark.asyncio
async def test_query_filters_canonical_skill_denied_by_permission(
    monkeypatch,
    tmp_path,
):
    command_loader, prompt_builder = await _build_prompt(
        monkeypatch,
        tmp_path,
        denied_skill=True,
    )

    command_loader.assert_called_once_with(str(tmp_path), None)
    assert prompt_builder.call_args.kwargs["skills"] == []


def test_real_get_skill_tool_commands_uses_session_catalog(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from clawcodex_ext.command_system import get_skill_tool_commands, skills_integration
    from clawcodex_ext.skills.catalog import get_skill_catalog, invalidate_skill_catalog

    workspace = tmp_path / "workspace"
    skill_dir = workspace / ".claude" / "skills" / "real-session-skill"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        "---\ndescription: Real session-scoped skill\n---\nUse the real catalog.",
        encoding="utf-8",
    )

    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))
    for name in (
        "CLAUDE_CONFIG_DIR",
        "CLAWCODEX_SKILLS_DIR",
        "CLAUDE_SKILLS_DIR",
        "CLAWCODEX_MANAGED_SKILLS_DIR",
        "CLAUDE_CODE_BARE_MODE",
        "CLAUDE_CODE_ADDITIONAL_DIRECTORIES",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("CLAUDE_MANAGED_CONFIG_DIR", str(tmp_path / "managed"))
    monkeypatch.setattr(
        "extensions.sop_converter.bundle_context.get_active_bundle",
        lambda: None,
    )

    catalog_calls: list[dict[str, object]] = []
    real_get_skill_catalog = skills_integration.get_skill_catalog

    def recording_get_skill_catalog(*args, **kwargs):
        catalog_calls.append(dict(kwargs))
        return real_get_skill_catalog(*args, **kwargs)

    monkeypatch.setattr(
        skills_integration,
        "get_skill_catalog",
        recording_get_skill_catalog,
    )

    invalidate_skill_catalog("test setup")
    try:
        session_a_commands = get_skill_tool_commands(
            str(workspace),
            "session-a",
        )
        session_b_commands = get_skill_tool_commands(
            str(workspace),
            "session-b",
        )
        session_a_snapshot = get_skill_catalog(
            project_root=workspace,
            session_id="session-a",
        )
        session_b_snapshot = get_skill_catalog(
            project_root=workspace,
            session_id="session-b",
        )
    finally:
        invalidate_skill_catalog("test cleanup")

    assert catalog_calls == [
        {"project_root": str(workspace), "session_id": "session-a"},
        {"project_root": str(workspace), "session_id": "session-b"},
    ]
    assert session_a_snapshot is not session_b_snapshot
    assert session_a_snapshot.session_id == "session-a"
    assert session_b_snapshot.session_id == "session-b"
    assert any(command.name == "real-session-skill" for command in session_a_commands)
    assert any(command.name == "real-session-skill" for command in session_b_commands)


def test_prompt_assembly_fallback_keeps_append_prompt_and_logs_exception(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    skill_tool = MagicMock()
    skill_tool.name = "Skill"
    skill_tool.aliases = ()
    skill_tool.is_enabled.return_value = True

    context = ToolContext(workspace_root=tmp_path)
    config = QueryEngineConfig(
        cwd=tmp_path,
        provider=MagicMock(model="test-model"),
        tool_registry=ToolRegistry([skill_tool]),
        tools=[skill_tool],
        tool_context=context,
        append_system_prompt="SOP OVERVIEW MARKER",
    )
    monkeypatch.setattr(
        engine_module,
        "fetch_system_prompt_parts",
        AsyncMock(return_value=SimpleNamespace(user_context={}, system_context={})),
    )
    assembly_error = RuntimeError("skill catalog exploded")
    command_module = ModuleType("clawcodex_ext.command_system")
    command_module.get_skill_tool_commands = MagicMock(side_effect=assembly_error)
    monkeypatch.setitem(sys.modules, "clawcodex_ext.command_system", command_module)
    monkeypatch.setattr(
        engine_module,
        "build_context_prompt",
        MagicMock(return_value="legacy"),
    )

    with caplog.at_level(logging.WARNING, logger=engine_module.__name__):
        prompt, user_context, system_context = asyncio.run(QueryEngine(config)._build_system_prompt_parts())

    assert prompt == "legacy\n\nSOP OVERVIEW MARKER"
    assert user_context == {}
    assert system_context == {}
    record = next(record for record in caplog.records if "system prompt assembly failed" in record.getMessage())
    assert record.exc_info is not None
    assert record.exc_info[1] is assembly_error
