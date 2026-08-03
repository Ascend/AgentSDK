"""Integration tests for the merge order in
``src.agent.load_agents_dir.get_agent_definitions_with_overrides``.

These pin the contract that the decoupled ``clawcodex_ext`` and
``extensions`` tiers participate in last-wins merge, with the
priority sequence

    built-in < plugin < clawcodex_ext < extensions < user < project < managed

and that custom agents from these tiers can override built-in
``agent_type`` slots.
"""

from __future__ import annotations


# pylint: disable=E0611,W0621
from pathlib import Path

import pytest

from clawcodex_ext.agent.registry import (
    SOURCE_CLAWCODEX_EXT,
    SOURCE_EXTENSIONS,
    AgentRegistry,
)


@pytest.fixture
def isolated_cwd(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Run the test in an empty working directory so on-disk discovery
    from user/project/managed layers stays empty.
    """
    monkeypatch.chdir(tmp_path)
    # Also clear the agent-definitions cache that load_agents_dir keeps
    # per-cwd — otherwise a previous test in the same session could
    # leak its results in.
    from src.agent import load_agents_dir

    load_agents_dir.clear_agent_definitions_cache()
    yield tmp_path


def _reload_extension_agents(tmp_path: Path):
    """Re-point the markdown discovery defaults at ``tmp_path`` so the
    test can drop ``agents/<name>.md`` files into the layout it
    controls. We patch the module-level constants rather than the
    call site to keep the production code path unchanged.
    """
    from clawcodex_ext.agent import markdown_discovery

    markdown_discovery.DEFAULT_CLAWCODEX_EXT_AGENTS_DIR = tmp_path / "clawcodex"
    markdown_discovery.DEFAULT_EXTENSIONS_ROOT = tmp_path / "extensions"
    (tmp_path / "clawcodex" / "agents").mkdir(parents=True, exist_ok=True)
    (tmp_path / "extensions").mkdir(parents=True, exist_ok=True)


def _write_md(directory: Path, name: str, *, description: str, body: str) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    file_path = directory / f"{name}.md"
    file_path.write_text(
        f"---\ndescription: {description}\ntools:\n  - Read\n---\n\n{body}\n",
        encoding="utf-8",
    )
    return file_path


def test_extensions_override_clawcodex_ext(tmp_path: Path, isolated_cwd):
    """Two tiers register the same ``agent_type``; the ``extensions``
    entry wins because it sits higher in the merge order.
    """
    _reload_extension_agents(tmp_path)

    AgentRegistry.clear()
    AgentRegistry.register_definition(_make_def("dup", "from clawcodex_ext", SOURCE_CLAWCODEX_EXT))
    _write_md(
        tmp_path / "extensions" / "demo" / "agents",
        "dup",
        description="from extensions md",
        body="extensions body",
    )

    from src.agent.load_agents_dir import get_agent_definitions_with_overrides

    active = get_agent_definitions_with_overrides(str(isolated_cwd))
    match = next((a for a in active if a.agent_type == "dup"), None)
    assert match is not None
    assert match.source == SOURCE_EXTENSIONS
    assert match.get_system_prompt().strip() == "extensions body"


def test_clawcodex_ext_can_override_builtin(tmp_path: Path, isolated_cwd):
    """A ``clawcodex_ext`` agent of type ``general-purpose`` wins over
    the built-in ``general-purpose`` agent from
    ``src.agent.agent_definitions``.
    """
    _reload_extension_agents(tmp_path)
    AgentRegistry.clear()
    AgentRegistry.register_definition(_make_def("general-purpose", "from clawcodex_ext", SOURCE_CLAWCODEX_EXT))

    from src.agent.load_agents_dir import get_agent_definitions_with_overrides

    active = get_agent_definitions_with_overrides(str(isolated_cwd))
    matches = [a for a in active if a.agent_type == "general-purpose"]
    assert len(matches) == 1
    assert matches[0].source == SOURCE_CLAWCODEX_EXT
    assert matches[0].when_to_use == "from clawcodex_ext"


def test_user_overrides_extension(tmp_path: Path, isolated_cwd, monkeypatch):
    """A user-level on-disk agent of type ``dup`` beats both
    ``extensions`` and ``clawcodex_ext`` entries of the same type.

    The user source is rooted at ``$HOME/.claude/agents/``; we
    redirect ``HOME`` to ``tmp_path`` so the test can place the
    file at ``tmp_path/.claude/agents/dup.md`` and have it resolve
    as ``source="user"``.
    """
    _reload_extension_agents(tmp_path)
    AgentRegistry.clear()
    AgentRegistry.register_definition(_make_def("dup", "from clawcodex_ext", SOURCE_CLAWCODEX_EXT))
    _write_md(
        tmp_path / "extensions" / "demo" / "agents",
        "dup",
        description="from extensions md",
        body="extensions body",
    )

    # user-level override via the on-disk custom-agent path.
    monkeypatch.setenv("HOME", str(tmp_path))
    user_agents_dir = tmp_path / ".claude" / "agents"
    _write_md(
        user_agents_dir,
        "dup",
        description="from user",
        body="user body",
    )

    from src.agent.load_agents_dir import get_agent_definitions_with_overrides

    active = get_agent_definitions_with_overrides(str(isolated_cwd))
    match = next(a for a in active if a.agent_type == "dup")
    assert match.source == "user"
    assert match.get_system_prompt().strip() == "user body"


def test_bundled_agents_appear_in_active_list(tmp_path: Path, isolated_cwd):
    """The bundled agents registered at conftest-import time show up
    in the active list under ``source="clawcodex_ext"``.
    """
    from src.agent.load_agents_dir import get_agent_definitions_with_overrides

    active = get_agent_definitions_with_overrides(str(isolated_cwd))
    types = {a.agent_type for a in active}
    assert {"code-reviewer", "docs-writer", "test-runner"}.issubset(types)
    for t in ("code-reviewer", "docs-writer", "test-runner"):
        m = next(a for a in active if a.agent_type == t)
        assert m.source == SOURCE_CLAWCODEX_EXT


def _make_def(agent_type: str, when_to_use: str, source: str):
    from src.agent.agent_definitions import AgentDefinition

    return AgentDefinition(
        agent_type=agent_type,
        when_to_use=when_to_use,
        tools=["Read"],
        source=source,  # type: ignore[arg-type]
        base_dir="test",
    )
