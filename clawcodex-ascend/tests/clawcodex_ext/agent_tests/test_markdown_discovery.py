"""Unit tests for ``clawcodex_ext.agent.markdown_discovery``."""

from __future__ import annotations


# pylint: disable=E0611
from pathlib import Path

from clawcodex_ext.agent.markdown_discovery import (
    discover_clawcodex_ext_agents,
    discover_extension_agents,
)
from clawcodex_ext.agent.registry import SOURCE_CLAWCODEX_EXT, SOURCE_EXTENSIONS


def _write_agent_md(directory: Path, name: str, *, description: str, body: str) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    file_path = directory / f"{name}.md"
    file_path.write_text(
        f"---\ndescription: {description}\ntools:\n  - Read\n---\n\n{body}\n",
        encoding="utf-8",
    )
    return file_path


def test_discover_clawcodex_ext_agents_loads_markdown(tmp_path: Path):
    _write_agent_md(
        tmp_path,
        "demo",
        description="Demo decoupled agent.",
        body="You are a demo agent for Claw Codex.",
    )

    agents = discover_clawcodex_ext_agents(root=tmp_path)
    assert len(agents) == 1
    a = agents[0]
    assert a.agent_type == "demo"
    assert a.source == SOURCE_CLAWCODEX_EXT
    assert a.when_to_use == "Demo decoupled agent."
    assert a.get_system_prompt().strip() == "You are a demo agent for Claw Codex."


def test_discover_extension_agents_walks_subpackages(tmp_path: Path):
    # Two extension packages, each with an ``agents/`` subdir.
    for ext_name in ("alpha", "beta"):
        ext_agents_dir = tmp_path / ext_name / "agents"
        _write_agent_md(
            ext_agents_dir,
            f"{ext_name}-worker",
            description=f"Worker for {ext_name}.",
            body=f"You are the {ext_name} worker.",
        )
    # A package without an ``agents/`` subdir should be ignored.
    (tmp_path / "no-agents-pkg").mkdir()

    agents = discover_extension_agents(extensions_root=tmp_path)
    by_type = {a.agent_type: a for a in agents}
    assert set(by_type) == {"alpha-worker", "beta-worker"}
    for a in agents:
        assert a.source == SOURCE_EXTENSIONS
    assert by_type["alpha-worker"].get_system_prompt().strip() == "You are the alpha worker."


def test_discover_skips_missing_or_empty_roots(tmp_path: Path):
    # Non-existent paths return [] without raising.
    assert discover_clawcodex_ext_agents(root=tmp_path / "does-not-exist") == []
    assert discover_extension_agents(extensions_root=tmp_path / "does-not-exist") == []


def test_discover_skips_unreadable_files(tmp_path: Path, caplog):
    # File with broken frontmatter is silently dropped (with a debug log).
    bad = tmp_path / "broken.md"
    bad.write_text("not a real frontmatter block", encoding="utf-8")

    caplog.set_level("DEBUG", logger="clawcodex_ext.agent.markdown_discovery")
    agents = discover_clawcodex_ext_agents(root=tmp_path)
    # The contract is that no exception escapes and the file is dropped.
    assert all(a.agent_type != "broken" for a in agents)
