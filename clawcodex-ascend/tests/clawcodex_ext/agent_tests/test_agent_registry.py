#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# -------------------------------------------------------------------------
# This file is part of the AgentSDK project.
#
# Originally from Clawd Codex:
# https://github.com/agentforce314/clawcodex
# Copyright (c) 2026 Clawd Codex Team
# Licensed under the MIT License. See clawcodex-ascend/LICENSE.clawcodex.
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

"""Unit tests for ``clawcodex_ext.agent.registry.AgentRegistry``."""

from __future__ import annotations


# pylint: disable=E0611
from clawcodex_ext.agent.agent_definitions import AgentDefinition
from clawcodex_ext.agent.constants import ALL_AGENT_DISALLOWED_TOOLS

from clawcodex_ext.agent.registry import (
    SOURCE_CLAWCODEX_EXT,
    SOURCE_EXTENSIONS,
    AgentRegistry,
)


def test_register_decorator_attaches_definition():
    @AgentRegistry.register(
        "demo",
        when_to_use="Demo agent for unit tests.",
        tools=["Read"],
    )
    def _prompt() -> str:
        return "You are a demo."

    stored = AgentRegistry.find("demo")
    assert stored is not None
    assert stored.agent_type == "demo"
    assert stored.when_to_use == "Demo agent for unit tests."
    assert stored.tools == ["Read"]
    assert stored.source == SOURCE_CLAWCODEX_EXT
    # The hard-coded deny list is always layered in.
    for t in ALL_AGENT_DISALLOWED_TOOLS:
        assert t in (stored.disallowed_tools or [])
    # System prompt is produced by the decorated function.
    assert stored.get_system_prompt() == "You are a demo."


def test_top_level_register_alias_attaches_clawcodex_ext_definition():
    from clawcodex_ext.agent import register

    @register("top-level", when_to_use="Top-level decorator.", tools=["Read"])
    def _prompt() -> str:
        return "top-level prompt"

    stored = AgentRegistry.find("top-level")
    assert stored is not None
    assert stored.source == SOURCE_CLAWCODEX_EXT
    assert stored.get_system_prompt() == "top-level prompt"


def test_extensions_register_defaults_to_extensions_source():
    from extensions.agents import register

    @register("third-party-default", when_to_use="Third-party decorator.", tools=["Read"])
    def _prompt() -> str:
        return "third-party prompt"

    stored = AgentRegistry.find("third-party-default")
    assert stored is not None
    assert stored.source == SOURCE_EXTENSIONS
    assert stored.base_dir == "extensions"
    assert stored.get_system_prompt() == "third-party prompt"


def test_register_definition_explicit_path():
    agent = AgentDefinition(
        agent_type="explicit",
        when_to_use="Explicitly registered.",
        tools=None,
        source=SOURCE_EXTENSIONS,
        base_dir="extensions/sample",
    )
    returned = AgentRegistry.register_definition(agent)

    assert returned is agent
    assert AgentRegistry.find("explicit") is agent
    assert AgentRegistry.by_source(SOURCE_EXTENSIONS) == [agent]


def test_register_is_last_wins_by_agent_type(caplog):
    @AgentRegistry.register(
        "dup",
        when_to_use="first",
        tools=["Read"],
    )
    def _first_prompt() -> str:
        return "first prompt"

    first = AgentRegistry.find("dup")
    assert first is not None

    with caplog.at_level("INFO", logger="clawcodex_ext.agent.registry"):

        @AgentRegistry.register(
            "dup",
            when_to_use="second",
            tools=["Read"],
        )
        def _second_prompt() -> str:
            return "second prompt"

    # Last-wins: only one entry, and it is the second one.
    matches = [a for a in AgentRegistry.all() if a.agent_type == "dup"]
    assert len(matches) == 1
    assert matches[0].when_to_use == "second"
    assert matches[0].get_system_prompt() == "second prompt"

    # Override was logged at INFO level.
    assert any("overridden" in rec.message for rec in caplog.records)


def test_by_source_filters_correctly():
    @AgentRegistry.register(
        "internal",
        when_to_use="Internal decoupled agent.",
        tools=["Read"],
    )
    def _internal() -> str:
        return "internal"

    ext = AgentDefinition(
        agent_type="third-party",
        when_to_use="Third-party agent.",
        tools=["Read"],
        source=SOURCE_EXTENSIONS,
        base_dir="extensions/sample",
    )
    AgentRegistry.register_definition(ext)

    clawcodex_ext_types = {a.agent_type for a in AgentRegistry.by_source(SOURCE_CLAWCODEX_EXT)}
    # The bundled agents are also clawcodex_ext-sourced, so use subset semantics.
    assert "internal" in clawcodex_ext_types
    assert {a.agent_type for a in AgentRegistry.by_source(SOURCE_EXTENSIONS)} == {"third-party"}


def test_clear_drops_all_definitions():
    @AgentRegistry.register("a", when_to_use="a", tools=["Read"])
    def _a() -> str:
        return "a"

    @AgentRegistry.register("b", when_to_use="b", tools=["Read"])
    def _b() -> str:
        return "b"

    assert len(AgentRegistry.all()) >= 2

    AgentRegistry.clear()
    assert AgentRegistry.all() == []
    assert AgentRegistry.find("a") is None
    assert AgentRegistry.find("b") is None


def test_bundled_agents_are_registered_on_import():
    """Eager import of ``clawcodex_ext.agent`` registers the three bundled agents.

    Force a re-import of the bundled-agent submodules so the @register
    decorators re-execute against the registry state captured by the
    conftest fixture. The outer module ``clawcodex_ext.agent`` is
    already in ``sys.modules`` (imported by the conftest), so the
    explicit ``del sys.modules[..._bundled_agents...]`` purge is what
    actually re-triggers the side effects.
    """
    import importlib
    import sys

    AgentRegistry.clear()
    try:
        # Drop the bundled-agent submodules so the @register side
        # effects run again on import.
        for mod in list(sys.modules):
            if mod.startswith("clawcodex_ext.agent._bundled_agents"):
                del sys.modules[mod]
        importlib.import_module("clawcodex_ext.agent._bundled_agents")
        types = {a.agent_type for a in AgentRegistry.all()}
    finally:
        # Re-seed the conftest-saved state (the autouse fixture will
        # restore it on teardown, but the clear() above clobbered it
        # — restore manually so other tests in the session still see
        # the bundled agents).
        for mod in list(sys.modules):
            if mod.startswith("clawcodex_ext.agent._bundled_agents"):
                del sys.modules[mod]
        importlib.import_module("clawcodex_ext.agent._bundled_agents")

    assert {"code-reviewer", "docs-writer", "test-runner"}.issubset(types)
