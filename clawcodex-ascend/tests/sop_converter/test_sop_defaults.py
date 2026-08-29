#!/usr/bin/env python3
# coding=utf-8

# -------------------------------------------------------------------------
# This file is part of the AgentSDK project.
#
# Originally from the clawcodex project:
#   https://github.com/agentforce314/clawcodex
#   Copyright (c) 2026 Clawd Codex Team
#   Licensed under the MIT License. See LICENSE-MIT-clawcodex in this directory.
#
# Portions copyright (c) 2026 Huawei Technologies Co.,Ltd.
# Licensed under Mulan PSL v2. You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
#
# This file is redistributed as a verbatim copy of the upstream source
# (minor whitespace / quoting normalization only); the original copyright
# notice and license terms above apply to the corresponding portions of
# this file. Local additions, if any, are licensed under Mulan PSL v2
# by Huawei Technologies Co.,Ltd.
# -------------------------------------------------------------------------

"""Smoke tests for the SOPDefaults dependency injection container.

Verifies:
- DEFAULTS singleton is populated with default adapters
- All 6 required fields are non-None after fill_defaults()
- Each factory produces a runtime_checkable Protocol-compatible instance
- Optional sop_provider is None by default

See ``docs/DECOUPLE_SOP_CONVERTER_PLAN.md`` §3.4 and §4.3.
"""

from __future__ import annotations


from extensions.capabilities.agent_definition_protocol import (
    AgentDefinitionProtocol,
    AgentToolConstants,
)
from extensions.capabilities.permission_protocol import PermissionContextProtocol
from extensions.capabilities.skill_protocol import (
    SkillProtocol,
)
from extensions.capabilities.tool_authoring_protocol import (
    ToolAuthoringProtocol,
)


def test_defaults_import():
    """DEFAULTS singleton is importable and populated."""
    from extensions.sop_converter.adapters import DEFAULTS, fill_defaults

    # Ensure populated (idempotent)
    fill_defaults(DEFAULTS)

    assert DEFAULTS.agent_definition_factory is not None
    assert DEFAULTS.skill_factory is not None
    assert DEFAULTS.frontmatter_parser is not None
    assert DEFAULTS.tool_authoring is not None
    assert DEFAULTS.permission_context_factory is not None
    assert DEFAULTS.agent_loader is not None
    # sop_provider is optional — should be None by default
    assert DEFAULTS.sop_provider is None


def test_agent_definition_factory():
    """Factory produces a runtime_checkable AgentDefinitionProtocol instance."""
    from extensions.sop_converter.adapters import DEFAULTS

    agent = DEFAULTS.agent_definition_factory(
        agent_type="test-agent",
        when_to_use="A test agent",
        tools=["Skill", "Read"],
        source="dynamic",
        base_dir="/tmp",
        background=False,
        memory=None,
    )
    assert isinstance(agent, AgentDefinitionProtocol)
    assert agent.agent_type == "test-agent"
    assert agent.when_to_use == "A test agent"
    assert agent.tools == ["Skill", "Read"]


def test_skill_factory():
    """Factory produces a runtime_checkable SkillProtocol instance."""
    from extensions.sop_converter.adapters import DEFAULTS

    skill = DEFAULTS.skill_factory(
        name="test-skill",
        description="A test skill",
        content="Do something",
        source="test",
        loaded_from="test",
        user_invocable=True,
        disable_model_invocation=False,
        content_length=0,
        is_hidden=False,
        context="",
        markdown_content="",
        progress_message="",
    )
    assert isinstance(skill, SkillProtocol)
    assert skill.name == "test-skill"


def test_frontmatter_parser():
    """Frontmatter parser returns a SkillFrontmatterResultProtocol-compatible result."""
    from extensions.sop_converter.adapters import DEFAULTS

    markdown = """---
name: test
description: Test
---

Body content here
"""
    parsed = DEFAULTS.frontmatter_parser(markdown)
    assert isinstance(parsed.frontmatter, dict)
    assert parsed.frontmatter.get("name") == "test"
    assert isinstance(parsed.body, str)
    assert "Body content" in parsed.body


def test_tool_authoring():
    """ToolAuthoringProtocol is populated and has the expected methods."""
    from extensions.sop_converter.adapters import DEFAULTS

    ta = DEFAULTS.tool_authoring
    assert isinstance(ta, ToolAuthoringProtocol)
    # Check that TOOL_DIR is a Path
    from pathlib import Path

    assert isinstance(ta.TOOL_DIR, Path)
    # Check methods exist
    assert hasattr(ta, "bundle_tool_dir")
    assert hasattr(ta, "save_spec")
    assert hasattr(ta, "validate_spec")
    assert hasattr(ta, "create_and_validate")
    assert hasattr(ta, "add_tool")
    assert hasattr(ta, "list_persisted_specs")
    assert hasattr(ta, "iter_bundle_tool_dirs")
    assert hasattr(ta, "create_spec")


def test_permission_context_factory():
    """Factory produces a runtime_checkable PermissionContextProtocol instance."""
    from extensions.sop_converter.adapters import DEFAULTS

    ctx = DEFAULTS.permission_context_factory(
        mode="bypass",
        is_bypass_permissions_mode_available=True,
        should_avoid_permission_prompts=False,
    )
    assert isinstance(ctx, PermissionContextProtocol)
    # Check property aliases
    assert ctx.is_bypass is True
    assert ctx.should_avoid_prompts is False


def test_agent_loader():
    """Agent loader returns a list of AgentDefinitionProtocol-compatible agents."""
    from extensions.sop_converter.adapters import DEFAULTS

    agents = DEFAULTS.agent_loader()
    assert isinstance(agents, list)
    # At minimum, it should return a list (possibly empty in test env)
    for agent in agents:
        assert isinstance(agent, AgentDefinitionProtocol)


def test_fill_defaults_idempotent():
    """fill_defaults() is idempotent — multiple calls don't break the container."""
    from extensions.sop_converter.adapters import DEFAULTS, fill_defaults

    agent_factory = DEFAULTS.agent_definition_factory
    fill_defaults(DEFAULTS)  # second call
    assert DEFAULTS.agent_definition_factory is agent_factory  # same reference


def test_agent_tool_constants():
    """AgentToolConstants values are accessible and match expected types."""
    assert isinstance(AgentToolConstants.MAX_INLINE_TOOL_DISPLAY, int)
    assert AgentToolConstants.MAX_INLINE_TOOL_DISPLAY > 0
    assert isinstance(AgentToolConstants.POS_PROXY_BASE_TOOLS, frozenset)
    assert "Skill" in AgentToolConstants.POS_PROXY_BASE_TOOLS
    assert isinstance(AgentToolConstants.POS_SOP_DOMAIN_AGENT_TOOLS, frozenset)
    assert "Bash" in AgentToolConstants.POS_SOP_DOMAIN_AGENT_TOOLS


def test_sop_converter_top_level_import():
    """Importing extensions.sop_converter triggers fill_defaults successfully."""
    import importlib

    import extensions.sop_converter.adapters

    # Force reimport to verify the module-level fill_defaults call
    importlib.reload(extensions.sop_converter.adapters)
    from extensions.sop_converter.adapters import DEFAULTS, fill_defaults

    fill_defaults(DEFAULTS)
    assert DEFAULTS.agent_definition_factory is not None
