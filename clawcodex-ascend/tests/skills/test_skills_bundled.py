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

from __future__ import annotations

import pytest
from src.skills.bundled_skills import (
    BundledSkillDefinition,
    clear_bundled_skills,
    get_bundled_skills,
    register_bundled_skill,
)


@pytest.fixture(autouse=True)
def _clean_registry() -> None:
    clear_bundled_skills()
    yield  # type: ignore[misc]
    clear_bundled_skills()


class TestBundledSkills:
    def test_register_and_get(self) -> None:
        register_bundled_skill(
            BundledSkillDefinition(
                name="test-skill",
                description="A test skill",
                get_prompt_for_command=lambda args: f"prompt: {args}",
            )
        )
        skills = get_bundled_skills()
        assert len(skills) == 1
        assert skills[0].name == "test-skill"
        assert skills[0].description == "A test skill"
        assert skills[0].source == "bundled"
        assert skills[0].loaded_from == "bundled"

    def test_multiple_skills(self) -> None:
        for i in range(3):
            register_bundled_skill(
                BundledSkillDefinition(
                    name=f"skill-{i}",
                    description=f"Skill {i}",
                    get_prompt_for_command=lambda args: args,
                )
            )
        assert len(get_bundled_skills()) == 3

    def test_clear(self) -> None:
        # Test the in-process behavior of `clear_bundled_skills`:
        # registering a one-off, then clearing, must wipe that one-off
        # from the registry. We don't assert "len == 0" after clear
        # because clearing also re-arms the lazy-init flag, so the next
        # `get_bundled_skills()` call seeds the always-on bundled
        # catalogue (simplify, debug, loop, stuck, verify-content).
        # Instead we assert the temporary skill is gone.
        from src.skills.bundled_skills import _bundled_skills

        register_bundled_skill(
            BundledSkillDefinition(
                name="temp",
                description="temp",
                get_prompt_for_command=lambda a: a,
            )
        )
        assert len(get_bundled_skills()) == 1  # lazy-init suppressed by register
        clear_bundled_skills()
        # After clear, the in-memory list is empty until the next
        # consumer triggers lazy-init.
        assert _bundled_skills == []
        seeded = get_bundled_skills()
        names = {s.name for s in seeded}
        assert "temp" not in names  # one-off was wiped
        assert "simplify" in names  # bundled catalogue re-seeded

    def test_get_returns_copy(self) -> None:
        register_bundled_skill(
            BundledSkillDefinition(
                name="s1",
                description="d",
                get_prompt_for_command=lambda a: a,
            )
        )
        skills1 = get_bundled_skills()
        skills2 = get_bundled_skills()
        assert skills1 is not skills2

    def test_skill_properties(self) -> None:
        register_bundled_skill(
            BundledSkillDefinition(
                name="advanced",
                description="Advanced skill",
                get_prompt_for_command=lambda a: "prompt",
                aliases=["adv"],
                when_to_use="when testing",
                argument_hint="<arg>",
                allowed_tools=["Bash"],
                disable_model_invocation=True,
                user_invocable=False,
            )
        )
        skill = get_bundled_skills()[0]
        assert skill.when_to_use == "when testing"
        assert skill.argument_hint == "<arg>"
        assert "Bash" in skill.allowed_tools
        assert skill.disable_model_invocation is True
        assert skill.user_invocable is False
        assert skill.is_hidden is True

    def test_get_prompt(self) -> None:
        register_bundled_skill(
            BundledSkillDefinition(
                name="prompt-test",
                description="test",
                get_prompt_for_command=lambda args: f"Hello {args}",
            )
        )
        skill = get_bundled_skills()[0]
        assert skill.get_prompt("world") == "Hello world"
