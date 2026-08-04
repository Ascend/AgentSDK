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

"""Bundled adapter for the portable ``spec-audit`` skill package."""

from __future__ import annotations

from functools import lru_cache

from ..argument_substitution import substitute_arguments
from ..bundled_skills import BundledSkillDefinition, register_bundled_skill
from ..frontmatter import parse_frontmatter
from .resource_loader import load_bundled_text_resources


_RESOURCE_PACKAGE = "clawcodex_ext.skills.bundled.spec_audit_resources"
_SKILL_NAME = "spec-audit"


@lru_cache(maxsize=1)
def _load_portable_skill() -> tuple[str, str, tuple[tuple[str, str], ...]]:
    resources = load_bundled_text_resources(_RESOURCE_PACKAGE)
    try:
        skill_markdown = resources["SKILL.md"]
    except KeyError as exc:
        raise FileNotFoundError(f"{_RESOURCE_PACKAGE} does not contain SKILL.md") from exc

    parsed = parse_frontmatter(skill_markdown)
    if parsed.frontmatter.get("name") != _SKILL_NAME:
        raise ValueError(f"bundled {_SKILL_NAME} package has invalid name frontmatter")
    description = parsed.frontmatter.get("description")
    if not isinstance(description, str) or not description.strip():
        raise ValueError(f"bundled {_SKILL_NAME} package has no description frontmatter")
    if not parsed.body.strip():
        raise ValueError(f"bundled {_SKILL_NAME} package has an empty prompt body")

    return description.strip(), parsed.body, tuple(resources.items())


def _build_spec_audit_prompt(args: str) -> str:
    _, prompt, _ = _load_portable_skill()
    return substitute_arguments(prompt, args, append_if_no_placeholder=True)


def register_spec_audit_skill() -> bool:
    """Register the portable Spec-Audit package in the bundled catalogue.

    Returns:
        ``True`` when the bundled registry accepts the definition.

    Raises:
        FileNotFoundError: If the packaged skill or its ``SKILL.md`` is missing.
        ValueError: If the packaged ``SKILL.md`` metadata or body is invalid.
    """

    description, _, resource_items = _load_portable_skill()
    return register_bundled_skill(
        BundledSkillDefinition(
            name=_SKILL_NAME,
            description=description,
            user_invocable=True,
            files=dict(resource_items),
            requires_resources=True,
            get_prompt_for_command=_build_spec_audit_prompt,
        )
    )


__all__ = ["register_spec_audit_skill"]
