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
#

"""Skill + Frontmatter Protocols — interface for SOP-convertible skills.

* :class:`SkillProtocol` — mirrors ``clawcodex_ext.skills.model.Skill``
  (field subset per ``docs/DECOUPLE_SOP_CONVERTER_PLAN.md`` §3.3).
* :class:`SkillFrontmatterProtocol` — the ``parse_frontmatter`` boundary
  for ``bundle_agents.py`` / ``bundle_skills.py``, keeping the
  ``(frontmatter, body)`` shape of ``FrontmatterParseResult``.
"""

from __future__ import annotations

from typing import Any, Callable, Optional, Protocol, runtime_checkable

__all__ = [
    "SkillProtocol",
    "SkillFrontmatterProtocol",
    "SkillFrontmatterResultProtocol",
]


@runtime_checkable
class SkillProtocol(Protocol):
    """Protocol for a parsed skill that the SOP converter can read.

    Field names align 1:1 with ``clawcodex_ext.skills.model.Skill`` so
    runtime ``isinstance`` checks pass without an adapter; optional
    fields stay optional.
    """

    name: str
    description: str
    content: str
    source: str
    loaded_from: str
    user_invocable: bool
    disable_model_invocation: bool
    content_length: int
    is_hidden: bool
    skill_root: Optional[str]
    aliases: list[str]
    allowed_tools: list[str]
    argument_hint: Optional[str]
    argument_names: list[str]
    when_to_use: Optional[str]
    version: Optional[str]
    model: Optional[str]
    context: str
    agent: Optional[str]
    effort: Optional[str | int]
    paths: Optional[list[str]]
    display_name: Optional[str]
    has_user_specified_description: bool
    base_dir: Optional[str]
    markdown_content: str
    progress_message: str
    hooks: Optional[dict]
    shell: Optional[str]
    get_prompt_for_command: Optional[Callable[[str], str]]
    is_enabled_fn: Optional[Callable[[], bool]]

    def user_facing_name(self) -> str: ...

    def get_prompt(self, args: str = "") -> str: ...

    def is_enabled(self) -> bool: ...


@runtime_checkable
class SkillFrontmatterResultProtocol(Protocol):
    """Result of parsing a markdown frontmatter block.

    Duck-typed mirror of ``clawcodex_ext.skills.frontmatter.FrontmatterParseResult``.
    """

    frontmatter: dict[str, Any]
    body: str


@runtime_checkable
class SkillFrontmatterProtocol(Protocol):
    """Frontmatter parser boundary used by ``bundle_agents`` / ``bundle_skills``.

    Implementations MUST return a value exposing ``.frontmatter`` /
    ``.body``; the default wraps
    ``clawcodex_ext.skills.frontmatter.parse_frontmatter``.
    """

    def __call__(self, markdown: str) -> SkillFrontmatterResultProtocol: ...
