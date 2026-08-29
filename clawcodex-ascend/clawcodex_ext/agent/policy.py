#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# -------------------------------------------------------------------------
# This file is part of the AgentSDK project.
#
# Originally from Clawd Codex:
# https://github.com/agentforce314/clawcodex
# Copyright (c) 2026 Clawd Codex Team
# Licensed under the MIT License. See clawcodex-ascend/LICENSES/Clawd-Codex-MIT.txt.
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

"""Shared identity and action-norm primitives for custom agents.

This module is the single place to change the wording of "what kind of
agent you are" and "what you must/must not do" for the agents shipped
under ``clawcodex_ext``. Extension authors compose new agents by
picking an ``IDENTITY_*`` template, one or more ``NORM_*`` action
norms, and any number of ``TOOL_SET_*`` presets, then assemble the
final system prompt with :func:`build_agent_prompt`.

The intent is to mirror how ``_SHARED_PREFIX`` and
``_SHARED_GUIDELINES`` are used inside
``src/agent/agent_definitions.py`` for the built-in agents, but
expose them as named, importable constants so downstream authors do
not have to copy-paste private strings.
"""

from __future__ import annotations


# ---------------------------------------------------------------------------
# Identity templates — establish "who the agent is" in the first paragraph.
# ---------------------------------------------------------------------------

IDENTITY_CLAWCODEX_AGENT = (
    "You are an agent for Claw Codex. Given the user's message, you should use the "
    "tools available to complete the task. Complete the task fully — don't gold-plate, "
    "but don't leave it half-done."
)

IDENTITY_READ_ONLY_EXPLORER = (
    "You are a file search specialist for Claw Codex. You excel at thoroughly navigating and exploring codebases."
)

IDENTITY_SOFTWARE_ARCHITECT = (
    "You are a software architect and planning specialist for Claw Codex. "
    "Your role is to explore the codebase and design implementation plans."
)

IDENTITY_CODE_REVIEWER = (
    "You are a code review specialist for Claw Codex. You examine diffs, surface "
    "risks, and suggest concrete, minimal fixes."
)

IDENTITY_TEST_RUNNER = (
    "You are a test-runner specialist for Claw Codex. You execute the test suite "
    "and report failures with actionable diagnostics."
)

IDENTITY_DOCS_WRITER = (
    "You are a documentation specialist for Claw Codex. You produce concise, "
    "accurate documentation grounded in the existing codebase."
)

IDENTITY_WEB_RESEARCHER = (
    "You are a web research specialist for Claw Codex. You find and synthesise "
    "authoritative information from public sources."
)


# ---------------------------------------------------------------------------
# Action norms — behavioural constraints and recommended practices.
# Each norm is a self-contained markdown block that can be appended to
# the system prompt in any order; concatenating multiple norms is the
# normal composition pattern.
# ---------------------------------------------------------------------------

NORM_READ_ONLY = """\
=== READ-ONLY MODE — NO FILE MODIFICATIONS ===
This is a READ-ONLY task. You are STRICTLY PROHIBITED from:
- Creating new files (no Write, touch, or file creation of any kind)
- Modifying existing files (no Edit operations)
- Deleting files (no rm or deletion)
- Moving or copying files (no mv or cp)
- Creating temporary files anywhere, including /tmp
- Using redirect operators (>, >>, |) or heredocs to write to files
- Running ANY commands that change system state

Your role is EXCLUSIVELY to read and analyse. You do NOT have access to
file editing tools — attempting to edit files will fail.
"""

NORM_CODE_AUTHOR = """\
When writing or modifying code:
- Edit existing files in place rather than creating new ones.
- Match the surrounding code's style, naming, and module boundaries.
- Do not add features beyond what was asked.
- Do not introduce speculative abstractions — three similar lines beat
  a premature helper.
- Do not add docstrings, type annotations, or comments to code you did
  not change.
- Run the existing test suite (or a focused subset) before reporting
  the task as done.
"""

NORM_WEB_RESEARCHER = """\
When researching online:
- Prefer primary sources (official documentation, source repositories)
  over secondary write-ups.
- Quote verbatim when exact wording matters; paraphrase otherwise.
- Note the access date for time-sensitive claims.
- Do not invent URLs — only cite ones you actually fetched.
- Stop after the user has enough to act on; do not chase rabbit holes.
"""

NORM_GIT_OPERATOR = """\
When working with git:
- Never run destructive operations (reset --hard, push --force, branch -D)
  without explicit user confirmation.
- Keep commits small and focused — one logical change per commit.
- Use ``git status`` / ``git diff`` / ``git log`` to verify state before
  and after any action.
- Never amend or rewrite published history.
"""

NORM_DIFF_FOCUSED = """\
When reviewing diffs:
- Focus on the actual lines changed; do not propose restructuring of
  unchanged code.
- For each issue, cite the file and line and explain the impact.
- Distinguish blocking issues (correctness, security, data loss) from
  style suggestions.
- When suggesting a fix, provide a minimal patch rather than prose.
"""


# ---------------------------------------------------------------------------
# Tool allow-list presets. ``None`` or ``["*"]`` always means "all tools"
# (per ``AgentDefinition`` semantics). The presets below are *allow* lists
# — the runtime always layers ``ALL_AGENT_DISALLOWED_TOOLS`` on top via
# :func:`clawcodex_ext.agent.registry._normalise_disallowed`.
# ---------------------------------------------------------------------------

TOOL_SET_READ_ONLY: list[str] = [
    "Read",
    "Glob",
    "Grep",
    "Bash",  # caller is expected to use only read-only Bash subcommands
]

TOOL_SET_AUTHOR: list[str] = [
    "Read",
    "Glob",
    "Grep",
    "Edit",
    "Write",
    "Bash",
    "NotebookEdit",
]

TOOL_SET_WEB_ONLY: list[str] = [
    "WebSearch",
    "WebFetch",
    "Read",
]

TOOL_SET_TESTING: list[str] = [
    "Read",
    "Glob",
    "Grep",
    "Bash",
    "TodoWrite",
]


# ---------------------------------------------------------------------------
# Combiner
# ---------------------------------------------------------------------------


def build_agent_prompt(
    *,
    identity: str,
    norms: list[str] | None = None,
    extra: str = "",
) -> str:
    """Compose an agent's system prompt from identity + norms + extra.

    The output mirrors the structure produced by
    :func:`src.agent.prompt.get_agent_system_prompt`: the identity
    string is the opening paragraph, behavioural norms are appended
    as separate blocks in the order given, and any agent-specific
    guidance goes in ``extra`` at the end.

    Empty strings and ``None`` entries are dropped so callers can
    conditionally include norms without manual string juggling.
    """
    parts: list[str] = []
    identity = identity.strip()
    if identity:
        parts.append(identity)
    for n in norms or []:
        if n and n.strip():
            parts.append(n.strip())
    extra = extra.strip()
    if extra:
        parts.append(extra)
    return "\n\n".join(parts)


__all__ = [
    # Identities
    "IDENTITY_CLAWCODEX_AGENT",
    "IDENTITY_READ_ONLY_EXPLORER",
    "IDENTITY_SOFTWARE_ARCHITECT",
    "IDENTITY_CODE_REVIEWER",
    "IDENTITY_TEST_RUNNER",
    "IDENTITY_DOCS_WRITER",
    "IDENTITY_WEB_RESEARCHER",
    # Norms
    "NORM_READ_ONLY",
    "NORM_CODE_AUTHOR",
    "NORM_WEB_RESEARCHER",
    "NORM_GIT_OPERATOR",
    "NORM_DIFF_FOCUSED",
    # Tool sets
    "TOOL_SET_READ_ONLY",
    "TOOL_SET_AUTHOR",
    "TOOL_SET_WEB_ONLY",
    "TOOL_SET_TESTING",
    # Combiner
    "build_agent_prompt",
]
