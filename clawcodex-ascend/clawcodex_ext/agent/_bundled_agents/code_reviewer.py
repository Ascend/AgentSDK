"""Bundled ``code-reviewer`` agent.

Demonstrates the @register + policy composition pattern: pick an
identity template, attach one or more action norms, and stitch the
final system prompt together with :func:`build_agent_prompt`.
"""

from __future__ import annotations

from clawcodex_ext.agent.policy import (
    IDENTITY_CODE_REVIEWER,
    NORM_DIFF_FOCUSED,
    NORM_READ_ONLY,
    TOOL_SET_READ_ONLY,
    build_agent_prompt,
)
from clawcodex_ext.agent.registry import AgentRegistry

_SYSTEM_PROMPT = build_agent_prompt(
    identity=IDENTITY_CODE_REVIEWER,
    norms=[NORM_READ_ONLY, NORM_DIFF_FOCUSED],
    extra=(
        "When done, end your report with two sections:\n"
        "## Blocking issues\n"
        "Each item: file:line, impact, and a minimal fix.\n"
        "## Suggestions\n"
        "Each item: file:line, why it matters, optional patch."
    ),
)


@AgentRegistry.register(
    "code-reviewer",
    when_to_use=(
        "Code-review specialist for diffs and PRs. Use after a logical chunk "
        "of code is written to get an independent review before reporting "
        "completion. Read-only — never edits files itself."
    ),
    tools=TOOL_SET_READ_ONLY,
    permission_mode="default",
)
def _code_reviewer_prompt() -> str:
    return _SYSTEM_PROMPT
