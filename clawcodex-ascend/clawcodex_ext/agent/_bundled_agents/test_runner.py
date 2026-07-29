"""Bundled ``test-runner`` agent."""

from __future__ import annotations

from clawcodex_ext.agent.policy import (
    IDENTITY_TEST_RUNNER,
    NORM_GIT_OPERATOR,
    TOOL_SET_TESTING,
    build_agent_prompt,
)
from clawcodex_ext.agent.registry import AgentRegistry

_SYSTEM_PROMPT = build_agent_prompt(
    identity=IDENTITY_TEST_RUNNER,
    norms=[NORM_GIT_OPERATOR],
    extra=(
        "Run the smallest test subset that exercises the change first; "
        "only run the full suite if the focused subset passes. For each "
        "failure report: file:line, expected vs actual, and your best "
        "guess at root cause. Do not modify any files — surface the "
        "fix as a recommendation to the caller."
    ),
)


@AgentRegistry.register(
    "test-runner",
    when_to_use=(
        "Test-runner specialist. Use after writing or modifying code to "
        "execute the relevant test suite and report results. Read-only — "
        "never edits source code itself."
    ),
    tools=TOOL_SET_TESTING,
    permission_mode="default",
)
def _test_runner_prompt() -> str:
    return _SYSTEM_PROMPT
