"""Unit tests for ``clawcodex_ext.agent.policy`` primitives."""

from __future__ import annotations


# pylint: disable=E0611
from clawcodex_ext.agent.policy import (
    IDENTITY_CLAWCODEX_AGENT,
    IDENTITY_CODE_REVIEWER,
    NORM_CODE_AUTHOR,
    NORM_DIFF_FOCUSED,
    NORM_READ_ONLY,
    TOOL_SET_AUTHOR,
    TOOL_SET_READ_ONLY,
    build_agent_prompt,
)


def test_build_agent_prompt_identity_only():
    out = build_agent_prompt(identity=IDENTITY_CLAWCODEX_AGENT)
    assert out == IDENTITY_CLAWCODEX_AGENT.strip()


def test_build_agent_prompt_joins_in_order():
    out = build_agent_prompt(
        identity=IDENTITY_CODE_REVIEWER,
        norms=[NORM_READ_ONLY, NORM_DIFF_FOCUSED],
        extra="Final notes.",
    )
    # Order is preserved: identity → norms → extra.
    assert out.index(IDENTITY_CODE_REVIEWER) < out.index("READ-ONLY MODE")
    assert out.index("READ-ONLY MODE") < out.index("When reviewing diffs")
    assert out.index("When reviewing diffs") < out.index("Final notes.")
    # All three blocks are present.
    assert "READ-ONLY MODE" in out
    assert "When reviewing diffs" in out
    assert "Final notes." in out


def test_build_agent_prompt_drops_empty_norms():
    out = build_agent_prompt(
        identity=IDENTITY_CLAWCODEX_AGENT,
        norms=[NORM_READ_ONLY, "", None, NORM_CODE_AUTHOR],
    )
    # Empty / None norms are dropped — no double blank lines, no "None" literal.
    assert "\n\n\n" not in out
    assert "None" not in out
    assert "READ-ONLY MODE" in out
    assert "When writing or modifying code" in out


def test_build_agent_prompt_extra_optional():
    # Extra is optional and an empty value should be dropped.
    out = build_agent_prompt(identity=IDENTITY_CLAWCODEX_AGENT, extra="")
    assert out == IDENTITY_CLAWCODEX_AGENT.strip()


def test_tool_sets_are_lists():
    for preset in (TOOL_SET_READ_ONLY, TOOL_SET_AUTHOR):
        assert isinstance(preset, list)
        assert all(isinstance(t, str) and t for t in preset)
