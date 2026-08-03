"""Unit tests for :mod:`src.agent.routing` (F-88 P88-C).

The classifier is a pure function — no I/O, no globals. The
contract being pinned here:

* "explore" / "find files" style prompts → Explore
* "make a plan" / "design a" style prompts → Plan
* Unrelated prompts → general-purpose
* Case-insensitive matching
* Ties go to Plan
* Type not in ``available`` → general-purpose
* Empty prompt → general-purpose
"""

from __future__ import annotations

import pytest

from src.agent.routing import (
    GENERAL_PURPOSE_FALLBACK,
    classify_prompt_to_subagent_type,
)


# All known agent types for the routing tests. The classifier must
# respect ``available``; we hand it the full set so the winner is
# never constrained by availability unless a test explicitly says so.
_ALL_TYPES = frozenset({"Explore", "Plan", "general-purpose"})


# ---------------------------------------------------------------------------
# Explore intent
# ---------------------------------------------------------------------------


def test_explore_phrase_scores_explore() -> None:
    assert classify_prompt_to_subagent_type("explore the codebase", _ALL_TYPES) == "Explore"


def test_find_files_is_explore() -> None:
    assert classify_prompt_to_subagent_type("find files matching the pattern", _ALL_TYPES) == "Explore"


def test_where_is_is_explore() -> None:
    assert classify_prompt_to_subagent_type("where is the database config", _ALL_TYPES) == "Explore"


# ---------------------------------------------------------------------------
# Plan intent
# ---------------------------------------------------------------------------


def test_make_a_plan_is_plan() -> None:
    assert classify_prompt_to_subagent_type("make a plan for the new auth flow", _ALL_TYPES) == "Plan"


def test_design_a_is_plan() -> None:
    assert classify_prompt_to_subagent_type("design a new caching layer", _ALL_TYPES) == "Plan"


def test_step_by_step_is_plan() -> None:
    assert classify_prompt_to_subagent_type("walk me through this step by step", _ALL_TYPES) == "Plan"


# ---------------------------------------------------------------------------
# Fallback to general-purpose
# ---------------------------------------------------------------------------


def test_ambiguous_prompt_falls_back() -> None:
    """A prompt with no Explore/Plan phrases falls through to general-purpose."""
    assert classify_prompt_to_subagent_type("fix the typo in the README", _ALL_TYPES) == GENERAL_PURPOSE_FALLBACK


def test_empty_prompt_is_general_purpose() -> None:
    assert classify_prompt_to_subagent_type("", _ALL_TYPES) == GENERAL_PURPOSE_FALLBACK


def test_whitespace_only_prompt_is_general_purpose() -> None:
    assert classify_prompt_to_subagent_type("   \n\t  ", _ALL_TYPES) == GENERAL_PURPOSE_FALLBACK


# ---------------------------------------------------------------------------
# Case-insensitive
# ---------------------------------------------------------------------------


def test_case_insensitive_explore() -> None:
    assert classify_prompt_to_subagent_type("EXPLORE THE CODEBASE", _ALL_TYPES) == "Explore"


def test_case_insensitive_plan() -> None:
    assert classify_prompt_to_subagent_type("Make A Plan to refactor", _ALL_TYPES) == "Plan"


# ---------------------------------------------------------------------------
# Substring boundary
# ---------------------------------------------------------------------------


def test_phrase_must_be_substring_not_word_boundary() -> None:
    """``"planetary"`` must NOT match the Plan phrase ``"plan the"``
    / ``"plan out"``. We use a substring match, so a phrase like
    ``"plan"`` would over-match; verify the phrases are
    multi-word so accidental containment is rare.
    """
    # "planetary alignment" contains "plan" only as a prefix of
    # "planetary"; none of the multi-word Plan phrases match.
    assert classify_prompt_to_subagent_type("planetary alignment check", _ALL_TYPES) == GENERAL_PURPOSE_FALLBACK


# ---------------------------------------------------------------------------
# Tie-breaking
# ---------------------------------------------------------------------------


def test_plan_wins_on_tie() -> None:
    """A prompt that contains both an Explore and a Plan phrase
    (equal score 1 each) routes to Plan — deliberate product call
    documented in the module docstring.
    """
    prompt = "make a plan to explore the codebase"  # 1 plan + 1 explore
    assert classify_prompt_to_subagent_type(prompt, _ALL_TYPES) == "Plan"


def test_higher_score_wins() -> None:
    """Two Explore phrases > one Plan phrase → Explore."""
    prompt = "find all the files where is the config"
    # 2 explore phrases (find all, where is) vs 0 plan phrases
    assert classify_prompt_to_subagent_type(prompt, _ALL_TYPES) == "Explore"


# ---------------------------------------------------------------------------
# Availability constraint
# ---------------------------------------------------------------------------


def test_unavailable_type_falls_back() -> None:
    """If the winning type is not in ``available``, fall back."""
    # The winning type would be Explore, but we say Explore is not
    # available — should fall back to general-purpose.
    only_general = frozenset({"general-purpose"})
    assert classify_prompt_to_subagent_type("explore the codebase", only_general) == GENERAL_PURPOSE_FALLBACK


def test_available_subset_still_routes() -> None:
    """If the available set includes Explore but not Plan, an Explore
    prompt still routes correctly (no regression).
    """
    avail = frozenset({"Explore", "general-purpose"})
    assert classify_prompt_to_subagent_type("find files", avail) == "Explore"


# ---------------------------------------------------------------------------
# Return-type contract
# ---------------------------------------------------------------------------


def test_returns_string() -> None:
    """The return type is plain ``str`` (not an enum / Literal)."""
    result = classify_prompt_to_subagent_type("explore the codebase", _ALL_TYPES)
    assert isinstance(result, str)
    assert result in {"Explore", "Plan", GENERAL_PURPOSE_FALLBACK}


def test_fallback_constant_value() -> None:
    """The fallback constant is the documented string ``"general-purpose"``."""
    assert GENERAL_PURPOSE_FALLBACK == "general-purpose"


# ---------------------------------------------------------------------------
# Parametrised sanity sweep — one of each
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "prompt,expected",
    [
        ("search for the auth module", "Explore"),
        ("scan the tests directory", "Explore"),
        ("trace through the request lifecycle", "Explore"),
        ("draft a plan to ship the new feature", "Plan"),
        ("architect a multi-tenant data layer", "Plan"),
        ("what would it take to support streaming", "Plan"),
        ("add a typo fix in line 42", GENERAL_PURPOSE_FALLBACK),
        ("explain the build process", GENERAL_PURPOSE_FALLBACK),
    ],
)
def test_sweep_prompts_route_correctly(prompt: str, expected: str) -> None:
    assert classify_prompt_to_subagent_type(prompt, _ALL_TYPES) == expected
