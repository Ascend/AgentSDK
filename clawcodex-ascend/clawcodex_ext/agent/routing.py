"""F-88 P88-C — prompt-to-subagent-type classifier.

When the Agent tool is invoked without an explicit ``subagent_type``,
this module decides whether the user's prompt semantically points at
the Explore or Plan one-shot agents, or should fall through to
``general-purpose``.

The classifier is a pure function — no I/O, no LLM call, no
side-effects. It runs on every Agent-tool invocation, so it must be
fast (sub-millisecond) and deterministic.

Phrase tables
-------------
The two tables below are the source of truth for routing. Each entry
is a substring matched (case-insensitively) against the user's
prompt. We use substrings (not whole-word regex) for two reasons:

1. The intent phrase in natural language rarely has clean word
   boundaries. "find all the python files" contains "find all" as a
   substring and the user almost certainly means Explore; a
   whole-word match would miss it.
2. The table is small (40 entries total) and the prompts are short
   (≤ 1 KB), so the O(n*m) cost is irrelevant.

Tie-breaking
------------
If both tables score > 0, the higher score wins. Ties go to Plan.
The reasoning: a planning request is a stronger signal than an
exploration request because the user is committing to "act on this";
they can always issue a follow-up Explore with a more focused
prompt. The Plan-tie-break is a deliberate product call — debate
is deferrable, the user can override with explicit
``subagent_type``.
"""

from __future__ import annotations

from typing import Iterable, Final


# --- Public constants ---

GENERAL_PURPOSE_FALLBACK: Final[str] = "general-purpose"


# --- Phrase tables (immutable tuples) ---

# Substrings (case-insensitive) that signal an Explore intent. Each
# entry is the canonical phrase as it typically appears in a prompt.
_EXPLORE_PHRASES: Final[tuple[str, ...]] = (
    "explore the codebase",
    "look around",
    "find files",
    "find all",
    "what files",
    "where is",
    "where are",
    "search for",
    "grep for",
    "scan the",
    "list the files",
    "show me the files",
    "understand the structure",
    "how is the code organized",
    "what modules",
    "trace through",
    "walk me through",
    "summarize the",
    "overview of",
    "what does this project",
)

# Substrings that signal a Plan intent.
_PLAN_PHRASES: Final[tuple[str, ...]] = (
    "make a plan",
    "create a plan",
    "draft a plan",
    "build a plan",
    "plan out",
    "plan the",
    "design the",
    "design a",
    "come up with a plan",
    "propose a plan",
    "outline the steps",
    "outline how",
    "step by step",
    "what's the best approach",
    "how should we approach",
    "what would it take to",
    "how would you implement",
    "architect a",
    "architect the",
    "implementation plan",
)


def _score(prompt_lower: str, phrases: tuple[str, ...]) -> int:
    """Count how many distinct phrases from ``phrases`` appear in
    ``prompt_lower``. Multiple occurrences of the same phrase count
    once — the classifier is asking "is this intent present?", not
    "how emphatically?".
    """
    return sum(1 for p in phrases if p in prompt_lower)


def classify_prompt_to_subagent_type(
    prompt: str,
    available: Iterable[str],
) -> str:
    """Return the best ``subagent_type`` for ``prompt``.

    The return is one of ``"Explore"``, ``"Plan"``, or
    :data:`GENERAL_PURPOSE_FALLBACK`. The winner is constrained to be
    in ``available`` — if the winning type is not available, the
    function returns the fallback (we never name a type the runtime
    cannot dispatch).

    Decision rules, in order:

    1. Empty/whitespace-only prompt → fallback.
    2. Lower-case the prompt, score each table.
    3. Higher score wins.
    4. Tie (both > 0, equal) → Plan.
    5. Both zero → fallback.
    6. The chosen type must be in ``available``; otherwise fallback.
    """
    if not prompt or not prompt.strip():
        return GENERAL_PURPOSE_FALLBACK

    prompt_lower = prompt.lower()
    explore_score = _score(prompt_lower, _EXPLORE_PHRASES)
    plan_score = _score(prompt_lower, _PLAN_PHRASES)

    if explore_score == 0 and plan_score == 0:
        return GENERAL_PURPOSE_FALLBACK

    # Plan wins on ties (deliberate product call — see module docstring).
    if plan_score >= explore_score:
        winner = "Plan"
    else:
        winner = "Explore"

    available_set = set(available)
    if winner not in available_set:
        return GENERAL_PURPOSE_FALLBACK
    return winner


__all__ = [
    "GENERAL_PURPOSE_FALLBACK",
    "classify_prompt_to_subagent_type",
]
