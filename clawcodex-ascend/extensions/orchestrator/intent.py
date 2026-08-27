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

"""Operator intent & comment-command semantics for issue trackers.

Split out of ``tracker.py``: pure parsing / merging logic with no
dependency on the adapter contract.  Callers that need ``Intent`` /
``Command`` resolution or the priority-merge rules import from here
directly; ``tracker.py`` re-exports these symbols for back-compat.
"""

from __future__ import annotations

import re
from enum import Enum

# Default label conventions for the three retry intents. Adapters accept an
# override at construction time; the keys map Intent values to label names.
DEFAULT_INTENT_LABELS: dict[str, str] = {
    "retry": "agent:retry",
    "followup": "agent:follow-up",
    "blocked": "agent:blocked",
    "rebase": "agent:rebase",
}


# Regex for ``/agent <subcommand> [args]`` at the start of a line / body.
# Permissive trailing text: any args / reason after the subcommand.
# Includes ``rebase`` in the recognized subcommand set.
_AGENT_COMMAND_RE = re.compile(
    r"^/agent\s+(retry|follow-up|unblock|rebase)\b[^\n]*",
    re.IGNORECASE | re.MULTILINE,
)


def _command_eligible_body(body: str) -> str:
    """Return only the lines of ``body`` where an operator command is valid.

    Comment bodies are Markdown. A ``/agent`` line pasted inside a
    fenced or indented code block, or inside a blockquote, is example
    text / log output rather than a real command — honoring it would
    trigger side-effecting orchestration (retry / rebase). Reassemble
    only plain-paragraph lines before applying :data:`_AGENT_COMMAND_RE`.
    """
    eligible: list[str] = []
    fence: str | None = None
    for line in body.splitlines():
        stripped = line.strip()
        if fence is not None:
            if stripped.startswith(fence):
                fence = None
            continue
        if stripped.startswith(("```", "~~~")):
            fence = stripped[:3]
            continue
        if line.startswith((" ", "\t", ">")):
            continue
        eligible.append(line)
    return "\n".join(eligible)


class Intent(str, Enum):
    """Operator intent expressed via issue labels or comment commands.

    Each issue may carry an intent that overrides the default
    4-layer "already handled" defense in the orchestrator.

      - NONE: no operator intent recorded
      - RETRY: reset local registry entry + close remote PR + new run
      - FOLLOWUP: keep PR, append commit on same branch
      - BLOCKED: permanently skip the issue
      - REBASE: orchestrator rebases the existing PR's feature branch
        onto the latest base and force-pushes. Agent reentry only
        triggered if rebase leaves content conflicts.
    """

    NONE = "none"
    RETRY = "retry"
    FOLLOWUP = "followup"
    BLOCKED = "blocked"
    REBASE = "rebase"


def intent_from_label_set(
    labels: list[str] | None,
    intent_labels: dict[str, str] | None = None,
) -> Intent:
    """Resolve an Intent from a list of issue labels.

    Priority rules:
      - ``agent:blocked`` wins over any other intent (permanent skip).
      - ``agent:rebase`` wins over RETRY/FOLLOWUP (rebase touches the
        remote history directly; treat as higher priority than
        follow-up commit appending).
      - ``agent:retry`` + ``agent:follow-up`` together → FOLLOWUP is more
        conservative (keeps PR evidence), so it wins.
      - Otherwise return whichever single intent label is present, or NONE.
    """
    if not labels:
        return Intent.NONE
    mapping = intent_labels or DEFAULT_INTENT_LABELS
    retry_label = _normalize_label(mapping.get("retry", ""))
    followup_label = _normalize_label(mapping.get("followup", ""))
    blocked_label = _normalize_label(mapping.get("blocked", ""))
    rebase_label = _normalize_label(mapping.get("rebase", ""))
    normalized = {_normalize_label(label) for label in labels if label}
    if blocked_label and blocked_label in normalized:
        return Intent.BLOCKED
    if rebase_label and rebase_label in normalized:
        return Intent.REBASE
    if followup_label and followup_label in normalized:
        return Intent.FOLLOWUP
    if retry_label and retry_label in normalized:
        return Intent.RETRY
    return Intent.NONE


class Command(str, Enum):
    """Operator command expressed via an issue comment.

    Distinct from ``Intent`` because commands may carry side effects
    (e.g. UNBLOCK clears an abandoned status) and because not every
    command maps to a run-mode intent.

    ``REBASE``: a comment-issued rebase request that maps to
    ``Intent.REBASE``.
    """

    RETRY = "retry"
    FOLLOWUP = "followup"
    UNBLOCK = "unblock"
    REBASE = "rebase"


def parse_agent_command(body: str | None) -> Command | None:
    """Extract a ClawCodex operator command from a comment body.

    Recognized forms (case-insensitive, at the start of any line in the body):
      - ``/agent retry [reason...]``
      - ``/agent follow-up [note...]``
      - ``/agent unblock``
      - ``/agent rebase [reason...]``

    Returns the matched ``Command`` or ``None`` if no recognized command
    is present. Only the first match is returned — operators that
    pile commands into one comment will get the first one honored.

    Lines inside fenced / indented code blocks and blockquotes are
    ignored so that pasted examples or logs never parse as a command.
    """
    if not body:
        return None
    match = _AGENT_COMMAND_RE.search(_command_eligible_body(body))
    if not match:
        return None
    raw = match.group(1).lower()
    if raw == "retry":
        return Command.RETRY
    if raw == "follow-up":
        return Command.FOLLOWUP
    if raw == "unblock":
        return Command.UNBLOCK
    if raw == "rebase":
        return Command.REBASE
    return None


def command_to_intent(command: Command) -> Intent:
    """Map a Command to the Intent the orchestrator should run with.

    ``UNBLOCK`` is a state-clearing meta-command and has no direct
    run-mode intent; it returns Intent.NONE so the next poll re-
    applies the label-based intent (or stays NONE if the operator
    removed the agent:blocked label too).

    ``Command.REBASE`` maps to ``Intent.REBASE``.
    """
    if command is Command.RETRY:
        return Intent.RETRY
    if command is Command.FOLLOWUP:
        return Intent.FOLLOWUP
    if command is Command.REBASE:
        return Intent.REBASE
    return Intent.NONE


def merge_intents(label_intent: Intent, command_intent: Intent) -> Intent:
    """Merge a label-derived Intent with a command-derived Intent.

    Design rationale: a comment command can override a label intent,
    but BLOCKED is sticky — it is a permanent skip and only the unblock
    command / CLI override can lift it. Between RETRY and FOLLOWUP the
    conservative rule wins: FOLLOWUP preserves PR evidence. This mirrors
    the label-only priority in :func:`intent_from_label_set`.

    Precedence (high → low):
      1. Intent.BLOCKED — sticky permanent skip.
      2. Intent.REBASE — orchestrator-side rebase is a
         remote-history-touching operation that beats the more
         conservative RETRY/FOLLOWUP branches.
      3. The more conservative of {RETRY, FOLLOWUP} = FOLLOWUP.
      4. Otherwise: command_intent wins over label_intent.
      5. Otherwise: whichever is non-NONE; else NONE.
    """
    if label_intent is Intent.BLOCKED or command_intent is Intent.BLOCKED:
        return Intent.BLOCKED
    if label_intent is Intent.REBASE or command_intent is Intent.REBASE:
        return Intent.REBASE
    if label_intent is Intent.FOLLOWUP or command_intent is Intent.FOLLOWUP:
        return Intent.FOLLOWUP
    if command_intent is not Intent.NONE:
        return command_intent
    if label_intent is not Intent.NONE:
        return label_intent
    return Intent.NONE


def merge_intents_with_cli(
    label_intent: Intent,
    command_intent: Intent,
    cli_intent: Intent,
) -> Intent:
    """Merge three intent sources (label / comment / CLI).

    Used by :meth:`Orchestrator._resolve_intent` to combine the three
    ways an operator can drive a retry:

      1. **Label** — ``agent:retry`` / ``agent:follow-up`` /
         ``agent:blocked`` / ``agent:rebase`` on the issue.
      2. **Comment** — ``/agent retry`` / ``/agent follow-up`` /
         ``/agent unblock`` / ``/agent rebase`` in the issue thread.
      3. **CLI** — ``clawcodex-dev orchestrator issue retry --mode
         reset|followup|unblock|rebase`` which writes ``registry.intent``
         with ``intent_source="cli"``. This is the
         operator's authoritative local command and is the ONLY
         source that survives even when the remote issue tracker is
         unreachable / read-only / local-only (LocalTracker).

    Precedence (high → low):
      1. Intent.BLOCKED — sticky permanent skip (any source).
      2. Intent.REBASE — remote-history rebase beats
         retry/followup which only affect local state.
      3. The more conservative of {RETRY, FOLLOWUP} = FOLLOWUP.
      4. CLI intent — operator's local command beats remote signals.
      5. Comment command beats label-only intent.
      6. Otherwise: whichever is non-NONE; else NONE.

    Why CLI wins: the CLI is the operator's deliberate, authenticated
    local action. Remote signals (label, comment) can be stale,
    spoofed, or lost. When the operator runs ``clawcodex-dev orchestrator
    issue retry --id 5 --mode reset --force``, they expect the daemon
    to honor that intent on the next poll regardless of what the
    remote tracker says.
    """
    if Intent.BLOCKED in (label_intent, command_intent, cli_intent):
        return Intent.BLOCKED
    if Intent.REBASE in (label_intent, command_intent, cli_intent):
        return Intent.REBASE
    if Intent.FOLLOWUP in (label_intent, command_intent, cli_intent):
        return Intent.FOLLOWUP
    if cli_intent is not Intent.NONE:
        return cli_intent
    if command_intent is not Intent.NONE:
        return command_intent
    if label_intent is not Intent.NONE:
        return label_intent
    return Intent.NONE


def _normalize_label(value: str) -> str:
    """Normalize a label for case-insensitive comparison."""
    return value.strip().lower()
