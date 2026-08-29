#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# -------------------------------------------------------------------------
# This file is part of the AgentSDK project.
#
# Originally from the clawcodex project:
#   https://github.com/agentforce314/clawcodex
#   Copyright (c) 2026 Clawd Codex Team
#   Licensed under the MIT License. See clawcodex-ascend/LICENSES/Clawd-Codex-MIT.txt.
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

"""Framework-agnostic completer primitives shared by the REPL and TUI.

Both prompt_toolkit (``clawcodex_ext.repl.core``) and Textual
(``clawcodex_ext.tui.widgets.prompt_input``) implement the same slash
and message-history completion behavior. This module is the single
source of truth for the pure-function parts: token detection, fuzzy
matching, and ranking. The framework glue (Completer subclass, OptionList
rendering) stays at the call site.

Keep this module import-clean: no prompt_toolkit, no textual, no Rich.
"""

from __future__ import annotations

from typing import Any, Iterable


def current_slash_token(text: str) -> tuple[str | None, int]:
    """Return ``(token, start_index)`` for the slash command under the cursor.

    Byte-identical to the previous implementations in
    ``clawcodex_ext.repl.core._SlashOnlyCompleter._current_slash_token``
    and ``clawcodex_ext.tui.widgets.prompt_input._current_slash_token``.

    Semantics locked in by ``tests.input.test_slash_completer`` and
    ``tests.tui.test_slash_token_parser``:

    * Empty text → ``(None, 0)``.
    * ``/`` at the start of the buffer, no space yet → ``(text, 0)``.
    * ``/`` mid-input preceded by whitespace, no space in the suffix
      → ``(text[i:], i)``.
    * Any other layout → ``(None, 0)``.
    """

    if not text:
        return None, 0
    if text.startswith("/"):
        space_idx = text.find(" ")
        if space_idx != -1:
            return None, 0
        return text, 0
    for i in range(len(text) - 1, -1, -1):
        ch = text[i]
        if ch == "/":
            if i > 0 and not text[i - 1].isspace():
                return None, 0
            token = text[i:]
            if " " in token:
                return None, 0
            return token, i
        if ch.isspace():
            return None, 0
    return None, 0


def fuzzy_match(name: str, partial: str) -> bool:
    """Prefix wins; subsequence is the fallback.

    Equivalent to the old ``_fuzzy_match`` in the TUI (which was a
    superset of the REPL's ``_fuzzy_subseq``). Empty ``partial`` is
    treated as a match.
    """

    if not partial:
        return True
    if name.startswith(partial):
        return True
    i = 0
    for ch in name:
        if ch == partial[i]:
            i += 1
            if i == len(partial):
                return True
    return False


# Back-compat alias for callers that import ``fuzzy_subseq`` by name.
fuzzy_subseq = fuzzy_match


def rank_suggestions(
    suggestions: Iterable[Any],
    partial: str,
    *,
    max_results: int | None = None,
) -> list[tuple[Any, str | None]]:
    """Filter + rank rich command suggestions.

    Returns ``[(suggestion, matched_alias), ...]`` sorted by:

    * Rank: exact name (0) → exact alias (1) → prefix name (2) →
      prefix alias (3) → fuzzy name (5) → fuzzy alias (6).
    * Secondary: insertion index for empty partial (preserves the
      provider's ordering of built-ins before skills), otherwise
      ``len(name)`` so shorter names win ties.
    * Tertiary: ``name.lower()`` alphabetical.

    Identical algorithm to the prior REPL ``_rich_completions`` and
    TUI ``_options_from_suggestions`` — extracted to one place so the
    two surfaces can no longer drift.
    """

    scored: list[tuple[int, int, Any, str | None]] = []
    seen: set[str] = set()
    for idx, sugg in enumerate(suggestions):
        name = getattr(sugg, "name", None)
        if not isinstance(name, str) or not name:
            continue
        name_lc = name.lower()
        if name_lc in seen:
            continue
        aliases = tuple(getattr(sugg, "aliases", ()) or ())
        matched_alias: str | None = None
        rank: int | None = None
        if not partial:
            rank = 0
        elif name_lc == partial:
            rank = 0
        else:
            alias_exact = next((a for a in aliases if a.lower() == partial), None)
            if alias_exact:
                rank = 1
                matched_alias = alias_exact
            elif name_lc.startswith(partial):
                rank = 2
            else:
                alias_prefix = next(
                    (a for a in aliases if a.lower().startswith(partial)),
                    None,
                )
                if alias_prefix:
                    rank = 3
                    matched_alias = alias_prefix
                elif fuzzy_match(name_lc, partial):
                    rank = 5
                else:
                    alias_fuzzy = next(
                        (a for a in aliases if fuzzy_match(a.lower(), partial)),
                        None,
                    )
                    if alias_fuzzy:
                        rank = 6
                        matched_alias = alias_fuzzy
        if rank is None:
            continue
        seen.add(name_lc)
        secondary = idx if not partial else len(name)
        scored.append((rank, secondary, sugg, matched_alias))

    scored.sort(key=lambda t: (t[0], t[1], t[2].name.lower()))
    if max_results is not None:
        scored = scored[:max_results]
    return [(sugg, alias) for _, _, sugg, alias in scored]


def rank_message_history(
    messages: Iterable[str],
    partial: str,
    *,
    limit: int = 5,
) -> list[str]:
    """Score previous user messages against the typed partial.

    Returns the top ``limit`` matches, ordered: exact match first,
    then prefix matches (most recent first — input order matters when
    callers pass messages oldest-first), then subsequence matches.

    Mirrors the prior REPL ``_MessageHistoryCompleter`` and TUI
    ``_refresh_message_suggestions`` scoring. The two sides differ
    only in the ``limit`` (REPL=5, TUI=10) and the caller's
    post-processing for display truncation; both are caller-side.
    """

    partial_lower = partial.lower()
    scored: list[tuple[int, int, str]] = []
    seen: set[str] = set()
    for idx, msg in enumerate(messages):
        if not isinstance(msg, str):
            continue
        msg_key = msg.lower()
        if msg_key in seen:
            continue
        seen.add(msg_key)
        if msg_key == partial_lower:
            scored.append((0, idx, msg))
        elif msg.lower().startswith(partial_lower):
            scored.append((1, idx, msg))
        elif fuzzy_match(msg.lower(), partial_lower):
            scored.append((2, idx, msg))
    scored.sort(key=lambda t: (t[0], t[1]))
    return [m for _, _, m in scored[:limit]]
