# -------------------------------------------------------------------------
# This file is part of the AgentSDK project.
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
#
# Copyright (c) 2026 Clawd Codex Team
# SPDX-License-Identifier: MIT
# Source: https://github.com/agentforce314/clawcodex
# ClawCodex-derived portions remain licensed under the MIT License.
# See clawcodex-ascend/LICENSE.clawcodex.

"""Interactive REPL for Claw Codex."""

from __future__ import annotations
# pylint: disable=C0302,C0412,E0611,E1102,E1128,R0916,R1702,R1711,W0105,W0201,W0221,W0404,W0602,W0621,W0640

import asyncio
from collections import deque
from datetime import datetime
import json
import logging
from pathlib import Path
import sys
import threading
import time
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    from clawcodex_ext.command_system.engine import CommandResult

from clawcodex_ext.utils.completers import (
    current_slash_token,
    rank_message_history,
    rank_suggestions,
)
from clawcodex_ext.permissions.types import (
    PermissionAskReply,
    PermissionAskRequest,
    PermissionMode,
)
from clawcodex_ext.repl.color_scheme import REPLPalette
from src.config import get_selection_mode

try:
    from prompt_toolkit import PromptSession
    from prompt_toolkit.history import FileHistory
    from prompt_toolkit.auto_suggest import AutoSuggestFromHistory, Suggestion
    from prompt_toolkit.styles import Style
    from prompt_toolkit.completion import Completer, Completion, WordCompleter
    from prompt_toolkit.input import ansi_escape_sequences as _pt_ansi_seq
    from prompt_toolkit.keys import Keys as _PTKeys

    # Teach prompt_toolkit to distinguish Shift+Enter from plain Enter.
    #
    # Two distinct Shift+Enter sequences are in the wild; we route both to
    # the same two-key tuple Meta+Enter uses (Escape + ControlM), so a
    # single ``escape, c-m`` binding covers them all:
    #
    # 1. ``\x1b[13;2u`` — Kitty keyboard protocol (Kitty, WezTerm, Ghostty,
    #    iTerm2 with CSI u mode). Not known to prompt_toolkit at all.
    # 2. ``\x1b[27;2;13~`` — xterm ``modifyOtherKeys`` level 2 (xterm with
    #    modifyOtherKeys on, some VSCode configurations). prompt_toolkit
    #    maps this to plain ``ControlM``, so by default it's
    #    indistinguishable from Enter — we override it.
    #
    # This matches the TypeScript reference's behavior in ``useTextInput.ts``
    # which explicitly treats both CSI 13;2u and CSI 27;2;13~ as "insert
    # newline" on Shift+Enter.
    if not hasattr(_pt_ansi_seq, "_clawcodex_shift_enter_registered"):
        _pt_ansi_seq.ANSI_SEQUENCES["\x1b[13;2u"] = (
            _PTKeys.Escape,
            _PTKeys.ControlM,
        )
        _pt_ansi_seq.ANSI_SEQUENCES["\x1b[27;2;13~"] = (
            _PTKeys.Escape,
            _PTKeys.ControlM,
        )
        _pt_ansi_seq._clawcodex_shift_enter_registered = True  # type: ignore[attr-defined]
    try:
        from prompt_toolkit.completion import FuzzyCompleter
    except Exception:  # pragma: no cover
        FuzzyCompleter = None  # type: ignore
    from prompt_toolkit.key_binding import KeyBindings

    _HAS_PROMPT_TOOLKIT = True
except ModuleNotFoundError:  # pragma: no cover
    _HAS_PROMPT_TOOLKIT = False

    class FileHistory:  # type: ignore
        def __init__(self, *args, **kwargs):
            pass

    class AutoSuggestFromHistory:  # type: ignore
        def __init__(self, *args, **kwargs):
            pass

    class Style:  # type: ignore
        @staticmethod
        def from_dict(*args, **kwargs):
            return None

    class WordCompleter:  # type: ignore
        def __init__(self, *args, **kwargs):
            pass

    class Completer:  # type: ignore
        def get_completions(self, *args, **kwargs):
            return iter(())

    class Completion:  # type: ignore
        def __init__(self, *args, **kwargs):
            pass

    FuzzyCompleter = None  # type: ignore

    class KeyBindings:  # type: ignore
        def __init__(self, *args, **kwargs):
            pass

    class PromptSession:  # type: ignore
        def __init__(self, *args, **kwargs):
            pass

        def prompt(self, *args, **kwargs):
            raise EOFError()


class _SlashOnlyCompleter(Completer):
    """Trigger autocompletion only for slash commands, matching the reference
    Claude Code behavior.

    Rules (mirrors ``typescript/src/utils/suggestions/commandSuggestions.ts``):

    * If the whole buffer starts with ``/`` and the cursor is on the first
      token, complete slash commands (prefix match against the command name).
    * If the cursor sits on a ``/``-prefixed token preceded by whitespace,
      complete that mid-input slash command.
    * In every other case (plain words like ``hello``, ``ex``, etc.) return
      no completions so the user can type freely without a suggestion popup.

    When a ``suggestions_provider`` is supplied it carries descriptions and
    optional ``[workflow]`` tags, which surface in the prompt_toolkit menu
    as ``display_meta`` — the same two-column layout the TS reference uses.
    The legacy ``words_provider`` is still honoured for callers that only
    have the flat name list.
    """

    def __init__(self, words_provider, suggestions_provider=None):
        self._words_provider = words_provider
        self._suggestions_provider = suggestions_provider

    def get_completions(self, document, complete_event):  # type: ignore[override]
        text = document.text_before_cursor
        token, token_start = current_slash_token(text)
        if token is None:
            return
        partial = token[1:].lower()  # strip leading '/'
        start_position = token_start - len(text)

        if self._suggestions_provider is not None:
            try:
                suggestions = self._suggestions_provider() or []
            except Exception:  # nosec B110
                suggestions = []
            yield from self._rich_completions(suggestions, partial, start_position)
            return

        words = self._words_provider() or []
        seen: set[str] = set()
        for word in words:
            if not isinstance(word, str) or not word.startswith("/"):
                continue
            name = word[1:]
            key = name.lower()
            if key in seen:
                continue
            if not partial or key.startswith(partial):
                seen.add(key)
                yield Completion(
                    text=word,
                    start_position=start_position,
                    display=word,
                )

    def _rich_completions(self, suggestions, partial, start_position):
        """Yield ``Completion`` entries with ``display`` + ``display_meta``.

        Ranks suggestions via the shared ``rank_suggestions`` helper
        (exact name → exact alias → prefix name → prefix alias → fuzzy
        subsequence) and renders each as a two-column prompt_toolkit
        entry: ``/name (alias)`` left, ``[tag] description`` right.
        Aliases are surfaced in ``(alias)`` only when the typed prefix
        matched the alias, so an unmatched partial does not pollute the
        menu with every alternate name.
        """

        for sugg, matched_alias in rank_suggestions(suggestions, partial):
            alias_text = f" ({matched_alias})" if matched_alias else ""
            display_text = f"/{sugg.name}{alias_text}"
            display_styled = [("class:completion.command", display_text)]
            description = (getattr(sugg, "description", "") or "").strip()
            tag = getattr(sugg, "tag", None)
            meta_parts: list[tuple[str, str]] = []
            if tag:
                meta_parts.append(("class:completion.tag", f"[{tag}] "))
            if description:
                # Collapse internal whitespace so multi-line descriptions
                # render as one row in the prompt_toolkit menu.
                meta_parts.append(("class:completion.description", " ".join(description.split())))
            yield Completion(
                text=f"/{sugg.name}",
                start_position=start_position,
                display=display_styled,
                display_meta=meta_parts if meta_parts else None,
            )


# Back-compat: downstream subclasses (e.g. ClawCodexExtREPL in
# clawcodex_ext/repl/app.py) reach into the static method
# ``_SlashOnlyCompleter._current_slash_token(...)`` to detect slash
# tokens for their own key bindings. Keep the public shape stable by
# aliasing to the shared module function.
_SlashOnlyCompleter._current_slash_token = staticmethod(current_slash_token)


class _MessageHistoryCompleter(Completer):
    """Trigger autocompletion from previous user messages in the session.

    This completer provides "smart" completion based on the conversation
    history: when the user starts typing a word that matches the beginning
    of a previous user message, it surfaces that full message as a
    completion candidate.

    Rules:

    * Only triggers when the cursor is at position 0 (start of buffer)
      OR when the cursor sits on a word that follows whitespace
      (i.e. mid-input completion on a whitespace-delimited token).
    * Collects user message text from ``history_messages`` (a list of
      strings supplied by the caller via a property accessor).
    * Ranks: exact match of entire message first, then longest-prefix
      match, then by recency (most recent first).
    * Yields at most 5 candidates to keep the popup uncluttered.
    """

    def __init__(self, history_provider):
        """Initialize with a callable that returns a list of user message
        strings from previous turns.

        Args:
            history_provider: A callable returning ``list[str]`` of
                previous user messages. May raise; failures are
                silently ignored so completions never crash the REPL.
        """
        self._history_provider = history_provider

    def get_completions(self, document, complete_event):
        text = document.text_before_cursor
        cursor_pos = document.cursor_position

        # Only complete when cursor is at the very start, OR when
        # the cursor sits inside a word that follows whitespace.
        # This avoids interfering with normal typing mid-sentence.
        current_word = document.get_word_under_cursor()
        if not current_word:
            return

        # Check: is the word at position 0 (start of buffer)?
        if cursor_pos != len(current_word):
            # Is there whitespace (or nothing) before this word?
            if cursor_pos < len(text):
                prefix = text[: cursor_pos - len(current_word)]
                if prefix and not prefix[-1].isspace():
                    return  # mid-word completion — don't trigger
                word_start = cursor_pos - len(current_word)
                if word_start > 0 and not text[word_start - 1].isspace():
                    return
            else:
                return

        partial = current_word

        try:
            history = self._history_provider() or []
        except Exception:
            return

        # Rank matches via the shared helper. The REPL surfaces at most
        # 5 suggestions, mirrors the prior behaviour.
        start_position = -len(current_word)
        for full_msg in rank_message_history(history, partial.lower(), limit=5):
            yield Completion(
                text=full_msg,
                start_position=start_position,
                display=full_msg[:80] + ("..." if len(full_msg) > 80 else ""),
                display_meta="history",
            )


try:
    from rich.cells import cell_len
    from rich.console import Console, Group
    from rich.align import Align
    from rich.markup import escape
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text
    from rich.markdown import Markdown
    from rich.columns import Columns
except ModuleNotFoundError:  # pragma: no cover

    class Console:  # type: ignore
        def print(self, *args, **kwargs):
            return None

    def cell_len(s):  # type: ignore
        return len(s)

    Group = None  # type: ignore
    Align = None  # type: ignore
    Panel = None  # type: ignore
    Table = None  # type: ignore
    Text = None  # type: ignore
    Columns = None  # type: ignore

    class Markdown:  # type: ignore
        def __init__(self, text: str):
            self.text = text


logger = logging.getLogger(__name__)


def _format_goal_footer_elapsed(seconds: int) -> str:
    """Format the compact elapsed value used by Claude Code's goal footer."""

    seconds = max(int(seconds), 0)
    if seconds < 60:
        return f"{seconds}s"
    minutes = seconds // 60
    if minutes < 60:
        return f"{minutes}m"
    hours, remaining_minutes = divmod(minutes, 60)
    return f"{hours}h" if remaining_minutes == 0 else f"{hours}h {remaining_minutes}m"


# Heavy runtime deps are loaded lazily via ``_load_heavy_runtime()`` so
# ``from src.repl import ClawcodexREPL`` stays within the Stage-6 perf
# budget. Full stack (Session, providers, tools, commands) loads on first
# REPL instantiation, before the interactive loop starts.
_heavy_runtime_loaded = False
_cron_runtime_loaded = False
_HAS_CRON = False
attach_cron_runtime = None  # type: ignore[assignment,misc]
replace_cron_tools = None  # type: ignore[assignment,misc]
claim_cron_run = None  # type: ignore[assignment,misc]
finalize_cron_run = None  # type: ignore[assignment,misc]


# Lazy runtime placeholders — ``get_provider_config`` / ``get_provider_class``
# / ``build_provider_from_config`` / ``Session`` are referenced at module
# load time (Stage-6 perf budget) before ``_load_heavy_runtime`` runs. The
# decorator marks them so the heavy loader can detect and replace them
# with the real imports from ``src.*``. ``_LazySession`` keeps the same
# shape (``create`` / ``resume`` / ``load``) so existing call sites like
# ``Session.create(...)`` keep working without a second import.
def _lazy_runtime_placeholder(func: Callable[..., Any]) -> Callable[..., Any]:
    setattr(func, "_clawcodex_lazy_runtime_placeholder", True)
    return func


def _is_lazy_runtime_placeholder(name: str) -> bool:
    return bool(
        getattr(
            globals().get(name),
            "_clawcodex_lazy_runtime_placeholder",
            False,
        )
    )


@_lazy_runtime_placeholder
def get_provider_config(provider: str) -> dict[str, Any]:
    from src.config import get_provider_config as _get_provider_config

    return _get_provider_config(provider)


@_lazy_runtime_placeholder
def get_provider_class(provider_name: str) -> Any:
    from src.providers import get_provider_class as _get_provider_class

    return _get_provider_class(provider_name)


@_lazy_runtime_placeholder
def build_provider_from_config(provider_name: str, model: str | None = None) -> Any:
    from src.providers.runtime import build_provider_from_config as _build_provider

    return _build_provider(provider_name, model)


class _LazySession:
    _clawcodex_lazy_runtime_placeholder = True

    @staticmethod
    def create(*args: Any, **kwargs: Any) -> Any:
        from src.agent import Session as _Session

        return _Session.create(*args, **kwargs)

    @staticmethod
    def resume(*args: Any, **kwargs: Any) -> Any:
        patched_load = getattr(_LazySession, "load", None)
        if type(patched_load).__module__ == "unittest.mock":
            return patched_load(*args, **kwargs)
        from src.agent import Session as _Session

        return _Session.resume(*args, **kwargs)

    @staticmethod
    def load(*args: Any, **kwargs: Any) -> Any:
        from src.agent import Session as _Session

        return _Session.load(*args, **kwargs)


Session = _LazySession


def _session_id_from_session(session: Any) -> str | None:
    session_id = getattr(session, "session_id", None)
    if isinstance(session_id, str) and session_id.strip():
        return session_id
    return None


def _load_cron_runtime() -> None:
    """Import cron helpers without pulling the full REPL runtime stack."""
    global _cron_runtime_loaded, _HAS_CRON, attach_cron_runtime
    global replace_cron_tools, claim_cron_run, finalize_cron_run

    if _cron_runtime_loaded:
        return

    try:
        from clawcodex_ext.cron_system.runtime import attach_cron_runtime, replace_cron_tools
        from clawcodex_ext.cron_system.runs import claim_cron_run, finalize_cron_run

        _HAS_CRON = True
    except ImportError:
        _HAS_CRON = False
        attach_cron_runtime = None  # type: ignore[assignment]
        replace_cron_tools = None  # type: ignore[assignment]
        claim_cron_run = None  # type: ignore[assignment]
        finalize_cron_run = None  # type: ignore[assignment]

    _cron_runtime_loaded = True


def _load_heavy_runtime() -> None:
    """Import agent/provider/tool/command deps on first REPL use."""
    global _heavy_runtime_loaded
    global Session, get_provider_config, resolve_output_style
    global build_provider_from_config, get_provider_class, tool_to_api_schema
    global ToolContext, build_default_registry, ToolCall
    global ToolEvent, summarize_tool_result, summarize_tool_use
    global StreamEvent
    global NO_CONTENT_MESSAGE, AssistantMessage, SystemMessage, UserMessage
    global TextBlock, ToolUseBlock, ToolResultBlock, AbortController
    global CostTracker, HistoryLog, AgentMentionCompleter, AtFileCompleter
    global LiveStatus, _HAS_CRON, attach_cron_runtime, replace_cron_tools
    global claim_cron_run, finalize_cron_run
    global format_advisor_status, permission_mode_short_title
    global compute_session_cost, format_cost_usd
    # Note: QueryEngine / QueryEngineConfig are NOT imported here. They are
    # only needed when actually driving the query loop (i.e. after the user
    # submits a non-slash prompt), so we import them locally inside
    # ``ClawcodexREPL.chat`` — pulling in ~1.1s of query engine startup
    # before first input would dominate REPL cold start.
    # Note: command_system symbols (CommandRegistry, register_builtin_commands,
    # etc.) are also NOT imported here. ``_init_command_system`` does its own
    # local imports and is itself lazy — see ``_ensure_command_system`` which
    # fires on first slash command. Pulling command_system in
    # ``_load_heavy_runtime`` adds ~0.8s of import cost that 90% of REPL
    # sessions will never pay for.

    if _heavy_runtime_loaded:
        return

    from src.agent import Session as _Session
    from src.config import get_provider_config as _get_provider_config
    from src.outputStyles import resolve_output_style
    from src.providers.runtime import build_provider_from_config as _build_provider_from_config

    # Note: AnthropicProvider / MinimaxProvider / ChatMessage are NOT imported
    # here. Callers that need them (``_provider_uses_system_kwarg``,
    # ``clawcodex_ext.query.query``, ``clawcodex_ext.utils.advisor``) do their
    # own internal imports, so pulling them into the REPL's heavy-runtime
    # bootstrap would load provider modules users may never touch.
    from src.providers import get_provider_class as _get_provider_class
    from clawcodex_ext.services.api.claude import tool_to_api_schema
    from src.tool_system.context import ToolContext
    from src.tool_system.defaults import build_default_registry
    from clawcodex_ext.tool_system.protocol import ToolCall
    from src.tool_system.renderers import ToolEvent, summarize_tool_result, summarize_tool_use
    from src.query.query import StreamEvent
    from clawcodex_ext.types.messages import (
        NO_CONTENT_MESSAGE,
        AssistantMessage,
        SystemMessage,
        UserMessage,
    )
    from clawcodex_ext.types.content_blocks import TextBlock, ToolUseBlock, ToolResultBlock
    from src.utils.abort_controller import AbortController
    from src.cost_tracker import CostTracker
    from src.history import HistoryLog
    from src.repl.agent_mention_completer import AgentMentionCompleter
    from src.repl.at_file_completer import AtFileCompleter
    from clawcodex_ext.repl.live_status import LiveStatus

    # Hot-path footer helpers — ``_bottom_toolbar`` runs on every prompt
    # redraw (per keystroke), so these are resolved once here instead of
    # being re-imported on each call.
    from src.utils.advisor import format_advisor_status
    from clawcodex_ext.permissions import permission_mode_short_title
    from clawcodex_ext.services.pricing import compute_session_cost, format_cost_usd

    if _is_lazy_runtime_placeholder("get_provider_config"):
        get_provider_config = _get_provider_config
    if _is_lazy_runtime_placeholder("get_provider_class"):
        get_provider_class = _get_provider_class
    if _is_lazy_runtime_placeholder("build_provider_from_config"):
        build_provider_from_config = _build_provider_from_config
    if not _is_lazy_runtime_placeholder("Session"):
        Session = _Session

    _load_cron_runtime()

    _heavy_runtime_loaded = True


_CRON_WAKE = object()

_CRON_PROMPT_PRELUDE = "This prompt was generated automatically from a scheduled task."


def _wrap_cron_prompt(prompt: str, task_id: str = "", run_id: str = "") -> str:
    """Wrap a cron prompt with context so the LLM knows it's automated.

    F-22-G-2: signature now matches the
    ``Callable[[prompt, task_id, run_id], str]`` shape expected by
    :class:`CronDispatchBridge`. ``run_id`` is currently unused but
    is reserved for future runs-by-id display.
    """
    _ = run_id  # reserved for future display
    now = datetime.now()
    time_str = now.strftime("%b %d %-I:%M%p").lower()
    header = f"✻ Running scheduled task ({time_str})"
    if task_id:
        header += f" · {task_id}"
    return f"{header}\n\n{_CRON_PROMPT_PRELUDE}\n\n---\n\n{prompt}"


try:
    from prompt_toolkit.patch_stdout import patch_stdout as _pt_patch_stdout
except ModuleNotFoundError:  # pragma: no cover - prompt_toolkit guarded above
    from contextlib import nullcontext as _pt_patch_stdout  # type: ignore


def _format_edit_summary_text(adds: int, removes: int) -> str:
    """Format an "Added X lines, removed Y lines" summary.

    Mirrors the pluralization in the TS reference component
    (``FileEditToolUpdatedMessage.tsx``) — sentence-cased standalone
    clauses, lowercase ``removed`` after a comma.
    """

    if adds <= 0 and removes <= 0:
        return ""
    parts: list[str] = []
    if adds > 0:
        parts.append(f"Added {adds} {'line' if adds == 1 else 'lines'}")
    if removes > 0:
        verb = "Removed" if adds == 0 else "removed"
        parts.append(f"{verb} {removes} {'line' if removes == 1 else 'lines'}")
    return ", ".join(parts)


# Tool names whose consecutive calls should be coalesced into a single
# ``TaskListV2`` snapshot in the transcript. See
# ``typescript/src/components/TaskListV2.tsx`` for the reference UI.
_TASK_WIDGET_TOOL_NAMES: set[str] = {
    "TaskCreate",
    "TaskUpdate",
    "TaskList",
    "TaskGet",
    "TodoWrite",
}


def _ghost_hint_for(key: str, *, has_tab_alias: bool = True) -> str:
    """Build the trailing ghost-text hint for the configured accept key.

    When *has_tab_alias* is true and *key* is not already tab, the hint
    also mentions ``TAB`` as a context-aware secondary key. Mirrors the
    upstream ``useTypeahead`` behaviour where Tab is bound to
    ``autocomplete:accept`` only while a ghost-text suggestion is shown.
    """
    from clawcodex_ext.utils.key_format import display_key, to_prompt_toolkit_key

    base = display_key(key)
    if has_tab_alias and to_prompt_toolkit_key(key) != "tab":
        base = f"{base} or {display_key('tab')}"
    return f" ({base} to accept)"


# Module-level state tracking ghost-text suggestion visibility.
#
# prompt_toolkit's ``Condition`` filter accepts a no-arg callable — the
# filter cannot see the current buffer. We instead observe the
# ``_HintedAutoSuggest.get_suggestion`` calls (which the framework
# invokes on every keystroke) and stash a tuple of
# ``(suggestion_text, complete_state_active)`` for the registered
# ``tab`` binding's filter closure to read.
#
# NOTE: a single shared state is fine because only one REPL/PromptSession
# is typically mounted at a time per process. Multi-session REPLs (none
# exist upstream) would need to thread per-buffer state through the
# filter — out of scope for plan 3.
_ghost_state: dict[str, object] = {"suggestion": None, "complete_active": False}


class _HintedAutoSuggest(AutoSuggestFromHistory):
    """Append ``(CTRL + e to accept)`` to ghost-text suggestions.

    The accept key is parameterised so users can move the binding to
    ``tab`` or another key via ``settings.accept_suggestion_key`` while
    keeping the displayed hint in sync.
    """

    def __init__(self, accept_key: str = "c-e", *, has_tab_alias: bool = True) -> None:
        super().__init__()
        self._accept_key = accept_key
        self._hint = _ghost_hint_for(accept_key, has_tab_alias=has_tab_alias)

    def get_suggestion(self, buffer, document):
        suggestion = super().get_suggestion(buffer, document)
        # Refresh the module-level visibility snapshot so the Tab
        # binding's filter can decide whether to fire.
        _ghost_state["suggestion"] = suggestion.text if suggestion else None
        _ghost_state["complete_active"] = bool(getattr(buffer, "complete_state", None))
        if suggestion and suggestion.text:
            return Suggestion(suggestion.text + self._hint)
        return suggestion


def _patch_accept_suggestion_bindings(bindings, accept_key: str = "c-e", *, has_tab_alias: bool = True):
    """Override the accept key so it strips the hint before inserting.

    *accept_key* is the prompt_toolkit spelling (e.g. ``"c-e"``, ``"tab"``,
    ``"c-j"``). The hint stripped from the inserted text always matches
    the one rendered by :class:`_HintedAutoSuggest` for the same key.

    When *has_tab_alias* is true, a context-aware ``tab`` binding is
    also registered: it fires only when the ghost-text suggestion is
    visible AND the completion menu is closed, so it never competes
    with the slash/``@``/history completion navigation Tab. When the
    filter is false, prompt_toolkit's default Tab handler runs and
    cycles through the active completion menu.
    """

    from prompt_toolkit.filters import Condition

    from clawcodex_ext.utils.key_format import to_prompt_toolkit_key

    pt_key = to_prompt_toolkit_key(accept_key)
    hint = _ghost_hint_for(accept_key, has_tab_alias=has_tab_alias)

    def _accept_ghost(buf):
        suggestion = buf.suggestion
        if suggestion and suggestion.text.endswith(hint):
            buf.insert_text(suggestion.text[: -len(hint)])
        elif suggestion:
            buf.insert_text(suggestion.text)

    @bindings.add(pt_key)
    def _accept(event):
        _accept_ghost(event.current_buffer)

    if has_tab_alias and pt_key != "tab":
        # Context-aware Tab: only fire when ghost is showing and no
        # completion popup is active. The filter is a no-arg closure
        # that reads the module-level snapshot kept fresh by
        # ``_HintedAutoSuggest.get_suggestion``. When the filter
        # returns False, prompt_toolkit's default Tab handler runs
        # (cycle completion menu / insert a tab).
        def _tab_filter() -> bool:
            return _ghost_state.get("suggestion") is not None and not _ghost_state.get("complete_active", False)

        @bindings.add("tab", filter=Condition(_tab_filter))
        def _tab_accept(event):
            _accept_ghost(event.current_buffer)


async def _drain_history(history) -> None:
    """Prime a ``prompt_toolkit`` ``History`` by driving ``History.load()``.

    ``History.get_strings()`` only reads the in-memory ``_loaded_strings``
    cache, which is populated lazily by the async ``History.load()``
    generator. ``PromptSession`` does not await it before the first
    prompt, so on the first keystroke ``AutoSuggestFromHistory`` always
    sees an empty list and never produces a ghost-text suggestion.
    Exhausting the generator once at REPL construction time forces the
    cache to be populated so the very first suggestion lookup works.
    """
    async for _ in history.load():
        pass


class ClawcodexREPL:
    """Interactive REPL for Claw Codex."""

    def __init__(
        self,
        provider_name: str = "glm",
        stream: bool = False,
        *,
        permission_mode: str = "default",
        is_bypass_permissions_mode_available: bool = False,
        **kwargs: Any,
    ):
        _load_heavy_runtime()

        # ``is_interactive`` is set during bootstrap phase 2 by
        # ``src.init.run_pre_action`` (called from ``cli.main``) before
        # the REPL constructor runs. Previously we set it here too,
        # but that was the M7.1 gap closed in plan phase 1 of
        # ch02-bootstrap. The REPL can rely on
        # ``get_is_interactive()`` already being ``True`` by the time
        # this constructor runs.

        # Stash the resolved permission state so ``ToolContext`` honors
        # ``--dangerously-skip-permissions`` / ``--permission-mode`` flags
        # passed at startup. See ``src/cli.py:_resolve_permission_state``.
        self._permission_mode = permission_mode
        self._is_bypass_permissions_mode_available = bool(is_bypass_permissions_mode_available)

        from clawcodex_ext.repl.color_scheme import (
            build_rich_theme,
            build_ptk_style,
            DARK as _REPL_DARK,
        )
        from rich.theme import Theme as _RichTheme

        self._repl_ptk_style = build_ptk_style(_REPL_DARK)
        self._repl_palette = _REPL_DARK
        self.console = Console(theme=_RichTheme(build_rich_theme(_REPL_DARK)), highlight=False)
        self.provider_name = provider_name
        self.stream = stream
        self._thinking_visible = False

        # Load configuration
        config = get_provider_config(provider_name)
        if not config.get("api_key"):
            self.console.print("[error]Error: API key not configured.[/error]")
            self.console.print("Run [bold]clawcodex login[/bold] to configure.")
            sys.exit(1)

        # Initialize provider
        provider_class = get_provider_class(provider_name)
        self.provider = provider_class(
            api_key=config["api_key"],
            base_url=config.get("base_url"),
            model=config.get("default_model"),
        )

        # Create session
        self.session = Session.create(provider_name, self.provider.model)
        session_id = _session_id_from_session(self.session)

        # Late-binding closure: ``tool_context`` is built below, but the
        # Agent tool's prompt builder won't read this until much later,
        # so reading ``self.tool_context.mcp_clients`` lazily is safe.
        def _get_mcp_servers_for_prompt() -> list[str]:
            ctx = getattr(self, "tool_context", None)
            if ctx is None:
                return []
            clients = getattr(ctx, "mcp_clients", None) or {}
            return list(clients.keys())

        self.tool_registry = build_default_registry(
            provider=self.provider,
            get_available_mcp_servers=_get_mcp_servers_for_prompt,
            defer_extended_tools=True,
        )
        if _HAS_CRON:
            replace_cron_tools(self.tool_registry)
        self._engine_messages: list[Any] = []
        from src.permissions.types import ToolPermissionContext

        self.tool_context = ToolContext(
            workspace_root=Path.cwd(),
            permission_context=ToolPermissionContext(
                mode=self._permission_mode,  # type: ignore[arg-type]
                is_bypass_permissions_mode_available=(self._is_bypass_permissions_mode_available),
            ),
            session_id=session_id,
        )
        if _HAS_CRON:
            attach_cron_runtime(
                self.tool_context,
                autostart=True,
                is_loading=lambda: self._active_live_status is not None,
            )
        self._cron_active_tasks: dict[str, str] = {}
        from clawcodex_ext.runtime.tool_context_binding import bind_tool_context_runtime

        bind_tool_context_runtime(
            self.tool_context,
            tool_registry=self.tool_registry,
            session=self.session,
            provider=self.provider,
        )
        self.tool_context.ask_user = self._ask_user_questions
        # Permission handler with status control for proper input handling
        self._current_status = None
        if self._permission_mode == "bypassPermissions":
            # The bypass mode short-circuits the registry's permission check
            # before the handler is ever consulted, but a few tools call the
            # handler directly (e.g. the doc-write gate). Auto-allow there
            # too so the user's explicit opt-in is honored end-to-end.
            self.tool_context.allow_docs = True
            self.tool_context.permission_handler = lambda _tn, _msg, _sug: (True, False)
        else:
            self.tool_context.permission_handler = self._handle_permission_ask_request

        # Runtime permission controller — the single chokepoint for
        # Shift+Tab cycles and ``/permissions`` picks. The controller
        # owns the threading.Lock that serializes the multi-field swap
        # (``permission_context`` + ``permission_handler`` +
        # ``allow_docs``) so the agent worker thread never sees a torn
        # write. ``default_handler`` is the snapshotted non-bypass
        # handler restored on every cycle out of ``bypassPermissions``.
        # ``app_state_store`` is None on the REPL — there's no reactive
        # ``AppState`` store wired here today, so the controller skips
        # the AppState write (the CCR/SDK listener chain). The TUI path
        # wires the store at controller construction time.
        from clawcodex_ext.permissions.runtime import (
            RuntimePermissionController,
        )

        self._runtime_permission_controller = RuntimePermissionController(
            tool_context_factory=lambda: self.tool_context,
            default_handler=self._handle_permission_ask_request,
            app_state_store=None,
            notify=self._notify_permission_mode_change,
        )

        # Persistent bottom-toolbar accumulators. Mirrors the TS Ink
        # status line that always shows model · provider · cwd · turn /
        # token totals.
        self._stats_turns: int = 0
        self._stats_input_tokens: int = 0
        self._stats_output_tokens: int = 0
        self._direct_abort_controller: AbortController | None = None
        self._im_active_cancel: Callable[[], None] | None = None
        # IM-driven permission wait state. Populated only while
        # ``_handle_permission_request`` is blocked in
        # ``_wait_im_permission_choice`` waiting for a WeChat reply. The
        # ``permission_probe`` on ReplGatewayClient.deliver() resolves it.
        # Guarded by ``_im_permission_lock`` (probe runs on the IPC reader
        # thread, the wait runs on the main thread).
        self._im_permission_lock = threading.Lock()
        self._im_permission_wait: dict | None = None

        # Messages the user typed into LiveStatus while the agent was
        # working. The main run() loop drains this before falling back to
        # ``prompt_session.prompt()`` so queued prompts are sent back-to-back
        # without the user having to retype them — matches the TS Ink
        # reference's "type while it's still thinking" affordance.
        # ``deque(maxlen=100)`` silently drops the oldest entry once full
        # so a long-running session can't accumulate an unbounded queue
        # under the per-turn lock. See ``clear_pending_turn_buffers`` for
        # the turn-end reset that runs even on small queues.
        self._queued_prompts: deque[str] = deque(maxlen=100)
        self._cron_queued_prompts: deque[str] = deque(maxlen=100)
        self._queued_prompts_lock = threading.Lock()
        self._background_outputs: list[str] = []
        self._background_outputs_lock = threading.Lock()
        # Permission dialogs can be requested from different worker paths
        # (e.g. subagents/tools). Serialize interactive prompts so we never
        # mount competing prompt_toolkit applications at once.
        self._permission_prompt_lock = threading.Lock()
        # Reserved for compatibility with older extensions that inspect the
        # field. Plain "allow this action" decisions are intentionally not
        # cached; each permission prompt must represent the current action.
        self._permission_decision_cache: dict[str, bool] = {}

        # The currently mounted ``LiveStatus`` (if any). ``_safe_input``
        # pauses it before reading a synchronous answer (e.g. permission
        # prompts) so two prompt_toolkit Applications don't fight over
        # the TTY and tear the spinner row.
        self._active_live_status: LiveStatus | None = None

        # Bounded stash of (label, full_content) pairs for blocks rendered
        # truncated in the transcript (currently only Write previews).
        # ``ctrl+o`` re-prints the most recent entry as a fresh block
        # below — see ``_do_expand_last``. Bounded so the deque doesn't
        # grow unboundedly during a long session.
        self._expandable_blocks: deque[tuple[str, str]] = deque(maxlen=20)

        # Streaming buffer of ``Thinking…`` text chunks. ``_expand_thinking``
        # concatenates them into the spinner label. ``deque(maxlen=1000)``
        # silently drops the oldest chunk once full so a runaway / backgrounded
        # session can't grow this buffer past ~1k strings.
        # ``clear_pending_turn_buffers`` additionally empties it at every turn
        # boundary for tight memory budgets (the WSL2 3.8 GB OOM repro).
        self._thinking_chunks: deque[str] = deque(maxlen=1000)

        # Original built-in commands - define this FIRST!
        self._original_built_ins = [
            "/",
            "/help",
            "/exit",
            "/quit",
            "/q",
            "/clear",
            "/save",
            "/load",
            "/stream",
            "/render-last",
            "/tools",
            "/tool",
            "/skills",
            "/init",
            "/tui",
            # TUI-only commands — listed here so the "unknown command → palette"
            # intercept (line 3056) does NOT swallow them before the TUI-only
            # handler (line 3089) can print the proper message.
            "/diff",
            "/mcp",
            "/tasks",
            "/rewind",
            "/repl",
            "/effort",
            "/history",
            "/idle",
            "/theme",
            "/permission",
        ]
        self._built_in_commands = list(self._original_built_ins)

        # Command system is built lazily on first slash command via
        # ``_ensure_command_system``. ``_load_heavy_runtime`` does NOT import
        # command_system symbols — pulling ``src.command_system`` here would
        # add ~0.8s of import cost that most sessions never pay back. The
        # cost moves to the first ``/`` keystroke path (``handle_command``)
        # which fires well after startup completes.

        # Prompt toolkit with tab completion
        from clawcodex_ext.debug.agent_debug import resolve_repl_history_file

        history_file = resolve_repl_history_file()
        history_file.parent.mkdir(parents=True, exist_ok=True)

        # ``_SlashOnlyCompleter`` handles ``/`` slash commands; the
        # ``AtFileCompleter`` adds ``@``-mention file completion that
        # mirrors the TS Ink reference (see
        # ``typescript/src/hooks/fileSuggestions.ts``). Merging keeps
        # both behaviors active simultaneously without either side
        # interfering with the other's trigger.
        from prompt_toolkit.completion import merge_completers

        # TTL cache for the slash-command suggestion list. ``build_command
        # _suggestions`` walks the user/project/managed skills dirs on every
        # call (~1.1s cold), and prompt_toolkit asks the completer on every
        # keystroke while typing — so without a cache the first ``/`` press
        # blocks the input row for over a second. Refreshed lazily; the
        # background warm below populates the cache before the user can
        # plausibly press ``/``. Invalidated on a 30 s TTL so newly-added
        # skills surface within a turn or two.
        self._slash_suggestions_cache: list[Any] | None = None
        self._slash_suggestions_cache_at: float = 0.0

        self._slash_completer = _SlashOnlyCompleter(
            self._get_slash_command_words,
            suggestions_provider=self._get_slash_command_suggestions,
        )
        self._at_completer = AtFileCompleter(cwd=str(self.tool_context.workspace_root))
        self._agent_completer = AgentMentionCompleter(self._available_agents)
        self._message_history_completer = _MessageHistoryCompleter(self._get_user_message_history)
        self.completer = merge_completers(
            [
                self._slash_completer,
                self._at_completer,
                self._agent_completer,
                self._message_history_completer,
            ]
        )

        # Warm the slash-command suggestion cache in the background so the
        # very first ``/`` keystroke doesn't pay the cold import + disk-walk
        # cost. Daemon thread so it can't block REPL shutdown.
        threading.Thread(
            target=self._warm_slash_suggestions_cache,
            name="slash-suggestions-warm",
            daemon=True,
        ).start()

        # Key bindings.
        #
        # Multiline-entry contract (mirrors
        # ``typescript/src/hooks/useTextInput.ts#handleEnter``):
        #
        #   * plain Enter          -> submit
        #   * Shift+Enter          -> insert newline  (terminals with
        #                             Kitty-protocol CSI 13;2u, iTerm2
        #                             or VSCode configured via
        #                             /terminal-setup)
        #   * Meta/Alt/Option+Enter -> insert newline  (universally
        #                             supported: the terminal sends
        #                             "\x1b\r", which prompt_toolkit
        #                             parses as Escape+ControlM)
        #   * ``\`` + Enter        -> insert newline  (portable fallback
        #                             that works on ANY terminal — the
        #                             trailing backslash is removed and
        #                             replaced by a real newline)
        #
        # The buffer is always created in ``multiline=True`` mode so that
        # real newlines can live in it; we override the default Enter
        # behavior below so Enter still submits (prompt_toolkit's default
        # in multiline mode is "insert newline").
        self.bindings = KeyBindings()
        if hasattr(self.bindings, "add"):

            @self.bindings.add("/")  # type: ignore[attr-defined]
            def _show_slash_completions(event):  # type: ignore[no-untyped-def]
                # Always insert the literal ``/`` — earlier versions
                # short-circuited when the buffer was non-empty and
                # silently swallowed the keystroke, so paths like
                # ``src/repl/core.py`` were untypable. Only auto-pop
                # the slash-command menu when ``/`` is the first
                # character of the buffer (mirrors the TS reference's
                # ``commandSuggestions`` trigger rule).
                buf = event.current_buffer
                was_empty = buf.text == ""
                buf.insert_text("/")
                if was_empty:
                    buf.start_completion(select_first=False)

            def _refresh_slash_menu_after_deletion(event, deleter):  # type: ignore[no-untyped-def]
                # prompt_toolkit's ``complete_while_typing`` only fires on
                # ``insert_text`` (buffer.py:1248-1252) — text deletions
                # close the completion popup but never reopen it. That's
                # what makes ``/exit`` → backspace to ``/ex`` go silent:
                # the popup closes when the menu's selected completion no
                # longer matches, and nothing re-triggers it. So we
                # explicitly restart completion after the deletion when
                # the cursor is still on a slash token.
                buf = event.current_buffer
                deleter(buf)
                if not (buf.completer and buf.complete_while_typing()):
                    return
                token, _ = _SlashOnlyCompleter._current_slash_token(buf.document.text_before_cursor)
                if token is not None:
                    buf.start_completion(select_first=False)

            @self.bindings.add("backspace")  # type: ignore[attr-defined]
            def _backspace_refreshes_slash_menu(event):  # type: ignore[no-untyped-def]
                _refresh_slash_menu_after_deletion(event, lambda b: b.delete_before_cursor(count=1))

            @self.bindings.add("delete")  # type: ignore[attr-defined]
            def _delete_refreshes_slash_menu(event):  # type: ignore[no-untyped-def]
                _refresh_slash_menu_after_deletion(event, lambda b: b.delete(count=1))

            @self.bindings.add("c-m")  # type: ignore[attr-defined]
            def _enter_submits_or_backslash_newline(event):  # type: ignore[no-untyped-def]
                """Enter: submit, or convert trailing ``\\`` into a newline.

                Exactly mirrors the TypeScript ``handleEnter`` logic. When a
                completion popup is open we accept the current selection and
                close the popup (prompt_toolkit's default Enter behavior) so
                the slash-command menu still works as expected.
                """
                buf = event.current_buffer
                if buf.complete_state:
                    buf.complete_state = None
                    return
                text = buf.text
                pos = buf.cursor_position
                if pos > 0 and text[pos - 1] == "\\":
                    buf.delete_before_cursor(count=1)
                    buf.insert_text("\n")
                    return
                buf.validate_and_handle()

            @self.bindings.add("escape", "c-m")  # type: ignore[attr-defined]
            def _meta_or_shift_enter_inserts_newline(event):  # type: ignore[no-untyped-def]
                """Meta+Enter (and Kitty-protocol Shift+Enter): insert ``\\n``."""
                event.current_buffer.insert_text("\n")

            @self.bindings.add("c-o")  # type: ignore[attr-defined]
            def _expand_last(event):  # type: ignore[no-untyped-def]
                """Ctrl+O: re-print the most recent truncated block in
                full as a fresh block below the prompt. ``run_in_terminal``
                temporarily exits the prompt loop so the output doesn't
                fight the prompt's redraw.
                """

                try:
                    from prompt_toolkit.application import run_in_terminal

                    run_in_terminal(self._do_expand_last)
                except Exception:
                    # Fallback: print directly. Prompt may redraw oddly
                    # but at least the expansion lands in scrollback.
                    self._do_expand_last()

        # ``AutoSuggestFromHistory`` reads its suggestions via
        # ``history.get_strings()``, which on the base ``History`` class
        # is a thin wrapper around the in-memory ``_loaded_strings`` cache
        # (see ``prompt_toolkit/history.py:History.get_strings``). That
        # cache is only populated by the async ``History.load()`` generator
        # — there is no ``__iter__`` on the base class in 3.0.52.
        # ``PromptSession`` itself does not await ``history.load()`` before
        # the first prompt, so without an explicit priming step the
        # autosuggest always sees an empty list on the first keystroke
        # and never produces a ghost-text suggestion. Drain ``load()`` once
        # here so the very first ``he`` typed by the user already sees
        # the matching ``hello`` from disk.
        file_history = FileHistory(str(history_file))
        asyncio.run(_drain_history(file_history))
        # Expose the shared history so ``LiveStatus`` (background agent
        # input) can pass it to its ``Buffer`` and support up/down
        # history navigation identical to the foreground prompt.
        self._file_history = file_history

        # Read the configured accept key (default ``c-e``) and whether
        # Tab should be a context-aware alias, so the ghost-text hint
        # and binding registration agree.
        try:
            from src.settings.settings import get_settings as _get_settings

            _settings = _get_settings()
            _accept_key = getattr(_settings, "accept_suggestion_key", "c-e") or "c-e"
            _accept_tab_alias = bool(getattr(_settings, "accept_suggestion_tab_alias", True))
        except Exception:
            _accept_key = "c-e"
            _accept_tab_alias = True

        self.prompt_session = PromptSession(
            history=file_history,
            auto_suggest=_HintedAutoSuggest(
                accept_key=_accept_key,
                has_tab_alias=_accept_tab_alias,
            ),
            completer=self.completer,
            style=Style.from_dict(self._repl_ptk_style),
            key_bindings=self.bindings,
            complete_while_typing=True,
            multiline=True,
            prompt_continuation=self._prompt_continuation,
            bottom_toolbar=self._bottom_toolbar,
        )
        _patch_accept_suggestion_bindings(
            self.bindings,
            accept_key=_accept_key,
            has_tab_alias=_accept_tab_alias,
        )

    def _bottom_toolbar(self):
        """Single-line status footer for the input prompt.

        Mirrors the TS Ink reference's persistent status row at the
        bottom: provider, model, current working directory, and
        accumulated turn / token counts for the session. Kept terse so
        it doesn't compete with the input row for attention.
        """

        try:
            provider = getattr(self.provider, "provider_name", None) or self.provider_name or "?"
            model = getattr(self.provider, "model", "") or "?"
            cwd_full = str(self.tool_context.cwd or self.tool_context.workspace_root)
            cwd = self._shorten_path_text(cwd_full) or cwd_full
            # Optional advisor segment — appears between cwd and turns
            # when ``/advisor`` is set. Mode label (server/client/inactive)
            # reflects what the NEXT request will do given the current
            # provider + main model, so a stale config under an
            # unsupported provider shows "(inactive)" rather than lying.
            advisor_seg = format_advisor_status(self.provider, model)
            advisor_part = f" {advisor_seg} ·" if advisor_seg else ""
            # Advisor token counts — accumulated on the ToolContext
            # by ``src/tool_system/tools/advisor.py`` per consultation.
            # Surface them next to the worker's counts so the user can
            # see how much of the spend went to the reviewer model.
            # Hidden when zero so the toolbar stays compact for users
            # who haven't enabled the advisor.
            adv_in = int(getattr(self.tool_context, "advisor_input_tokens", 0) or 0)
            adv_out = int(getattr(self.tool_context, "advisor_output_tokens", 0) or 0)
            advisor_tokens = f" (advisor: {adv_in} in / {adv_out} out)" if (adv_in or adv_out) else ""
            # Model context window — show max context length for the
            # current model (e.g. "ctx: 200k"). Falls back silently
            # when model name is unknown or lookup fails.
            proactive_part = ""
            try:
                from clawcodex_ext.repl.proactive_integration import (
                    format_proactive_status,
                )

                proactive_status = format_proactive_status()
                if proactive_status:
                    proactive_part = f" 路 {proactive_status}"
            except Exception:  # nosec B110
                proactive_part = ""
            ctx_part = ""
            try:
                from clawcodex_ext.context_system.context_analyzer import (
                    get_context_window_for_model,
                )

                ctx_win = get_context_window_for_model(model)
                if ctx_win > 0:
                    ctx_part = f" · ctx: {ctx_win // 1000}k"
            except Exception:  # nosec B110
                ctx_part = ""
            # USD cost — directional estimate based on the upstream
            # model's published per-token price. Proxies (litellm,
            # openrouter, bedrock) may charge different rates; the
            # displayed number is the upstream-list cost, not the
            # exact invoice. Hidden when zero (no API turns yet this
            # session).
            try:
                from src.settings.settings import get_settings as _gs

                _settings = _gs()
                _advisor_model = (getattr(_settings, "advisor_model", "") or "").strip()
            except Exception:  # nosec B110
                _advisor_model = ""
            worker_cost, advisor_cost, total_cost = compute_session_cost(
                worker_model=model,
                worker_input_tokens=self._stats_input_tokens,
                worker_output_tokens=self._stats_output_tokens,
                advisor_model=_advisor_model,
                advisor_input_tokens=adv_in,
                advisor_output_tokens=adv_out,
            )
            # Space-separated label (matches TUI's "cost N" pattern;
            # avoids the REPL/TUI label-style split critic flagged).
            cost_part = f" · cost {format_cost_usd(total_cost)}" if total_cost > 0 else ""
            _in = self._stats_input_tokens
            _out = self._stats_output_tokens

            def _fmt(n: int) -> str:
                return f"{n / 1000:.1f}k" if n >= 1000 else str(n)

            goal_segment = self._goal_footer_status()
            goal_part = f" · {goal_segment}" if goal_segment else ""
            status_segments = advisor_part + proactive_part
            return status_segments + (
                f" {provider} · {model} · {cwd} · "
                f"mode: {permission_mode_short_title(self._permission_mode)} · "
                f"turns: {self._stats_turns} · "
                f"tokens: {_fmt(_in)} in / "
                f"{_fmt(_out)} out"
                f"{ctx_part}"
                f"{advisor_tokens}"
                f"{cost_part}"
                f"{goal_part}"
                f" "
            )
        except Exception:
            # Never let the toolbar break the input prompt. Runs on every
            # redraw (per-keystroke), so log at debug level to surface the
            # cause when troubleshooting without flooding normal sessions.
            logger.debug("bottom toolbar render failed", exc_info=True)
            return ""

    def _goal_footer_status(self) -> str | None:
        """Return Claude Code's active-goal footer segment for the REPL."""

        service = getattr(getattr(self, "tool_context", None), "goal_service", None)
        thread_id = getattr(getattr(self, "tool_context", None), "goal_thread_id", None)
        thread_id = thread_id or getattr(getattr(self, "tool_context", None), "session_id", None)
        if service is None or not thread_id:
            return None
        try:
            from clawcodex_ext.goal.model import ThreadGoalStatus

            goal = service.get_goal(str(thread_id))
            if goal is None or goal.status is not ThreadGoalStatus.ACTIVE:
                self._goal_footer_id = None
                self._goal_footer_started_at = None
                return None

            now = time.monotonic()
            if (
                getattr(self, "_goal_footer_id", None) != goal.goal_id
                or getattr(self, "_goal_footer_started_at", None) is None
            ):
                self._goal_footer_id = goal.goal_id
                self._goal_footer_started_at = now - max(int(goal.time_used_seconds), 0)
            elapsed = max(int(now - self._goal_footer_started_at), 0)
            return f"◎ /goal active ({_format_goal_footer_elapsed(elapsed)})"
        except Exception:
            logger.debug("goal footer render failed", exc_info=True)
            return None

    def _has_active_evaluator_goal_for_control_flow(self) -> bool:
        """Authoritatively gate paths that would bypass goal evaluation.

        Storage failures fail closed: the canonical query path can surface the
        underlying error, while direct streaming could silently skip a live
        evaluator goal and incorrectly return control to the user.
        """

        context = getattr(self, "tool_context", None)
        service = getattr(context, "goal_service", None)
        thread_id = getattr(context, "goal_thread_id", None) or getattr(context, "session_id", None)
        if service is None or not thread_id:
            return False
        try:
            from clawcodex_ext.goal.model import (
                GoalCompletionMode,
                ThreadGoalStatus,
            )

            goal = service.get_goal(str(thread_id))
            return bool(
                goal is not None
                and goal.status is ThreadGoalStatus.ACTIVE
                and goal.completion_mode is GoalCompletionMode.EVALUATOR
            )
        except Exception:
            logger.warning(
                "goal state lookup failed; disabling direct stream",
                exc_info=True,
            )
            return True

    def _echo_user_input(self, text: str) -> None:
        """Print a user message to the transcript (transparent background).

        Used for queued submissions (typed during agent work via
        :class:`LiveStatus`) and any other path that needs to surface a
        user-authored message into scrollback.

        Uses ``prompt_fg`` (terminal default foreground) for the ``❯``
        prefix so queued messages match the visual style of normal input
        typed directly in the prompt buffer.
        """

        try:
            color = self._repl_palette.prompt_fg
        except Exception:
            color = ""
        from rich.text import Text

        body = text.replace("\n", "\n  ")
        prefix = Text("❯ ", style=f"bold {color}")
        self.console.print(
            prefix + Text(body),
            markup=False,
            soft_wrap=True,
        )

    def _prompt_continuation(self, width, line_number, is_soft_wrap):
        """Continuation prompt for wrapped / multi-line input.

        Logical lines get ``"… "`` so it's obvious we're in an in-progress
        multi-line prompt; soft wraps get blank padding so long lines
        flow naturally. Width-padded to keep the text column aligned
        with the primary ``❯ `` prompt.
        """
        if is_soft_wrap:
            return " " * width
        marker = "… "
        if width <= len(marker):
            return marker[:width]
        return marker.rjust(width)

    def _run_arrow_menu(
        self,
        options: list[tuple[str, str]],
        *,
        title: str = "",
        allow_other: bool = False,
        multi_select: bool = False,
    ) -> int | list[int] | None:
        """Show an arrow-key navigable menu using prompt_toolkit key bindings.

        Args:
            options: List of (label, description) pairs for each option.
            title: Optional title shown above the menu.
            allow_other: If True, add an "Other" option at the end.
            multi_select: If True, allow multiple selections with Space toggle.

        Returns:
            Single-select: int (0-based index) or None for cancel.
            Multi-select: list[int] of 0-based indices, or None for cancel.
        """
        if not _HAS_PROMPT_TOOLKIT:
            return None

        total = len(options) + (1 if allow_other else 0)
        if total == 0:
            return None

        # Mutable state captured by closures
        cursor = [0]  # list for mutation in closure
        selected: set[int] | None = set() if multi_select else None

        from prompt_toolkit.layout import Layout
        from prompt_toolkit.layout.containers import Window
        from prompt_toolkit.layout.controls import FormattedTextControl

        # Build the display text
        def get_menu_fragments():
            fragments: list[tuple[str, str]] = []
            if title:
                fragments.append(("[bold]", f"\n{title}\n\n"))
            for i, (label, desc) in enumerate(options):
                is_cursor = i == cursor[0]
                is_sel = multi_select and i in (selected or set())
                prefix = "▸" if is_cursor else " "
                check = "✓" if is_sel else " "
                item_style = "class:arrow-cursor" if is_cursor else ""
                fragments.append((item_style, f"  {prefix} {check} {i + 1}. {label}"))
                if desc:
                    fragments.append(("class:dim", f"    {desc}"))
                fragments.append(("", "\n"))
            if allow_other:
                i = len(options)
                is_cursor = i == cursor[0]
                prefix = "▸" if is_cursor else " "
                item_style = "class:arrow-cursor" if is_cursor else ""
                fragments.append((item_style, f"  {prefix}   {i + 1}. Other"))
                fragments.append(("class:dim", "  (provide custom text)"))
                fragments.append(("", "\n"))
            if multi_select:
                hint = "  ↑↓ navigate · Space toggle · Enter confirm · 1-9 quick select · Esc cancel"
            else:
                hint = "  ↑↓ navigate · Enter select · 1-9 quick select · Esc cancel"
            fragments.append(("class:dim", f"\n{hint}"))
            return fragments

        kb = KeyBindings()

        @kb.add("up")
        def _move_up(event):
            cursor[0] = max(0, cursor[0] - 1)
            event.app.invalidate()

        @kb.add("down")
        def _move_down(event):
            cursor[0] = min(total - 1, cursor[0] + 1)
            event.app.invalidate()

        @kb.add("enter")
        def _handle_enter(event):
            if multi_select:
                sel_list = sorted(selected) if selected else [0]
                event.app.exit(result=sel_list)
            else:
                event.app.exit(result=cursor[0])

        @kb.add("space")
        def _handle_space(event):
            if multi_select:
                if cursor[0] < len(options):
                    s = selected
                    if s is not None:
                        if cursor[0] in s:
                            s.discard(cursor[0])
                        else:
                            s.add(cursor[0])
                        event.app.invalidate()
            else:
                event.app.exit(result=cursor[0])

        @kb.add("escape")
        def _handle_escape(event):
            event.app.exit(result=None)

        @kb.add("c-c")
        def _handle_ctrl_c(event):
            event.app.exit(result=None)

        # Number keys as fallback quick-select (1-9)
        for digit in range(1, min(10, total + 1)):

            @kb.add(str(digit))
            def _handle_digit(event, idx=digit):
                actual = idx - 1
                if multi_select:
                    if actual < len(options):
                        s = selected
                        if s is not None:
                            if actual in s:
                                s.discard(actual)
                            else:
                                s.add(actual)
                            event.app.invalidate()
                else:
                    event.app.exit(result=actual)

        from prompt_toolkit.application import Application

        pt_style = Style.from_dict(
            {
                "arrow-cursor": "bold",
                "dim": "fg:gray",
            }
        )

        app = Application(
            layout=Layout(
                Window(
                    FormattedTextControl(
                        get_menu_fragments,
                    )
                )
            ),
            key_bindings=kb,
            style=pt_style,
            full_screen=False,
            mouse_support=False,
        )

        live = self._active_live_status
        if live is not None:
            with live.paused():
                result = app.run()
        else:
            result = app.run()

        if result is None:
            return None
        if multi_select and selected:
            sel_list = sorted(selected)
            return sel_list if sel_list else [0]
        if multi_select and not selected:
            return [0]
        return result

    def _arrow_select(self, options, title="", allow_other=False, multi_select=False):
        """Public wrapper for :meth:`_run_arrow_menu` callable from outside the class.

        This exists so :class:`ReplUIHost` can receive a simple callable
        without coupling to the method signature of ``_run_arrow_menu``.
        """
        return self._run_arrow_menu(
            options,
            title=title,
            allow_other=allow_other,
            multi_select=multi_select,
        )

    def _ask_user_questions(self, questions: list[dict]) -> dict[str, str]:
        # Stop the Rich status spinner if running, so we can get clean input
        if self._current_status is not None:
            try:
                self._current_status.stop()
            except Exception:  # nosec B110
                pass

        answers: dict[str, str] = {}
        use_arrow = get_selection_mode() == "arrow"
        for q in questions:
            if isinstance(q, str):
                q = {"question": q}
            if not isinstance(q, dict):
                continue
            question_text = str(q.get("question", "")).strip()
            options = q.get("options") or []
            multi = bool(q.get("multiSelect", False))
            if not question_text or not isinstance(options, list) or len(options) < 2:
                continue

            # Build labels and option pairs from the question options
            labels: list[str] = []
            opt_pairs: list[tuple[str, str]] = []
            for opt in options:
                if isinstance(opt, str):
                    opt = {"label": opt, "description": ""}
                if not isinstance(opt, dict):
                    continue
                label = str(opt.get("label", "")).strip()
                desc = str(opt.get("description", "")).strip()
                labels.append(label)
                opt_pairs.append((label, desc))

            if use_arrow:
                result = self._run_arrow_menu(
                    opt_pairs,
                    title=question_text,
                    allow_other=True,
                    multi_select=multi,
                )
                other_idx = len(labels)

                if result is None:
                    selected_labels = [labels[0]]
                elif isinstance(result, list):
                    selected_labels = []
                    for idx in result:
                        if idx == other_idx:
                            free = self._safe_input("Other > ").strip()
                            if free:
                                selected_labels.append(free)
                        elif 0 <= idx < len(labels):
                            selected_labels.append(labels[idx])
                    if not selected_labels:
                        selected_labels = [labels[0]]
                else:
                    if result == other_idx:
                        free = self._safe_input("Other > ").strip()
                        selected_labels = [free] if free else [labels[0]]
                    elif 0 <= result < len(labels):
                        selected_labels = [labels[result]]
                    else:
                        selected_labels = [labels[0]]

                answers[question_text] = ", ".join(selected_labels) if multi else selected_labels[0]
            else:
                self.console.print(f"\n[bold]{question_text}[/bold]")
                for i, (label, desc) in enumerate(opt_pairs, start=1):
                    self.console.print(f"  {i}. {label}  [dim]{desc}[/dim]")
                other_idx = len(labels) + 1
                self.console.print(f"  {other_idx}. Other  [dim]Provide custom text[/dim]")

                prompt = "Select (comma-separated) > " if multi else "Select > "
                raw = self._safe_input(prompt).strip()
                if not raw:
                    choice_str = "1"
                else:
                    choice_str = raw

                selected: list[str] = []
                parts = [p.strip() for p in choice_str.split(",") if p.strip()]
                if not parts:
                    parts = ["1"]
                for part in parts:
                    try:
                        idx = int(part)
                    except ValueError:
                        idx = -1
                    if idx == other_idx:
                        free = self._safe_input("Other > ").strip()
                        if free:
                            selected.append(free)
                        continue
                    if 1 <= idx <= len(labels):
                        selected.append(labels[idx - 1])
                if not selected:
                    selected = [labels[0]]
                answers[question_text] = ", ".join(selected) if multi else selected[0]

        # Restart spinner after getting answers
        if self._current_status is not None:
            try:
                self._current_status.start()
            except Exception:  # nosec B110
                pass

        return answers

    def _handle_permission_request(
        self,
        tool_name: str,
        message: str,
        suggestion: str | None,
        tool_input: Any = None,
    ) -> tuple[bool, bool]:
        """Handle interactive permission requests from tools.

        Args:
            tool_name: Name of the tool requesting permission.
            message: Message explaining what permission is needed.
            suggestion: Optional suggestion for enabling the setting.
            tool_input: Optional tool input dict for per-tool preview rendering.

        Returns:
            Tuple of (allowed: bool, continue_without_caching: bool).
            continue_without_caching is always False since we don't cache in REPL.
        """
        with self._permission_prompt_lock:
            # Stop the Rich status spinner if running, so we can get clean input
            if self._current_status is not None:
                try:
                    self._current_status.stop()
                except Exception:  # nosec B110
                    pass

            self.console.print("")
            self.console.print("[bold][warning]⚠ Permission Required[/warning][/bold]")
            self.console.print(f"  {message}")
            # Render per-tool preview (Bash → command, Write → path+content, etc.)
            self._render_permission_preview(tool_name, tool_input)
            self.console.print("")

            # Determine if this is a setting that can be enabled
            can_enable_setting = False
            setting_to_enable: str | None = None

            msg_lower = message.lower()
            if "allow_docs" in msg_lower or "documentation files" in msg_lower:
                if not self.tool_context.allow_docs:
                    can_enable_setting = True
                    setting_to_enable = "allow_docs"

            # Build options
            options: list[tuple[str, str]] = [
                ("y", "Yes, allow this action"),
                ("n", "No, deny this action"),
            ]
            if can_enable_setting:
                options.insert(0, ("e", f"Enable {setting_to_enable} and allow"))

            im_reply = getattr(self, "_im_reply_controller", None)
            send_permission_prompt = getattr(im_reply, "send_permission_prompt", None)
            # IM-driven turns may have the permission decision come from
            # WeChat (a menu number/letter reply). Keyboard-driven turns
            # keep using the terminal as before. ``peek_reply_origin`` is
            # non-None only while an IM message is driving this turn.
            im_origin = None
            im_client = getattr(im_reply, "_client", None) if im_reply is not None else None
            peek_origin = getattr(im_client, "peek_reply_origin", None)
            if callable(peek_origin):
                try:
                    im_origin = peek_origin()
                except Exception:  # nosec B110
                    im_origin = None

            if im_origin and callable(send_permission_prompt):
                choice = (
                    self._wait_im_permission_choice(
                        message=message,
                        suggestion=suggestion,
                        options=options,
                        allow_choices={key for key, _desc in options if key != "n"},
                    )
                    .strip()
                    .lower()
                )
            else:
                if callable(send_permission_prompt):
                    try:
                        send_permission_prompt(
                            message=message,
                            suggestion=suggestion,
                            options=options,
                        )
                    except Exception:  # nosec B110
                        pass

                if get_selection_mode() == "arrow":
                    opt_pairs: list[tuple[str, str]] = []
                    for key, desc in options:
                        opt_pairs.append((f"[{key}] {desc}", ""))
                    result = self._run_arrow_menu(
                        opt_pairs,
                        title="Permission Required",
                        allow_other=False,
                        multi_select=False,
                    )
                    if result is None:
                        return False, False
                    idx = result if isinstance(result, int) else (result[0] if result else 0)
                    if can_enable_setting:
                        if idx == 0:
                            self._enable_permission_setting(setting_to_enable)
                            return True, False
                        elif idx == 1:
                            return True, False
                        else:
                            return False, False
                    else:
                        if idx == 0:
                            return True, False
                        else:
                            return False, False

                self.console.print("[bold]Options:[/bold]")
                for i, (key, desc) in enumerate(options, start=1):
                    self.console.print(f"  {i}. [{key}] {desc}")
                self.console.print("")

                choice = self._safe_input("Select option> ").strip().lower()

            if can_enable_setting:
                if choice in ("1", "e", "enable"):
                    self._enable_permission_setting(setting_to_enable)
                    return True, False
                elif choice in ("2", "y", "yes", ""):
                    return True, False
                elif choice in ("3", "n", "no"):
                    return False, False
            else:
                if choice in ("1", "y", "yes", ""):
                    return True, False
                elif choice in ("2", "n", "no"):
                    return False, False

            self.console.print("[dim]Invalid choice, defaulting to deny.[/dim]")
            return False, False

    def _render_permission_preview(
        self,
        tool_name: str | None,
        tool_input: Any,
    ) -> None:
        """Render a per-tool permission preview (Bash → command, Write → path+content, etc.).

        Reuses the TUI's :func:`preview_for_tool` renderers so both surfaces
        show the same detail.  Silently degrades when the preview module is
        unavailable (headless / minimal installs).
        """
        if not tool_input:
            return
        try:
            from clawcodex_ext.tui.screens.permission_modal import preview_for_tool
        except ImportError:
            return
        try:
            renderable = preview_for_tool(tool_name, tool_input)
        except Exception:  # nosec B110
            renderable = None
        if renderable is not None:
            self.console.print(renderable)

    def _handle_permission_ask_request(
        self,
        request: PermissionAskRequest,
    ) -> PermissionAskReply:
        """Handle the structured permission request used by the registry."""
        with self._permission_prompt_lock:
            if self._current_status is not None:
                try:
                    self._current_status.stop()
                except Exception:  # nosec B110
                    pass

            self.console.print("")
            self.console.print("[bold][warning]⚠ Permission Required[/warning][/bold]")
            self.console.print(f"  {request.message}")
            # Render per-tool preview (Bash → command, Write → path+content, etc.)
            self._render_permission_preview(request.tool_name, request.tool_input)
            self.console.print("")

            can_enable_setting = False
            setting_to_enable: str | None = None
            msg_lower = request.message.lower()
            if "allow_docs" in msg_lower or "documentation files" in msg_lower:
                if not self.tool_context.allow_docs:
                    can_enable_setting = True
                    setting_to_enable = "allow_docs"

            session_label: str | None = None
            if request.suggestions:
                from clawcodex_ext.permissions.updates import session_option_label

                session_label = session_option_label(
                    request.suggestions,
                    request.tool_name,
                    request.tool_input,
                )

            option_actions: list[tuple[str, str, str]] = []
            if can_enable_setting:
                option_actions.append(("e", f"Enable {setting_to_enable} and allow", "enable_setting"))
            option_actions.append(("y", "Yes, allow this action", "allow_once"))
            if session_label:
                option_actions.append(("s", f"Yes, {session_label}", "allow_session"))
            option_actions.append(("n", "No, deny this action", "deny"))

            options = [(key, desc) for key, desc, _action in option_actions]
            action_by_key: dict[str, str] = {key.lower(): action for key, _desc, action in option_actions}
            for idx, (_key, _desc, action) in enumerate(option_actions, start=1):
                action_by_key[str(idx)] = action
            action_by_key["yes"] = "allow_once"
            action_by_key["no"] = "deny"
            action_by_key[""] = "allow_once"
            if can_enable_setting:
                action_by_key["enable"] = "enable_setting"
            if session_label:
                action_by_key["session"] = "allow_session"

            im_reply = getattr(self, "_im_reply_controller", None)
            send_permission_prompt = getattr(im_reply, "send_permission_prompt", None)
            im_origin = None
            im_client = getattr(im_reply, "_client", None) if im_reply is not None else None
            peek_origin = getattr(im_client, "peek_reply_origin", None)
            if callable(peek_origin):
                try:
                    im_origin = peek_origin()
                except Exception:  # nosec B110
                    im_origin = None

            if im_origin and callable(send_permission_prompt):
                choice = (
                    self._wait_im_permission_choice(
                        message=request.message,
                        suggestion=None,
                        options=options,
                        allow_choices={key for key, _desc, action in option_actions if action != "deny"},
                    )
                    .strip()
                    .lower()
                )
            else:
                if callable(send_permission_prompt):
                    try:
                        send_permission_prompt(
                            message=request.message,
                            suggestion=None,
                            options=options,
                        )
                    except Exception:  # nosec B110
                        pass

                if get_selection_mode() == "arrow":
                    opt_pairs = [(f"[{key}] {desc}", "") for key, desc in options]
                    result = self._run_arrow_menu(
                        opt_pairs,
                        title="Permission Required",
                        allow_other=False,
                        multi_select=False,
                    )
                    if result is None:
                        return PermissionAskReply(behavior="deny")
                    idx = result if isinstance(result, int) else (result[0] if result else 0)
                    if not 0 <= idx < len(option_actions):
                        return PermissionAskReply(behavior="deny")
                    action = option_actions[idx][2]
                    return self._permission_reply_for_action(
                        action,
                        setting_to_enable,
                        request.suggestions,
                    )

                self.console.print("[bold]Options:[/bold]")
                for i, (key, desc) in enumerate(options, start=1):
                    self.console.print(f"  {i}. [{key}] {desc}")
                self.console.print("")

                choice = self._safe_input("Select option> ").strip().lower()

            action = action_by_key.get(choice)
            if action is None:
                self.console.print("[dim]Invalid choice, defaulting to deny.[/dim]")
                return PermissionAskReply(behavior="deny")
            return self._permission_reply_for_action(
                action,
                setting_to_enable,
                request.suggestions,
            )

    def _permission_reply_for_action(
        self,
        action: str,
        setting_to_enable: str | None,
        suggestions: tuple,
    ) -> PermissionAskReply:
        if action == "enable_setting":
            if setting_to_enable is None:
                return PermissionAskReply(behavior="deny")
            self._enable_permission_setting(setting_to_enable)
            return PermissionAskReply(behavior="allow")
        if action == "allow_once":
            return PermissionAskReply(behavior="allow")
        if action == "allow_session":
            if not suggestions:
                return PermissionAskReply(behavior="deny")
            return PermissionAskReply(behavior="allow", chosen_updates=tuple(suggestions))
        return PermissionAskReply(behavior="deny")

    def _wait_im_permission_choice(
        self,
        *,
        message: str,
        options: list[tuple[str, str]],
        suggestion: str | None = None,
        allow_choices: set[str] | None = None,
    ) -> str:
        """Wait for a WeChat reply to resolve an IM-driven permission prompt.

        Mirrors the menu to WeChat immediately (via the one-shot thread path
        in ``_ImReplyController._send_outbound_text``) then blocks the main
        thread on an event that is set by ``_handle_im_permission_reply``
        (the ``permission_probe`` on ``ReplGatewayClient.deliver``) when a
        matching reply arrives. ``Ctrl-C`` and a 300s timeout default to
        deny so the REPL never hangs indefinitely waiting for WeChat.

        Returns the chosen menu key/number (lowercased) for the existing
        parser in ``_handle_permission_request`` to interpret.
        """
        import os
        import time

        valid: set[str] = set()
        for idx, (key, _desc) in enumerate(options, start=1):
            valid.add(str(idx))
            valid.add(key.lower())
        valid.update({"yes", "no"})

        state: dict = {"event": threading.Event(), "choice": None, "valid": valid}
        # Lazy-init the lock: the running REPL is ClawCodexExtREPL, whose
        # __init__ overrides without super().__init__(), so the attr set in
        # ClawcodexREPL.__init__ may be absent. _handle_permission_request is
        # serialized by _permission_prompt_lock, so only one wait runs at a
        # time — the lazy create is race-free in practice, and the probe
        # (_handle_im_permission_reply) reads the same attr after we publish
        # _im_permission_wait below.
        lock = getattr(self, "_im_permission_lock", None)
        if lock is None:
            lock = threading.Lock()
            self._im_permission_lock = lock
        with lock:
            self._im_permission_wait = state

        im_reply = getattr(self, "_im_reply_controller", None)
        send_permission_prompt = getattr(im_reply, "send_permission_prompt", None)
        if callable(send_permission_prompt):
            try:
                send_permission_prompt(
                    message=message,
                    suggestion=suggestion,
                    options=options,
                    interactive=True,
                    allow_choices=allow_choices,
                )
            except Exception:  # nosec B110
                pass

        try:
            timeout = float(os.environ.get("CLAWCODEX_IM_PERMISSION_TIMEOUT", "300"))
        except (TypeError, ValueError):
            timeout = 300.0
        deadline = time.monotonic() + timeout
        try:
            while not state["event"].wait(timeout=0.5):
                if time.monotonic() >= deadline:
                    self.console.print("[dim]IM permission reply timed out, defaulting to deny.[/dim]")
                    state["choice"] = "n"
                    break
        except KeyboardInterrupt:
            state["choice"] = "n"
        finally:
            with lock:
                self._im_permission_wait = None
        return state["choice"] or "n"

    def _enable_permission_setting(self, setting_name: str | None) -> None:
        """Enable a permission setting in the tool context."""
        if not setting_name:
            return

        self.console.print(f"\n[dim]Enabling {setting_name}...[/dim]")

        if setting_name == "allow_docs":
            self.tool_context.allow_docs = True
            self.console.print(f"[success]✓ {setting_name} enabled for this session[/success]")
            return

        self.console.print(f"[dim]Could not enable {setting_name}.[/dim]")

    def _ensure_command_system(self):
        """Idempotently build the new command system on first slash command.

        ``_load_heavy_runtime`` deliberately does NOT import command_system
        symbols — pulling ``src.command_system`` (~0.8s of transitive deps
        including the away_summary / intent_forecast / cli registrations)
        into the heavy runtime would inflate every REPL cold start regardless
        of whether the user ever types ``/``.

        Instead, ``__init__`` skips command-system setup and we build it here
        on first invocation. The cheap sentinel check makes subsequent calls
        free. Cost moves to the first ``/xxx`` keystroke, which by definition
        happens after the user has had time to read the startup banner.

        Local imports are intentional — see ``_load_heavy_runtime`` note.
        """
        if getattr(self, "command_registry", None) is not None:
            return

        # All command_system symbols imported locally; see module top docstring
        # of ``_load_heavy_runtime`` for the rationale.
        from src.command_system import (
            CommandRegistry,
            create_command_context,
            load_and_register_skills,
            register_builtin_commands,
        )

        # Also register to global registry so execute_command_async can find commands
        register_builtin_commands(None)  # None = use global registry

        # Create command registry and register built-ins
        self.command_registry = CommandRegistry()
        register_builtin_commands(self.command_registry)
        try:
            from clawcodex_ext.away_summary.registration import (
                register_away_summary_commands,
            )
            from clawcodex_ext.intent_forecast.registration import (
                register_intent_forecast_commands,
            )

            register_away_summary_commands(None)
            register_away_summary_commands(self.command_registry)
            register_intent_forecast_commands(None)
            register_intent_forecast_commands(self.command_registry)
        except Exception:  # nosec B110
            pass

        # F-53: auto-expose non-core tools as /<tool-name> slash commands.
        # The runtime ``tool_registry`` is captured lazily at invocation
        # time via ``context.tool_registry`` (set by
        # ``attach_downstream_context`` below), so we only need a
        # schema snapshot at registration time. The default registry
        # gives us that without paying for the SOP tool bridge.
        try:
            from clawcodex_ext.cli.tool_cmd import register_tool_commands

            register_tool_commands(self.command_registry)
            register_tool_commands(None)  # also register in global registry
        except Exception:  # nosec B110
            pass

        # Prompt skills must be present in both registries: the instance
        # registry drives REPL recognition/completion, while
        # ``execute_command_async`` resolves from the global registry.
        # Without this wiring, a typed ``/<skill>`` falls through as ordinary
        # model input and only works if the model happens to call SkillTool.
        try:
            load_and_register_skills(
                project_root=Path.cwd(),
                registry=self.command_registry,
            )
            load_and_register_skills(
                project_root=Path.cwd(),
                registry=None,
            )
        except Exception:  # nosec B110
            pass

        # Create cost tracker and history
        self.cost_tracker = CostTracker()
        self.history_log = HistoryLog()

        # Wire the surface-agnostic UIHost so interactive commands (port of
        # TS ``local-jsx``) can drive a menu / prompt on the REPL. We import
        # lazily to avoid pulling the interactive-command subsystem into the
        # import graph for non-REPL consumers.
        from clawcodex_ext.repl.ui_host import ReplUIHost

        # Create command context
        self.command_context = create_command_context(
            workspace_root=Path.cwd(),
            conversation=self.session.conversation,
            cost_tracker=self.cost_tracker,
            history=self.history_log,
            provider=self.provider,
            ui=ReplUIHost(self._safe_input, self.console, arrow_select=self._arrow_select),
            tool_context=self.tool_context,
        )

        # Merge new commands with built-in list for completion
        self._update_built_in_commands_with_command_system()

    # Backwards-compatible alias — kept so any third-party extension or unit
    # test that calls ``repl._init_command_system()`` (e.g. in fixtures) still
    # works. New code should call ``_ensure_command_system`` which is
    # idempotent.
    def _init_command_system(self):  # noqa: D401 — kept for backward compat
        """Backward-compatible alias for :meth:`_ensure_command_system`."""
        self._ensure_command_system()

    def _update_built_in_commands_with_command_system(self):
        """Update the built-in commands list with commands from the new system."""
        # Start with original built-ins
        self._built_in_commands = list(self._original_built_ins)

        # Add commands from the new command system
        try:
            for cmd in self.command_registry.list_commands():
                cmd_name = f"/{cmd.name}"
                if cmd_name not in self._built_in_commands:
                    self._built_in_commands.append(cmd_name)
                # Add aliases
                for alias in cmd.aliases:
                    alias_name = f"/{alias}"
                    if alias_name not in self._built_in_commands:
                        self._built_in_commands.append(alias_name)
        except Exception:  # nosec B110
            pass

    def _try_execute_new_command(self, command: str, args: str) -> tuple[bool, str | None]:
        """Try to execute a command using the new command system (sync path for LocalCommand only).

        Returns:
            Tuple of (handled: bool, result_text: str | None)
        """
        # Local import — ``execute_command_sync`` is no longer pulled in by
        # ``_load_heavy_runtime``. ``_try_execute_new_command`` only fires on
        # ``/xxx`` slash input, which is the same lazy trigger as
        # ``_ensure_command_system``; no incremental cost.
        from src.command_system import execute_command_sync

        try:
            success, result_text, error = execute_command_sync(command, args, self.command_context)
            if success:
                return True, result_text
            else:
                return False, error
        except Exception as e:
            return False, str(e)

    async def _try_execute_command_async(self, command: str, args: str) -> CommandResult:  # noqa: F821 — forward ref under ``from __future__ import annotations``
        """Execute a command asynchronously, supporting both LocalCommand and PromptCommand.

        Returns:
            CommandResult with the execution result
        """
        # Local import — same lazy rationale as ``_try_execute_new_command``.
        from src.command_system import (
            CommandResult,
            execute_command_async,
        )

        try:
            return await execute_command_async(command, args, self.command_context)
        except Exception as e:
            return CommandResult.error(command, str(e))

    def _run_command_async_with_status(
        self,
        command: str,
        args: str,
        *,
        status_message: str | None = None,
    ) -> CommandResult:  # noqa: F821 — forward ref under ``from __future__ import annotations``
        """Run async slash-command execution without freezing the visible REPL."""

        import concurrent.futures

        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(
                asyncio.run,
                self._try_execute_command_async(command, args),
            )
            if status_message:
                status_ref: list[LiveStatus] = []

                def _on_submit(text: str) -> None:
                    self._enqueue_prompt(text)
                    if status_ref:
                        n = self._queued_count()
                        if n == 0:
                            status_ref[0].update(status_message)
                        else:
                            status_ref[0].update(f"{status_message} ({n} queued)")

                try:
                    with _pt_patch_stdout(raw=True):
                        with LiveStatus(
                            status_message,
                            on_submit=_on_submit,
                            on_expand=self._do_expand_last,
                            on_permission_cycle=self._apply_permission_mode_cycle,
                            completer=self.completer,
                            history=self._file_history,
                            toolbar_text=self._bottom_toolbar,
                        ) as status:
                            status_ref.append(status)
                            self._active_live_status = status
                            try:
                                while not future.done():
                                    time.sleep(0.05)
                            finally:
                                self._active_live_status = None
                except Exception:  # nosec B110
                    self._active_live_status = None
            return future.result()

    def _handle_command_result(self, result: CommandResult) -> bool:  # noqa: F821 — forward ref under ``from __future__ import annotations``
        """Handle the result of a command execution.

        Returns True if the command was handled, False otherwise.
        """
        if not result.success:
            if result.error:
                self.console.print(f"[error]{result.error}[/error]")
            return True

        if result.result_type == "text":
            if result.text:
                if result.display == "assistant":
                    self.console.print()
                    self.console.print(Markdown(result.text))
                    from clawcodex_ext.types.messages import create_message

                    self._engine_messages.append(create_message("assistant", result.text))
                    self.console.print()
                    return True
                if getattr(result, "transient", False):
                    self._print_transient_text(
                        result.text,
                        command=result.command_name,
                    )
                # F-122-F: long /btw answers carry scrollable=True. Route
                # them through the keyboard-scrolled viewer so the user
                # can navigate instead of seeing a wall of text scroll
                # past. Falls back to a flat print when prompt_toolkit is
                # unavailable or the body fits without paging.
                elif getattr(result, "scrollable", False):
                    self._print_scrollable_text(
                        result.text,
                        command=result.command_name,
                    )
                else:
                    self._print_local_command_text(
                        result.text,
                        command=result.command_name,
                    )
                if not getattr(result, "transient", False):
                    self.console.print()
            if result.should_query:
                self._continue_goal_if_idle()
            return True

        elif result.result_type == "prompt":
            # For PromptCommand, extract the text content and send to LLM
            prompt_text = ""
            for item in result.prompt_content:
                if item.get("type") == "text":
                    prompt_text = item.get("text", "")
                    break

            if prompt_text:
                # Send the prompt to the LLM for interactive execution
                # Use higher max_turns for complex commands like /init
                self.console.print("[dim]Initializing workspace setup...[/dim]")
                self.chat(prompt_text)
            return True

        elif result.result_type == "skip":
            # Command handled silently
            return True

        return False

    def _continue_goal_if_idle(self) -> bool:
        """Start an active goal continuation after a local slash command."""

        _load_heavy_runtime()
        try:
            from clawcodex_ext.goal.runtime import goal_runtime_for_context

            goal_runtime = goal_runtime_for_context(self.tool_context)
        except Exception:  # nosec B110
            return False
        if goal_runtime is None:
            return False

        continuation = goal_runtime.continue_if_idle()
        if continuation is None or not goal_runtime.claim_continuation(continuation):
            return False

        messages = list(continuation.messages)
        if not messages:
            return False

        from clawcodex_ext.query.agent_loop_compat import (
            build_effective_system_prompt,
            run_query_as_agent_loop,
        )

        style_name = getattr(self.tool_context, "output_style_name", None)
        style_dir = getattr(self.tool_context, "output_style_dir", None)
        append_prompt = resolve_output_style(style_name, style_dir).prompt
        extra = getattr(self, "_append_system_prompt", "")
        if extra:
            append_prompt = f"{append_prompt}\n\n{extra}"

        effective_system_prompt = build_effective_system_prompt(
            append_prompt,
            self.tool_context,
        )
        initial_messages = [
            *list(getattr(self, "_engine_messages", [])),
            *messages,
        ]
        persisted_messages = list(initial_messages)
        streamed_text = False
        tool_names: dict[str, str] = {}
        continuation_usage = {"input_tokens": 0, "output_tokens": 0}
        abort_controller = AbortController()
        previous_abort_controller = getattr(self.tool_context, "abort_controller", None)
        self.tool_context.abort_controller = abort_controller

        def _on_text_chunk(chunk: str) -> None:
            nonlocal streamed_text
            if not chunk:
                return
            streamed_text = True
            self.console.print(chunk, end="", markup=False, highlight=False, soft_wrap=True)

        def _on_message(message: Any) -> None:
            persisted_messages.append(message)
            message_usage = getattr(message, "usage", None)
            if isinstance(message_usage, dict):
                continuation_usage["input_tokens"] += int(message_usage.get("input_tokens", 0) or 0)
                continuation_usage["output_tokens"] += int(message_usage.get("output_tokens", 0) or 0)
            add_existing = getattr(self.session.conversation, "add_existing_message", None)
            if callable(add_existing):
                add_existing(message)
            else:
                self.session.conversation.add_message(message.role, message.content)
            if isinstance(message, SystemMessage):
                subtype = getattr(message, "subtype", None)
                if subtype in {
                    "goal_evaluation",
                    "goal_achieved",
                    "goal_evaluator_error",
                }:
                    # The initial `/goal <condition>` run uses this adapter,
                    # not QueryEngine.submit_message(). Keep its transcript
                    # visibility identical to ordinary goal continuations.
                    self.console.print()
                    if subtype == "goal_evaluator_error":
                        self._last_chat_outcome = "goal_evaluator_error"
                    style = (
                        "success"
                        if subtype == "goal_achieved"
                        else "warning"
                        if subtype == "goal_evaluator_error"
                        else "dim"
                    )
                    self.console.print(f"[{style}]{escape(str(message.content))}[/{style}]")

        def _on_tool_event(event: ToolEvent) -> None:
            tool_use_id = str(event.tool_use_id or "")
            if event.kind == "tool_use":
                if tool_use_id:
                    tool_names[tool_use_id] = event.tool_name
                summary = summarize_tool_use(event.tool_name, event.tool_input or {})
                suffix = f" ({escape(summary)})" if summary else ""
                self.console.print(f"[success]●[/success] [bold][info]{event.tool_name}[/info][/bold]{suffix}")
                return

            tool_name = event.tool_name or tool_names.get(tool_use_id, "tool")
            output = event.error if event.is_error else event.tool_output
            summary = summarize_tool_result(tool_name, output)
            style = "error" if event.is_error else "dim"
            self.console.print(f"[{style}]  ⎿  {escape(str(summary))}[/{style}]")

        async def _run_query():
            return await run_query_as_agent_loop(
                initial_messages=initial_messages,
                provider=self.provider,
                tool_registry=self.tool_registry,
                tool_context=self.tool_context,
                system_prompt=effective_system_prompt,
                # Claude-style /goal keeps running until its evaluator says
                # the condition is met (or the user cancels/clears it).
                max_turns=0,
                on_event=_on_tool_event,
                on_text_chunk=_on_text_chunk if self.stream else None,
                on_message=_on_message,
                abort_controller=abort_controller,
            )

        def _cancel_goal_continuation() -> None:
            self._last_chat_outcome = "cancelled"
            abort_controller.abort("user_interrupt")
            status = getattr(self, "_active_live_status", None)
            if status is not None:
                try:
                    status.update("[warning]Cancelling…[/warning]")
                except Exception:  # nosec B110
                    pass

        status_ref: list[LiveStatus] = []

        def _on_submit_goal(text: str) -> None:
            self._enqueue_prompt(text)
            if status_ref:
                status_ref[0].update(self._status_message())

        background_requested = False

        def _on_background_goal() -> None:
            nonlocal background_requested
            background_requested = True
            self._last_chat_outcome = "cancelled"
            abort_controller.abort("background")

        self.console.print("\n[bold]Assistant[/bold]")
        try:
            loop = self._get_chat_loop()
        except RuntimeError:
            loop = None

        result = None
        status = None
        cancelled = False
        from clawcodex_ext.utils.abort_controller import AbortError

        self._last_chat_outcome = "success"
        self._im_active_cancel = _cancel_goal_continuation
        try:
            with _pt_patch_stdout(raw=True):
                with LiveStatus(
                    self._status_message(),
                    on_cancel=_cancel_goal_continuation,
                    on_submit=_on_submit_goal,
                    on_expand=self._do_expand_last,
                    on_background=_on_background_goal,
                    on_permission_cycle=self._apply_permission_mode_cycle,
                    completer=self.completer,
                    history=self._file_history,
                    toolbar_text=self._bottom_toolbar,
                ) as status:
                    status_ref.append(status)
                    self._active_live_status = status
                    try:
                        if loop is None or loop.is_closed():
                            result = asyncio.run(_run_query())
                        elif loop.is_running():
                            import concurrent.futures

                            with concurrent.futures.ThreadPoolExecutor() as pool:
                                result = pool.submit(lambda: asyncio.run(_run_query())).result()
                        else:
                            result = loop.run_until_complete(_run_query())
                    finally:
                        self._active_live_status = None
        except (AbortError, asyncio.CancelledError):
            self._last_chat_outcome = "cancelled"
            cancelled = True
        except KeyboardInterrupt:
            self._last_chat_outcome = "cancelled"
            abort_controller.abort("user_interrupt")
            raise
        except Exception as exc:
            if self._last_chat_outcome == "goal_evaluator_error":
                self._engine_messages = persisted_messages
                self._stats_turns += int(getattr(exc, "num_turns", 0) or 0)
                self._stats_input_tokens += continuation_usage["input_tokens"]
                self._stats_output_tokens += continuation_usage["output_tokens"]
                try:
                    self.session.save_transcript()
                except Exception:  # nosec B110
                    pass
                return False
            self._last_chat_outcome = "failure"
            self.console.print(f"\n[error]Error: {escape(str(exc))}[/error]")
            return False
        finally:
            if self._im_active_cancel is _cancel_goal_continuation:
                self._im_active_cancel = None
            self._active_live_status = None
            self.tool_context.abort_controller = previous_abort_controller

        if status is not None:
            pending = getattr(status, "_pending_text", "")
            if pending:
                self._enqueue_prompt(pending)
        if background_requested:
            self._handle_background_escape()
            return False
        if abort_controller.signal.aborted:
            cancelled = True
        if cancelled:
            self.console.print()
            return False
        if result is None:
            return False

        self._engine_messages = persisted_messages
        usage = result.usage or {}
        self._stats_turns += int(result.num_turns or 0)
        self._stats_input_tokens += int(usage.get("input_tokens", 0) or 0)
        self._stats_output_tokens += int(usage.get("output_tokens", 0) or 0)
        response_text = result.response_text
        if response_text and not streamed_text:
            self.console.print(Markdown(response_text))
        self.console.print()
        try:
            self.session.save_transcript()
        except Exception:  # nosec B110
            pass
        return True

    def _print_local_command_text(self, text: str, *, command: str = "") -> None:
        """Print local command output, rendering only /recap as Markdown."""

        if command == "recap" and self._is_recap_text(text):
            self.console.print()
            self.console.print(Markdown(text))
            return
        self.console.print("\n" + text)

    def _print_transient_text(self, text: str, *, command: str = "") -> None:
        """Show a short status view that disappears after Esc/Enter/q."""

        if not text:
            return
        if not _HAS_PROMPT_TOOLKIT:
            self._print_local_command_text(text, command=command)
            return

        from prompt_toolkit.application import Application
        from prompt_toolkit.key_binding import KeyBindings
        from prompt_toolkit.layout import Layout
        from prompt_toolkit.layout.containers import Window
        from prompt_toolkit.layout.controls import FormattedTextControl
        from prompt_toolkit.styles import Style

        body = text.strip()
        if body.startswith("Goal\n\n"):
            body = body[len("Goal\n\n") :]
        label = command or "status"

        def _fragments():
            return [
                ("class:transient-title", f"\n  {label.title()}\n\n"),
                ("", body + "\n"),
                ("class:transient-footer", "\n  Esc to dismiss"),
            ]

        bindings = KeyBindings()

        def _close(event) -> None:
            event.app.exit(result=None)

        bindings.add("escape")(_close)
        bindings.add("enter")(_close)
        bindings.add("q")(_close)
        bindings.add("Q")(_close)
        bindings.add("c-c")(_close)

        application = Application(
            layout=Layout(Window(FormattedTextControl(_fragments))),
            key_bindings=bindings,
            style=Style.from_dict(
                {
                    "transient-title": "bold fg:cyan",
                    "transient-footer": "fg:gray italic",
                }
            ),
            full_screen=False,
            erase_when_done=True,
            mouse_support=False,
        )
        live = self._active_live_status
        try:
            if live is not None:
                with live.paused():
                    application.run()
            else:
                application.run()
        except (EOFError, KeyboardInterrupt):
            return

    # ------------------------------------------------------------------
    # F-122-F: scrollable answer viewer for /btw side questions
    # ------------------------------------------------------------------
    # /btw answers can be long (multi-paragraph explanations). Dumping them
    # in a single block makes them scroll past unreadable. When the engine
    # marks a result as scrollable, render it inside a prompt_toolkit
    # Application with a viewport window the user can navigate with
    # ↑↓ / PgUp / PgDn / Home / End and dismiss with Space / Enter / Esc /
    # q / Ctrl-C. Falls back to a flat print when prompt_toolkit is not
    # available or the body already fits in the terminal height.

    _SCROLL_VIEWER_RESERVED_LINES = 4  # header (2) + footer hint (2)
    _SCROLL_VIEWER_MIN_WINDOW = 5  # never paginate fewer than this many lines

    def _print_scrollable_text(self, text: str, *, command: str = "") -> None:
        """Render *text* in a keyboard-scrollable viewer (F-122-F).

        The viewer opens only if the body exceeds one terminal page; if the
        whole answer fits, we degrade to a flat print (no extra keystroke
        needed to dismiss). When prompt_toolkit is unavailable we also
        degrade to a flat print.
        """
        if not text:
            return

        # Strip the leading newline that _handle_command_result's caller
        # would have inserted; the viewer renders its own header.
        body = text.lstrip("\n")

        # Estimate line count cheaply so we can decide whether to paginate
        # at all. If the body fits on the terminal, skip the viewer.
        lines = body.splitlines() or [""]
        try:
            import shutil

            term_height = shutil.get_terminal_size((100, 24)).lines
        except Exception:  # nosec B110
            term_height = 24
        window = max(
            self._SCROLL_VIEWER_MIN_WINDOW,
            term_height - self._SCROLL_VIEWER_RESERVED_LINES,
        )
        if len(lines) <= window:
            # Body fits on one screen — no viewer needed.
            self.console.print("\n" + body)
            return

        if not _HAS_PROMPT_TOOLKIT:
            # Best-effort fallback: print everything, mark a hint line.
            self.console.print("\n" + body)
            self.console.print("[dim](prompt_toolkit unavailable — answer not paginated)[/dim]")
            return

        self._run_scroll_viewer(body, lines=lines, window=window, command=command)

    def _run_scroll_viewer(
        self,
        body: str,
        *,
        lines: list[str],
        window: int,
        command: str,
    ) -> None:
        """Open the prompt_toolkit Application that paginates *lines*."""
        from prompt_toolkit.application import Application
        from prompt_toolkit.key_binding import KeyBindings
        from prompt_toolkit.layout import Layout
        from prompt_toolkit.layout.containers import Window
        from prompt_toolkit.layout.controls import FormattedTextControl
        from prompt_toolkit.styles import Style

        total = len(lines)
        # Mutable scroll offset captured by closures.
        offset = [0]

        def clamp_offset() -> None:
            offset[0] = max(0, min(offset[0], max(0, total - window)))

        def get_fragments():
            clamp_offset()
            start = offset[0]
            end = min(start + window, total)
            fragments: list[tuple[str, str]] = []
            # Header — no leading 💡: the body already carries the
            # answer's decoration prefix; the header is purely a
            # navigational banner (command + cursor).
            label = command or "answer"
            fragments.append(
                ("class:scroll-header", f"\n  /{label}  "),
            )
            fragments.append(
                ("class:scroll-meta", f"(lines {start + 1}-{end} of {total})\n\n"),
            )
            # Visible window
            for i in range(start, end):
                fragments.append(("", lines[i] + "\n"))
            # Pad short pages so the footer stays at a stable position.
            shown = end - start
            for _ in range(window - shown):
                fragments.append(("", "~\n"))
            # Footer hint
            fragments.append(
                (
                    "class:scroll-footer",
                    "\n  ↑↓ scroll · PgUp/PgDn page · Home/End jump · Space/Enter/Esc/q close",
                ),
            )
            return fragments

        kb = KeyBindings()

        def _close(event) -> None:
            event.app.exit(result=None)

        def _line_up(event) -> None:
            offset[0] -= 1
            event.app.invalidate()

        def _line_down(event) -> None:
            offset[0] += 1
            event.app.invalidate()

        def _page_up(event) -> None:
            offset[0] -= window
            event.app.invalidate()

        def _page_down(event) -> None:
            offset[0] += window
            event.app.invalidate()

        def _home(event) -> None:
            offset[0] = 0
            event.app.invalidate()

        def _end(event) -> None:
            offset[0] = total
            event.app.invalidate()

        kb.add("up")(_line_up)
        kb.add("down")(_line_down)
        kb.add("pageup")(_page_up)
        kb.add("pagedown")(_page_down)
        kb.add("home")(_home)
        kb.add("end")(_end)
        kb.add("space")(_close)
        kb.add("enter")(_close)
        kb.add("escape")(_close)
        kb.add("q")(_close)
        kb.add("Q")(_close)
        kb.add("c-c")(_close)

        pt_style = Style.from_dict(
            {
                "scroll-header": "bold fg:cyan",
                "scroll-meta": "fg:gray",
                "scroll-footer": "fg:gray italic",
            }
        )

        app = Application(
            layout=Layout(Window(FormattedTextControl(get_fragments))),
            key_bindings=kb,
            style=pt_style,
            full_screen=False,
            mouse_support=False,
        )

        live = self._active_live_status
        try:
            if live is not None:
                with live.paused():
                    app.run()
            else:
                app.run()
        except (EOFError, KeyboardInterrupt):
            return

    @staticmethod
    def _is_recap_text(text: str) -> bool:
        # Accept either a single newline (legacy persisted recaps) or the
        # double newline used by ``format_away_summary_for_display`` so
        # Markdown renders the prefix and body as separate paragraphs.
        return text.strip().startswith(("Recapitulate\n", "Recapitulate\n\n", "Away Summary\n", "Away Summary\n\n"))

    def _get_slash_command_words(self) -> list[str]:
        words = list(self._built_in_commands)
        try:
            from src.skills.loader import get_all_skills

            cwd = self.tool_context.cwd or self.tool_context.workspace_root
            for s in get_all_skills(project_root=cwd):
                words.append(f"/{s.name}")
        except Exception:  # nosec B110
            pass
        deduped: list[str] = []
        seen: set[str] = set()
        for w in words:
            lw = w.lower()
            if lw in seen:
                continue
            seen.add(lw)
            deduped.append(w)
        return deduped

    def _get_user_message_history(self) -> list[str]:
        """Return previous user message text from the session.

        Reads from ``self.session.conversation.messages`` which is the
        canonical message store shared with the QueryEngine. Extracts
        only ``UserMessage`` entries so the completer only suggests
        text the user actually typed (not assistant responses).
        Returns messages in chronological order (oldest first).
        """

        try:
            conv = getattr(self, "session", None)
            if conv is None:
                return []
            messages = getattr(conv, "messages", None)
            if messages is None:
                return []
            from clawcodex_ext.types.messages import UserMessage

            result: list[str] = []
            for msg in messages:
                if isinstance(msg, UserMessage):
                    # ``content`` can be str or list[ContentBlock].
                    # We extract plain text for completion purposes.
                    content = msg.content
                    if isinstance(content, str):
                        result.append(content)
                    elif isinstance(content, list):
                        for block in content:
                            if hasattr(block, "text"):
                                result.append(block.text)
            return result
        except Exception:  # nosec B110
            return []

    # REPL-only built-ins not covered by the shared TUI ``LOCAL_BUILTINS``.
    # Used to seed descriptions for the prompt_toolkit completion menu so
    # ``/save`` etc. show meta text alongside the registry-backed entries.
    _REPL_EXTRA_BUILTIN_DESCRIPTIONS: dict[str, str] = {
        "save": "Save the conversation to a file",
        "load": "Load a saved conversation",
        "tool": "Inspect or invoke a single tool",
        "init": "Initialize a CLAUDE.md for this workspace",
        "tui": "Switch to the Textual TUI",
        "gateway": "Connect, status, or disconnect the IM gateway",
    }

    _SLASH_SUGGESTIONS_TTL_S = 30.0

    def _get_slash_command_suggestions(self) -> list[Any]:
        """Return rich slash-command entries (name + description + tag).

        Drives the prompt_toolkit completion menu's two-column display
        (command name on the left, description as ``display_meta`` on
        the right) and stays in lock-step with the TUI palette by
        reusing :func:`src.tui.commands.build_command_suggestions`. Adds
        the REPL-only built-ins (``/save``, ``/load``, ``/tool``,
        ``/init``, ``/tui``) that the shared builder doesn't know about.

        Cached with a 30-second TTL: the builder walks user/project/managed
        skills dirs (~1.1s cold, ~0.4 ms warm) and prompt_toolkit calls
        this on every keystroke after ``/``, so rebuilding the 500-entry
        list each keystroke is what made the popup feel laggy.
        """

        now = time.monotonic()
        cached = self._slash_suggestions_cache
        if cached is not None and (now - self._slash_suggestions_cache_at) < self._SLASH_SUGGESTIONS_TTL_S:
            return cached

        try:
            from src.tui.commands import CommandSuggestion, build_command_suggestions

            cwd = self.tool_context.cwd or self.tool_context.workspace_root
            base = build_command_suggestions(cwd, self.tool_context)

            have = {s.name.lower() for s in base if hasattr(s, "name")}
            extra: list[Any] = []
            for name, description in self._REPL_EXTRA_BUILTIN_DESCRIPTIONS.items():
                if name in have:
                    continue
                extra.append(CommandSuggestion(name=name, description=description))
            # Built-ins lead the menu, then registry/skills (the order
            # ``build_command_suggestions`` already produces).
            result: list[Any] = [
                *(s for s in base if getattr(s, "source", "") == "builtin"),
                *extra,
                *(s for s in base if getattr(s, "source", "") != "builtin"),
            ]
        except Exception:  # nosec B110
            result = []

        self._slash_suggestions_cache = result
        self._slash_suggestions_cache_at = now
        return result

    def _warm_slash_suggestions_cache(self) -> None:
        """Pre-populate the slash-command suggestion cache off the main thread.

        Called once from ``__init__``. Building the suggestion list cold
        is ~1.1 s on a populated skills tree, which is what the user
        perceives as latency on the very first ``/`` press. By doing the
        work in a daemon thread during REPL startup the cache is already
        warm by the time the user presses ``/``.
        """

        try:
            self._get_slash_command_suggestions()
        except Exception:  # nosec B110
            # Warming is a best-effort optimization; falling back to the
            # lazy cold path on the next ``/`` press is acceptable.
            pass

    def _refresh_completer(self) -> None:
        # The slash + ``@``-file completers are stable for the lifetime
        # of the REPL: ``_SlashOnlyCompleter`` reads its word list
        # lazily, and ``AtFileCompleter`` rebuilds its file index on
        # its own TTL. We just rebind the merged completer onto the
        # PromptSession in case anything in the tool-system replaced
        # ``self.completer`` with a stub.
        try:
            from prompt_toolkit.completion import merge_completers

            if not hasattr(self, "_at_completer") or self._at_completer is None:
                self._at_completer = AtFileCompleter(cwd=str(self.tool_context.workspace_root))
            if not hasattr(self, "_slash_completer") or self._slash_completer is None:
                self._slash_completer = _SlashOnlyCompleter(
                    self._get_slash_command_words,
                    suggestions_provider=self._get_slash_command_suggestions,
                )
            if not hasattr(self, "_agent_completer") or self._agent_completer is None:
                self._agent_completer = AgentMentionCompleter(self._available_agents)
            self.completer = merge_completers(
                [
                    self._slash_completer,
                    self._at_completer,
                    self._agent_completer,
                    self._message_history_completer,
                ]
            )
            if hasattr(self, "prompt_session") and getattr(self.prompt_session, "completer", None) is not None:
                self.prompt_session.completer = self.completer
        except Exception:  # nosec B110
            return

    def _show_slash_palette(self, query: str | None = None) -> None:
        q = (query or "").strip().lower()
        self.console.print("\n[bold]Available commands and skills:[/bold]")

        # Collect all commands
        all_commands: list[tuple[str, str, str]] = []  # (name, description, type)
        seen: set[str] = set()

        def add_command(name: str, desc: str, cmd_type: str = "command") -> None:
            if name in seen:
                return
            seen.add(name)
            if q and q not in name.lower() and q not in desc.lower():
                return
            all_commands.append((name, desc, cmd_type))

        # Add built-in commands
        for cmd in self._original_built_ins:
            if cmd == "/":
                continue
            add_command(cmd, "", "command")

        # Add commands from new command system
        try:
            for cmd in self.command_registry.list_commands():
                cmd_name = f"/{cmd.name}"
                if cmd_name in self._original_built_ins:
                    continue
                alias_str = f" (aliases: {', '.join(cmd.aliases)})" if cmd.aliases else ""
                add_command(f"{cmd_name}{alias_str}", cmd.description, "command")
        except Exception:  # nosec B110
            pass

        # Add skills
        try:
            from src.skills.loader import get_all_skills

            cwd = self.tool_context.cwd or self.tool_context.workspace_root
            skills = list(get_all_skills(project_root=cwd))
            skills.sort(key=lambda s: s.name.lower())
            for s in skills:
                desc = (s.description or "").strip()
                add_command(f"/{s.name}", desc, "skill")
        except Exception:  # nosec B110
            pass

        # Sort and display
        all_commands.sort(key=lambda x: x[0].lower())
        if not all_commands and q:
            # No matches for the query — show a helpful hint instead of an empty list
            self.console.print(f"  [dim]No matching commands for [warning]/{q}[/warning].[/dim]")
            self.console.print(
                "  [dim]Type [secondary]/[/secondary] to browse all available commands or [secondary]/help[/secondary] for details.[/dim]"
            )
        else:
            for name, desc, cmd_type in all_commands:
                if cmd_type == "skill":
                    self.console.print(f"  [secondary]{name}[/secondary]")
                    if desc:
                        self.console.print(f"    [dim]{desc}[/dim]")
                else:
                    if desc:
                        self.console.print(f"  {name}  [dim]- {desc}[/dim]")
                    else:
                        self.console.print(f"  {name}")

        self.console.print()

    _MAX_PREVIEW_LINES = 3

    _EDIT_DIFF_MAX_LINES = 30

    def _format_edit_diff_preview(self, hunks: list[dict]):
        """Render an Edit/MultiEdit structured patch as a Rich :class:`Group`.

        Mirrors the TUI's ``EditActivity`` body: a one-line
        ``Added X lines, removed Y lines`` summary above the line-numbered
        diff with red/green markers and shaded backgrounds. Long diffs are
        truncated with a ``… +N more diff lines`` footer to keep the
        scrollback compact.
        """

        adds = 0
        removes = 0
        for hunk in hunks:
            for raw in hunk.get("lines") or []:
                if raw.startswith("+"):
                    adds += 1
                elif raw.startswith("-"):
                    removes += 1

        summary = _format_edit_summary_text(adds, removes) or "no changes"
        summary_text = Text(summary, style="dim")

        # Snap to a sane width: ``self.console.width`` falls back to 80
        # when stdout is not a TTY. Fenced to 1 so degenerate widths don't
        # produce negative padding.
        console_width = max(1, getattr(self.console, "width", 0) or 80)

        # Color bar starts at the line-number column and ends 7 cols
        # short of the right terminal edge, leaving visible breathing
        # room on the right so the bar doesn't run flush against the
        # screen border.
        target_right = max(1, console_width - 7)

        diff = Text()
        rendered = 0
        truncated = False
        for hunk in hunks:
            if truncated:
                break
            old_lineno = int(hunk.get("oldStart", 0) or 0)
            new_lineno = int(hunk.get("newStart", 0) or 0)
            for raw in hunk.get("lines") or []:
                if rendered >= self._EDIT_DIFF_MAX_LINES:
                    truncated = True
                    break
                # Edit's structuredPatch carries lines that already retain
                # their source ``\n`` (Edit calls splitlines(keepends=True)
                # before unified_diff). Strip it here so we don't double up
                # on newlines and produce blank rows between every entry.
                stripped = raw.rstrip("\n").rstrip("\r")
                if stripped.startswith("+"):
                    # Colors mirror ``typescript/src/utils/theme.ts darkTheme``
                    # (``diffAdded: 'rgb(34,92,43)'``,
                    # ``diffRemoved: 'rgb(122,41,54)'``). The bar begins
                    # one column to the left of the line-number digits
                    # (i.e. the gutter carries a 1-col leading bg pad).
                    body = stripped[1:]
                    num_str = str(new_lineno)
                    lead = " " * max(0, 4 - len(num_str) - 1)
                    gutter = f" {num_str} "
                    visible = len(gutter) + 1 + cell_len(body)
                    padding = max(0, target_right - len(lead) - visible)
                    diff.append(lead)
                    diff.append(gutter, style=f"on {self._repl_palette.diff_add}")
                    diff.append("+", style=f"bold on {self._repl_palette.diff_add}")
                    diff.append(body + " " * padding, style=f"on {self._repl_palette.diff_add}")
                    diff.append("\n")
                    new_lineno += 1
                elif stripped.startswith("-"):
                    body = stripped[1:]
                    num_str = str(old_lineno)
                    lead = " " * max(0, 4 - len(num_str) - 1)
                    gutter = f" {num_str} "
                    visible = len(gutter) + 1 + cell_len(body)
                    padding = max(0, target_right - len(lead) - visible)
                    diff.append(lead)
                    diff.append(gutter, style=f"on {self._repl_palette.diff_remove}")
                    diff.append("-", style=f"bold on {self._repl_palette.diff_remove}")
                    diff.append(body + " " * padding, style=f"on {self._repl_palette.diff_remove}")
                    diff.append("\n")
                    old_lineno += 1
                else:
                    body = stripped[1:] if stripped.startswith(" ") else stripped
                    # Context lines have no bg; keep gutter width aligned
                    # with add/remove rows so columns line up.
                    diff.append(f"{old_lineno:>4}  " + body + "\n", style="dim")
                    old_lineno += 1
                    new_lineno += 1
                rendered += 1

        if truncated:
            total = sum(len(h.get("lines") or []) for h in hunks)
            remaining = max(0, total - rendered)
            diff.append(
                f"     … +{remaining} more diff {'line' if remaining == 1 else 'lines'}\n",
                style="dim",
            )

        return Group(summary_text, diff) if Group is not None else summary_text

    def _format_tool_result_preview(
        self,
        block: "ToolResultBlock",
        tool_info: tuple[str, dict] | None,
    ):
        """Return either a plain string or a Rich renderable (Edit diff)."""
        import json as _json

        raw = block.content if isinstance(block.content, str) else str(block.content)
        tool_name = tool_info[0] if tool_info else ""

        # Prefer the original ToolResult.output threaded through as
        # in-process metadata — `block.content` is the API-mapped string
        # (e.g. "The file X has been updated successfully.") and no longer
        # carries structured fields like Edit's `structuredPatch`.
        parsed: dict | None = None
        meta_output = getattr(block, "metadata", None)
        if isinstance(meta_output, dict):
            tool_output = meta_output.get("tool_output")
            if isinstance(tool_output, dict):
                parsed = tool_output
        if parsed is None:
            try:
                parsed = _json.loads(raw)
                if not isinstance(parsed, dict):
                    parsed = None
            except Exception:  # nosec B110
                pass

        if tool_name == "Read":
            if parsed:
                t = parsed.get("type", "")
                if t == "file_unchanged":
                    return "Unchanged since last read"
                f = parsed.get("file", {})
                n = f.get("numLines", 0)
                if t == "text":
                    return f"Read {n} {'line' if n == 1 else 'lines'}"
                if t == "notebook":
                    cells = f.get("cells", [])
                    return f"Read {len(cells)} cells"
                if t == "pdf":
                    return "Read PDF"
                if t == "image":
                    return "Read image"
            if "unchanged" in raw.lower():
                return "Unchanged since last read"
            return "Read file"

        if tool_name == "Bash":
            stdout = raw
            if parsed:
                stdout = parsed.get("stdout", "")
                stderr = parsed.get("stderr", "")
                if not stdout and not stderr:
                    return "(No output)"
                if not stdout:
                    stdout = stderr
            if not stdout or not stdout.strip():
                return "(No output)"
            lines = stdout.rstrip("\n").split("\n")
            total_chars = len(stdout)
            if len(lines) <= self._MAX_PREVIEW_LINES + 1 and total_chars <= 200:
                return stdout.rstrip("\n")
            if len(lines) <= self._MAX_PREVIEW_LINES + 1:
                first_line = lines[0]
                if len(first_line) > 120:
                    return f"{first_line[:120]}…\n… +{total_chars - 120} chars"
                return f"{stdout[:200]}…\n… +{total_chars - 200} chars"
            preview = "\n".join(lines[: self._MAX_PREVIEW_LINES])
            remaining = len(lines) - self._MAX_PREVIEW_LINES
            return f"{preview}\n… +{remaining} lines"

        if tool_name == "Glob":
            if parsed:
                n = parsed.get("numFiles", 0)
                return f"Found {n} {'file' if n == 1 else 'files'}"
            return "done"

        if tool_name == "Grep":
            if parsed:
                mode = parsed.get("mode", "files_with_matches")
                if mode == "content":
                    n = parsed.get("numLines", 0)
                    return f"Found {n} {'line' if n == 1 else 'lines'}"
                if mode == "count":
                    n = parsed.get("numMatches", 0)
                    nf = parsed.get("numFiles", 0)
                    return f"Found {n} {'match' if n == 1 else 'matches'} across {nf} {'file' if nf == 1 else 'files'}"
                n = parsed.get("numFiles", 0)
                return f"Found {n} {'file' if n == 1 else 'files'}"
            return "done"

        if tool_name == "Write":
            # Port of ``typescript/src/tools/FileWriteTool/UI.tsx`` —
            # ``FileWriteToolCreatedMessage`` renders ``Wrote N lines to
            # <path>`` followed by the first MAX_LINES_TO_RENDER (10) lines
            # of the new content and a ``… +M lines`` footer when truncated.
            # Update results render a diff in the TS UI; we keep that as a
            # follow-up and only show the header for now.
            path = ""
            content = ""
            if tool_info and isinstance(tool_info[1], dict):
                path = tool_info[1].get("file_path") or tool_info[1].get("filePath") or ""
                c = tool_info[1].get("content")
                if isinstance(c, str):
                    content = c
            if parsed:
                path = parsed.get("filePath") or parsed.get("path") or path
                if not content:
                    c = parsed.get("content")
                    if isinstance(c, str):
                        content = c
            if not path:
                return "done"

            # Distinguish create vs update from the API result string emitted
            # by ``_map_result_to_api`` in ``src/tool_system/tools/write.py``.
            is_update = "has been updated successfully" in raw

            short = self._shorten_path_text(path)
            # ``countLines`` parity: trailing newline is a terminator.
            if content:
                parts = content.split("\n")
                n = len(parts) - 1 if content.endswith("\n") else len(parts)
            else:
                n = 0

            header = f"Wrote {n} {'line' if n == 1 else 'lines'} to {short}" if n else f"Wrote to {short}"
            if is_update or not content:
                return header

            MAX = 10
            content_lines = content.split("\n")
            # Drop the trailing empty element produced by a terminator newline
            # so we don't render a phantom blank line N+1.
            if content.endswith("\n") and content_lines and content_lines[-1] == "":
                content_lines = content_lines[:-1]
            preview_lines = content_lines[:MAX]
            body = "\n".join(f"     {i:>3}  {line}" for i, line in enumerate(preview_lines, start=1))
            extra = max(0, len(content_lines) - MAX)
            footer = f"\n     … +{extra} {'line' if extra == 1 else 'lines'} (ctrl+o to expand)" if extra else ""
            if extra:
                # Stash the full content so ``ctrl+o`` can re-print it as
                # a fresh block below. We can't mutate the truncated
                # block in scrollback once it's printed, so the
                # expansion appends instead of swapping in place.
                self._stash_expandable(f"Write({short})", content)
            if not body:
                return header
            return f"{header}\n{body}{footer}"

        if tool_name in ("Edit", "MultiEdit"):
            # Port of ``typescript/src/components/FileEditToolUpdatedMessage.tsx``:
            # show ``Added X lines, removed Y lines`` plus the line-numbered
            # diff with red/green markers, instead of a bare ``done``.
            if parsed:
                hunks = parsed.get("structuredPatch") or []
                if hunks:
                    return self._format_edit_diff_preview(hunks)
                if parsed.get("type") == "create":
                    path = parsed.get("filePath") or parsed.get("path") or ""
                    content = parsed.get("content") or ""
                    if path:
                        if content:
                            parts = content.split("\n")
                            n = len(parts) - 1 if content.endswith("\n") else len(parts)
                        else:
                            n = 0
                        short = self._shorten_path_text(path)
                        return f"Wrote {n} {'line' if n == 1 else 'lines'} to {short}" if n else f"Wrote to {short}"
            return "done"

        if tool_name == "TaskCreate":
            if parsed:
                task = parsed.get("task") or {}
                subject = task.get("subject") or ""
                task_id = task.get("id") or ""
                if subject:
                    return f"Created task #{task_id}: {subject}"
                if task_id:
                    return f"Created task #{task_id}"
            return "Task created"

        if tool_name == "TaskUpdate":
            if parsed:
                changed = parsed.get("updatedFields") or []
                task_id = parsed.get("taskId") or ""
                status_change = parsed.get("statusChange") or {}
                if status_change:
                    return f"Task #{task_id}: {status_change.get('from')} → {status_change.get('to')}"
                if "deleted" in changed:
                    return f"Task #{task_id} deleted"
                if changed:
                    return f"Task #{task_id} updated ({', '.join(changed)})"
            return "Task updated"

        if tool_name == "TaskList":
            if parsed:
                tasks = parsed.get("tasks") or []
                return f"Listed {len(tasks)} task{'' if len(tasks) == 1 else 's'}"
            return "Listed tasks"

        if tool_name == "TaskGet":
            if parsed and parsed.get("task"):
                t = parsed["task"]
                return f"Task #{t.get('id')}: {t.get('subject')} ({t.get('status')})"
            return "Task not found"

        if tool_name in ("Agent", "Task"):
            # Show the subagent's terminal outcome instead of the raw JSON
            # envelope (which dumps prompt / agent_id / token counts inline).
            content_text = ""
            agent_type = ""
            tool_uses_count: int | None = None
            duration_ms: int | None = None
            if parsed:
                agent_type = str(parsed.get("agent_type") or "")
                blocks = parsed.get("content")
                if isinstance(blocks, list):
                    parts = []
                    for b in blocks:
                        if isinstance(b, dict) and b.get("type") == "text":
                            t = b.get("text")
                            if isinstance(t, str):
                                parts.append(t)
                    content_text = "\n".join(parts).strip()
                elif isinstance(blocks, str):
                    content_text = blocks.strip()
                tu = parsed.get("total_tool_use_count")
                if isinstance(tu, int):
                    tool_uses_count = tu
                dur = parsed.get("total_duration_ms")
                if isinstance(dur, int):
                    duration_ms = dur
            head_bits: list[str] = []
            if agent_type:
                head_bits.append(f"@{agent_type}")
            stats: list[str] = []
            if isinstance(tool_uses_count, int):
                stats.append(f"{tool_uses_count} tool use{'' if tool_uses_count == 1 else 's'}")
            if isinstance(duration_ms, int) and duration_ms > 0:
                if duration_ms >= 1000:
                    stats.append(f"{duration_ms / 1000:.1f}s")
                else:
                    stats.append(f"{duration_ms}ms")
            if stats:
                head_bits.append("(" + ", ".join(stats) + ")")
            head = " ".join(head_bits) if head_bits else "Agent done"
            if not content_text:
                return head
            # Show the first non-empty content line plus an ellipsis hint when
            # there's more underneath — keeps the result block compact.
            lines = [ln for ln in content_text.splitlines() if ln.strip()]
            if not lines:
                return head
            first = lines[0]
            if len(first) > 200:
                first = first[:197] + "…"
            if len(lines) > 1:
                return f"{head}\n{first}\n… +{len(lines) - 1} more line{'' if len(lines) - 1 == 1 else 's'}"
            return f"{head}\n{first}"

        if tool_name == "TodoWrite":
            if parsed:
                new = parsed.get("newTodos") or []
                done = sum(1 for t in new if t.get("status") == "completed")
                in_prog = sum(1 for t in new if t.get("status") == "in_progress")
                pending = sum(1 for t in new if t.get("status") == "pending")
                return (
                    f"{len(new)} todo{'' if len(new) == 1 else 's'} "
                    f"({done} done, {in_prog} in progress, {pending} open)"
                )
            return "Todos updated"

        if not raw or len(raw) < 80:
            return raw or "done"
        lines = raw.rstrip("\n").split("\n")
        total_chars = len(raw)
        if len(lines) <= self._MAX_PREVIEW_LINES + 1 and total_chars <= 200:
            return raw.rstrip("\n")
        if len(lines) <= self._MAX_PREVIEW_LINES + 1:
            first_line = lines[0]
            if len(first_line) > 120:
                return f"{first_line[:120]}…\n… +{total_chars - 120} chars"
            return f"{raw[:200]}…\n… +{total_chars - 200} chars"
        preview = "\n".join(lines[: self._MAX_PREVIEW_LINES])
        remaining = len(lines) - self._MAX_PREVIEW_LINES
        return f"{preview}\n… +{remaining} lines"

    def _record_tool_result_message(self, message: Any) -> bool:
        """Keep an engine-emitted tool result in the persisted conversation."""
        content = getattr(message, "content", None)
        if not isinstance(content, list) or not any(isinstance(block, ToolResultBlock) for block in content):
            return False
        self.session.conversation.add_existing_message(message)
        return True

    def _available_agents(self) -> list[Any]:
        """Return the list of agent definitions that can be invoked via ``@agent-...``.

        Calls the on-disk loader so user / project / managed / plugin
        agents participate in the same ``@agent-<type>`` lookup the
        TypeScript ``processAgentMentions`` performs. ``options.agent_definitions``
        is still honored as an SDK-side override and supports both the
        canonical ``{"active_agents": [...]}`` shape and a legacy flat
        list/dict form so existing harnesses keep working.
        """
        try:
            from clawcodex_ext.agent.agent_definitions import get_built_in_agents
            from clawcodex_ext.agent.load_agents_dir import get_agents_for_mentions
        except Exception:
            return []

        extra = getattr(
            getattr(self.tool_context, "options", None),
            "agent_definitions",
            None,
        )
        if isinstance(extra, dict):
            active = extra.get("active_agents")
            if isinstance(active, list) and active:
                return list(active)

        try:
            cwd = self.tool_context.cwd or self.tool_context.workspace_root
            return get_agents_for_mentions(
                cwd,
                tool_context=self.tool_context,
                runtime_context=getattr(self, "runtime_context", None),
            )
        except Exception:
            return list(get_built_in_agents())

    def _enqueue_prompt(self, text: str) -> None:
        """Append a user-typed prompt to the user queue from any thread."""

        text = (text or "").strip()
        if not text:
            return
        with self._queued_prompts_lock:
            self._queued_prompts.append(text)

    def _wake_prompt_for_im(self) -> None:
        """Wake the REPL prompt loop so an IM-enqueued prompt is drained.

        The IM gateway opt-in enqueues inbound WeChat messages via
        ``_enqueue_prompt`` from the gateway IPC read loop (on
        ``_cron_loop``). But the main loop blocks on
        ``prompt_async('❯ ')`` waiting for keyboard input, and the only
        other wake (``_watch_outbox``) fires solely on cron outbox events
        — so without an explicit wake the enqueued prompt would sit in
        ``_queued_prompts`` indefinitely: never displayed, processed, or
        replied to.

        This schedules ``app.exit(_CRON_WAKE)`` on ``_cron_loop`` when a
        prompt is actually pending, mirroring ``_watch_outbox``. After the
        wake, the loop iterates and ``_pop_queued_prompt`` drains the
        queued IM prompt on the next cycle. Safe to call from any thread
        (uses ``call_soon_threadsafe``); a no-op when no prompt is active
        (the next loop iteration drains the queue naturally).
        """
        loop = getattr(self, "_cron_loop", None)
        if loop is None or not loop.is_running():
            return
        loop.call_soon_threadsafe(self._exit_pending_prompt_for_im)

    def _exit_pending_prompt_for_im(self) -> None:
        """Exit the pending prompt_async so the loop drains queued prompts.

        Runs on ``_cron_loop``. Only pokes ``app.exit`` when a prompt is
        actually pending (``app.future`` set) — calling it with no active
        prompt crashes prompt_toolkit. Same guard ``_watch_outbox`` uses.
        """
        ps = getattr(self, "prompt_session", None)
        if ps is None:
            return
        app = getattr(ps, "app", None)
        if app is None:
            return
        if getattr(app, "future", None) is not None:
            # This is a programmatic wake, not a submitted keyboard prompt.
            # Ask prompt_toolkit to erase the abandoned prompt row; otherwise
            # app.exit() commits an empty prompt line to scrollback immediately
            # before the queued IM message is echoed.
            app.erase_when_done = True
            app.exit(result=_CRON_WAKE)

    def _get_chat_loop(self):
        """Return the event loop ``chat()`` should pump for the turn.

        Prefers the long-lived ``_cron_loop`` (where the IM gateway IPC
        reader + heartbeat live) so the loop is pumped during the turn —
        otherwise WeChat ``/stop`` and permission replies arriving mid-turn
        are never processed until the turn ends (the IPC reader is starved).
        ``_cron_loop`` is created in ``run()`` via ``new_event_loop()`` but
        never set as the thread's current loop, so ``asyncio.get_event_loop()``
        returns a DIFFERENT loop — hence the explicit preference. Falls back
        to the thread's current loop when ``_cron_loop`` is absent
        (headless/test paths without ``run()``), creating one when Python
        3.11+ has no current loop configured.
        """
        loop = getattr(self, "_cron_loop", None)
        if loop is not None:
            return loop
        try:
            return asyncio.get_event_loop()
        except RuntimeError:
            # Since Python 3.11, get_event_loop() can raise after an
            # asyncio.run() call (or whenever the policy has no loop for the
            # current thread).  Keep the pre-3.11 synchronous REPL behaviour:
            # provision a reusable loop for the caller to pump.
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            return loop

    def _interrupt_active_chat_from_im(self) -> bool:
        """Cancel the currently running REPL turn from an IM control command."""
        cancel = self._im_active_cancel
        if cancel is not None:
            try:
                cancel()
                return True
            except Exception:
                return False
        abort_controller = getattr(self, "_direct_abort_controller", None)
        if abort_controller is not None and not abort_controller.signal.aborted:
            abort_controller.abort("user_interrupt")
            return True
        return False

    def _enqueue_cron_prompt(self, text: str) -> None:
        """Append a cron-generated prompt to the cron queue from any thread.

        Cron prompts are consumed with lower priority than user input —
        see ``_pop_queued_prompt``.
        """
        text = (text or "").strip()
        if not text:
            return
        with self._queued_prompts_lock:
            self._cron_queued_prompts.append(text)

    def _pop_queued_prompt(self) -> tuple[str, str] | None:
        """Pop next prompt. User queue has priority over cron queue.

        Returns (text, source) where source is ``'user'`` or ``'cron'``,
        or None if both queues are empty.
        """
        with self._queued_prompts_lock:
            if self._queued_prompts:
                return (self._queued_prompts.popleft(), "user")
            if self._cron_queued_prompts:
                return (self._cron_queued_prompts.popleft(), "cron")
            return None

    def _ensure_background_output_queue(self) -> None:
        if not hasattr(self, "_background_outputs_lock"):
            self._background_outputs_lock = threading.Lock()
        if not hasattr(self, "_background_outputs"):
            self._background_outputs = []

    def _enqueue_background_output(self, text: str) -> None:
        self._ensure_background_output_queue()
        with self._background_outputs_lock:
            self._background_outputs.append(text)

    def _drain_background_outputs(self) -> None:
        self._ensure_background_output_queue()
        with self._background_outputs_lock:
            outputs = list(self._background_outputs)
            self._background_outputs.clear()
        for text in outputs:
            if self._is_recap_text(text):
                self._print_local_command_text(text, command="recap")
                continue
            self.console.print(text)

    async def _prompt_with_cron_watch(self) -> str | None:
        """Run prompt_async with a concurrent outbox watcher.

        The watcher runs in the same event loop and checks for cron events
        every 1 second. When events are found, it calls app.exit(_CRON_WAKE)
        from within the event loop, avoiding cross-thread issues.

        The watcher runs in the same event loop as the prompt, calls
        app.exit() from within the loop — no cross-thread issues, no
        exception propagation issues.
        """
        app = self.prompt_session.app
        original_erase_when_done = getattr(app, "erase_when_done", False)

        async def _watch_outbox():
            """Watch for cron events and wake the prompt when found."""
            while True:
                await asyncio.sleep(1.0)
                outbox = getattr(self.tool_context, "outbox", None)
                if outbox and getattr(app, "future", None) is not None:
                    app.erase_when_done = True
                    app.exit(result=_CRON_WAKE)
                    return

        watch_task = asyncio.ensure_future(_watch_outbox())
        buffer_changed_handler = None
        default_buffer = None
        try:
            controller = getattr(self, "_intent_forecast_controller", None)
            default_buffer = getattr(getattr(self.prompt_session, "app", None), "default_buffer", None)
            if controller is not None and default_buffer is not None:

                def _on_text_changed(_sender) -> None:
                    try:
                        controller.on_prompt_draft_changed(str(getattr(default_buffer, "text", "") or ""))
                    except Exception:  # nosec B110
                        pass

                buffer_changed_handler = _on_text_changed
                try:
                    default_buffer.on_text_changed += buffer_changed_handler
                except Exception:  # nosec B110
                    buffer_changed_handler = None
            return await self.prompt_session.prompt_async("❯ ")
        finally:
            # A programmatic wake temporarily enables erase_when_done so its
            # empty prompt row is not left in scrollback. Restore the normal
            # interactive behavior for the next prompt.
            app.erase_when_done = original_erase_when_done
            if buffer_changed_handler is not None and default_buffer is not None:
                try:
                    default_buffer.on_text_changed -= buffer_changed_handler
                except Exception:  # nosec B110
                    pass
            watch_task.cancel()
            try:
                await watch_task
            except asyncio.CancelledError:
                pass

    def _drain_cron_outbox(self) -> None:
        _load_cron_runtime()
        """Drain ``cron_prompt`` / ``cron_missed`` events from the
        tool context outbox and enqueue them as user-submitted prompts.

        Called every iteration in ``run()`` before the normal prompt check,
        so a background cron firing is injected as if the user typed it.

        Cron prompts are wrapped with a context prelude so the LLM
        understands this is an automated scheduled task execution,
        not a new user request. This prevents the LLM from asking
        clarifying questions (e.g. "when should I remind you?")
        instead of directly executing the prompt.

        Accumulation guard: if a task_id is already being processed
        (present in ``_cron_active_tasks``), the duplicate prompt is
        discarded and its run is finalized as "cancelled". This prevents
        outbox pile-up when a recurring task's interval is shorter than
        its execution time.

        F-22-G-2: the typed-or-dict parsing and prompt wrapping is now
        delegated to :class:`CronDispatchBridge.drain` /
        :meth:`CronDispatchBridge.drain_missed`. The accumulation
        guard stays here because it depends on the per-REPL-session
        ``_cron_active_tasks`` dictionary, not on the bridge.
        """
        if not _HAS_CRON:
            return
        if not hasattr(self, "_cron_active_tasks"):
            self._cron_active_tasks = {}
        outbox = getattr(self.tool_context, "outbox", None)
        if not outbox:
            return
        # Lazy import to avoid hard dependency at module import time;
        # ``_HAS_CRON`` already guards downstream call sites.
        from clawcodex_ext.cron_system.dispatch import CronDispatchBridge

        bridge = CronDispatchBridge(
            self.tool_context.workspace_root,
            wrap_prompt=_wrap_cron_prompt,
        )
        drained: list[str] = []
        # cron_prompt: typed-or-dict parsing + wrap done by bridge;
        # accumulation guard stays local.
        for event in bridge.drain(outbox):
            if event.task_id and event.task_id in self._cron_active_tasks:
                self._finalize_cron_run(event.run_id, "cancelled")
                continue
            if event.task_id and event.run_id:
                self._cron_active_tasks[event.task_id] = event.run_id
            drained.append(event.wrapped_prompt)
        # cron_missed: notifications are delivered verbatim (no wrap).
        for missed in bridge.drain_missed(outbox):
            drained.append(missed.notification)
        for text in drained:
            self._enqueue_cron_prompt(text)

    def _extract_cron_task_id(self, user_input: str) -> str | None:
        first_line = user_input.split("\n", 1)[0]
        if not first_line.startswith("✻ Running scheduled task"):
            return None
        sep = " · "
        idx = first_line.find(sep)
        if idx == -1:
            return None
        task_id = first_line[idx + len(sep) :].strip()
        return task_id if task_id else None

    def _claim_cron_task(self, task_id: str) -> str | None:
        active = getattr(self, "_cron_active_tasks", None)
        if active is None:
            return None
        run_id = active.get(task_id)
        if not run_id:
            return None
        if claim_cron_run is None:
            return run_id
        try:
            claimed = claim_cron_run(self.tool_context.workspace_root, run_id)
        except Exception:  # nosec B110
            return run_id
        return claimed.id if claimed is not None else run_id

    def _finalize_cron_run(
        self,
        run_id: str | None,
        status: str,
        *,
        error: str | None = None,
    ) -> None:
        if not run_id or finalize_cron_run is None:
            return
        try:
            finalize_cron_run(
                self.tool_context.workspace_root,
                run_id,
                status,  # type: ignore[arg-type]
                error=error,
            )
        except Exception:  # nosec B110
            pass

    def _finalize_cron_task(
        self,
        task_id: str,
        status: str = "completed",
        *,
        error: str | None = None,
    ) -> None:
        active = getattr(self, "_cron_active_tasks", None)
        if active is None:
            return
        run_id = active.pop(task_id, None)
        self._finalize_cron_run(run_id, status, error=error)

    def _queued_count(self) -> int:
        with self._queued_prompts_lock:
            return len(self._queued_prompts) + len(self._cron_queued_prompts)

    def clear_pending_turn_buffers(self) -> None:
        """Reset per-turn transient UI buffers at the end of each chat() turn.

        Called from the main ``chat()`` loop immediately after the
        :class:`LiveStatus` context manager has torn down (so the spinner
        row is no longer drawing) and before ``_stats_turns`` is
        incremented. At this point the REPL is single-threaded between
        turns, so the deque ``.clear()`` calls are safe without locks.

        Cleared here:
        * ``_thinking_chunks`` — streaming ``Thinking…`` text appended while
          the engine ran. The spinner is gone, so the buffer is dead
          memory. Bounded by ``deque(maxlen=1000)`` regardless, but
          per-turn clearing keeps the working set minimal (the
          WSL2 3.8 GB OOM repro was driven by this buffer).

        NOT cleared here (and intentionally so):
        * ``_queued_prompts`` — these are *user input* the LiveStatus
          spinner captured while the engine was running (the
          "type while it's still thinking" affordance, ported from
          the TS Ink reference). They must survive the turn boundary
          so the outer ``run()`` loop can drain them via
          ``_pop_queued_prompt()`` on the next iteration. Wiping them
          here would silently drop whatever the user typed during the
          turn. The ``deque(maxlen=100)`` is the actual memory cap; if
          the user truly floods the spinner, the oldest entry is
          dropped FIFO, not the whole queue.
        * ``_expandable_blocks`` — its ``maxlen=20`` already bounds it,
          and clearing would defeat the ``ctrl+o`` re-expand feature
          that replays the most recent block.
        """
        self._thinking_chunks.clear()

    @property
    def _p(self) -> "REPLPalette":
        """Convenience access to the active REPL palette."""
        return self._repl_palette

    def _status_message(self) -> str:
        """Spinner status text. Includes queued-prompt count when non-zero."""

        n = self._queued_count()
        if n == 0:
            return "Thinking…"
        return f"Thinking… ({n} queued)"

    def _safe_input(self, prompt: str) -> str:
        """Read a line from the user.

        Tries ``prompt_toolkit.prompt`` first because it cooperates with
        :func:`prompt_toolkit.patch_stdout.patch_stdout`. Falls back to
        bare ``input()`` if prompt_toolkit isn't available or the runtime
        can't open a TTY (e.g. piped stdin).

        If a :class:`~src.repl.live_status.LiveStatus` is currently
        mounted (we're inside ``chat()``), pause it for the duration of
        the read. Two prompt_toolkit Applications cannot share a TTY —
        without pausing, the spinner row keeps redrawing and shreds the
        user's keystrokes.
        """

        live = self._active_live_status

        def _do_read() -> str:
            if _HAS_PROMPT_TOOLKIT:
                try:
                    from prompt_toolkit import prompt as pt_prompt

                    return pt_prompt(prompt)
                except Exception:  # nosec B110
                    pass
            return input(prompt)

        if live is not None:
            with live.paused():
                return _do_read()
        return _do_read()

    # ------------------------------------------------------------------
    # ctrl+o expansion (truncated tool-result blocks)
    # ------------------------------------------------------------------
    def _stash_expandable(self, label: str, content: str) -> None:
        """Record a truncated block so ``ctrl+o`` can re-print its
        full content. Bounded by ``self._expandable_blocks`` ``maxlen``.
        """

        if not content:
            return
        self._expandable_blocks.append((label, content))

    def _do_expand_last(self) -> None:
        """Print the most recently stashed truncated block in full.

        Invoked via ``prompt_toolkit.application.run_in_terminal`` from
        the ``ctrl+o`` keybindings on both the idle prompt and the
        ``LiveStatus`` live region, so the print doesn't fight either
        Application's redraw.
        """

        # First, expand any stashed thinking content
        self._expand_thinking()

        if not self._expandable_blocks:
            return
        label, content = self._expandable_blocks[-1]
        lines = content.split("\n")
        # Trim the trailing empty element produced by a terminator newline
        # so we don't render a phantom blank line at the end.
        if content.endswith("\n") and lines and lines[-1] == "":
            lines = lines[:-1]
        self.console.print(f"  [dim]── Expanded {label} ──[/dim]", highlight=False)
        for i, line in enumerate(lines, start=1):
            # markup=False / highlight=False so a stray ``[`` or ``$`` in
            # the file content can't be interpreted as Rich markup or a
            # syntax token.
            self.console.print(
                f"     {i:>3}  {line}",
                markup=False,
                highlight=False,
                soft_wrap=True,
            )
        self.console.print("  [dim]── End ──[/dim]", highlight=False)

    def _expand_thinking(self) -> None:
        """Print stashed thinking content when user presses ctrl+o."""
        if not self._thinking_chunks:
            return
        thinking_text = "".join(self._thinking_chunks)
        lines = thinking_text.split("\n")
        if thinking_text.endswith("\n") and lines and lines[-1] == "":
            lines = lines[:-1]
        self.console.print("  [dim]── Expanded thinking ──[/dim]", highlight=False)
        for i, line in enumerate(lines, start=1):
            self.console.print(
                f"     {i:>3}  {line}",
                markup=False,
                highlight=False,
                soft_wrap=True,
            )
        self.console.print("  [dim]── End ──[/dim]", highlight=False)
        self._thinking_chunks.clear()

    def _shorten_path_text(self, text: str) -> str:
        root = str(self.tool_context.workspace_root)
        cwd = str(self.tool_context.cwd or self.tool_context.workspace_root)
        for base in (cwd, root):
            prefix = base.rstrip("/") + "/"
            if text.startswith(prefix):
                return "./" + text[len(prefix) :]
            text = text.replace(prefix, "")
        return text

    # ------------------------------------------------------------------
    # Task widget (coalesced Task* / TodoWrite snapshot)
    # ------------------------------------------------------------------
    #
    # Mirrors ``typescript/src/components/TaskListV2.tsx``: instead of
    # printing one bullet per ``TaskCreate``/``TaskUpdate`` call, we wait
    # until a run of task-management calls is finished and then render
    # the current task-state once.

    def _render_task_snapshot(self) -> None:
        """Print a compact snapshot of the current task / todo list."""
        # Prefer V2 tasks (interactive mode); fall back to V1 todos.
        tasks = self._collect_task_entries()
        if not tasks:
            return

        def _sort_key(entry: dict[str, Any]) -> tuple[int, str]:
            try:
                return (0, f"{int(entry['id']):08d}")
            except (TypeError, ValueError):
                return (1, str(entry.get("id", "")))

        sorted_tasks = sorted(tasks, key=_sort_key)

        completed = sum(1 for t in sorted_tasks if t["status"] == "completed")
        in_progress = sum(1 for t in sorted_tasks if t["status"] == "in_progress")
        pending = len(sorted_tasks) - completed - in_progress

        parts = [f"[bold]{completed}[/bold] done"]
        if in_progress > 0:
            parts.append(f"[bold]{in_progress}[/bold] in progress")
        parts.append(f"[bold]{pending}[/bold] open")
        header = (
            f"[success]●[/success] [bold][info]Tasks[/info][/bold] "
            f"[dim]([bold]{len(sorted_tasks)}[/bold] total: {', '.join(parts)})[/dim]"
        )
        self.console.print(header)

        unresolved = {t["id"] for t in sorted_tasks if t["status"] != "completed"}

        for task in sorted_tasks:
            status = task["status"]
            subject = str(task.get("subject") or "")
            if status == "completed":
                icon, style, subject_style = "✓", "green", "dim strike"
            elif status == "in_progress":
                icon, style, subject_style = "◼", "cyan", "bold"
            else:
                icon, style, subject_style = "◻", "dim", ""

            blocked_by = [bid for bid in (task.get("blockedBy") or []) if bid in unresolved]
            owner = task.get("owner")
            suffix_parts: list[str] = []
            if owner:
                suffix_parts.append(f"[dim] (@{owner})[/dim]")
            if blocked_by:
                blockers = ", ".join(f"#{bid}" for bid in sorted(blocked_by))
                suffix_parts.append(f"[dim] ▸ blocked by {blockers}[/dim]")
            suffix = "".join(suffix_parts)

            subject_markup = f"[{subject_style}]{subject}[/{subject_style}]" if subject_style else subject
            self.console.print(f"  [{style}]{icon}[/{style}] {subject_markup}{suffix}")

    def _collect_task_entries(self) -> list[dict[str, Any]]:
        """Return a normalised list of task dicts from the tool context.

        Uses V2 ``tasks`` if populated, otherwise falls back to the V1
        ``todos`` list written by ``TodoWrite``. Both are coalesced into
        the same shape: ``{id, status, subject, owner?, blockedBy?}``.
        """
        entries: list[dict[str, Any]] = []
        v2 = getattr(self.tool_context, "tasks", None) or {}
        if isinstance(v2, dict) and v2:
            for tid, t in v2.items():
                if not isinstance(t, dict):
                    continue
                entries.append(
                    {
                        "id": str(t.get("id", tid)),
                        "status": t.get("status", "pending"),
                        "subject": t.get("subject", ""),
                        "owner": t.get("owner"),
                        "blockedBy": list(t.get("blockedBy") or []),
                    }
                )
            return entries

        todos = getattr(self.tool_context, "todos", None) or []
        for td in todos:
            if not isinstance(td, dict):
                continue
            entries.append(
                {
                    "id": str(td.get("id", "")),
                    "status": td.get("status", "pending"),
                    "subject": td.get("content") or td.get("activeForm") or "",
                    "owner": None,
                    "blockedBy": [],
                }
            )
        return entries

    def _display_cwd(self) -> str:
        cwd = str(Path.cwd())
        home = str(Path.home())
        if cwd.startswith(home):
            return cwd.replace(home, "~", 1)
        return cwd

    def _truncate_middle(self, text: str, limit: int) -> str:
        if limit <= 0 or len(text) <= limit:
            return text
        if limit <= 3:
            return text[:limit]
        head = max(1, (limit - 1) // 2)
        tail = max(1, limit - head - 1)
        return f"{text[:head]}…{text[-tail:]}"

    def _handoff_to_textual_tui(self) -> None:
        """Switch from the Rich REPL into the Textual TUI for this session.

        Runs the Textual app inline. When the user quits the TUI (``Ctrl+D``
        or ``/exit`` inside it), control returns here and the caller's REPL
        loop resumes with the same session, provider, tool registry and
        tool context — so conversation history is preserved across the
        handoff.
        """
        try:
            from src.tui.app import ClawCodexTUI
        except Exception as exc:
            self.console.print(
                f"[error]Textual TUI is unavailable: {exc}[/error]\n"
                "[dim]Install it with `pip install 'textual>=0.79'`.[/dim]"
            )
            return

        if self.tool_context is None:
            self.console.print(
                "[error]TUI requires an API key to function.[/error]\n"
                "[dim]Use [bold]/login[/bold] to configure, or set [info]ANTHROPIC_API_KEY[/info] env var, then restart.[/dim]"
            )
            return

        self.console.print(
            "[dim]Entering Textual TUI. Press Ctrl+B to exit to shell, or /exit / Ctrl+D to return to CLI.[/dim]"
        )
        app = ClawCodexTUI(
            provider=self.provider,
            provider_name=self.provider_name,
            workspace_root=self.tool_context.workspace_root,
            tool_registry=self.tool_registry,
            tool_context=self.tool_context,
            session=self.session,
            stream=True,
            runtime_context=getattr(self, "runtime_context", None),
            replay_exit_snapshot_from_start=False,
        )
        result = None
        try:
            result = app.run()
        except KeyboardInterrupt:
            pass
        except Exception as exc:
            self.console.print(f"[error]TUI exited with error: {exc}[/error]")
        finally:
            # Dump the transcript we captured right before exit so the
            # conversation stays in the host's scrollback — matching
            # ink's non-fullscreen behaviour.
            snapshot = getattr(app, "exit_snapshot", None) or []
            for piece in snapshot:
                try:
                    self.console.print(piece)
                except Exception:  # nosec B110
                    continue

            # Ctrl+B → background exit (agent running in background)
            if isinstance(result, tuple) and result[0] == "__BACKGROUND_EXIT__":
                session_id = result[1] if len(result) > 1 else ""
                has_bg_agent = result[2] if len(result) > 2 else False
                if has_bg_agent:
                    self.console.print("\n  [bold][success]Agent is running in background[/success][/bold]")
                else:
                    if session_id:
                        self.console.print(f"\n  [bold][warning]Session {session_id} saved.[/warning][/bold]")
                    else:
                        self.console.print("\n  [warning]Session saved.[/warning]")
                self.console.print("[dim]Exiting clawcodex...[/dim]")
                raise SystemExit(0)

            # Ctrl+B → full exit to terminal shell (not back to CLI)
            if isinstance(result, tuple) and result[0] == "__FULL_EXIT__":
                session_id = result[1] if len(result) > 1 else ""
                if session_id:
                    self.console.print(
                        f"\n  [bold][warning]Session {session_id} saved.[/warning][/bold] Resume with:\n"
                        f"    [info]clawcodex --tui --resume {session_id}[/info]"
                    )
                else:
                    self.console.print("\n  [dim]Session saved.[/dim]")
                self.console.print("[dim]Exiting clawcodex...[/dim]")
                raise SystemExit(0)

            self.console.print("[dim]Returned from Textual TUI.[/dim]")

    def _sync_conversation_from_transcript(self, session_id: str) -> None:
        """Sync conversation from JSONL transcript to get full history.

        The .json session file is a snapshot saved at fork time and doesn't
        include background agent output. The JSONL transcript has the complete
        history and is used by TailFollower in TUI --resume mode.

        F-49 Phase 0.4.2: if Session.resume() already populated messages
        (via the core Phase 0.4.1 fix in session.py), this method is a
        quick-return no-op — kept as a defensive double-check.
        """
        # F-49 Phase 0.4.2: quick-return if messages already populated
        # by Session.resume()'s JSONL back-fill (Phase 0.4.1).
        if self.session.conversation.messages:
            return

        try:
            from src.services.session_storage import SessionStorage
            from clawcodex_ext.types.messages import message_from_dict

            storage = SessionStorage(session_id=session_id)
            entries = storage.read_transcript()

            if not entries:
                return

            # Rebuild message list from transcript
            messages = []
            for entry in entries:
                if entry.get("role") == "system" and entry.get("content") == "__background_complete__":
                    continue  # Skip completion marker
                try:
                    msg = message_from_dict(entry)
                    messages.append(msg)
                except Exception:  # nosec B110
                    pass

            if messages:
                self.session.conversation.messages = messages
        except Exception:  # nosec B110
            pass  # Best-effort, don't fail resume

    def _warn_if_background_runner_active(self, session_id: str) -> None:
        """Detect a still-running background agent for ``session_id`` and warn the user.

        When the user hits Ctrl+B a forked child continues the agent loop
        headlessly, writing to the JSONL transcript as it goes. If the
        user runs ``clawcodex --resume <sid>`` while that child is still
        active, the transcript only contains what the child has written
        *so far* — later turns land after this resume finishes loading.

        This method reads ``.background-runner.json`` (via
        :func:`get_background_runner_status`, which also corrects stale
        ``running`` markers whose PID has died) and, when the child is
        genuinely still alive, prints a clear notice telling the user:

        * the background agent (pid N) is still running,
        * the history shown now is partial,
        * they should re-run ``--resume <sid>`` after it finishes to see
          the complete output.

        Best-effort and silent on any failure — this must never block
        resume. Mirrors the defensive style of
        :meth:`_sync_conversation_from_transcript`.
        """
        try:
            from src.agent.background_runner import get_background_runner_status

            info = get_background_runner_status(session_id)
        except Exception:  # nosec B110
            return

        if not info or info.get("status") != "running":
            return

        pid = info.get("pid")
        pid_str = f" (pid {pid})" if pid is not None else ""
        self.console.print(f"\n[warning]⏎ Background agent{pid_str} is still running for this session.[/warning]")
        self.console.print(
            "[dim]The history shown below is partial — it only reflects what the "
            "background agent has produced so far.[/dim]"
        )
        self.console.print(
            f"[dim]To see the complete output once it finishes, exit and re-run:[/dim]\n"
            f"  [info]clawcodex --resume {session_id}[/info]"
        )

    def _replay_resume_history(self) -> None:
        """Replay full conversation history on resume, rendering identically to live chat.

        Produces the same visual output the user saw before exiting:
        * User messages with ``❯`` prefix
        * Assistant text rendered as Rich Markdown
        * Tool calls as ``● ToolName(args)`` headers
        * Tool results as ``  ⎿  result`` lines (with edit diff support)
        * Thinking blocks rendered if visible, or stashed for Ctrl+O expansion

        This mirrors the rendering logic in ``chat()``'s engine-stream handler
        but operates on the static message list rather than a live stream.
        """
        from clawcodex_ext.types.content_blocks import TextBlock, ToolUseBlock, ToolResultBlock
        from clawcodex_ext.types.content_blocks import ThinkingBlock, RedactedThinkingBlock
        from src.tool_system.renderers import summarize_tool_use

        self.console.print()
        self.console.print("[dim]─── resumed conversation history ───[/dim]")

        # Build a tool_use_id → (name, input) map so we can show
        # the right header above each ToolResultBlock.
        tool_use_map: dict[str, tuple[str, dict]] = {}
        tool_block_needs_leading_space = False

        for msg in self.session.conversation.messages:
            role = getattr(msg, "role", "") or ""
            content = getattr(msg, "content", None)

            if role == "system":
                # F-103: Render away_summary (Recapitulate) system messages
                # instead of skipping them, matching live-chat behaviour where
                # _print_local_command_text renders /recap output as Markdown.
                subtype = getattr(msg, "subtype", None) or ""
                if subtype == "away_summary":
                    try:
                        from clawcodex_ext.away_summary.messages import (
                            format_away_summary_for_display,
                        )

                        display = format_away_summary_for_display(getattr(msg, "content", "") or "")
                    except Exception:
                        display = str(getattr(msg, "content", "") or "")
                    self.console.print()
                    self.console.print(Markdown(display))
                elif subtype == "intent_forecast":
                    display = str(getattr(msg, "content", "") or "")
                    if display:
                        self.console.print()
                        self.console.print(Markdown(display))
                elif subtype in {
                    "goal_set",
                    "goal_cleared",
                    "goal_evaluation",
                    "goal_achieved",
                    "goal_evaluator_error",
                }:
                    display = str(getattr(msg, "content", "") or "")
                    if display:
                        self.console.print()
                        style = (
                            "error"
                            if subtype == "goal_evaluator_error"
                            else "success"
                            if subtype == "goal_achieved"
                            else "dim"
                        )
                        self.console.print(f"[{style}]{escape(display)}[/{style}]")
                continue

            if role == "user":
                if isinstance(content, list):
                    for block in content:
                        if isinstance(block, ToolResultBlock):
                            # Print result line
                            if block.is_error:
                                err_text = block.content if isinstance(block.content, str) else str(block.content)
                                self.console.print(f"[error]  ⎿  {escape(err_text) if err_text else 'Error'}[/error]")
                            else:
                                preview = self._format_tool_result_preview(
                                    block,
                                    tool_use_map.get(block.tool_use_id),
                                )
                                if isinstance(preview, str):
                                    if "\n" in preview:
                                        first, *rest = preview.split("\n")
                                        self.console.print(f"[dim]  ⎿  {first}[/dim]")
                                        for ln in rest:
                                            self.console.print(f"[dim]     {ln}[/dim]")
                                    else:
                                        self.console.print(f"[dim]  ⎿  {preview}[/dim]")
                                else:
                                    self.console.print("[dim]  ⎿  [/dim]", end="")
                                    self.console.print(preview)
                            tool_block_needs_leading_space = True
                        elif isinstance(block, TextBlock):
                            text = block.text or ""
                            if text:
                                self._echo_user_input(text)
                elif isinstance(content, str) and content:
                    self._echo_user_input(content)
                continue

            if role == "assistant":
                # Use the agent type from tool_context (set by @agent-mention
                # or agent config) just like live chat does — fall back to
                # "Assistant" for the default case.
                _agent_label = getattr(getattr(self, "tool_context", None), "agent_type", None) or "Assistant"
                # Only print the agent label when the message has visible
                # text content.  Tool-only messages (ToolUseBlock without
                # any TextBlock) suppress the label — the deferred
                # ``● ToolName(args)`` header already identifies the call
                # when its result arrives.  Without this check every
                # tool-only assistant message would produce a lonely
                # "Assistant" line with nothing underneath, stacking up
                # as visually confusing blank labels.
                _has_text = False
                if isinstance(content, list):
                    _has_text = any(
                        isinstance(b, TextBlock) and bool((b.text or "").strip()) and b.text != "[No content]"
                        for b in content
                    )
                elif isinstance(content, str):
                    _has_text = bool(content.strip())
                if _has_text:
                    self.console.print(f"\n[bold]{_agent_label}[/bold]")
                if isinstance(content, list):
                    for block in content:
                        if isinstance(block, TextBlock) and (block.text or "").strip():
                            # Suppress NO_CONTENT_MESSAGE placeholder that
                            # create_assistant_message injects for empty
                            # responses — matches live-chat behaviour where
                            # _run_query skips empty TextBlocks.
                            if block.text == "[No content]":
                                continue
                            self.console.print(Markdown(block.text))
                        elif isinstance(block, ToolUseBlock):
                            tool_use_map[block.id] = (block.name, block.input)
                            summary = summarize_tool_use(block.name, block.input)
                            if isinstance(summary, str) and summary:
                                summary = self._shorten_path_text(summary)
                            if summary:
                                call_args = f"[dim]([/dim]{escape(summary)}[dim])[/dim]"
                            else:
                                call_args = ""
                            # Some persisted transcripts have no matching
                            # ToolResultBlock. Print the call at its original
                            # position so a later recap cannot overtake it.
                            if tool_block_needs_leading_space:
                                self.console.print()
                            self.console.print(
                                f"[success]●[/success] [bold][info]{block.name}[/info][/bold]"
                                + (f" {call_args}" if call_args else "")
                            )
                            tool_block_needs_leading_space = True
                        elif isinstance(block, ThinkingBlock):
                            # Replay thinking blocks matching live-chat behaviour:
                            # visible → print directly; hidden → stash for Ctrl+O.
                            thinking_text = block.thinking or ""
                            if thinking_text:
                                if self._thinking_visible:
                                    self.console.print(
                                        thinking_text,
                                        end="",
                                        markup=False,
                                        highlight=False,
                                        soft_wrap=True,
                                    )
                                else:
                                    self._thinking_chunks.append(thinking_text)
                        elif isinstance(block, RedactedThinkingBlock):
                            # Redacted thinking is never expandable — skip silently
                            # in replay, same as live chat (no visible output).
                            pass
                elif isinstance(content, str) and content:
                    self.console.print(Markdown(content))
                continue

        self.console.print("\n[dim]─── end of history ───[/dim]")

    def _flatten_message_content(self, content: Any) -> str:
        """Normalise Message.content (string or block list) to text."""
        if content is None:
            return ""
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: list[str] = []
            for item in content:
                # Handle dataclass blocks (TextBlock, ToolUseBlock, ToolResultBlock, etc.)
                item_type = getattr(item, "type", None) if hasattr(item, "type") else None
                if item_type is None and isinstance(item, dict):
                    item_type = item.get("type")

                if item_type == "text":
                    text = getattr(item, "text", None) or (item.get("text") if isinstance(item, dict) else "")
                    if text:
                        parts.append(text)
                elif item_type == "tool_use":
                    name = getattr(item, "name", None) or (item.get("name") if isinstance(item, dict) else "")
                    if not name and isinstance(item, dict):
                        name = item.get("input", {}).get("description", "")
                    if name:
                        parts.append(f"[tool:{name}]")
                elif item_type == "tool_result":
                    result = getattr(item, "content", None) or (item.get("content") if isinstance(item, dict) else "")
                    if result:
                        parts.append(str(result))
                elif item_type is None and isinstance(item, str):
                    parts.append(item)
            return "\n".join(p for p in parts if p).strip()
        return str(content)

    def _format_history_line(self, msg: Any) -> str | None:
        """Render one message for the /load history preview.

        Returns None for messages the snapshot should hide: meta /
        virtual system injections, the ``[No content]`` placeholder
        that ``create_assistant_message`` injects for empty
        responses, the ``[AWAY SUMMARY]`` system injection, and any
        other message whose flattened content is empty.
        """
        if getattr(msg, "isMeta", False) or getattr(msg, "isVirtual", False):
            return None
        # AWAY SUMMARY uses subtype="away_summary" with isMeta=False,
        # so the isMeta check above does not catch it. Mirror
        # clawcodex_ext/away_summary/fingerprint.py:81-83 detection.
        if msg.role == "system" and getattr(msg, "subtype", None) == "away_summary":
            return None
        text = self._flatten_message_content(msg.content).strip()
        if not text:
            return None
        # Same convention as core.py:2809-2814 and tui/app.py:1283-1286:
        # hide the no-content placeholder injected for empty responses.
        if text == NO_CONTENT_MESSAGE:
            return None
        # Belt-and-braces: if subtype was lost on disk round-trip,
        # content-prefix still catches the AWAY SUMMARY injection.
        if text.startswith("[AWAY SUMMARY]"):
            return None
        role_colors = {
            "user": "blue",
            "assistant": "green",
            "system": "magenta",
        }
        role_color = role_colors.get(msg.role, "yellow")
        preview = text[:100].replace("\n", " ")
        suffix = "..." if len(text) > 100 else ""
        return f"[{role_color}]{msg.role}[/{role_color}]: {preview}{suffix}"

    def _print_startup_header(self):
        from src import __version__

        display_path = self._display_cwd()
        provider_label = f"{self.provider_name.upper()} Provider"
        model_label = (self.provider.model if self.provider else None) or "N/A"

        if Panel is None or Group is None or Align is None or Table is None or Text is None or Columns is None:
            print(f"ClawCodex v{__version__}")
            print(f"{model_label} · {provider_label}")
            print(f"{display_path}\n")
            return

        width = getattr(self.console, "width", 80)
        content_width = max(28, min(width - 12, 72))
        table = Table.grid(padding=(0, 1))
        table.add_column(style="bright_black", justify="right", no_wrap=True)
        table.add_column(style="white", ratio=1)
        table.add_row(
            "Version",
            Text.assemble(
                ("ClawCodex", f"bold {self._p.text}"),
                ("  ", ""),
                (f"v{__version__}", f"bold {self._p.info}"),
            ),
        )
        table.add_row("Model", Text(model_label, style=f"bold {self._p.secondary}"))
        table.add_row("Provider", Text(provider_label, style=f"bold {self._p.success}"))
        table.add_row(
            "Workspace",
            Text(
                self._truncate_middle(display_path, content_width - 12),
                style=f"bold {self._p.primary}",
            ),
        )

        footer = Text("/help  •  /tools  •  /tui  •  /stream  •  /exit", style="dim")

        # F-97 telemetry notice — show when both stats collection and error
        # reporting are enabled.  Best-effort & swallowed on failure so a
        # misconfigured telemetry package never blocks REPL startup.
        try:
            from telemetry.config import load_config as _load_telemetry_cfg

            _tc = _load_telemetry_cfg()
            if _tc.enabled and _tc.reporting.reporting_enabled:
                telemetry_notice = Group(
                    Align.center(
                        Text(
                            "Telemetry: stats ✓ · error reporting ✓  — /telemetry to configure",
                            style="dim italic",
                        )
                    ),
                    Align.center(
                        Text(
                            "Collects usage data & error reports; may be uploaded periodically.",
                            style="dim italic",
                        )
                    ),
                )
                body = Group(table, telemetry_notice, Text(""), Align.center(footer))
            else:
                body = Group(table, Text(""), Align.center(footer))
        except Exception:  # nosec B110
            body = Group(table, Text(""), Align.center(footer))
        header = Panel(
            body,
            border_style="bright_black",
            title="[bold][info] CLAWCODEX [/info][/bold]",
            subtitle="[dim]interactive terminal[/dim]",
            padding=(1, 2),
        )
        self.console.print(header)

        # Show coordinator mode badge when active
        from src.coordinator.mode import is_coordinator_mode

        if is_coordinator_mode():
            self.console.print(
                "[bold][warning]  ⚡ Coordinator Mode ACTIVE[/warning][/bold]  "
                "[dim]— Agent / SendMessage / TaskStop only[/dim]"
            )
            self.console.print()

        self.console.print()

    def run(self):
        _load_heavy_runtime()
        """Run the REPL."""
        self._print_startup_header()

        if getattr(self, "_api_key_missing", False):
            self.console.print("[warning]No API key configured — REPL is in read-only mode.[/warning]")
            self.console.print(
                "Use [bold]/login[/bold] to configure, or set [info]ANTHROPIC_API_KEY[/info] env var, then restart."
            )
            self.console.print("Type [bold]/exit[/bold] to quit.\n")

        # Print conversation history when resuming a session
        # — renders identically to live chat output so the user sees
        # the same transcript they would have seen before exiting.
        resumed = getattr(self, "_resume_session_id", None)
        if resumed and self.session.conversation.messages:
            self._replay_resume_history()

        try:
            from clawcodex_ext.debug.agent_debug import emit_agent_debug_marker

            emit_agent_debug_marker(
                "repl.ready",
                {
                    "session_id": _session_id_from_session(self.session),
                    "surface": "repl",
                    "stream": bool(self.stream),
                },
            )
        except Exception:  # nosec B110
            pass

        # Phase B-2 wake: spin up a long-lived asyncio loop on the
        # main thread so the in-loop cron watcher can call app.exit()
        # from within the event loop. The loop is closed on exit.
        self._cron_loop = asyncio.new_event_loop()
        # IM gateway opt-in: connect + register now that the loop is up
        # (installed by install_repl_extensions via _gateway_init).
        im_init = getattr(self, "_gateway_init", None)
        if im_init is not None:
            try:
                self._cron_loop.run_until_complete(im_init(self._cron_loop))
            except Exception:  # nosec B110
                pass
        try:
            self._run_main_loop()
        finally:
            # tear down the IM gateway client if it was connected
            im_client = getattr(self, "_gateway_client", None)
            if im_client is not None:
                try:
                    self._cron_loop.run_until_complete(im_client.close())
                except Exception:  # nosec B110
                    pass
            try:
                self._cron_loop.close()
            except Exception:  # nosec B110
                pass

    def _run_main_loop(self) -> None:
        """Inner body of the REPL main loop. Extracted so we can
        own the asyncio event loop lifecycle cleanly in ``run``.
        """
        import asyncio

        while True:
            try:
                user_input = None
                self._refresh_completer()
                self._drain_cron_outbox()
                self._drain_background_outputs()
                result = self._pop_queued_prompt()
                if result is not None:
                    queued, source = result
                    # Echo queued submissions with a dim background so
                    # they read as a discrete user-message block when
                    # they land in scrollback alongside the agent's
                    # transcript output.
                    # For cron prompts, only display the header line (first line)
                    # to avoid showing the prelude text to the user.
                    if source == "cron":
                        first_line = queued.split("\n")[0]
                        self._echo_user_input(first_line)
                    else:
                        self._echo_user_input(queued)
                    user_input = queued
                else:
                    # Blank line of breathing room between the previous
                    # transcript and the next prompt. The bg highlight
                    # on the prompt itself (PromptSession ``style``)
                    # provides the visual cue that the next row is
                    # user input — no divider needed.
                    self.console.print()
                    # The prompt session is configured with ``multiline=True``
                    # up front so that newlines (via Shift+Enter / Meta+Enter
                    # / ``\`` + Enter) can live in the buffer. Plain Enter
                    # still submits via our custom ``c-m`` binding.
                    if getattr(self, "_api_key_missing", False):
                        user_input = input("❯ ")
                    else:
                        # Phase B-2 wake: run prompt_async with a
                        # concurrent outbox watcher in the same event loop.
                        # The watcher checks for cron events every 1 second
                        # and calls app.exit(_CRON_WAKE) from within the loop
                        # when events are found. This avoids all cross-thread
                        # issues.
                        with _pt_patch_stdout(raw=True):
                            self._current_prompt_task = self._cron_loop.create_task(self._prompt_with_cron_watch())
                            try:
                                user_input = self._cron_loop.run_until_complete(self._current_prompt_task)
                            finally:
                                self._current_prompt_task = None

                if user_input is None:
                    # app.exit() was called (e.g., Ctrl+B)
                    try:
                        self.session.save()
                    except Exception:  # nosec B110
                        pass
                    self._print_resume_hint()
                    self.console.print("\n[primary]Goodbye![/primary]")
                    break

                if user_input is _CRON_WAKE:
                    # app.exit(_CRON_WAKE) returned normally.
                    # Drain the outbox and re-prompt.
                    self._drain_cron_outbox()
                    continue

                if not user_input.strip():
                    continue

                if user_input.startswith("/"):
                    controller = getattr(self, "_intent_forecast_controller", None)
                    if controller is not None:
                        controller.on_user_interaction("slash")
                    self.handle_command(user_input)
                    continue

                if user_input.startswith("!"):
                    controller = getattr(self, "_intent_forecast_controller", None)
                    if controller is not None:
                        controller.on_user_interaction("bash")
                    # Bash mode: direct execution, no agent turn.
                    # Feeds the bash input + output into the conversation
                    # (so the model sees what happened on its next turn).
                    from src.services.bash_mode import run_bash_mode_command

                    command = user_input[1:]
                    self._echo_user_input(f"! {command}")
                    outcome = run_bash_mode_command(command, self.tool_context)

                    # Append conversation texts so the model sees them
                    # on the next agent turn.
                    for conv_text in outcome.conversation_texts:
                        self.session.conversation.add_user_message(conv_text)

                    # Display result in the console.
                    if outcome.ok:
                        if outcome.stdout:
                            self.console.print(outcome.stdout, markup=False, highlight=False)
                        if outcome.stderr:
                            self.console.print(
                                f"[dim]{outcome.stderr}[/dim]",
                                markup=False,
                                highlight=False,
                            )
                    else:
                        self.console.print(
                            f"[error]! {command}: {outcome.error or outcome.stderr or 'Unknown error'}[/error]",
                            markup=False,
                            highlight=False,
                        )
                    continue

                if user_input.startswith("#"):
                    # C9 memory-note append: persist the note to
                    # ~/.claude/CLAUDE.md and show an acknowledgement,
                    # no agent turn.
                    from src.services.memory_append import (
                        append_memory_note,
                        pick_saving_message,
                    )
                    from pathlib import Path

                    note = user_input[1:]
                    ok = append_memory_note(str(Path.home() / ".claude" / "CLAUDE.md"), note)
                    if ok:
                        self.console.print(f"[success]{pick_saving_message()}[/success]")
                    else:
                        self.console.print("[error]Failed to save memory note[/error]")
                    continue

                _cron_task_id = self._extract_cron_task_id(user_input)
                if _cron_task_id:
                    self._claim_cron_task(_cron_task_id)
                try:
                    _chat_success = self.chat(user_input)
                except KeyboardInterrupt:
                    if _cron_task_id:
                        self._finalize_cron_task(_cron_task_id, "cancelled")
                    raise
                except SystemExit:
                    if _cron_task_id:
                        self._finalize_cron_task(_cron_task_id, "cancelled")
                    raise
                except Exception as exc:
                    if _cron_task_id:
                        self._finalize_cron_task(
                            _cron_task_id,
                            "failed",
                            error=f"{type(exc).__name__}: {exc}",
                        )
                    raise
                if _cron_task_id:
                    if _chat_success is False:
                        self._finalize_cron_task(
                            _cron_task_id,
                            "failed",
                            error="Scheduled task execution returned without completing.",
                        )
                    else:
                        self._finalize_cron_task(_cron_task_id)

            except asyncio.CancelledError:
                # Safety net: app.exit() returns normally, so this path
                # should rarely be hit. Kept for edge cases (e.g. external
                # cancellation, prompt_toolkit internals).
                self._drain_cron_outbox()
                continue
            except KeyboardInterrupt:
                try:
                    self.session.save()
                except Exception:  # nosec B110
                    pass
                self.console.print("\n[warning]Interrupted. Type /exit or press Ctrl+D to quit.[/warning]")
                continue
            except EOFError:
                try:
                    self.session.save()
                except Exception:  # nosec B110
                    pass
                self._print_resume_hint()
                self.console.print("\n[primary]Goodbye![/primary]")
                break

    def _send_im_command_feedback(self, command: str, *, success: bool) -> None:
        raw = str(command or "").strip()
        if not raw.startswith("/") or raw == "/":
            return
        im_reply = getattr(self, "_im_reply_controller", None)
        send_feedback = getattr(im_reply, "send_command_feedback", None)
        if not callable(send_feedback):
            return
        try:
            send_feedback(raw, success=success)
        except Exception:  # noqa: BLE001
            logger.debug("IM command feedback failed", exc_info=True)

    def handle_command(self, command: str):
        """Handle slash commands."""
        _load_heavy_runtime()
        # Lazy command-system build — fires on first ``/`` keystroke. The
        # cost (~0.8s of command_system import + registration) was previously
        # paid up front in ``__init__``; moving it here saves that on every
        # session that opens with non-slash input.
        self._ensure_command_system()
        success = False
        try:
            result = self._handle_command(command)
            success = True
            return result
        finally:
            self._send_im_command_feedback(command, success=success)

    def _handle_command(self, command: str):
        raw = command.strip()
        if raw == "/":
            self._show_slash_palette()
            return

        # ── REPL-native handlers — must run BEFORE the palette check ───
        if raw.startswith("/") and " " not in raw:
            candidate = raw[1:].lower()
            if candidate == "diff":
                self._handle_repl_diff()
                return
            if candidate == "mcp":
                self._handle_repl_mcp()
                return
            if candidate == "tasks":
                self._handle_repl_tasks()
                return
            if candidate == "rewind":
                self._handle_repl_rewind()
                return
            if candidate == "effort":
                self._handle_repl_effort("")
                return
            if candidate == "history":
                self._handle_repl_history()
                return
            if candidate == "idle":
                self._handle_repl_idle()
                return

        if raw.startswith("/") and " " not in raw and raw.lower() not in (c.lower() for c in self._built_in_commands):
            query = raw[1:]
            if query:
                self._show_slash_palette(query=query)
                return

        # ── New-command-system try path ──────────────────────────────────
        if raw.startswith("/"):
            parts = raw[1:].split(maxsplit=1)
            cmd_name = parts[0].lower()
            args_raw = parts[1] if len(parts) > 1 else ""

            # Check if this command exists in the new command system
            # but skip the ones we handle specially
            # Note: /context, /compact, /skill need special handling, don't route through new system
            # /init is handled via new command system (PromptCommand) so it's NOT in special_commands
            special_commands = {
                "exit",
                "quit",
                "q",
                "help",
                "tools",
                "tool",
                "save",
                "load",
                "stream",
                "render-last",
                "skill",
                "context",
                "compact",  # These need special handling
                "permission",  # REPL-native permission mode command
                "tui",  # handoff to Textual TUI
                "resume",  # REPL-native session resume with browser
                # TUI-only commands — keep only the truly TUI-specific ones here
                "repl",
                "theme",
                # F-43 runtime commands: /provider and /model are routed via
                # the new command system (clawcodex_ext/cli/runtime_commands.py)
                # and work in both REPL and TUI; do NOT mark them TUI-only.
                "",
            }

            # Handle truly TUI-only commands (/repl, /theme)
            if cmd_name in ("repl", "theme"):
                self.console.print(f"[dim]/{cmd_name} is only available in the Textual TUI. Use /tui to switch.[/dim]")
                return

            # ── REPL-native implementations for Phase-2/3 commands ──────
            if cmd_name == "diff":
                self._handle_repl_diff()
                return
            if cmd_name == "mcp":
                self._handle_repl_mcp()
                return
            if cmd_name == "tasks":
                self._handle_repl_tasks()
                return
            if cmd_name == "rewind":
                self._handle_repl_rewind()
                return
            if cmd_name == "effort":
                self._handle_repl_effort(args_raw)
                return
            if cmd_name == "history":
                self._handle_repl_history()
                return
            if cmd_name == "idle":
                self._handle_repl_idle()
                return

            # Alias for the remaining downstream code that uses `args`
            args = args_raw

            # Handle /init through the new command system (PromptCommand path)
            if cmd_name == "init":
                # Use async path for PromptCommand
                try:
                    result = self._run_command_async_with_status(cmd_name, args)

                    if result.success:
                        self._handle_command_result(result)
                    elif result.error:
                        self.console.print(f"[error]{result.error}[/error]")
                except Exception as e:
                    self.console.print(f"[error]Error executing /init: {e}[/error]")
                return

            if cmd_name == "permission":
                self._handle_permission_command(args)
                return

            if cmd_name not in special_commands:
                # Try to execute via new command system
                # First try sync path for LocalCommand (faster)
                try:
                    handled, result_text = self._try_execute_new_command(cmd_name, args)
                    if handled:
                        if result_text:
                            self.console.print("\n" + result_text)
                        self.console.print()
                        return
                except Exception:  # nosec B110
                    pass

                # Use async path for PromptCommand / InteractiveCommand
                # Run in a new event loop since we're in a sync context
                try:
                    result = self._run_command_async_with_status(
                        cmd_name,
                        args,
                        status_message="Recapping..."
                        if cmd_name == "recap"
                        else "Answering..."
                        if cmd_name == "btw"
                        else None,
                    )

                    if result.success:
                        if self._handle_command_result(result):
                            return
                    elif self.command_registry.get(cmd_name) is not None:
                        # The command IS registered but its execution returned an
                        # error (e.g. an interactive command whose surface could
                        # not be opened). Surface that real error and stop —
                        # otherwise the request slides into the fallback chain
                        # below and is misreported as "Unknown command".
                        self._handle_command_result(result)
                        return
                except Exception:  # nosec B110
                    pass

        # Fall back to original command handling
        cmd = raw.lower()

        if cmd in ["/exit", "/quit", "/q"]:
            # Save session and print resume hint before exiting.
            try:
                self.session.save()
            except Exception:  # nosec B110
                pass
            self.console.print("[primary]Goodbye![/primary]")
            # Delegate to the centralised helper so the hint format matches
            # CCB's ``printResumeHint()`` and shares the process-wide
            # idempotency latch with the atexit cleanup.
            self._print_resume_hint()
            raise SystemExit(0)

        elif cmd == "/login":
            self.console.print(
                "[info]Use [bold]clawcodex login[/bold] in a separate terminal to configure your API key.[/info]"
            )
            self.console.print("[dim]Then restart clawcodex to use the REPL.[/dim]")

        elif cmd == "/tui":
            self._handoff_to_textual_tui()

        elif cmd == "/help":
            self.show_help()

        elif cmd == "/tools":
            names = [spec.name for spec in self.tool_registry.list_tools()]
            names.sort(key=str.lower)
            self.console.print("\n[bold]Available tools:[/bold]")
            for name in names:
                self.console.print(f"  - {name}")
            self.console.print()

        elif cmd.startswith("/tool"):
            parts = command.strip().split(maxsplit=2)
            if len(parts) < 2:
                self.console.print("[error]Usage: /tool <name> <json-input>[/error]")
                return
            name = parts[1]
            payload = {}
            if len(parts) == 3:
                try:
                    payload = json.loads(parts[2])
                except json.JSONDecodeError as e:
                    self.console.print(f"[error]Invalid JSON input: {e}[/error]")
                    return
            try:
                result = self.tool_registry.dispatch(ToolCall(name=name, input=payload), self.tool_context)
            except Exception as e:
                self.console.print(f"[error]Tool error: {e}[/error]")
                return
            self.console.print("\n[bold]Tool result:[/bold]")
            self.console.print(json.dumps(result.output, indent=2, ensure_ascii=False))
            self.console.print()

        elif cmd == "/clear":
            # Try new command system first, fall back to original
            try:
                handled, result_text = self._try_execute_new_command("clear", "")
                if handled:
                    if result_text:
                        self.console.print("\n[success]" + result_text + "[/success]")
                    return
                if result_text:
                    self.console.print(f"[error]{escape(result_text)}[/error]")
                    return
            except Exception as exc:
                self.console.print(f"[error]{escape(str(exc))}[/error]")
                return
            # Original implementation
            try:
                from clawcodex_ext.goal.service import clear_goal_for_context

                clear_goal_for_context(self.tool_context)
            except Exception as exc:
                self.console.print(f"[error]{escape(str(exc))}[/error]")
                return
            self.session.conversation.clear()
            self._engine_messages = []
            self.console.print("[success]Conversation cleared.[/success]")

        elif cmd == "/save":
            self.save_session()

        elif cmd == "/stream" or cmd.startswith("/stream "):
            parts = raw.split(maxsplit=1)
            if len(parts) == 1:
                status = "enabled" if self.stream else "disabled"
                self.console.print(f"[success]Stream mode {status}.[/success]")
                return

            action = parts[1].strip().lower()
            if action in {"on", "true", "1", "enable", "enabled"}:
                self.stream = True
            elif action in {"off", "false", "0", "disable", "disabled"}:
                self.stream = False
            elif action == "toggle":
                self.stream = not self.stream
            else:
                self.console.print("[error]Usage: /stream [on|off|toggle][/error]")
                return

            status = "enabled" if self.stream else "disabled"
            self.console.print(f"[success]Stream mode {status}.[/success]")

        elif cmd == "/render-last":
            rendered = self._render_last_assistant_message()
            if not rendered:
                self.console.print("[warning]No assistant response available to render.[/warning]")

        elif cmd.startswith("/load"):
            parts = command.strip().split(maxsplit=1)
            if len(parts) < 2:
                self.console.print("[error]Usage: /load <session-id>[/error]")
            else:
                session_id = parts[1]
                self.load_session(session_id)

        elif cmd.startswith("/resume"):
            parts = command.strip().split(maxsplit=1)
            if len(parts) >= 2 and parts[1].strip():
                # Session ID provided — load directly
                self.load_session(parts[1].strip())
            else:
                # No session ID — show interactive browser
                try:
                    from clawcodex_ext.repl.session_browser import (
                        browse_sessions_interactive,
                    )

                    selected_id = browse_sessions_interactive()
                    if selected_id:
                        self.load_session(selected_id)
                    else:
                        self.console.print("[dim]Session selection cancelled.[/dim]")
                except Exception:
                    # Fallback: list sessions as text
                    try:
                        from src.services.session_storage import SessionStorage

                        metas = SessionStorage.list_sessions(limit=50)
                        if not metas:
                            self.console.print("[yellow]No past sessions found.[/yellow]")
                        else:
                            self.console.print("\n[bold]Available sessions:[/bold]")
                            for i, m in enumerate(metas, 1):
                                sid = m.session_id[:12]
                                preview = getattr(m, "title", "") or getattr(m, "last_user_input", "") or ""
                                if preview:
                                    preview = preview[:60]
                                self.console.print(f"  {i:>3}. {sid}…  {preview}")
                            self.console.print(
                                "\n[dim]Use [bold]/resume <session-id>[/bold] "
                                "or [bold]/load <session-id>[/bold] "
                                "to restore a session.[/dim]"
                            )
                    except Exception:
                        self.console.print("[yellow]No past sessions found. Interactive browser unavailable.[/yellow]")

        elif cmd == "/skill":
            self._handle_skill_command()

        elif cmd == "/context":
            # Populate command context config for context analysis
            self.command_context.config["provider"] = self.provider
            self.command_context.config["model"] = self.provider.model
            self.command_context.config["tool_schemas"] = [
                tool_to_api_schema(spec) for spec in self.tool_registry.list_tools()
            ]
            self.command_context.config["system_prompt"] = ""
            # Try new command system
            try:
                handled, result_text = self._try_execute_new_command("context", "")
                if handled and result_text:
                    self.console.print(Markdown(result_text))
                    return
            except Exception:  # nosec B110
                pass
            self.console.print("[warning]/context analysis unavailable in this context.[/warning]")

        elif cmd == "/compact":
            # Populate command context config for compact
            self.command_context.config["provider"] = self.provider
            self.command_context.config["model"] = self.provider.model
            self.command_context.config["system_prompt"] = ""
            # Try new command system
            try:
                handled, result_text = self._try_execute_new_command("compact", "")
                if handled and result_text:
                    self.console.print("\n[success]" + result_text + "[/success]")
                    return
            except Exception:  # nosec B110
                pass
            # Simple fallback: just clear conversation
            self.session.conversation.clear()
            self._engine_messages = []
            self.console.print("[success]Conversation cleared.[/success]")

        elif cmd.startswith("/provider"):
            # Safety fallback: /provider may have failed through the new
            # command system path (F-43). Try direct sync execution.
            parts = raw.split(maxsplit=1)
            provider_args = parts[1] if len(parts) > 1 else ""
            try:
                handled, text = self._try_execute_new_command("provider", provider_args)
                if handled:
                    if text:
                        self.console.print("\n" + text)
                    self.console.print()
                else:
                    # Execution failed — show error text to the user
                    self.console.print(f"\n[error]{text or 'Unknown error'}[/error]\n")
            except Exception as exc:
                self.console.print(f"\n[error]Failed to execute /provider command: {exc}[/error]\n")

        elif cmd.startswith("/model"):
            # Safety fallback: same as /provider above.
            parts = raw.split(maxsplit=1)
            model_args = parts[1] if len(parts) > 1 else ""
            try:
                handled, text = self._try_execute_new_command("model", model_args)
                if handled:
                    if text:
                        self.console.print("\n" + text)
                    self.console.print()
                else:
                    self.console.print(f"\n[error]{text or 'Unknown error'}[/error]\n")
            except Exception as exc:
                self.console.print(f"\n[error]Failed to execute /model command: {exc}[/error]\n")

        else:
            if raw.startswith("/"):
                if self._try_run_skill_slash(raw):
                    return
            self.console.print(f"[error]Unknown command: {command}[/error]")
            self.console.print(
                "  [dim]Type [secondary]/[/secondary] to browse all available commands or [secondary]/help[/secondary] for details.[/dim]"
            )

    # ── REPL-native handlers for Phase-2/3 commands ────────────────────

    def _handle_repl_diff(self) -> None:
        """Show pending file diffs as formatted text."""
        files: list[tuple[str, str, str]] = []  # (path, patch, summary)
        # Collect diffs from conversation's last tool results
        for msg in reversed(self.session.conversation.messages):
            content = msg.content
            if isinstance(content, list):
                for block in content:
                    item_type = getattr(block, "type", None)
                    if item_type == "tool_result":
                        result_text = getattr(block, "content", None) or ""
                        if isinstance(result_text, str) and "patch" in result_text.lower():
                            # Try to extract file path + patch from structured edit result
                            result_data = getattr(block, "content", None)
                            if isinstance(result_data, str):
                                files.append(("(inline)", result_data, ""))
            if files:
                break  # Only look at the last tool result batch

        if not files:
            # Fall back: show git diff for workspace
            try:
                from src.utils.git import get_session_diff

                diff = get_session_diff(cwd=str(self.tool_context.workspace_root))
                if diff.files_changed:
                    self.console.print(
                        f"\n[bold][info]Pending changes[/info][/bold] [dim]({diff.files_changed} files, +{diff.insertions} -{diff.deletions})[/dim]"
                    )
                    self.console.print(
                        diff.patch[:4000] + ("\n[dim]… (truncated)[/dim]" if len(diff.patch) > 4000 else "")
                    )
                    self.console.print()
                    return
            except Exception:  # nosec B110
                pass
            self.console.print("[dim]No pending diffs to display.[/dim]")
            return

        self.console.print(f"\n[bold][info]Diff — {len(files)} file(s) changed[/info][/bold]")
        for path, patch, _summary in files[:10]:
            self.console.print(f"  [bold]{path}[/bold]")
            lines = patch.splitlines()
            for line in lines[:30]:
                if line.startswith("+"):
                    self.console.print(f"    [success]{line}[/success]")
                elif line.startswith("-"):
                    self.console.print(f"    [error]{line}[/error]")
                elif line.startswith("@@"):
                    self.console.print(f"    [info]{line}[/info]")
            if len(lines) > 30:
                self.console.print(f"    [dim]… {len(lines) - 30} more lines[/dim]")
        self.console.print()

    def _handle_repl_mcp(self) -> None:
        """List configured MCP servers."""
        try:
            from src.config import load_config

            cfg = load_config() or {}
            raw = cfg.get("mcp_servers") or cfg.get("mcpServers") or {}
        except Exception:
            raw = {}

        servers: list[dict[str, str]] = []
        if isinstance(raw, dict):
            for server_id, entry in raw.items():
                name = entry.get("name", server_id) if isinstance(entry, dict) else server_id
                status = entry.get("status", "disconnected") if isinstance(entry, dict) else "disconnected"
                tools = entry.get("tools", []) if isinstance(entry, dict) else []
                servers.append(
                    {
                        "id": str(server_id),
                        "name": str(name),
                        "status": str(status),
                        "tools": str(len(tools)),
                    }
                )

        if not servers:
            self.console.print("[dim]No MCP servers configured.[/dim]")
            return

        self.console.print(f"\n[bold][info]MCP Servers ({len(servers)})[/info][/bold]")
        for s in servers:
            status_style = (
                "green"
                if s["status"] == "connected"
                else "yellow"
                if s["status"] in ("connecting", "running")
                else "dim"
            )
            self.console.print(
                f"  [bold]{s['name']}[/bold] — [{status_style}]{s['status']}[/{status_style}]  [dim]({s['tools']} tools)[/dim]"
            )
        self.console.print()

    def _handle_repl_tasks(self) -> None:
        """Show background task snapshot."""
        tasks = self._collect_task_entries()
        if not tasks:
            self.console.print("[dim]No active tasks.[/dim]")
            return
        self._render_task_snapshot()

    def _handle_repl_rewind(self) -> None:
        """List conversation messages and let user rewind to a chosen turn."""
        msgs = self.session.conversation.messages
        user_msgs = [(i, m) for i, m in enumerate(msgs) if m.role == "user"]

        if not user_msgs:
            self.console.print("[dim]Nothing to rewind — no user messages in conversation.[/dim]")
            return

        self.console.print(f"\n[bold][info]Conversation history ({len(user_msgs)} user turns)[/info][/bold]")
        for idx, (orig_idx, msg) in enumerate(user_msgs):
            text = self._flatten_message_content(msg.content)
            preview = text[:80].replace("\n", " ")
            self.console.print(
                f"  [bold]{idx + 1}[/bold]  {preview}[dim]…[/dim]"
                if len(text) > 80
                else f"  [bold]{idx + 1}[/bold]  {preview}"
            )

        self.console.print()
        choice = self._safe_input("Rewind to turn (number) or leave empty to cancel: ").strip()
        if not choice:
            self.console.print("[dim]Rewind cancelled.[/dim]")
            return

        try:
            target = int(choice) - 1
            if target < 0 or target >= len(user_msgs):
                self.console.print("[error]Invalid turn number.[/error]")
                return
            orig_idx = user_msgs[target][0]
            # Truncate conversation to before this message
            self.session.conversation.messages = msgs[:orig_idx]
            self._engine_messages = []
            self.console.print(f"[success]Rewound to turn {target + 1}.[/success]")
        except (ValueError, IndexError):
            self.console.print("[error]Invalid input. Use a number from the list.[/error]")

    def _handle_repl_effort(self, args: str) -> None:
        """Show or set reasoning effort level."""
        current = getattr(self, "_effort", None)
        args = args.strip()

        if not args:
            self.console.print(
                f"\n[bold][info]Reasoning effort[/info][/bold]  [dim]current:[/dim] {current or '[success]auto[/success]'}"
            )
            self.console.print(
                "  Usage: [bold]/effort <level>[/bold]  where level is: [dim]auto, low, medium, high[/dim]"
            )
            self.console.print()
            return

        valid = {"auto", "low", "medium", "high"}
        if args.lower() in valid:
            self._effort = args.lower()
            self.console.print(f"[success]Reasoning effort set to {self._effort}.[/success]")
        else:
            self.console.print(f"[error]Invalid effort level: {args}. Use one of: {', '.join(sorted(valid))}[/error]")

    def _handle_repl_history(self) -> None:
        """Show recent session history."""
        events = list(self.history_log.events)
        if not events:
            self.console.print("[dim]No history events recorded this session.[/dim]")
            return

        self.console.print(f"\n[bold][info]Session History ({len(events)} events)[/info][/bold]")
        for ev in events[-20:]:  # Show last 20
            self.console.print(f"  [bold]{ev.title}[/bold]")
            if ev.detail:
                self.console.print(f"    [dim]{ev.detail[:120]}[/dim]")
        self.console.print()

    def _handle_repl_idle(self) -> None:
        """Show idle / away-summary configuration."""
        try:
            from clawcodex_ext.away_summary.config import load_away_summary_config

            cfg = load_away_summary_config()
        except Exception:
            cfg = None

        self.console.print("\n[bold][info]Idle Configuration[/info][/bold]")
        if cfg is not None:
            self.console.print(
                "  Auto-summary:    [success]enabled[/success]"
                if cfg.enabled
                else "  Auto-summary:    [dim]disabled[/dim]"
            )
            self.console.print(f"  Idle timeout:    [bold]{cfg.idle_seconds}s[/bold] ({cfg.idle_seconds // 60} min)")
            self.console.print(f"  Min turns:       {cfg.min_turns}")
            self.console.print(
                "  /recap command:  [success]available[/success]"
                if cfg.recap_command_enabled
                else "  /recap command:  [dim]disabled[/dim]"
            )
            self.console.print()
            self.console.print("  [dim]Set these via [bold]settings.away_summary[/bold] in your config file.[/dim]")
        else:
            self.console.print("  [dim]Away summary config not available.[/dim]")
        self.console.print()
        self.console.print("  [dim]On idle return you can:[/dim]")
        self.console.print("  [dim]  Continue — resume the conversation[/dim]")
        self.console.print("  [dim]  /clear   — start a fresh conversation[/dim]")
        self.console.print()

    # ── End of REPL-native handlers ─────────────────────────────────────

    def _handle_permission_command(self, args: str = "") -> None:
        """Handle the /permission command.

        Without arguments: show current mode + interactive selection menu.
        With a mode name: directly set the permission mode.
        """
        from clawcodex_ext.permissions import (
            EXTERNAL_PERMISSION_MODES,
            permission_mode_short_title,
            permission_mode_title,
        )

        mode = args.strip()

        if mode:
            # Direct mode selection
            if mode not in EXTERNAL_PERMISSION_MODES:
                valid = ", ".join(EXTERNAL_PERMISSION_MODES)
                self.console.print(f"[error]Invalid permission mode: '{mode}'[/error]\n[dim]Valid modes: {valid}[/dim]")
                return

            self._apply_permission_mode(mode)
            title = permission_mode_title(mode)
            self.console.print(f"[success]Permission mode set to: {title}[/success]")
            return

        # Interactive mode: show current mode + numbered menu
        current = self._permission_mode
        current_title = permission_mode_title(current)
        current_short = permission_mode_short_title(current)

        self.console.print()
        self.console.print(f"[bold]Current permission mode:[/bold] {current_title} ({current_short})")
        self.console.print()
        self.console.print("[bold]Select a permission mode:[/bold]")

        modes = list(EXTERNAL_PERMISSION_MODES)
        for i, m in enumerate(modes, 1):
            title = permission_mode_title(m)
            desc = self._permission_mode_description(m)
            marker = " ✓" if m == current else ""
            self.console.print(f"  [info]{i}.[/info] {title}{' [success]' + marker + '[/success]' if marker else ''}")
            self.console.print(f"       [dim]{desc}[/dim]")

        self.console.print()
        self.console.print("  [dim]or any other key to cancel[/dim]")
        self.console.print()

        try:
            choice = self._safe_input("Choose mode [1-5]: ").strip()
        except (EOFError, KeyboardInterrupt):
            self.console.print("[dim]Cancelled.[/dim]")
            return

        if not choice:
            return

        try:
            idx = int(choice)
            if 1 <= idx <= len(modes):
                chosen = modes[idx - 1]
                if chosen == current:
                    self.console.print("[dim]Already in that mode.[/dim]")
                    return
                self._apply_permission_mode(chosen)
                title = permission_mode_title(chosen)
                self.console.print(f"[success]Permission mode set to: {title}[/success]")
            else:
                self.console.print(f"[error]Invalid choice: {idx}. Enter 1–{len(modes)}.[/error]")
        except ValueError:
            self.console.print("[dim]Cancelled.[/dim]")

    @staticmethod
    def _permission_mode_description(mode: PermissionMode) -> str:
        """Return a human-readable description for each permission mode."""
        descriptions = {
            "default": "Prompt before every tool use (default behavior)",
            "plan": "No write operations — only plan and read code",
            "acceptEdits": "Auto-accept file edits; ask for other tools",
            "bypassPermissions": "Auto-approve all tool requests (caution!)",
            "dontAsk": "Never prompt — fail if permission would be needed",
        }
        return descriptions.get(mode, "")

    def _apply_permission_mode(self, mode: PermissionMode) -> None:
        """Apply a specific permission mode via the runtime controller.

        Called by the ``/permissions`` slash command picker. Routes
        through :class:`RuntimePermissionController` so the same lock /
        AppState-write / handler-restore logic is shared with the
        Shift+Tab keybinding path. Returns the mode applied; the
        caller is responsible for printing a user-facing confirmation.
        """
        next_mode = self._runtime_permission_controller.set_mode(mode)
        self._permission_mode = next_mode
        return next_mode

    def _apply_permission_mode_cycle(self) -> None:
        """Shift+Tab: cycle to the next permission mode.

        Bound to ``LiveStatus.on_permission_cycle`` at all three
        construction sites so the keybinding fires during the agent
        run (the prompt_toolkit background thread remains alive while
        ``_run_query`` blocks on the worker).
        """
        next_mode = self._runtime_permission_controller.cycle()
        self._permission_mode = next_mode

    def _notify_permission_mode_change(self, mode: str) -> None:
        """Surface a mode change in the live status row or console.

        Called by the runtime controller after the multi-field swap.
        When a :class:`LiveStatus` is mounted (i.e. the agent is
        running), update its visible message — the user sees the new
        mode inline in the spinner row. Otherwise fall through to
        :meth:`console.print` so the change is visible between turns.
        """
        status = getattr(self, "_active_live_status", None)
        if status is not None:
            try:
                status.update(f"mode: {mode}")
            except Exception:  # nosec B110
                pass
            return
        try:
            self.console.print(f"[dim]Permission mode: {mode}[/dim]")
        except Exception:  # nosec B110
            pass

    def _try_run_skill_slash(self, raw: str) -> bool:
        text = raw.strip()
        if not text.startswith("/"):
            return False
        body = text[1:]
        if not body:
            return False
        if body.split(maxsplit=1)[0].lower() in {c.lstrip("/").lower() for c in self._built_in_commands if c != "/"}:
            return False

        parts = body.split(maxsplit=1)
        skill_name = parts[0].strip()
        args = parts[1] if len(parts) > 1 else ""
        if not skill_name:
            return False

        try:
            from clawcodex_ext.tool_system.tools.skill import run_user_invoked_skill

            result = run_user_invoked_skill(skill_name, args, self.tool_context)
        except Exception as e:
            self.console.print(f"[error]Skill error: {e}[/error]")
            return True

        payload = result.output if isinstance(result.output, dict) else {}
        if result.is_error or not payload.get("success"):
            err = payload.get("error") if isinstance(payload.get("error"), str) else "Unknown skill error"
            self.console.print(f"[error]{err}[/error]")
            return True

        self.console.print(f"[dim]Launching skill: {payload.get('commandName', skill_name)}[/dim]")
        meta_parts: list[str] = []
        loaded = payload.get("loadedFrom")
        if isinstance(loaded, str) and loaded:
            meta_parts.append(f"source={loaded}")
        model = payload.get("model")
        if isinstance(model, str) and model:
            meta_parts.append(f"model={model}")
        tools = payload.get("allowedTools")
        if isinstance(tools, list) and tools:
            shown = ", ".join(str(t) for t in tools[:6])
            more = f" (+{len(tools) - 6})" if len(tools) > 6 else ""
            meta_parts.append(f"tools={shown}{more}")
        if meta_parts:
            self.console.print(f"[dim]{' · '.join(meta_parts)}[/dim]")

        if payload.get("status") in {"fork", "forked"}:
            result_text = payload.get("result")
            if isinstance(result_text, str) and result_text.strip():
                self.console.print()
                self.console.print(Markdown(result_text))
                from clawcodex_ext.types.messages import create_message

                self._engine_messages.append(create_message("assistant", result_text))
                self.console.print()
            return True

        prompt = payload.get("prompt")
        if not isinstance(prompt, str) or not prompt.strip():
            self.console.print("[error]Skill produced empty prompt[/error]")
            return True

        self.chat(prompt)
        return True

    def show_help(self):
        """Show help message."""
        help_text = r"""
**Available Commands:**

- `/` - Show all commands and skills
- `/help` - Show this help message
- `/exit`, `/quit`, `/q` - Exit the REPL
- `/clear`, `/reset`, `/new` - Clear conversation history
- `/save` - Save current session
- `/load <session-id>` - Load a previous session
- `/stream [on|off|toggle]` - Toggle live response rendering
- `/render-last` - Re-render the last assistant reply as Markdown
- `/tools` - List available built-in tools
- `/tool <name> <json>` - Run a tool directly
- `/skills` - List all available skills
- `/init` - Create CLAUDE.md file for the project
- `/cost` - Show session cost and usage
- `/compact` - Compact conversation to save context space
- `/tui` - Switch into the Textual-based full-screen TUI (opt-in)

**Usage:**
- Type your message and press Enter to chat
- Use ↓/↑ and Enter to accept a slash-command suggestion
- Press Ctrl+C to interrupt current operation
- Press Ctrl+D to exit
- Multi-line input: Shift+Enter, Meta/Alt+Enter, or `\` + Enter inserts a newline; plain Enter submits
"""
        self.console.print(Markdown(help_text))

    def _handle_skill_command(self) -> None:
        """Handle /skill command - list all available skills."""
        try:
            from src.skills.loader import get_all_skills

            cwd = self.tool_context.cwd or self.tool_context.workspace_root
            skills = list(get_all_skills(project_root=cwd))
            skills.sort(key=lambda s: s.name.lower())

            if not skills:
                self.console.print("\n[bold]Available Skills:[/bold]")
                self.console.print("[dim]No skills found.[/dim]")
                self.console.print(
                    "[dim]Create skills in ~/.clawcodex/skills/ or ~/.claude/skills/ or .clawcodex/skills/ in your project.[/dim]"
                )
                return

            # Group skills by source
            from collections import defaultdict

            by_source: dict[str, list] = defaultdict(list)
            for s in skills:
                loaded = getattr(s, "loaded_from", "") or "unknown"
                by_source[loaded].append(s)

            self.console.print(f"\n[bold]Available Skills ({len(skills)}):[/bold]")
            for source in sorted(by_source.keys()):
                source_skills = by_source[source]
                self.console.print(f"\n[info]{source.title()} Skills:[/info]")
                for s in source_skills:
                    desc = (getattr(s, "description", None) or "").strip()
                    user_invocable = getattr(s, "user_invocable", True)
                    inv_str = "" if user_invocable else " [dim](not user-invocable)[/dim]"
                    self.console.print(f"  [success]/{s.name}[/success]{inv_str}")
                    if desc:
                        self.console.print(f"    [dim]{desc}[/dim]")
            self.console.print()
        except Exception as e:
            self.console.print(f"[error]Error loading skills: {e}[/error]")

    def _is_recoverable_tool_error(self, tool_name: str, tool_output) -> bool:
        if not isinstance(tool_name, str):
            return False
        if not isinstance(tool_output, dict):
            return False
        name = tool_name.strip().lower()
        err = tool_output.get("error")
        if not isinstance(err, str):
            return False
        e = err.lower()
        if name == "read" and e.startswith("file not found:"):
            p = err.split(":", 2)[-1].strip()
            if (
                "/.clawcodex/skills/" in p
                or "\\.clawcodex\\skills\\" in p
                or "/.claude/skills/" in p
                or "\\.claude\\skills\\" in p
            ):
                return True
        return False

    def _provider_uses_system_kwarg(self) -> bool:
        from src.providers.anthropic_provider import AnthropicProvider
        from src.providers.minimax_provider import MinimaxProvider

        return isinstance(self.provider, (AnthropicProvider, MinimaxProvider))

    def _build_direct_stream_payload(self) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        style_name = getattr(self.tool_context, "output_style_name", None)
        style_dir = getattr(self.tool_context, "output_style_dir", None)
        style_prompt = resolve_output_style(style_name, style_dir).prompt

        if self._provider_uses_system_kwarg():
            return self.session.conversation.get_messages(), ({"system": style_prompt} if style_prompt.strip() else {})

        messages: list[dict[str, Any]] = []
        for msg in self.session.conversation.messages:
            if isinstance(msg.content, str):
                messages.append({"role": msg.role, "content": msg.content})
        if style_prompt.strip():
            messages = [{"role": "system", "content": style_prompt}, *messages]
        return messages, {}

    def _should_try_direct_stream(self, user_input: str) -> bool:
        if not self.stream:
            return False
        # Goal turns must use the canonical query loop so the independent
        # completion evaluator runs at the natural stop boundary.
        if self._has_active_evaluator_goal_for_control_flow():
            return False
        text = user_input.strip().lower()
        if not text or text.startswith("/"):
            return False
        if len(text) > 240:
            return False

        code_task_markers = (
            "/",
            "\\",
            "src/",
            "tests/",
            ".py",
            ".ts",
            ".md",
            "file",
            "files",
            "read",
            "write",
            "edit",
            "modify",
            "change",
            "search",
            "grep",
            "glob",
            "bash",
            "shell",
            "command",
            "run",
            "test",
            "fix",
            "bug",
            "refactor",
            "repo",
            "repository",
            "project",
            "workspace",
            "folder",
            "directory",
            "function",
            "class",
            "module",
            "code",
            "implementation",
            "readme",
            "pyproject",
            "package.json",
            "git",
            "commit",
            "diff",
            "tool",
            "文件",
            "代码",
            "仓库",
            "项目",
            "目录",
            "读取",
            "写入",
            "修改",
            "搜索",
            "运行",
            "测试",
            "修复",
            "命令",
            "工具",
            "函数",
            "类",
        )
        return not any(marker in text for marker in code_task_markers)

    def _stream_direct_response(self, on_text_chunk=None) -> str | None:
        """Stream a response via ``chat_stream_response`` which supports abort.

        Uses the upstream provider's ``chat_stream_response`` (with built-in
        ``StreamAbortGuard``) instead of ``chat_stream``, so CTRL+C via the
        ``AbortController`` terminates the HTTP connection immediately
        instead of waiting for the next chunk to arrive.
        """

        abort_controller = getattr(self, "_direct_abort_controller", None)
        if abort_controller is None:
            abort_controller = AbortController()
        abort_signal = abort_controller.signal
        # If user already cancelled before we got here, bail fast.
        if abort_signal.aborted:
            return None

        self._last_direct_response_usage = None
        try:
            api_messages, call_kwargs = self._build_direct_stream_payload()
            try:
                response = self.provider.chat_stream_response(
                    api_messages,
                    tools=None,
                    abort_signal=abort_signal,
                    on_text_chunk=on_text_chunk,
                    **call_kwargs,
                )
                content = response.content if response else None
                full_response = content if isinstance(content, str) and content else None
                if full_response is not None:
                    usage = getattr(response, "usage", None) if response is not None else None
                    self._last_direct_response_usage = usage if isinstance(usage, dict) else None
                    self.session.conversation.add_assistant_message(full_response)
                    return full_response
            except NotImplementedError:
                pass

            chunks: list[str] = []
            try:
                stream = self.provider.chat_stream(api_messages, tools=None, **call_kwargs)
                for chunk in stream:
                    if abort_signal.aborted:
                        return None
                    if not chunk:
                        continue
                    chunks.append(chunk)
                    if on_text_chunk is not None:
                        on_text_chunk(chunk)
            except Exception:  # nosec B110
                if abort_signal.aborted:
                    return None
                return None
            full_response = "".join(chunks) or None
        except Exception:  # nosec B110
            # Provider raised (e.g. AbortError from user cancel, or a
            # real error). If the abort controller was tripped, the user
            # cancelled — return None without logging an error.
            if abort_signal.aborted:
                return None
            raise

        if full_response:
            self.session.conversation.add_assistant_message(full_response)
        return full_response

    def _account_direct_goal_turn(self, usage: dict[str, Any] | None) -> None:
        try:
            from clawcodex_ext.goal.runtime import goal_runtime_for_context

            goal_runtime = goal_runtime_for_context(self.tool_context)
        except Exception:
            return
        if goal_runtime is None:
            return
        try:
            turn_id = goal_runtime.on_turn_start(plan_mode=bool(getattr(self.tool_context, "plan_mode", False)))
        except Exception:
            return
        try:
            goal_runtime.on_token_usage(turn_id, usage or {})
        except Exception as exc:
            try:
                goal_runtime.on_turn_error(turn_id, exc)
            except Exception:  # nosec B110
                pass
            return
        try:
            goal_runtime.on_turn_stop(turn_id)
        except Exception:  # nosec B110
            pass

    def _get_last_assistant_text(self) -> str | None:
        for message in reversed(self.session.conversation.messages):
            if message.role != "assistant":
                continue
            content = message.content
            if isinstance(content, str) and content.strip():
                return content
            if isinstance(content, list):
                parts: list[str] = []
                for block in content:
                    block_type = getattr(block, "type", None)
                    if block_type == "text":
                        text = getattr(block, "text", "")
                        if isinstance(text, str) and text:
                            parts.append(text)
                joined = "".join(parts).strip()
                if joined:
                    return joined
        return None

    def _render_last_assistant_message(self) -> bool:
        text = self._get_last_assistant_text()
        if not text:
            return False
        self.console.print("\n[bold]Last Assistant Response[/bold]")
        self.console.print(Markdown(text))
        self.console.print()
        return True

    def _sanitize_conversation_for_api_error(self, msg: Any) -> None:
        """If the assistant message signals an API error that requires
        history sanitization, mutate ``session.conversation.messages``
        in place to match the engine's sanitized state.

        Today only ``image_unsupported`` triggers a strip: the user's
        image-bearing UserMessage stays in ``session.conversation``
        otherwise, and the direct-stream path (line ~2186 of this file)
        reads ``session.conversation.messages`` rather than
        ``engine.get_messages()`` — so without this mirror, a short
        text-only follow-up routed through ``_stream_direct_response``
        would hit the same provider with the still-cached image.

        Extracted into a method so the REPL handler stays terse AND so
        this load-bearing behaviour has a direct unit-test surface
        (test_repl_conversation_sanitization).
        """
        if getattr(msg, "_api_error", None) == "image_unsupported":
            from clawcodex_ext.context_system.microcompact import (
                strip_images_from_typed_messages,
            )

            self.session.conversation.messages = strip_images_from_typed_messages(self.session.conversation.messages)

    def chat(self, user_input: str, max_turns: int | None = None):
        """Send message to LLM and display response.

        Uses the new QueryEngine (WS-4) state machine to drive the query loop.

        Args:
            user_input: The user message to send.
            max_turns: Maximum number of tool call turns. None means unlimited
                (matching TS interactive REPL behavior). Only set for SDK/non-interactive mode.
        """
        _load_heavy_runtime()
        self._last_chat_outcome = "success"
        from src.repl.background_escape import BackgroundEscape

        # Expand ``@path`` mentions into context attachments before the model
        # sees the message. Port of
        # ``typescript/src/utils/attachments.ts#processAtMentionedFiles``.
        from src.command_system.input_processing import (
            build_image_content_blocks,
            expand_at_mentions,
            format_at_mention_attachments,
        )
        from clawcodex_ext.types.content_blocks import TextBlock

        cwd_for_mentions = str(self.tool_context.cwd or self.tool_context.workspace_root)
        _, at_attachments = expand_at_mentions(user_input, cwd=cwd_for_mentions)

        available_agents = self._available_agents()
        from src.command_system.input_processing import (
            expand_agent_mentions,
            find_unknown_agent_mentions,
            format_unknown_agent_mention_error,
        )

        unknown_agents = find_unknown_agent_mentions(user_input, available_agents)
        if unknown_agents:
            self.console.print(f"[error]{format_unknown_agent_mention_error(unknown_agents, available_agents)}[/error]")
            return True

        # Port of ``processAgentMentions`` from
        # ``typescript/src/utils/attachments.ts``: if the user types
        # ``@agent-explore`` (or the autocomplete form ``@"explore (agent)"``),
        # attach a system-reminder telling the model to delegate to that
        # agent via the Agent tool. Mentions of unknown agents are ignored so
        # we don't polute context with misleading reminders.
        # ponytail: skip direct @agent- shortcut when SOP bundle is active;
        # in bundle mode the overview agent must handle delegation with full
        # conversation context so stage agents get history, not one-shot prompts.
        _sop_bundle_active = False
        try:
            from extensions.sop_converter.bundle_context import get_active_bundle

            _sop_bundle_active = get_active_bundle() is not None
        except ImportError:
            pass

        agent_attachments = expand_agent_mentions(user_input, available_agents)

        if agent_attachments:
            for att in agent_attachments:
                if _sop_bundle_active:
                    self.console.print(f"[dim]  ⎿  @{att['agent_type']} → delegating via overview agent[/dim]")
                else:
                    self.console.print(f"[dim]  ⎿  Invoking agent @{att['agent_type']}[/dim]")

        if agent_attachments and not at_attachments and not _sop_bundle_active:
            from clawcodex_ext.repl.mentioned_agent import (
                run_mentioned_agent_direct,
                should_run_mentioned_agent_directly,
            )

            if should_run_mentioned_agent_directly(agent_attachments, at_attachments):
                if run_mentioned_agent_direct(
                    self,
                    agent_type=agent_attachments[0]["agent_type"],
                    user_input=user_input,
                ):
                    return True

        all_attachments = list(at_attachments) + list(agent_attachments)
        if all_attachments:
            attachment_text = format_at_mention_attachments(all_attachments)
            user_input = f"{attachment_text}\n\n{user_input}" if attachment_text else user_input
            for att in at_attachments:
                kind = att.get("kind")
                if kind == "image":
                    # TS shows "Read 1 file (ctrl+o to expand)" for image
                    # @-mentions; we mirror the user-facing intent without
                    # the count (one mention -> one line).
                    self.console.print(f"[dim]  ⎿  Read image {att['display_path']}[/dim]")
                elif kind == "binary":
                    # Binary file (PDF, archive, ...) — show what happened
                    # so the user isn't surprised that no content was
                    # inlined. The reminder text already nudges the model
                    # toward the Read tool.
                    self.console.print(f"[dim]  ⎿  Skipped binary file {att['display_path']}[/dim]")
                else:
                    label = "directory" if kind == "directory" else "file"
                    self.console.print(
                        f"[dim]  ⎿  Listed {label} {att['display_path']}{'/' if kind == 'directory' else ''}[/dim]"
                    )

        # Companion intro — build and prepend companion intro attachment
        # if a companion has been hatched and not yet announced.
        intro_attachments: list[dict[str, Any]] = []
        from src.buddy.prompt import (
            build_companion_intro_attachment,
            format_companion_intro_attachments,
        )

        intro_attachments = build_companion_intro_attachment(
            self.session.conversation.messages,  # type: ignore[attr-defined]
        )
        if intro_attachments:
            intro_text = format_companion_intro_attachments(intro_attachments)
            if intro_text:
                user_input = f"{intro_text}\n\n{user_input}" if user_input else intro_text
            from clawcodex_ext.types.messages import AttachmentMessage

            self.session.conversation.messages.append(AttachmentMessage(attachments=intro_attachments))

        # Image @-mentions become real image content blocks on the user
        # message so the model sees the image directly (matches TS's
        # auto-Read-on-@image behaviour, and stops the model from
        # hallucinating about mojibake'd PNG bytes in a system-reminder).
        # When images are present, ``user_message_content`` is a mixed
        # ``[TextBlock, ImageBlock, ...]`` list; otherwise it is just the
        # text string. Both shapes are accepted by ``add_user_message``
        # and ``engine.submit_message`` (``MessageContent = str |
        # list[ContentBlock]``).
        image_blocks = build_image_content_blocks(at_attachments)
        if image_blocks:
            user_message_content: str | list[Any] = [
                TextBlock(text=user_input),
                *image_blocks,
            ]
        else:
            user_message_content = user_input

        self.session.conversation.add_user_message(user_message_content)

        try:
            # Show the agent name (from @agent-mention) or default to "Assistant"
            _agent_label = "Assistant"
            if agent_attachments:
                _agent_label = agent_attachments[0].get("agent_type", "Assistant")
            self.console.print(f"\n[bold]{_agent_label}[/bold]")

            stream_started = False

            def _stop_status_once() -> None:
                nonlocal stream_started
                if self._current_status is not None and not stream_started:
                    try:
                        self._current_status.stop()
                    except Exception:  # nosec B110
                        pass
                stream_started = True

            # Direct-stream skips the tool loop; it can only carry plain
            # text. If the user attached an image, fall through to the
            # full engine path so the image content block survives.
            if not image_blocks and self._should_try_direct_stream(user_input):

                def on_text_chunk_direct(chunk: str) -> None:
                    if not chunk:
                        return
                    _stop_status_once()
                    self.console.print(chunk, end="", markup=False, highlight=False, soft_wrap=True)

                self._direct_abort_controller = AbortController()

                def _cancel_direct_stream() -> None:
                    self._last_chat_outcome = "cancelled"
                    if self._direct_abort_controller is not None:
                        self._direct_abort_controller.abort("user_interrupt")
                    # Immediate visual feedback — update the LiveStatus message
                    status = getattr(self, "_active_live_status", None)
                    if status is not None:
                        try:
                            status.update("[warning]Cancelling…[/warning]")
                        except Exception:  # nosec B110
                            pass

                _direct_status_ref: list[LiveStatus] = []

                def _on_submit_direct(text: str) -> None:
                    self._enqueue_prompt(text)
                    if _direct_status_ref:
                        _direct_status_ref[0].update(self._status_message())

                # Ctrl+B background escape flag — set by the
                # LiveStatus keybinding and checked after the
                # with-block to raise BackgroundEscape.
                _background_requested_direct = False

                def _on_background_direct() -> None:
                    nonlocal _background_requested_direct
                    _background_requested_direct = True
                    # Also cancel the direct stream so it stops
                    # consuming tokens immediately.
                    if self._direct_abort_controller is not None:
                        self._direct_abort_controller.abort("background")

                self._im_active_cancel = _cancel_direct_stream
                try:
                    with _pt_patch_stdout(raw=True):
                        with LiveStatus(
                            self._status_message(),
                            on_cancel=_cancel_direct_stream,
                            on_submit=_on_submit_direct,
                            on_expand=self._do_expand_last,
                            on_background=_on_background_direct,
                            on_permission_cycle=self._apply_permission_mode_cycle,
                            completer=self.completer,
                            history=self._file_history,
                            toolbar_text=self._bottom_toolbar,
                        ) as status:
                            _direct_status_ref.append(status)
                            self._active_live_status = status
                            try:
                                direct_response = self._stream_direct_response(
                                    on_text_chunk=on_text_chunk_direct,
                                )
                            finally:
                                self._active_live_status = None
                finally:
                    if self._im_active_cancel is _cancel_direct_stream:
                        self._im_active_cancel = None
                # Capture unsubmitted buffer text before returning so the
                # user doesn't lose what they were typing when the agent
                # finishes mid-keystroke.
                pending = status._pending_text
                if pending:
                    self._enqueue_prompt(pending)
                if direct_response is not None:
                    self._account_direct_goal_turn(getattr(self, "_last_direct_response_usage", None))
                    self.console.print("\n")
                    # Per-turn save: persist JSONL transcript only (lightweight).
                    try:
                        self.session.save_transcript()
                    except Exception:  # nosec B110
                        pass
                    self._continue_goal_if_idle()
                    return True
                # User pressed ESC/Ctrl+C during direct stream — skip the
                # engine path entirely instead of falling through.
                if self._direct_abort_controller is not None and self._direct_abort_controller.signal.aborted:
                    self._last_chat_outcome = "cancelled"
                    return False
                if _background_requested_direct:
                    raise BackgroundEscape()

            from src.outputStyles import resolve_output_style

            style_name = getattr(self.tool_context, "output_style_name", None)
            style_dir = getattr(self.tool_context, "output_style_dir", None)
            style_prompt = resolve_output_style(style_name, style_dir).prompt

            from clawcodex_ext.tool_system import get_team_aware_tool_list
            from clawcodex_ext.agent.agent_tool_utils import filter_tools_for_startup_agent

            tools = get_team_aware_tool_list(self.tool_registry, self.tool_context.team)
            startup_agent = getattr(self.tool_context, "startup_agent", None)
            if startup_agent is None:
                rc = getattr(self, "runtime_context", None)
                if rc is not None:
                    startup_agent = getattr(rc.options, "startup_agent", None)
            tools = filter_tools_for_startup_agent(tools, startup_agent)

            # Coordinator Mode — when ``CLAUDE_CODE_COORDINATOR_MODE=true``,
            # restrict the tool list to read-only + delegation tools
            # (Agent / SendMessage / TaskStop / Read / WebSearch / WebFetch),
            # replace the system prompt with the coordinator-specific prompt,
            # and inject the worker-tools context block.
            from src.coordinator.mode import (
                is_coordinator_mode,
                filter_coordinator_tools,
                get_coordinator_user_context,
            )
            from src.coordinator.prompt import get_coordinator_system_prompt

            if is_coordinator_mode():
                tools = filter_coordinator_tools(tools)
                # Get MCP clients for worker context block
                mcp_clients = getattr(self.tool_context, "mcp_clients", None) or {}
                engine_system_prompt = get_coordinator_system_prompt()
                engine_user_context = get_coordinator_user_context(
                    mcp_clients=mcp_clients.values(),
                )
                # style_prompt is irrelevant in coordinator mode — the
                # coordinator has its own prompt body that replaces the
                # default system prompt entirely.
                append_prompt: str | None = None
            else:
                engine_system_prompt = None
                engine_user_context = None
                append_prompt = style_prompt
                # Inject resolved agent system prompt if present
                extra = getattr(self, "_append_system_prompt", "")
                if extra:
                    append_prompt = f"{append_prompt}\n\n{extra}"

            prior_messages = list(self._engine_messages)

            # Local import — ``clawcodex_ext.query.engine`` pulls in ~1.1s of
            # transitive deps (query protocol + state machine). Kept out of
            # ``_load_heavy_runtime`` so REPL cold start doesn't pay the cost
            # until the user actually submits a non-slash prompt.
            from clawcodex_ext.query.engine import QueryEngine, QueryEngineConfig

            engine_config = QueryEngineConfig(
                cwd=self.tool_context.workspace_root,
                provider=self.provider,
                tool_registry=self.tool_registry,
                tools=tools,
                tool_context=self.tool_context,
                system_prompt=engine_system_prompt,
                user_context=engine_user_context,
                append_system_prompt=append_prompt,
                max_turns=max_turns,
                initial_messages=prior_messages,
            )
            engine = QueryEngine(engine_config)

            response_text = ""
            last_text_was_printed = False

            async def _run_query() -> tuple[str, bool]:
                nonlocal stream_started
                last_text = ""
                last_text_was_printed = False
                api_call_count = 0
                tool_use_map: dict[str, tuple[str, dict]] = {}
                # Per-turn token totals — surfaced to the spinner suffix
                # via ``status.set_tokens(...)``. Local to the closure so
                # they reset every turn; ``self._stats_*`` remain the
                # session-cumulative counters for ``/stats``.
                turn_tokens = 0
                # Track whether a Task*/TodoWrite round is "in flight" so we
                # can coalesce a run of task-management calls into a single
                # TaskListV2-style snapshot instead of dumping one ``●`` bullet
                # per call. This mirrors the behaviour of
                # ``typescript/src/components/TaskListV2.tsx``, which re-renders
                # a single widget each time the ``tasks`` slice of AppState
                # changes.
                pending_task_flush = False
                task_tool_ids: set[str] = set()
                # When the assistant emits multiple tool_use blocks in one
                # message, printing all ``● Tool(args)`` lines eagerly and
                # then dumping every ``⎿ preview`` underneath stacks the
                # output into one tall, hard-to-scan block. Defer each
                # header so it prints right above its matching result —
                # this is what produces the per-call "small block" look
                # in the TS Ink reference (see
                # ``typescript/src/components/REPL.tsx``).
                pending_tool_use_prints: dict[str, str] = {}
                # When True, the next tool-use header should be preceded by a
                # blank-line spacer so consecutive tool calls render as
                # discrete, scannable blocks instead of a dense wall of
                # identical-looking lines. Reset to False after each header
                # we print; set to True after we emit a result line.
                tool_block_needs_leading_space = False

                def _flush_task_snapshot_if_any() -> None:
                    nonlocal pending_task_flush
                    if not pending_task_flush:
                        return
                    pending_task_flush = False
                    self._render_task_snapshot()

                def _on_thinking_chunk(chunk: str) -> None:
                    """Accumulate thinking chunks for later expansion."""
                    if self._thinking_visible:
                        # Print thinking content directly when visible
                        self.console.print(chunk, end="", markup=False, highlight=False, soft_wrap=True)
                    else:
                        # Stash for later expansion via ctrl+o
                        self._thinking_chunks.append(chunk)

                async for msg in engine.submit_message(
                    user_message_content,
                    on_thinking_chunk=_on_thinking_chunk,
                ):
                    if isinstance(msg, StreamEvent):
                        if msg.type == "stream_request_start":
                            api_call_count += 1
                            # Reset spinner status to "Thinking…" when a
                            # new API round-trip starts, so the user sees
                            # the LLM is reasoning again (rather than the
                            # previous tool name left from the last turn).
                            if _engine_status_ref:
                                _engine_status_ref[0].update(self._status_message())
                        continue

                    if isinstance(msg, AssistantMessage):
                        self.session.conversation.add_assistant_message(msg.content)
                        # Engine-side mirror: the engine has just stripped
                        # image blocks from _mutable_messages; keep
                        # session.conversation in sync so the persisted
                        # JSONL and the direct-stream path (which reads
                        # session.conversation directly) don't carry stale
                        # image content. See QueryEngine.submit_message for
                        # the engine side of this pair.
                        self._sanitize_conversation_for_api_error(msg)
                        usage = getattr(msg, "usage", None)
                        if isinstance(usage, dict):
                            in_toks = int(usage.get("input_tokens", 0) or 0)
                            out_toks = int(usage.get("output_tokens", 0) or 0)
                            self._stats_input_tokens += in_toks
                            self._stats_output_tokens += out_toks
                            turn_tokens += in_toks + out_toks
                            if _engine_status_ref:
                                _engine_status_ref[0].set_tokens(turn_tokens)
                        content = msg.content
                        if isinstance(content, str):
                            if content:
                                last_text = content
                                last_text_was_printed = False
                                _stop_status_once()
                                stream_started = True
                        elif isinstance(content, list):
                            for block in content:
                                if isinstance(block, TextBlock) and block.text:
                                    # New assistant text -> first flush any
                                    # pending task snapshot so the widget
                                    # lands above the explanatory text.
                                    _flush_task_snapshot_if_any()
                                    last_text = block.text
                                    _stop_status_once()
                                    stream_started = True
                                    self.console.print(Markdown(block.text))
                                    last_text_was_printed = True
                                elif isinstance(block, ToolUseBlock):
                                    tool_use_map[block.id] = (block.name, block.input)
                                    # Show the tool name in the spinner
                                    # so "Thinking…" → "● Bash" / "● Read"
                                    # instead of a static message.
                                    if _engine_status_ref:
                                        _engine_status_ref[0].update(f"● {block.name}")
                                    if block.name in _TASK_WIDGET_TOOL_NAMES:
                                        task_tool_ids.add(block.id)
                                        pending_task_flush = True
                                        continue
                                    # Any non-task tool call terminates the
                                    # current task widget run; flush the
                                    # snapshot before printing the new call.
                                    _flush_task_snapshot_if_any()
                                    summary = summarize_tool_use(block.name, block.input)
                                    if isinstance(summary, str) and summary:
                                        summary = self._shorten_path_text(summary)
                                    # Mirror the compact ``● ToolName(args)``
                                    # rendering used by
                                    # ``typescript/src/components/REPL.tsx``
                                    # (and Claude Code's Ink UI). The function-
                                    # call shape is less noisy than the old
                                    # ``• ToolName (args) running…`` format
                                    # and is easier to scan.
                                    # Args go inside parens in a dim style;
                                    # omit them entirely when we have nothing
                                    # meaningful to show so ``● ToolName`` reads
                                    # cleaner than a literal ``ToolName()``.
                                    if summary:
                                        call_args = f"[dim]([/dim]{escape(summary)}[dim])[/dim]"
                                    else:
                                        call_args = ""
                                    pending_tool_use_prints[block.id] = (
                                        f"[success]●[/success] [bold][info]{block.name}[/info][/bold]"
                                        + (f" {call_args}" if call_args else "")
                                    )
                        continue

                    if isinstance(msg, SystemMessage):
                        subtype = getattr(msg, "subtype", None)
                        if subtype == "max_turns_reached":
                            _stop_status_once()
                            stream_started = True
                            self.console.print(
                                "[warning]Reached maximum number of turns. The task may be incomplete.[/warning]"
                            )
                        elif subtype in {"goal_evaluation", "goal_achieved"}:
                            _stop_status_once()
                            stream_started = True
                            evaluator_usage = getattr(msg, "usage", None)
                            if isinstance(evaluator_usage, dict):
                                in_toks = int(evaluator_usage.get("input_tokens", 0) or 0)
                                out_toks = int(evaluator_usage.get("output_tokens", 0) or 0)
                                self._stats_input_tokens += in_toks
                                self._stats_output_tokens += out_toks
                                turn_tokens += in_toks + out_toks
                                if _engine_status_ref:
                                    _engine_status_ref[0].set_tokens(turn_tokens)
                            style = "success" if subtype == "goal_achieved" else "dim"
                            self.console.print(f"[{style}]{escape(str(msg.content))}[/{style}]")
                            add_existing = getattr(
                                self.session.conversation,
                                "add_existing_message",
                                None,
                            )
                            if callable(add_existing):
                                add_existing(msg)
                            else:
                                self.session.conversation.add_message(msg.role, msg.content)
                        elif subtype == "goal_evaluator_error":
                            _stop_status_once()
                            stream_started = True
                            self._last_chat_outcome = "goal_evaluator_error"
                            evaluator_usage = getattr(msg, "usage", None)
                            if isinstance(evaluator_usage, dict):
                                in_toks = int(evaluator_usage.get("input_tokens", 0) or 0)
                                out_toks = int(evaluator_usage.get("output_tokens", 0) or 0)
                                self._stats_input_tokens += in_toks
                                self._stats_output_tokens += out_toks
                                turn_tokens += in_toks + out_toks
                                if _engine_status_ref:
                                    _engine_status_ref[0].set_tokens(turn_tokens)
                            self.console.print(f"[warning]{escape(str(msg.content))}[/warning]")
                            add_existing = getattr(
                                self.session.conversation,
                                "add_existing_message",
                                None,
                            )
                            if callable(add_existing):
                                add_existing(msg)
                            else:
                                self.session.conversation.add_message(msg.role, msg.content)
                        continue

                    if isinstance(msg, UserMessage):
                        content = msg.content
                        if isinstance(content, list):
                            # The engine yields tool results as UserMessages.
                            # Preserve them in the session conversation as
                            # well as rendering them; save_transcript() writes
                            # from this conversation, so omitting this append
                            # made every result disappear after --resume.
                            self._record_tool_result_message(msg)
                            for block in content:
                                if isinstance(block, ToolResultBlock):
                                    # Suppress per-call ``⎿ ...`` result
                                    # output for task widget tools — the
                                    # flushed snapshot already reflects the
                                    # post-call state. Errors still surface
                                    # so the user sees validation problems.
                                    if block.tool_use_id in task_tool_ids:
                                        # Task tool headers were never
                                        # buffered (we render a snapshot
                                        # instead) so nothing to flush.
                                        if block.is_error:
                                            _flush_task_snapshot_if_any()
                                            err_text = (
                                                block.content if isinstance(block.content, str) else str(block.content)
                                            )
                                            self.console.print(
                                                f"[error]  ⎿  {escape(err_text) if err_text else 'Error'}[/error]"
                                            )
                                        continue
                                    # Print the deferred ``● Tool(args)``
                                    # header right above this result so each
                                    # call renders as a self-contained block.
                                    header = pending_tool_use_prints.pop(block.tool_use_id, None)
                                    if header is not None:
                                        if tool_block_needs_leading_space:
                                            self.console.print()
                                        self.console.print(header)
                                    # Match the TS UI's tool-result prefix
                                    # ``  ⎿  `` (see
                                    # ``typescript/src/components/MessageResponse.tsx``).
                                    if block.is_error:
                                        err_text = (
                                            block.content if isinstance(block.content, str) else str(block.content)
                                        )
                                        self.console.print(
                                            f"[error]  ⎿  {escape(err_text) if err_text else 'Error'}[/error]"
                                        )
                                    else:
                                        preview = self._format_tool_result_preview(
                                            block,
                                            tool_use_map.get(block.tool_use_id),
                                        )
                                        if isinstance(preview, str):
                                            # Multi-line previews indent
                                            # continuation lines under the
                                            # ``⎿`` glyph so they read as part
                                            # of the same result block.
                                            if "\n" in preview:
                                                first, *rest = preview.split("\n")
                                                self.console.print(f"[dim]  ⎿  {first}[/dim]")
                                                for ln in rest:
                                                    self.console.print(f"[dim]     {ln}[/dim]")
                                            else:
                                                self.console.print(f"[dim]  ⎿  {preview}[/dim]")
                                        else:
                                            # Rich renderable (e.g. Edit diff
                                            # Group) — emit the prefix then
                                            # the renderable so its internal
                                            # styling survives the dim wrap.
                                            self.console.print("[dim]  ⎿  [/dim]", end="")
                                            self.console.print(preview)
                                    tool_block_needs_leading_space = True
                        continue

                # Flush any trailing task snapshot at end-of-turn so the
                # final "N tasks (...)" summary lands in the transcript.
                _flush_task_snapshot_if_any()

                # If a tool_use never received a matching result (turn cut
                # short, error mid-loop), surface the headers we were
                # holding so the user can still see what was attempted.
                for header in pending_tool_use_prints.values():
                    if tool_block_needs_leading_space:
                        self.console.print()
                    self.console.print(header)
                    tool_block_needs_leading_space = True
                pending_tool_use_prints.clear()

                return last_text, last_text_was_printed

            def _cancel_engine() -> None:
                self._last_chat_outcome = "cancelled"
                try:
                    engine.interrupt()
                except Exception:  # nosec B110
                    pass
                # Immediate visual feedback — update the LiveStatus message
                # so the user sees "Cancelling…" without waiting for the
                # abort to propagate through the provider stream.
                status = getattr(self, "_active_live_status", None)
                if status is not None:
                    try:
                        status.update("[warning]Cancelling…[/warning]")
                    except Exception:  # nosec B110
                        pass

            _engine_status_ref: list[LiveStatus] = []

            def _on_submit_engine(text: str) -> None:
                self._enqueue_prompt(text)
                if _engine_status_ref:
                    _engine_status_ref[0].update(self._status_message())

            # Ctrl+B background escape flag — set by the
            # LiveStatus keybinding and raised after the
            # with-block exits.
            _background_requested_engine = False

            def _on_background_engine() -> None:
                nonlocal _background_requested_engine
                _background_requested_engine = True
                # Also cancel the engine so it stops consuming
                # tokens and tool calls immediately.
                try:
                    engine.interrupt()
                except Exception:  # nosec B110
                    pass

            self._im_active_cancel = _cancel_engine
            try:
                with _pt_patch_stdout(raw=True):
                    with LiveStatus(
                        self._status_message(),
                        on_cancel=_cancel_engine,
                        on_submit=_on_submit_engine,
                        on_expand=self._do_expand_last,
                        on_background=_on_background_engine,
                        on_permission_cycle=self._apply_permission_mode_cycle,
                        completer=self.completer,
                        history=self._file_history,
                        toolbar_text=self._bottom_toolbar,
                    ) as status:
                        _engine_status_ref.append(status)
                        self._active_live_status = status
                        try:
                            loop = self._get_chat_loop()
                            if loop.is_running():
                                import concurrent.futures

                                with concurrent.futures.ThreadPoolExecutor() as pool:
                                    response_text, last_text_was_printed = pool.submit(
                                        lambda: asyncio.run(_run_query())
                                    ).result()
                            else:
                                response_text, last_text_was_printed = loop.run_until_complete(_run_query())
                        except RuntimeError:
                            response_text, last_text_was_printed = asyncio.run(_run_query())
                        finally:
                            self._active_live_status = None
            finally:
                if self._im_active_cancel is _cancel_engine:
                    self._im_active_cancel = None

            # Capture unsubmitted buffer text before proceeding so the user
            # doesn't lose what they were typing when the agent finishes
            # mid-keystroke (the engine path).
            pending = status._pending_text
            if pending:
                self._enqueue_prompt(pending)

            engine.reset_abort_controller()

            self._engine_messages = engine.get_messages()
            # Drop per-turn *UI* buffers (spinner streaming text) before
            # the next turn starts so a long session can't accumulate
            # buffered references. Safe to call here: the LiveStatus
            # context manager has already torn down (no spinner is
            # reading these buffers), and the REPL is single-threaded
            # between turns. Note: this deliberately does NOT touch
            # ``_queued_prompts`` — those are pending user input that
            # ``run()`` will drain on the next loop iteration. See
            # ``clear_pending_turn_buffers`` for the OOM-repro rationale.
            self.clear_pending_turn_buffers()
            self._stats_turns += 1

            # Companion observer — fire per-turn reaction if relevant keywords
            # appear in the user's message. Currently a no-op until Textual
            # sprite rendering is available.
            from src.buddy.observer import fire_companion_observer

            def _no_op_reaction(quip: str | None) -> None:
                return

            fire_companion_observer(
                self.session.conversation.messages,  # type: ignore[attr-defined]
                _no_op_reaction,
            )

            if not last_text_was_printed and response_text:
                self.console.print(Markdown(response_text))
            self.console.print()

            # Per-turn save: persist JSONL transcript only (lightweight).
            try:
                self.session.save_transcript()
            except Exception:  # nosec B110
                pass

            # If Ctrl+B was pressed during the engine run, raise
            # BackgroundEscape *after* the LiveStatus is torn down
            # and the engine's abort controller is reset.  This keeps
            # the background-fork logic out of the LiveStatus handler.
            if _background_requested_engine:
                raise BackgroundEscape()

        except BackgroundEscape:
            self._last_chat_outcome = "cancelled"
            self._handle_background_escape()
            return False
        except Exception as e:
            if self._last_chat_outcome != "cancelled":
                self._last_chat_outcome = "failure"
            error_str = str(e)

            if "401" in error_str or "authentication" in error_str.lower() or "令牌" in error_str:
                self.console.print(f"\n[error]❌ Authentication Error: {escape(str(e))}[/error]")
                self.console.print("\n[warning]Your API key appears to be invalid or expired.[/warning]")

                from rich.prompt import Prompt

                choice = Prompt.ask(
                    "\nWould you like to reconfigure your API key now?",
                    choices=["y", "n"],
                    default="y",
                )

                if choice == "y":
                    self._handle_relogin()
                else:
                    self.console.print(
                        "\n[dim]You can run [bold]clawcodex login[/bold] later to update your API key.[/dim]"
                    )
            else:
                self.console.print(f"\n[error]Error: {escape(str(e))}[/error]")
                import traceback

                traceback.print_exc()
            return False

        if self._last_chat_outcome != "goal_evaluator_error":
            self._continue_goal_if_idle()
        return True

    def _print_resume_hint(self) -> None:
        """Print a hint showing the session ID for ``--resume``, matching CCB's
        ``printResumeHint()``. Delegates to the centralised helper so the
        inline ``/exit`` path and the atexit cleanup share one
        implementation (including the process-wide idempotency latch).
        """
        from clawcodex_ext.utils.resume_hint import print_resume_hint

        print_resume_hint(getattr(self.session, "session_id", None))

    def _handle_background_escape(self) -> None:
        """Handle Ctrl+B background escape: fork the agent into a background process.

        Called when ``chat()`` catches a :class:`BackgroundEscape`.  Saves
        the session, calls :func:`launch_background_runner` to fork (Unix)
        or spawn (Windows) a child that continues the agent loop headlessly,
        then prints a resume hint so the user can re-attach later with
        ``--resume <session_id>``.

        The ``_print_resume_hint`` call claims the process-wide latch in
        ``clawcodex_ext.utils.resume_hint``, suppressing the duplicate
        ``Resume this session with: ...`` line that the atexit cleanup
        would otherwise emit on top of the inline hint here.
        """
        from src.agent.background_runner import launch_background_runner

        # Save the conversation state so the child process can pick up
        # where the parent left off.
        try:
            self.session.save()
        except Exception:  # nosec B110
            pass

        # Determine max_turns for the background runner.  In interactive
        # mode there is no limit (None), matching the REPL's default.
        pid = launch_background_runner(
            session=self.session,
            provider=self.provider,
            tool_registry=self.tool_registry,
            tool_context=self.tool_context,
            max_turns=0,  # 0 = unlimited in the headless runner
        )

        if pid is not None:
            self.console.print(f"\n[success]⏎ Agent sent to background (pid {pid}).[/success]")
            self.console.print("[dim]Exiting...[/dim]")
            self._print_resume_hint()
            raise SystemExit(0)
        else:
            # Windows graceful degradation — no os.fork(), subprocess
            # launch may also have failed.
            self.console.print("\n[warning]Background mode is not supported on this platform.[/warning]")
            self.console.print("[dim]Press Ctrl+C to cancel the current run instead.[/dim]")

    def _handle_relogin(self):
        """Handle re-authentication when credentials fail."""
        from rich.prompt import Prompt
        from src.config import set_api_key, set_default_provider
        from src.providers import PROVIDER_INFO

        self.console.print("\n[bold][primary]Reconfigure Provider Credentials[/primary][/bold]\n")

        provider_names = list(PROVIDER_INFO.keys())
        self.console.print("[bold]Available providers:[/bold]")
        for name, info in PROVIDER_INFO.items():
            self.console.print(f"  [info]{name}[/info] - {info['label']} (default model: {info['default_model']})")
        self.console.print()

        provider = Prompt.ask(
            "Select LLM provider",
            choices=provider_names,
            default=self.provider_name if self.provider_name in provider_names else "anthropic",
        )

        info = PROVIDER_INFO[provider]

        if provider == "openai-codex":
            from src.auth.codex_oauth import login_codex_device_flow
            from src.config import get_provider_config

            login_codex_device_flow(console=self.console)
            config = get_provider_config(provider)
            self.console.print(f"\n[dim]Available models:[/dim] {', '.join(info['available_models'])}")
            self.console.print(f"[dim]Default:[/dim] [bold]{info['default_model']}[/bold]")
            default_model = Prompt.ask(
                f"{provider.upper()} Default Model",
                default=config.get("default_model") or info["default_model"],
            )
            set_api_key(
                provider,
                api_key="",
                base_url=config.get("base_url") or info["default_base_url"],
                default_model=default_model,
            )
            set_default_provider(provider)
            self.console.print("\n[success]OpenAI Codex login completed successfully![/success]\n")
        else:
            api_key = Prompt.ask(f"Enter {provider.upper()} API Key", password=True)

            if not api_key:
                self.console.print("\n[error]Error: API Key cannot be empty[/error]")
                return

            self.console.print(f"\n[dim]Default:[/dim] {info['default_base_url']}")
            base_url = Prompt.ask(f"{provider.upper()} Base URL", default=info["default_base_url"])

            self.console.print(f"\n[dim]Available models:[/dim] {', '.join(info['available_models'])}")
            self.console.print(f"[dim]Default:[/dim] [bold]{info['default_model']}[/bold]")
            default_model = Prompt.ask(f"{provider.upper()} Default Model", default=info["default_model"])

            set_api_key(provider, api_key=api_key, base_url=base_url, default_model=default_model)
            set_default_provider(provider)

            self.console.print(f"\n[success]{provider.upper()} API Key updated successfully![/success]\n")

        self.provider = build_provider_from_config(provider)
        self.provider_name = provider

        # Rebuild tool registry with new provider so Agent tool works
        def _get_mcp_servers_for_prompt() -> list[str]:
            ctx = getattr(self, "tool_context", None)
            if ctx is None:
                return []
            clients = getattr(ctx, "mcp_clients", None) or {}
            return list(clients.keys())

        self.tool_registry = build_default_registry(
            provider=self.provider,
            get_available_mcp_servers=_get_mcp_servers_for_prompt,
            defer_extended_tools=True,
        )

        self.console.print("[success]✓ Provider reinitialized. You can continue chatting![/success]\n")

    def save_session(self):
        """Save current session."""
        self.session.save()
        self.console.print(f"[success]Session saved: {self.session.session_id}[/success]")

    def load_session(self, session_id: str):
        """Load a previous session.

        Ch03 round-2 (R2.2): delegates to ``Session.resume`` so the
        bootstrap singleton's session id and cost counters are updated
        in lockstep with the on-disk reconstruction. Without this
        wiring the loaded conversation persists under the loaded id
        but every bootstrap reader still sees the bootstrap-generated
        UUID, and ``total_cost_usd`` restarts at 0.

        Args:
            session_id: Session ID to load
        """
        from src.agent import Session
        from src.bootstrap.state import get_total_cost_usd

        loaded_session = Session.resume(session_id)
        if loaded_session is None:
            self.console.print(f"[error]Session not found: {session_id}[/error]")
            return

        # Replace current session (bootstrap id + cost already restored
        # by Session.resume).
        self.session = loaded_session
        from clawcodex_ext.runtime.tool_context_binding import bind_tool_context_runtime

        bind_tool_context_runtime(
            self.tool_context,
            tool_registry=self.tool_registry,
            session=self.session,
            provider=self.provider,
        )
        # Goals follow the resumed session rather than any stale protocol
        # override left on an injected ToolContext.
        self.tool_context.goal_thread_id = None
        try:
            from clawcodex_ext.goal.runtime import (
                restore_goal_runtime_after_session_resume,
            )

            restore_goal_runtime_after_session_resume(self.tool_context)
        except Exception:  # nosec B110
            # Session loading must remain usable when the optional goal store
            # is disabled or unavailable.
            pass

        command_context = getattr(self, "command_context", None)
        if command_context is not None:
            command_context.session = loaded_session
            command_context.conversation = loaded_session.conversation
            command_context.tool_context = self.tool_context
        # Populate _engine_messages from the restored conversation so the
        # next chat() call's QueryEngine sees the full history rather than
        # starting with an empty mutable-message list (which would cause
        # the model to lose all prior context on resume).
        self._engine_messages = list(loaded_session.conversation.messages)
        self.console.print(f"[success]Session loaded: {session_id}[/success]")
        self.console.print(f"[dim]Provider: {loaded_session.provider}, Model: {loaded_session.model}[/dim]")
        self.console.print(f"[dim]Messages: {len(loaded_session.conversation.messages)}[/dim]")
        restored_cost = get_total_cost_usd()
        if restored_cost > 0:
            self.console.print(f"[dim]Restored cost: ${restored_cost:.4f}[/dim]")

        # Show last 5 messages; meta/virtual injections and empty turns are skipped.
        if loaded_session.conversation.messages:
            self.console.print("\n[bold]Conversation History:[/bold]")
            for msg in loaded_session.conversation.messages[-5:]:
                line = self._format_history_line(msg)
                if line is None:
                    continue
                self.console.print(line)
