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

"""Prompt input widget — bottom ``❯`` line with slash palette + history.

Port of ``typescript/src/components/PromptInput/PromptInput.tsx`` at the
fidelity required to feel like the ink reference:

* Multi-line editing via ``Shift+Enter`` / ``Ctrl+J`` (Textual's default)
  with plain ``Enter`` submitting the prompt.
* Slash-command palette opens when the current token starts with ``/``
  and fuzzy-filters as the user types.
* ``@``-file completions open when the current token starts with ``@``,
  using the same matching logic as the REPL (``src/utils/at_file_completer``).
* Up / Down navigate the in-session history when the palette is closed;
  when it is open, arrow keys drive the option list.
* ``Escape`` closes the palette without losing the draft.
* ``Ctrl+L`` clears the draft.

Phase 1 deliberately keeps the implementation single-line-in-practice
(using Textual's ``Input``) but exposes :meth:`set_multiline` for
Phase 2 to swap in a ``TextArea`` without changing the public surface.
"""

from __future__ import annotations
# pylint: disable=C0302,E0611,R1702,W0201

import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from rich.text import Text
from textual import events
from textual.app import ComposeResult
from textual.containers import Vertical
from textual.message import Message
from textual.widgets import Input, OptionList, Static
from textual.widgets.option_list import Option

from ..commands import CommandSuggestion
from ..messages import (
    CancelRequested,
    PermissionModeCycleRequested,
    PromptDraftChanged,
    PromptPasted,
)
from ..paste import PasteInfo, classify_paste
from ..vim import VimState
from .prompt_input_footer import PromptInputFooter
from .prompt_input_mode_indicator import PromptInputModeIndicator
from ..rainbow_highlight import highlight_triggers
from clawcodex_ext.services.ultraplan.feature_gates import is_ultraplan_rainbow_enabled
from clawcodex_ext.services.ultraplan.keyword_detector import find_ultraplan_trigger_positions

from clawcodex_ext.utils.completers import (
    current_slash_token,
    fuzzy_match,
    rank_message_history,
    rank_suggestions,
)


# Reuse the same matching machinery as the prompt_toolkit REPL.
# ``_AT_TOKEN_RE`` matches ``@<query>`` at the cursor position;
# ``_is_path_like_token`` / ``_path_completions`` handle absolute
# and explicit-relative paths by walking the filesystem directly;
# ``_filter_candidates`` ranks project-file hits by substring,
# subsequence and fuzzy signals with a 26-bit bitmap pre-filter.
# ``_list_git_files`` / ``_walk_filesystem`` provide the candidate list.
from src.utils.at_file_completer import (  # type: ignore[import-untyped]
    _AT_TOKEN_RE,
    _build_path_bitmap,
    _filter_candidates,
    _is_path_like_token,
    _list_git_files,
    _MAX_SUGGESTIONS as _AT_FILE_MAX_SUGGESTIONS,
    _path_completions,
    _walk_filesystem,
)

# How long we cache the project-file index (same as ``AtFileCompleter``).
_FILE_CACHE_TTL = 5.0


@dataclass
class PromptSubmitted(Message):
    """User pressed Enter on a non-empty prompt."""

    text: str


class _PasteAwareInput(Input):
    """``Input`` subclass that routes bracketed paste through the host.

    The stock Textual ``Input._on_paste`` truncates the payload to the
    first line and bypasses any custom paste handler on the parent
    widget. That destroys the very property chapter 14 calls out as
    "critical for security" — the bracketed-paste envelope must travel
    intact so the host can decide whether it is a literal text insert,
    an image-file drag, or an empty-paste image-clipboard sentinel.

    This subclass intercepts ``events.Paste`` before the stock handler
    runs and forwards it to the surrounding :class:`PromptInput` via
    ``parent.handle_paste``. The Input itself does not insert anything;
    the host owns the buffer mutation so vim-mode shims, slash-popup
    suppression, and history-pointer hygiene all happen in one place.
    """

    def _on_paste(self, event: events.Paste) -> None:  # type: ignore[override]
        # Walk up the widget tree looking for a parent that knows how
        # to absorb a bracketed paste. ``hasattr`` avoids a forward
        # reference to :class:`PromptInput` (which is declared below
        # to keep the file readable top-down).
        node = self.parent
        for _ in range(8):  # bounded walk; the tree is always shallow.
            if node is None:
                break
            handler = getattr(node, "handle_paste", None)
            if callable(handler):
                handler(event.text)
                # ``prevent_default`` is what stops Textual's MRO
                # walker from also invoking the stock
                # ``Input._on_paste`` (which would truncate the
                # payload to the first line and double-insert).
                # ``stop`` then halts bubbling to ancestor widgets so
                # the host does not see the raw Paste twice.
                event.prevent_default()
                event.stop()
                return
            node = getattr(node, "parent", None)
        # Fallback to the stock behaviour if we've been re-parented
        # outside a :class:`PromptInput` host.
        super()._on_paste(event)


_NAME_COLUMN_WIDTH = 22  # left column reserved for "/name (alias)"
_MAX_VISIBLE_SUGGESTIONS = 6

# Paste placeholder thresholds — when a paste exceeds these, the input
# shows a compact ``[Pasted text #N +K lines]`` placeholder instead of
# dumping raw content into the single-line buffer.
_PASTE_CHARS_THRESHOLD = 800
_PASTE_NEWLINES_THRESHOLD = 2
_PASTE_PLACEHOLDER_RE = re.compile(r"\[Pasted text #(\d+)(?: \+(\d+) lines)?\]")


class _SlashSuggestions(OptionList):
    DEFAULT_CSS = """
    _SlashSuggestions {
        max-height: 10;
        border: none;
        padding: 0;
        background: $background;
    }
    _SlashSuggestions > .option-list--option {
        padding: 0 1;
    }
    _SlashSuggestions > .option-list--option-highlighted {
        background: $boost;
        text-style: bold;
    }
    _SlashSuggestions.-hidden {
        display: none;
    }
    """


class _MessageSuggestions(OptionList):
    """Popup for message-history completions."""

    DEFAULT_CSS = """
    _MessageSuggestions {
        max-height: 6;
        border: none;
        padding: 0;
        background: $background;
        height: auto;
    }
    _MessageSuggestions > .option-list--option {
        padding: 0 1;
    }
    _MessageSuggestions > .option-list--option-highlighted {
        background: $boost;
        text-style: bold;
    }
    _MessageSuggestions.-hidden {
        display: none;
    }
    """


class _AtFileSuggestions(OptionList):
    """Popup for ``@`` file-path completions."""

    DEFAULT_CSS = """
    _AtFileSuggestions {
        max-height: 12;
        border: none;
        padding: 0;
        background: $background;
        height: auto;
    }
    _AtFileSuggestions > .option-list--option {
        padding: 0 1;
    }
    _AtFileSuggestions > .option-list--option-highlighted {
        background: $boost;
        text-style: bold;
    }
    _AtFileSuggestions.-hidden {
        display: none;
    }
    """


class _AgentSuggestions(OptionList):
    """Popup for ``@agent-<type>`` completions."""

    DEFAULT_CSS = """
    _AgentSuggestions {
        max-height: 12;
        border: none;
        padding: 0;
        background: $background;
        height: auto;
    }
    _AgentSuggestions > .option-list--option {
        padding: 0 1;
    }
    _AgentSuggestions > .option-list--option-highlighted {
        background: $boost;
        text-style: bold;
    }
    _AgentSuggestions.-hidden {
        display: none;
    }
    """


def _current_at_token(text_before_cursor: str) -> str | None:
    """Return the query after ``@`` at the current cursor position, or None.

    A valid ``@`` token must be at the start of the buffer or preceded
    by whitespace — ``foo@bar`` (e.g. an email address) does not count.
    This mirrors the same rule in ``AtFileCompleter.get_completions``.
    """
    match = _AT_TOKEN_RE.search(text_before_cursor)
    if match is None:
        return None
    at_pos = match.start()
    if at_pos > 0 and not text_before_cursor[at_pos - 1].isspace():
        return None
    return match.group(1)


# Regex to detect ``@agent-<partial>`` mention tokens for the TUI.
# Same pattern used by the prompt_toolkit ``AgentMentionCompleter``.
_AGENT_COMPLETE_RE = re.compile(r"(?:^|(?<=\s))@(agent-[\w:.@\-]*)$")


def _current_agent_token(text_before_cursor: str) -> str | None:
    """Return the partial agent type after ``@agent-``, or None if the
    cursor is not on an ``@agent-`` token.

    A valid token must be at the start of the buffer or preceded by
    whitespace — ``foo@agent-bar`` does not count.
    """
    match = _AGENT_COMPLETE_RE.search(text_before_cursor)
    if match is None:
        return None
    at_pos = match.start()
    if at_pos > 0 and not text_before_cursor[at_pos - 1].isspace():
        return None
    token = match.group(1)  # e.g. "agent-explor" or "agent-"
    return token[len("agent-") :]  # e.g. "explor" or ""


class PromptInput(Vertical):
    """Input line plus slash-command suggestion popup."""

    DEFAULT_CSS = """
    PromptInput {
        height: auto;
        padding: 0 0;
    }
    PromptInput > Input {
        border: round $primary-darken-2;
        padding: 0 1;
    }
    PromptInput > #ghost-suggestion {
        height: auto;
        padding: 0 2;
        color: $text 40%;
    }
    PromptInput > #ghost-suggestion.-hidden {
        display: none;
    }
    PromptInput > #ultraplan-trigger-preview {
        height: auto;
        padding: 0 2;
        color: $text 70%;
    }
    PromptInput > #ultraplan-trigger-preview.-hidden {
        display: none;
    }
    """

    BINDINGS = [
        ("ctrl+l", "clear_draft", "Clear draft"),
    ]

    def __init__(
        self,
        *,
        words_provider: Callable[[], list[str]],
        suggestions_provider: Callable[[], list[CommandSuggestion]] | None = None,
        message_history_provider: Callable[[], list[str]] | None = None,
        agents_provider: Callable[[], list[Any]] | None = None,
        files_provider: Callable[[], list[str]] | None = None,
        cwd: str | os.PathLike[str] | None = None,
        vim_mode: bool = False,
        initial_history: list[str] | None = None,
    ) -> None:
        super().__init__()
        self._words_provider = words_provider
        self._suggestions_provider = suggestions_provider
        self._message_history_provider = message_history_provider
        self._agents_provider = agents_provider
        self._files_provider = files_provider
        self._message_completions: list[str] = []
        self._message_completion_pos: int | None = None
        # Seed with cross-session history from HistoryStore (most recent
        # entries first, so ↑/↓ order matches submit order).
        self._history: list[str] = list(initial_history or [])
        self._history_pos: int | None = None
        self._input = _PasteAwareInput(placeholder="Type a prompt, or / for commands")
        self._suggestions = _SlashSuggestions(classes="-hidden")
        self._message_suggestions = _MessageSuggestions(classes="-hidden")
        self._at_file_suggestions = _AtFileSuggestions(classes="-hidden")
        self._agent_suggestions = _AgentSuggestions(classes="-hidden")
        self._ghost_suggestion = Static("", id="ghost-suggestion", classes="-hidden")
        self._ghost_suggestion.renderable = ""
        self._ultraplan_trigger_preview = Static(
            "",
            id="ultraplan-trigger-preview",
            classes="-hidden",
            markup=False,
        )
        self._ultraplan_trigger_preview.renderable = ""
        self._vim = VimState(enabled=vim_mode)
        self._yank_buffer: str = ""
        # Round 2 / WI-R2.5: most-recent bracketed paste classification.
        # Test seam — the host reads :class:`PromptPasted` instead.
        self._last_paste: PasteInfo | None = None
        # Paste placeholder state: blobs store original text indexed by
        # auto-increment ID; the input displays a compact placeholder.
        self._paste_blobs: dict[int, str] = {}
        self._paste_counter: int = 0
        # Round 2 / WI-R2.4: passive status surfaces around the input.
        # The mode indicator subscribes to ``VimState`` directly; the
        # footer reads ``VimState.enabled`` lazily. Both mounted as
        # siblings of the existing ``Input`` + ``OptionList`` pair.
        self._mode_indicator = PromptInputModeIndicator(vim_state=self._vim)
        self._footer = PromptInputFooter(vim_state=self._vim)
        # ---- ``@`` file-completion state ----
        self._cwd = Path(cwd or os.getcwd()).resolve()
        self._file_cache: list[str] = []
        self._file_cache_bitmaps: list[int] = []
        self._file_cache_built_at: float = 0.0
        # ---- suggest mode: tracks which popup type is active for
        #      unified key handling (F-38 Enter/Tab separation) ----
        self._suggest_mode: str = ""  # "slash" | "at_file" | "at_agent" | "message" | ""

    def compose(self) -> ComposeResult:
        yield self._mode_indicator
        yield self._input
        yield self._ultraplan_trigger_preview
        yield self._ghost_suggestion
        yield self._suggestions
        yield self._message_suggestions
        yield self._at_file_suggestions
        yield self._agent_suggestions
        yield self._footer

    def on_mount(self) -> None:
        self._input.focus()

    # ---- external API ----
    def focus_input(self) -> None:
        self._input.focus()

    def clear(self) -> None:
        self._input.value = ""
        self._hide_suggestions()
        self._hide_message_suggestions()
        self._hide_ghost_suggestion()
        self._hide_ultraplan_trigger_preview()

    def set_value(self, value: str) -> None:
        """Replace the draft text in the prompt (used by /history)."""

        self._input.value = value or ""
        self._hide_suggestions()
        self._hide_message_suggestions()
        self._hide_ghost_suggestion()
        self._refresh_ultraplan_trigger_preview(self._input.value)

    # ---- bracketed paste ----
    def handle_paste(self, text: str) -> PasteInfo:
        """Insert a bracketed-paste payload as a single atomic operation.

        Bypasses the slash-popup recomputer, the history pointer, and
        the vim chord tracker — pasted characters must always reach the
        buffer literally even when they happen to spell a chord prefix.
        Chapter 14 calls this the ``isPasted`` invariant; the Python
        port honours it by funnelling every bracketed paste through
        this single entry point.

        Large pastes (over 800 chars or 3+ lines) are stored in
        ``_paste_blobs`` and shown as a ``[Pasted text #N +K lines]``
        placeholder instead of dumping raw content into the single-line
        input buffer. The placeholder is expanded back to the original
        text in :meth:`expand_pastes` before submission.

        Returns the :class:`PasteInfo` so callers (typically tests) can
        assert on classification without having to listen for the
        :class:`PromptPasted` message.
        """

        info = classify_paste(text)
        self._last_paste = info
        if not info.is_empty:
            inp = self._input
            value = inp.value or ""
            pos = max(0, min(inp.cursor_position, len(value)))
            if info.length > _PASTE_CHARS_THRESHOLD or info.line_count > _PASTE_NEWLINES_THRESHOLD:
                # Large paste: store blob, show placeholder
                self._paste_counter += 1
                blob_id = self._paste_counter
                self._paste_blobs[blob_id] = info.text
                extra = f" +{info.line_count} lines" if info.line_count > 1 else ""
                placeholder = f"[Pasted text #{blob_id}{extra}]"
                new_value = value[:pos] + placeholder + value[pos:]
                inp.value = new_value
                inp.cursor_position = pos + len(placeholder)
            else:
                new_value = value[:pos] + info.text + value[pos:]
                inp.value = new_value
                inp.cursor_position = pos + info.length
        # Pasted content must never reopen the slash palette; the user
        # paste-bombed the prompt and likely wants to keep typing or
        # submit. Same rationale as the TS ``usePasteHandler`` reset.
        self._hide_suggestions()
        # Paste must never disturb the history cursor — leaving
        # ``_history_pos`` untouched here is the explicit contract.
        self.post_message(PromptPasted(info=info))
        return info

    @property
    def last_paste(self) -> PasteInfo | None:
        """Most recent classified paste (test seam)."""

        return self._last_paste

    def expand_pastes(self, text: str) -> str:
        """Replace all paste placeholders with their original content.

        Called from :meth:`on_input_submitted` so the model sees
        the full pasted text, not the ``[Pasted text #N]`` placeholder.
        Unknown / expired blob IDs are left as-is (defensive).
        """

        def _replace(m: re.Match) -> str:
            blob_id = int(m.group(1))
            orig = self._paste_blobs.get(blob_id)
            if orig is not None:
                return orig
            return m.group(0)  # unknown blob — leave placeholder intact

        return _PASTE_PLACEHOLDER_RE.sub(_replace, text)

    def _splice_at_mention(self, selected_id: str) -> None:
        """Replace the ``@<token>`` under the cursor with the chosen ID.

        Called when the user presses Enter/Tab on a highlighted @file
        or @agent suggestion.  Finds the ``@`` token at the cursor
        position and replaces the entire token (including the ``@``
        prefix) with the selected completion text, leaving the cursor
        at the end of the replacement so the user can keep typing.
        """
        inp = self._input
        value = inp.value or ""
        pos = inp.cursor_position
        # Walk backward from cursor to find the start of the @token
        start = pos
        while start > 0 and not value[start - 1].isspace():
            start -= 1
        # Walk forward from cursor to find the end of the @token
        end = pos
        while end < len(value) and not value[end].isspace():
            end += 1
        # Replace the entire @token with the selected completion
        prefix = value[:start]
        suffix = value[end:]
        inp.value = prefix + selected_id + suffix
        inp.cursor_position = len(prefix) + len(selected_id)

    def action_clear_draft(self) -> None:
        self.clear()

    def set_enabled(self, enabled: bool) -> None:
        """Enable / disable the input (used when a modal steals focus)."""

        self._input.disabled = not enabled
        if enabled:
            self._input.focus()

    # ---- vim mode ----
    def set_vim_mode(self, enabled: bool) -> None:
        """Toggle vim-mode on the prompt."""

        self._vim.set_enabled(enabled)
        # ``VimState`` listeners only fire on actual mode transitions
        # (Round 2 / WI-R2.1), so the enable/disable flip itself needs
        # an explicit refresh on the surrounding status surfaces.
        try:
            self._mode_indicator.refresh_mode()
            self._footer.refresh_hints()
        except Exception:  # nosec B110
            # Pre-mount: ``compose`` will render with the right state.
            pass

    @property
    def vim_mode(self) -> bool:
        return self._vim.enabled

    @property
    def vim_state(self) -> VimState:  # exposed for tests / status line
        return self._vim

    # ---- input events ----
    def on_input_changed(self, event: Input.Changed) -> None:
        self.post_message(PromptDraftChanged(text=event.value or ""))
        self._refresh_ultraplan_trigger_preview(event.value or "")
        # When the user is navigating the in-session history with
        # Up/Down, :meth:`_navigate_history` programmatically sets the
        # input value to a previous prompt. Because Textual fires
        # ``Input.Changed`` for every value assignment — programmatic
        # or not — the usual handler would call
        # :meth:`_refresh_message_suggestions` and pop a multi-row
        # history-completion OptionList under the field. The user did
        # not ask for completions; they asked to recall. Showing the
        # popup there renders ~6 rows of empty-looking space below the
        # field (perceived as "extra blank lines"), and the ghost-text
        # hint would also flash a longer history match under the
        # field. Suppress all suggestion surfaces until the user
        # diverges from the recalled entry.
        if self._history_pos is not None:
            if self._history_pos >= len(self._history) or event.value != self._history[self._history_pos]:
                # The user has typed past the recalled entry — drop out
                # of history-navigation mode so subsequent keystrokes
                # behave normally.
                self._history_pos = None
            else:
                self._hide_suggestions()
                self._hide_message_suggestions()
                self._hide_at_file_suggestions()
                self._hide_agent_suggestions()
                self._hide_ghost_suggestion()
                self._hide_ultraplan_trigger_preview()
                return
        self._refresh_suggestions(event.value, event.input.cursor_position)
        # Show ghost-text suggestion from history when no popup is open.
        if (
            self._suggestions.has_class("-hidden")
            and self._message_suggestions.has_class("-hidden")
            and self._at_file_suggestions.has_class("-hidden")
            and self._agent_suggestions.has_class("-hidden")
        ):
            self._refresh_ghost_suggestion(event.value or "")
        else:
            self._hide_ghost_suggestion()

    async def on_input_submitted(self, event: Input.Submitted) -> None:
        raw = (event.value or "").strip()
        text = self.expand_pastes(raw)
        if not text:
            return

        # If any suggestion popup is open and a row is highlighted,
        # accept the selection (fill / splice) but do NOT submit —
        # the user must press Enter again to actually send.  Mirrors the
        # upstream behaviour where Enter on a highlighted row is "select
        # + keep typing", not "select + submit" (F-38 unified path).
        if self._suggest_mode:
            if self._accept_suggestion():
                return

        self._history.append(text)
        self._history_pos = None
        self._hide_suggestions()
        self._hide_at_file_suggestions()
        self._hide_agent_suggestions()
        self._hide_message_suggestions()
        self._hide_ghost_suggestion()
        self._hide_ultraplan_trigger_preview()
        self._input.value = ""
        ## _log(f'[prompt_input] posting PromptSubmitted: {text}')
        self.post_message(PromptSubmitted(text=text))

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        """Handle Enter/Space on a suggestion row — insert the command."""
        if event.option.id:
            selected_text = event.option.id
            # Check takes_args for slash commands (F-38 _arg_lookup)
            if event.option_list is self._suggestions and self._suggestions_provider is not None:
                try:
                    suggestions = self._suggestions_provider() or []
                    for s in suggestions:
                        if isinstance(s, CommandSuggestion) and s.slash == selected_text:
                            if s.takes_args and not selected_text.endswith(" "):
                                selected_text += " "
                            break
                except Exception:  # nosec B110
                    pass
            self._input.value = selected_text
            self._input.cursor_position = len(selected_text)
            # ``OptionList.OptionSelected`` exposes its source list via
            # ``option_list`` (``Message.sender`` is not an attribute on
            # the base ``Message`` in this Textual version, so the old
            # ``event.sender is …`` comparison raised AttributeError).
            if event.option_list is self._at_file_suggestions:
                self._hide_at_file_suggestions()
            elif event.option_list is self._message_suggestions:
                self._hide_message_suggestions()
            elif event.option_list is self._agent_suggestions:
                self._hide_agent_suggestions()
            else:
                self._hide_suggestions()

    async def on_key(self, event: events.Key) -> None:
        key = event.key

        # Shift+Tab: cycle permission mode. Intercept here so it works
        # even when the Input child has focus (Textual's default focus
        # navigation would otherwise consume the key before the screen
        # binding fires).
        if key == "shift+tab":
            self.post_message(PermissionModeCycleRequested())
            event.stop()
            return

        # Vim mode: consume chord-owned keys before the Input sees them.
        if self._vim.enabled:
            result = self._vim.handle(key)
            if result.consumed:
                if result.action is not None:
                    self._apply_vim_action(result.action)
                event.stop()
                return

        if key == "escape" and not self._suggestions.has_class("-hidden"):
            self._hide_suggestions()
            event.stop()
            return
        if key == "escape" and not self._message_suggestions.has_class("-hidden"):
            self._hide_message_suggestions()
            event.stop()
            return
        if key == "escape" and not self._at_file_suggestions.has_class("-hidden"):
            self._hide_at_file_suggestions()
            event.stop()
            return
        if key == "escape":
            # Bubble up to the app; it decides whether to actually
            # cancel based on whether the agent bridge is busy.
            # Mirrors the TS reference's chat:cancel keybinding.
            self.post_message(CancelRequested())
            event.stop()
            return

        # Tab: if any suggestion popup is open, accept the highlighted
        # item (F-38 Enter/Tab separation). Otherwise accept the
        # ghost-text suggestion when one is visible. If neither,
        # let the event bubble up to the App's default ``focus_next``
        # binding — crucially we do NOT call ``event.stop()`` in that
        # case so widget-to-widget focus navigation still works.
        if key == "tab":
            if self._suggest_mode:
                accepted = self._accept_suggestion()
                if accepted:
                    event.stop()
                    return
            if not self._ghost_suggestion.has_class("-hidden"):
                self._accept_ghost_suggestion()
                event.stop()
                return

        if key in ("up", "down"):
            # @ file suggestions take top priority
            if not self._at_file_suggestions.has_class("-hidden"):
                if key == "up":
                    self._at_file_suggestions.action_cursor_up()
                else:
                    self._at_file_suggestions.action_cursor_down()
                event.stop()
                return
            # Message-history popup takes second priority
            if not self._message_suggestions.has_class("-hidden"):
                if key == "up":
                    self._message_suggestions.action_cursor_up()
                else:
                    self._message_suggestions.action_cursor_down()
                event.stop()
                return
            if not self._suggestions.has_class("-hidden"):
                self._suggestions.focus()
                if key == "up":
                    self._suggestions.action_cursor_up()
                else:
                    self._suggestions.action_cursor_down()
                event.stop()
                return
            self._navigate_history(1 if key == "up" else -1)
            event.stop()
            return

    # ---- vim action application ----
    def _apply_vim_action(self, action: str) -> None:
        inp = self._input
        value = inp.value or ""
        pos = inp.cursor_position
        if action == "insert-before":
            return
        if action == "insert-after":
            inp.cursor_position = min(len(value), pos + 1)
        elif action == "insert-line-start":
            inp.cursor_position = 0
        elif action == "insert-line-end":
            inp.cursor_position = len(value)
        elif action == "move-left":
            inp.cursor_position = max(0, pos - 1)
        elif action == "move-right":
            inp.cursor_position = min(len(value), pos + 1)
        elif action == "move-start":
            inp.cursor_position = 0
        elif action == "move-end":
            inp.cursor_position = len(value)
        elif action == "move-word-next":
            inp.cursor_position = _next_word(value, pos)
        elif action == "move-word-prev":
            inp.cursor_position = _prev_word(value, pos)
        elif action == "delete-char":
            if pos < len(value):
                inp.value = value[:pos] + value[pos + 1 :]
        elif action == "delete-line":
            self._yank_buffer = value
            inp.value = ""
        elif action == "yank-line":
            self._yank_buffer = value
        elif action == "paste-after":
            if self._yank_buffer:
                inp.value = value[: pos + 1] + self._yank_buffer + value[pos + 1 :]
                inp.cursor_position = pos + 1 + len(self._yank_buffer)
        elif action == "paste-before":
            if self._yank_buffer:
                inp.value = value[:pos] + self._yank_buffer + value[pos:]
                inp.cursor_position = pos + len(self._yank_buffer)
        elif action == "submit":
            text = value.strip()
            if text:
                self._history.append(text)
                self._history_pos = None
                inp.value = ""
                self.post_message(PromptSubmitted(text=text))

    # ---- ghost-text suggestion from history ----

    def _find_history_suggestion(self, text: str) -> str | None:
        """Return the most recent history entry starting with *text*, or None."""
        if not text:
            return None
        for entry in reversed(self._history):
            if entry.startswith(text) and entry != text:
                return entry
        return None

    def _refresh_ghost_suggestion(self, text: str) -> None:
        """Show a dim ghost-text suggestion from history below the input."""
        match = self._find_history_suggestion(text)
        if match is not None:
            suffix = match[len(text) :]
            # Show "TAB to accept" hint next to the dimmed completion.
            hint = Text()
            hint.append(suffix, style="dim")
            hint.append(" (TAB to accept)", style="dim cyan")
            self._ghost_suggestion.renderable = hint
            self._ghost_suggestion.update(hint)
            self._ghost_suggestion.remove_class("-hidden")
        else:
            self._hide_ghost_suggestion()

    def _hide_ghost_suggestion(self) -> None:
        if not self._ghost_suggestion.has_class("-hidden"):
            self._ghost_suggestion.add_class("-hidden")
            self._ghost_suggestion.renderable = ""
            self._ghost_suggestion.update("")

    def _refresh_ultraplan_trigger_preview(self, text: str) -> None:
        if not text or not is_ultraplan_rainbow_enabled():
            self._hide_ultraplan_trigger_preview()
            return
        hits = find_ultraplan_trigger_positions(text)
        leading = len(text) - len(text.lstrip())
        if not hits or hits[0].start != leading:
            self._hide_ultraplan_trigger_preview()
            return
        rendered = Text("ultraplan: ", style="dim")
        rendered.append_text(highlight_triggers(text, hits))
        self._ultraplan_trigger_preview.renderable = rendered
        self._ultraplan_trigger_preview.update(rendered)
        self._ultraplan_trigger_preview.remove_class("-hidden")

    def _hide_ultraplan_trigger_preview(self) -> None:
        if not self._ultraplan_trigger_preview.has_class("-hidden"):
            self._ultraplan_trigger_preview.add_class("-hidden")
            self._ultraplan_trigger_preview.renderable = ""
            self._ultraplan_trigger_preview.update("")

    def _accept_ghost_suggestion(self) -> None:
        """Accept the ghost-text suggestion, appending it to the input."""
        text = self._input.value or ""
        match = self._find_history_suggestion(text)
        if match is not None:
            self._input.value = match
            self._input.cursor_position = len(match)
            self._hide_ghost_suggestion()

    # ---- suggestion plumbing ----
    def _refresh_suggestions(self, text: str, cursor: int) -> None:
        token, _ = _current_slash_token(text[:cursor])
        if token is not None:
            # Slash mode: show command suggestions
            self._hide_message_suggestions()
            self._hide_at_file_suggestions()
            self._hide_agent_suggestions()
            partial = token[1:].lower()

            options = self._build_suggestion_options(partial)
            if not options:
                self._hide_suggestions()
                self._suggest_mode = ""
                return
            self._suggestions.clear_options()
            self._suggestions.add_options(options)
            self._suggestions.highlighted = 0
            self._suggestions.remove_class("-hidden")
            self._suggest_mode = "slash"
            return

        # Check for ``@agent-<type>`` mention token (takes priority over
        # plain ``@`` file completions so the agent names popup doesn't
        # compete with file suggestions for the same token).
        agent_query = _current_agent_token(text[:cursor])
        if agent_query is not None:
            self._hide_suggestions()
            self._hide_message_suggestions()
            self._hide_at_file_suggestions()
            self._refresh_agent_suggestions(agent_query)
            return

        # Check for ``@`` file-completion token
        at_query = _current_at_token(text[:cursor])
        if at_query is not None:
            self._hide_suggestions()
            self._hide_message_suggestions()
            self._hide_agent_suggestions()
            self._refresh_at_file_suggestions(at_query)
            return

        # Non-slash, non-@ mode: hide all popups, check message history
        self._hide_suggestions()
        self._hide_at_file_suggestions()
        self._hide_agent_suggestions()
        self._refresh_message_suggestions(text, cursor)

    def _build_suggestion_options(self, partial: str) -> list[Option]:
        """Return the rich Option rows to show under the prompt.

        Prefers the rich ``suggestions_provider`` (two-column ``/name +
        description`` layout to match the TS reference); falls back to
        the ``words_provider`` string list when no rich source exists
        (legacy tests, embedded uses).
        """

        if self._suggestions_provider is not None:
            try:
                suggestions = self._suggestions_provider() or []
            except Exception:
                suggestions = []
            return _options_from_suggestions(suggestions, partial)

        words = self._words_provider() or []
        return _options_from_words(words, partial)

    def _hide_suggestions(self) -> None:
        if not self._suggestions.has_class("-hidden"):
            self._suggestions.add_class("-hidden")
            self._suggestions.clear_options()
        if self._suggest_mode == "slash":
            self._suggest_mode = ""

    def _refresh_message_suggestions(self, text: str, cursor: int) -> None:
        """Refresh message-history completions when not in slash mode.

        Collects the current token (non-whitespace word) under the cursor.
        If it matches the start of any previous user message, shows them
        in a popup below the input line.
        """
        if not self._message_history_provider:
            self._hide_message_suggestions()
            return

        # Guard against stale cursor positions (e.g. after programmatic
        # ``_input.value = ""`` while the widget still reports the old
        # cursor offset). Without this, ``prefix[i]`` below IndexErrors.
        if cursor > len(text):
            self._hide_message_suggestions()
            return

        # Extract the current token (non-whitespace word) under cursor.
        prefix = text[:cursor]
        # Find the start of the current token
        i = cursor - 1
        while i >= 0 and not prefix[i].isspace():
            i -= 1
        current_token = prefix[i + 1 : cursor]

        if not current_token:
            self._hide_message_suggestions()
            return

        token_lower = current_token.lower()

        try:
            history = self._message_history_provider() or []
        except Exception:
            self._hide_message_suggestions()
            return

        # Rank via shared helper. The TUI surfaces up to
        # ``_MAX_VISIBLE_SUGGESTIONS`` (6) entries to keep the popup compact.
        ranked = rank_message_history(history, token_lower, limit=_MAX_VISIBLE_SUGGESTIONS)

        if not ranked:
            self._hide_message_suggestions()
            return

        self._message_suggestions.clear_options()
        for full_msg in ranked:
            display = full_msg[:100] + ("..." if len(full_msg) > 100 else "")
            # Textual 0.79 的 OptionList.add_option 不再接受 id=
            # kwarg；通过 Option 包装传入。id 是选中后回填到 input
            # 的全文（display 仅作展示截断）。
            self._message_suggestions.add_option(Option(display, id=full_msg))
        self._message_suggestions.highlighted = 0
        self._message_suggestions.remove_class("-hidden")
        self._suggest_mode = "message"

    def _hide_message_suggestions(self) -> None:
        if not self._message_suggestions.has_class("-hidden"):
            self._message_suggestions.add_class("-hidden")
            self._message_suggestions.clear_options()
        if self._suggest_mode == "message":
            self._suggest_mode = ""

    # ---- unified suggestion accept ----

    def _accept_suggestion(self) -> bool:
        """Accept the currently highlighted suggestion from whichever popup
        is open. Returns True if a suggestion was accepted, False if no
        popup is visible or nothing was highlighted.

        This is the single entry point for both Enter (when a popup is open)
        and Tab (F-38) — the caller decides whether to stop the key event.
        """
        # Slash command popup
        if not self._suggestions.has_class("-hidden"):
            idx = self._suggestions.highlighted
            if idx is not None:
                option = self._suggestions.get_option_at_index(idx)
                if option is not None and option.id:
                    # Check if the selected command takes arguments:
                    # if so, append a trailing space so the user can type
                    # args immediately (F-38 _arg_lookup).
                    selected_text = option.id
                    current_text = (self._input.value or "").strip()
                    takes_args = False
                    # Check suggestions_provider for takes_args info
                    if self._suggestions_provider is not None:
                        try:
                            suggestions = self._suggestions_provider() or []
                            for s in suggestions:
                                if isinstance(s, CommandSuggestion) and s.slash == selected_text:
                                    takes_args = bool(s.takes_args)
                                    break
                        except Exception:  # nosec B110
                            pass
                    if current_text == selected_text and not takes_args:
                        self._hide_suggestions()
                        return False
                    if takes_args and not selected_text.endswith(" "):
                        selected_text += " "
                    self._input.value = selected_text
                    self._input.cursor_position = len(selected_text)
                    self._hide_suggestions()
                    return True
            return False

        # @ file suggestions → splice
        if not self._at_file_suggestions.has_class("-hidden"):
            idx = self._at_file_suggestions.highlighted
            if idx is not None:
                option = self._at_file_suggestions.get_option_at_index(idx)
                if option is not None and option.id:
                    self._splice_at_mention(option.id)
                    self._hide_at_file_suggestions()
                    return True
            return False

        # @ agent suggestions → splice
        if not self._agent_suggestions.has_class("-hidden"):
            idx = self._agent_suggestions.highlighted
            if idx is not None:
                option = self._agent_suggestions.get_option_at_index(idx)
                if option is not None and option.id:
                    self._splice_at_mention(option.id)
                    self._hide_agent_suggestions()
                    return True
            return False

        # Message-history suggestions → fill input
        if not self._message_suggestions.has_class("-hidden"):
            idx = self._message_suggestions.highlighted
            if idx is not None:
                option = self._message_suggestions.get_option_at_index(idx)
                if option is not None and option.id:
                    self._input.value = option.id
                    self._input.cursor_position = len(option.id)
                    self._hide_message_suggestions()
                    return True
            return False

        return False

    # ---- ``@`` file suggestion plumbing ----

    def _get_cached_file_list(self) -> tuple[list[str], list[int]]:
        """Return a snapshot of the project-file index, rebuilding if stale.

        Mirrors the 5-second TTL of ``AtFileCompleter._candidates_snapshot``.
        Returns ``(paths, bitmaps)`` — the bitmaps are 26-bit ints encoding
        lowercase a-z presence for fast pre-filtering in ``_filter_candidates``.
        """
        now = time.monotonic()
        if self._file_cache and (now - self._file_cache_built_at) < _FILE_CACHE_TTL:
            return self._file_cache, self._file_cache_bitmaps

        paths = _list_git_files(self._cwd)
        if paths is None:
            paths = _walk_filesystem(self._cwd)
        paths.sort(key=str.lower)
        self._file_cache = paths
        self._file_cache_bitmaps = [_build_path_bitmap(p) for p in paths]
        self._file_cache_built_at = time.monotonic()
        return self._file_cache, self._file_cache_bitmaps

    def _refresh_at_file_suggestions(self, query: str) -> None:
        """Refresh the ``@`` file-suggestion popup based on the typed query.

        Two code paths:
        * If a custom ``files_provider`` is set, use it exclusively.
        * Path-like tokens (``@/...``, ``@~/...``, ``@./...``, ``@../...``)
          walk the filesystem directly via ``_path_completions``.
        * Plain queries (``@src/uti``) are matched against the cached
          project-file index via ``_filter_candidates``.
        """
        # Custom files_provider takes priority when set (F-38).
        if self._files_provider is not None:
            try:
                entries = self._files_provider()
            except Exception:
                entries = None
            if not entries:
                self._hide_at_file_suggestions()
                return
            matches = [p for p in entries if query.lower() in p.lower()]
            matches = matches[:_AT_FILE_MAX_SUGGESTIONS]
            if not matches:
                self._hide_at_file_suggestions()
                return
            self._at_file_suggestions.clear_options()
            for path in matches:
                self._at_file_suggestions.add_option(Option(path, id="@" + path))
            self._at_file_suggestions.highlighted = 0
            self._at_file_suggestions.remove_class("-hidden")
            self._suggest_mode = "at_file"
            return

        if _is_path_like_token(query):
            entries = _path_completions(query, _AT_FILE_MAX_SUGGESTIONS)
            if not entries:
                self._hide_at_file_suggestions()
                return
            self._at_file_suggestions.clear_options()
            for entry in entries:
                # Textual 0.79: id= kwarg 不再被 add_option 接受，
                # 改用 Option 包装。
                self._at_file_suggestions.add_option(
                    Option(entry.display, id="@" + entry.text),
                )
            self._at_file_suggestions.highlighted = 0
            self._at_file_suggestions.remove_class("-hidden")
            self._suggest_mode = "at_file"
            return

        candidates, bitmaps = self._get_cached_file_list()
        if not candidates:
            self._hide_at_file_suggestions()
            return

        matches = _filter_candidates(
            candidates,
            query,
            _AT_FILE_MAX_SUGGESTIONS,
            bitmaps=bitmaps,
        )
        if not matches:
            self._hide_at_file_suggestions()
            return

        self._at_file_suggestions.clear_options()
        for path in matches:
            # Textual 0.79: id= kwarg 不再被 add_option 接受，
            # 改用 Option 包装。
            self._at_file_suggestions.add_option(Option(path, id="@" + path))
        self._at_file_suggestions.highlighted = 0
        self._at_file_suggestions.remove_class("-hidden")
        self._suggest_mode = "at_file"

    def _hide_at_file_suggestions(self) -> None:
        if not self._at_file_suggestions.has_class("-hidden"):
            self._at_file_suggestions.add_class("-hidden")
            self._at_file_suggestions.clear_options()
        if self._suggest_mode == "at_file":
            self._suggest_mode = ""

    def _refresh_agent_suggestions(self, query: str) -> None:
        """Refresh the ``@agent-<type>`` suggestion popup.

        Matches the typed partial against known agent ``agent_type`` values.
        """
        if not self._agents_provider:
            self._hide_agent_suggestions()
            return

        try:
            agents = self._agents_provider()
        except Exception:
            self._hide_agent_suggestions()
            return

        if not agents:
            self._hide_agent_suggestions()
            return

        query_lower = query.lower()
        matches: list[tuple[str, str]] = []  # (agent_type, name)

        for agent in agents:
            agent_type = getattr(agent, "agent_type", None) or (
                agent.get("agent_type") if isinstance(agent, dict) else None
            )
            if not isinstance(agent_type, str) or not agent_type:
                continue
            if query_lower not in agent_type.lower():
                continue
            name = getattr(agent, "name", None) or (agent.get("name", "") if isinstance(agent, dict) else "")
            matches.append((agent_type, str(name) if name else ""))

        if not matches:
            self._hide_agent_suggestions()
            return

        self._agent_suggestions.clear_options()
        for agent_type, name in matches:
            display = f"{agent_type}  — {name}" if name else agent_type
            self._agent_suggestions.add_option(
                Option(display, id=f"@agent-{agent_type}"),
            )
        self._agent_suggestions.highlighted = 0
        self._agent_suggestions.remove_class("-hidden")
        self._suggest_mode = "at_agent"

    def _hide_agent_suggestions(self) -> None:
        if not self._agent_suggestions.has_class("-hidden"):
            self._agent_suggestions.add_class("-hidden")
            self._agent_suggestions.clear_options()
        if self._suggest_mode == "at_agent":
            self._suggest_mode = ""

    def _navigate_history(self, direction: int) -> None:
        """``direction`` = +1 means older (Up); -1 means newer (Down)."""
        if not self._history:
            return
        if direction > 0:
            if self._history_pos is None:
                self._history_pos = len(self._history) - 1
            else:
                self._history_pos = max(0, self._history_pos - 1)
        else:
            if self._history_pos is None:
                return
            self._history_pos += 1
            if self._history_pos >= len(self._history):
                self._history_pos = None
                self._input.value = ""
                return
        self._input.value = self._history[self._history_pos]
        self._input.cursor_position = len(self._input.value)


def _options_from_suggestions(
    suggestions: list[CommandSuggestion],
    partial: str,
) -> list[Option]:
    """Filter + rank rich command suggestions, then render two-column rows.

    Delegates the matching/scoring to the shared
    :func:`clawcodex_ext.utils.completers.rank_suggestions` (same
    algorithm as the REPL's ``_rich_completions``), then renders each
    ranked entry as a two-column Rich ``Text`` row.

    Aliases are surfaced only when the typed prefix matched the alias
    so an unmatched ``/com<TAB>`` does not get polluted with the full
    alias list.
    """

    ranked = rank_suggestions(
        (s for s in suggestions if isinstance(s, CommandSuggestion)),
        partial,
        max_results=_MAX_VISIBLE_SUGGESTIONS,
    )
    return [Option(_render_suggestion_row(sugg, matched_alias), id=sugg.slash) for sugg, matched_alias in ranked]


def _options_from_words(words: list[Any], partial: str) -> list[Option]:
    """Fallback renderer for the legacy ``words_provider`` path.

    Produces the same plain ``/name`` rows the older popup used, with
    the same fuzzy filter — so older callers / tests keep working
    while richer surfaces opt into the two-column layout.
    """

    matches: list[str] = []
    seen: set[str] = set()
    for word in words:
        if not isinstance(word, str) or not word.startswith("/"):
            continue
        key = word[1:].lower()
        if key in seen:
            continue
        if not partial or fuzzy_match(key, partial):
            seen.add(key)
            matches.append(word)
            if len(matches) >= _MAX_VISIBLE_SUGGESTIONS:
                break
    return [Option(word, id=word) for word in matches]


def _render_suggestion_row(
    sugg: CommandSuggestion,
    matched_alias: str | None,
) -> Text:
    """Render one slash-command row as a Rich ``Text`` with two columns.

    Column 1: ``/name`` (plus ``(alias)`` if the user typed an alias),
    padded to :data:`_NAME_COLUMN_WIDTH`. Column 2: optional
    ``[tag]`` then the command description. The whole row is dim by
    default; OptionList's highlighted-row CSS overlays bold + boost
    background on the selected entry (see ``_SlashSuggestions``).
    """

    alias_text = f" ({matched_alias})" if matched_alias else ""
    name_segment = f"{sugg.slash}{alias_text}"
    if len(name_segment) > _NAME_COLUMN_WIDTH - 1:
        name_segment = name_segment[: _NAME_COLUMN_WIDTH - 2] + "…"
    pad = max(1, _NAME_COLUMN_WIDTH - len(name_segment))

    row = Text(no_wrap=True, overflow="ellipsis", style="dim")
    row.append(name_segment, style="bold")
    row.append(" " * pad)
    if sugg.tag:
        row.append(f"[{sugg.tag}] ", style="italic cyan")
    if sugg.description:
        # Collapse internal whitespace so multi-line descriptions render
        # cleanly on a single row (the OptionList truncates with "…").
        row.append(" ".join(sugg.description.split()))
    return row


def _next_word(text: str, pos: int) -> int:
    """Return the cursor index of the next word start.

    ``pos`` is clamped to ``[0, len(text)]``. A word is any run of
    non-whitespace characters; we skip the current word first, then
    any whitespace.
    """

    n = len(text)
    pos = max(0, min(pos, n))
    # skip to end of current word
    while pos < n and not text[pos].isspace():
        pos += 1
    # skip intervening whitespace
    while pos < n and text[pos].isspace():
        pos += 1
    return pos


def _prev_word(text: str, pos: int) -> int:
    """Return the cursor index of the previous word start."""

    pos = max(0, min(pos, len(text)))
    # step back over whitespace
    while pos > 0 and text[pos - 1].isspace():
        pos -= 1
    # step back to start of word
    while pos > 0 and not text[pos - 1].isspace():
        pos -= 1
    return pos


# Back-compat: ``tests.tui.test_slash_token_parser`` imports
# ``_current_slash_token`` via the ``src.tui.widgets.prompt_input``
# lazy proxy. Re-export the shared function under the same name so
# the proxy can find it and the spec-locking test continues to pass.
_current_slash_token = current_slash_token
