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

"""Plan 3: REPL-side context-aware Tab registration.

The Python port of claude-code-best's ``useTypeahead`` registers a
``tab`` binding with a filter alongside the configured accept key.
This test pins the registration shape so the contract is explicit:

* The configured accept key is bound (default ``c-e``).
* A ``tab`` binding is also registered, with a non-trivial filter
  (``Condition`` instance wrapping a callable that inspects the
  current buffer's ``suggestion`` and ``complete_state``).
* The hint string is consistent across the configured key and Tab.
"""

from __future__ import annotations

# ``src.repl`` is the legacy proxy: importing it first bootstraps the
# lazy ``__getattr__`` chain that pulls ``clawcodex_ext.repl.core`` into
# ``sys.modules`` before we touch it. Without this bootstrap the
# partially-initialised module makes the import below raise
# ``ImportError: cannot import name 'ClawcodexREPL' from 'src.repl.core'``.
import src.repl  # noqa: F401

from prompt_toolkit.filters import Condition
from prompt_toolkit.key_binding import KeyBindings

from clawcodex_ext.repl.core import (
    _HintedAutoSuggest,
    _ghost_hint_for,
    _patch_accept_suggestion_bindings,
)


def _bindings_keys(bindings) -> list[str]:
    """Return the list of registered key string values (in registration order).

    ``Binding.keys`` is a tuple of ``prompt_toolkit.keys.Keys`` enum
    instances, NOT plain strings. ``Tab`` is stored as ``Keys.ControlI``
    (value ``'c-i'``) because Tab and Ctrl+I share ASCII 9 — we surface
    the underlying string value so callers can compare against literals
    like ``"c-e"`` / ``"c-i"``.
    """
    return [k.value for b in bindings.bindings for k in b.keys]


def _find_binding(bindings, *aliases: str):
    """Return the first binding whose key matches any of *aliases*."""
    wanted = set(aliases)
    for b in bindings.bindings:
        for k in b.keys:
            if k.value in wanted:
                return b
    raise AssertionError(f"no binding found for keys {aliases!r}")


def test_default_accept_key_registers_c_e():
    bindings = KeyBindings()
    _patch_accept_suggestion_bindings(bindings)
    keys = _bindings_keys(bindings)
    assert "c-e" in keys
    # Tab is also bound (context-aware secondary key). prompt_toolkit
    # stores the Tab key under the ControlI alias — ASCII 9.
    assert "c-i" in keys


def test_hint_mentions_tab_when_primary_is_not_tab():
    """The ghost-text hint should advertise ``TAB`` as an alias."""
    hint = _ghost_hint_for("c-e")
    assert "CTRL + E" in hint
    assert "TAB" in hint


def test_hint_omits_tab_alias_when_primary_is_tab():
    """If the user already set the primary key to tab, the hint is
    just ``(TAB to accept)`` — no double mention.
    """
    hint = _ghost_hint_for("tab")
    assert hint == " (TAB to accept)"


def test_tab_binding_has_a_filter():
    """The tab binding must be filtered to ghost-only contexts."""
    bindings = KeyBindings()
    _patch_accept_suggestion_bindings(bindings)
    # Tab is stored under the ControlI alias in prompt_toolkit.
    tab_binding = _find_binding(bindings, "tab", "c-i")
    # The filter is a ``Condition`` instance wrapping the closure.
    assert isinstance(tab_binding.filter, Condition)


def test_can_disable_tab_alias():
    """``has_tab_alias=False`` skips the tab binding entirely."""
    bindings = KeyBindings()
    _patch_accept_suggestion_bindings(bindings, has_tab_alias=False)
    keys = _bindings_keys(bindings)
    assert "c-e" in keys
    assert "c-i" not in keys
    # And the hint no longer mentions TAB.
    hint = _ghost_hint_for("c-e", has_tab_alias=False)
    assert hint == " (CTRL + E to accept)"


def test_hinted_auto_suggest_hint_includes_tab_alias():
    """The class-level hint must match the patched binding hint."""
    sas = _HintedAutoSuggest(accept_key="c-e")
    assert "CTRL + E" in sas._hint
    assert "TAB" in sas._hint
    # And the class hint equals the helper output for the same key.
    assert sas._hint == _ghost_hint_for("c-e")


def test_hinted_auto_suggest_can_disable_tab_alias():
    sas = _HintedAutoSuggest(accept_key="c-e", has_tab_alias=False)
    assert sas._hint == _ghost_hint_for("c-e", has_tab_alias=False)
    assert "TAB" not in sas._hint


def test_tab_filter_returns_false_when_no_suggestion():
    """The filter must return False when no ghost is showing.

    prompt_toolkit ``Condition`` filters are no-arg callables in this
    version, so the filter reads visibility from the module-level
    ``_ghost_state`` snapshot that ``_HintedAutoSuggest.get_suggestion``
    keeps fresh. We exercise that snapshot directly.
    """
    from clawcodex_ext.repl import core as _repl_core

    bindings = KeyBindings()
    _patch_accept_suggestion_bindings(bindings)
    tab_binding = _find_binding(bindings, "tab", "c-i")
    cond = tab_binding.filter

    # No suggestion at all -> filter False.
    _repl_core._ghost_state["suggestion"] = None
    _repl_core._ghost_state["complete_active"] = False
    assert cond() is False

    # Suggestion + completion menu open -> filter False (popup wins).
    _repl_core._ghost_state["suggestion"] = "hello"
    _repl_core._ghost_state["complete_active"] = True
    assert cond() is False

    # Suggestion only (the ghost case) -> filter True.
    _repl_core._ghost_state["suggestion"] = "hello "
    _repl_core._ghost_state["complete_active"] = False
    assert cond() is True
