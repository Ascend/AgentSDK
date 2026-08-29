#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# -------------------------------------------------------------------------
# This file is part of the AgentSDK project.
#
# Originally from Clawd Codex:
# https://github.com/agentforce314/clawcodex
# Copyright (c) 2026 Clawd Codex Team
# Licensed under the MIT License. See clawcodex-ascend/LICENSES/Clawd-Codex-MIT.txt.
#
# Portions copyright (c) 2026 Huawei Technologies Co.,Ltd.
# Licensed under Mulan PSL v2. You may obtain a copy of Mulan PSL v2 at:
#
#          http://license.coscl.org.cn/MulanPSL2
#
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
# -------------------------------------------------------------------------

"""Voice-mode gating logic.

Mirrors TS ``src/voice/voiceModeEnabled.ts``: a three-layer gate that
decides whether voice input is available and which STT backend to use.

Layers
------
1. **Feature flag** — ``FEATURE_VOICE_MODE`` env var (``"1"`` / ``"true"``).
 Off by default; the project ships the stack but the user must opt in.
 adds an independent ``FEATURE_DIALOGUE_MODE`` so the two
 paths (half-duplex PTT and full-duplex dialogue) can be released
 separately.
2. **Kill-switch** — ``CLAWCODEX_VOICE_DISABLED`` env var. Default unset =
 not disabled. A negative gate like TS ``tengu_amber_quartz_disabled``:
 the *absence* of the env var means "available".
3. **Auth (Anthropic backend only)** — Anthropic STT requires an OAuth
 token (claude.ai subscription), not an API key. The doubao backend
 uses an independent credential file and skips this check.
 MiniMax Realtime (the main path) uses an API key + group_id, so
 it's treated like doubao: presence of either env var or the
 credentials file is enough.

Read-side
---------
* :func:`is_voice_mode_enabled` — Anthropic path: flag + kill-switch + OAuth.
* :func:`is_voice_available` — Provider-agnostic: flag + kill-switch only.
  Used by ``/voice`` to decide whether to show the command at all, and by
  the doubao path which doesn't need OAuth.
* :func:`get_voice_provider` — The persisted backend, defaulting to
  ``"anthropic"`` when unset (so a fresh install has a defined default).
* :func:`is_voice_enabled` — The master on/off switch written by ``/voice``.

Dialogue-mode additions are gated independently, so existing
installations keep working unchanged:

* :func:`is_dialogue_feature_enabled` — independent flag.
* :func:`has_dialogue_auth` — MiniMax credentials present.
* :func:`is_dialogue_available` — flag ∧ ¬kill-switch.
* :func:`is_dialogue_enabled` — master switch (``settings.dialogue_enabled``).
* :func:`get_dialogue_provider` — backend choice, default ``"minimax"``.

The settings reads go through ``get_settings`` (cached) and are
invalidated by ``set_voice_provider`` / ``set_voice_enabled`` so the gate
reflects mid-session changes immediately.
"""

from __future__ import annotations

# Sibling clawcodex_ext packages (auth) are migrated in separate branches;
# suppress E0611 until the full series lands.
# pylint: disable=no-name-in-module

import os
from pathlib import Path
from typing import Literal

__all__ = [
    "VoiceProvider",
    "VOICE_PROVIDERS",
    "is_voice_feature_enabled",
    "is_voice_disabled_by_kill_switch",
    "has_voice_auth",
    "is_voice_mode_enabled",
    "is_voice_available",
    "is_voice_enabled",
    "get_voice_provider",
    "DialogueProvider",
    "DIALOGUE_PROVIDERS",
    "is_dialogue_feature_enabled",
    "has_dialogue_auth",
    "is_dialogue_available",
    "is_dialogue_enabled",
    "get_dialogue_provider",
]

VoiceProvider = Literal["anthropic", "doubao"]
VOICE_PROVIDERS: tuple[str, ...] = ("anthropic", "doubao")

# dialogue providers. Mirrored on VOICE_PROVIDERS so the
# registry/CLI surface is symmetric. Today only ``"minimax"`` is wired
# (MiniMaxRealtimeDialogueProvider); ``"openai-realtime"`` is reserved
# for the P65-E reference adapter.
DialogueProvider = Literal["minimax", "openai-realtime"]
DIALOGUE_PROVIDERS: tuple[str, ...] = ("minimax", "openai-realtime")

# Credentials file location for the MiniMax Realtime dialogue backend
# (same convention as ``minimax_stt``: ``~/.clawcodex/tts/minimax/
# credentials.json``). Used by :func:`has_dialogue_auth` to short-circuit
# the ``MINIMAX_API_KEY`` env probe and surface a hint when neither is
# configured.
MINIMAX_REALTIME_CREDENTIALS_PATH = Path("~/.clawcodex/tts/minimax/credentials.json")


def is_voice_feature_enabled() -> bool:
    """Layer 1 — compile/runtime feature flag ``FEATURE_VOICE_MODE``.

    Accepts ``"1"`` / ``"true"`` / ``"yes"`` (case-insensitive). Empty /
    unset / any other value = disabled. This is the *only* layer that
    defaults off; layers 2 and 3 default to "available".
    """
    raw = os.environ.get("FEATURE_VOICE_MODE", "").strip().lower()
    return raw in ("1", "true", "yes", "on")


def is_voice_disabled_by_kill_switch() -> bool:
    """Layer 2 — negative kill-switch ``CLAWCODEX_VOICE_DISABLED``.

    Mirrors TS ``tengu_amber_quartz_disabled`` (default false = not
    disabled). Any truthy value here disables voice; absence = available.
    """
    raw = os.environ.get("CLAWCODEX_VOICE_DISABLED", "").strip().lower()
    return raw in ("1", "true", "yes", "on")


def has_voice_auth() -> bool:
    """Layer 3 — Anthropic OAuth check (claude.ai subscription token).

    Distinct from an API key: the Anthropic STT WebSocket endpoint
    (``voice_stream``) is gated on an OAuth bearer, not a sk-ant- key.
    Doubao bypasses this entirely (independent credentials file).

    Lazy import avoids pulling the auth stack at module-import time, and
    degrades to ``False`` if the auth module is unavailable (e.g. running
    under a stripped-down test harness) — voice simply reports unavailable
    rather than crashing the gate.
    """
    try:
        from clawcodex_ext.auth.oauth import OAuthTokens  # noqa: F401
        from clawcodex_ext.auth.auth import load_api_key  # noqa: F401
    except Exception:
        return False
    # The real OAuth-token presence check lives in the auth subsystem; we
    # delegate to a lightweight probe so the gate stays decoupled from the
    # auth internals. Implementation detail: read the persisted OAuth store
    # and confirm a non-expired token exists. Falls back to False on any
    # failure so a misconfigured auth never crashes the REPL.
    try:
        from clawcodex_ext.auth.codex_store import has_valid_oauth_token

        return has_valid_oauth_token()
    except Exception:
        # Function may not exist on all builds — treat as "no auth" rather
        # than blocking the gate. The STT connection itself will surface a
        # proper error if the user actually tries to record.
        return False


def is_voice_mode_enabled() -> bool:
    """Anthropic path gate: flag ∧ ¬kill-switch ∧ OAuth.

    Use this for the Anthropic STT backend specifically. For doubao or
    provider-agnostic availability checks use :func:`is_voice_available`.
    """
    if not is_voice_feature_enabled():
        return False
    if is_voice_disabled_by_kill_switch():
        return False
    return has_voice_auth()


def is_voice_available() -> bool:
    """Provider-agnostic gate: flag ∧ ¬kill-switch.

    OAuth is intentionally NOT checked here — the doubao backend uses an
    independent credential file. The REPL uses this to decide whether to
    show the ``/voice`` command and whether the push-to-talk hotkey should
    even be armed; the per-provider connection then enforces its own auth.
    """
    if not is_voice_feature_enabled():
        return False
    return not is_voice_disabled_by_kill_switch()


def is_voice_enabled() -> bool:
    """Master on/off switch (``settings.voice_enabled``).

    Written by ``/voice`` (no-arg toggle). When false the push-to-talk
    hotkey is inert regardless of provider configuration. Decoupled from
    :func:`get_voice_provider` — provider records the *chosen backend*,
    this records whether the user has opted in at all.
    """
    try:
        from src.settings.settings import get_settings

        return bool(get_settings().voice_enabled)
    except Exception:
        # Settings unavailable (e.g. bootstrap-before-config) → treat as
        # off. ``/voice`` will re-flip it once settings are reachable.
        return False


def get_voice_provider() -> VoiceProvider:
    """The persisted STT backend, defaulting to ``"anthropic"`` when unset.

    Reads ``settings.voice_provider``; empty string / unrecognised value /
    settings-unavailable all fall back to ``"anthropic"`` so a fresh
    install has a defined default without needing an explicit ``/voice
    anthropic`` invocation.
    """
    try:
        from src.settings.settings import get_settings

        raw = (get_settings().voice_provider or "").strip().lower()
    except Exception:
        raw = ""
    if raw in VOICE_PROVIDERS:
        return raw  # type: ignore[return-value]
    return "anthropic"


# ──: full-duplex dialogue gating ────────────────────────────────


def is_dialogue_feature_enabled() -> bool:
    """Layer 1 — dialog feature flag ``FEATURE_DIALOGUE_MODE``.

    Independent from ``FEATURE_VOICE_MODE`` so the STT path can ship
    without dragging in full-duplex dependencies (and vice versa).
    Same truthy grammar: ``"1"`` / ``"true"`` / ``"yes"`` / ``"on"``.
    """
    raw = os.environ.get("FEATURE_DIALOGUE_MODE", "").strip().lower()
    return raw in ("1", "true", "yes", "on")


def has_dialogue_auth() -> bool:
    """Layer 3 — MiniMax Realtime credentials probe.

    The MiniMax backend (current P65-A only adapter) uses API key + group_id,
    not OAuth. We probe ``MINIMAX_API_KEY`` first, then fall back to the
    credentials file's ``api_key`` field — the same file that
    :class:`MiniMaxSTTProvider` reads, so the user configures one file
    for both voice paths.
    """
    if os.environ.get("MINIMAX_API_KEY"):
        return True
    try:
        import json

        path = MINIMAX_REALTIME_CREDENTIALS_PATH.expanduser()
        if not path.is_file():
            return False
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return False
    return isinstance(data, dict) and bool(data.get("api_key"))


def is_dialogue_available() -> bool:
    """Provider-agnostic gate for full-duplex dialogue: flag ∧ ¬kill-switch.

    Same shape as :func:`is_voice_available` so the half- and full-duplex
    surfaces roll out symmetrically. The dialogue backend enforces its
    own auth at connection time, mirroring ``provider_registry``.
    """
    if not is_dialogue_feature_enabled():
        return False
    return not is_voice_disabled_by_kill_switch()


def is_dialogue_enabled() -> bool:
    """Master on/off switch (``settings.dialogue_enabled``).

    Written by ``/dialogue`` (no-arg toggle). When false the dialogue
    session refuses to start regardless of provider configuration.
    Decoupled from :func:`get_dialogue_provider` for the same reason
    the voice_provider / voice_enabled pair is split: the user can
    set up a backend in advance and then just toggle the switch on.
    """
    try:
        from src.settings.settings import get_settings

        return bool(getattr(get_settings(), "dialogue_enabled", False))
    except Exception:
        return False


def get_dialogue_provider() -> DialogueProvider:
    """The persisted dialogue backend, defaulting to ``"minimax"``.

    Reads ``settings.dialogue_provider``; empty / unrecognised values
    fall back to ``"minimax"`` so a fresh install has a defined default
    (the only currently-implemented adapter).
    """
    try:
        from src.settings.settings import get_settings

        raw = (getattr(get_settings(), "dialogue_provider", "") or "").strip().lower()
    except Exception:
        raw = ""
    if raw in DIALOGUE_PROVIDERS:
        return raw  # type: ignore[return-value]
    return "minimax"
