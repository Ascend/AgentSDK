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

"""Voice — ``/voice`` command.

Mirrors TS ``src/commands/voice/voice.ts``: toggles voice mode on/off and
selects the STT backend. The persisted state lives in
``settings.voice_enabled`` (master switch) and ``settings.voice_provider``
(``"anthropic"`` | ``"doubao"``); both are written via
:func:`src.config.set_voice_enabled` / :func:`src.config.set_voice_provider`,
which invalidate the settings cache so the next ``get_settings`` reflects
the change mid-session.

Usage
-----
* ``/voice`` — toggle voice mode on/off (current provider kept).
* ``/voice anthropic`` — enable voice mode + select Anthropic STT.
* ``/voice doubao`` — enable voice mode + select Doubao ASR.
* ``/voice off`` — disable voice mode (provider kept).
* ``/voice status`` — show current state + availability diagnostics.
* ``/voice help`` — usage text.

Design decisions (mirrors TS)
------------------------------
* ``/voice <provider>`` flips *both* the master switch and the provider
  atomically, so the user doesn't have to run two commands to start using
  a new backend. ``/voice off`` only flips the switch (provider retained
  for next enable).
* Provider validation happens here (not in settings) so the command can
  surface a clear "unknown backend" message with the valid list, rather
  than the settings layer silently coercing.
* Availability diagnostics in ``status`` surface the three gate
  layers (feature flag / kill-switch / per-provider auth) so the user can
  see *why* voice is unavailable without reading source.
* Follows the project's :class:`LocalCommand` convention (matches
  ``/cost``, ``/context``, …): a plain ``LocalCommand`` instance bound to
  a free ``voice_command_call`` function via :meth:`set_call`, rather than
  a subclass that overrides ``call``. The :func:`set_call` indirection is
  required by ``test_all_builtins_have_call_impl`` which asserts every
  registered ``LocalCommand`` has ``_call_impl`` set.
"""

from __future__ import annotations

from contextlib import suppress

from .types import CommandContext, LocalCommand, LocalCommandResult
from clawcodex_ext.services.voice.voice_mode_enabled import (  # pylint: disable=no-name-in-module
    VOICE_PROVIDERS,
    get_voice_provider,
    has_voice_auth,
    is_voice_available,
    is_voice_disabled_by_kill_switch,
    is_voice_enabled,
    is_voice_feature_enabled,
)
from src.config import set_voice_enabled, set_voice_provider

__all__ = ["VOICE_COMMAND", "voice_command_call"]

_HELP = (
    "Usage: /voice [anthropic|doubao|off|status|help]\n\n"
    "Voice mode enables push-to-talk speech input. Hold the spacebar to "
    "record; release to transcribe and submit.\n\n"
    "Commands:\n"
    "- /voice              Toggle voice mode on/off\n"
    "- /voice anthropic    Enable with Anthropic STT (requires OAuth login)\n"
    "- /voice doubao       Enable with Doubao ASR (requires credentials file)\n"
    "- /voice off          Disable voice mode\n"
    "- /voice status       Show current state and availability\n"
    "- /voice help         Show this help\n\n"
    "Tip: for full-duplex voice dialogue (simultaneous speech in + out "
    "with barge-in), see /dialogue."
)


def _status_text() -> str:
    """Render the current voice state + per-layer availability diagnostics.

    Surfaces all three gate layers so the user can see *why* voice
    is unavailable (feature flag off / kill-switch set / OAuth missing)
    without having to read source or check env vars by hand.

    Also surfaces the dialogue status line so a user
    encountering the half-duplex code path discovers the full-duplex
    sibling without reading the docs.
    """
    enabled = is_voice_enabled()
    provider = get_voice_provider()
    lines = [f"Voice mode: {'on' if enabled else 'off'}", f"Provider: {provider}"]
    lines.append("")
    lines.append("Availability:")
    lines.append(f"  feature flag (FEATURE_VOICE_MODE): {'on' if is_voice_feature_enabled() else 'off'}")
    lines.append(
        f"  kill-switch (CLAWCODEX_VOICE_DISABLED): {'set' if is_voice_disabled_by_kill_switch() else 'clear'}"
    )
    lines.append(f"  overall available: {'yes' if is_voice_available() else 'no'}")
    lines.append(f"  Anthropic OAuth token: {'present' if has_voice_auth() else 'missing'}")
    lines.append("")
    if not is_voice_feature_enabled():
        lines.append("Tip: set FEATURE_VOICE_MODE=1 to enable voice.")
    if is_voice_disabled_by_kill_switch():
        lines.append("Tip: unset CLAWCODEX_VOICE_DISABLED to release the kill-switch.")
    if provider == "anthropic" and not has_voice_auth():
        lines.append("Tip: run /login to obtain an Anthropic OAuth token for STT.")
    if provider == "doubao":
        lines.append("Tip: configure ~/.clawcodex/tts/doubao/credentials.json for Doubao ASR.")
    # cross-reference. The dialogue status block is a single
    # line — enough to make the user aware that the full-duplex path
    # exists without duplicating the status command's output.
    with suppress(Exception):
        from clawcodex_ext.services.voice.voice_mode_enabled import (  # pylint: disable=no-name-in-module
            is_dialogue_enabled,
            is_dialogue_feature_enabled,
            has_dialogue_auth,
        )

        dia_state = (
            "on"
            if (is_dialogue_enabled() and has_dialogue_auth())
            else "off"
            if is_dialogue_enabled()
            else "available — enable with /dialogue"
            if (is_dialogue_feature_enabled() and has_dialogue_auth())
            else "configured (provider unconfigured)"
        )
        lines.append("")
        lines.append(f"Full-duplex dialogue (/dialogue): {dia_state}")
    return "\n".join(lines)


def voice_command_call(args: str, context: CommandContext) -> LocalCommandResult:
    """``/voice`` handler — toggle voice mode and/or select STT backend.

    Every path returns a :class:`LocalCommandResult` text message; no UI
    picker is needed, so ``/voice`` is a :class:`LocalCommand` (not an
    :class:`InteractiveCommand`) and works on headless / SDK surfaces too.
    """
    raw = (args or "").strip()
    a = raw.lower()

    # 1. help (headless).
    if a in ("help", "-h", "--help"):
        return LocalCommandResult(type="text", value=_HELP)

    # 2. status (headless) — diagnostics for the three gate layers.
    if a in ("status", "current", "show"):
        return LocalCommandResult(type="text", value=_status_text())

    # 3. off — disable voice mode, keep provider.
    if a == "off":
        set_voice_enabled(False)
        return LocalCommandResult(type="text", value="Voice mode disabled.")

    # 4. explicit provider — enable + select backend atomically.
    if a in VOICE_PROVIDERS:
        set_voice_provider(a)
        set_voice_enabled(True)
        auth_hint = ""
        if a == "anthropic" and not has_voice_auth():
            auth_hint = " (warning: no OAuth token — run /login before recording)"
        elif a == "doubao":
            auth_hint = " (configure ~/.clawcodex/tts/doubao/credentials.json before recording)"
        return LocalCommandResult(
            type="text",
            value=f"Voice mode enabled with {a} backend{auth_hint}.",
        )

    # 5. no args — toggle on/off (keep current provider).
    if not a:
        currently = is_voice_enabled()
        set_voice_enabled(not currently)
        provider = get_voice_provider()
        if not currently:
            return LocalCommandResult(
                type="text",
                value=f"Voice mode enabled (provider: {provider}). Hold spacebar to record.",
            )
        return LocalCommandResult(type="text", value="Voice mode disabled.")

    # 6. unknown arg.
    return LocalCommandResult(
        type="text",
        value=(
            f"Unknown argument: {raw}. Valid options: " + ", ".join(list(VOICE_PROVIDERS) + ["off", "status", "help"])
        ),
    )


VOICE_COMMAND = LocalCommand(
    name="voice",
    description="Toggle voice input mode and select STT backend",
    argument_hint="[anthropic|doubao|off|status|help]",
    supports_non_interactive=True,
)
VOICE_COMMAND.set_call(voice_command_call)
