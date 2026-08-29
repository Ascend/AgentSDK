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

"""tts — ``/tts`` command (P64-E5 Text-to-Speech).

Symmetric to ``/voice`` (the STT side): toggles TTS playback on/off,
selects the TTS backend, sets the voice id, and offers a ``say`` preview
sub-command. Persisted state lives in ``settings.tts_enabled`` /
``settings.tts_provider`` / ``settings.tts_voice``; written via
:func:`src.config.set_tts_enabled` / :func:`set_tts_provider` /
:func:`set_tts_voice` which invalidate the settings cache so the next
``get_settings()`` reflects the change mid-session.

Usage
-----
* ``/tts`` — toggle TTS playback on/off (current provider kept).
* ``/tts openai`` — enable TTS + select OpenAI backend.
* ``/tts minimax`` — enable TTS + select MiniMax T2A backend.
* ``/tts gemini`` — enable TTS + select Gemini TTS backend.
* ``/tts off`` — disable TTS playback (provider kept).
* ``/tts voice <name>`` — set the provider-specific voice id.
* ``/tts say <text>`` — preview: synthesize and play a sample phrase.
* ``/tts status`` — show current state + provider list.
* ``/tts help`` — usage text.

Design decisions (mirrors /voice)
----------------------------------
* ``/tts <provider>`` flips both the master switch and the provider
 atomically. ``/tts off`` only flips the switch (provider retained
 for next enable).
* Provider validation happens here (not in settings) so the command
 surfaces a clear "unknown backend" message with the valid list.
* ``say`` is a fire-and-forget preview: it constructs the provider via the
 registry, calls ``synthesize()`` (batch path), and pipes the PCM to
 the default :class:`AudioPlayer`. Errors surface as text rather than
 audio (so the user sees what went wrong).
* Follows the project's :class:`LocalCommand` convention (matches
 ``/voice``, ``/cost``, …): a plain ``LocalCommand`` bound to a free
 ``tts_command_call`` function via :meth:`set_call`.
"""

from __future__ import annotations

import asyncio
import logging

from .types import CommandContext, LocalCommand, LocalCommandResult
from clawcodex_ext.services.voice.tts import TTSConfig  # pylint: disable=no-name-in-module
from clawcodex_ext.services.voice.provider_registry import (  # pylint: disable=no-name-in-module
    list_tts_providers,
    get_tts_provider,
)
from src.config import set_tts_enabled, set_tts_provider, set_tts_voice

logger = logging.getLogger(__name__)

__all__ = ["TTS_COMMAND", "tts_command_call", "TTS_PROVIDERS"]

# Built-in TTS provider whitelist (mirrors the registry's builtins). Used
# for argument validation; the registry is the source of truth for what's
# actually instantiable (a provider may be registered but fail at factory
# time if its dep is missing — that surfaces as a clear error, not here).
TTS_PROVIDERS: tuple[str, ...] = ("openai", "minimax", "gemini")

_HELP = (
    "Usage: /tts [openai|minimax|gemini|off|voice <name>|say <text>|status|help]\n\n"
    "TTS mode enables speech output for agent replies. When enabled, agent\n"
    "text is synthesized to PCM and played through the audio device.\n\n"
    "Commands:\n"
    "- /tts                    Toggle TTS playback on/off\n"
    "- /tts openai             Enable with OpenAI TTS (requires OPENAI_API_KEY)\n"
    "- /tts minimax            Enable with MiniMax T2A (requires MINIMAX_API_KEY)\n"
    "- /tts gemini             Enable with Gemini TTS (requires google-genai + GEMINI_API_KEY)\n"
    "- /tts off                Disable TTS playback\n"
    "- /tts voice <name>       Set the provider-specific voice id\n"
    "- /tts say <text>           Preview: synthesize and play a sample phrase\n"
    "- /tts status             Show current state and provider list\n"
    "- /tts help               Show this help"
)


def _get_tts_provider_name() -> str:
    """Read the persisted TTS provider, defaulting to ``openai`` if unset."""
    try:
        from src.settings.settings import get_settings

        raw = (get_settings().tts_provider or "").strip().lower()
    except Exception:
        raw = ""
    if raw in TTS_PROVIDERS:
        return raw
    return "openai"


def _is_tts_enabled() -> bool:
    try:
        from src.settings.settings import get_settings

        return bool(get_settings().tts_enabled)
    except Exception:
        return False


def _get_tts_voice() -> str:
    try:
        from src.settings.settings import get_settings

        return (get_settings().tts_voice or "").strip()
    except Exception:
        return ""


def _status_text() -> str:
    enabled = _is_tts_enabled()
    provider = _get_tts_provider_name()
    voice = _get_tts_voice() or "(provider default)"
    lines = [
        f"TTS mode: {'on' if enabled else 'off'}",
        f"Provider: {provider}",
        f"Voice: {voice}",
        "",
        "Available providers: " + ", ".join(list_tts_providers()),
    ]
    return "\n".join(lines)


def _say(args: str) -> LocalCommandResult:
    """Preview path: synthesize ``args`` via the current provider + play it.

    Runs the async synthesis + playback on a fresh event loop (the command
    runs synchronously from the UI thread). Errors are caught and surfaced
    as text — the user sees what went wrong rather than silence.
    """
    text = (args or "").strip()
    if not text:
        return LocalCommandResult(type="text", value="Usage: /tts say <text> — provide a phrase to synthesize.")
    provider_name = _get_tts_provider_name()
    try:
        provider = get_tts_provider(provider_name)
    except KeyError:
        return LocalCommandResult(
            type="text",
            value=f"Unknown TTS provider {provider_name!r}. Valid: {list_tts_providers()}",
        )
    except ImportError as exc:
        return LocalCommandResult(
            type="text",
            value=f"TTS backend {provider_name!r} unavailable: {exc}",
        )

    voice = _get_tts_voice()
    cfg = TTSConfig(voice=voice) if voice else TTSConfig()

    async def _run() -> tuple[bool, str]:
        try:
            pcm = await provider.synthesize(text, cfg)
        except Exception as exc:
            return False, f"Synthesis failed: {exc}"
        if not pcm:
            return False, "Synthesis returned empty audio."
        # Play via the audio player (P64-E8). Lazy import so the player
        # backend (PyAudio / SoX / ffplay) is only probed when the user
        # actually runs a preview, not at module import.
        try:
            from clawcodex_ext.services.voice.audio_player import play_pcm  # pylint: disable=no-name-in-module

            play_pcm(pcm, sample_rate=cfg.sample_rate)
        except Exception as exc:
            return False, f"Playback failed: {exc}"
        return True, ""

    try:
        loop = asyncio.new_event_loop()
        try:
            ok, err = loop.run_until_complete(_run())
        finally:
            loop.close()
    except Exception as exc:
        return LocalCommandResult(type="text", value=f"Preview failed: {exc}")
    if not ok:
        return LocalCommandResult(type="text", value=err)
    return LocalCommandResult(type="text", value=f"Preview played ({len(text)} chars via {provider_name}).")


def tts_command_call(args: str, context: CommandContext) -> LocalCommandResult:
    """``/tts`` handler — toggle TTS, select backend, set voice, or preview."""
    raw = (args or "").strip()
    a = raw.lower()
    # Split into subcommand + payload once, tolerating multiple spaces
    # between them (e.g. "/tts voice   alloy" or "/tts say   hello").
    parts = raw.split(maxsplit=1)
    head = parts[0].lower() if parts else ""
    payload = parts[1] if len(parts) > 1 else ""

    # 1. help.
    if a in ("help", "-h", "--help"):
        return LocalCommandResult(type="text", value=_HELP)

    # 2. status.
    if a in ("status", "current", "show"):
        return LocalCommandResult(type="text", value=_status_text())

    # 3. off.
    if a == "off":
        set_tts_enabled(False)
        return LocalCommandResult(type="text", value="TTS playback disabled.")

    # 4. voice <name> — set the voice id (case-preserved).
    if head == "voice":
        voice = payload.strip()
        if not voice:
            return LocalCommandResult(type="text", value="Usage: /tts voice <name> — provide a voice id.")
        set_tts_voice(voice)
        return LocalCommandResult(type="text", value=f"TTS voice set to {voice!r} (applies on next synthesis).")

    # 5. say <text> — preview.
    if head == "say":
        return _say(payload)

    # 6. explicit provider — enable + select backend atomically.
    if a in TTS_PROVIDERS:
        set_tts_provider(a)
        set_tts_enabled(True)
        return LocalCommandResult(
            type="text",
            value=f'TTS playback enabled with {a} backend. Run /tts say "hello" to preview.',
        )

    # 7. no args — toggle on/off (keep current provider).
    if not a:
        currently = _is_tts_enabled()
        set_tts_enabled(not currently)
        provider = _get_tts_provider_name()
        if not currently:
            return LocalCommandResult(
                type="text",
                value=f"TTS playback enabled (provider: {provider}). Agent replies will be spoken.",
            )
        return LocalCommandResult(type="text", value="TTS playback disabled.")

    # 8. unknown arg.
    return LocalCommandResult(
        type="text",
        value=(
            f"Unknown argument: {raw}. Valid options: "
            + ", ".join(list(TTS_PROVIDERS) + ["off", "voice <name>", "say <text>", "status", "help"])
        ),
    )


TTS_COMMAND = LocalCommand(
    name="tts",
    description="Toggle text-to-speech output and select TTS backend",
    argument_hint="[openai|minimax|gemini|off|voice <name>|say <text>|status|help]",
    supports_non_interactive=True,
)
TTS_COMMAND.set_call(tts_command_call)
