#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# -------------------------------------------------------------------------
# This file is part of the AgentSDK project.
# Copyright (c) 2026 Huawei Technologies Co.,Ltd.
# Copyright (c) 2026 Clawd Codex Team
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

"""STT + TTS + Dialogue provider registries — F-64 / F-65.

Three parallel registries that map provider names to *factory* callables
producing :class:`STTProvider` / :class:`TTSProvider` /
:class:`FullDuplexDialogueProvider` instances. The factory pattern lets us
defer heavy imports (``websockets`` for Anthropic/MiniMax, ``doubaoime-asr``
for doubao, ``google-genai`` for Gemini — all optional) until the user
actually starts a recording or synthesis, rather than at REPL boot.

This lives in the patch layer (``clawcodex_ext``) so third-party
extensions in ``extensions/`` could register additional STT/TTS/dialogue
backends via the same entry points without touching upstream.
"""

from __future__ import annotations

from typing import Protocol

from .stt import STTProvider
from .tts import TTSProvider

__all__ = [
    "STTProviderFactory",
    "TTSProviderFactory",
    "DialogueProviderFactory",
    "register_stt_provider",
    "register_tts_provider",
    "register_dialogue_provider",
    "get_stt_provider",
    "get_tts_provider",
    "get_dialogue_provider",
    "list_stt_providers",
    "list_tts_providers",
    "list_dialogue_providers",
    "STT_REGISTRY",
    "TTS_REGISTRY",
    "DIALOGUE_REGISTRY",
]


class STTProviderFactory(Protocol):
    def __call__(self) -> STTProvider: ...


class TTSProviderFactory(Protocol):
    def __call__(self) -> TTSProvider: ...


class DialogueProviderFactory(Protocol):
    """Factory returning a fresh full-duplex dialogue provider.

    Mirrors the STT/TTS factory signatures: a no-arg callable that
    constructs a brand-new instance. ``FullDuplexDialogueProvider`` is
    stateful (one WebSocket per session) so the registry helper
    :func:`get_dialogue_provider` constructs a new instance per call —
    callers own its lifecycle via ``start`` / ``stop`` / ``close``.
    """

    def __call__(self) -> "FullDuplexDialogueProvider": ...  # type: ignore[name-defined]  # noqa: F821


STT_REGISTRY: dict[str, STTProviderFactory] = {}
TTS_REGISTRY: dict[str, TTSProviderFactory] = {}
DIALOGUE_REGISTRY: dict[str, DialogueProviderFactory] = {}


def register_stt_provider(name: str, factory: STTProviderFactory) -> None:
    STT_REGISTRY[name.lower()] = factory


def register_tts_provider(name: str, factory: TTSProviderFactory) -> None:
    TTS_REGISTRY[name.lower()] = factory


def register_dialogue_provider(name: str, factory: DialogueProviderFactory) -> None:
    DIALOGUE_REGISTRY[name.lower()] = factory


def get_stt_provider(name: str) -> STTProvider:
    factory = STT_REGISTRY.get(name.lower())
    if factory is None:
        raise KeyError(f"STT provider not registered: {name!r}")
    return factory()


def get_tts_provider(name: str) -> TTSProvider:
    factory = TTS_REGISTRY.get(name.lower())
    if factory is None:
        raise KeyError(f"TTS provider not registered: {name!r}")
    return factory()


def get_dialogue_provider(name: str) -> "FullDuplexDialogueProvider":  # type: ignore[name-defined]  # noqa: F821
    """Construct (not reuse) a full-duplex dialogue provider.

    Unlike STT/TTS providers which are typically stateless wrappers, the
    F-65 dialogue provider is *session-scoped* (a live WebSocket + pump
    tasks per session). Returning a fresh instance per call sidesteps
    any cross-session state leak; the caller (``DialogueSessionManager``
    in P65-B) owns it for the duration of one session.
    """
    # Local import: ABC lives in dialogue.py which imports ``websockets``
    # not at module scope, but keeping the import local makes the
    # dependency graph explicit at the only call site that materialises
    # the type. Anything that uses STT/TTS paths never touches
    # ``dialogue.py`` so the cold-start cost stays zero (F-64 STG-6 perf).

    factory = DIALOGUE_REGISTRY.get(name.lower())
    if factory is None:
        raise KeyError(f"Dialogue provider not registered: {name!r}")
    return factory()


def list_stt_providers() -> list[str]:
    return sorted(STT_REGISTRY)


def list_tts_providers() -> list[str]:
    return sorted(TTS_REGISTRY)


def list_dialogue_providers() -> list[str]:
    return sorted(DIALOGUE_REGISTRY)


def _register_builtins() -> None:
    # STT factories
    def _anthropic_factory() -> STTProvider:
        from .anthropic_stt import AnthropicSTTProvider

        return AnthropicSTTProvider()

    def _doubao_factory() -> STTProvider:
        from .doubao_stt import DoubaoSTTProvider

        return DoubaoSTTProvider()

    def _minimax_factory() -> STTProvider:
        from .minimax_stt import MiniMaxSTTProvider

        return MiniMaxSTTProvider()

    register_stt_provider("anthropic", _anthropic_factory)
    register_stt_provider("doubao", _doubao_factory)
    register_stt_provider("minimax", _minimax_factory)

    # TTS factories (P64-E7)
    def _openai_tts_factory() -> TTSProvider:
        from .openai_tts import OpenAITTSProvider

        return OpenAITTSProvider()

    def _minimax_tts_factory() -> TTSProvider:
        from .minimax_tts import MiniMaxTTSProvider

        return MiniMaxTTSProvider()

    def _gemini_tts_factory() -> TTSProvider:
        from .gemini_tts import GeminiTTSProvider

        return GeminiTTSProvider()

    register_tts_provider("openai", _openai_tts_factory)
    register_tts_provider("minimax", _minimax_tts_factory)
    register_tts_provider("gemini", _gemini_tts_factory)

    # F-65 P65-A: full-duplex dialogue factories. Local-import the
    # provider class inside the factory so REPL cold-start doesn't pay
    # for the websockets import unless the user actually starts a
    # dialogue session. Matches the F-64 STT/TTS lazy-load pattern.
    def _minimax_dialogue_factory() -> "FullDuplexDialogueProvider":  # type: ignore[name-defined]  # noqa: F821
        from .minimax_realtime_dialogue import MiniMaxRealtimeDialogueProvider

        return MiniMaxRealtimeDialogueProvider()

    register_dialogue_provider("minimax", _minimax_dialogue_factory)
    # ``openai-realtime`` is reserved for the P65-E reference adapter;
    # registering here would require pulling OpenAI-specific deps at
    # import time. Left as a future entry; ``/dialogue openai-realtime``
    # will surface "provider not registered" until that ships.


_register_builtins()
