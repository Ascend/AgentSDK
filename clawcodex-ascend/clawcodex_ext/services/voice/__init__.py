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

"""Voice subsystem — Voice Mode + Voice Dialogue.

Provides speech-to-text and voice activity detection, plus the
push-to-talk recording controller and STT provider registry. P64-E
adds the symmetric text-to-speech surface (TTSProvider ABC, OpenAI /
MiniMax / Gemini providers, AudioOutQueue, AudioPlayer).

Layering * Detection + STT abstract base — mirror TS ``voice/`` interfaces.
* :mod:`voice_mode_enabled` — three-layer gate (flag / kill-switch / OAuth).
* :mod:`provider_registry` — STT + TTS backend factory registries.
* :mod:`anthropic_stt` — Anthropic Nova 3 WebSocket streaming (P64-A + P64-C).
* :mod:`doubao_stt` — Doubao ASR AsyncGenerator adapter (P64-A).
* :mod:`minimax_stt` — MiniMax Realtime API STT (P64-D1, voice-in → text-out).
* :mod:`tts` — TTSProvider ABC + TTSConfig + TTSChunk + TTSSynthesis (P64-E1).
* :mod:`openai_tts` — OpenAI audio.speech TTS (P64-E2, reference impl).
* :mod:`minimax_tts` — MiniMax T2A HTTP TTS (P64-E3).
* :mod:`gemini_tts` — Gemini generate_content TTS (P64-E4).
* :mod:`audio_chunk_queue` — push→pull async audio bridge (P64-C, STT side).
* :mod:`audio_out_queue` — bounded async TTS frame queue (P64-E8).
* :mod:`audio_player` — PyAudio / SoX / ffplay PCM player (P64-E8).
* :mod:`audio_recorder` — cross-platform PCM capture (P64-B).
* :mod:`push_to_talk` — recording session orchestrator (P64-B).

Layering (P65-A/B/C/D):
* :mod:`dialogue` — FullDuplexDialogueProvider ABC (new).
* :mod:`minimax_realtime_dialogue` — MiniMax Realtime full-duplex adapter.
* :mod:`interrupt` — energy-based VAD for barge-in.
* :mod:`dialogue_session` — DialogueSessionManager state machine.
"""

from __future__ import annotations

from .audio_chunk_queue import AudioChunkQueue
from .audio_out_queue import AudioOutQueue
from .audio_player import AudioPlayer, has_ffplay, has_pyaudio, has_sox, play_pcm
from .audio_recorder import (
    AudioRecorder,
    PyAudioRecorder,
    SoXRecorder,
    make_recorder,
)
from .detection import (
    VoiceActivityConfig,
    VoiceActivityDetector,
    VoiceActivityState,
)
from .doubao_stt import (
    DOUBAO_CREDENTIALS_PATH,
    DoubaoCredentialsError,
    DoubaoSTTProvider,
    DoubaoStreamConnection,
)
from .anthropic_stt import (
    ANTHROPIC_VOICE_ENDPOINT,
    AnthropicSTTProvider,
    VoiceAuthError,
    VoiceStreamConnection,
)
from .minimax_stt import (
    MINIMAX_CREDENTIALS_PATH,
    MINIMAX_REALTIME_ENDPOINTS,
    MiniMaxCredentialsError,
    MiniMaxSTTProvider,
    MiniMaxStreamConnection,
)
from .tts import TTSChunk, TTSConfig, TTSProvider, TTSSynthesis
from .openai_tts import (
    OPENAI_TTS_ENDPOINT,
    OpenAITTSCredentialsError,
    OpenAITTSProvider,
)
from .minimax_tts import (
    MINIMAX_SUPPORTED_MODELS,
    MINIMAX_SYSTEM_VOICES,
    MINIMAX_T2A_ENDPOINTS,
    MINIMAX_TTS_CREDENTIALS_PATH,
    MiniMaxTTSCredentialsError,
    MiniMaxTTSProvider,
)
from .gemini_tts import (
    GEMINI_TTS_DEFAULT_MODEL,
    GEMINI_TTS_DEFAULT_VOICE,
    GeminiTTSCredentialsError,
    GeminiTTSProvider,
)
from .provider_registry import (
    DIALOGUE_REGISTRY,
    DialogueProviderFactory,
    STT_REGISTRY,
    STTProviderFactory,
    TTS_REGISTRY,
    TTSProviderFactory,
    get_stt_provider,
    get_tts_provider,
    list_dialogue_providers,
    list_stt_providers,
    list_tts_providers,
    register_dialogue_provider,
    register_stt_provider,
    register_tts_provider,
)
from .push_to_talk import (
    PushToTalkController,
    VoiceSessionResult,
    VoiceSessionState,
)
from .stt import STTConfig, STTProvider, STTResult
from .voice_mode_enabled import (
    VOICE_PROVIDERS,
    DIALOGUE_PROVIDERS,
    DialogueProvider,
    VoiceProvider,
    get_dialogue_provider as get_dialogue_provider_settings,
    get_voice_provider,
    has_dialogue_auth,
    has_voice_auth,
    is_dialogue_available,
    is_dialogue_enabled,
    is_dialogue_feature_enabled,
    is_voice_available,
    is_voice_disabled_by_kill_switch,
    is_voice_enabled,
    is_voice_feature_enabled,
    is_voice_mode_enabled,
)

# dialogue modules — dialogue.py / interrupt.py / dialogue_session
# .py / minimax_realtime_dialogue.py — are deliberately NOT imported
# here. They live behind a ``__getattr__`` lazy resolver below to keep
# REPL cold-start unaffected: a typical voice-only install never
# touches ``/dialogue`` and shouldn't pay the import cost (Stage-6
# perf invariant). The provider factory function (``get_dialogue_provider``)
# is the canonical entry point used by the ``/dialogue`` command and
# resolves the dialogue modules on demand.

__all__ = [
    # Detection + STT base
    "STTConfig",
    "STTProvider",
    "STTResult",
    "VoiceActivityConfig",
    "VoiceActivityDetector",
    "VoiceActivityState",
    # Voice-mode gating
    "VOICE_PROVIDERS",
    "VoiceProvider",
    "get_voice_provider",
    "has_voice_auth",
    "is_voice_available",
    "is_voice_disabled_by_kill_switch",
    "is_voice_enabled",
    "is_voice_feature_enabled",
    "is_voice_mode_enabled",
    # Dialogue-mode gating
    "DIALOGUE_PROVIDERS",
    "DialogueProvider",
    "get_dialogue_provider_settings",
    "has_dialogue_auth",
    "is_dialogue_available",
    "is_dialogue_enabled",
    "is_dialogue_feature_enabled",
    # Provider registries (STT + TTS + Dialogue)
    "STT_REGISTRY",
    "STTProviderFactory",
    "TTS_REGISTRY",
    "TTSProviderFactory",
    "DIALOGUE_REGISTRY",
    "DialogueProviderFactory",
    "get_stt_provider",
    "get_tts_provider",
    "list_stt_providers",
    "list_tts_providers",
    "register_stt_provider",
    "register_tts_provider",
    "register_dialogue_provider",
    "list_dialogue_providers",
    # Anthropic STT backend
    "ANTHROPIC_VOICE_ENDPOINT",
    "AnthropicSTTProvider",
    "VoiceAuthError",
    "VoiceStreamConnection",
    # Doubao STT backend
    "DOUBAO_CREDENTIALS_PATH",
    "DoubaoCredentialsError",
    "DoubaoSTTProvider",
    "DoubaoStreamConnection",
    # MiniMax STT backend (P64-D1)
    "MINIMAX_CREDENTIALS_PATH",
    "MINIMAX_REALTIME_ENDPOINTS",
    "MiniMaxCredentialsError",
    "MiniMaxSTTProvider",
    "MiniMaxStreamConnection",
    # TTS abstraction (P64-E1)
    "TTSChunk",
    "TTSConfig",
    "TTSProvider",
    "TTSSynthesis",
    # TTS providers (P64-E2/E3/E4)
    "OPENAI_TTS_ENDPOINT",
    "OpenAITTSCredentialsError",
    "OpenAITTSProvider",
    "MINIMAX_SUPPORTED_MODELS",
    "MINIMAX_SYSTEM_VOICES",
    "MINIMAX_T2A_ENDPOINTS",
    "MINIMAX_TTS_CREDENTIALS_PATH",
    "MiniMaxTTSCredentialsError",
    "MiniMaxTTSProvider",
    "GEMINI_TTS_DEFAULT_MODEL",
    "GEMINI_TTS_DEFAULT_VOICE",
    "GeminiTTSCredentialsError",
    "GeminiTTSProvider",
    # Audio pipeline (STT in + TTS out)
    "AudioChunkQueue",
    "AudioOutQueue",
    "AudioPlayer",
    "AudioRecorder",
    "PyAudioRecorder",
    "SoXRecorder",
    "has_ffplay",
    "has_pyaudio",
    "has_sox",
    "make_recorder",
    "play_pcm",
    # Push-to-talk orchestrator
    "PushToTalkController",
    "VoiceSessionResult",
    "VoiceSessionState",
]


def __getattr__(name: str):
    """Lazy attribute resolution for dialogue types.

    ``dialogue`` / ``interrupt`` / ``dialogue_session`` /
    ``minimax_realtime_dialogue`` aren't imported at module scope — that
    would force the (potentially unused) ``websockets`` import at REPL
    boot. They're pulled in on first attribute access via this hook.
    Keeping the resolver here means ``from clawcodex_ext.services.
    voice import FullDuplexDialogueProvider`` Just Works without
    changing import semantics for the surface.
    """
    if name in (
        "DialogueConfig",
        "DialogueEvent",
        "FullDuplexDialogueProvider",
        "DialogueState",
        "DialogueModality",
    ):
        from . import dialogue as _dialogue_mod

        return getattr(_dialogue_mod, name)
    if name in (
        "InterruptConfig",
        "InterruptDecision",
        "InterruptDetector",
    ):
        from . import interrupt as _interrupt_mod

        return getattr(_interrupt_mod, name)
    if name in (
        "DialogueSessionManager",
        "DialogueSessionState",
        "DialogueSessionCallbacks",
        "DialogueSessionOptions",
    ):
        from . import dialogue_session as _session_mod

        return getattr(_session_mod, name)
    if name in (
        "MINIMAX_REALTIME_CREDENTIALS_PATH",
        "MINIMAX_REALTIME_ENDPOINTS",
        "MiniMaxCredentialsError",
        "MiniMaxRealtimeDialogueProvider",
    ):
        from . import minimax_realtime_dialogue as _minimax_mod

        return getattr(_minimax_mod, name)
    if name == "get_dialogue_provider":
        # The registry helper (factory) is the one used by /dialogue.
        # The settings-backed getter is exposed as
        # ``get_dialogue_provider_settings`` to avoid ambiguity.
        from .provider_registry import get_dialogue_provider as _factory

        return _factory
    raise AttributeError(f"module 'clawcodex_ext.services.voice' has no attribute {name!r}")
