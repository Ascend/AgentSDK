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

"""Voice activity detection.

Mirrors TypeScript voice/detection.ts — detects speech in audio streams.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from enum import Enum
from typing import Callable


class VoiceActivityState(str, Enum):
    IDLE = "idle"
    LISTENING = "listening"
    SPEAKING = "speaking"
    PROCESSING = "processing"


@dataclass
class VoiceActivityConfig:
    """Configuration for voice activity detection."""

    silence_threshold_db: float = -40.0
    speech_threshold_db: float = -25.0
    min_speech_duration_ms: float = 200.0
    max_silence_duration_ms: float = 1500.0
    sample_rate: int = 16000


class VoiceActivityDetector:
    """Detects voice activity in audio streams.

    Uses simple energy-based detection. Full implementation would use
    a proper VAD model (e.g., Silero VAD).
    """

    def __init__(self, config: VoiceActivityConfig | None = None) -> None:
        self._config = config or VoiceActivityConfig()
        self._state = VoiceActivityState.IDLE
        self._speech_start: float | None = None
        self._last_speech: float | None = None
        self._listeners: list[Callable[[VoiceActivityState], None]] = []

    @property
    def state(self) -> VoiceActivityState:
        return self._state

    @property
    def is_speaking(self) -> bool:
        return self._state == VoiceActivityState.SPEAKING

    def start(self) -> None:
        """Start listening for voice activity."""
        self._set_state(VoiceActivityState.LISTENING)

    def stop(self) -> None:
        """Stop listening."""
        self._set_state(VoiceActivityState.IDLE)
        self._speech_start = None
        self._last_speech = None

    def process_audio_level(self, level_db: float) -> VoiceActivityState:
        """Process an audio level reading and update state.

        Args:
            level_db: Audio level in decibels.

        Returns:
            Current voice activity state.
        """
        now = time.time() * 1000  # ms

        if self._state == VoiceActivityState.IDLE:
            return self._state

        if level_db >= self._config.speech_threshold_db:
            if self._speech_start is None:
                self._speech_start = now
            self._last_speech = now

            speech_duration = now - self._speech_start
            if speech_duration >= self._config.min_speech_duration_ms:
                if self._state != VoiceActivityState.SPEAKING:
                    self._set_state(VoiceActivityState.SPEAKING)
        elif self._last_speech is not None:
            silence_duration = now - self._last_speech
            if silence_duration >= self._config.max_silence_duration_ms:
                if self._state == VoiceActivityState.SPEAKING:
                    self._set_state(VoiceActivityState.PROCESSING)
                self._speech_start = None
                self._last_speech = None

        return self._state

    def on_state_change(self, listener: Callable[[VoiceActivityState], None]) -> Callable[[], None]:
        """Register a state change listener. Returns unsubscribe."""
        self._listeners.append(listener)

        def unsubscribe() -> None:
            if listener in self._listeners:
                self._listeners.remove(listener)

        return unsubscribe

    def _set_state(self, new_state: VoiceActivityState) -> None:
        if new_state != self._state:
            self._state = new_state
            for listener in self._listeners:
                try:
                    listener(new_state)
                except Exception:  # nosec B110 - a faulty VAD listener must not break state updates
                    pass
