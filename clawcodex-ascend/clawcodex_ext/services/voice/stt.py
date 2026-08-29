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

"""Speech-to-text provider.

Mirrors TypeScript voice/stt.ts — abstract STT interface and configuration.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class STTConfig:
    """Speech-to-text configuration."""

    language: str = "en"
    model: str = "whisper-1"
    sample_rate: int = 16000
    encoding: str = "pcm_s16le"
    interim_results: bool = True


@dataclass
class STTResult:
    """Result of a speech-to-text transcription."""

    text: str
    confidence: float = 1.0
    is_final: bool = True
    language: str = "en"
    duration_ms: float = 0.0


class STTProvider(ABC):
    """Abstract speech-to-text provider."""

    @abstractmethod
    async def transcribe(self, audio_data: bytes, config: STTConfig | None = None) -> STTResult:
        """Transcribe audio data to text."""

    @abstractmethod
    async def start_streaming(self, config: STTConfig | None = None) -> None:
        """Start streaming transcription."""

    @abstractmethod
    async def feed_audio(self, chunk: bytes) -> STTResult | None:
        """Feed an audio chunk. Returns interim result if available."""

    @abstractmethod
    async def stop_streaming(self) -> STTResult:
        """Stop streaming and return final result."""

    @abstractmethod
    async def close(self) -> None:
        """Release resources."""
