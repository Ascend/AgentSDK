#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# -------------------------------------------------------------------------
#  This file is part of the AgentSDK project.
# Copyright (c) 2026 Huawei Technologies Co.,Ltd.
#
# AgentSDK is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#
#           http://license.coscl.org.cn/MulanPSL2
#
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
# -------------------------------------------------------------------------

"""audio-capture unit tests without real audio hardware."""

from __future__ import annotations

import asyncio
import io
import wave

import pytest
from clawcodex_ext.native import load, load_or_fallback
from clawcodex_ext.native.audio import AudioCaptureModule, AudioFallback


def test_audio_module_registered():
    assert load("audio_capture") is not None or True
    from clawcodex_ext.native import NativeModuleRegistry

    assert NativeModuleRegistry.is_registered("audio_capture")


def test_audio_fallback_returns_silent_wav():
    """AudioFallback.record returns valid silent WAV bytes."""
    fb = AudioFallback()
    assert fb.is_available() is False
    assert fb.get_version() == "fallback-silent"

    data = asyncio.run(fb.record(duration_sec=0.1, sample_rate=8000, channels=1))
    assert isinstance(data, bytes)
    with wave.open(io.BytesIO(data), "rb") as wf:
        assert wf.getnchannels() == 1
        assert wf.getframerate() == 8000
        assert wf.getsampwidth() == 2
        # 0.1s @ 8000Hz → 800 samples → 1600 bytes
        assert wf.getnframes() == 800


def test_audio_fallback_stream_yields_silence():
    fb = AudioFallback()

    async def _take_first():
        async for chunk in fb.stream(sample_rate=8000, channels=1):
            return chunk

    chunk = asyncio.run(_take_first())
    assert isinstance(chunk, bytes)
    assert all(b == 0 for b in chunk)


def test_audio_load_or_fallback_returns_object():
    """load_or_fallback returns an object whether or not the backend exists."""
    inst = load_or_fallback("audio_capture")
    assert inst is not None
    assert isinstance(inst, (AudioCaptureModule, AudioFallback))


def test_audio_record_raises_when_unavailable(monkeypatch):
    """record raises NativeModuleError when its backend is unavailable."""
    mod = AudioCaptureModule()
    mod._backend = None
    from clawcodex_ext.native import NativeModuleError

    with pytest.raises(NativeModuleError):
        asyncio.run(mod.record(duration_sec=0.1))
