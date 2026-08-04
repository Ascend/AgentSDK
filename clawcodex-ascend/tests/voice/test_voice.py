#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# -------------------------------------------------------------------------
#  This file is part of the AgentSDK project.
# Copyright (c) 2026 Huawei Technologies Co.,Ltd.
# Copyright (c) 2026 Clawd Codex Team
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

"""Tests for Voice subsystem — F-64 P64-A/B/C.

Covers:
* Detection + STT abstract types (existing baseline).
* :mod:`voice_mode_enabled` three-layer gate.
* :mod:`provider_registry` registration + lookup.
* :mod:`audio_chunk_queue` push→pull async bridge.
* :class:`PushToTalkController` lifecycle with a stub recorder + stub provider.
* ``/voice`` command (toggle / provider select / status / help / errors).
* Settings schema round-trip of ``voice_provider`` / ``voice_enabled``.
"""

from __future__ import annotations

import asyncio
import os
import unittest
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from unittest import mock

from clawcodex_ext.services.voice.audio_chunk_queue import AudioChunkQueue
from clawcodex_ext.services.voice.audio_recorder import AudioRecorder
from clawcodex_ext.services.voice.detection import (
    VoiceActivityConfig,
    VoiceActivityDetector,
    VoiceActivityState,
)
from clawcodex_ext.services.voice.provider_registry import (
    STT_REGISTRY,
    get_stt_provider,
    list_stt_providers,
    register_stt_provider,
)
from clawcodex_ext.services.voice.push_to_talk import (
    PushToTalkController,
    VoiceSessionState,
)
from clawcodex_ext.services.voice.stt import STTConfig, STTProvider, STTResult
from clawcodex_ext.services.voice import voice_mode_enabled as vme


# ── Existing baseline tests (preserved) ────────────────────────────────────


class TestSTTTypes(unittest.TestCase):
    def test_stt_config_defaults(self) -> None:
        config = STTConfig()
        self.assertEqual(config.language, "en")
        self.assertEqual(config.sample_rate, 16000)
        self.assertTrue(config.interim_results)

    def test_stt_result(self) -> None:
        result = STTResult(text="hello world", confidence=0.95, is_final=True)
        self.assertEqual(result.text, "hello world")
        self.assertEqual(result.confidence, 0.95)
        self.assertTrue(result.is_final)


class TestVoiceActivityDetector(unittest.TestCase):
    def test_initial_state(self) -> None:
        vad = VoiceActivityDetector()
        self.assertEqual(vad.state, VoiceActivityState.IDLE)
        self.assertFalse(vad.is_speaking)

    def test_start_stop(self) -> None:
        vad = VoiceActivityDetector()
        vad.start()
        self.assertEqual(vad.state, VoiceActivityState.LISTENING)
        vad.stop()
        self.assertEqual(vad.state, VoiceActivityState.IDLE)

    def test_speech_detection(self) -> None:
        config = VoiceActivityConfig(
            min_speech_duration_ms=0,
            speech_threshold_db=-25.0,
        )
        vad = VoiceActivityDetector(config=config)
        vad.start()
        state = vad.process_audio_level(-10.0)
        self.assertEqual(state, VoiceActivityState.SPEAKING)
        self.assertTrue(vad.is_speaking)

    def test_silence_after_speech(self) -> None:
        config = VoiceActivityConfig(
            min_speech_duration_ms=0,
            max_silence_duration_ms=0,
        )
        vad = VoiceActivityDetector(config=config)
        vad.start()
        vad.process_audio_level(-10.0)
        self.assertTrue(vad.is_speaking)
        state = vad.process_audio_level(-50.0)
        self.assertEqual(state, VoiceActivityState.PROCESSING)

    def test_idle_ignores_audio(self) -> None:
        vad = VoiceActivityDetector()
        state = vad.process_audio_level(-10.0)
        self.assertEqual(state, VoiceActivityState.IDLE)

    def test_state_change_listener(self) -> None:
        vad = VoiceActivityDetector()
        states: list[VoiceActivityState] = []
        unsub = vad.on_state_change(lambda s: states.append(s))
        vad.start()
        vad.stop()
        self.assertEqual(states, [VoiceActivityState.LISTENING, VoiceActivityState.IDLE])
        unsub()
        vad.start()
        self.assertEqual(len(states), 2)

    def test_voice_activity_config_defaults(self) -> None:
        config = VoiceActivityConfig()
        self.assertEqual(config.silence_threshold_db, -40.0)
        self.assertEqual(config.speech_threshold_db, -25.0)
        self.assertEqual(config.min_speech_duration_ms, 200.0)
        self.assertEqual(config.max_silence_duration_ms, 1500.0)

    def test_voice_activity_config_custom(self) -> None:
        config = VoiceActivityConfig(speech_threshold_db=-30.0, sample_rate=8000)
        self.assertEqual(config.speech_threshold_db, -30.0)
        self.assertEqual(config.sample_rate, 8000)


# ── P64-A: voice_mode_enabled three-layer gate ─────────────────────────────


class TestVoiceModeEnabled(unittest.TestCase):
    def setUp(self) -> None:
        # Clear F-64 env vars so each test starts from a known state.
        for k in ("FEATURE_VOICE_MODE", "CLAWCODEX_VOICE_DISABLED"):
            os.environ.pop(k, None)

    def test_feature_flag_off_by_default(self) -> None:
        self.assertFalse(vme.is_voice_feature_enabled())

    def test_feature_flag_accepts_truthy_values(self) -> None:
        for v in ("1", "true", "True", "yes", "on", "ON"):
            os.environ["FEATURE_VOICE_MODE"] = v
            self.assertTrue(vme.is_voice_feature_enabled(), f"failed for {v!r}")
        os.environ.pop("FEATURE_VOICE_MODE", None)

    def test_feature_flag_rejects_falsy_values(self) -> None:
        for v in ("", "0", "false", "no", "off", "anything"):
            os.environ["FEATURE_VOICE_MODE"] = v
            self.assertFalse(vme.is_voice_feature_enabled(), f"failed for {v!r}")
        os.environ.pop("FEATURE_VOICE_MODE", None)

    def test_kill_switch_off_by_default(self) -> None:
        self.assertFalse(vme.is_voice_disabled_by_kill_switch())

    def test_kill_switch_engaged_by_truthy_value(self) -> None:
        for v in ("1", "true", "yes", "on"):
            os.environ["CLAWCODEX_VOICE_DISABLED"] = v
            self.assertTrue(vme.is_voice_disabled_by_kill_switch(), f"failed for {v!r}")
        os.environ.pop("CLAWCODEX_VOICE_DISABLED", None)

    def test_is_voice_available_flag_off(self) -> None:
        os.environ.pop("FEATURE_VOICE_MODE", None)
        self.assertFalse(vme.is_voice_available())

    def test_is_voice_available_flag_on_no_kill(self) -> None:
        os.environ["FEATURE_VOICE_MODE"] = "1"
        self.assertTrue(vme.is_voice_available())

    def test_is_voice_available_flag_on_kill_set(self) -> None:
        os.environ["FEATURE_VOICE_MODE"] = "1"
        os.environ["CLAWCODEX_VOICE_DISABLED"] = "1"
        self.assertFalse(vme.is_voice_available())

    def test_is_voice_mode_enabled_needs_auth(self) -> None:
        # Anthropic path requires OAuth; with no auth module available in
        # the test env, has_voice_auth returns False, so the gate is off
        # even with the flag on and kill-switch clear.
        os.environ["FEATURE_VOICE_MODE"] = "1"
        os.environ.pop("CLAWCODEX_VOICE_DISABLED", None)
        with mock.patch.object(vme, "has_voice_auth", return_value=True):
            self.assertTrue(vme.is_voice_mode_enabled())
        with mock.patch.object(vme, "has_voice_auth", return_value=False):
            self.assertFalse(vme.is_voice_mode_enabled())

    def test_get_voice_provider_defaults_to_anthropic(self) -> None:
        with mock.patch(
            "src.settings.settings.get_settings",
            side_effect=RuntimeError("no settings in test"),
        ):
            self.assertEqual(vme.get_voice_provider(), "anthropic")

    def test_get_voice_provider_reads_settings(self) -> None:
        @dataclass
        class _S:
            voice_provider: str = "doubao"

        with mock.patch("src.settings.settings.get_settings", return_value=_S()):
            self.assertEqual(vme.get_voice_provider(), "doubao")

    def test_get_voice_provider_unknown_falls_back(self) -> None:
        @dataclass
        class _S:
            voice_provider: str = "watson"

        with mock.patch("src.settings.settings.get_settings", return_value=_S()):
            self.assertEqual(vme.get_voice_provider(), "anthropic")


# ── P64-A: provider_registry ───────────────────────────────────────────────


class _StubProvider(STTProvider):
    """Minimal STTProvider for registry/push-to-talk tests."""

    async def transcribe(self, audio_data, config=None):
        return STTResult(text="stub")

    async def start_streaming(self, config=None):
        return None

    async def feed_audio(self, chunk):
        return None

    async def stop_streaming(self):
        return STTResult(text="stub-final")

    async def close(self):
        return None


class TestProviderRegistry(unittest.TestCase):
    def test_builtin_providers_registered(self) -> None:
        names = list_stt_providers()
        self.assertIn("anthropic", names)
        self.assertIn("doubao", names)

    def test_get_unknown_provider_raises(self) -> None:
        with self.assertRaises(KeyError):
            get_stt_provider("nonexistent")

    def test_register_custom_provider(self) -> None:
        register_stt_provider("stub", lambda: _StubProvider())
        try:
            self.assertIn("stub", list_stt_providers())
            provider = get_stt_provider("stub")
            self.assertIsInstance(provider, _StubProvider)
        finally:
            STT_REGISTRY.pop("stub", None)

    def test_register_is_case_insensitive(self) -> None:
        register_stt_provider("MyStub", lambda: _StubProvider())
        try:
            self.assertIn("mystub", STT_REGISTRY)
            provider = get_stt_provider("MYSTUB")
            self.assertIsInstance(provider, _StubProvider)
        finally:
            STT_REGISTRY.pop("mystub", None)


# ── P64-C: AudioChunkQueue ─────────────────────────────────────────────────


class TestAudioChunkQueue(unittest.TestCase):
    def test_push_then_iterate(self) -> None:
        async def run() -> None:
            q = AudioChunkQueue()
            q.push(b"a")
            q.push(b"b")
            q.push(None)  # close
            chunks = [c async for c in q]
            self.assertEqual(chunks, [b"a", b"b"])

        asyncio.run(run())

    def test_close_without_data_terminates(self) -> None:
        async def run() -> None:
            q = AudioChunkQueue()
            q.push(None)
            chunks = [c async for c in q]
            self.assertEqual(chunks, [])

        asyncio.run(run())

    def test_push_after_close_is_noop(self) -> None:
        async def run() -> None:
            q = AudioChunkQueue()
            q.push(None)
            q.push(b"late")  # should be ignored
            chunks = [c async for c in q]
            self.assertEqual(chunks, [])

        asyncio.run(run())

    def test_async_wait_for_chunks(self) -> None:
        """Consumer blocks on an empty queue and wakes when producer pushes."""

        async def run() -> None:
            q = AudioChunkQueue()
            received: list[bytes] = []

            async def consumer() -> None:
                async for c in q:
                    received.append(c)

            task = asyncio.create_task(consumer())
            await asyncio.sleep(0)  # let consumer enter __anext__
            q.push(b"frame-1")
            q.push(b"frame-2")
            q.push(None)
            await task
            self.assertEqual(received, [b"frame-1", "frame-2".encode()])

        asyncio.run(run())


# ── P64-B: PushToTalkController with stubs ─────────────────────────────────


class _StubRecorder(AudioRecorder):
    """Records nothing; lets tests drive the controller without a mic."""

    def __init__(self) -> None:
        self._recording = False
        self._cb: Optional[object] = None

    def start(self, on_chunk) -> None:
        self._recording = True
        self._cb = on_chunk

    def stop(self) -> None:
        self._recording = False

    @property
    def is_recording(self) -> bool:
        return self._recording


class _StubStreamConnection:
    """Mimics VoiceStreamConnection / DoubaoStreamConnection for unit tests."""

    def __init__(self, *, final_text: str = "hello") -> None:
        self._final = final_text
        self.feed_calls: list[bytes] = []
        self.closed = False

    def feed_audio(self, chunk: bytes) -> None:
        self.feed_calls.append(chunk)

    async def finalize(self) -> str:
        return self._final

    async def close(self) -> None:
        self.closed = True


class _StubStreamingProvider(STTProvider):
    """Provider whose ``connect_stream`` returns a stub connection."""

    def __init__(self, *, final_text: str = "transcribed text") -> None:
        self._final = final_text
        self.last_connection: Optional[_StubStreamConnection] = None
        self.last_kwargs: dict = {}

    def connect_stream(self, **kwargs):
        self.last_kwargs = kwargs
        self.last_connection = _StubStreamConnection(final_text=self._final)
        # Fire on_ready immediately to mimic the doubao path.
        on_ready = kwargs.get("on_ready")
        if on_ready:
            on_ready()
        return self.last_connection

    async def transcribe(self, audio_data, config=None):
        return STTResult(text=self._final)

    async def start_streaming(self, config=None):
        return None

    async def feed_audio(self, chunk):
        return None

    async def stop_streaming(self):
        return STTResult(text=self._final)

    async def close(self):
        return None


class TestPushToTalkController(unittest.TestCase):
    def setUp(self) -> None:
        os.environ["FEATURE_VOICE_MODE"] = "1"
        os.environ.pop("CLAWCODEX_VOICE_DISABLED", None)
        # Patches kept alive for the whole test via self.addCleanup.
        self._patches: list[object] = []

    def tearDown(self) -> None:
        os.environ.pop("FEATURE_VOICE_MODE", None)
        for p in self._patches:
            p.stop()

    def _patch(self, target: str, **kwargs) -> mock.MagicMock:
        p = mock.patch(target, **kwargs)
        self._patches.append(p)
        return p.start()

    def _make_controller(self, *, provider: _StubStreamingProvider, enabled: bool = True) -> PushToTalkController:
        register_stt_provider("__test_stub", lambda: provider)
        recorder = _StubRecorder()
        self._patch(
            "clawcodex_ext.services.voice.push_to_talk.get_voice_provider",
            return_value="__test_stub",
        )
        self._patch(
            "clawcodex_ext.services.voice.push_to_talk.is_voice_enabled",
            return_value=enabled,
        )
        return PushToTalkController(recorder=recorder)

    def test_can_start_requires_enabled(self) -> None:
        controller = self._make_controller(provider=_StubStreamingProvider(), enabled=False)
        self.assertFalse(controller.can_start())

    def test_can_start_true_when_enabled(self) -> None:
        controller = self._make_controller(provider=_StubStreamingProvider(), enabled=True)
        self.assertTrue(controller.can_start())

    def test_start_transitions_to_recording(self) -> None:
        provider = _StubStreamingProvider()
        controller = self._make_controller(provider=provider, enabled=True)
        states: list[VoiceSessionState] = []
        controller._on_state_change = states.append  # type: ignore[assignment]
        errors: list[str] = []
        controller._on_error = errors.append  # type: ignore[assignment]
        ok = controller.start()
        self.assertTrue(ok, f"start() returned False; errors={errors}; can_start={controller.can_start()}")
        self.assertEqual(controller.state, VoiceSessionState.RECORDING)
        self.assertIn(VoiceSessionState.RECORDING, states)

    def test_stop_returns_final_transcript(self) -> None:
        provider = _StubStreamingProvider(final_text="final words")
        controller = self._make_controller(provider=provider, enabled=True)
        controller.start()
        result = asyncio.run(controller.stop())
        self.assertEqual(result.text, "final words")
        self.assertEqual(result.provider, "__test_stub")
        self.assertIsNone(result.error)

    def test_start_feeds_audio_to_connection(self) -> None:
        provider = _StubStreamingProvider()
        controller = self._make_controller(provider=provider, enabled=True)
        controller.start()
        # Simulate the recorder thread pushing a frame.
        controller._on_audio_chunk(b"pcm-frame")
        self.assertEqual(provider.last_connection.feed_calls, [b"pcm-frame"])
        asyncio.run(controller.stop())

    def test_disarm_cleans_up(self) -> None:
        provider = _StubStreamingProvider()
        controller = self._make_controller(provider=provider, enabled=True)
        controller.start()
        controller.disarm()
        self.assertEqual(controller.state, VoiceSessionState.IDLE)

    def test_unknown_provider_emits_error(self) -> None:
        recorder = _StubRecorder()
        self._patch(
            "clawcodex_ext.services.voice.push_to_talk.get_voice_provider",
            return_value="nonexistent",
        )
        self._patch(
            "clawcodex_ext.services.voice.push_to_talk.is_voice_enabled",
            return_value=True,
        )
        controller = PushToTalkController(recorder=recorder)
        errors: list[str] = []
        controller._on_error = errors.append  # type: ignore[assignment]
        ok = controller.start()
        self.assertFalse(ok)
        self.assertEqual(controller.state, VoiceSessionState.ERROR)
        self.assertTrue(errors)


# ── /voice command ─────────────────────────────────────────────────────────


def _make_context() -> object:
    """Build a minimal CommandContext for /voice tests."""
    from clawcodex_ext.command_system.types import CommandContext

    return CommandContext(workspace_root=Path("/tmp"), cwd=Path("/tmp"))


class TestVoiceCommand(unittest.TestCase):
    def setUp(self) -> None:
        os.environ["FEATURE_VOICE_MODE"] = "1"
        os.environ.pop("CLAWCODEX_VOICE_DISABLED", None)

    def tearDown(self) -> None:
        os.environ.pop("FEATURE_VOICE_MODE", None)

    def _run(self, args: str) -> str:
        from clawcodex_ext.command_system.voice_command import voice_command_call

        ctx = _make_context()
        result = voice_command_call(args, ctx)  # type: ignore[arg-type]
        return result.value

    def test_help(self) -> None:
        out = self._run("help")
        self.assertIn("Usage: /voice", out)

    def test_status_shows_layers(self) -> None:
        out = self._run("status")
        self.assertIn("Voice mode:", out)
        self.assertIn("Provider:", out)
        self.assertIn("feature flag", out)
        self.assertIn("kill-switch", out)

    def test_off_disables(self) -> None:
        with mock.patch("clawcodex_ext.command_system.voice_command.set_voice_enabled") as m:
            out = self._run("off")
        self.assertEqual(out, "Voice mode disabled.")
        m.assert_called_once_with(False)

    def test_anthropic_enables_and_sets_provider(self) -> None:
        with (
            mock.patch("clawcodex_ext.command_system.voice_command.set_voice_enabled") as m_en,
            mock.patch("clawcodex_ext.command_system.voice_command.set_voice_provider") as m_prov,
        ):
            out = self._run("anthropic")
        self.assertIn("enabled", out)
        self.assertIn("anthropic", out)
        m_en.assert_called_once_with(True)
        m_prov.assert_called_once_with("anthropic")

    def test_doubao_enables_and_sets_provider(self) -> None:
        with (
            mock.patch("clawcodex_ext.command_system.voice_command.set_voice_enabled") as m_en,
            mock.patch("clawcodex_ext.command_system.voice_command.set_voice_provider") as m_prov,
        ):
            out = self._run("doubao")
        self.assertIn("doubao", out)
        m_en.assert_called_once_with(True)
        m_prov.assert_called_once_with("doubao")

    def test_toggle_off_when_currently_on(self) -> None:
        with (
            mock.patch(
                "clawcodex_ext.command_system.voice_command.is_voice_enabled",
                return_value=True,
            ),
            mock.patch("clawcodex_ext.command_system.voice_command.set_voice_enabled") as m,
        ):
            out = self._run("")
        self.assertEqual(out, "Voice mode disabled.")
        m.assert_called_once_with(False)

    def test_toggle_on_when_currently_off(self) -> None:
        with (
            mock.patch(
                "clawcodex_ext.command_system.voice_command.is_voice_enabled",
                return_value=False,
            ),
            mock.patch(
                "clawcodex_ext.command_system.voice_command.get_voice_provider",
                return_value="anthropic",
            ),
            mock.patch("clawcodex_ext.command_system.voice_command.set_voice_enabled") as m,
        ):
            out = self._run("")
        self.assertIn("enabled", out)
        self.assertIn("anthropic", out)
        m.assert_called_once_with(True)

    def test_unknown_arg(self) -> None:
        out = self._run("watson")
        self.assertIn("Unknown argument", out)
        self.assertIn("anthropic", out)


# ── Settings schema round-trip ─────────────────────────────────────────────


class TestSettingsVoiceFields(unittest.TestCase):
    def test_defaults_empty_and_disabled(self) -> None:
        from clawcodex_ext.settings.types import SettingsSchema

        s = SettingsSchema()
        self.assertEqual(s.voice_provider, "")
        self.assertFalse(s.voice_enabled)

    def test_round_trip(self) -> None:
        from clawcodex_ext.settings.types import SettingsSchema

        s = SettingsSchema(voice_provider="doubao", voice_enabled=True)
        d = s.to_dict()
        self.assertEqual(d["voice_provider"], "doubao")
        self.assertTrue(d["voice_enabled"])
        s2 = SettingsSchema.from_dict(d)
        self.assertEqual(s2.voice_provider, "doubao")
        self.assertTrue(s2.voice_enabled)

    def test_from_dict_unknown_goes_to_extra(self) -> None:
        from clawcodex_ext.settings.types import SettingsSchema

        s = SettingsSchema.from_dict({"voice_provider": "anthropic", "future_field": 1})
        self.assertEqual(s.voice_provider, "anthropic")
        self.assertEqual(s.extra.get("future_field"), 1)


# ── set_voice_provider / set_voice_enabled persistence ─────────────────────


class TestVoicePersistence(unittest.TestCase):
    def test_set_voice_provider_normalizes(self) -> None:
        from src.config import set_voice_provider

        mgr = mock.MagicMock()
        mgr.load_global.return_value = {}
        with (
            mock.patch("src.config._get_default_manager", return_value=mgr),
            mock.patch("src.settings.settings.invalidate_settings_cache"),
        ):
            set_voice_provider("DOUBAO")
        section = mgr.save_global.call_args.args[0]["settings"]
        self.assertEqual(section["voice_provider"], "doubao")

    def test_set_voice_provider_invalid_coerces_empty(self) -> None:
        from src.config import set_voice_provider

        mgr = mock.MagicMock()
        mgr.load_global.return_value = {}
        with (
            mock.patch("src.config._get_default_manager", return_value=mgr),
            mock.patch("src.settings.settings.invalidate_settings_cache"),
        ):
            set_voice_provider("nonsense")
        section = mgr.save_global.call_args.args[0]["settings"]
        self.assertEqual(section["voice_provider"], "")

    def test_set_voice_enabled_writes_bool(self) -> None:
        from src.config import set_voice_enabled

        mgr = mock.MagicMock()
        mgr.load_global.return_value = {}
        with (
            mock.patch("src.config._get_default_manager", return_value=mgr),
            mock.patch("src.settings.settings.invalidate_settings_cache"),
        ):
            set_voice_enabled(True)
        section = mgr.save_global.call_args.args[0]["settings"]
        self.assertTrue(section["voice_enabled"])


if __name__ == "__main__":
    unittest.main()
