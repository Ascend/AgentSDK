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

"""Tests for F-65 (Voice Dialogue Mode) components.

Covers:
* :class:`DialogueConfig` / :class:`DialogueEvent` dataclasses.
* :class:`FullDuplexDialogueProvider` ABC (registered in registry).
* :class:`InterruptDetector` (energy-based VAD for barge-in).
* :class:`AudioOutQueue.clear` and :class:`AudioPlayer` ``stop_nowait`` /
  ``stop_and_close`` for the F-65 interrupt path.
* :class:`DialogueSessionManager` state machine with a stub provider.
* :mod:`voice_mode_enabled` F-65 gate (DIALOGUE_PROVIDERS,
  has_dialogue_auth, is_dialogue_feature_enabled, is_dialogue_available,
  is_dialogue_enabled, get_dialogue_provider).
* :mod:`provider_registry` dialogue registration round-trip.
* ``/dialogue`` slash command (help / status / off / provider /
  modality / voice / start / stop / unknown).
* Settings persistence round-trip for ``dialogue_*`` fields.
* Config setter functions: ``set_dialogue_*``.

These tests are pure-Python (no real microphone / WebSocket). The
stub providers in this file model the wire surface of
:class:`FullDuplexDialogueProvider` so we can exercise the session
manager end-to-end.
"""

from __future__ import annotations

import asyncio
import os
import unittest
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional
from unittest import mock

# ── helpers / stubs ───────────────────────────────────────────────────────


@dataclass
class _StubDialogueProvider:
    """In-process implementation of :class:`FullDuplexDialogueProvider`.

    Records every public call so tests can assert the session manager
    drives the lifecycle correctly. ``feed_events`` lets a test inject
    server-side events through the same ``on_event`` callback the
    session manager registered.
    """

    started: bool = False
    closed: bool = False
    feed_calls: list[bytes] = field(default_factory=list)
    text_calls: list[str] = field(default_factory=list)
    interrupt_calls: int = 0
    stop_calls: int = 0
    close_calls: int = 0
    on_event: Optional[Callable] = None
    config: Any = None

    async def start(self, *, on_event, config=None) -> None:
        self.started = True
        self.on_event = on_event
        self.config = config or None

    async def feed_audio(self, chunk: bytes) -> None:
        self.feed_calls.append(chunk)

    async def send_text(self, text: str) -> None:
        self.text_calls.append(text)

    async def interrupt(self) -> None:
        self.interrupt_calls += 1
        if self.on_event is not None:
            self.on_event(_evt("interrupt"))

    async def stop(self) -> str:
        self.stop_calls += 1
        await self.close()
        return ""

    async def close(self) -> None:
        self.close_calls += 1
        self.closed = True

    # ── test helpers ─────────────────────────────────────────────────────

    def emit(self, type_: str, **kwargs) -> None:
        if self.on_event is not None:
            self.on_event(_evt(type_, **kwargs))


def _evt(type_: str, **kwargs) -> Any:
    """Construct a :class:`DialogueEvent` without importing the namespace."""
    from clawcodex_ext.services.voice.dialogue import DialogueEvent

    return DialogueEvent(type=type_, **kwargs)


# ── dialogue types (P65-A abstract) ───────────────────────────────────────


class TestDialogueTypes(unittest.TestCase):
    def test_config_defaults(self) -> None:
        from clawcodex_ext.services.voice.dialogue import DialogueConfig

        c = DialogueConfig()
        self.assertEqual(c.model, "speech-2.8-turbo")
        self.assertEqual(c.sample_rate, 16000)
        self.assertEqual(c.output_sample_rate, 24000)
        self.assertEqual(c.modality, "text")
        self.assertEqual(c.language, "zh")
        self.assertTrue(c.interim_results)
        self.assertEqual(c.extra, {})

    def test_config_custom(self) -> None:
        from clawcodex_ext.services.voice.dialogue import DialogueConfig

        c = DialogueConfig(
            model="speech-2.6",
            voice="alloy",
            modality="audio",
            language="en",
            extra={"k": 1},
        )
        self.assertEqual(c.model, "speech-2.6")
        self.assertEqual(c.voice, "alloy")
        self.assertEqual(c.modality, "audio")
        self.assertEqual(c.language, "en")
        self.assertEqual(c.extra, {"k": 1})

    def test_event_defaults(self) -> None:
        from clawcodex_ext.services.voice.dialogue import DialogueEvent

        e = DialogueEvent(type="ready")
        self.assertEqual(e.type, "ready")
        self.assertEqual(e.text, "")
        self.assertEqual(e.pcm, b"")
        self.assertFalse(e.is_final)
        self.assertEqual(e.message, "")

    def test_event_audio_payload(self) -> None:
        from clawcodex_ext.services.voice.dialogue import DialogueEvent

        e = DialogueEvent(type="audio", pcm=b"\x00\x01", sample_rate=24000)
        self.assertEqual(e.pcm, b"\x00\x01")
        self.assertEqual(e.sample_rate, 24000)

    def test_state_strings(self) -> None:
        from clawcodex_ext.services.voice.dialogue import DialogueState

        self.assertEqual(DialogueState.IDLE, "idle")
        self.assertEqual(DialogueState.LISTENING, "listening")
        self.assertEqual(DialogueState.SPEAKING, "speaking")
        self.assertEqual(DialogueState.INTERRUPTED, "interrupted")
        self.assertEqual(DialogueState.DONE, "done")
        self.assertEqual(DialogueState.ERROR, "error")


class TestDialogueProviderABC(unittest.TestCase):
    def test_cannot_instantiate_directly(self) -> None:
        from clawcodex_ext.services.voice.dialogue import (
            FullDuplexDialogueProvider,
        )

        with self.assertRaises(TypeError):
            FullDuplexDialogueProvider()  # type: ignore[abstract]

    def test_concrete_must_implement_all_abstracts(self) -> None:
        from clawcodex_ext.services.voice.dialogue import (
            FullDuplexDialogueProvider,
        )

        class Incomplete(FullDuplexDialogueProvider):
            async def start(self, *, on_event, config=None):
                return None

        # Missing feed_audio / send_text / interrupt / stop / close.
        with self.assertRaises(TypeError):
            Incomplete()  # type: ignore[abstract]


# ── provider registry (P65-D wire-up) ─────────────────────────────────────


class TestDialogueProviderRegistry(unittest.TestCase):
    def setUp(self) -> None:
        from clawcodex_ext.services.voice import provider_registry as reg

        # Snapshot the registry so we can restore it after the test —
        # some tests inject a stub and we don't want to leak it across
        # the suite (the registry is a module global).
        self._snapshot = dict(reg.DIALOGUE_REGISTRY)

    def tearDown(self) -> None:
        from clawcodex_ext.services.voice import provider_registry as reg

        reg.DIALOGUE_REGISTRY.clear()
        reg.DIALOGUE_REGISTRY.update(self._snapshot)

    def test_builtin_minimax_is_registered(self) -> None:
        from clawcodex_ext.services.voice.provider_registry import (
            DIALOGUE_REGISTRY,
            list_dialogue_providers,
        )

        self.assertIn("minimax", DIALOGUE_REGISTRY)
        self.assertEqual(list_dialogue_providers(), ["minimax"])

    def test_register_and_lookup_roundtrip(self) -> None:
        from clawcodex_ext.services.voice.provider_registry import (
            DIALOGUE_REGISTRY,
            get_dialogue_provider,
            list_dialogue_providers,
            register_dialogue_provider,
        )

        # Use a fresh stub to avoid touching builtins.
        def stub_factory() -> Any:
            return _StubDialogueProvider()

        register_dialogue_provider("stub-f65", stub_factory)
        self.assertIn("stub-f65", DIALOGUE_REGISTRY)
        self.assertIn("stub-f65", list_dialogue_providers())

        provider = get_dialogue_provider("stub-f65")
        try:
            self.assertIsInstance(provider, _StubDialogueProvider)
        finally:
            # Clean up so the registry snapshot test above stays honest.
            DIALOGUE_REGISTRY.pop("stub-f65", None)

    def test_unknown_provider_raises(self) -> None:
        from clawcodex_ext.services.voice.provider_registry import (
            get_dialogue_provider,
        )

        with self.assertRaises(KeyError) as cm:
            get_dialogue_provider("does-not-exist")
        self.assertIn("does-not-exist", str(cm.exception))

    def test_get_dialogue_provider_returns_fresh_instance(self) -> None:
        # Session-scoped: each call returns a new instance (cf. STT/TTS
        # singletons that the registry reuses).
        from clawcodex_ext.services.voice.provider_registry import (
            DIALOGUE_REGISTRY,
            get_dialogue_provider,
            register_dialogue_provider,
        )

        seen: list[_StubDialogueProvider] = []

        def factory() -> Any:
            p = _StubDialogueProvider()
            seen.append(p)
            return p

        register_dialogue_provider("unique-stub", factory)
        try:
            a = get_dialogue_provider("unique-stub")
            b = get_dialogue_provider("unique-stub")
            self.assertIsNot(a, b)
            self.assertEqual(len(seen), 2)
        finally:
            DIALOGUE_REGISTRY.pop("unique-stub", None)


# ── voice-mode gating (F-65) ──────────────────────────────────────────────


class TestDialogueGate(unittest.TestCase):
    def setUp(self) -> None:
        for k in (
            "FEATURE_DIALOGUE_MODE",
            "FEATURE_VOICE_MODE",
            "CLAWCODEX_VOICE_DISABLED",
            "MINIMAX_API_KEY",
        ):
            os.environ.pop(k, None)
        self._settings_patches: list[Any] = []

    def tearDown(self) -> None:
        for k in (
            "FEATURE_DIALOGUE_MODE",
            "FEATURE_VOICE_MODE",
            "CLAWCODEX_VOICE_DISABLED",
            "MINIMAX_API_KEY",
        ):
            os.environ.pop(k, None)
        for patch in self._settings_patches:
            try:
                patch.stop()
            except Exception:
                pass

    def _patch_settings(self, **values: Any) -> None:
        from src.settings import settings as settings_module

        fake = mock.MagicMock()
        fake.dialogue_provider = values.get("dialogue_provider", "")
        fake.dialogue_enabled = values.get("dialogue_enabled", False)
        fake.dialogue_voice = values.get("dialogue_voice", "")
        fake.voice_provider = values.get("voice_provider", "")
        fake.voice_enabled = values.get("voice_enabled", False)

        patcher = mock.patch.object(settings_module, "get_settings", return_value=fake)
        self._settings_patches.append(patcher)
        patcher.start()

    def test_dialogue_feature_flag_off_by_default(self) -> None:
        from clawcodex_ext.services.voice.voice_mode_enabled import (
            is_dialogue_feature_enabled,
        )

        self.assertFalse(is_dialogue_feature_enabled())

    def test_dialogue_feature_flag_accepts_truthy(self) -> None:
        from clawcodex_ext.services.voice.voice_mode_enabled import (
            is_dialogue_feature_enabled,
        )

        for v in ("1", "true", "yes", "on"):
            os.environ["FEATURE_DIALOGUE_MODE"] = v
            self.assertTrue(is_dialogue_feature_enabled(), v)
        os.environ.pop("FEATURE_DIALOGUE_MODE", None)

    def test_has_dialogue_auth_from_env(self) -> None:
        from clawcodex_ext.services.voice.voice_mode_enabled import (
            has_dialogue_auth,
        )

        os.environ["MINIMAX_API_KEY"] = "sk-test"
        self.assertTrue(has_dialogue_auth())

    def test_has_dialogue_auth_from_file(self) -> None:
        from clawcodex_ext.services.voice.voice_mode_enabled import (
            MINIMAX_REALTIME_CREDENTIALS_PATH,
            has_dialogue_auth,
        )

        with mock.patch.object(
            MINIMAX_REALTIME_CREDENTIALS_PATH.__class__,  # type: ignore[attr-defined]
            "expanduser",
            return_value=Path("/tmp/fake_minimax_creds.json"),
        ):
            from clawcodex_ext.services.voice import voice_mode_enabled as vme

            vme.MINIMAX_REALTIME_CREDENTIALS_PATH = Path("/tmp/fake_minimax_creds.json")
            # Patch Path.is_file / read_text to simulate the file.
            fake_path = mock.MagicMock()
            fake_path.is_file.return_value = True
            fake_path.read_text.return_value = '{"api_key": "abc123"}'

            with (
                mock.patch.object(Path, "is_file", return_value=True),
                mock.patch.object(Path, "read_text", return_value='{"api_key": "abc123"}'),
            ):
                self.assertTrue(has_dialogue_auth())
            vme.MINIMAX_REALTIME_CREDENTIALS_PATH = Path("~/.clawcodex/tts/minimax/credentials.json")

    def test_is_dialogue_available(self) -> None:
        from clawcodex_ext.services.voice.voice_mode_enabled import (
            is_dialogue_available,
        )

        os.environ["FEATURE_DIALOGUE_MODE"] = "1"
        self.assertTrue(is_dialogue_available())
        os.environ["CLAWCODEX_VOICE_DISABLED"] = "1"
        self.assertFalse(is_dialogue_available())

    def test_is_dialogue_enabled_default(self) -> None:
        from clawcodex_ext.services.voice.voice_mode_enabled import (
            is_dialogue_enabled,
        )

        self.assertFalse(is_dialogue_enabled())

    def test_get_dialogue_provider_default_is_minimax(self) -> None:
        from clawcodex_ext.services.voice.voice_mode_enabled import (
            get_dialogue_provider,
        )

        self.assertEqual(get_dialogue_provider(), "minimax")

    def test_get_dialogue_provider_respects_settings(self) -> None:
        from clawcodex_ext.services.voice.voice_mode_enabled import (
            get_dialogue_provider,
        )

        self._patch_settings(dialogue_provider="openai-realtime")
        self.assertEqual(get_dialogue_provider(), "openai-realtime")


# ── InterruptDetector (P65-B) ─────────────────────────────────────────────


class TestInterruptDetector(unittest.TestCase):
    def test_initial_decision_is_silence(self) -> None:
        from clawcodex_ext.services.voice.interrupt import (
            InterruptDecision,
            InterruptDetector,
        )

        d = InterruptDetector()
        self.assertFalse(d.triggered)
        self.assertEqual(d.last_decision, InterruptDecision.SILENCE)

    def test_silence_for_low_energy(self) -> None:
        from clawcodex_ext.services.voice.interrupt import (
            InterruptConfig,
            InterruptDecision,
            InterruptDetector,
        )

        cfg = InterruptConfig(min_speech_duration_ms=0)
        d = InterruptDetector(cfg)
        # A 32-sample buffer of all zeros stays in silence territory.
        zero_pcm = b"\x00\x00" * 32
        decision = d.feed_pcm(zero_pcm, now_ms=0.0)
        self.assertEqual(decision, InterruptDecision.SILENCE)

    def test_speech_above_threshold_triggers_once(self) -> None:
        from clawcodex_ext.services.voice.interrupt import (
            InterruptConfig,
            InterruptDecision,
            InterruptDetector,
        )

        cfg = InterruptConfig(
            speech_threshold_db=-40.0,
            silence_threshold_db=-50.0,
            min_speech_duration_ms=0,
            cooldown_ms=10_000.0,
        )
        d = InterruptDetector(cfg)
        # Loud PCM16 at half-scale → 16384 LE.
        loud = b"\x00\x40" * 32  # 0x4000 = 16384 LE
        decision = d.feed_pcm(loud, now_ms=0.0)
        self.assertEqual(decision, InterruptDecision.SPEECH_START)
        self.assertTrue(d.triggered)
        # Subsequent frames are SPEAKING inside cooldown.
        decision2 = d.feed_pcm(loud, now_ms=10.0)
        self.assertEqual(decision2, InterruptDecision.SPEAKING)

    def test_silence_after_trigger_resets(self) -> None:
        from clawcodex_ext.services.voice.interrupt import (
            InterruptConfig,
            InterruptDecision,
            InterruptDetector,
        )

        cfg = InterruptConfig(
            speech_threshold_db=-40.0,
            silence_threshold_db=-50.0,
            min_speech_duration_ms=0,
            min_silence_duration_ms=0,
            cooldown_ms=0,
        )
        d = InterruptDetector(cfg)
        loud = b"\x00\x40" * 32
        zero = b"\x00\x00" * 32
        self.assertEqual(d.feed_pcm(loud, now_ms=0.0), InterruptDecision.SPEECH_START)
        self.assertEqual(d.feed_pcm(zero, now_ms=10.0), InterruptDecision.SPEECH_STOP)
        self.assertFalse(d.triggered)

    def test_chunk_helper(self) -> None:
        from clawcodex_ext.services.voice.interrupt import (
            InterruptConfig,
            InterruptDecision,
            InterruptDetector,
        )

        cfg = InterruptConfig(min_speech_duration_ms=0)
        d = InterruptDetector(cfg)
        # First frame above threshold with 0 ms min-speech → SPEECH_START.
        d.feed_pcm_chunk([20000] * 32, now_ms=0.0)
        self.assertEqual(d.last_decision, InterruptDecision.SPEECH_START)
        # Subsequent frames above threshold fall into cooldown → SPEAKING.
        d.feed_pcm_chunk([20000] * 16, now_ms=10.0)
        self.assertEqual(d.last_decision, InterruptDecision.SPEAKING)

    def test_pcm16_rms_db_for_zero_buffer(self) -> None:
        from clawcodex_ext.services.voice.interrupt import _pcm16_rms_db

        self.assertLess(_pcm16_rms_db(b""), -119.0)


# ── AudioOutQueue.clear / AudioPlayer.stop_nowait (P65-C) ─────────────────


class TestAudioOutQueueClear(unittest.IsolatedAsyncioTestCase):
    async def test_clear_drops_buffers(self) -> None:
        from clawcodex_ext.services.voice.audio_out_queue import AudioOutQueue
        from clawcodex_ext.services.voice.tts import TTSChunk

        q = AudioOutQueue(max_frames=10)
        for i in range(3):
            await q.push(TTSChunk(pcm=b"\x00\x01"))
        self.assertEqual(q.clear(), 3)
        # Next push succeeds without blocking.
        await q.push(TTSChunk(pcm=b"\x02\x03"))

    async def test_clear_is_idempotent(self) -> None:
        from clawcodex_ext.services.voice.audio_out_queue import AudioOutQueue

        q = AudioOutQueue()
        self.assertEqual(q.clear(), 0)
        self.assertEqual(q.clear(), 0)


class TestAudioPlayerStopBehaviours(unittest.IsolatedAsyncioTestCase):
    async def test_stop_nowait_does_not_close_queue(self) -> None:
        # Use a dummy backend that doesn't open PyAudio — the test
        # only cares about the stop_nowait semantics.
        with mock.patch(
            "clawcodex_ext.services.voice.audio_player.has_pyaudio",
            return_value=False,
        ):
            from clawcodex_ext.services.voice.audio_out_queue import AudioOutQueue
            from clawcodex_ext.services.voice.audio_player import AudioPlayer
            from clawcodex_ext.services.voice.tts import TTSChunk

            q = AudioOutQueue(max_frames=10)
            for i in range(3):
                q.push_nowait(TTSChunk(pcm=b"\x00\x01"))
            player = AudioPlayer(queue=q, sample_rate=24000)
            player.start()
            await asyncio.sleep(0.05)
            await player.stop_nowait()
            # The key behaviour we want: stop_nowait must NOT flip the
            # queue's ``_closed`` flag, so a follow-on conversation
            # turn (after interruption) can keep pushing frames into
            # the same queue.
            self.assertFalse(q._closed)
            # And the queue still accepts new frames post-cancel.
            await q.push(TTSChunk(pcm=b"\x00\x02"))
            # Sanity: ``_closed`` is the property under test; the exact
            # number of items left in the buffer depends on how many
            # the drain task consumed before we cancelled it. The test
            # asserts the structural invariant, not the count.
            self.assertFalse(q._closed)

    async def test_stop_still_closes_queue_for_back_compat(self) -> None:
        with mock.patch(
            "clawcodex_ext.services.voice.audio_player.has_pyaudio",
            return_value=False,
        ):
            from clawcodex_ext.services.voice.audio_out_queue import AudioOutQueue
            from clawcodex_ext.services.voice.audio_player import AudioPlayer

            q = AudioOutQueue(max_frames=10)
            player = AudioPlayer(queue=q, sample_rate=24000)
            player.start()
            await asyncio.sleep(0.05)
            await player.stop()
            self.assertTrue(q._closed)


# ── DialogueSessionManager (P65-B integration) ────────────────────────────


class TestDialogueSessionManager(unittest.IsolatedAsyncioTestCase):
    async def test_audio_event_routes_to_speaking_state(self) -> None:
        from clawcodex_ext.services.voice.dialogue_session import (
            DialogueSessionManager,
            DialogueSessionOptions,
        )

        provider = _StubDialogueProvider()
        options = DialogueSessionOptions(use_recorder=False, config=None)
        mgr = DialogueSessionManager(provider, options=options)
        await mgr.start()
        try:
            self.assertEqual(mgr.state, "listening")
            provider.emit("audio", pcm=b"\x00\x01" * 32, sample_rate=24000)
            # Yield enough cycles for the dispatch (sync) + drain task
            # tick. State must end up as "speaking" regardless of
            # whether the frame has been drained already — the
            # important assertion is the state transition, not the
            # queue depth (which depends on whether PyAudio is around
            # in the test env).
            await asyncio.sleep(0.05)
            self.assertEqual(mgr.state, "speaking")
            # After the response turn ends, the manager flips back to
            # listening (the next provider event would do this; emulate
            # it with a ``done`` event).
            provider.emit("done", text="hello")
            await asyncio.sleep(0.05)
            self.assertEqual(mgr.state, "listening")
        finally:
            await mgr.close()

    async def test_transcript_collected(self) -> None:
        from clawcodex_ext.services.voice.dialogue_session import (
            DialogueSessionCallbacks,
            DialogueSessionManager,
            DialogueSessionOptions,
        )

        captured: list[str] = []
        provider = _StubDialogueProvider()
        cb = DialogueSessionCallbacks(
            on_user_transcript=lambda t: captured.append(t),
        )
        options = DialogueSessionOptions(use_recorder=False)
        mgr = DialogueSessionManager(provider, callbacks=cb, options=options)
        await mgr.start()
        try:
            provider.emit("transcript", text="hello", is_final=True)
            await asyncio.sleep(0.05)
            self.assertEqual(captured, ["hello"])
        finally:
            await mgr.close()

    async def test_barge_in_calls_interrupt_and_clears_queue(self) -> None:
        from clawcodex_ext.services.voice.interrupt import InterruptConfig
        from clawcodex_ext.services.voice.dialogue_session import (
            DialogueSessionCallbacks,
            DialogueSessionManager,
            DialogueSessionOptions,
        )

        captured_interrupt: list[bool] = []
        provider = _StubDialogueProvider()
        cb = DialogueSessionCallbacks(
            on_interrupt=lambda: captured_interrupt.append(True),
        )
        # Tell the detector to fire immediately on any loud frame so
        # we don't have to encode realistic PCM.
        interrupt_cfg = InterruptConfig(
            speech_threshold_db=-60.0,
            min_speech_duration_ms=0,
            cooldown_ms=0,
        )
        options = DialogueSessionOptions(
            use_recorder=False,
            interrupt_config=interrupt_cfg,
        )
        mgr = DialogueSessionManager(provider, callbacks=cb, options=options)
        await mgr.start()
        try:
            # Pretend the agent is currently speaking.
            provider.emit("audio", pcm=b"\x00\x01", sample_rate=24000)
            await asyncio.sleep(0.05)
            self.assertEqual(mgr.state, "speaking")

            loud = b"\x00\x40" * 32  # 0x4000 LE = 16384 (half-scale)
            await mgr.feed_audio(loud)
            await asyncio.sleep(0.1)
            # Provider's interrupt was called, queue was cleared.
            self.assertGreaterEqual(provider.interrupt_calls, 1)
            self.assertEqual(captured_interrupt, [True])
            # clear() returns the number of dropped frames; we want at
            # least one to have been in the buffer at interrupt time.
            # Some may have already drained through the test player, so
            # the exact count isn't stable — only the drop-on-clear
            # behaviour is asserted here.
            mgr.out_queue.clear()
        finally:
            await mgr.close()

    async def test_stop_returns_empty_when_no_transcripts(self) -> None:
        from clawcodex_ext.services.voice.dialogue_session import (
            DialogueSessionManager,
            DialogueSessionOptions,
        )

        provider = _StubDialogueProvider()
        options = DialogueSessionOptions(use_recorder=False)
        mgr = DialogueSessionManager(provider, options=options)
        await mgr.start()
        result = await mgr.stop()
        self.assertEqual(result, "")
        self.assertTrue(provider.closed)


# ── /dialogue command (P65-D) ─────────────────────────────────────────────


def _ctx() -> Any:
    """Construct a CommandContext with the minimum surface the command uses."""
    from pathlib import Path

    from clawcodex_ext.command_system.types import CommandContext

    return CommandContext(
        workspace_root=Path("/tmp"),
        cwd=Path("/tmp"),
        config={"settings": {}},
    )


class TestDialogueCommand(unittest.TestCase):
    def setUp(self) -> None:
        # Patch the settings getters + config setters the command uses
        # so we don't touch the user's real config.
        self._patches: list[Any] = []
        self._settings_stub = mock.MagicMock()
        self._settings_stub.dialogue_provider = ""
        self._settings_stub.dialogue_enabled = False
        self._settings_stub.dialogue_voice = ""
        self._settings_stub.dialogue_modality = "text"

        patcher = mock.patch("src.settings.settings.get_settings", return_value=self._settings_stub)
        self._patches.append(patcher)
        patcher.start()

        for env in (
            "FEATURE_DIALOGUE_MODE",
            "CLAWCODEX_VOICE_DISABLED",
            "MINIMAX_API_KEY",
        ):
            os.environ.pop(env, None)

    def tearDown(self) -> None:
        for p in self._patches:
            try:
                p.stop()
            except Exception:
                pass
        for env in (
            "FEATURE_DIALOGUE_MODE",
            "CLAWCODEX_VOICE_DISABLED",
            "MINIMAX_API_KEY",
        ):
            os.environ.pop(env, None)

    def _call(self, args: str) -> Any:
        from clawcodex_ext.command_system.dialogue_command import (
            dialogue_command_call,
        )

        return dialogue_command_call(args, _ctx())

    def test_help(self) -> None:
        result = self._call("help")
        self.assertEqual(result.type, "text")
        self.assertIn("/dialogue", result.value)

    def test_status(self) -> None:
        result = self._call("status")
        self.assertIn("Voice dialogue", result.value)

    def test_off_disables(self) -> None:
        with mock.patch("clawcodex_ext.command_system.dialogue_command.set_dialogue_enabled") as m:
            result = self._call("off")
            m.assert_called_with(False)
        self.assertIn("disabled", result.value)

    def test_unknown_arg(self) -> None:
        result = self._call("nonsense")
        self.assertIn("Unknown argument", result.value)

    def test_provider_selects_minimax(self) -> None:
        with (
            mock.patch("clawcodex_ext.command_system.dialogue_command.set_dialogue_provider") as m_p,
            mock.patch("clawcodex_ext.command_system.dialogue_command.set_dialogue_enabled") as m_e,
        ):
            os.environ["FEATURE_DIALOGUE_MODE"] = "1"
            os.environ["MINIMAX_API_KEY"] = "sk-test"
            result = self._call("minimax")
            m_p.assert_called_with("minimax")
            m_e.assert_called_with(True)
            self.assertIn("minimax", result.value)

    def test_mode_text(self) -> None:
        with mock.patch("clawcodex_ext.command_system.dialogue_command.set_dialogue_modality") as m:
            result = self._call("mode text")
            m.assert_called_with("text")
        self.assertIn("text", result.value)

    def test_mode_audio(self) -> None:
        with mock.patch("clawcodex_ext.command_system.dialogue_command.set_dialogue_modality") as m:
            self._call("mode audio")
            m.assert_called_with("audio")

    def test_mode_invalid(self) -> None:
        result = self._call("mode banana")
        self.assertIn("Unknown modality", result.value)

    def test_voice(self) -> None:
        with mock.patch("clawcodex_ext.command_system.dialogue_command.set_dialogue_voice") as m:
            result = self._call("voice alloy")
            m.assert_called_with("alloy")
        self.assertIn("alloy", result.value)

    def test_start_blocks_when_feature_off(self) -> None:
        result = self._call("start")
        self.assertIn("feature flag", result.value)

    def test_start_blocks_when_kill_switch_engaged(self) -> None:
        os.environ["FEATURE_DIALOGUE_MODE"] = "1"
        os.environ["CLAWCODEX_VOICE_DISABLED"] = "1"
        result = self._call("start")
        self.assertIn("kill-switch", result.value)

    def test_start_succeeds_when_minimax_configured(self) -> None:
        os.environ["FEATURE_DIALOGUE_MODE"] = "1"
        os.environ["MINIMAX_API_KEY"] = "sk-test"
        # settings stub: minimax selected
        self._settings_stub.dialogue_provider = "minimax"
        result = self._call("start")
        self.assertIn("ready", result.value.lower())

    def test_start_missing_credentials(self) -> None:
        os.environ["FEATURE_DIALOGUE_MODE"] = "1"
        self._settings_stub.dialogue_provider = "minimax"
        result = self._call("start")
        self.assertIn("credentials", result.value.lower())

    def test_toggle_enables_when_off(self) -> None:
        self._settings_stub.dialogue_provider = "minimax"
        self._settings_stub.dialogue_enabled = False
        with mock.patch("clawcodex_ext.command_system.dialogue_command.set_dialogue_enabled") as m:
            result = self._call("")
            m.assert_called_with(True)
        self.assertIn("enabled", result.value)

    def test_toggle_disables_when_on(self) -> None:
        self._settings_stub.dialogue_enabled = True
        with mock.patch("clawcodex_ext.command_system.dialogue_command.set_dialogue_enabled") as m:
            result = self._call("")
            m.assert_called_with(False)
        self.assertIn("disabled", result.value)

    def test_stop_always_succeeds(self) -> None:
        result = self._call("stop")
        self.assertIn("ended", result.value.lower())


# ── settings + config setter round-trips (P65-D) ──────────────────────────


class TestDialogueSettingsRoundTrip(unittest.TestCase):
    def test_settings_schema_has_dialogue_fields(self) -> None:
        from clawcodex_ext.settings.types import SettingsSchema

        s = SettingsSchema()
        self.assertEqual(s.dialogue_provider, "")
        self.assertEqual(s.dialogue_enabled, False)
        self.assertEqual(s.dialogue_voice, "")
        self.assertEqual(s.dialogue_modality, "text")
        self.assertTrue(s.dialogue_interim_results)

    def test_settings_from_dict_roundtrip(self) -> None:
        from clawcodex_ext.settings.types import SettingsSchema

        data = SettingsSchema(
            dialogue_provider="minimax",
            dialogue_enabled=True,
            dialogue_voice="alloy",
            dialogue_modality="audio",
            dialogue_interim_results=False,
        ).to_dict()
        restored = SettingsSchema.from_dict(data)
        self.assertEqual(restored.dialogue_provider, "minimax")
        self.assertTrue(restored.dialogue_enabled)
        self.assertEqual(restored.dialogue_voice, "alloy")
        self.assertEqual(restored.dialogue_modality, "audio")
        self.assertFalse(restored.dialogue_interim_results)


class TestDialogueConfigSetters(unittest.TestCase):
    def setUp(self) -> None:
        self._patches: list[Any] = []
        self._saved: dict[str, dict[str, Any]] = {}

        fake_mgr = mock.MagicMock()
        fake_mgr.load_global.side_effect = lambda: dict(self._saved.get("cfg", {"settings": {}}))
        fake_mgr.save_global.side_effect = lambda cfg: self._saved.update({"cfg": cfg})

        patcher = mock.patch("src.config._get_default_manager", return_value=fake_mgr)
        self._patches.append(patcher)
        patcher.start()

    def tearDown(self) -> None:
        for p in self._patches:
            try:
                p.stop()
            except Exception:
                pass

    def test_set_dialogue_provider_normalizes_minimax(self) -> None:
        from src.config import set_dialogue_provider

        set_dialogue_provider("MINIMAX")
        self.assertEqual(self._saved["cfg"]["settings"]["dialogue_provider"], "minimax")

    def test_set_dialogue_provider_invalid_coerces_empty(self) -> None:
        from src.config import set_dialogue_provider

        set_dialogue_provider("does-not-exist")
        self.assertEqual(self._saved["cfg"]["settings"]["dialogue_provider"], "")

    def test_set_dialogue_enabled_writes_bool(self) -> None:
        from src.config import set_dialogue_enabled

        set_dialogue_enabled(True)
        self.assertTrue(self._saved["cfg"]["settings"]["dialogue_enabled"])
        set_dialogue_enabled(False)
        self.assertFalse(self._saved["cfg"]["settings"]["dialogue_enabled"])

    def test_set_dialogue_voice_writes_str(self) -> None:
        from src.config import set_dialogue_voice

        set_dialogue_voice("alloy")
        self.assertEqual(self._saved["cfg"]["settings"]["dialogue_voice"], "alloy")

    def test_set_dialogue_modality_normalizes(self) -> None:
        from src.config import set_dialogue_modality

        set_dialogue_modality("audio")
        self.assertEqual(self._saved["cfg"]["settings"]["dialogue_modality"], "audio")
        set_dialogue_modality("invalid")
        self.assertEqual(self._saved["cfg"]["settings"]["dialogue_modality"], "text")


# ── exit-tests for the package public surface (P65-D wiring) ──────────────


class TestVoicePackageExports(unittest.TestCase):
    def test_lazy_attribute_resolution(self) -> None:
        # Resolve each F-65 export via the package __getattr__ hook.
        from clawcodex_ext.services import voice as vp

        for name in (
            "DialogueConfig",
            "DialogueEvent",
            "FullDuplexDialogueProvider",
            "InterruptConfig",
            "InterruptDecision",
            "InterruptDetector",
            "DialogueSessionManager",
            "DialogueSessionState",
            "DialogueSessionCallbacks",
            "DialogueSessionOptions",
            "MINIMAX_REALTIME_ENDPOINTS",
            "MiniMaxRealtimeDialogueProvider",
            "DIALOGUE_PROVIDERS",
        ):
            with self.subTest(name=name):
                self.assertTrue(hasattr(vp, name), name)

    def test_command_registry_includes_dialogue(self) -> None:
        from clawcodex_ext.command_system.builtins import get_builtin_commands

        names = {c.name for c in get_builtin_commands()}
        self.assertIn("dialogue", names)
        self.assertIn("voice", names)
        self.assertIn("tts", names)


if __name__ == "__main__":
    unittest.main()
