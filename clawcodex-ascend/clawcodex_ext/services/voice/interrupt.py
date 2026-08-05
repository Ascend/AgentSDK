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

"""Interrupt detector — F-65 P65-B.

Watches an inbound PCM stream for *barge-in*: the user starts speaking
while the agent is mid-reply. Triggered by a sustained rise in audio
energy above a configurable threshold, with a debounce window so a
single noisy frame doesn't kill a fresh reply.

The detector is intentionally simple — energy-based VAD on rolling PCM
windows. The F-64 :class:`VoiceActivityDetector` in :mod:`detection`
already does this for the speech/state machine; rather than wrap that
class (which is session-lifecycle-coupled via ``start`` / ``stop``),
:Class:`InterruptDetector` keeps its own numeric state and exposes
only the one decision the dialogue session manager needs:

> "Did the user just start speaking in a way that should cancel the
> agent reply?"

Design choices
--------------
* **Per-call, not event-streamed** — :meth:`feed_pcm` accepts a single
  PCM frame and returns the current decision (``"silence"`` /
  ``"speech_start"`` / ``"speaking"`` / ``"speech_stop"``). The session
  manager drives it from its existing audio-recording loop; no
  callbacks needed. Keeping the API "call me once per frame" means
  tests don't have to wrestle with async listener plumbing.
* **Energy, not ML** — RMS in dBFS against a configurable threshold,
  averaged over the last ``window_ms`` of frames. Good enough for the
  barge-in trigger (sensitivity tuning lives in
  :class:`InterruptConfig`, no model weights required). A real Silero
  VAD would be a drop-in replacement once we add it.
* **Half-duplex aware** — when the agent isn't speaking (nothing to
  barge into) the detector still works fine; the session manager simply
  ignores its decisions. Skipping the work entirely would couple the
  detector to session state we don't want to know about here.
* **Cooldown after a trigger** — prevents flapping at the speech edge
  if the user's first frame is borderline energy; configurable.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from enum import Enum
from typing import Iterable, Optional


class InterruptDecision(str, Enum):
    """Outcome of one :meth:`InterruptDetector.feed_pcm` call.

    * ``"silence"`` — below threshold, agent should keep speaking.
    * ``"speaking"`` — above threshold but the trigger has already
      fired for this turn (caller has nothing to do).
    * ``"speech_start"`` — first frame of sustained speech *after*
      silence: this is the moment to interrupt.
    * ``"speech_stop"`` — voice energy dropped back below threshold
      long enough to count as a turn end (caller may commit audio).
    """

    SILENCE = "silence"
    SPEAKING = "speaking"
    SPEECH_START = "speech_start"
    SPEECH_STOP = "speech_stop"


@dataclass
class InterruptConfig:
    """Tuning knobs for the barge-in VAD."""

    # dBFS threshold above which a frame counts as speech. -30 dBFS is
    # a reasonable conversational default; lower = more sensitive
    # (catches whispers + noise), higher = stricter (more delay before
    # the trigger fires).
    speech_threshold_db: float = -30.0
    # dBFS threshold below which a frame counts as silence. Separate
    # from the speech threshold to introduce hysteresis — without it
    # a noisy room makes the trigger flap.
    silence_threshold_db: float = -45.0
    # Minimum continuous speech duration before :data:`InterruptDecision
    # SPEECH_START` fires. Suppresses single-frame noise spikes.
    min_speech_duration_ms: float = 200.0
    # Continuous silence required before ``SPEECH_STOP`` fires. Lets
    # the user pause mid-sentence without closing the turn.
    min_silence_duration_ms: float = 600.0
    # PCM sample width in bytes (2 = PCM16). We do RMS in samples not
    # bytes because the math is endian-agnostic.
    sample_width_bytes: int = 2
    # Sample rate of the PCM stream — used to convert frame duration
    # to ms for the min_*_duration checks. Defaults to the dialogue
    # input rate; override if your recorder runs at 48 kHz.
    sample_rate: int = 16000
    # Cooldown after a trigger: for this many ms further ``speech_start``
    # decisions are suppressed (returned as ``SPEAKING``). Prevents the
    # trigger from firing twice in quick succession on a single turn.
    cooldown_ms: float = 500.0


@dataclass
class _InterruptState:
    speech_start_time_ms: Optional[float] = None
    last_silence_time_ms: Optional[float] = None
    triggered: bool = False
    trigger_time_ms: Optional[float] = None


class InterruptDetector:
    """Energy-threshold + debounce VAD for barge-in detection.

    Stateless across sessions: construct one per dialogue session so
    cooldown / hysteresis settings start clean each time. The class is
    deliberately not async — it's called from a sync audio callback
    inside the session manager's recorder loop.

    Usage::

        vad = InterruptDetector(InterruptConfig())
        for pcm_frame in recorder:
            decision = vad.feed_pcm(pcm_frame)
            if decision == InterruptDecision.SPEECH_START and agent_playing:
                await agent.interrupt()

    For tests, :meth:`feed_pcm_chunk` accepts a flat iterable of samples
    and runs the same detection loop across them — no real audio device
    needed.
    """

    def __init__(self, config: Optional[InterruptConfig] = None) -> None:
        self._config = config or InterruptConfig()
        self._state = _InterruptState()
        self._last_decision: InterruptDecision = InterruptDecision.SILENCE

    @property
    def config(self) -> InterruptConfig:
        return self._config

    @property
    def last_decision(self) -> InterruptDecision:
        return self._last_decision

    @property
    def triggered(self) -> bool:
        """True if a barge-in trigger has fired and not yet been reset."""
        return self._state.triggered

    def reset(self) -> None:
        """Clear the trigger state — call at the start of each session."""
        self._state = _InterruptState()
        self._last_decision = InterruptDecision.SILENCE

    def feed_pcm(self, pcm: bytes, *, now_ms: Optional[float] = None) -> InterruptDecision:
        """Process one PCM frame and return the current decision.

        ``pcm`` is mono PCM16 little-endian at ``self._config.sample_rate``.
        ``now_ms`` is injected in tests; defaults to a monotonic clock
        in real use.
        """
        if not pcm:
            return self._last_decision
        level_db = _pcm16_rms_db(pcm)
        return self._evaluate(level_db, now_ms=now_ms)

    def feed_pcm_chunk(self, samples: Iterable[int], *, now_ms: Optional[float] = None) -> InterruptDecision:
        """Same as :meth:`feed_pcm` but takes a flat iterable of int samples.

        Convenience for tests that want to exercise the detector without
        having to encode PCM16 by hand. ``samples`` are signed 16-bit
        values in the canonical [-32768, 32767] range.
        """
        sq_sum = 0
        count = 0
        for s in samples:
            sq_sum += int(s) * int(s)
            count += 1
        if count == 0:
            return self._last_decision
        rms = math.sqrt(sq_sum / count)
        if rms <= 0:
            level_db = -120.0  # treat total silence as effectively zero
        else:
            level_db = 20.0 * math.log10(rms / 32768.0)
        return self._evaluate(level_db, now_ms=now_ms)

    # ── core state machine ────────────────────────────────────────────────

    def _evaluate(self, level_db: float, *, now_ms: Optional[float]) -> InterruptDecision:
        cfg = self._config
        now = now_ms if now_ms is not None else time.time() * 1000.0

        # Cooldown: ignore trigger fires until the cooldown window expires
        # from the last trigger.
        if (
            self._state.triggered
            and self._state.trigger_time_ms is not None
            and (now - self._state.trigger_time_ms) < cfg.cooldown_ms
        ):
            # Still inside cooldown — only the speech/silence binary matters.
            if level_db >= cfg.speech_threshold_db:
                self._last_decision = InterruptDecision.SPEAKING
            else:
                self._last_decision = InterruptDecision.SILENCE
            return self._last_decision

        # 1. Speech above threshold
        if level_db >= cfg.speech_threshold_db:
            if self._state.speech_start_time_ms is None:
                self._state.speech_start_time_ms = now
                self._state.last_silence_time_ms = None

            speech_dur = now - self._state.speech_start_time_ms
            if speech_dur >= cfg.min_speech_duration_ms and not self._state.triggered:
                # First sustained speech after silence → Barge-in!
                self._state.triggered = True
                self._state.trigger_time_ms = now
                self._last_decision = InterruptDecision.SPEECH_START
                return self._last_decision
            self._last_decision = InterruptDecision.SPEAKING
            return self._last_decision

        # 2. Below speech threshold. Update silence bookkeeping.
        if self._state.last_silence_time_ms is None:
            self._state.last_silence_time_ms = now
        silence_dur = now - self._state.last_silence_time_ms

        # End of speech if we had speech that has now been silent long enough.
        if silence_dur >= cfg.min_silence_duration_ms and self._state.speech_start_time_ms is not None:
            self._state.speech_start_time_ms = None
            self._state.triggered = False  # reset trigger for next turn
            self._last_decision = InterruptDecision.SPEECH_STOP
            return self._last_decision

        self._last_decision = InterruptDecision.SILENCE
        return self._last_decision


def _pcm16_rms_db(pcm: bytes) -> float:
    """Return dBFS (RMS) of a PCM16 little-endian buffer.

    Empty input returns ``-120 dBFS`` (effectively muted) so callers
    can compare against the silence threshold without a special case.
    """
    if not pcm:
        return -120.0
    sw = 2  # PCM16
    if len(pcm) % sw != 0:
        # Truncate defensively; some devices emit trailing padding.
        pcm = pcm[: -(len(pcm) % sw)]
    if not pcm:
        return -120.0
    total = 0
    count = 0
    # Use ``memoryview`` slicing to avoid a numpy dependency for an
    # 8-line routine. ~9k samples/ms is fine for the barge-in path.
    mv = memoryview(pcm)
    for i in range(0, len(mv) - sw + 1, sw):
        # Little-endian signed 16.
        sample = int.from_bytes(mv[i : i + sw], byteorder="little", signed=True)
        total += sample * sample
        count += 1
    if count == 0:
        return -120.0
    rms = math.sqrt(total / count)
    if rms <= 0:
        return -120.0
    # Full-scale PCM16 = 32768. dBFS = 20·log10(rms / 32768).
    return 20.0 * math.log10(rms / 32768.0)
