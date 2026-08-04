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

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from datetime import datetime, timezone
from typing import Any

from clawcodex_ext.query.outbox_types import ProactivePromptEvent  # pylint: disable=no-name-in-module
from clawcodex_ext.services.kairos import TickConfig, TickEvent, TickScheduler

from .constants import DEFAULT_JITTER_FRACTION, TICK_INTERVAL_MS, TICK_TAG
from .controller import ProactiveController, get_default_controller

logger = logging.getLogger(__name__)

PromptSink = Callable[[str], None]
SkipPredicate = Callable[[], bool]
CompactPredicate = Callable[[], bool]
CompactCallback = Callable[[], None]


class TickEmitter:
    def __init__(
        self,
        *,
        controller: ProactiveController | None = None,
        outbox: list[Any] | None = None,
        prompt_sink: PromptSink | None = None,
        interval_ms: int = TICK_INTERVAL_MS,
        jitter_fraction: float = DEFAULT_JITTER_FRACTION,
        scheduler: TickScheduler | None = None,
        should_skip: SkipPredicate | None = None,
        should_compact_first: CompactPredicate | None = None,
        compact_callback: CompactCallback | None = None,
        start: bool = False,
    ) -> None:
        self._ctrl = controller or get_default_controller()
        self._outbox = outbox
        self._prompt_sink = prompt_sink
        self._should_skip = should_skip
        self._should_compact_first = should_compact_first
        self._compact_callback = compact_callback
        self._scheduler = scheduler or TickScheduler(
            TickConfig(
                id="proactive",
                interval_seconds=interval_ms / 1000,
                enabled=False,
                jitter_fraction=jitter_fraction,
                name="Proactive tick emitter",
            )
        )
        self._scheduler.subscribe(self._on_tick_event)
        if start:
            self.start()

    @property
    def scheduler(self) -> TickScheduler:
        return self._scheduler

    def start(self) -> bool:
        started = self._scheduler.start()
        self._ctrl.set_next_tick_at((time.time() * 1000) + self._scheduler.config.interval_seconds * 1000)
        return started

    def stop(self) -> bool:
        self._ctrl.set_next_tick_at(None)
        return self._scheduler.stop()

    def pause(self) -> None:
        self._scheduler.pause()
        self._ctrl.pause()

    def resume(self) -> None:
        self._ctrl.resume()
        self._scheduler.resume()

    def emit_now(self) -> str | None:
        event = TickEvent(
            scheduler_id="proactive-manual",
            tick_number=self._scheduler.tick_count + 1,
            scheduled_at=time.time(),
            actual_at=time.time(),
        )
        return self._on_tick_event(event)

    def _on_tick_event(self, event: TickEvent) -> str | None:
        if not self._ctrl.should_tick():
            self._reschedule()
            return None
        if self._should_skip is not None and self._should_skip():
            self._reschedule()
            return None
        if self._should_compact_first is not None and self._should_compact_first():
            try:
                if self._compact_callback is not None:
                    self._compact_callback()
            except Exception:
                logger.exception("proactive pre-tick compaction failed")
                self._ctrl.set_context_blocked(True)
                self._reschedule()
                return None

        now = datetime.now(tz=timezone.utc).strftime("%H:%M:%S")
        tick_text = f"<{TICK_TAG}>{now}</{TICK_TAG}>"
        self._deliver(tick_text)
        self._ctrl.record_tick(at_ms=event.actual_at * 1000, summary=f"Injected {tick_text}")
        self._reschedule()
        return tick_text

    def _deliver(self, tick_text: str) -> None:
        if self._prompt_sink is not None:
            self._prompt_sink(tick_text)
            return
        if self._outbox is not None:
            self._outbox.append(ProactivePromptEvent(prompt=tick_text, source="tick"))

    def _reschedule(self) -> None:
        self._ctrl.set_next_tick_at((time.time() * 1000) + self._scheduler.config.interval_seconds * 1000)
