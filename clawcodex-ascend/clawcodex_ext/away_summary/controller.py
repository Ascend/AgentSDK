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

"""Idle controller for automatic Away Summary generation."""

from __future__ import annotations

import logging
import threading
from typing import Any, Callable, Protocol

from clawcodex_ext.away_summary.config import (  # pylint: disable=no-name-in-module
    AwaySummaryConfig,
    load_away_summary_config,
)
from clawcodex_ext.away_summary.fingerprint import (  # pylint: disable=no-name-in-module
    conversation_fingerprint,
    last_away_summary_fingerprint,
    session_turn_count,
)
from clawcodex_ext.away_summary.messages import (  # pylint: disable=no-name-in-module
    format_away_summary_for_display,
)
from clawcodex_ext.away_summary.service import AwaySummaryService  # pylint: disable=no-name-in-module

from .memory import get_session_memory_content

logger = logging.getLogger(__name__)


class TimerHandle(Protocol):
    def cancel(self) -> None: ...


class TimerFactory(Protocol):
    def call_later(self, seconds: float, callback: Callable[[], None]) -> TimerHandle: ...


class ThreadingTimerFactory:
    def call_later(self, seconds: float, callback: Callable[[], None]) -> TimerHandle:
        timer = threading.Timer(seconds, callback)
        timer.daemon = True
        timer.start()
        return timer


class AwaySummaryController:
    # Only a new run resets the timer; ``"user"`` preserves the legacy path.
    _USER_INTERACTION_RESET_REASONS = frozenset({"submit", "new_prompt", "user"})

    def __init__(
        self,
        *,
        conversation: Any,
        provider_getter: Callable[[], Any],
        model_getter: Callable[[], str | None],
        session_getter: Callable[[], Any | None],
        display: Callable[[str], None] | None = None,
        config_loader: Callable[[], AwaySummaryConfig] = load_away_summary_config,
        timer_factory: TimerFactory | None = None,
        interactive: bool = True,
        memory_getter: Callable[[], str | None] | None = None,
    ) -> None:
        self.conversation = conversation
        self.provider_getter = provider_getter
        self.model_getter = model_getter
        self.session_getter = session_getter
        self.display = display
        self.config_loader = config_loader
        self.timer_factory = timer_factory or ThreadingTimerFactory()
        self.interactive = interactive
        self.memory_getter = memory_getter
        self._lock = threading.RLock()
        self._timer: TimerHandle | None = None
        self._busy = False
        self._run_completed = False
        self._armed_fingerprint: str | None = None
        self._running = False
        self._generation = 0
        self._armed_generation: int | None = None
        self._closed = False

    def on_user_interaction(self, reason: str = "user") -> None:
        """Cancel the idle timer only when a new agent run starts."""
        if reason not in self._USER_INTERACTION_RESET_REASONS:
            return
        with self._lock:
            self._cancel_locked()

    def on_run_start(self) -> None:
        with self._lock:
            self._busy = True
            self._run_completed = False
            self._cancel_locked()

    def on_run_finish(self) -> None:
        with self._lock:
            self._busy = False
            self._run_completed = True

    def on_assistant_turn_complete(self) -> None:
        with self._lock:
            # Ignore intermediate turn-complete callbacks.
            if not self._run_completed:
                return
            self._run_completed = False
            self._busy = False
            if self._closed or not self.interactive:
                return
            cfg = self.config_loader()
            if not cfg.enabled:
                return
            if session_turn_count(self.conversation) < cfg.min_turns:
                return
            fingerprint = conversation_fingerprint(self.conversation)
            if fingerprint == last_away_summary_fingerprint(self.conversation):
                return
            self._cancel_locked()
            self._armed_fingerprint = fingerprint
            self._armed_generation = self._generation
            self._timer = self.timer_factory.call_later(
                cfg.idle_seconds,
                self._on_idle_timer,
            )

    def close(self) -> None:
        with self._lock:
            self._closed = True
            self._cancel_locked()

    def _on_idle_timer(self) -> None:
        with self._lock:
            if self._closed or self._busy or self._running:
                return
            cfg = self.config_loader()
            if not self.interactive or not cfg.enabled:
                return
            fingerprint = conversation_fingerprint(self.conversation)
            if fingerprint != self._armed_fingerprint:
                return
            if fingerprint == last_away_summary_fingerprint(self.conversation):
                return
            generation = self._armed_generation
            if generation is None:
                return
            self._running = True
            self._timer = None

        session: Any | None = None
        try:
            session = self.session_getter()
            self._generate_summary(
                cfg,
                session,
                generation=generation,
                fingerprint=fingerprint,
            )
        except Exception:
            logger.exception(
                "Away Summary failed: trigger=auto fingerprint=%s",
                self._armed_fingerprint,
            )
        finally:
            with self._lock:
                self._running = False
                if self._armed_generation == generation:
                    self._armed_fingerprint = None
                    self._armed_generation = None

    def _generate_summary(
        self,
        cfg: AwaySummaryConfig,
        session: Any | None,
        *,
        generation: int,
        fingerprint: str,
    ) -> None:
        memory = self._load_memory(session) if cfg.include_session_memory else None
        service = AwaySummaryService(
            conversation=self.conversation,
            provider=self.provider_getter(),
            model=self.model_getter(),
            session=session,
            config=cfg,
            memory=memory,
        )
        result = service.generate(trigger="auto", persist=False)
        if not result.generated:
            return
        with self._lock:
            if not self._idle_snapshot_is_current(generation, fingerprint):
                logger.info("Away Summary discarded because the idle snapshot changed")
                return
            service.persist_result(result, trigger="auto")
        if self.display is not None:
            self.display(format_away_summary_for_display(result.summary))

    def _idle_snapshot_is_current(
        self,
        generation: int,
        fingerprint: str,
    ) -> bool:
        if self._closed or self._busy:
            return False
        if generation != self._generation:
            return False
        if fingerprint != self._armed_fingerprint:
            return False
        if conversation_fingerprint(self.conversation) != fingerprint:
            return False
        return last_away_summary_fingerprint(self.conversation) != fingerprint

    def _load_memory(self, session: Any | None) -> str | None:
        try:
            if self.memory_getter is not None:
                return self.memory_getter()
            sid = getattr(session, "session_id", None) if session is not None else None
            return get_session_memory_content(session_id=sid)
        except Exception:
            logger.debug("Away Summary: memory_getter raised, continuing without memory", exc_info=True)
            return None

    def _cancel_locked(self) -> None:
        self._generation += 1
        if self._timer is not None:
            try:
                self._timer.cancel()
            except Exception:
                logger.debug("Away Summary timer cancellation failed", exc_info=True)
        self._timer = None
        self._armed_fingerprint = None
        self._armed_generation = None
