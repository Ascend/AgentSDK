#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# -------------------------------------------------------------------------
# This file is part of the AgentSDK project.
# Copyright (c) 2026 Clawd Codex Team
# Copyright (c) 2026 Huawei Technologies Co.,Ltd.
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

"""Idle controller for automatic Intent Forecast generation."""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Protocol

from clawcodex_ext.intent_forecast.config import IntentForecastConfig, load_intent_forecast_config
from clawcodex_ext.intent_forecast.learning import (
    build_feedback_features,
    looks_like_correction,
    looks_like_followup,
    record_feedback,
)
from clawcodex_ext.intent_forecast.messages import ForecastResult
from clawcodex_ext.intent_forecast.persistence import save_forecast_result
from clawcodex_ext.intent_forecast.service import IntentForecastService

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


_AUTO_RETRY_MAX = 3
_AUTO_RETRY_BASE_SECONDS = 30


@dataclass
class IntentForecastController:
    provider_getter: Callable[[], Any]
    model_getter: Callable[[], str | None]
    session_getter: Callable[[], Any | None]
    workspace_root: Path
    display: Callable[[ForecastResult], None]
    submit: Callable[[str], None] | None = None
    config_loader: Callable[[], IntentForecastConfig] = load_intent_forecast_config
    conversation_getter: Callable[[], Any | None] | None = None
    timer_factory: TimerFactory | None = None
    interactive: bool = True

    def __post_init__(self) -> None:
        self.timer_factory = self.timer_factory or ThreadingTimerFactory()
        self._lock = threading.RLock()
        self._timer: TimerHandle | None = None
        self._busy = False
        self._running = False
        self._generation_id = 0
        self._last_result: ForecastResult | None = None
        self._shown_fingerprints: set[str] = set()
        self._pending_acceptance: dict[str, Any] | None = None
        self._prompt_draft_text = ""
        self._valid_input_seen = False
        self._auto_retry_count = 0

    @property
    def last_result(self) -> ForecastResult | None:
        return self._last_result

    def remember(self, result: ForecastResult) -> None:
        self._last_result = result

    def on_mount(self) -> None:
        with self._lock:
            self._arm_locked()

    def on_user_interaction(self, reason: str = "user") -> None:
        with self._lock:
            self._generation_id += 1
            self._cancel_locked()

    def on_prompt_draft_changed(self, text: str) -> None:
        self._prompt_draft_text = text
        if text:
            self._maybe_record_user_acceptance_outcome(text)
            self.on_user_interaction("draft")
        else:
            with self._lock:
                self._arm_locked()

    def on_run_start(self) -> None:
        with self._lock:
            self._valid_input_seen = True
            self._busy = True
            self._generation_id += 1
            self._cancel_locked()

    def on_run_finish(self) -> None:
        self._record_acceptance_outcome("accepted_completed")
        with self._lock:
            self._busy = False
            self._arm_locked()

    def accept(self, selection: int | str = 1) -> bool:
        from clawcodex_ext.intent_forecast.messages import parse_selection

        result = self._last_result
        if result is None:
            return False
        suggestion = parse_selection(str(selection), result.suggestions)
        if suggestion is None:
            return False
        cfg = self.config_loader()
        if cfg.feedback_enabled:
            features = build_feedback_features(suggestion=suggestion, trigger="accept")
            record_feedback(
                "accepted_started",
                suggestion=suggestion,
                cwd=self.workspace_root,
                fingerprint=result.fingerprint,
                features=features,
            )
            self._pending_acceptance = {
                "suggestion": suggestion,
                "fingerprint": result.fingerprint,
                "features": features,
            }
        if self.submit is not None:
            self.submit(suggestion.prompt)
        self._last_result = None
        return True

    def dismiss(self) -> None:
        result = self._last_result
        cfg = self.config_loader()
        if cfg.feedback_enabled:
            record_feedback(
                "dismissed",
                cwd=self.workspace_root,
                fingerprint=result.fingerprint if result else "",
            )
        self._last_result = None
        self.on_user_interaction("dismiss")

    def close(self) -> None:
        self._record_acceptance_outcome("accepted_aborted")
        with self._lock:
            self._generation_id += 1
            self._cancel_locked()

    def _arm_locked(self) -> None:
        cfg = self.config_loader()
        reasons = self._guard_reasons(cfg)
        if reasons:
            logger.debug("_arm_locked skipped: %s", reasons)
            return
        self._cancel_locked()
        self._timer = self.timer_factory.call_later(cfg.idle_seconds, self._on_idle_timer)
        logger.debug(
            "_arm_locked armed: idle_seconds=%s busy=%s draft=%r valid_input_seen=%s conv_has_user=%s",
            cfg.idle_seconds,
            self._busy,
            self._prompt_draft_text,
            self._valid_input_seen,
            self._conversation_has_user_input(),
        )

    def _on_idle_timer(self) -> None:
        with self._lock:
            if self._busy or self._running:
                logger.debug("_on_idle_timer skipped: busy=%s running=%s", self._busy, self._running)
                return
            cfg = self.config_loader()
            reasons = self._guard_reasons(cfg)
            if reasons:
                logger.debug("_on_idle_timer skipped: %s", reasons)
                return
            self._running = True
            self._timer = None
            self._generation_id += 1
            generation_id = self._generation_id

        logger.debug(
            "_on_idle_timer firing: generation_id=%s retry=%d",
            generation_id,
            self._auto_retry_count,
        )
        success = False
        try:
            service = IntentForecastService(
                conversation=self._conversation(),
                provider=self.provider_getter(),
                model=self.model_getter(),
                workspace_root=self.workspace_root,
                config=cfg,
            )
            result = service.generate(trigger="auto")
            with self._lock:
                stale = generation_id != self._generation_id
            logger.debug(
                "_on_idle_timer generated: stale=%s fingerprint=%s suggestions=%d",
                stale,
                result.fingerprint,
                len(result.suggestions),
            )
            try:
                save_forecast_result(
                    result,
                    trigger="auto",
                    cwd=self.workspace_root,
                    model=self.model_getter(),
                    stale=stale,
                )
            except Exception:
                logger.exception("Intent Forecast history save failed: trigger=auto")
            if stale:
                if cfg.feedback_enabled:
                    record_feedback(
                        "stale",
                        cwd=self.workspace_root,
                        fingerprint=result.fingerprint,
                    )
                return
            if result.fingerprint in self._shown_fingerprints:
                return
            self._shown_fingerprints.add(result.fingerprint)
            if result.generated:
                self._last_result = result
            self.display(result)
            success = True
        except Exception as exc:
            logger.warning(
                "Intent Forecast auto trigger failed (attempt %d/%d): %s: %s",
                self._auto_retry_count + 1,
                _AUTO_RETRY_MAX,
                type(exc).__name__,
                exc,
            )
            logger.debug("Intent Forecast auto trigger traceback:", exc_info=True)
            with self._lock:
                if self._auto_retry_count < _AUTO_RETRY_MAX:
                    self._auto_retry_count += 1
                    self._timer = self.timer_factory.call_later(_AUTO_RETRY_BASE_SECONDS, self._on_idle_timer)
                    logger.warning(
                        "Intent Forecast auto trigger will retry in %ds (attempt %d/%d).",
                        _AUTO_RETRY_BASE_SECONDS,
                        self._auto_retry_count + 1,
                        _AUTO_RETRY_MAX,
                    )
                else:
                    logger.warning(
                        "Intent Forecast auto trigger giving up after %d failed attempts; "
                        "use /forecast run to retry manually.",
                        _AUTO_RETRY_MAX,
                    )
        finally:
            with self._lock:
                self._running = False
                if success:
                    self._auto_retry_count = 0

    def _conversation(self) -> Any | None:
        if self.conversation_getter is not None:
            return self.conversation_getter()
        session = self.session_getter()
        return getattr(session, "conversation", None)

    def _conversation_has_user_input(self) -> bool:
        conversation = self._conversation()
        messages = list(getattr(conversation, "messages", []) or [])
        for msg in messages:
            if str(getattr(msg, "role", "") or "") != "user":
                continue
            content = getattr(msg, "content", "")
            if _content_has_text(content):
                return True
        return False

    def _maybe_record_user_acceptance_outcome(self, text: str) -> None:
        if self._pending_acceptance is None:
            return
        if looks_like_correction(text):
            self._record_acceptance_outcome("accepted_corrected")
        elif looks_like_followup(text):
            self._record_acceptance_outcome("accepted_followup")

    def _record_acceptance_outcome(self, event: str) -> None:
        pending = self._pending_acceptance
        if pending is None:
            return
        cfg = self.config_loader()
        if cfg.feedback_enabled:
            record_feedback(
                event,
                suggestion=pending.get("suggestion"),
                cwd=self.workspace_root,
                fingerprint=str(pending.get("fingerprint") or ""),
                features=pending.get("features") if isinstance(pending.get("features"), dict) else {},
            )
        self._pending_acceptance = None

    def _cancel_locked(self) -> None:
        if self._timer is not None:
            self._timer.cancel()
        self._timer = None

    def _guard_reasons(self, cfg: IntentForecastConfig) -> list[str]:
        reasons: list[str] = []
        if not self.interactive:
            reasons.append("non-interactive")
        if not cfg.enabled:
            reasons.append("cfg.enabled=False")
        if not cfg.auto_display:
            reasons.append("cfg.auto_display=False")
        if self._busy:
            reasons.append("busy")
        if self._prompt_draft_text.strip():
            reasons.append(f"prompt_draft={self._prompt_draft_text[:40]!r}")
        if self._valid_input_seen:
            reasons.append("valid_input_seen=True")
        if self._conversation_has_user_input():
            reasons.append("conversation_has_user_input")
        return reasons


def _content_has_text(content: Any) -> bool:
    if content is None:
        return False
    if isinstance(content, str):
        return bool(content.strip())
    if isinstance(content, list):
        return any(_content_has_text(item) for item in content)
    if isinstance(content, dict):
        return _content_has_text(content.get("text") or content.get("content") or "")
    text = getattr(content, "text", None)
    if text is not None:
        return bool(str(text).strip())
    return bool(str(content).strip())
