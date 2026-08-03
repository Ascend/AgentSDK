# -------------------------------------------------------------------------
# This file is part of the AgentSDK project.
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
#
# Copyright (c) 2026 Clawd Codex Team
# SPDX-License-Identifier: MIT
# Source: https://github.com/agentforce314/clawcodex
# ClawCodex-derived portions remain licensed under the MIT License.
# See clawcodex-ascend/LICENSE.clawcodex.

"""Status line widget — parity with ``StatusLine.tsx`` + ``SpinnerWithVerb``.

The ink reference renders a single line at the bottom of the transcript
that shows:

* On the left: provider · model.
* In the middle: the current "verb" (``Synthesizing…``, ``Gathering…``,
  ``Running bash…``) paired with an animated spinner while the agent is
  busy, plus an elapsed-time indicator.
* On the right: turn count, queued prompts pill, and usage (total
  tokens) once a run has completed.

Phase 1 renders this as a static-looking Rich ``Text`` updated by an
interval timer. The animation is strictly cosmetic — the authoritative
"am I busy?" signal comes from :class:`AppState.is_thinking`.
"""

from __future__ import annotations
# pylint: disable=E0611,W0201

import time
from pathlib import Path
from typing import Any

from rich.text import Text
from textual.reactive import reactive
from textual.widgets import Static

from ..state import AppState


# Braille spinner frames (standard) + sparkle characters for visual variety.
# Every third tick, the spinner shows a sparkle instead of the next braille
# frame to give a "✨ processing" feel that matches the ink reference's
# sparkle-based busy indicator.
_SPINNER_FRAMES = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
_SPARKLE_CHARS = "✨✦✧❋⚡"


class StatusLine(Static):
    """Single-line status footer."""

    DEFAULT_CSS = """
    StatusLine {
        height: 1;
        background: $surface;
        color: $text;
        padding: 0 1;
    }
    """

    turns: reactive[int] = reactive(0)
    is_thinking: reactive[bool] = reactive(False)
    queued: reactive[int] = reactive(0)
    permission_mode: reactive[str] = reactive("")

    def __init__(
        self,
        *,
        provider: str,
        model: str,
        workspace_root: Path,
        app_state: AppState | None = None,
        provider_instance: object | None = None,
    ) -> None:
        self._provider = provider
        self._model = model
        self._workspace_root = Path(workspace_root)
        self._app_state = app_state
        # Live provider instance (BaseProvider) — used ONLY for the
        # advisor status segment so ``format_advisor_status`` can call
        # ``is_advisor_enabled(provider)`` and pick the right mode
        # label. Optional: when omitted, the advisor segment shows
        # "(client)" as the conservative default for any configured
        # advisor (since SERVER_SIDE requires the instance to verify
        # first-party).
        self._provider_instance = provider_instance
        self._frame = 0
        self._sparkle_frame = 0
        self._timer = None
        initial = Text(f"{provider} · {model}    ready    turn 0")
        super().__init__(initial, markup=False)

    # ---- lifecycle ----
    def on_mount(self) -> None:
        self._timer = self.set_interval(1 / 10, self._tick)
        self._redraw()

    def _tick(self) -> None:
        if self._app_state is not None:
            self.is_thinking = self._app_state.is_thinking
            self.queued = len(self._app_state.queued_prompts)
        if self.is_thinking:
            self._frame = (self._frame + 1) % len(_SPINNER_FRAMES)
            # Cycle through sparkle chars on every 3rd tick for variety
            if self._frame % 3 == 0:
                self._sparkle_frame = (self._sparkle_frame + 1) % len(_SPARKLE_CHARS)
        self._redraw()

    # ---- public API ----
    def bind_state(self, state: AppState) -> None:
        self._app_state = state
        self._redraw()

    def refresh_identity(
        self,
        *,
        provider: str | None = None,
        model: str | None = None,
    ) -> None:
        if provider is not None:
            self._provider = provider
        if model is not None:
            self._model = model
        self._redraw()

    def bump_turn(self) -> None:
        self.turns += 1

    def set_busy(self, verb: str = "Synthesizing") -> None:
        self.is_thinking = True
        if self._app_state is not None:
            self._app_state.set_thinking(True, verb=verb)

    def set_idle(self) -> None:
        self.is_thinking = False
        if self._app_state is not None:
            self._app_state.set_thinking(False)

    def set_permission_mode(self, mode: str | None) -> None:
        self.permission_mode = mode or ""

    # ---- render ----
    def bind_footer(self, footer: Any) -> None:
        """Wire this StatusLine's thinking state to a PromptInputFooter.

        When ``is_thinking`` flips, push the new value to the footer's
        ``set_loading`` method so the context-aware hint row under the
        prompt input switches to "esc to interrupt" while the agent is
        busy.  The ``Any`` type avoids a circular import with
        ``prompt_input_footer``.
        """
        self._footer_ref = footer
        # Push the current state immediately so the footer is in sync
        # even if ``is_thinking`` hasn't changed yet.
        footer.set_loading(self.is_thinking)

    def watch_is_thinking(self, value: bool) -> None:
        if hasattr(self, "_footer_ref"):
            self._footer_ref.set_loading(value)
        self._redraw()

    def watch_turns(self, _: int) -> None:
        self._redraw()

    def watch_queued(self, _: int) -> None:
        self._redraw()

    def watch_permission_mode(self, _: str) -> None:
        self._redraw()

    def watch_goal(self, _: dict | None) -> None:
        self._redraw()

    def _redraw(self) -> None:
        if self.is_thinking:
            # Show sparkle on every 4th tick for visual variety
            if self._frame % 4 == 0:
                spinner = _SPARKLE_CHARS[self._sparkle_frame]
            else:
                spinner = _SPINNER_FRAMES[self._frame]
        else:
            spinner = " "
        self.update(self._compose_text(spinner=spinner))

    def _compose_text(self, *, spinner: str) -> Text:
        state = self._app_state
        verb = state.verb if state else ("thinking" if self.is_thinking else "ready")
        elapsed = ""
        if state and state.is_thinking and state.verb_started_at:
            secs = int(time.time() - state.verb_started_at)
            if secs > 0:
                elapsed = f" {secs}s"

        left_parts = [f"{self._provider} · {self._model}"]
        # Permission mode segment — reflects the active permission mode.
        if self.permission_mode:
            left_parts.append(f"mode: {self.permission_mode}")
        # Optional advisor segment — appears next to provider/model
        # when ``/advisor`` is configured. Mode label reflects what
        # the NEXT request will actually do (server/client/inactive)
        # so a stale config under an unsupported provider doesn't
        # silently lie. Shared formatter with the legacy REPL
        # bottom_toolbar so both surfaces render identically.
        try:
            from src.utils.advisor import format_advisor_status

            # Pass the live provider instance when available so the
            # mode label (server/client/inactive) reflects what the
            # next request will actually do. Falls back to None (=
            # "client" default) when the instance isn't plumbed.
            advisor_seg = format_advisor_status(
                self._provider_instance,
                self._model,
            )
        except Exception:
            advisor_seg = None
        if advisor_seg:
            left_parts.append(advisor_seg)
        left = " · ".join(left_parts)
        cwd = self._display_cwd()
        middle = f"{spinner} {verb}{elapsed}" if self.is_thinking else verb
        right_bits: list[str] = [f"turn {self.turns}"]
        goal_segment = _goal_status_segment(getattr(state, "goal_status", None))
        if goal_segment:
            right_bits.append(goal_segment)
        if self.queued:
            right_bits.append(f"queued {self.queued}")
        # Real-time token display — shown both during and after a run
        # so the user sees live metrics while the agent is busy.
        in_t = 0
        out_t = 0
        if state and state.usage:
            in_t = state.usage.get("input_tokens", 0)
            out_t = state.usage.get("output_tokens", 0)
        total = in_t + out_t
        if total:
            right_bits.append(f"tokens {in_t} in / {out_t} out")
            # Advisor token segment — appears next to worker tokens
            # whenever the advisor has been consulted this session.
            # ``state.usage["advisor_*"]`` is mirrored from
            # ``tool_context.advisor_*`` by ``agent_bridge.py`` after
            # each run; the underlying ctx counter is accumulated by
            # ``AdvisorTool._advisor_call`` on every consultation.
            # Hidden when zero so the bar stays compact for users who
            # haven't enabled the advisor yet.
            adv_in = state.usage.get("advisor_input_tokens", 0)
            adv_out = state.usage.get("advisor_output_tokens", 0)
            if adv_in or adv_out:
                right_bits.append(f"advisor {adv_in}/{adv_out}")
            # USD cost segment — uses the shared compute_session_cost
            # helper so REPL and TUI render identical numbers for the
            # same usage. Directional estimate based on upstream model
            # prices; proxies (litellm/openrouter/bedrock) may bill
            # differently. Hidden when zero.
            try:
                from clawcodex_ext.services.pricing import (
                    compute_session_cost,
                    format_cost_usd,
                )
                from src.settings.settings import get_settings

                _adv_model = (getattr(get_settings(), "advisor_model", "") or "").strip()
                _, _, total_cost = compute_session_cost(
                    worker_model=self._model,
                    worker_input_tokens=in_t,
                    worker_output_tokens=out_t,
                    advisor_model=_adv_model,
                    advisor_input_tokens=adv_in,
                    advisor_output_tokens=adv_out,
                )
                if total_cost > 0:
                    right_bits.append(f"cost {format_cost_usd(total_cost)}")
            except Exception:  # nosec B110
                pass
        right = " · ".join(right_bits)
        proactive_text = ""
        try:
            from clawcodex_ext.repl.proactive_integration import (
                format_proactive_status,
            )

            proactive_status = format_proactive_status()
            if proactive_status:
                proactive_text = f" 路 {proactive_status}"
        except Exception:
            proactive_text = ""
        return Text(f"{left}    {middle}    {cwd}    {right}{proactive_text}")

    def _display_cwd(self) -> str:
        try:
            home = Path.home()
            rel = self._workspace_root.relative_to(home)
            return f"~/{rel}" if str(rel) != "." else "~"
        except Exception:
            return str(self._workspace_root)


def _goal_status_segment(goal: dict | None, *, now: float | None = None) -> str | None:
    if not isinstance(goal, dict):
        return None
    status = str(goal.get("status") or "")
    if status not in {
        "active",
        "paused",
        "blocked",
        "usage_limited",
        "budget_limited",
        "complete",
    }:
        return None

    elapsed = _goal_int(goal.get("timeUsedSeconds", goal.get("time_used_seconds", 0)))
    active_since = goal.get("activeSince", goal.get("active_since"))
    try:
        if status == "active" and active_since is not None:
            current = time.time() if now is None else now
            elapsed = max(int(current - float(active_since)), 0)
    except (TypeError, ValueError):
        pass
    return f"◎ /goal {status} ({_format_goal_elapsed_seconds(elapsed)})"


def _goal_int(value: object) -> int:
    try:
        return max(int(value or 0), 0)
    except (TypeError, ValueError):
        return 0


def _format_goal_elapsed_seconds(seconds: int) -> str:
    seconds = max(seconds, 0)
    if seconds < 60:
        return f"{seconds}s"
    minutes = seconds // 60
    if minutes < 60:
        return f"{minutes}m"
    hours = minutes // 60
    remaining_minutes = minutes % 60
    if hours >= 24:
        days = hours // 24
        remaining_hours = hours % 24
        return f"{days}d {remaining_hours}h {remaining_minutes}m"
    if remaining_minutes == 0:
        return f"{hours}h"
    return f"{hours}h {remaining_minutes}m"
