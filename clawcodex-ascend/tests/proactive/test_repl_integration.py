from __future__ import annotations

import time

from clawcodex_ext.repl.proactive_integration import format_proactive_status
from clawcodex_ext.services.proactive.state import AutomationState


def test_format_proactive_status_hides_inactive() -> None:
    assert format_proactive_status(AutomationState(phase="inactive")) == ""


def test_format_proactive_status_shows_active_countdown() -> None:
    state = AutomationState(
        phase="active",
        next_tick_at=time.time() * 1000 + 5_000,
    )

    text = format_proactive_status(state)

    assert text.startswith("proactive:active ")
    assert text.endswith("s")


def test_format_proactive_status_shows_paused() -> None:
    assert format_proactive_status(AutomationState(phase="paused")) == "proactive:paused"
