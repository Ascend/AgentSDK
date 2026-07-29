from __future__ import annotations

from clawcodex_ext.services.kairos import TickEvent
from clawcodex_ext.services.proactive import ProactiveController, TickEmitter


def test_tick_emitter_injects_tick_to_outbox() -> None:
    ctrl = ProactiveController()
    outbox = []
    emitter = TickEmitter(controller=ctrl, outbox=outbox)
    ctrl.activate("test")

    text = emitter._on_tick_event(
        TickEvent(
            scheduler_id="test",
            tick_number=1,
            scheduled_at=1.0,
            actual_at=1.0,
        )
    )

    assert text is not None
    assert text.startswith("<tick>")
    assert outbox[0].get("type") == "proactive_prompt"
    assert outbox[0].prompt == text
    assert ctrl.state.tick_count == 1


def test_tick_emitter_blocks_when_compact_fails() -> None:
    ctrl = ProactiveController()
    ctrl.activate("test")

    def compact() -> None:
        raise RuntimeError("boom")

    emitter = TickEmitter(
        controller=ctrl,
        outbox=[],
        should_compact_first=lambda: True,
        compact_callback=compact,
    )

    assert emitter.emit_now() is None
    assert ctrl.state.phase == "blocked"
