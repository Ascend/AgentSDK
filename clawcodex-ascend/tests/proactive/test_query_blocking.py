from __future__ import annotations

from clawcodex_ext.query.query import _mark_proactive_context_blocked_on_error
from clawcodex_ext.services.proactive import (
    get_default_controller,
    reset_default_controller_for_tests,
)


def test_query_error_blocks_active_proactive_context() -> None:
    ctrl = reset_default_controller_for_tests()
    ctrl.activate("test")

    _mark_proactive_context_blocked_on_error(RuntimeError("api failed"))

    assert get_default_controller().state.phase == "blocked"


def test_query_error_does_not_activate_inactive_proactive_context() -> None:
    ctrl = reset_default_controller_for_tests()

    _mark_proactive_context_blocked_on_error(RuntimeError("api failed"))

    assert ctrl.state.phase == "inactive"
