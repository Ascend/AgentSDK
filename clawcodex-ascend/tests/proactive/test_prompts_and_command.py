from __future__ import annotations

import os
from pathlib import Path

from clawcodex_ext.command_system.builtins import execute_command_sync
from clawcodex_ext.command_system.engine import create_command_context
from clawcodex_ext.command_system.proactive_command import _EMITTERS_BY_CONTEXT_ID
from clawcodex_ext.feature_gate import FeatureFlag, get_registry
from clawcodex_ext.goal.model import ThreadGoalStatus
from clawcodex_ext.goal.store import GoalStore
from clawcodex_ext.services.proactive import (
    get_default_controller,
    get_proactive_section,
    reset_default_controller_for_tests,
)
from clawcodex_ext.tool_system.context import ToolContext


def test_proactive_section_is_goal_aware(tmp_path: Path) -> None:
    # Point CLAWCODEX_HOME at tmp_path so _active_goal_block's GoalStore()
    # shares the same SQLite database as the test setup.
    os.environ["CLAWCODEX_HOME"] = str(tmp_path)
    store = GoalStore()
    store.insert_thread_goal(
        thread_id="sess",
        objective="Ship F-89",
        status=ThreadGoalStatus.ACTIVE,
        token_budget=None,
    )
    ctrl = reset_default_controller_for_tests()
    ctrl.activate("test")

    section = get_proactive_section("medium", session_id="sess")

    assert section is not None
    assert "<active-goal" in section
    assert "Ship F-89" in section


def test_proactive_command_controls_state(tmp_path: Path) -> None:
    reset_default_controller_for_tests()
    reg = get_registry()
    if reg.get_flag("PROACTIVE") is None:
        reg.register(FeatureFlag("PROACTIVE", default=False))
    reg.enable_feature("PROACTIVE")
    context = create_command_context(tmp_path)

    success, result, error = execute_command_sync(
        "proactive",
        "on focus=minimal",
        context,
    )

    assert success
    assert error is None
    assert "<system-reminder>" in (result or "")
    assert get_default_controller().state.phase == "active"
    assert get_default_controller().state.focus == "minimal"


def test_proactive_command_rejects_invalid_focus(tmp_path: Path) -> None:
    reset_default_controller_for_tests()
    reg = get_registry()
    if reg.get_flag("PROACTIVE") is None:
        reg.register(FeatureFlag("PROACTIVE", default=False))
    reg.enable_feature("PROACTIVE")
    context = create_command_context(tmp_path)

    success, result, error = execute_command_sync("proactive", "on focus=nope", context)

    assert success
    assert error is None
    assert "Invalid proactive focus" in (result or "")
    assert get_default_controller().state.phase == "inactive"


def test_proactive_command_starts_emitter_when_outbox_available(tmp_path: Path) -> None:
    reset_default_controller_for_tests()
    reg = get_registry()
    if reg.get_flag("PROACTIVE") is None:
        reg.register(FeatureFlag("PROACTIVE", default=False))
    reg.enable_feature("PROACTIVE")
    tool_context = ToolContext(workspace_root=tmp_path)
    context = create_command_context(tmp_path, tool_context=tool_context)

    success, _, error = execute_command_sync("proactive", "on", context)
    emitter = _EMITTERS_BY_CONTEXT_ID.get(id(tool_context))

    assert success
    assert error is None
    assert emitter is not None
    assert emitter.scheduler.start() is False

    off_success, _, off_error = execute_command_sync("proactive", "off", context)
    assert off_success
    assert off_error is None
    assert id(tool_context) not in _EMITTERS_BY_CONTEXT_ID
