from __future__ import annotations

from pathlib import Path

from clawcodex_ext.services.proactive import reset_default_controller_for_tests
from clawcodex_ext.tool_system.context import ToolContext
from clawcodex_ext.tool_system.tools.sleep import _sleep_call
from extensions.remote_api.core import _response_metadata
from extensions.remote_api.state_reporter import current_automation_state


def test_sleep_tool_enters_proactive_sleep(tmp_path: Path) -> None:
    ctrl = reset_default_controller_for_tests()
    ctrl.activate("test")
    context = ToolContext(workspace_root=tmp_path)

    result = _sleep_call({"seconds": 0.2}, context)

    assert result.output["proactive"] is True
    assert ctrl.state.phase == "sleeping"


def test_remote_metadata_reports_automation_state() -> None:
    ctrl = reset_default_controller_for_tests()
    ctrl.activate("test", focus="minimal")

    state = current_automation_state()
    metadata = _response_metadata()

    assert state["phase"] == "active"
    assert metadata["automation_state"]["focus"] == "minimal"
