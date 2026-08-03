"""End-to-end integration tests for F-88 P88-C + P88-D.

Verifies that the auto-routing classifier and the report-persist
hook fire correctly through the full Agent tool dispatch.
"""

from __future__ import annotations


# pylint: disable=C0412,E0611,W0621
import json
from pathlib import Path
from unittest.mock import patch

import pytest

from src.bootstrap.state import get_session_id, reset_state_for_tests
from src.tool_system.context import ToolContext, ToolUseOptions
from src.tool_system.defaults import build_default_registry
from clawcodex_ext.tool_system.protocol import ToolCall
from clawcodex_ext.types.content_blocks import TextBlock
from src.types.messages import AssistantMessage


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_context(tmp_path: Path) -> ToolContext:
    ctx = ToolContext(workspace_root=tmp_path)
    ctx.options = ToolUseOptions(is_non_interactive_session=False)
    return ctx


@pytest.fixture
def fresh_state(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Reset bootstrap state so each test gets a fresh session id."""
    reset_state_for_tests()
    monkeypatch.setenv("PYTEST_CURRENT_TEST", "f88")


# ---------------------------------------------------------------------------
# P88-C — auto-routing through the Agent tool dispatch
# ---------------------------------------------------------------------------


def test_auto_routing_picks_explore(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fresh_state: None) -> None:
    """A prompt that matches the Explore phrase table is dispatched
    with ``subagent_type='Explore'`` — even though the model did not
    pass one explicitly.
    """
    registry = build_default_registry(provider=object())
    context = _make_context(tmp_path)

    captured: dict[str, object] = {}

    async def _fake_run_agent(params):
        captured["agent_type"] = params.agent_definition.agent_type
        yield AssistantMessage(content=[TextBlock(text="done exploring")])

    with patch("src.tool_system.tools.agent.run_agent", _fake_run_agent):
        result = registry.dispatch(
            ToolCall(
                name="Agent",
                input={"description": "scan", "prompt": "explore the codebase"},
            ),
            context,
        )

    assert result.is_error is False
    assert captured["agent_type"] == "Explore"


def test_auto_routing_picks_plan(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fresh_state: None) -> None:
    """A prompt that matches the Plan phrase table is dispatched
    with ``subagent_type='Plan'``.
    """
    registry = build_default_registry(provider=object())
    context = _make_context(tmp_path)

    captured: dict[str, object] = {}

    async def _fake_run_agent(params):
        captured["agent_type"] = params.agent_definition.agent_type
        yield AssistantMessage(content=[TextBlock(text="plan ready")])

    with patch("src.tool_system.tools.agent.run_agent", _fake_run_agent):
        result = registry.dispatch(
            ToolCall(
                name="Agent",
                input={"description": "plan", "prompt": "make a plan for auth"},
            ),
            context,
        )

    assert result.is_error is False
    assert captured["agent_type"] == "Plan"


def test_auto_routing_falls_back_to_general_purpose(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fresh_state: None
) -> None:
    """A prompt with no Explore / Plan phrase falls through to
    ``general-purpose`` — the existing default dispatch is
    preserved.
    """
    registry = build_default_registry(provider=object())
    context = _make_context(tmp_path)

    captured: dict[str, object] = {}

    async def _fake_run_agent(params):
        captured["agent_type"] = params.agent_definition.agent_type
        yield AssistantMessage(content=[TextBlock(text="ok")])

    with patch("src.tool_system.tools.agent.run_agent", _fake_run_agent):
        result = registry.dispatch(
            ToolCall(
                name="Agent",
                input={"description": "fix", "prompt": "fix the typo on line 42"},
            ),
            context,
        )

    assert result.is_error is False
    assert captured["agent_type"] == "general-purpose"


def test_explicit_subagent_type_overrides_routing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fresh_state: None
) -> None:
    """If the model passes ``subagent_type`` explicitly, it wins —
    the classifier is a fallback, not an override.
    """
    registry = build_default_registry(provider=object())
    context = _make_context(tmp_path)

    captured: dict[str, object] = {}

    async def _fake_run_agent(params):
        captured["agent_type"] = params.agent_definition.agent_type
        yield AssistantMessage(content=[TextBlock(text="done")])

    with patch("src.tool_system.tools.agent.run_agent", _fake_run_agent):
        # The prompt would route to Explore, but the model explicitly
        # asked for general-purpose.
        result = registry.dispatch(
            ToolCall(
                name="Agent",
                input={
                    "description": "explicit",
                    "prompt": "explore the codebase",
                    "subagent_type": "general-purpose",
                },
            ),
            context,
        )

    assert result.is_error is False
    assert captured["agent_type"] == "general-purpose"


# ---------------------------------------------------------------------------
# P88-D — Explore / Plan reports persist to disk
# ---------------------------------------------------------------------------


def test_explore_report_persisted_to_disk(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fresh_state: None) -> None:
    """An auto-routed Explore call writes both ``.md`` and ``.json``
    files to ``$CLAWCODEX_HOME/reports/explore/<session_id>/<agent_id>.*``.
    """
    monkeypatch.setenv("CLAWCODEX_HOME", str(tmp_path))
    registry = build_default_registry(provider=object())
    context = _make_context(tmp_path)

    async def _fake_run_agent(_params):
        yield AssistantMessage(
            content=[
                TextBlock(
                    text=(
                        "# Demo Explore\n\n"
                        "Found two things.\n\n"
                        "- One thing\n"
                        "- Two thing\n\n"
                        "### Critical Files for Implementation\n"
                        "- src/foo.py\n"
                    )
                )
            ]
        )

    with patch("src.tool_system.tools.agent.run_agent", _fake_run_agent):
        result = registry.dispatch(
            ToolCall(
                name="Agent",
                input={"description": "scan", "prompt": "explore the codebase"},
            ),
            context,
        )

    assert result.is_error is False

    # The session id from bootstrap state must match what the report
    # store wrote under.
    sid = get_session_id()
    report_dir = tmp_path / "reports" / "explore" / str(sid)
    assert report_dir.exists(), f"expected {report_dir} to exist"

    md_files = list(report_dir.glob("*.md"))
    json_files = list(report_dir.glob("*.json"))
    assert len(md_files) == 1
    assert len(json_files) == 1
    assert md_files[0].stem == json_files[0].stem

    payload = json.loads(json_files[0].read_text(encoding="utf-8"))
    assert payload["kind"] == "explore"
    assert payload["title"] == "Demo Explore"
    assert "Found two things." in payload["summary"]
    assert "One thing" in payload["findings"]
    assert payload["critical_files"] == ["src/foo.py"]


def test_plan_report_persisted_to_disk(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fresh_state: None) -> None:
    """An auto-routed Plan call writes both ``.md`` and ``.json``
    files to ``$CLAWCODEX_HOME/reports/plan/<session_id>/<agent_id>.*``.
    """
    monkeypatch.setenv("CLAWCODEX_HOME", str(tmp_path))
    registry = build_default_registry(provider=object())
    context = _make_context(tmp_path)

    async def _fake_run_agent(_params):
        yield AssistantMessage(
            content=[
                TextBlock(
                    text=(
                        "# Demo Plan\n\n"
                        "A high-level approach.\n\n"
                        "1. Gather requirements\n"
                        "2. Design the schema\n"
                        "3. Implement\n\n"
                        "### Critical Files for Implementation\n"
                        "- src/auth/login.py\n"
                        "- tests/test_auth.py\n"
                    )
                )
            ]
        )

    with patch("src.tool_system.tools.agent.run_agent", _fake_run_agent):
        result = registry.dispatch(
            ToolCall(
                name="Agent",
                input={"description": "plan", "prompt": "make a plan for auth"},
            ),
            context,
        )

    assert result.is_error is False

    sid = get_session_id()
    report_dir = tmp_path / "reports" / "plan" / str(sid)
    assert report_dir.exists()

    md_files = list(report_dir.glob("*.md"))
    json_files = list(report_dir.glob("*.json"))
    assert len(md_files) == 1
    assert len(json_files) == 1

    payload = json.loads(json_files[0].read_text(encoding="utf-8"))
    assert payload["kind"] == "plan"
    assert payload["title"] == "Demo Plan"
    assert "Gather requirements" in payload["steps"]
    assert "Design the schema" in payload["steps"]
    assert "Implement" in payload["steps"]
    assert payload["critical_files"] == ["src/auth/login.py", "tests/test_auth.py"]


def test_general_purpose_does_not_persist_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fresh_state: None
) -> None:
    """A general-purpose agent does NOT write to the report store —
    only the one-shot agents (Explore / Plan) are persisted.
    """
    monkeypatch.setenv("CLAWCODEX_HOME", str(tmp_path))
    registry = build_default_registry(provider=object())
    context = _make_context(tmp_path)

    async def _fake_run_agent(_params):
        yield AssistantMessage(content=[TextBlock(text="general done")])

    with patch("src.tool_system.tools.agent.run_agent", _fake_run_agent):
        result = registry.dispatch(
            ToolCall(
                name="Agent",
                input={"description": "fix", "prompt": "fix the typo on line 42"},
            ),
            context,
        )

    assert result.is_error is False

    sid = get_session_id()
    explore_dir = tmp_path / "reports" / "explore" / str(sid)
    plan_dir = tmp_path / "reports" / "plan" / str(sid)
    assert not explore_dir.exists()
    assert not plan_dir.exists()
