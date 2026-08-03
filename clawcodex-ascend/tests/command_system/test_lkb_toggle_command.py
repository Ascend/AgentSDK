"""Tests for the bare ``/lkb`` interactive on/off toggle (merged flag).

Covers the ``LkbCommand`` picker flow: selecting "on"/"off" persists the
single ``LKB_PLAN_GRAPH`` feature flag via the feature-gate registry, and
cancelling leaves the registry untouched. Subcommand args still route to
the plain-text ``_lkb_call`` path.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from clawcodex_ext.command_system.lkb_command import LKB_COMMAND, LkbCommand
from clawcodex_ext.command_system.types import (
    CommandContext,
    InteractiveCommand,
    InteractiveOutcome,
)


class _FakeUI:
    """Minimal UIHost stand-in: records ``select`` calls, returns a scripted pick."""

    def __init__(self, picked: str | None) -> None:
        self._picked = picked
        self.select_calls: list[tuple[str, object, object]] = []

    async def select(self, title, options, current=None):
        self.select_calls.append((title, options, current))
        return self._picked


@pytest.fixture
def registry_spy(monkeypatch: pytest.MonkeyPatch) -> dict:
    """Spy on set_override/save_config of the feature-gate registry."""
    from clawcodex_ext.feature_gate import get_registry

    reg = get_registry()
    spy = {"overrides": [], "saved": 0}
    monkeypatch.setattr(reg, "set_override", lambda name, enabled: spy["overrides"].append((name, enabled)))
    monkeypatch.setattr(reg, "save_config", lambda: spy.__setitem__("saved", spy["saved"] + 1))
    return spy


def _ctx(tmp_path: Path, picked: str | None) -> CommandContext:
    return CommandContext(workspace_root=tmp_path, cwd=tmp_path, ui=_FakeUI(picked))


def test_lkb_command_is_interactive() -> None:
    assert isinstance(LKB_COMMAND, InteractiveCommand)
    assert isinstance(LKB_COMMAND, LkbCommand)
    assert LKB_COMMAND.name == "lkb"


@pytest.mark.asyncio
async def test_toggle_on_persists_flag(tmp_path: Path, registry_spy: dict) -> None:
    outcome = await LKB_COMMAND.run("", _ctx(tmp_path, "on"))
    assert isinstance(outcome, InteractiveOutcome)
    assert registry_spy["overrides"] == [("LKB_PLAN_GRAPH", True)]
    assert registry_spy["saved"] == 1
    assert "LKB enabled" in (outcome.message or "")


@pytest.mark.asyncio
async def test_toggle_off_persists_flag(tmp_path: Path, registry_spy: dict) -> None:
    outcome = await LKB_COMMAND.run("", _ctx(tmp_path, "off"))
    assert isinstance(outcome, InteractiveOutcome)
    assert registry_spy["overrides"] == [("LKB_PLAN_GRAPH", False)]
    assert registry_spy["saved"] == 1
    assert "LKB disabled" in (outcome.message or "")


@pytest.mark.asyncio
async def test_toggle_cancel_touches_nothing(tmp_path: Path, registry_spy: dict) -> None:
    outcome = await LKB_COMMAND.run("", _ctx(tmp_path, None))
    assert isinstance(outcome, InteractiveOutcome)
    assert "Cancelled" in (outcome.message or "")
    assert registry_spy["overrides"] == []
    assert registry_spy["saved"] == 0


@pytest.mark.asyncio
async def test_subcommand_routes_to_text_path(tmp_path: Path, registry_spy: dict) -> None:
    """`/lkb status` skips the picker and goes through `_lkb_call`."""
    ctx = _ctx(tmp_path, "on")  # picker must not be consulted
    outcome = await LKB_COMMAND.run("status", ctx)
    assert isinstance(outcome, InteractiveOutcome)
    # No tool_context on the ctx → the text path reports that (flag guard
    # would otherwise produce the not-enabled message — either proves the
    # subcommand bypassed the picker).
    assert ctx.ui.select_calls == []
    message = outcome.message or ""
    assert "tool_context" in message or "not currently enabled" in message
    assert registry_spy["overrides"] == []
    assert registry_spy["saved"] == 0
