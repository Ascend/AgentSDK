#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# -------------------------------------------------------------------------
#  This file is part of the AgentSDK project.
# Copyright (c) 2026 Huawei Technologies Co.,Ltd.
#
# AgentSDK is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#
#           http://license.coscl.org.cn/MulanPSL2
#
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
# -------------------------------------------------------------------------

"""Tests for stage3g bottom toolbar."""

from __future__ import annotations

import pytest


@pytest.fixture(scope="module")
def _heavy_runtime():
    """Test helper for heavy runtime."""
    from src.repl.core import ClawcodexREPL, _load_heavy_runtime

    _load_heavy_runtime()
    return ClawcodexREPL


class TestStage3gBottomToolbarSource:
    """Tests for TestStage3gBottomToolbarSource."""

    def test_no_orphan_goal_part_reference(self):
        """Verify no orphan goal part reference."""
        import inspect
        import re

        from src.repl.core import ClawcodexREPL

        src = inspect.getsource(ClawcodexREPL._bottom_toolbar)
        # Only flag orphan when ``goal_part`` is referenced in an f-string
        # but has no local assignment. Using it as a regular variable name
        # outside f-strings (e.g. as a plain string concat) is allowed and
        # never raises NameError.
        goal_part_in_fstring = bool(re.search(r'f"\{goal_part\}"', src))
        goal_part_assigned = bool(re.search(r"\bgoal_part\s*(?::\s*\S+)?\s*=", src))
        assert not (goal_part_in_fstring and not goal_part_assigned), (
            "_bottom_toolbar references 'goal_part' in an f-string but "
            "the local assignment was removed (orphan variable regression, "
            "see commit 0293f5e1). Either restore the assignment or "
            "delete the f-string reference."
        )

    def test_all_interpolated_vars_are_assigned(self):
        """Verify all interpolated vars are assigned."""
        import inspect
        import re

        from src.repl.core import ClawcodexREPL

        src = inspect.getsource(ClawcodexREPL._bottom_toolbar)
        refs = set(re.findall(r'f"\{([a-z_][a-z_0-9]*)\}"', src))
        # Skip ``f"{NAME:FORMAT}"`` style with explicit format spec.
        refs = {r for r in refs if ":" not in r}
        for ref in refs:
            # Accept any local assignment shape: NAME = ..., NAME=..., NAME: T = ...
            assigned = bool(re.search(rf"\b{re.escape(ref)}\s*(?::\s*\S+)?\s*=", src))
            assert assigned, (
                f"_bottom_toolbar interpolates f'{{{ref}}}' but never "
                f"assigns it locally — orphan variable reference. Either "
                f"restore the definition or delete the f-string reference."
            )


class TestStage3gBottomToolbarRuntime:
    """Tests for TestStage3gBottomToolbarRuntime."""

    @staticmethod
    def _make_stub(
        *,
        provider_name="anthropic",
        model="claude-sonnet-4-6",
        cwd="/tmp",
        turns=0,
        in_tokens=0,
        out_tokens=0,
        advisor_in=0,
        advisor_out=0,
    ):
        class _Stub:
            pass

        stub = _Stub()
        stub.provider = type(
            "P",
            (),
            {"provider_name": provider_name, "model": model},
        )()
        stub.provider_name = provider_name
        stub.tool_context = type(
            "T",
            (),
            {
                "cwd": cwd,
                "workspace_root": cwd,
                "advisor_input_tokens": advisor_in,
                "advisor_output_tokens": advisor_out,
                "tasks": {},
            },
        )()
        stub._permission_mode = "default"
        stub._stats_turns = turns
        stub._stats_input_tokens = in_tokens
        stub._stats_output_tokens = out_tokens
        stub._shorten_path_text = staticmethod(lambda p: p)
        # ``_bottom_toolbar`` calls ``self._goal_footer_status()``; without
        # a no-op binding the call raises AttributeError on the stub and
        # the outer ``except Exception`` swallows it, returning "".
        stub._goal_footer_status = lambda: None
        stub._goal_footer_id = None
        stub._goal_footer_started_at = None
        stub._task_toolbar_part = lambda: ""
        return stub

    def test_renders_non_empty_string(self, _heavy_runtime):
        """Verify renders non empty string."""
        ClawcodexREPL = _heavy_runtime
        stub = self._make_stub()
        result = ClawcodexREPL._bottom_toolbar(stub)
        assert result, f"expected non-empty status bar, got {result!r}"
        assert "anthropic" in result
        assert "claude-sonnet-4-6" in result
        assert "/tmp" in result

    def test_no_name_error_on_render(self, _heavy_runtime):
        """Verify no name error on render."""
        ClawcodexREPL = _heavy_runtime
        for perm_mode in ("default", "plan", "acceptEdits", "bypassPermissions"):
            stub = self._make_stub()
            stub._permission_mode = perm_mode
            try:
                result = ClawcodexREPL._bottom_toolbar(stub)
            except NameError as e:
                raise AssertionError(
                    f"_bottom_toolbar raised NameError({e}) under "
                    f"permission_mode={perm_mode!r} — orphan variable "
                    f"reference regression (see commit 0293f5e1)."
                ) from e
            assert result, (
                f"permission_mode={perm_mode!r}: got empty result, "
                f"likely a silent failure swallowed by except Exception"
            )

    def test_zero_advisor_hides_advisor_part(self, _heavy_runtime):
        """Verify zero advisor hides advisor part."""
        ClawcodexREPL = _heavy_runtime
        stub = self._make_stub(advisor_in=0, advisor_out=0)
        result = ClawcodexREPL._bottom_toolbar(stub)
        assert "advisor" not in result, f"advisor tokens are 0 but result contains 'advisor': {result!r}"

    def test_nonzero_advisor_renders_advisor_part(self, _heavy_runtime):
        """Verify nonzero advisor renders advisor part."""
        ClawcodexREPL = _heavy_runtime
        stub = self._make_stub(advisor_in=1234, advisor_out=567)
        result = ClawcodexREPL._bottom_toolbar(stub)
        assert "advisor" in result, f"advisor tokens are non-zero but result lacks 'advisor': {result!r}"

    def test_zero_cost_hides_cost_part(self, _heavy_runtime):
        """Verify zero cost hides cost part."""
        ClawcodexREPL = _heavy_runtime
        stub = self._make_stub(in_tokens=0, out_tokens=0)
        result = ClawcodexREPL._bottom_toolbar(stub)
        assert "cost" not in result, f"tokens are 0 but result contains 'cost': {result!r}"

    def test_missing_tool_context_returns_empty(self, _heavy_runtime):
        """Verify missing tool context returns empty."""
        ClawcodexREPL = _heavy_runtime

        class _BareStub:
            provider = None
            provider_name = "anthropic"
            tool_context = None
            _permission_mode = "default"
            _stats_turns = 0
            _stats_input_tokens = 0
            _stats_output_tokens = 0
            _shorten_path_text = staticmethod(lambda p: p)

        result = ClawcodexREPL._bottom_toolbar(_BareStub())
        assert result == "", f"expected empty string when tool_context is None, got {result!r}"

    def test_unknown_model_renders_without_crash(self, _heavy_runtime):
        """Verify unknown model renders without crash."""
        ClawcodexREPL = _heavy_runtime
        stub = self._make_stub(
            provider_name="anthropic",
            model="some-future-unknown-model-2099",
        )
        result = ClawcodexREPL._bottom_toolbar(stub)
        assert result, f"expected non-empty result for unknown model, got {result!r}"
        assert "anthropic" in result

    def test_task_progress_is_persistent_in_toolbar(self, _heavy_runtime):
        """Task progress remains visible without requiring another TaskList call."""

        ClawcodexREPL = _heavy_runtime
        stub = self._make_stub()
        stub.tool_context.tasks = {
            "T-1": {"id": "T-1", "status": "completed", "metadata": {}},
            "T-2": {"id": "T-2", "status": "in_progress", "metadata": {}},
            "T-3": {
                "id": "T-3",
                "status": "pending",
                "metadata": {},
                "lkb": {"derivedStatus": "blocked"},
            },
        }
        stub._task_toolbar_part = lambda: ClawcodexREPL._task_toolbar_part(stub)

        result = ClawcodexREPL._bottom_toolbar(stub)

        assert "tasks: 1/3" in result
        assert "1 running" in result
        assert "1 blocked" in result
