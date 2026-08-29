# ruff: noqa
# pylint: skip-file
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

"""Session macro MVP: parse/validate and overlay resolver."""

from __future__ import annotations

import time
import unittest
import unittest.mock
from dataclasses import FrozenInstanceError
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from clawcodex_ext.agent.tool_authoring.spec import AgentToolSpec
from clawcodex_ext.tool_system.build_tool import Tool, build_tool
from clawcodex_ext.tool_system.context import (
    RetrievalPlan,
    ToolContext,
    ToolUseOptions,
)
from clawcodex_ext.tool_system.registry import ToolRegistry
from extensions.sop_converter.bundle_context import (
    build_bundle_context,
    filter_tools_for_bundle,
)
from extensions.sop_converter.runtime.macros.errors import MacroConvertError
from extensions.sop_converter.runtime.macros.resolve_tool import resolve_tool_for_context
from extensions.sop_converter.runtime.macros.session import (
    SessionMacroOverlay,
    SessionMacroSnapshot,
    clear_session_macros_for_context,
    is_session_macro_tool,
    iter_effective_tools,
    mark_session_macro_tool,
    register_session_macro,
    sync_effective_tools,
)
from extensions.sop_converter.runtime.macros.session_parse import (
    parse_session_macro_definition,
    parse_session_macro_route,
)
from extensions.sop_converter.runtime.macros.validation import (
    ValidatedSessionMacro,
    validate_session_macro_definition,
)


def _make_tool(name: str) -> Tool:
    return build_tool(
        name=name,
        input_schema={"type": "object", "properties": {}},
        call=lambda _i, _c: None,
        prompt=name,
    )


def _empty_snapshot(
    *,
    owner_session_id: str,
    generation: int = 1,
    tools: dict[str, Tool] | None = None,
    covered_base_tools: dict[str, Tool] | None = None,
    success_timestamps: tuple[float, ...] = (),
) -> SessionMacroSnapshot:
    return SessionMacroSnapshot(
        owner_session_id=owner_session_id,
        generation=generation,
        definitions={},
        specs={},
        tools=tools or {},
        tool_specs={},
        routes=(),
        success_timestamps=success_timestamps,
        covered_base_tools=covered_base_tools or {},
    )


def _minimal_session_macro_dict(**overrides) -> dict:
    data = {
        "version": 1,
        "name": "session-demo-macro",
        "description": "A session macro for tests",
        "scope": "session",
        "enabled": True,
        "workflow": {
            "inputs": {
                "query": {"type": "string", "required": True},
            },
            "steps": [
                {
                    "id": "run",
                    "kind": "tool",
                    "callable_ref": "echo-tool",
                    "args": {"text": "$input.query"},
                }
            ],
            "outputs": {"result": "$steps.run.output"},
        },
        "routing": {
            "phrases": ["run session demo"],
            "keywords": ["demo"],
            "selection": "prefer",
            "priority": 100,
        },
        "provenance": {"kind": "session_nl"},
    }
    data.update(overrides)
    return data


class TestToolContextSessionMacroFields(unittest.TestCase):
    def test_tool_context_declares_session_macro_fields(self) -> None:
        ctx = ToolContext(workspace_root=".")
        self.assertIsNone(ctx.session_macro_overlay)
        self.assertIsNone(ctx.confirm_session_macro_plan)
        self.assertIs(ctx.allow_session_macro_registration, False)


class TestParseSessionMacroStrict(unittest.TestCase):
    def test_parse_rejects_unknown_top_level_field(self) -> None:
        data = _minimal_session_macro_dict()
        data["extra_field"] = "nope"
        with self.assertRaises(MacroConvertError) as ctx:
            parse_session_macro_definition(data)
        self.assertEqual(ctx.exception.error_code, "macro_unknown_field")
        self.assertEqual(ctx.exception.field, "extra_field")

    def test_parse_rejects_illegal_selection_enum(self) -> None:
        with self.assertRaises(MacroConvertError) as ctx:
            parse_session_macro_route({"selection": "maybe"}, default_target="session-demo-macro")
        self.assertEqual(ctx.exception.error_code, "macro_schema_invalid")
        self.assertEqual(ctx.exception.field, "routing.selection")

        data = _minimal_session_macro_dict()
        data["routing"] = {**data["routing"], "selection": "maybe"}
        with self.assertRaises(MacroConvertError) as ctx2:
            parse_session_macro_definition(data)
        self.assertEqual(ctx2.exception.error_code, "macro_schema_invalid")


class TestValidateSessionMacro(unittest.TestCase):
    def test_non_kebab_name_rejected(self) -> None:
        data = _minimal_session_macro_dict(name="Not_Kebab")
        macro = parse_session_macro_definition(data)
        with self.assertRaises(MacroConvertError) as ctx:
            validate_session_macro_definition(macro, tool_index={"echo-tool"})
        self.assertEqual(ctx.exception.error_code, "macro_name_invalid")

    def test_exclusive_selection_rejected(self) -> None:
        data = _minimal_session_macro_dict()
        data["routing"] = {
            **data["routing"],
            "selection": "exclusive",
            "verified": True,
            "intent_key": "demo",
            "covered_tools": ["echo-tool"],
        }
        macro = parse_session_macro_definition(data)
        with self.assertRaises(MacroConvertError) as ctx:
            validate_session_macro_definition(macro, tool_index={"echo-tool"})
        self.assertEqual(ctx.exception.error_code, "macro_selection_forbidden")

    def test_target_tool_mismatch_rejected(self) -> None:
        data = _minimal_session_macro_dict()
        data["routing"] = {**data["routing"], "target_tool": "other-name"}
        macro = parse_session_macro_definition(data)
        with self.assertRaises(MacroConvertError) as ctx:
            validate_session_macro_definition(macro, tool_index={"echo-tool"})
        self.assertEqual(ctx.exception.error_code, "macro_target_mismatch")

    def test_workflow_callable_forbidden(self) -> None:
        data = _minimal_session_macro_dict()
        data["workflow"] = {
            "inputs": {},
            "steps": [
                {
                    "id": "nested",
                    "kind": "tool",
                    "callable_ref": "other-session-macro",
                    "args": {},
                }
            ],
            "outputs": {"result": "$steps.nested.output"},
        }
        macro = parse_session_macro_definition(data)
        with self.assertRaises(MacroConvertError) as ctx:
            validate_session_macro_definition(
                macro,
                tool_index={"other-session-macro", "echo-tool"},
                forbid_workflow_tools={"other-session-macro"},
            )
        self.assertEqual(ctx.exception.error_code, "macro_step_forbidden")

    def test_validate_returns_validated_session_macro_with_normalized_target(self) -> None:
        data = _minimal_session_macro_dict()
        # empty target_tool → normalize to name
        data["routing"] = {k: v for k, v in data["routing"].items() if k != "target_tool"}
        macro = parse_session_macro_definition(data)
        result = validate_session_macro_definition(macro, tool_index={"echo-tool"})
        self.assertIsInstance(result, ValidatedSessionMacro)
        self.assertEqual(result.definition.name, "session-demo-macro")
        self.assertEqual(result.definition.routing.target_tool, "session-demo-macro")
        self.assertEqual(result.definition.routing.selection, "prefer")
        self.assertEqual(result.definition.scope, "session")
        self.assertEqual(result.workflow.name, "session-demo-macro")
        self.assertEqual(result.tool_spec.name, "session-demo-macro")
        self.assertEqual(result.tool_spec.call_type, "workflow")
        self.assertEqual(
            result.tool_spec.call_impl,
            {"catalog_id": "session:session-demo-macro"},
        )


class TestResolveToolForContext(unittest.TestCase):
    def test_resolver_prefers_overlay_over_registry_and_options(self) -> None:
        name = "session-demo-macro"
        overlay_tool = mark_session_macro_tool(_make_tool(name))
        registry_tool = _make_tool(name)
        options_tool = _make_tool(name)

        overlay = SessionMacroOverlay()
        overlay.commit(
            _empty_snapshot(
                owner_session_id="sess-1",
                tools={name: overlay_tool},
            )
        )
        registry = ToolRegistry([registry_tool])
        ctx = ToolContext(
            workspace_root=".",
            session_id="sess-1",
            session_macro_overlay=overlay,
            options=ToolUseOptions(tools=[options_tool]),
        )

        resolved = resolve_tool_for_context(ctx, name, base_registry=registry)
        self.assertIs(resolved, overlay_tool)
        self.assertIsNot(resolved, registry_tool)
        self.assertIsNot(resolved, options_tool)

    def test_snapshot_cow_replace_is_atomic(self) -> None:
        overlay = SessionMacroOverlay()
        tool_a = mark_session_macro_tool(_make_tool("macro-a"))
        tool_b = mark_session_macro_tool(_make_tool("macro-b"))

        first = _empty_snapshot(
            owner_session_id="sess-1",
            generation=1,
            tools={"macro-a": tool_a},
        )
        overlay.commit(first)
        held = overlay.read()
        self.assertIsNotNone(held)
        assert held is not None
        self.assertEqual(held.generation, 1)
        self.assertIn("macro-a", held.tools)
        self.assertNotIn("macro-b", held.tools)

        second = _empty_snapshot(
            owner_session_id="sess-1",
            generation=2,
            tools={"macro-b": tool_b},
        )
        overlay.commit(second)
        latest = overlay.read()
        self.assertIsNotNone(latest)
        assert latest is not None

        # Readers keep the prior immutable snapshot; commit replaces wholesale.
        self.assertIsNot(held, latest)
        self.assertEqual(held.generation, 1)
        self.assertIn("macro-a", held.tools)
        self.assertNotIn("macro-b", held.tools)
        self.assertEqual(latest.generation, 2)
        self.assertIn("macro-b", latest.tools)
        self.assertNotIn("macro-a", latest.tools)

        with self.assertRaises(FrozenInstanceError):
            held.generation = 99  # type: ignore[misc]

    def test_resolver_ignores_overlay_when_owner_session_mismatches(self) -> None:
        name = "session-demo-macro"
        overlay_tool = mark_session_macro_tool(_make_tool(name))
        stale_options_tool = mark_session_macro_tool(_make_tool(name))
        base_tool = _make_tool(name)

        overlay = SessionMacroOverlay()
        overlay.commit(
            _empty_snapshot(
                owner_session_id="old-sess",
                tools={name: overlay_tool},
            )
        )
        registry = ToolRegistry([base_tool])
        ctx = ToolContext(
            workspace_root=".",
            session_id="new-sess",
            session_macro_overlay=overlay,
            options=ToolUseOptions(tools=[stale_options_tool]),
        )

        resolved = resolve_tool_for_context(ctx, name, base_registry=registry)
        self.assertIs(resolved, base_tool)
        self.assertIsNot(resolved, overlay_tool)
        self.assertIsNot(resolved, stale_options_tool)
        self.assertTrue(is_session_macro_tool(overlay_tool))
        self.assertTrue(is_session_macro_tool(stale_options_tool))
        self.assertFalse(is_session_macro_tool(base_tool))


class TestSessionMacroCleanupAndPool(unittest.TestCase):
    def test_session_switch_removes_overlay_tools_from_options_but_restores_covered_base(
        self,
    ) -> None:
        name = "covered-echo"
        base_tool = _make_tool(name)
        session_tool = mark_session_macro_tool(_make_tool(name))
        other = _make_tool("keep-me")

        overlay = SessionMacroOverlay()
        overlay.commit(
            _empty_snapshot(
                owner_session_id="sess-old",
                tools={name: session_tool},
                covered_base_tools={name: base_tool},
                success_timestamps=(1.0, 2.0),
            )
        )
        ctx = ToolContext(
            workspace_root=".",
            session_id="sess-old",
            session_macro_overlay=overlay,
            options=ToolUseOptions(tools=[session_tool, other]),
        )

        clear_session_macros_for_context(ctx)

        self.assertIsNone(overlay.read())
        tools = list(ctx.options.tools or [])
        self.assertNotIn(session_tool, tools)
        self.assertFalse(any(is_session_macro_tool(t) for t in tools))
        self.assertIn(other, tools)
        self.assertIn(base_tool, tools)
        self.assertIs(tools[tools.index(base_tool)], base_tool)
        # Old overlay macro must not remain callable; covered base restored by identity.
        resolved = resolve_tool_for_context(ctx, name)
        self.assertIs(resolved, base_tool)
        self.assertIsNot(resolved, session_tool)

    def test_session_switch_clears_retrieval_hidden_and_plan(self) -> None:
        name = "session-hidden-macro"
        session_tool = mark_session_macro_tool(_make_tool(name))
        base_hidden = _make_tool("atomic-hidden")

        overlay = SessionMacroOverlay()
        overlay.commit(
            _empty_snapshot(
                owner_session_id="sess-1",
                tools={name: session_tool},
            )
        )
        ctx = ToolContext(
            workspace_root=".",
            session_id="sess-1",
            session_macro_overlay=overlay,
            options=ToolUseOptions(tools=[session_tool]),
            retrieval_hidden_tools=[session_tool, base_hidden],
            retrieval_suppressed_tools={name, "atomic-hidden"},
            retrieval_plan=RetrievalPlan(
                query="q",
                selected_macros=[name],
                suppressed_tools=[name, "atomic-hidden"],
            ),
        )

        clear_session_macros_for_context(ctx)

        self.assertIsNone(overlay.read())
        self.assertFalse(any(is_session_macro_tool(t) for t in ctx.retrieval_hidden_tools))
        self.assertIn(base_hidden, ctx.retrieval_hidden_tools)
        self.assertIsNone(ctx.retrieval_plan)
        self.assertEqual(ctx.retrieval_suppressed_tools, set())

    def test_restore_retrieval_tools_does_not_revive_foreign_session_macros(self) -> None:
        foreign = mark_session_macro_tool(_make_tool("foreign-session-macro"))
        atomic = _make_tool("atomic-tool")
        ctx = ToolContext(
            workspace_root=".",
            session_id="sess-new",
            options=ToolUseOptions(tools=[]),
            retrieval_hidden_tools=[foreign, atomic],
            retrieval_suppressed_tools={"foreign-session-macro", "atomic-tool"},
            retrieval_plan=RetrievalPlan(query="q", selected_macros=["foreign-session-macro"]),
        )

        ctx.restore_retrieval_tools()

        tools = list(ctx.options.tools or [])
        self.assertIn(atomic, tools)
        self.assertNotIn(foreign, tools)
        self.assertFalse(any(is_session_macro_tool(t) for t in tools))
        self.assertEqual(ctx.retrieval_hidden_tools, [])
        self.assertIsNone(ctx.retrieval_plan)

    def test_bundle_filter_keeps_session_macros(self) -> None:
        session_tool = mark_session_macro_tool(_make_tool("session-demo-macro"))
        foreign_deferred = build_tool(
            name="foreign-bundle-tool",
            input_schema={"type": "object", "properties": {}},
            call=lambda _i, _c: None,
            prompt="foreign",
            should_defer=True,
        )
        read_tool = _make_tool("Read")
        bundle = build_bundle_context(
            bundle_path=Path("/tmp/bundle-a"),
            skill_names=[],
            skill_dirs=[],
            tool_names=["some-bundle-tool"],
        )

        filtered = filter_tools_for_bundle(
            [read_tool, foreign_deferred, session_tool],
            bundle,
        )
        names = {tool.name for tool in filtered}
        self.assertIn("Read", names)
        self.assertIn("session-demo-macro", names)
        self.assertNotIn("foreign-bundle-tool", names)
        self.assertIn(session_tool, filtered)

    def test_bundle_filter_keeps_session_macro_register_tools(self) -> None:
        """Register/promote tools must survive --agent bundle isolation."""
        register_tool = _make_tool("register-macro-workflow")
        from_trace_tool = _make_tool("register-macro-from-trace")
        promote_tool = _make_tool("promote-macro-workflow")
        foreign = build_tool(
            name="foreign-bundle-tool",
            input_schema={"type": "object", "properties": {}},
            call=lambda _i, _c: None,
            prompt="foreign",
            should_defer=True,
        )
        bundle = build_bundle_context(
            bundle_path=Path("/tmp/adf-bundle"),
            skill_names=[],
            skill_dirs=[],
            tool_names=["skills-skill-handlers-list-operations"],
        )

        filtered = filter_tools_for_bundle(
            [register_tool, from_trace_tool, promote_tool, foreign],
            bundle,
        )
        names = {tool.name for tool in filtered}
        self.assertIn("register-macro-workflow", names)
        self.assertIn("register-macro-from-trace", names)
        self.assertIn("promote-macro-workflow", names)
        self.assertNotIn("foreign-bundle-tool", names)

    def test_iter_effective_tools_merges_overlay_over_base(self) -> None:
        name = "session-demo-macro"
        base = _make_tool("bash")
        covered = _make_tool(name)
        overlay_tool = mark_session_macro_tool(_make_tool(name))
        overlay = SessionMacroOverlay()
        overlay.commit(
            _empty_snapshot(
                owner_session_id="sess-1",
                tools={name: overlay_tool},
                covered_base_tools={name: covered},
            )
        )
        ctx = ToolContext(
            workspace_root=".",
            session_id="sess-1",
            session_macro_overlay=overlay,
            options=ToolUseOptions(tools=[base, covered]),
        )

        effective = iter_effective_tools(ctx, [base, covered])
        self.assertIn(base, effective)
        self.assertIn(overlay_tool, effective)
        self.assertNotIn(covered, effective)

        sync_effective_tools(ctx)
        self.assertIn(base, ctx.options.tools)
        self.assertIn(overlay_tool, ctx.options.tools)
        self.assertNotIn(covered, ctx.options.tools)


def _create_tool_from_spec(spec: AgentToolSpec) -> Tool:
    return mark_session_macro_tool(
        build_tool(
            name=spec.name,
            input_schema=dict(spec.input_schema or {"type": "object", "properties": {}}),
            call=lambda _i, _c: {"ok": True},
            prompt=spec.description or spec.name,
        )
    )


def _register_ctx(
    *,
    allow: bool = True,
    confirm=None,
    session_id: str = "sess-1",
    overlay: SessionMacroOverlay | None = None,
    tools: list[Tool] | None = None,
) -> ToolContext:
    if confirm is None:

        def _always_confirm(_plan: object) -> bool:
            return True

        confirm = _always_confirm
    return ToolContext(
        workspace_root=".",
        session_id=session_id,
        session_macro_overlay=overlay if overlay is not None else SessionMacroOverlay(),
        allow_session_macro_registration=allow,
        confirm_session_macro_plan=confirm,
        options=ToolUseOptions(tools=list(tools or [_make_tool("echo-tool")])),
    )


class TestRegisterSessionMacro(unittest.TestCase):
    def test_capability_gate_blocks_even_when_interactive_confirm_would_pass(self) -> None:
        ctx = _register_ctx(allow=False, confirm=lambda _p: True)
        with self.assertRaises(MacroConvertError) as raised:
            register_session_macro(
                ctx,
                _minimal_session_macro_dict(),
                replace=False,
                tool_index={"echo-tool"},
                workflow_tool_names=set(),
                protected_builtin_exclusive_targets=set(),
                create_tool=_create_tool_from_spec,
            )
        self.assertEqual(raised.exception.error_code, "macro_capability_denied")
        self.assertIsNone(ctx.session_macro_overlay.read())

    def test_confirm_false_writes_nothing(self) -> None:
        ctx = _register_ctx(confirm=lambda _p: False)
        with self.assertRaises(MacroConvertError) as raised:
            register_session_macro(
                ctx,
                _minimal_session_macro_dict(),
                replace=False,
                tool_index={"echo-tool"},
                workflow_tool_names=set(),
                protected_builtin_exclusive_targets=set(),
                create_tool=_create_tool_from_spec,
            )
        self.assertEqual(raised.exception.error_code, "macro_registration_denied")
        self.assertIsNone(ctx.session_macro_overlay.read())

    def test_confirm_exception_writes_nothing(self) -> None:
        def _boom(_plan):
            raise RuntimeError("ui failed")

        ctx = _register_ctx(confirm=_boom)
        with self.assertRaises(MacroConvertError) as raised:
            register_session_macro(
                ctx,
                _minimal_session_macro_dict(),
                replace=False,
                tool_index={"echo-tool"},
                workflow_tool_names=set(),
                protected_builtin_exclusive_targets=set(),
                create_tool=_create_tool_from_spec,
            )
        self.assertEqual(raised.exception.error_code, "macro_registration_denied")
        self.assertIsNone(ctx.session_macro_overlay.read())

    def test_session_id_changes_during_confirm_aborts(self) -> None:
        overlay = SessionMacroOverlay()
        holder: dict = {}

        def _flip(_plan):
            holder["ctx"].session_id = "sess-other"
            return True

        ctx = _register_ctx(confirm=_flip, overlay=overlay, session_id="sess-1")
        holder["ctx"] = ctx
        with self.assertRaises(MacroConvertError) as raised:
            register_session_macro(
                ctx,
                _minimal_session_macro_dict(),
                replace=False,
                tool_index={"echo-tool"},
                workflow_tool_names=set(),
                protected_builtin_exclusive_targets=set(),
                create_tool=_create_tool_from_spec,
            )
        self.assertEqual(raised.exception.error_code, "macro_stale_session")
        self.assertIsNone(overlay.read())

    def test_create_tool_failure_no_partial_commit(self) -> None:
        def _fail(_spec: AgentToolSpec) -> Tool:
            raise RuntimeError("factory boom")

        ctx = _register_ctx()
        with self.assertRaises(MacroConvertError) as raised:
            register_session_macro(
                ctx,
                _minimal_session_macro_dict(),
                replace=False,
                tool_index={"echo-tool"},
                workflow_tool_names=set(),
                protected_builtin_exclusive_targets=set(),
                create_tool=_fail,
            )
        self.assertEqual(raised.exception.error_code, "macro_create_tool_failed")
        self.assertIsNone(ctx.session_macro_overlay.read())

    def test_rate_limit_and_macro_count_and_size_limits(self) -> None:
        # Size: >64KiB payload
        huge = _minimal_session_macro_dict(description="x" * (65 * 1024))
        ctx = _register_ctx()
        with self.assertRaises(MacroConvertError) as raised:
            register_session_macro(
                ctx,
                huge,
                replace=False,
                tool_index={"echo-tool"},
                workflow_tool_names=set(),
                protected_builtin_exclusive_targets=set(),
                create_tool=_create_tool_from_spec,
            )
        self.assertEqual(raised.exception.error_code, "macro_definition_too_large")

        # Rate: 5 successes in window already recorded
        now = time.monotonic()
        overlay = SessionMacroOverlay()
        overlay.commit(
            SessionMacroSnapshot(
                owner_session_id="sess-1",
                generation=1,
                success_timestamps=tuple(now - i for i in range(5)),
            )
        )
        ctx = _register_ctx(overlay=overlay)
        with self.assertRaises(MacroConvertError) as raised2:
            register_session_macro(
                ctx,
                _minimal_session_macro_dict(),
                replace=False,
                tool_index={"echo-tool"},
                workflow_tool_names=set(),
                protected_builtin_exclusive_targets=set(),
                create_tool=_create_tool_from_spec,
            )
        self.assertEqual(raised2.exception.error_code, "macro_registration_rate_limited")

        # Count: 32 macros already present
        defs = {}
        for i in range(32):
            name = f"macro-{i}"
            defs[name] = parse_session_macro_definition(_minimal_session_macro_dict(name=name))
        overlay2 = SessionMacroOverlay()
        overlay2.commit(
            SessionMacroSnapshot(
                owner_session_id="sess-1",
                generation=1,
                definitions=defs,
            )
        )
        ctx = _register_ctx(overlay=overlay2)
        with self.assertRaises(MacroConvertError) as raised3:
            register_session_macro(
                ctx,
                _minimal_session_macro_dict(name="macro-extra"),
                replace=False,
                tool_index={"echo-tool"},
                workflow_tool_names=set(),
                protected_builtin_exclusive_targets=set(),
                create_tool=_create_tool_from_spec,
            )
        self.assertEqual(raised3.exception.error_code, "macro_session_limit_exceeded")

        # String caps: name / route phrase / description char limits
        from extensions.sop_converter.runtime.macros.session import (
            MAX_DESCRIPTION_CHARS,
            MAX_STRING_CHARS,
        )

        long_name = "a" * (MAX_STRING_CHARS + 1)
        with self.assertRaises(MacroConvertError) as raised4:
            register_session_macro(
                _register_ctx(),
                _minimal_session_macro_dict(name=long_name),
                replace=False,
                tool_index={"echo-tool"},
                workflow_tool_names=set(),
                protected_builtin_exclusive_targets=set(),
                create_tool=_create_tool_from_spec,
            )
        self.assertEqual(raised4.exception.error_code, "macro_definition_too_large")

        long_phrase = "p" * (MAX_STRING_CHARS + 1)
        with self.assertRaises(MacroConvertError) as raised5:
            register_session_macro(
                _register_ctx(),
                _minimal_session_macro_dict(
                    routing={
                        "phrases": [long_phrase],
                        "keywords": ["demo"],
                        "selection": "prefer",
                        "priority": 100,
                        "target_tool": "session-demo-macro",
                    }
                ),
                replace=False,
                tool_index={"echo-tool"},
                workflow_tool_names=set(),
                protected_builtin_exclusive_targets=set(),
                create_tool=_create_tool_from_spec,
            )
        self.assertEqual(raised5.exception.error_code, "macro_definition_too_large")

        long_desc = "d" * (MAX_DESCRIPTION_CHARS + 1)
        with self.assertRaises(MacroConvertError) as raised6:
            register_session_macro(
                _register_ctx(),
                _minimal_session_macro_dict(description=long_desc),
                replace=False,
                tool_index={"echo-tool"},
                workflow_tool_names=set(),
                protected_builtin_exclusive_targets=set(),
                create_tool=_create_tool_from_spec,
            )
        self.assertEqual(raised6.exception.error_code, "macro_definition_too_large")

    def test_bypass_permissions_without_confirm_denies(self) -> None:
        from clawcodex_ext.permissions.types import ToolPermissionContext

        ctx = ToolContext(
            workspace_root=".",
            session_id="sess-1",
            session_macro_overlay=SessionMacroOverlay(),
            allow_session_macro_registration=True,
            confirm_session_macro_plan=None,
            permission_context=ToolPermissionContext(mode="bypassPermissions"),
            options=ToolUseOptions(tools=[_make_tool("echo-tool")]),
        )
        with self.assertRaises(MacroConvertError) as raised:
            register_session_macro(
                ctx,
                _minimal_session_macro_dict(),
                replace=False,
                tool_index={"echo-tool"},
                workflow_tool_names=set(),
                protected_builtin_exclusive_targets=set(),
                create_tool=_create_tool_from_spec,
            )
        self.assertEqual(raised.exception.error_code, "macro_registration_denied")
        self.assertIsNone(ctx.session_macro_overlay.read())

    def test_protected_builtin_exclusive_target_conflict(self) -> None:
        ctx = _register_ctx()
        with self.assertRaises(MacroConvertError) as raised:
            register_session_macro(
                ctx,
                _minimal_session_macro_dict(name="invoke-existing-agent"),
                replace=False,
                tool_index={"echo-tool"},
                workflow_tool_names=set(),
                protected_builtin_exclusive_targets={"invoke-existing-agent"},
                create_tool=_create_tool_from_spec,
            )
        self.assertEqual(raised.exception.error_code, "macro_route_conflict")
        self.assertIsNone(ctx.session_macro_overlay.read())

    def test_concurrent_replace_uses_generation_check(self) -> None:
        overlay = SessionMacroOverlay()
        first = register_session_macro(
            _register_ctx(overlay=overlay),
            _minimal_session_macro_dict(),
            replace=False,
            tool_index={"echo-tool"},
            workflow_tool_names=set(),
            protected_builtin_exclusive_targets=set(),
            create_tool=_create_tool_from_spec,
        )
        self.assertTrue(first["registered"])
        gen_before = overlay.read().generation

        def _race(_plan):
            # Simulate another commit during confirm (generation bump).
            snap = overlay.read()
            assert snap is not None
            overlay.commit(
                SessionMacroSnapshot(
                    owner_session_id=snap.owner_session_id,
                    generation=snap.generation + 1,
                    definitions=dict(snap.definitions),
                    specs=dict(snap.specs),
                    tools=dict(snap.tools),
                    tool_specs=dict(snap.tool_specs),
                    routes=tuple(snap.routes),
                    success_timestamps=tuple(snap.success_timestamps),
                    covered_base_tools=dict(snap.covered_base_tools),
                )
            )
            return True

        ctx = _register_ctx(overlay=overlay, confirm=_race)
        with self.assertRaises(MacroConvertError) as raised:
            register_session_macro(
                ctx,
                _minimal_session_macro_dict(
                    description="replaced during race",
                ),
                replace=True,
                tool_index={"echo-tool"},
                workflow_tool_names=set(),
                protected_builtin_exclusive_targets=set(),
                create_tool=_create_tool_from_spec,
            )
        self.assertEqual(raised.exception.error_code, "macro_concurrent_modification")
        self.assertEqual(overlay.read().generation, gen_before + 1)

    def test_concurrent_create_uses_generation_check(self) -> None:
        overlay = SessionMacroOverlay()
        # Seed an unrelated macro so generation is non-zero before the race.
        seed = register_session_macro(
            _register_ctx(overlay=overlay),
            _minimal_session_macro_dict(name="seed-macro"),
            replace=False,
            tool_index={"echo-tool"},
            workflow_tool_names=set(),
            protected_builtin_exclusive_targets=set(),
            create_tool=_create_tool_from_spec,
        )
        self.assertTrue(seed["registered"])
        gen_before = overlay.read().generation

        def _race(_plan):
            snap = overlay.read()
            assert snap is not None
            overlay.commit(
                SessionMacroSnapshot(
                    owner_session_id=snap.owner_session_id,
                    generation=snap.generation + 1,
                    definitions=dict(snap.definitions),
                    specs=dict(snap.specs),
                    tools=dict(snap.tools),
                    tool_specs=dict(snap.tool_specs),
                    routes=tuple(snap.routes),
                    success_timestamps=tuple(snap.success_timestamps),
                    covered_base_tools=dict(snap.covered_base_tools),
                )
            )
            return True

        ctx = _register_ctx(overlay=overlay, confirm=_race)
        with self.assertRaises(MacroConvertError) as raised:
            register_session_macro(
                ctx,
                _minimal_session_macro_dict(name="raced-create"),
                replace=False,
                tool_index={"echo-tool"},
                workflow_tool_names=set(),
                protected_builtin_exclusive_targets=set(),
                create_tool=_create_tool_from_spec,
            )
        self.assertEqual(raised.exception.error_code, "macro_concurrent_modification")
        self.assertEqual(overlay.read().generation, gen_before + 1)
        self.assertNotIn("raced-create", overlay.read().definitions)


class TestWireCallPathsToResolver(unittest.IsolatedAsyncioTestCase):
    """Task 5: every lookup path prefers resolve_tool_for_context / overlay."""

    async def test_main_tool_execution_uses_overlay_macro(self) -> None:
        from types import SimpleNamespace

        from clawcodex_ext.services.tool_execution.tool_execution import run_tool_use
        from src.types.messages import AssistantMessage

        name = "session-demo-macro"
        overlay_tool = mark_session_macro_tool(
            build_tool(
                name=name,
                input_schema={"type": "object", "properties": {}},
                call=lambda _i, _c: {"source": "overlay"},
                prompt=name,
            )
        )
        stale = build_tool(
            name=name,
            input_schema={"type": "object", "properties": {}},
            call=lambda _i, _c: {"source": "stale-options"},
            prompt=name,
        )
        overlay = SessionMacroOverlay()
        overlay.commit(_empty_snapshot(owner_session_id="sess-1", tools={name: overlay_tool}))
        ctx = ToolContext(
            workspace_root=".",
            session_id="sess-1",
            session_macro_overlay=overlay,
            options=ToolUseOptions(tools=[stale]),
        )

        async def _allow(*_a, **_k):
            return {"behavior": "allow"}

        tool_use = SimpleNamespace(name=name, id="tu-1", input={})
        updates = [u async for u in run_tool_use(tool_use, AssistantMessage(), _allow, ctx)]
        self.assertTrue(updates)
        # Overlay call wins over the same-named stale options tool.
        last = updates[-1]
        payload = getattr(last.message, "toolUseResult", None)
        self.assertIsNotNone(payload)
        if isinstance(payload, dict):
            self.assertEqual(payload.get("source"), "overlay")
        else:
            self.assertIn("overlay", str(payload))

    def test_execute_tool_resolves_overlay_not_stale_base(self) -> None:
        from clawcodex_ext.tool_system.tools.execute import _resolve_target

        name = "session-demo-macro"
        overlay_tool = mark_session_macro_tool(_make_tool(name))
        stale_base = _make_tool(name)
        overlay = SessionMacroOverlay()
        overlay.commit(_empty_snapshot(owner_session_id="sess-1", tools={name: overlay_tool}))
        registry = ToolRegistry([stale_base])
        ctx = ToolContext(
            workspace_root=".",
            session_id="sess-1",
            session_macro_overlay=overlay,
            tool_registry=registry,
            options=ToolUseOptions(tools=[stale_base]),
        )
        resolved = _resolve_target(name, ctx)
        self.assertIs(resolved, overlay_tool)
        self.assertIsNot(resolved, stale_base)

    def test_main_and_execute_paths_resolve_same_overlay_tool(self) -> None:
        """Main query resolver and Execute `_resolve_target` must share identity."""
        from clawcodex_ext.tool_system.tools.execute import _resolve_target

        name = "session-demo-macro"
        overlay_tool = mark_session_macro_tool(_make_tool(name))
        stale = _make_tool(name)
        overlay = SessionMacroOverlay()
        overlay.commit(_empty_snapshot(owner_session_id="sess-1", tools={name: overlay_tool}))
        registry = ToolRegistry([stale])
        ctx = ToolContext(
            workspace_root=".",
            session_id="sess-1",
            session_macro_overlay=overlay,
            tool_registry=registry,
            options=ToolUseOptions(tools=[stale]),
        )
        via_main = resolve_tool_for_context(ctx, name, base_registry=registry)
        via_execute = _resolve_target(name, ctx)
        self.assertIs(via_main, overlay_tool)
        self.assertIs(via_execute, overlay_tool)
        self.assertIs(via_main, via_execute)
