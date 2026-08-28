#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# -------------------------------------------------------------------------
# This file is part of the AgentSDK project.
# Copyright (c) 2026 Huawei Technologies Co.,Ltd.
# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# -------------------------------------------------------------------------
"""Unit tests for :mod:`extensions.sop_converter.core.dependency`.

Covers:

* Pairing heuristics (build ↔ invoke) for both positive and negative
  cases
* ``extract_shared_params`` fallbacks when the build op has no
  obvious return key
* ``detect_lifecycle_patterns`` building a complete
  :class:`ToolDependencyGraph` from synthetic components
* Writer emits valid YAML, atomic, idempotent
* Reader is tolerant of corruption and missing files
* ``merge_overrides`` replaces same-keyed intent groups and
  de-duplicates same-paired dependencies
* Wiring: ``register_component_tools`` produces a yaml on disk when
  ``bundle_dir`` is set
"""

from __future__ import annotations

import tempfile
import textwrap
import unittest
from pathlib import Path

from extensions.sop_converter.core.dependency import (
    load_tool_dependencies,
    write_tool_dependencies,
)
from extensions.sop_converter.core.dependency.detector import detect_lifecycle_patterns
from extensions.sop_converter.core.dependency.heuristics import (
    extract_shared_params,
    pair_build_invoke,
)
from extensions.sop_converter.core.dependency.models import (
    HiddenStep,
    IntentGroup,
    PriorityRoute,
    ToolDependency,
    ToolDependencyGraph,
)
from extensions.sop_converter.core.dependency.reader import parse_graph_payload
from extensions.sop_converter.source_parser import ParamSpec, SourceComponent, SourceOperation


def _op(
    name: str,
    *,
    class_name: str = "Ops",
    params: list[ParamSpec] | None = None,
    return_type: str | None = "Dict[str, Any]",
) -> SourceOperation:
    return SourceOperation(
        name=name,
        description=f"op {name}",
        parameters=params or [],
        return_type=return_type,
        class_name=class_name,
        file_stem="ops",
    )


def _param(name: str, *, required: bool = True, type_hint: str = "str") -> ParamSpec:
    return ParamSpec(name=name, type_hint=type_hint, required=required)


def _component(ops: list[SourceOperation], name: str = "Comp") -> SourceComponent:
    return SourceComponent(name=name, file_path="x.py", description="comp", operations=ops)


class TestPairingHeuristics(unittest.TestCase):
    """``pair_build_invoke`` + ``extract_shared_params`` smoke tests."""

    def test_no_ops_returns_empty(self) -> None:
        self.assertEqual(pair_build_invoke([]), [])

    def test_only_builds_no_pair(self) -> None:
        ops = [_op("build_agent"), _op("create_team_session")]
        self.assertEqual(pair_build_invoke(ops), [])

    def test_only_invokes_no_pair(self) -> None:
        ops = [_op("run_agent", params=[_param("agent_id")])]
        self.assertEqual(pair_build_invoke(ops), [])

    def test_build_agent_pairs_with_run_agent(self) -> None:
        ops = [
            _op("build_agent"),
            _op("run_agent", params=[_param("agent_id")]),
        ]
        pairs = pair_build_invoke(ops)
        self.assertEqual(len(pairs), 1)
        build, invoke, shared = pairs[0]
        self.assertEqual(build.name, "build_agent")
        self.assertEqual(invoke.name, "run_agent")
        self.assertIn("agent_id", shared)

    def test_create_session_pairs_with_run_session(self) -> None:
        """``create_session`` returns ``session_id``; ``run_session`` consumes it."""
        ops = [
            _op("create_session"),
            _op("run_session", params=[_param("session_id")]),
        ]
        pairs = pair_build_invoke(ops)
        self.assertEqual(len(pairs), 1)
        self.assertEqual(pairs[0][0].name, "create_session")
        self.assertIn("session_id", pairs[0][2])

    def test_invoke_without_id_param_does_not_pair(self) -> None:
        ops = [
            _op("build_agent"),
            _op("run_thing", params=[_param("name")]),
        ]
        self.assertEqual(pair_build_invoke(ops), [])

    def test_extract_shared_params_fallback_to_build_keys(self) -> None:
        build = _op("build_agent")
        invoke = _op("call_agent", params=[_param("agent_id")])
        shared = extract_shared_params(build, invoke)
        self.assertIn("agent_id", shared)


class TestDetectLifecyclePatterns(unittest.TestCase):
    """``detect_lifecycle_patterns`` end-to-end behaviour."""

    def test_no_components_returns_empty_graph(self) -> None:
        g = detect_lifecycle_patterns([])
        self.assertTrue(g.is_empty())

    def test_full_agent_lifecycle(self) -> None:
        comp = _component(
            [
                _op("build_agent"),
                _op("run_agent", params=[_param("agent_id")]),
                _op("invoke_agent", params=[_param("agent_id")]),
            ]
        )
        g = detect_lifecycle_patterns([comp])
        # Two pairs (build→run, build→invoke)
        self.assertEqual(len(g.dependencies), 2)
        # All three tools share the agent_lifecycle group
        group = g.get_intent_group("agent_lifecycle")
        self.assertIsNotNone(group)
        assert group is not None
        self.assertEqual(group.primary_entry, "comp.build-agent")
        self.assertIn("comp.run-agent", group.tools)
        # Priority routes: at least one create, one invoke
        routes = [r for r in g.priority_routes if r.intent_group == "agent_lifecycle"]
        self.assertTrue(any(r.entry_first for r in routes))
        self.assertTrue(any(not r.entry_first for r in routes))
        # Hidden steps attached to the "create → invoke" lifecycle
        create_routes = [d for d in g.dependencies if d.lifecycle == "create → invoke"]
        self.assertTrue(create_routes, "expected create→invoke dependency routes")
        self.assertTrue(all(d.hidden_steps for d in create_routes))

    def test_session_lifecycle(self) -> None:
        comp = _component(
            [
                _op("create_session"),
                _op("run_session", params=[_param("session_id")]),
            ]
        )
        g = detect_lifecycle_patterns([comp])
        self.assertEqual(len(g.dependencies), 1)
        self.assertIsNotNone(g.get_intent_group("session_lifecycle"))

    def test_spec_lifecycle(self) -> None:
        comp = _component(
            [
                _op("load_spec"),
                _op("run_spec", params=[_param("spec_id")]),
            ]
        )
        g = detect_lifecycle_patterns([comp])
        self.assertEqual(len(g.dependencies), 1)
        dep = g.dependencies[0]
        self.assertEqual(dep.lifecycle, "prepare → execute")
        # The "prepare → execute" template attaches a load_spec step
        self.assertTrue(any(s.action == "load_spec" for s in dep.hidden_steps))

    def test_isolated_ops_yield_empty_graph(self) -> None:
        comp = _component(
            [
                _op("unrelated_a"),
                _op("unrelated_b", params=[_param("x")]),
            ]
        )
        g = detect_lifecycle_patterns([comp])
        self.assertEqual(g.dependencies, [])
        self.assertEqual(g.intent_groups, [])

    def test_deterministic_output(self) -> None:
        comp = _component(
            [
                _op("build_agent"),
                _op("run_agent", params=[_param("agent_id")]),
            ]
        )
        g1 = detect_lifecycle_patterns([comp])
        g2 = detect_lifecycle_patterns([comp])
        self.assertEqual(g1.to_dict(), g2.to_dict())


class TestGraphMutation(unittest.TestCase):
    """Round-trip + merge behaviour for :class:`ToolDependencyGraph`."""

    def test_round_trip_dict(self) -> None:
        graph = ToolDependencyGraph(
            version=1,
            dependencies=[
                ToolDependency(
                    from_tool="a",
                    to_tool="b",
                    shared_params=["x"],
                    hidden_steps=[HiddenStep(action="s", description="d")],
                    lifecycle="create → invoke",
                )
            ],
            intent_groups=[
                IntentGroup(
                    name="ig",
                    description="desc",
                    tools=["a", "b"],
                    primary_entry="a",
                )
            ],
            priority_routes=[PriorityRoute(keywords=["k"], intent_group="ig", entry_first=True)],
        )
        g2 = ToolDependencyGraph.from_dict(graph.to_dict())
        self.assertEqual(g2.dependencies[0].from_tool, "a")
        self.assertEqual(g2.dependencies[0].hidden_steps[0].action, "s")
        self.assertEqual(g2.intent_groups[0].name, "ig")
        self.assertEqual(g2.priority_routes[0].intent_group, "ig")

    def test_merge_overrides_replaces_dependencies(self) -> None:
        base = ToolDependencyGraph(
            dependencies=[
                ToolDependency(from_tool="a", to_tool="b"),
                ToolDependency(from_tool="a", to_tool="c"),
            ]
        )
        override = ToolDependencyGraph(
            dependencies=[
                ToolDependency(from_tool="a", to_tool="b", lifecycle="new"),
            ]
        )
        base.merge_overrides(override)
        self.assertEqual(len(base.dependencies), 2)
        ab = next(d for d in base.dependencies if (d.from_tool, d.to_tool) == ("a", "b"))
        self.assertEqual(ab.lifecycle, "new")

    def test_merge_overrides_replaces_intent_group_by_name(self) -> None:
        base = ToolDependencyGraph(intent_groups=[IntentGroup(name="ig", description="old", tools=["a"])])
        override = ToolDependencyGraph(intent_groups=[IntentGroup(name="ig", description="new", tools=["a", "b"])])
        base.merge_overrides(override)
        self.assertEqual(len(base.intent_groups), 1)
        self.assertEqual(base.intent_groups[0].description, "new")
        self.assertEqual(base.intent_groups[0].tools, ["a", "b"])

    def test_merge_overrides_appends_routes(self) -> None:
        base = ToolDependencyGraph(priority_routes=[PriorityRoute(keywords=["k1"], intent_group="ig")])
        override = ToolDependencyGraph(priority_routes=[PriorityRoute(keywords=["k2"], intent_group="ig")])
        base.merge_overrides(override)
        self.assertEqual(len(base.priority_routes), 2)

    def test_merge_overrides_none_is_noop(self) -> None:
        base = ToolDependencyGraph(dependencies=[ToolDependency(from_tool="a", to_tool="b")])
        base.merge_overrides(None)
        self.assertEqual(len(base.dependencies), 1)


class TestWriterReaderRoundTrip(unittest.TestCase):
    """End-to-end disk round trip."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.bundle = self.tmp / "bundle"
        self.bundle.mkdir()

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_write_then_load_via_bundle_path(self) -> None:
        comp = _component(
            [
                _op("build_agent"),
                _op("run_agent", params=[_param("agent_id")]),
            ]
        )
        g = detect_lifecycle_patterns([comp])
        write_tool_dependencies(g, self.bundle / ".clawcodex" / "tool-dependencies.yaml")
        loaded = load_tool_dependencies(self.bundle)
        self.assertIsNotNone(loaded)
        assert loaded is not None
        self.assertEqual(len(loaded.dependencies), 1)
        self.assertEqual(loaded.dependencies[0].from_tool, "comp.build-agent")
        self.assertEqual(loaded.dependencies[0].to_tool, "comp.run-agent")

    def test_load_missing_returns_none(self) -> None:
        self.assertIsNone(load_tool_dependencies(self.bundle))

    def test_load_corrupt_file_returns_empty_graph(self) -> None:
        path = self.bundle / ".clawcodex" / "tool-dependencies.yaml"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("::: not yaml :::\n  - {oops", encoding="utf-8")
        loaded = load_tool_dependencies(self.bundle)
        self.assertIsNotNone(loaded)
        assert loaded is not None
        self.assertTrue(loaded.is_empty())

    def test_atomic_write_does_not_leave_tmp(self) -> None:
        g = ToolDependencyGraph(dependencies=[ToolDependency(from_tool="a", to_tool="b")])
        write_tool_dependencies(g, self.bundle / ".clawcodex" / "tool-dependencies.yaml")
        leftover = list((self.bundle / ".clawcodex").glob("*.tmp"))
        self.assertEqual(leftover, [])

    def test_minimal_yaml_loader_handles_writer_output(self) -> None:
        # Force the no-PyYAML path by patching the import
        import builtins

        original_import = builtins.__import__

        def _no_yaml(name: str, *args, **kwargs):
            if name == "yaml" or name.startswith("yaml."):
                raise ImportError("no yaml in test")
            return original_import(name, *args, **kwargs)

        builtins.__import__ = _no_yaml  # type: ignore[assignment]
        try:
            comp = _component(
                [
                    _op("build_agent"),
                    _op("run_agent", params=[_param("agent_id")]),
                ]
            )
            g = detect_lifecycle_patterns([comp])
            path = self.bundle / ".clawcodex" / "tool-dependencies.yaml"
            write_tool_dependencies(g, path)
            text = path.read_text(encoding="utf-8")
            # The no-PyYAML writer emits block-style YAML (not JSON), so
            # the reader's minimal loader can round-trip it.
            payload = parse_graph_payload(text)
            self.assertIsNotNone(payload)
            self.assertEqual(len(payload.dependencies), 1)
        finally:
            builtins.__import__ = original_import  # type: ignore[assignment]

    def test_minimal_yaml_loads_inline_map_sequence(self) -> None:
        """Block-style ``- key: value`` items with indented continuations.

        Regression for review feedback: the no-PyYAML loader must parse
        ``dependencies`` whose items are inline maps with continuation
        lines (PyYAML ``default_flow_style=False`` output) without
        dropping fields or skipping subsequent items.
        """
        import builtins

        from extensions.sop_converter.core.dependency import reader

        original_import = builtins.__import__

        def _no_yaml(name: str, *args, **kwargs):
            if name == "yaml" or name.startswith("yaml."):
                raise ImportError("no yaml in test")
            return original_import(name, *args, **kwargs)

        builtins.__import__ = _no_yaml  # type: ignore[assignment]
        try:
            payload = reader.parse_graph_payload(
                textwrap.dedent(
                    """\
                    version: 1
                    dependencies:
                    - from: build-agent
                      to: run-agent
                      shared_params:
                      - agent_id
                    - from: create-session
                      to: run-session
                    """
                )
            )
        finally:
            builtins.__import__ = original_import  # type: ignore[assignment]

        self.assertEqual(len(payload.dependencies), 2)
        first = payload.dependencies[0]
        self.assertEqual(first.from_tool, "build-agent")
        self.assertEqual(first.to_tool, "run-agent")
        self.assertIn("agent_id", first.shared_params)
        self.assertEqual(payload.dependencies[1].from_tool, "create-session")
        self.assertEqual(payload.dependencies[1].to_tool, "run-session")

    def test_minimal_yaml_loads_inline_map_with_single_item(self) -> None:
        """A single ``- key: value`` item must not skip the next sibling.

        Regression for review feedback: ``idx`` advancement after an
        inline mapping used to over-skip the following sequence item.
        """
        import builtins

        from extensions.sop_converter.core.dependency import reader

        original_import = builtins.__import__

        def _no_yaml(name: str, *args, **kwargs):
            if name == "yaml" or name.startswith("yaml."):
                raise ImportError("no yaml in test")
            return original_import(name, *args, **kwargs)

        builtins.__import__ = _no_yaml  # type: ignore[assignment]
        try:
            payload = reader.parse_graph_payload(
                textwrap.dedent(
                    """\
                    dependencies:
                    - from: a
                    - from: b
                    """
                )
            )
        finally:
            builtins.__import__ = original_import  # type: ignore[assignment]

        self.assertEqual(len(payload.dependencies), 2)
        self.assertEqual(payload.dependencies[0].from_tool, "a")
        self.assertEqual(payload.dependencies[1].from_tool, "b")


class TestWriterFallback(unittest.TestCase):
    """Writer degrades gracefully when PyYAML cannot represent the data."""

    def test_yaml_dump_falls_back_to_yaml_subset_on_representer_error(self) -> None:
        try:
            import yaml  # noqa: F401
        except ImportError:
            self.skipTest("PyYAML not installed")

        from unittest.mock import patch

        from extensions.sop_converter.core.dependency import reader, writer

        data = {"dependencies": [{"from": "a", "to": "b"}]}
        with patch("yaml.safe_dump", side_effect=Exception("cannot represent an object")):
            out = writer._yaml_dump(data)
        # Falls back to the built-in YAML-subset emitter so the writer
        # never raises and the ``.yaml`` payload stays valid YAML.
        self.assertTrue(out.lstrip().startswith("dependencies:"))
        parsed = reader._minimal_yaml_load(out)
        self.assertEqual(parsed["dependencies"][0]["to"], "b")

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.sdk = self.tmp / "sdk"
        self.sdk.mkdir()
        (self.sdk / "agent.py").write_text(
            textwrap.dedent(
                """
                from typing import Dict, Any
                class AgentBuilder:
                    def build_agent(self, name: str = "demo") -> Dict[str, Any]:
                        return {"agent_id": "id-1", "name": name}
                    def run_agent(self, agent_id: str, query: str = "") -> str:
                        return f"ok {agent_id} {query}"
                """
            ).strip(),
            encoding="utf-8",
        )
        (self.sdk / "__init__.py").write_text("", encoding="utf-8")
        self.bundle = self.tmp / "bundle"
        self.bundle.mkdir()

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_register_component_tools_emits_yaml(self) -> None:
        """Verify the bridge writes the yaml when ``bundle_dir`` is set."""
        from extensions.sop_converter.runtime.tool_registry_bridge import (
            register_component_tools,
        )

        register_component_tools(
            components=[],  # components list can be empty — we just want the yaml hook
            source_dir=str(self.sdk),
            bundle_dir=self.bundle,
            bundle_id="bundle",
        )
        # The bridge writes bundle/.clawcodex/tool-dependencies.yaml whenever
        # bundle_dir is set (even for an empty component list), so the file
        # must exist and round-trip through the reader.
        yaml_path = self.bundle / ".clawcodex" / "tool-dependencies.yaml"
        self.assertTrue(
            yaml_path.is_file(),
            "register_component_tools with bundle_dir should write bundle/.clawcodex/tool-dependencies.yaml",
        )
        payload = parse_graph_payload(yaml_path.read_text(encoding="utf-8"))
        self.assertIsNotNone(payload)
        self.assertEqual(len(payload.dependencies), 0)


if __name__ == "__main__":
    unittest.main()
