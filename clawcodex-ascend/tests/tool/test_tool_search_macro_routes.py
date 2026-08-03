"""Tests for F-57 Phase 3 MacroRoute direct routing."""

from __future__ import annotations


# pylint: disable=E0611
import unittest

from clawcodex_ext.tool_system.build_tool import build_tool
from clawcodex_ext.tool_system.registry import ToolRegistry
from clawcodex_ext.tool_system.tools.tool_search import make_tool_search_tool
from clawcodex_ext.tool_system.tools.tool_search_matching import rank_tool_matches
from extensions.sop_converter.runtime.macros.models import MacroRoute
from extensions.sop_converter.runtime.macros.routing import (
    DEFAULT_MACRO_ROUTE_CATALOG,
    MacroRouteCatalog,
    ensure_builtin_routes,
    match_macro_routes,
    resolve_macro_route,
)


class TestMacroRouteModel(unittest.TestCase):
    def test_default_selection_is_prefer(self) -> None:
        route = MacroRoute(target_tool="test-macro")
        self.assertEqual(route.selection, "prefer")

    def test_default_verified_is_false(self) -> None:
        route = MacroRoute(target_tool="test-macro")
        self.assertFalse(route.verified)

    def test_default_enabled_is_true(self) -> None:
        route = MacroRoute(target_tool="test-macro")
        self.assertTrue(route.enabled)

    def test_exclusive_requires_verified(self) -> None:
        route = MacroRoute(
            target_tool="test-macro",
            selection="exclusive",
            verified=False,
        )
        self.assertEqual(route.selection, "exclusive")
        self.assertFalse(route.verified)


class TestMacroRouteMatching(unittest.TestCase):
    def setUp(self) -> None:
        self.catalog = MacroRouteCatalog()
        self.catalog.register_route(
            MacroRoute(
                phrases=["用已创建的 agent 回复"],
                keywords=["agent", "回复"],
                negative_keywords=["创建", "配置"],
                target_tool="invoke-existing-agent",
                match_mode="all",
                selection="exclusive",
                priority=100,
                verified=True,
            )
        )
        self.catalog.register_route(
            MacroRoute(
                phrases=["创建并调用 agent"],
                keywords=["创建", "agent"],
                target_tool="create-and-invoke-agent",
                match_mode="all",
                selection="prefer",
                priority=90,
                verified=False,
            )
        )

    def test_keyword_match(self) -> None:
        matches = match_macro_routes("调用 agent 回复", catalog=self.catalog)
        self.assertEqual(len(matches), 1)
        self.assertEqual(matches[0].route.target_tool, "invoke-existing-agent")

    def test_phrase_match(self) -> None:
        matches = match_macro_routes("用已创建的 agent 回复", catalog=self.catalog)
        self.assertEqual(len(matches), 1)
        self.assertEqual(matches[0].route.target_tool, "invoke-existing-agent")
        self.assertTrue(matches[0].is_exact_phrase)

    def test_phrase_match_does_not_require_keywords(self) -> None:
        """Phrase hit is sufficient; match_mode=all only applies to keyword path."""
        catalog = MacroRouteCatalog()
        catalog.register_route(
            MacroRoute(
                phrases=["用已创建的 agent 回复"],
                keywords=["agent", "回复", "调用", "已有", "id"],
                target_tool="invoke-existing-agent",
                match_mode="all",
                selection="exclusive",
                verified=True,
            )
        )
        matches = match_macro_routes("用已创建的 agent 回复", catalog=catalog)
        self.assertEqual(len(matches), 1)
        self.assertEqual(matches[0].route.target_tool, "invoke-existing-agent")
        self.assertTrue(matches[0].is_exact_phrase)

    def test_negative_keyword_excludes(self) -> None:
        matches = match_macro_routes("创建 agent", catalog=self.catalog)
        self.assertEqual(len(matches), 1)
        self.assertEqual(matches[0].route.target_tool, "create-and-invoke-agent")

    def test_create_agent_excludes_invoke_route(self) -> None:
        matches = match_macro_routes("创建并调用已有的 agent", catalog=self.catalog)
        self.assertEqual(len(matches), 1)
        self.assertEqual(matches[0].route.target_tool, "create-and-invoke-agent")

    def test_no_match(self) -> None:
        matches = match_macro_routes("test query", catalog=self.catalog)
        self.assertEqual(len(matches), 0)


class TestMacroRouteResolution(unittest.TestCase):
    def setUp(self) -> None:
        self.tools = [
            build_tool(
                name="invoke-existing-agent",
                input_schema={"type": "object", "properties": {}},
                call=lambda _i, _c: None,
                prompt="invoke existing agent",
            ),
            build_tool(
                name="create-and-invoke-agent",
                input_schema={"type": "object", "properties": {}},
                call=lambda _i, _c: None,
                prompt="create and invoke agent",
            ),
            build_tool(
                name="noise-tool",
                input_schema={"type": "object", "properties": {}},
                call=lambda _i, _c: None,
                prompt="noise",
            ),
        ]

    def test_exclusive_route_returns_single_tool(self) -> None:
        catalog = MacroRouteCatalog()
        catalog.register_route(
            MacroRoute(
                keywords=["agent", "调用"],
                target_tool="invoke-existing-agent",
                selection="exclusive",
                verified=True,
            )
        )
        names, exclusive = resolve_macro_route("调用 agent", self.tools, catalog=catalog)
        self.assertEqual(names, ["invoke-existing-agent"])
        self.assertTrue(exclusive)

    def test_prefer_route_returns_macro(self) -> None:
        catalog = MacroRouteCatalog()
        catalog.register_route(
            MacroRoute(
                keywords=["创建", "agent"],
                target_tool="create-and-invoke-agent",
                selection="prefer",
                verified=False,
            )
        )
        names, exclusive = resolve_macro_route("创建 agent", self.tools, catalog=catalog)
        self.assertEqual(names, ["create-and-invoke-agent"])
        self.assertFalse(exclusive)

    def test_unverified_exclusive_is_skipped(self) -> None:
        catalog = MacroRouteCatalog()
        catalog.register_route(
            MacroRoute(
                keywords=["agent", "调用"],
                target_tool="invoke-existing-agent",
                selection="exclusive",
                verified=False,
            )
        )
        names, exclusive = resolve_macro_route("调用 agent", self.tools, catalog=catalog)
        self.assertEqual(names, [])
        self.assertFalse(exclusive)

    def test_tool_not_available_is_skipped(self) -> None:
        catalog = MacroRouteCatalog()
        catalog.register_route(
            MacroRoute(
                keywords=["agent", "调用"],
                target_tool="non-existent-tool",
                selection="exclusive",
                verified=True,
            )
        )
        names, exclusive = resolve_macro_route("调用 agent", self.tools, catalog=catalog)
        self.assertEqual(names, [])
        self.assertFalse(exclusive)


class TestBuiltinRoutes(unittest.TestCase):
    def test_ensure_builtin_routes_registers_invoke_existing_agent(self) -> None:
        catalog = MacroRouteCatalog()
        ensure_builtin_routes(catalog)
        routes = {route.target_tool: route for route in catalog.get_routes()}
        self.assertIn("invoke-existing-agent", routes)
        self.assertIn("resume-resource", routes)
        route = routes["invoke-existing-agent"]
        self.assertTrue(route.verified)
        self.assertEqual(route.selection, "exclusive")
        self.assertEqual(route.keywords, ["agent", "回复"])
        self.assertEqual(route.negative_keywords, ["创建", "配置", "删除", "列出"])
        self.assertEqual(route.intent_key, "agent.invoke_existing")
        self.assertEqual(route.covered_tools, ["llmagent-invoke", "send-to-agent"])
        self.assertIn("用已创建的 agent 回复", route.phrases)

    def test_ensure_builtin_routes_idempotent(self) -> None:
        catalog = MacroRouteCatalog()
        ensure_builtin_routes(catalog)
        ensure_builtin_routes(catalog)
        routes = catalog.get_routes()
        self.assertEqual(
            {route.target_tool for route in routes},
            {"invoke-existing-agent", "resume-resource"},
        )

    def test_builtin_matches_canonical_and_regression_phrases(self) -> None:
        catalog = MacroRouteCatalog()
        ensure_builtin_routes(catalog)
        for query in (
            "用已创建的 agent 回复",
            "用 verify-bot 回复 ping",
            "invoke existing agent by agent_id",
            "invoke llm agent send message to agent",
        ):
            with self.subTest(query=query):
                matches = match_macro_routes(query, catalog=catalog)
                self.assertEqual(len(matches), 1, query)
                self.assertEqual(matches[0].route.target_tool, "invoke-existing-agent")
                self.assertTrue(matches[0].is_exact_phrase, query)

    def test_builtin_matches_resume_resource_chinese_queries(self) -> None:
        catalog = MacroRouteCatalog()
        ensure_builtin_routes(catalog)
        for query in (
            "恢复资源",
            "请帮我恢复资源",
            "恢复一下资源",
            "把资源恢复一下",
            "资源恢复",
            "恢复 verify-bot 资源",
            "按 resource_ref 恢复",
            "resume-resource",
            "resume resource",
        ):
            with self.subTest(query=query):
                matches = match_macro_routes(query, catalog=catalog)
                self.assertEqual(
                    [m.route.target_tool for m in matches],
                    ["resume-resource"],
                    query,
                )

        for query in ("创建资源", "删除资源", "列出资源", "只要资源不要恢复"):
            with self.subTest(query=query):
                matches = match_macro_routes(query, catalog=catalog)
                self.assertNotIn(
                    "resume-resource",
                    [m.route.target_tool for m in matches],
                    query,
                )


class TestToolSearchMacroRouteIntegration(unittest.TestCase):
    def setUp(self) -> None:
        self.registry = ToolRegistry()
        self.registry.register(
            build_tool(
                name="invoke-existing-agent",
                input_schema={"type": "object", "properties": {}},
                call=lambda _i, _c: None,
                prompt="invoke existing agent",
            )
        )
        self.registry.register(
            build_tool(
                name="read-agent-config",
                input_schema={"type": "object", "properties": {}},
                call=lambda _i, _c: None,
                prompt="read agent config",
            )
        )
        self.registry.register(make_tool_search_tool(self.registry))

    def test_rank_tool_matches_uses_macro_route(self) -> None:
        """Assert MacroRoute path fires; not just normal scoring ranking the macro first."""
        ensure_builtin_routes(DEFAULT_MACRO_ROUTE_CATALOG)
        tools = list(self.registry.list_tools())
        query = "用已创建的 agent 回复"

        matched, exclusive = resolve_macro_route(query, tools, catalog=DEFAULT_MACRO_ROUTE_CATALOG)
        self.assertEqual(matched, ["invoke-existing-agent"])
        self.assertTrue(exclusive)

        matches = rank_tool_matches(
            query,
            tools,
            max_results=5,
            macro_route_catalog=DEFAULT_MACRO_ROUTE_CATALOG,
        )
        # exclusive must truncate, not merely place the macro at [0]
        self.assertEqual(matches, ["invoke-existing-agent"])

    def test_exclusive_truncates_to_macro_only(self) -> None:
        catalog = MacroRouteCatalog()
        catalog.register_route(
            MacroRoute(
                phrases=["用已创建的 agent 回复"],
                target_tool="invoke-existing-agent",
                selection="exclusive",
                verified=True,
            )
        )
        tools = [
            build_tool(
                name="invoke-existing-agent",
                input_schema={"type": "object", "properties": {}},
                call=lambda _i, _c: None,
                prompt="invoke existing agent",
            ),
            build_tool(
                name="other-agent-helper",
                input_schema={"type": "object", "properties": {}},
                call=lambda _i, _c: None,
                prompt="用已创建的 agent 回复 helper",
            ),
        ]
        matches = rank_tool_matches(
            "用已创建的 agent 回复",
            tools,
            max_results=5,
            macro_route_catalog=catalog,
        )
        self.assertEqual(matches, ["invoke-existing-agent"])

    def test_prefer_prepends_and_keeps_lower_candidates(self) -> None:
        catalog = MacroRouteCatalog()
        catalog.register_route(
            MacroRoute(
                keywords=["create", "agent"],
                target_tool="create-and-invoke-agent",
                match_mode="all",
                selection="prefer",
                verified=False,
            )
        )
        tools = [
            build_tool(
                name="create-and-invoke-agent",
                input_schema={"type": "object", "properties": {}},
                call=lambda _i, _c: None,
                prompt="macro create and invoke",
            ),
            build_tool(
                name="noise-alpha",
                input_schema={"type": "object", "properties": {}},
                call=lambda _i, _c: None,
                prompt="create agent noise-alpha helper",
            ),
            build_tool(
                name="noise-beta",
                input_schema={"type": "object", "properties": {}},
                call=lambda _i, _c: None,
                prompt="create agent noise-beta helper",
            ),
        ]
        matches = rank_tool_matches(
            "create agent",
            tools,
            max_results=5,
            macro_route_catalog=catalog,
        )
        self.assertEqual(matches[0], "create-and-invoke-agent")
        self.assertGreater(len(matches), 1)
        self.assertTrue({"noise-alpha", "noise-beta"} & set(matches))

    def test_macro_route_excludes_negative_keywords(self) -> None:
        ensure_builtin_routes(DEFAULT_MACRO_ROUTE_CATALOG)
        tools = list(self.registry.list_tools())
        matches = rank_tool_matches(
            "创建 agent",
            tools,
            max_results=5,
            macro_route_catalog=DEFAULT_MACRO_ROUTE_CATALOG,
        )
        self.assertNotEqual(matches[0], "invoke-existing-agent")

    def test_direct_macro_route_runs_before_lifecycle_chain(self) -> None:
        """§8.3: direct route → lifecycle-chain → scoring."""
        from types import SimpleNamespace

        catalog = MacroRouteCatalog()
        catalog.register_route(
            MacroRoute(
                phrases=["lifecycle-chain: agent create"],
                target_tool="invoke-existing-agent",
                selection="exclusive",
                verified=True,
                scope="builtin",
            )
        )
        tools = [
            build_tool(
                name="invoke-existing-agent",
                input_schema={"type": "object", "properties": {}},
                call=lambda _i, _c: None,
                prompt="macro",
            ),
            build_tool(
                name="create-llm-agent",
                input_schema={"type": "object", "properties": {}},
                call=lambda _i, _c: None,
                prompt="create",
            ),
        ]
        graph = SimpleNamespace(
            priority_routes=[
                SimpleNamespace(
                    keywords=["agent", "create"],
                    intent_group="agent_create",
                    primary_entry="create-llm-agent",
                )
            ],
            intent_groups=[
                SimpleNamespace(
                    name="agent_create",
                    description="create agent",
                    tools=["create-llm-agent"],
                )
            ],
        )
        matches = rank_tool_matches(
            "lifecycle-chain: agent create",
            tools,
            max_results=5,
            lifecycle_graph=graph,
            macro_route_catalog=catalog,
        )
        self.assertEqual(matches, ["invoke-existing-agent"])


class TestExclusiveTieAndScope(unittest.TestCase):
    def test_tied_exclusive_returns_both_candidates(self) -> None:
        catalog = MacroRouteCatalog()
        catalog.register_route(
            MacroRoute(
                keywords=["shared"],
                target_tool="macro-a",
                selection="exclusive",
                verified=True,
                priority=100,
                scope="bundle",
            )
        )
        catalog.register_route(
            MacroRoute(
                keywords=["shared"],
                target_tool="macro-b",
                selection="exclusive",
                verified=True,
                priority=100,
                scope="bundle",
            )
        )
        tools = [
            build_tool(
                name="macro-a",
                input_schema={"type": "object", "properties": {}},
                call=lambda _i, _c: None,
                prompt="a",
            ),
            build_tool(
                name="macro-b",
                input_schema={"type": "object", "properties": {}},
                call=lambda _i, _c: None,
                prompt="b",
            ),
        ]
        names, exclusive = resolve_macro_route("shared intent", tools, catalog=catalog)
        self.assertEqual(set(names), {"macro-a", "macro-b"})
        self.assertEqual(len(names), 2)
        # Tie must not silently pick a single exclusive winner
        self.assertFalse(exclusive)

    def test_session_scope_ranks_above_bundle(self) -> None:
        catalog = MacroRouteCatalog()
        catalog.register_route(
            MacroRoute(
                keywords=["agent", "call"],
                target_tool="bundle-macro",
                match_mode="all",
                selection="prefer",
                priority=100,
                scope="bundle",
            )
        )
        catalog.register_route(
            MacroRoute(
                keywords=["agent", "call"],
                target_tool="session-macro",
                match_mode="all",
                selection="prefer",
                priority=100,
                scope="session",
            )
        )
        matches = match_macro_routes("call agent", catalog=catalog)
        self.assertEqual(matches[0].route.target_tool, "session-macro")

    def test_builtin_safety_exclusive_not_overridden_by_session(self) -> None:
        catalog = MacroRouteCatalog()
        catalog.register_route(
            MacroRoute(
                keywords=["agent", "call"],
                target_tool="invoke-existing-agent",
                match_mode="all",
                selection="exclusive",
                verified=True,
                priority=50,
                scope="builtin",
            )
        )
        catalog.register_route(
            MacroRoute(
                keywords=["agent", "call"],
                target_tool="session-hijack",
                match_mode="all",
                selection="exclusive",
                verified=True,
                priority=200,
                scope="session",
            )
        )
        tools = [
            build_tool(
                name="invoke-existing-agent",
                input_schema={"type": "object", "properties": {}},
                call=lambda _i, _c: None,
                prompt="builtin",
            ),
            build_tool(
                name="session-hijack",
                input_schema={"type": "object", "properties": {}},
                call=lambda _i, _c: None,
                prompt="session",
            ),
        ]
        names, exclusive = resolve_macro_route("call agent", tools, catalog=catalog)
        self.assertEqual(names, ["invoke-existing-agent"])
        self.assertTrue(exclusive)


class TestLoadBundleMacroRoutes(unittest.TestCase):
    def test_load_bundle_macro_routes_from_yaml(self) -> None:
        import tempfile
        from pathlib import Path

        from extensions.sop_converter.bundle_context import load_bundle_macro_routes

        with tempfile.TemporaryDirectory() as tmp:
            macros = Path(tmp) / ".clawcodex" / "macros"
            macros.mkdir(parents=True)
            (macros / "demo.yaml").write_text(
                "\n".join(
                    [
                        "name: demo-macro",
                        "scope: bundle",
                        "enabled: true",
                        "routing:",
                        "  phrases:",
                        "    - run demo macro",
                        "  keywords:",
                        "    - demo",
                        "  target_tool: demo-macro",
                        "  intent_key: demo.run",
                        "  covered_tools:",
                        "    - demo-atomic",
                        "  selection: prefer",
                        "  priority: 80",
                    ]
                ),
                encoding="utf-8",
            )
            routes = load_bundle_macro_routes(Path(tmp))
            self.assertEqual(len(routes), 1)
            self.assertEqual(routes[0].target_tool, "demo-macro")
            self.assertEqual(routes[0].scope, "bundle")
            self.assertEqual(routes[0].intent_key, "demo.run")
            self.assertEqual(routes[0].covered_tools, ["demo-atomic"])
            self.assertIn("run demo macro", routes[0].phrases)


if __name__ == "__main__":
    unittest.main()
