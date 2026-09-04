"""Tests for SOP convert tool dependency inference."""

from __future__ import annotations

import tempfile
import textwrap
import unittest
from pathlib import Path

from extensions.sop_converter.core.import_alias_resolver import ModuleImportIndex
from extensions.sop_converter.core.source_parser import ParamSpec, SourceComponent, SourceOperation
from extensions.sop_converter.core.tool_dependencies import (
    _is_chain_builder_producer,
    build_tool_dependency_index,
    dependency_schema_fragment,
    enrich_input_schema_with_dependencies,
    sanitize_type_name,
    to_kebab_tool_name,
)
from extensions.sop_converter.tool_registry_bridge import operation_to_spec


def _comp(name: str, ops: list[SourceOperation]) -> SourceComponent:
    return SourceComponent(name=name, file_path=f"{name}/mod.py", description=name, operations=ops)


class TestToolDependencies(unittest.TestCase):
    def test_sanitize_skips_primitives(self) -> None:
        self.assertIsNone(sanitize_type_name("str"))
        self.assertEqual(sanitize_type_name("AgentSpec"), "agentspec")

    def test_sanitize_skips_generic_callback_types(self) -> None:
        # These generic types would blow up the enables list, so skip them.
        for hint in (
            "Callable",
            "Callable[..., Any]",
            "Callable[[int], str]",
            "Awaitable",
            "Awaitable[str]",
            "Coroutine",
            "Coroutine[Any, Any, None]",
            "AsyncGenerator",
            "AsyncIterator",
            "Type",
            "Type[Foo]",
            "ClassVar",
            "ClassVar[int]",
            "NoReturn",
            "Event",
            "Queue",
            "Lock",
            "Future",
            "asyncio.Task",
            "Protocol",
            "Generic",
        ):
            self.assertIsNone(
                sanitize_type_name(hint),
                f"expected {hint!r} to be filtered as primitive",
            )

    def test_callable_return_type_does_not_create_dependency(self) -> None:
        # A tool returning Callable must not mark every Callable-accepting
        # tool as downstream.
        factory = SourceOperation(
            name="get_validator",
            description="Get validator.",
            return_type="Callable",
        )
        consumer = SourceOperation(
            name="run_with_callback",
            description="Run with cb.",
            parameters=[ParamSpec(name="cb", type_hint="Callable")],
        )
        index = build_tool_dependency_index([_comp("a", [factory]), _comp("b", [consumer])])
        factory_deps = index[to_kebab_tool_name("a", factory)]
        consumer_deps = index[to_kebab_tool_name("b", consumer)]
        self.assertEqual(factory_deps.enables, [])
        self.assertEqual(consumer_deps.requires, [])

    def test_builds_prerequisite_edge(self) -> None:
        factory = SourceOperation(
            name="make_session",
            description="Create session",
            return_type="SessionHandle",
        )
        runner = SourceOperation(
            name="run_chat",
            description="Run chat",
            parameters=[ParamSpec(name="session", type_hint="SessionHandle")],
        )
        components = [
            _comp("teams.core", [factory]),
            _comp("teams.chat", [runner]),
        ]
        index = build_tool_dependency_index(components)
        runner_tool = to_kebab_tool_name("teams.chat", runner)
        factory_tool = to_kebab_tool_name("teams.core", factory)
        deps = index[runner_tool]
        self.assertIn(factory_tool, deps.requires)
        self.assertIn(runner_tool, index[factory_tool].enables)

    def test_operation_to_spec_embeds_dependencies(self) -> None:
        factory = SourceOperation(name="make_session", description="Create", return_type="SessionHandle")
        runner = SourceOperation(
            name="run_chat",
            description="Run",
            parameters=[ParamSpec(name="session", type_hint="SessionHandle")],
        )
        index = build_tool_dependency_index(
            [_comp("a", [factory]), _comp("b", [runner])],
        )
        runner_tool = to_kebab_tool_name("b", runner)
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            spec = operation_to_spec(
                runner,
                source_dir=tmp,
                script_path=f"{tmp}/script.py",
                comp_name="b",
                tool_deps=index[runner_tool],
            )
        fragment = spec.input_schema.get("x-sop-dependencies")
        self.assertIsNotNone(fragment)
        assert fragment is not None
        self.assertTrue(fragment.get("requires"))
        self.assertNotIn("Prerequisites", spec.description)
        self.assertNotIn("has-prerequisites", spec.tags)

    def test_enrich_schema_noop_without_edges(self) -> None:
        schema = {"type": "object", "properties": {}}
        enriched = enrich_input_schema_with_dependencies(schema, None)
        self.assertNotIn("x-sop-dependencies", enriched)
        self.assertIsNone(dependency_schema_fragment(None))

    def test_task_guide_omits_prerequisite_notes(self) -> None:
        from extensions.sop_converter.runtime.skill_grouper import SkillSpec
        from extensions.sop_converter.task_guide import generate_task_guide_markdown

        factory = SourceOperation(
            name="make_session",
            description="Create session handle for chat.",
            has_docstring=True,
            return_type="SessionHandle",
        )
        runner = SourceOperation(
            name="run_chat",
            description="Run chat with an existing session.",
            has_docstring=True,
            parameters=[ParamSpec(name="session", type_hint="SessionHandle")],
        )
        components = [
            _comp("teams.core", [factory]),
            _comp("teams.chat", [runner]),
        ]
        index = build_tool_dependency_index(components)
        factory_tool = to_kebab_tool_name("teams.core", factory)
        runner_tool = to_kebab_tool_name("teams.chat", runner)
        skill = SkillSpec(
            name="teams",
            description="Team chat",
            allowed_tools=[factory_tool, runner_tool],
        )
        guide = generate_task_guide_markdown(skill, components, tool_deps_index=index)
        self.assertNotIn("前置工具", guide)
        self.assertNotIn("后置工具", guide)

    def test_tool_search_document_omits_orchestration_metadata(self) -> None:
        from clawcodex_ext.agent.tool_authoring.factory import build_tool_from_spec
        from clawcodex_ext.tool_system.tools.tool_search_matching import tool_search_document

        factory = SourceOperation(name="make_session", description="Create", return_type="SessionHandle")
        runner = SourceOperation(
            name="run_chat",
            description="Run chat with session.",
            parameters=[ParamSpec(name="session", type_hint="SessionHandle")],
        )
        index = build_tool_dependency_index(
            [_comp("a", [factory]), _comp("b", [runner])],
        )
        runner_tool = to_kebab_tool_name("b", runner)
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            spec = operation_to_spec(
                runner,
                source_dir=tmp,
                script_path=f"{tmp}/script.py",
                comp_name="b",
                tool_deps=index[runner_tool],
            )
        tool = build_tool_from_spec(spec)
        document = tool_search_document(tool)
        self.assertIn("run chat", document)
        self.assertNotIn("prerequisites", document)
        self.assertNotIn("has-prerequisites", document)
        self.assertNotIn("make-session", document)

    def test_chain_builder_not_registered_as_producer(self) -> None:
        self.assertTrue(
            _is_chain_builder_producer(
                SourceOperation(
                    name="configure_model_provider",
                    description="Configure provider",
                    class_name="AgentConfig",
                    return_type="AgentConfig",
                )
            )
        )
        configure = SourceOperation(
            name="configure_model_provider",
            description="Configure provider",
            class_name="AgentConfig",
            return_type="AgentConfig",
            file_stem="react_agent",
        )
        consumer = SourceOperation(
            name="create_llm_agent",
            description="Create agent",
            parameters=[ParamSpec(name="agent_config", type_hint="AgentConfig")],
            file_stem="llm_agent",
        )
        components = [
            SourceComponent(
                name="demo_sdk.agents",
                file_path="demo_sdk/agents",
                description="agents",
                operations=[configure],
            ),
            SourceComponent(
                name="demo_sdk.app",
                file_path="demo_sdk/app",
                description="app",
                operations=[consumer],
            ),
        ]
        root = self._write_disambiguation_sdk()
        index = build_tool_dependency_index(components, source_dir=str(root))
        consumer_tool = to_kebab_tool_name("demo_sdk.app", consumer)
        configure_tool = to_kebab_tool_name("demo_sdk.agents", configure)
        self.assertNotIn(configure_tool, index[consumer_tool].requires)

    def test_qualified_types_do_not_cross_match(self) -> None:
        root = self._write_disambiguation_sdk()
        resolver = ModuleImportIndex(str(root))
        legacy_identity = resolver.resolve_type_identity(
            "demo_sdk.app.llm_agent",
            "AgentConfig",
        )
        new_identity = resolver.resolve_type_identity(
            "demo_sdk.agents.react_agent",
            "AgentConfig",
        )
        self.assertNotEqual(legacy_identity, new_identity)

    def _write_disambiguation_sdk(self) -> Path:
        import shutil

        tmpdir = tempfile.mkdtemp()
        root = Path(tmpdir)
        self.addCleanup(lambda: shutil.rmtree(root, ignore_errors=True))
        legacy_pkg = root / "demo_sdk" / "legacy"
        agents_pkg = root / "demo_sdk" / "agents"
        app_pkg = root / "demo_sdk" / "app"
        for pkg in (legacy_pkg, agents_pkg, app_pkg):
            pkg.mkdir(parents=True)
            (pkg / "__init__.py").write_text("", encoding="utf-8")
        (legacy_pkg / "config.py").write_text(
            textwrap.dedent(
                """
                class LegacyAgentConfig:
                    pass
                """
            ),
            encoding="utf-8",
        )
        (legacy_pkg / "__init__.py").write_text(
            textwrap.dedent(
                """
                from demo_sdk.legacy.config import LegacyAgentConfig as _LegacyAgentConfig
                LegacyAgentConfig = _LegacyAgentConfig
                """
            ),
            encoding="utf-8",
        )
        (agents_pkg / "react_agent.py").write_text(
            textwrap.dedent(
                """
                class AgentConfig:
                    def configure_model_provider(self, provider: str) -> "AgentConfig":
                        return self
                """
            ),
            encoding="utf-8",
        )
        (app_pkg / "llm_agent.py").write_text(
            textwrap.dedent(
                """
                from demo_sdk.legacy import LegacyAgentConfig as AgentConfig

                def create_llm_agent(agent_config: AgentConfig):
                    return agent_config
                """
            ),
            encoding="utf-8",
        )
        return root
