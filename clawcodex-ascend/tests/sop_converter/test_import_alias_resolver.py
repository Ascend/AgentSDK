"""Tests for module-qualified type identity resolution."""

from __future__ import annotations

import tempfile
import textwrap
import unittest
import shutil
from pathlib import Path

from extensions.sop_converter.core.import_alias_resolver import ModuleImportIndex


class TestImportAliasResolver(unittest.TestCase):
    def setUp(self) -> None:
        self.root = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.root)
        self._write_layout()

    def _write_layout(self) -> None:
        legacy_pkg = self.root / "demo_sdk" / "legacy"
        agents_pkg = self.root / "demo_sdk" / "agents"
        app_pkg = self.root / "demo_sdk" / "app"
        widgets_pkg = self.root / "demo_sdk" / "widgets"
        for pkg in (legacy_pkg, agents_pkg, app_pkg, widgets_pkg):
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
        (widgets_pkg / "types.py").write_text(
            textwrap.dedent(
                """
                class WidgetConfig:
                    pass
                """
            ),
            encoding="utf-8",
        )
        (widgets_pkg / "factory.py").write_text(
            textwrap.dedent(
                """
                from .types import WidgetConfig
                """
            ),
            encoding="utf-8",
        )
        (widgets_pkg / "runner.py").write_text(
            textwrap.dedent(
                """
                from .types import WidgetConfig as PublicConfig
                """
            ),
            encoding="utf-8",
        )

    def test_alias_resolves_to_legacy_config(self) -> None:
        resolver = ModuleImportIndex(str(self.root))
        identity = resolver.resolve_type_identity(
            "demo_sdk.app.llm_agent",
            "AgentConfig",
        )
        self.assertEqual(
            identity,
            "demo_sdk_legacy_config_legacyagentconfig",
        )

    def test_local_class_resolves_in_own_module(self) -> None:
        resolver = ModuleImportIndex(str(self.root))
        identity = resolver.resolve_type_identity(
            "demo_sdk.agents.react_agent",
            "AgentConfig",
        )
        self.assertEqual(
            identity,
            "demo_sdk_agents_react_agent_agentconfig",
        )

    def test_resolve_import_path_follows_alias_to_legacy_class(self) -> None:
        resolver = ModuleImportIndex(str(self.root))
        resolved = resolver.resolve_import_path(
            "demo_sdk.app.llm_agent",
            "AgentConfig",
        )
        self.assertEqual(
            resolved,
            ("demo_sdk.legacy.config", "LegacyAgentConfig"),
        )

    def test_resolve_import_path_prefers_local_definition(self) -> None:
        resolver = ModuleImportIndex(str(self.root))
        resolved = resolver.resolve_import_path(
            "demo_sdk.agents.react_agent",
            "AgentConfig",
        )
        self.assertEqual(
            resolved,
            ("demo_sdk.agents.react_agent", "AgentConfig"),
        )

    def test_relative_import_aliases_resolve_to_same_type(self) -> None:
        resolver = ModuleImportIndex(str(self.root))
        factory_identity = resolver.resolve_type_identity(
            "demo_sdk.widgets.factory",
            "WidgetConfig",
        )
        runner_identity = resolver.resolve_type_identity(
            "demo_sdk.widgets.runner",
            "PublicConfig",
        )
        self.assertEqual(factory_identity, "demo_sdk_widgets_types_widgetconfig")
        self.assertEqual(runner_identity, factory_identity)


if __name__ == "__main__":
    unittest.main()
