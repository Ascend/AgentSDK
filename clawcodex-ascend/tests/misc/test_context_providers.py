# -------------------------------------------------------------------------
# This file is part of the AgentSDK project.
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
#
# Copyright (c) 2026 Clawd Codex Team
# SPDX-License-Identifier: MIT
# Source: https://github.com/agentforce314/clawcodex
# ClawCodex-derived portions remain licensed under the MIT License.
# See clawcodex-ascend/LICENSE.clawcodex.
#
"""Tests for context providers (from_issue, from_ci, from_config).

Covers:
  - from_issue: with / without issue_info, empty labels, long description truncation
  - from_ci: with / without ci_status, empty string, various truthy values
  - from_config: YAML file reading, runtime_ctx override, missing file
  - Smoke: all three modules can be imported and register sections correctly
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import pytest

from clawcodex_ext.context_system.section_registry import (
    clear_section_registry,
    collect_new_sections,
    get_sections_by_tag,
)


# ---------------------------------------------------------------------------
# Helper: force-reimport a module so its register_section calls run again
# after clear_section_registry() has removed them.
# ---------------------------------------------------------------------------


def _reimport(modname: str) -> None:
    """Remove *modname* from sys.modules, then re-import it."""
    sys.modules.pop(modname, None)
    __import__(modname)


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture(autouse=True)
def _clean_registry():
    """Ensure clean registry before and after each test."""
    clear_section_registry()
    yield
    clear_section_registry()


# ============================================================================
# from_issue tests
# ============================================================================


class TestFromIssue:
    """Tests for the issue-tracker context provider."""

    def _import(self):
        """Force-reimport the module so registration happens."""
        _reimport("extensions.context_providers.from_issue")

    def test_registers_section_on_import(self):
        """Importing the module registers 'issue-context' in the registry."""
        self._import()
        sections = get_sections_by_tag("issue-tracker")
        ids = [s.id for s in sections]
        assert "issue-context" in ids

    def test_builder_with_issue_info(self):
        """With issue_info in runtime_ctx, builder returns formatted content."""
        self._import()
        ctx = {
            "issue_info": {
                "title": "Fix login bug",
                "description": "Users cannot log in with SSO.",
                "labels": ["bug", "auth"],
            },
            "workflow_phase": "coding",
        }
        results = collect_new_sections(ctx)
        issue_section = next((s for s in results if s.id == "issue-context"), None)
        assert issue_section is not None
        content = issue_section.content
        assert "Fix login bug" in content
        assert "Users cannot log in with SSO" in content
        assert "bug" in content
        assert "auth" in content
        assert "coding" in content
        assert issue_section.order == 55

    def test_builder_without_issue_info(self):
        """Without issue_info, builder returns None (section is excluded)."""
        self._import()
        results = collect_new_sections({})
        ids = [s.id for s in results]
        assert "issue-context" not in ids

    def test_builder_with_empty_labels(self):
        """Empty labels list is handled gracefully."""
        self._import()
        ctx = {
            "issue_info": {
                "title": "Fix login bug",
                "description": "Users cannot log in.",
                "labels": [],
            },
        }
        results = collect_new_sections(ctx)
        issue_section = next((s for s in results if s.id == "issue-context"), None)
        assert issue_section is not None
        content = issue_section.content
        assert "Labels:" not in content  # No labels → no Labels line

    def test_builder_with_missing_description(self):
        """Missing 'description' key is handled gracefully."""
        self._import()
        ctx = {
            "issue_info": {
                "title": "Fix login bug",
                "labels": ["bug"],
            },
        }
        results = collect_new_sections(ctx)
        issue_section = next((s for s in results if s.id == "issue-context"), None)
        assert issue_section is not None
        content = issue_section.content
        assert "Fix login bug" in content

    def test_long_description_truncation(self):
        """Very long descriptions are truncated to 500 chars."""
        self._import()
        long_desc = "A" * 1000
        ctx = {
            "issue_info": {
                "title": "Long issue",
                "description": long_desc,
                "labels": [],
            },
        }
        results = collect_new_sections(ctx)
        issue_section = next((s for s in results if s.id == "issue-context"), None)
        assert issue_section is not None
        content = issue_section.content
        # Should be truncated to ~500 chars + …
        assert "AAAA" in content
        assert "…" in content  # Truncation marker

    def test_tags(self):
        """Section carries expected tags."""
        self._import()
        sec = get_sections_by_tag("workflow", "issue-tracker")
        assert any(s.id == "issue-context" for s in sec)

    def test_order_and_scope(self):
        """Section has order=55 and cache_scope=REQUEST."""
        self._import()
        from clawcodex_ext.context_system.section_registry import (
            _registry,
        )

        sec = _registry.get("issue-context")
        assert sec is not None
        assert sec.order == 55
        assert sec.cache_scope.value == "request"


# ============================================================================
# from_ci tests
# ============================================================================


class TestFromCi:
    """Tests for the CI status context provider."""

    def _import(self):
        """Force-reimport the module so registration happens."""
        _reimport("extensions.context_providers.from_ci")

    def test_registers_section_on_import(self):
        """Import registers 'ci-status'."""
        self._import()
        sections = get_sections_by_tag("ci")
        ids = [s.id for s in sections]
        assert "ci-status" in ids

    @pytest.mark.parametrize("status", ["passing", "failing", "running", "pending"])
    def test_builder_with_ci_status(self, status):
        """Various ci_status values produce a CI Status block."""
        self._import()
        ctx = {"ci_status": status}
        results = collect_new_sections(ctx)
        ci_section = next((s for s in results if s.id == "ci-status"), None)
        assert ci_section is not None
        assert status in ci_section.content

    def test_builder_without_ci_status(self):
        """Without ci_status, builder returns None."""
        self._import()
        results = collect_new_sections({})
        ids = [s.id for s in results]
        assert "ci-status" not in ids

    def test_builder_with_none_ci_status(self):
        """ci_status=None is treated as absent."""
        self._import()
        ctx = {"ci_status": None}
        results = collect_new_sections(ctx)
        ids = [s.id for s in results]
        assert "ci-status" not in ids

    def test_builder_with_empty_string_ci(self):
        """ci_status='' is treated as absent."""
        self._import()
        ctx = {"ci_status": ""}
        results = collect_new_sections(ctx)
        ids = [s.id for s in results]
        assert "ci-status" not in ids

    def test_order_and_scope(self):
        """Section has order=56 and REQUEST scope."""
        self._import()
        from clawcodex_ext.context_system.section_registry import (
            _registry,
        )

        sec = _registry.get("ci-status")
        assert sec is not None
        assert sec.order == 56
        assert sec.cache_scope.value == "request"


# ============================================================================
# from_config tests
# ============================================================================


class TestFromConfig:
    """Tests for the YAML-config context provider."""

    def _import(self):
        """Force-reimport the module so registration happens."""
        _reimport("extensions.context_providers.from_config")

    def test_registers_section_on_import(self):
        """Import registers 'declared-config'."""
        self._import()
        sections = get_sections_by_tag("config")
        ids = [s.id for s in sections]
        assert "declared-config" in ids

    def test_builder_with_runtime_ctx_override(self):
        """declared_sections from runtime_ctx['custom'] is rendered."""
        self._import()
        ctx = {
            "custom": {
                "declared_sections": [
                    {"title": "Project Info", "content": "This is a test project."},
                ],
            },
        }
        results = collect_new_sections(ctx)
        cfg_section = next((s for s in results if s.id == "declared-config"), None)
        assert cfg_section is not None
        content = cfg_section.content
        assert "Project Info" in content
        assert "test project" in content

    def test_builder_with_empty_ctx(self):
        """Without config data, builder returns None."""
        self._import()
        results = collect_new_sections({})
        ids = [s.id for s in results]
        assert "declared-config" not in ids

    def test_builder_with_yaml_file(self):
        """A valid .clawcodex/context_sections.yaml is read correctly."""
        self._import()
        with tempfile.TemporaryDirectory() as tmpdir:
            dot_dir = Path(tmpdir) / ".clawcodex"
            dot_dir.mkdir()
            yaml_path = dot_dir / "context_sections.yaml"
            yaml_path.write_text(
                "sections:\n"
                "  - title: Sprint Goal\n"
                "    content: Complete the Q3 release.\n"
                "  - title: Team\n"
                "    content: Alice, Bob, Charlie\n",
                encoding="utf-8",
            )
            ctx = {"cwd": tmpdir}
            results = collect_new_sections(ctx)
            cfg_section = next((s for s in results if s.id == "declared-config"), None)
            assert cfg_section is not None
            content = cfg_section.content
            assert "Sprint Goal" in content
            assert "Q3 release" in content
            assert "Alice" in content

    def test_builder_with_missing_yaml(self):
        """Missing .clawcodex/context_sections.yaml → no section."""
        self._import()
        with tempfile.TemporaryDirectory() as tmpdir:
            ctx = {"cwd": tmpdir}
            results = collect_new_sections(ctx)
            ids = [s.id for s in results]
            assert "declared-config" not in ids

    def test_builder_with_malformed_yaml(self):
        """Malformed YAML is handled without raising."""
        self._import()
        with tempfile.TemporaryDirectory() as tmpdir:
            dot_dir = Path(tmpdir) / ".clawcodex"
            dot_dir.mkdir()
            yaml_path = dot_dir / "context_sections.yaml"
            yaml_path.write_text("not: valid: yaml: [[[", encoding="utf-8")
            ctx = {"cwd": tmpdir}
            # Should not raise
            results = collect_new_sections(ctx)
            ids = [s.id for s in results]
            assert "declared-config" not in ids

    def test_yaml_overrides_runtime_ctx(self):
        """YAML file is preferred over runtime_ctx data."""
        self._import()
        with tempfile.TemporaryDirectory() as tmpdir:
            dot_dir = Path(tmpdir) / ".clawcodex"
            dot_dir.mkdir()
            yaml_path = dot_dir / "context_sections.yaml"
            yaml_path.write_text(
                "sections:\n  - title: From YAML\n    content: This is from the yaml file.\n",
                encoding="utf-8",
            )
            ctx = {
                "cwd": tmpdir,
                "custom": {
                    "declared_sections": [
                        {"title": "Override", "content": "Should NOT appear."},
                    ],
                },
            }
            results = collect_new_sections(ctx)
            cfg_section = next((s for s in results if s.id == "declared-config"), None)
            assert cfg_section is not None
            content = cfg_section.content
            assert "From YAML" in content
            assert "Should NOT appear" not in content

    def test_order_and_scope(self):
        """Section has order=57 and SESSION scope."""
        self._import()
        from clawcodex_ext.context_system.section_registry import (
            _registry,
        )

        sec = _registry.get("declared-config")
        assert sec is not None
        assert sec.order == 57
        assert sec.cache_scope.value == "session"


# ============================================================================
# Cross-cutting: registration metadata correctness
# ============================================================================


class TestRegistrationMetadata:
    """Verify all three providers register with correct metadata."""

    def test_all_three_register(self):
        """Importing all three registers three non-canonical sections."""
        _reimport("extensions.context_providers.from_issue")
        _reimport("extensions.context_providers.from_ci")
        _reimport("extensions.context_providers.from_config")

        from clawcodex_ext.context_system.section_registry import _registry

        assert "issue-context" in _registry
        assert "ci-status" in _registry
        assert "declared-config" in _registry

    def test_sorting_order(self):
        """Sections appear in order 55 → 56 → 57."""
        _reimport("extensions.context_providers.from_issue")
        _reimport("extensions.context_providers.from_ci")
        _reimport("extensions.context_providers.from_config")

        ctx = {
            "issue_info": {"title": "T1", "description": "D1", "labels": []},
            "ci_status": "passing",
            "custom": {
                "declared_sections": [
                    {"title": "Cfg", "content": "Cfg content"},
                ],
            },
        }
        results = collect_new_sections(ctx)
        orders = [s.order for s in results]
        assert orders == sorted(orders), f"Sections not sorted: {orders}"
        assert orders == [55, 56, 57]
