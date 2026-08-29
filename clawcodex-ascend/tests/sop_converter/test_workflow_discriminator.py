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

"""Tests for the workflow discriminator."""

from __future__ import annotations

from pathlib import Path


from extensions.sop_converter.workflow_mode.discriminator import WorkflowDiscriminator
from extensions.sop_converter.workflow_mode.ast_helpers import (
    parse_enum_dict_mapping,
    resolve_enum_member,
)
import ast

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"


class TestAstEnumAttribute:
    def test_resolve_enum_member_attribute(self):
        dict_code = "{Stage.A: Stage.A}"
        dict_node = ast.parse(dict_code).body[0].value  # type: ignore[attr-defined]
        members = {"A": 1}
        ref = resolve_enum_member(dict_node.keys[0], {"Stage"}, members)
        assert ref == ("A", 1)

    def test_parse_enum_dict_mapping(self):
        code = """
from enum import IntEnum
class Stage(IntEnum):
    PREPROCESS = 1
    ANALYZE = 2
NEXT = {Stage.PREPROCESS: Stage.ANALYZE}
"""
        tree = ast.parse(code)
        dict_node = None
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign) and isinstance(node.value, ast.Dict):
                dict_node = node.value
        assert dict_node is not None
        members = {"PREPROCESS": 1, "ANALYZE": 2}
        pairs = parse_enum_dict_mapping(dict_node, {"Stage"}, members)
        assert pairs == [(1, 2)]

    def test_parse_linear_next_stage_dictcomp_enumerate(self):
        from extensions.sop_converter.workflow_mode.ast_helpers import (
            parse_linear_next_stage_dictcomp,
        )

        code = "{s: seq[i + 1] for i, s in enumerate(seq)}"
        dictcomp = ast.parse(code, mode="eval").body  # type: ignore[attr-defined]
        assert isinstance(dictcomp, ast.DictComp)
        pairs = parse_linear_next_stage_dictcomp(dictcomp, stage_sequence=[1, 2, 3])
        assert pairs == [(1, 2), (2, 3)]

    def test_parse_linear_next_stage_dictcomp_single_name(self):
        from extensions.sop_converter.workflow_mode.ast_helpers import (
            parse_linear_next_stage_dictcomp,
        )

        code = "{stage: fn(stage) for stage in STAGE_SEQUENCE}"
        dictcomp = ast.parse(code, mode="eval").body  # type: ignore[attr-defined]
        assert isinstance(dictcomp, ast.DictComp)
        pairs = parse_linear_next_stage_dictcomp(dictcomp, stage_sequence=[1, 2, 3])
        assert pairs == [(1, 2), (2, 3)]

    def test_parse_enum_dict_mapping_from_expr_dictcomp(self):
        from extensions.sop_converter.workflow_mode.ast_helpers import (
            parse_enum_dict_mapping_from_expr,
        )

        code = "{stage: seq[i + 1] for i, stage in enumerate(seq)}"
        dictcomp = ast.parse(code, mode="eval").body  # type: ignore[attr-defined]
        pairs = parse_enum_dict_mapping_from_expr(
            dictcomp,
            {"Stage"},
            {"A": 1, "B": 2, "C": 3},
            stage_sequence=[1, 2, 3],
        )
        assert pairs == [(1, 2), (2, 3)]


class TestWorkflowDiscriminator:
    def test_sdk_project(self):
        path = FIXTURES / "fixture_sdk_project"
        disc = WorkflowDiscriminator(path).discriminate()
        assert disc.mode == "sdk"
        assert disc.total_score < 0.3

    def test_hybrid_project(self):
        path = FIXTURES / "fixture_hybrid_project"
        disc = WorkflowDiscriminator(path).discriminate()
        assert disc.mode == "hybrid"
        assert 0.3 <= disc.total_score < 0.7

    def test_fwa_project(self):
        path = FIXTURES / "fixture_fwa_project"
        disc = WorkflowDiscriminator(path).discriminate()
        assert disc.mode == "fwa"
        assert disc.total_score >= 0.7
        assert disc.fwa_qualified

    def test_force_mode_sdk(self):
        path = FIXTURES / "fixture_fwa_project"
        disc = WorkflowDiscriminator(path).discriminate(force_mode="sdk")
        assert disc.mode == "sdk"
        assert disc.forced

    def test_force_mode_fwa(self):
        path = FIXTURES / "fixture_sdk_project"
        disc = WorkflowDiscriminator(path).discriminate(force_mode="fwa")
        assert disc.mode == "fwa"
        assert disc.forced


class TestDiscriminationSummary:
    def test_format_includes_mode_and_features(self):
        from extensions.sop_converter.workflow_mode.extractors.preview import (
            format_discrimination_summary,
        )

        path = FIXTURES / "fixture_fwa_project"
        disc = WorkflowDiscriminator(path).discriminate()
        text = format_discrimination_summary(disc)
        assert "Selected mode: fwa" in text
        assert "stage_enum" in text
        assert "Total score:" in text
        assert "FWA combo gate: passed" in text


class TestWalkPyFilesExcludesClawcodex:
    def test_clawcodex_output_dir_is_skipped(self, tmp_path):
        """Regression: ``walk_py_files`` must not descend into ``.clawcodex/``
        — generated bundle artifacts (agent-tools/scripts wrappers) are not
        source.  A stale wrapper with an invalid signature would otherwise
        trigger "Failed to parse ..." noise during workflow discrimination.
        """
        from extensions.sop_converter.workflow_mode.ast_helpers import walk_py_files

        (tmp_path / "real.py").write_text("X = 1\n")
        bundle = tmp_path / ".clawcodex" / "proj" / "agent-tools" / "scripts"
        bundle.mkdir(parents=True)
        # A valid generated wrapper — would be yielded if .clawcodex weren't
        # excluded, polluting the discriminator's file set.
        (bundle / "generated_wrapper.py").write_text("Y = 2\n")

        files = {p.name for p in walk_py_files(tmp_path)}
        assert "real.py" in files
        assert "generated_wrapper.py" not in files
