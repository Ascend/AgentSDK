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

# pylint: disable=relative-beyond-top-level
# tech_v26.2.0 has not merged package marker files (e.g. extensions/__init__.py)
# yet, so pylint cannot tell that sop_converter is a Python package and flags
# valid relative imports as E0402. Drop this tag once the package markers land.


"""Shared AST scan cache for workflow-mode analysis."""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
from pathlib import Path

from .ast_helpers import (
    collect_enum_member_names,
    find_enum_classes,
    get_enum_members,
    parse_ast,
    pick_primary_stage_enum,
    walk_py_files,
)


@dataclass
class SourceScanContext:
    source_dir: Path
    trees: dict[Path, ast.Module] = field(default_factory=dict)
    enum_member_names: set[str] = field(default_factory=set)
    primary_stage_enum: str | None = None
    member_to_value: dict[str, int] = field(default_factory=dict)
    enum_class_names: set[str] = field(default_factory=set)

    @classmethod
    def build(cls, source_dir: str | Path) -> SourceScanContext:
        root = Path(source_dir)
        ctx = cls(source_dir=root)
        for py_file in walk_py_files(root):
            tree = parse_ast(py_file)
            if tree is not None:
                ctx.trees[py_file] = tree
        ctx.primary_stage_enum = pick_primary_stage_enum(ctx.trees)
        ctx.enum_class_names = cls._collect_enum_class_names(ctx.trees)
        if ctx.primary_stage_enum:
            ctx.member_to_value = cls._members_for_class(ctx.trees, ctx.primary_stage_enum)
            ctx.enum_member_names = set(ctx.member_to_value.keys())
        else:
            ctx.enum_member_names = collect_enum_member_names(ctx.trees)
            for tree in ctx.trees.values():
                for enum_cls in find_enum_classes(tree):
                    ctx.member_to_value.update(get_enum_members(enum_cls))
        return ctx

    @staticmethod
    def _collect_enum_class_names(trees: dict[Path, ast.Module]) -> set[str]:
        names: set[str] = set()
        for tree in trees.values():
            for cls in find_enum_classes(tree):
                names.add(cls.name)
        return names

    @staticmethod
    def _members_for_class(trees: dict[Path, ast.Module], class_name: str) -> dict[str, int]:
        for tree in trees.values():
            for cls in find_enum_classes(tree):
                if cls.name == class_name:
                    return get_enum_members(cls)
        return {}
