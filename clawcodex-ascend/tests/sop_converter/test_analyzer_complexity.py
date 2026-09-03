#!/usr/bin/env python3
# coding=utf-8

# -------------------------------------------------------------------------
# This file is part of the AgentSDK project.
#
# Originally from the clawcodex project:
#   https://github.com/agentforce314/clawcodex
#   Copyright (c) 2026 Clawd Codex Team
#   Licensed under the MIT License. See LICENSE-MIT-clawcodex in this directory.
#
# Portions copyright (c) 2026 Huawei Technologies Co.,Ltd.
# Licensed under Mulan PSL v2. You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
#
# This file is redistributed as a verbatim copy of the upstream source
# (minor whitespace / quoting normalization only); the original copyright
# notice and license terms above apply to the corresponding portions of
# this file. Local additions, if any, are licensed under Mulan PSL v2
# by Huawei Technologies Co.,Ltd.
# -------------------------------------------------------------------------

"""Tests for F-50-C complexity scoring (real nesting depth, PR #771 review)."""

from __future__ import annotations

import ast

from extensions.sop_converter.workflow_mode.capability.analyzer import (
    _max_control_flow_depth,
    _score_complexity,
)


def _tree(source: str) -> ast.Module:
    return ast.parse(source)


class TestMaxControlFlowDepth:
    def test_sibling_loops_count_as_depth_one(self):
        # 30 top-level for loops at the same nesting level -> depth 1, not 30.
        source = "\n".join(f"for i{i} in range(3):\n    print(i{i})" for i in range(30))
        assert _max_control_flow_depth(_tree(source)) == 1

    def test_sibling_ifs_count_as_depth_one(self):
        source = "\n".join(f"if x{i}:\n    pass" for i in range(20))
        assert _max_control_flow_depth(_tree(source)) == 1

    def test_nested_for_if_while_depth_three(self):
        source = """
def run():
    for i in range(3):
        if i > 1:
            while i < 10:
                print(i)
"""
        assert _max_control_flow_depth(_tree(source)) == 3

    def test_try_with_mix_nesting(self):
        source = """
def run():
    try:
        with open("f") as fh:
            for line in fh:
                if "x" in line:
                    print(line)
    except OSError:
        pass
"""
        # try -> with -> for -> if  = 4 levels
        assert _max_control_flow_depth(_tree(source)) == 4

    def test_function_nesting_does_not_count(self):
        # def-in-def is not control flow; nesting depth must stay 0.
        source = """
def outer():
    def inner():
        pass
    return inner
"""
        assert _max_control_flow_depth(_tree(source)) == 0

    def test_empty_module(self):
        assert _max_control_flow_depth(_tree("")) == 0


class TestScoreComplexityRegression:
    def test_many_sibling_ifs_no_longer_inflate_complexity(self):
        # 20 sibling ifs inside one function: old logic scored max_depth=20
        # (depth_score=1.0), pushing complexity >= 0.4; real depth is 1.
        body = "\n".join(f"    if x{i}:\n        print(i{i})" for i in range(20))
        source = f"def run():\n{body}\n"
        tree = _tree(source)
        assert _max_control_flow_depth(tree) == 1
        # lines ~= 60 -> line_score 0.3 -> 0.5*0.3 = 0.15; depth contributes 0.02
        complexity = _score_complexity(tree)
        assert complexity < 0.4

    def test_line_and_import_contributions_still_count(self):
        source = """
import os
import sys

def run():
    total = 0
    for i in range(100):
        total += i
    return total
"""
        tree = _tree(source)
        complexity = _score_complexity(tree)
        # depth=1 -> 0.3*(1/15)=0.02; lines ~= 9 -> line_score 0.045 -> 0.0225
        # imports=2 -> 0.2*(2/20)=0.02
        assert 0.02 <= complexity <= 0.2

    def test_score_bounded(self):
        source = "\n".join(f"if x{i}:\n    pass" for i in range(100))
        assert 0.0 <= _score_complexity(_tree(source)) <= 1.0
