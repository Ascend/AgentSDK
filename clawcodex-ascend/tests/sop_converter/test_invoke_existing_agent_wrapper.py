#!/usr/bin/env python3
# -*- coding: utf-8 -*-

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

"""Regression tests for the invoke-existing-agent executable wrapper."""

from __future__ import annotations

import importlib.util
import json
import shutil
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
WRAPPER_PATH = (
    PROJECT_ROOT
    / "extensions"
    / "sop_converter"
    / "runtime"
    / "composite_tools"
    / "scripts"
    / "invoke_existing_agent_wrapper.py"
)


def _load_wrapper():
    spec = importlib.util.spec_from_file_location("invoke_existing_agent_wrapper_test", WRAPPER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_import_does_not_modify_sys_path() -> None:
    """Importing the wrapper must not change resolution for later imports."""
    before = list(sys.path)

    _load_wrapper()

    assert sys.path == before


def test_cli_validation_does_not_bootstrap_repo_root(monkeypatch, capsys) -> None:
    """Invalid CLI input must not change import resolution for the host process."""
    wrapper = _load_wrapper()
    repo_root = str(PROJECT_ROOT)
    path_without_repo = [entry for entry in sys.path if entry != repo_root]
    cases = (
        ([str(WRAPPER_PATH)], 2, "usage"),
        ([str(WRAPPER_PATH), "unknown", "{}"], 1, "unknown_method"),
        ([str(WRAPPER_PATH), "invoke_existing_agent", "{"], 1, "invalid_json"),
        ([str(WRAPPER_PATH), "invoke_existing_agent", "[]"], 1, "invalid_input"),
    )

    for argv, expected_return_code, expected_error_code in cases:
        monkeypatch.setattr(sys, "path", list(path_without_repo))
        before = list(sys.path)

        assert wrapper.main(argv) == expected_return_code
        assert sys.path == before
        assert json.loads(capsys.readouterr().out)["error_code"] == expected_error_code


def test_standalone_wrapper_reports_missing_repo_root(tmp_path: Path) -> None:
    """A relocated wrapper must explain why converter imports are unavailable."""
    relocated_wrapper = tmp_path / WRAPPER_PATH.name
    shutil.copy2(WRAPPER_PATH, relocated_wrapper)

    result = subprocess.run(
        [sys.executable, "-I", str(relocated_wrapper), "invoke_existing_agent", "{}"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 1
    assert result.stderr == ""
    assert json.loads(result.stdout)["error_code"] == "repo_root_not_found"
