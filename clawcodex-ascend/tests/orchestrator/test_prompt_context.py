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
"""Focused tests for prompt workspace context and Python resolution."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import extensions.orchestrator.prompt_context as prompt_context
from extensions.orchestrator.prompt_context import (
    _build_sequential_workspace_context,
    _detect_python_in_workspace,
    _get_operator_hints,
    _parse_conda_env_name,
    _parse_pyvenv_home,
    _resolve_agent_expansion_workspace,
    _resolve_workspace_path,
    _to_jinja_value,
    resolve_python_executable,
)


def test_build_sequential_workspace_context_contains_chain_metadata() -> None:
    session = SimpleNamespace(
        workspace_strategy="sequential",
        integration_branch="integration",
        start_commit_sha="start",
        base_commit_sha="base",
        previous_issue_id="F-1",
        sequence_index=2,
    )

    context = _build_sequential_workspace_context(session)

    assert "Sequential Workspace Context" in context
    assert "`integration`" in context
    assert "`F-1`" in context
    assert "`2`" in context


def test_workspace_path_helpers_accept_path_and_string(tmp_path: Path) -> None:
    path_session = SimpleNamespace(workspace=SimpleNamespace(path=tmp_path))
    string_session = SimpleNamespace(workspace=SimpleNamespace(path=str(tmp_path)))

    assert _resolve_workspace_path(path_session) == tmp_path
    assert _resolve_agent_expansion_workspace(string_session) == tmp_path
    assert _resolve_workspace_path(None) is None


def test_to_jinja_value_recurses_through_mapping_and_list() -> None:
    value = {1: [{2: "answer"}]}

    assert _to_jinja_value(value) == {"1": [{"2": "answer"}]}


def test_operator_hints_are_one_shot_but_repro_section_is_retained(tmp_path: Path) -> None:
    hints = tmp_path / ".operator_hints.md"
    hints.write_text(
        "## Inject Hints\nUse the narrow fix.\n\n## Reproduction established\nRun `pytest focused.py`.\n",
        encoding="utf-8",
    )

    first = _get_operator_hints(tmp_path)
    second = _get_operator_hints(tmp_path)

    assert first is not None and "narrow fix" in first
    assert second is not None and "Reproduction" in second
    assert "narrow fix" not in second


def test_python_metadata_parsers_soft_fail_and_strip_values(tmp_path: Path) -> None:
    pyvenv = tmp_path / "pyvenv.cfg"
    pyvenv.write_text('include-system-site-packages = false\nhome = "C:/Python311"\n', encoding="utf-8")
    conda = tmp_path / "environment.yml"
    conda.write_text("dependencies:\n  - python=3.11\nname: runtime-env\n", encoding="utf-8")

    assert _parse_pyvenv_home(pyvenv) == "C:/Python311"
    assert _parse_conda_env_name(conda) == "runtime-env"
    assert _parse_pyvenv_home(tmp_path / "missing.cfg") == ""
    assert _parse_conda_env_name(tmp_path / "missing.yml") == ""


def test_pyvenv_cfg_resolves_interpreter_from_venv_root(tmp_path: Path) -> None:
    venv_root = tmp_path / ".venv"
    interpreter = venv_root / "bin" / "python3"
    interpreter.parent.mkdir(parents=True)
    interpreter.touch()
    (venv_root / "pyvenv.cfg").write_text(
        f"home = {tmp_path / 'base-python'}\n",
        encoding="utf-8",
    )

    assert _detect_python_in_workspace(tmp_path, [".venv/pyvenv.cfg"]) == str(interpreter)


def test_conda_environment_resolves_windows_interpreter_layout(tmp_path: Path, monkeypatch) -> None:
    conda_root = tmp_path / "Miniconda3"
    interpreter = conda_root / "envs" / "runtime-env" / "python.exe"
    interpreter.parent.mkdir(parents=True)
    interpreter.touch()
    (tmp_path / "environment.yml").write_text("name: runtime-env\n", encoding="utf-8")
    monkeypatch.setattr(prompt_context, "_CONDA_ROOT_CANDIDATES", (str(conda_root),))
    monkeypatch.delenv("CONDA_PREFIX", raising=False)

    assert _detect_python_in_workspace(tmp_path, ["environment.yml"]) == str(interpreter)


def test_resolve_python_executable_honours_precedence(tmp_path: Path) -> None:
    agent_cfg = SimpleNamespace(python_executable="agent-python")
    workspace_cfg = SimpleNamespace(
        python_executable="workspace-python",
        python_auto_detect=False,
        python_detect_candidates=[],
    )

    assert (
        resolve_python_executable(
            workspace_path=tmp_path,
            agent_cfg=agent_cfg,
            workspace_cfg=workspace_cfg,
            issue_executable=" issue-python ",
        )
        == "issue-python"
    )
    assert (
        resolve_python_executable(
            workspace_path=tmp_path,
            agent_cfg=agent_cfg,
            workspace_cfg=workspace_cfg,
        )
        == "workspace-python"
    )


def test_resolve_python_executable_falls_back_to_agent_default() -> None:
    agent_cfg = SimpleNamespace(python_executable="agent-python")
    workspace_cfg = SimpleNamespace(
        python_executable="",
        python_auto_detect=False,
        python_detect_candidates=[],
    )

    assert (
        resolve_python_executable(
            workspace_path=None,
            agent_cfg=agent_cfg,
            workspace_cfg=workspace_cfg,
        )
        == "agent-python"
    )
