#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# -------------------------------------------------------------------------
# This file is part of the AgentSDK project.
#
# Originally from Clawd Codex:
# https://github.com/agentforce314/clawcodex
# Copyright (c) 2026 Clawd Codex Team
# Licensed under the MIT License. See clawcodex-ascend/LICENSE.clawcodex.
#
# Portions copyright (c) 2026 Huawei Technologies Co.,Ltd.
# Licensed under Mulan PSL v2. You may obtain a copy of Mulan PSL v2 at:
#
#          http://license.coscl.org.cn/MulanPSL2
#
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
# -------------------------------------------------------------------------

from __future__ import annotations


# pylint: disable=E0611
import json
import os
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from clawcodex_ext.agent.tool_authoring.call_handlers.bash import (
    BashCallError,
    _argv_for_json_args_template,
    resolve_bundle_venv_environment,
    resolve_agent_tool_bash_timeout_sec,
)
from clawcodex_ext.agent.tool_authoring.call_handlers import bash as bash_handler
from clawcodex_ext.agent.tool_authoring.factory import build_tool_from_spec
from clawcodex_ext.agent.tool_authoring.spec import AgentToolSpec
from clawcodex_ext.tool_system.context import ToolContext
from extensions.sop_converter.bundle_manifest import write_bundle_manifest


def test_resolve_agent_tool_bash_timeout_sec_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("AGENT_TOOL_BASH_TIMEOUT_SEC", raising=False)
    assert resolve_agent_tool_bash_timeout_sec() == 300.0


def test_resolve_agent_tool_bash_timeout_sec_env_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("AGENT_TOOL_BASH_TIMEOUT_SEC", "900")
    assert resolve_agent_tool_bash_timeout_sec() == 900.0


def test_sop_wrapper_failure_returns_structured_json() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        wrapper = tmp_path / "fail_wrapper.py"
        payload = {
            "created_persisted": False,
            "callable_by_agent_id": False,
            "agent_id_call_contract": "not_persisted",
            "error_code": "catalog_write_failed",
        }
        wrapper.write_text(
            f"import json, sys\nprint(json.dumps({payload!r}), file=sys.stderr)\nsys.exit(1)\n",
            encoding="utf-8",
        )
        tool = build_tool_from_spec(
            AgentToolSpec(
                name="demo-create-agent",
                description="demo",
                input_schema={"type": "object", "properties": {}},
                call_type="bash",
                call_impl=f'python3 "{wrapper}" create_agent \'{{json_args}}\'',
                source="agent-created",
            )
        )

        result = tool.call({"query": "hello"}, ToolContext(workspace_root=tmp_path))

        assert result.is_error
        assert result.output["error_code"] == "catalog_write_failed"
        assert result.output["created_persisted"] is False
        assert result.output["callable_by_agent_id"] is False
        assert result.output["agent_id_call_contract"] == "not_persisted"


def test_json_wrapper_argv_uses_host_python(tmp_path: Path) -> None:
    argv = _argv_for_json_args_template(
        'python3 "/tmp/wrapper.py" create_agent \'{json_args}\'',
        '{"id":"verify-bot"}',
    )

    assert argv[0] == sys.executable
    assert argv[-1] == '{"id":"verify-bot"}'


def test_resolve_bundle_environment_uses_ready_converted_venv(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle_dir = tmp_path / "bundle"
    sdk_dir = tmp_path / "sdk"
    bundle_dir.mkdir()
    sdk_dir.mkdir()
    write_bundle_manifest(
        bundle_dir,
        sdk_source_dir=sdk_dir,
        sdk_requirements=("jsonschema-path>=0.3",),
    )
    bundle_python = tmp_path / "bundle-venv" / "bin" / "python"
    bundle_python.parent.mkdir(parents=True)
    bundle_python.write_text("ready", encoding="utf-8")
    site_packages = tmp_path / "bundle-venv" / "lib" / "site-packages"
    site_packages.mkdir(parents=True)

    from extensions.sop_converter import bundle_venv

    monkeypatch.setattr(bundle_venv, "is_venv_ready", lambda *_args: True)
    monkeypatch.setattr(bundle_venv, "bundle_venv_python", lambda *_args: bundle_python)
    monkeypatch.setattr(
        bundle_venv,
        "bundle_venv_site_packages",
        lambda *_args: (site_packages,),
    )
    context = ToolContext(
        workspace_root=tmp_path,
        bundle_context=SimpleNamespace(bundle_path=bundle_dir),
    )

    resolved = resolve_bundle_venv_environment(context)
    assert resolved["VIRTUAL_ENV"] == str(bundle_python.parent.parent)
    assert resolved["PYTHONPATH"].split(os.pathsep)[0] == str(site_packages)
    assert resolved["PATH"].split(os.pathsep)[0] == str(bundle_python.parent)


def test_resolve_bundle_environment_fails_without_runtime_install(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle_dir = tmp_path / "bundle"
    sdk_dir = tmp_path / "sdk"
    bundle_dir.mkdir()
    sdk_dir.mkdir()
    write_bundle_manifest(
        bundle_dir,
        sdk_source_dir=sdk_dir,
        sdk_requirements=("jsonschema-path>=0.3",),
    )

    from extensions.sop_converter import bundle_venv

    monkeypatch.setattr(bundle_venv, "is_venv_ready", lambda *_args: False)
    context = ToolContext(
        workspace_root=tmp_path,
        bundle_context=SimpleNamespace(bundle_path=bundle_dir),
    )

    with pytest.raises(BashCallError, match="bundle_venv_not_ready"):
        resolve_bundle_venv_environment(context)

    tool = build_tool_from_spec(
        AgentToolSpec(
            name="demo-create-agent",
            description="demo",
            input_schema={"type": "object", "properties": {}},
            call_type="bash",
            call_impl='python3 "/tmp/wrapper.py" create_agent \'{json_args}\'',
            source="sop-converter",
        )
    )
    result = tool.call({}, context)
    assert result.is_error
    assert result.output["error_code"] == "bundle_venv_not_ready"
    assert result.output["recovery"] == "rerun_sop_convert"


def test_execute_bash_applies_resolved_bundle_environment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle_site = str(tmp_path / "bundle-venv" / "lib" / "site-packages")
    captured: dict[str, object] = {}

    monkeypatch.setattr(
        bash_handler,
        "resolve_bundle_venv_environment",
        lambda _context: {
            "PYTHONPATH": bundle_site,
            "VIRTUAL_ENV": str(tmp_path / "bundle-venv"),
        },
    )

    def fake_run(argv, **kwargs):
        captured["argv"] = argv
        captured["kwargs"] = kwargs
        return 0, "{}", "", False, False

    monkeypatch.setattr(bash_handler, "_run_subprocess_with_abort", fake_run)

    output = bash_handler.execute_bash(
        'python3 "/tmp/wrapper.py" create_agent \'{json_args}\' --catalog-metadata \'{"resource_type":"agent"}\'',
        {"json_args": '{"id":"verify-bot"}'},
        context=ToolContext(workspace_root=tmp_path),
    )

    assert output == "{}"
    assert captured["argv"][0] == sys.executable
    assert captured["kwargs"]["env"]["PYTHONPATH"] == bundle_site


def test_execute_bash_imports_bundle_site_packages(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle_site = tmp_path / "bundle-venv" / "lib" / "site-packages"
    bundle_site.mkdir(parents=True)
    (bundle_site / "fake_bundle_dep.py").write_text(
        'VALUE = "from-bundle-venv"\n',
        encoding="utf-8",
    )
    wrapper = tmp_path / "catalog_wrapper.py"
    wrapper.write_text(
        "import json\nimport fake_bundle_dep\nprint(json.dumps({'value': fake_bundle_dep.VALUE}))\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        bash_handler,
        "resolve_bundle_venv_environment",
        lambda _context: {
            "PYTHONPATH": str(bundle_site),
            "VIRTUAL_ENV": str(bundle_site.parents[2]),
        },
    )

    output = bash_handler.execute_bash(
        f'python3 "{wrapper}" create_agent \'{{json_args}}\' --catalog-metadata \'{{"resource_type":"agent"}}\'',
        {"json_args": "{}"},
        context=ToolContext(workspace_root=tmp_path),
    )

    assert json.loads(output) == {"value": "from-bundle-venv"}
