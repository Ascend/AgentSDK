#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# -------------------------------------------------------------------------
#  This file is part of the AgentSDK project.
# Copyright (c) 2026 Huawei Technologies Co.,Ltd.
# Copyright (c) 2026 Clawd Codex Team
#
# AgentSDK is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#
#           http://license.coscl.org.cn/MulanPSL2
#
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
# -------------------------------------------------------------------------

from __future__ import annotations

import json
from pathlib import Path

import pytest

from clawcodex_ext.configuration import (
    ConfigDomain,
    ConfigMutationRequest,
    ConfigOperation,
    ConfigScope,
    ConfigurationError,
    apply_configuration_snapshot,
    get_configuration_snapshot,
    invalidate_configuration,
    mutate_configuration,
    set_effort,
)
from clawcodex_ext.tool_system.tools.config import ConfigTool
from src.permissions.types import ToolPermissionContext
from src.tool_system.context import ToolContext


@pytest.fixture
def configuration_paths(tmp_path, monkeypatch):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    user_app = tmp_path / "home" / ".clawcodex" / "config.json"
    user_settings_dir = tmp_path / "claude-home"
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(user_settings_dir))
    monkeypatch.setattr("src.config.get_config_path", lambda: user_app)
    monkeypatch.setattr("src.config._find_git_root", lambda _cwd=None: workspace)
    invalidate_configuration("test setup")
    yield workspace, user_app, user_settings_dir / "settings.json"
    invalidate_configuration("test teardown")


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _context(workspace: Path, *, trusted: bool = False) -> ToolContext:
    return ToolContext(
        workspace_root=workspace,
        cwd=workspace,
        permission_context=ToolPermissionContext(mode="default"),
        workspace_trusted=trusted,
    )


def test_snapshot_precedence_legacy_compatibility_and_immutability(configuration_paths):
    workspace, user_app, user_settings = configuration_paths
    _write(
        user_app,
        {
            "unknownApp": {"keep": True},
            "settings": {
                "model": "legacy-user",
                "hookRuntime": {"timeout_ms": 111},
            },
        },
    )
    _write(user_settings, {"model": "canonical-user", "unknownSetting": 1})
    _write(
        workspace / ".claude" / "config.json",
        {"default_provider": "openai", "settings": {"model": "legacy-project"}},
    )
    _write(workspace / ".claude" / "settings.json", {"model": "canonical-project"})
    _write(workspace / ".claude" / "settings.local.json", {"model": "local"})

    snapshot = get_configuration_snapshot(workspace)

    assert snapshot.app["default_provider"] == "openai"
    assert snapshot.app["unknownApp"]["keep"] is True
    assert snapshot.settings["model"] == "local"
    assert snapshot.settings["unknownSetting"] == 1
    assert snapshot.typed_settings["hooks"]["timeout_ms"] == 111
    with pytest.raises(TypeError):
        snapshot.settings["model"] = "mutated"


def test_mutations_preserve_unknown_fields_and_never_write_merged_global(configuration_paths):
    workspace, user_app, _user_settings = configuration_paths
    _write(user_app, {"userOnly": 1, "unknown": {"keep": True}})
    project_app = workspace / ".claude" / "config.json"
    _write(project_app, {"projectOnly": 2})

    context = _context(workspace, trusted=True)
    result = mutate_configuration(
        ConfigMutationRequest(
            setting="projectFeature.enabled",
            value=True,
            scope=ConfigScope.PROJECT,
            domain=ConfigDomain.APP,
        ),
        context,
    )

    assert result.success is True
    assert json.loads(project_app.read_text(encoding="utf-8")) == {
        "projectOnly": 2,
        "projectFeature": {"enabled": True},
    }
    assert json.loads(user_app.read_text(encoding="utf-8")) == {
        "userOnly": 1,
        "unknown": {"keep": True},
    }


def test_array_operations_and_atomic_validation_failure(configuration_paths):
    workspace, _user_app, user_settings = configuration_paths
    _write(user_settings, {"allowed_tools": ["Read"], "unknown": {"keep": True}})

    context = _context(workspace)
    mutate_configuration(
        ConfigMutationRequest(
            setting="allowed_tools",
            value=["Read", "Bash"],
            domain="settings",
            operation="append_unique",
        ),
        context,
    )
    after_append = json.loads(user_settings.read_text(encoding="utf-8"))
    assert after_append["allowed_tools"] == ["Read", "Bash"]
    assert after_append["unknown"] == {"keep": True}

    mutate_configuration(
        ConfigMutationRequest(
            setting="allowed_tools",
            value="Read",
            domain="settings",
            operation=ConfigOperation.REMOVE,
        ),
        context,
    )
    assert json.loads(user_settings.read_text(encoding="utf-8"))["allowed_tools"] == ["Bash"]

    before = user_settings.read_bytes()
    with pytest.raises(ConfigurationError, match="settings.env"):
        mutate_configuration(
            ConfigMutationRequest(setting="env.BAD", value=123, domain="settings"),
            context,
        )
    assert user_settings.read_bytes() == before


def test_secret_scope_redaction_and_project_trust_gate(configuration_paths):
    workspace, user_app, _user_settings = configuration_paths
    _write(user_app, {"providers": {"openai": {"api_key": "secret-value"}}})
    untrusted = _context(workspace)

    read_result = mutate_configuration(
        ConfigMutationRequest(setting="providers.openai.api_key"),
        untrusted,
    )
    assert read_result.to_dict()["value"] == "***REDACTED***"

    with pytest.raises(ConfigurationError, match="user scope"):
        mutate_configuration(
            ConfigMutationRequest(
                setting="providers.openai.api_key",
                value="nope",
                scope="project",
            ),
            untrusted,
        )

    with pytest.raises(ConfigurationError, match="workspace trust"):
        mutate_configuration(
            ConfigMutationRequest(
                setting="permissions.allow",
                value=["Bash(git status)"],
                scope="project",
                domain="settings",
            ),
            untrusted,
        )


def test_runtime_projection_respects_trust_and_refreshes_without_restart(configuration_paths):
    workspace, _user_app, user_settings = configuration_paths
    _write(
        user_settings,
        {
            "env": {"USER_FLAG": "yes"},
            "permissions": {"allow": ["Read"]},
            "hooks": {"PreToolUse": [{"type": "command", "command": "echo user"}]},
        },
    )
    project_settings = workspace / ".claude" / "settings.json"
    _write(
        project_settings,
        {
            "env": {"PROJECT_FLAG": "yes"},
            "permissions": {"deny": ["Bash"]},
            "hooks": {"PostToolUse": [{"type": "command", "command": "echo project"}]},
        },
    )

    context = _context(workspace)
    apply_configuration_snapshot(context, get_configuration_snapshot(workspace))
    assert context.env == {"USER_FLAG": "yes"}
    assert context.permission_context.always_allow_rules["userSettings"] == ["Read"]
    assert "projectSettings" not in context.permission_context.always_deny_rules
    assert "PreToolUse" in context.hook_config_manager.snapshot.hooks
    assert "PostToolUse" not in context.hook_config_manager.snapshot.hooks

    context.workspace_trusted = True
    apply_configuration_snapshot(context, get_configuration_snapshot(workspace))
    assert context.env["PROJECT_FLAG"] == "yes"
    assert context.permission_context.always_deny_rules["projectSettings"] == ["Bash"]
    assert "PostToolUse" in context.hook_config_manager.snapshot.hooks

    mutate_configuration(
        ConfigMutationRequest(
            setting="permissions.allow",
            value="Glob",
            domain="settings",
            operation="append_unique",
        ),
        context,
    )
    assert context.permission_context.always_allow_rules["userSettings"] == ["Read", "Glob"]


def test_atomic_replace_failure_leaves_original_untouched(configuration_paths, monkeypatch):
    workspace, _user_app, user_settings = configuration_paths
    _write(user_settings, {"model": "before", "unknown": True})
    before = user_settings.read_bytes()

    def fail_replace(_source, _destination):
        raise OSError("simulated replace failure")

    monkeypatch.setattr("clawcodex_ext.configuration.service.os.replace", fail_replace)
    with pytest.raises(ConfigurationError, match="atomically update"):
        mutate_configuration(
            ConfigMutationRequest(setting="model", value="after", domain="settings"),
            _context(workspace),
        )

    assert user_settings.read_bytes() == before
    assert not list(user_settings.parent.glob(".settings.json.*.tmp"))


def test_effort_writer_uses_canonical_settings_file(configuration_paths):
    workspace, user_app, user_settings = configuration_paths
    _write(user_app, {"unknown": "preserved"})

    set_effort("high")

    assert json.loads(user_settings.read_text(encoding="utf-8"))["effort"] == "high"
    assert json.loads(user_app.read_text(encoding="utf-8")) == {"unknown": "preserved"}


def test_config_tool_supports_structured_operations_and_legacy_input(configuration_paths):
    workspace, _user_app, _user_settings = configuration_paths
    context = _context(workspace, trusted=True)

    old_result = ConfigTool.call(
        {"setting": "default_provider", "value": "openai"},
        context,
    ).output
    assert old_result["operation"] == "set"

    structured = ConfigTool.call(
        {
            "setting": "permissions.allow",
            "value": "Read",
            "scope": "project",
            "domain": "settings",
            "operation": "append_unique",
        },
        context,
    ).output
    assert structured["operation"] == "append_unique"
    assert structured["scope"] == "project"
    assert structured["domain"] == "settings"
