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

import dataclasses
import json

import pytest

from clawcodex_ext.configuration import (
    ConfigMutationRequest,
    ConfigurationError,
    configuration_json_schema,
    get_configuration_contract,
    infer_configuration_domain,
    invalidate_configuration,
    mutate_configuration,
)
from clawcodex_ext.settings.types import SettingsSchema
from src.tool_system.context import ToolContext


@pytest.fixture
def contract_paths(tmp_path, monkeypatch):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    user_app = tmp_path / "home" / ".clawcodex" / "config.json"
    settings_dir = tmp_path / "claude-home"
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(settings_dir))
    monkeypatch.setattr("src.config.get_config_path", lambda: user_app)
    monkeypatch.setattr("src.config._find_git_root", lambda _cwd=None: workspace)
    invalidate_configuration("contract test setup")
    yield workspace, user_app, settings_dir / "settings.json"
    invalidate_configuration("contract test teardown")


def _context(workspace):
    return ToolContext(workspace_root=workspace, cwd=workspace, workspace_trusted=True)


def test_every_settings_dataclass_field_derives_to_settings_domain():
    fields = {field.name for field in dataclasses.fields(SettingsSchema)} - {"extra"}

    assert fields
    assert {name for name in fields if infer_configuration_domain(name) != "settings"} == set()


@pytest.mark.parametrize(
    ("setting", "value"),
    [
        ("advisor_enabled", True),
        ("dialogue_enabled", True),
        ("dialogue_interim_results", False),
        ("dialogue_modality", "audio"),
        ("dialogue_provider", "minimax"),
        ("dialogue_voice", "voice-1"),
        ("disable_workflows", True),
        ("freeze.threshold_s", 30.0),
        ("model_provider", "openai"),
        ("spinner_verbs.mode", "append"),
        ("tts_enabled", True),
        ("tts_provider", "openai"),
        ("tts_silent_text_output", True),
        ("tts_voice", "alloy"),
        ("voice_enabled", True),
        ("voice_provider", "doubao"),
    ],
)
def test_previously_missing_roots_write_to_settings_without_explicit_domain(
    contract_paths,
    setting,
    value,
):
    workspace, user_app, user_settings = contract_paths

    result = mutate_configuration(ConfigMutationRequest(setting=setting, value=value), _context(workspace))

    assert result.domain.value == "settings"
    assert result.path == user_settings
    assert not user_app.exists()


def test_explicit_wrong_domain_is_actionable_and_has_no_side_effect(contract_paths):
    workspace, user_app, user_settings = contract_paths

    with pytest.raises(ConfigurationError, match="belongs to the 'settings' domain"):
        mutate_configuration(
            ConfigMutationRequest(setting="voice_enabled", value=True, domain="app"),
            _context(workspace),
        )

    assert not user_app.exists()
    assert not user_settings.exists()


def test_user_only_app_field_rejects_project_scope(contract_paths):
    workspace, _user_app, _user_settings = contract_paths

    with pytest.raises(ConfigurationError, match="allowed scopes: user"):
        mutate_configuration(
            ConfigMutationRequest(setting="theme", value="dark", scope="project"),
            _context(workspace),
        )

    assert not (workspace / ".claude" / "config.json").exists()


@pytest.mark.parametrize(
    ("setting", "domain", "manager"),
    [
        ("telemetry.enabled", "app", "/telemetry"),
        ("plugins.demo", "app", "plugin loader"),
        ("daemon.enabled", "app", "clawcodex daemon"),
        ("mcp_servers.demo", "settings", "clawcodex mcp"),
    ],
)
def test_dedicated_subsystems_are_readable_but_not_mutable_through_config(
    contract_paths,
    setting,
    domain,
    manager,
):
    workspace, user_app, user_settings = contract_paths
    context = _context(workspace)

    read = mutate_configuration(ConfigMutationRequest(setting=setting, domain=domain), context)
    assert read.success is True
    with pytest.raises(ConfigurationError, match=manager):
        mutate_configuration(
            ConfigMutationRequest(setting=setting, value=True, domain=domain),
            context,
        )
    assert not user_app.exists()
    assert not user_settings.exists()


def test_contract_schema_is_stable_value_free_and_covers_extensions(contract_paths):
    _workspace, user_app, _user_settings = contract_paths
    user_app.parent.mkdir(parents=True)
    user_app.write_text(
        json.dumps(
            {
                "providers": {
                    "private-provider": {
                        "api_key": "top-secret",
                        "base_url": "https://private.invalid",
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    app_schema = configuration_json_schema("app")
    settings_schema = configuration_json_schema("settings")
    encoded = json.dumps(app_schema)

    assert {"away_summary", "intent_forecast", "agentRouting", "agentModels"} <= set(settings_schema["properties"])
    assert {"theme", "logoColor", "editorMode", "copyFullResponse", "selection_mode"} <= set(app_schema["properties"])
    assert "private-provider" not in encoded
    assert "top-secret" not in encoded
    assert "https://private.invalid" not in encoded
    assert all("x-scopes" in field and "x-secret" in field for field in app_schema["properties"].values())
    provider_shape = app_schema["properties"]["providers"]["additionalProperties"]
    assert provider_shape["properties"]["api_key"]["x-secret"] is True


@pytest.mark.parametrize(
    ("setting", "value", "message"),
    [
        ("voice_provider", "unknown", "must be one of"),
        ("dialogue_modality", "video", "must be one of"),
        ("freeze.threshold_s", -1, "must be >= 0"),
        ("intent_forecast.min_confidence", 2.0, "must be <= 1"),
    ],
)
def test_contract_constraints_reject_invalid_writes_atomically(
    contract_paths,
    setting,
    value,
    message,
):
    workspace, _user_app, user_settings = contract_paths
    user_settings.parent.mkdir(parents=True)
    user_settings.write_text('{"unknown": {"keep": true}}\n', encoding="utf-8")
    before = user_settings.read_bytes()

    with pytest.raises(ConfigurationError, match=message):
        mutate_configuration(ConfigMutationRequest(setting=setting, value=value), _context(workspace))

    assert user_settings.read_bytes() == before


def test_internal_app_state_is_not_advertised_but_unknown_fields_remain_compatible():
    app_names = {field.name for field in get_configuration_contract() if field.domain == "app"}

    assert {"projects", "user_id", "companion_pet_at"}.isdisjoint(app_names)
    assert infer_configuration_domain("projects./workspace.hasTrustDialogAccepted") == "app"


def test_contract_callers_cannot_mutate_future_app_schema_results():
    first = get_configuration_contract()
    theme = next(field for field in first if field.name == "theme")
    theme.schema["injected"] = True

    refreshed = configuration_json_schema("app")
    assert "injected" not in refreshed["properties"]["theme"]
