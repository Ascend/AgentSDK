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

from clawcodex_ext.cli.model_cmd.registry import ModelRegistry
from clawcodex_ext.cli.model_cmd.resolver import resolve


def test_resolve_prefers_cli_provider_and_model(monkeypatch) -> None:
    monkeypatch.setenv("CLAWCODEX_PROVIDER", "glm")
    monkeypatch.setenv("CLAWCODEX_MODEL", "zai/glm-4")
    monkeypatch.setattr("clawcodex_ext.cli.model_cmd.resolver.get_default_provider", lambda: "openai")

    resolution = resolve(cli_provider="anthropic", cli_model="claude-sonnet-4-6")

    assert resolution.provider == "anthropic"
    assert resolution.model == "claude-sonnet-4-6"
    assert resolution.provider_source == "cli"
    assert resolution.model_source == "cli"


def test_resolve_uses_environment_before_user_defaults(monkeypatch) -> None:
    monkeypatch.setenv("CLAWCODEX_PROVIDER", "glm")
    monkeypatch.setenv("CLAWCODEX_MODEL", "zai/glm-4")
    monkeypatch.setattr("clawcodex_ext.cli.model_cmd.resolver.get_default_provider", lambda: "anthropic")

    resolution = resolve()

    assert resolution.provider == "zai"
    assert resolution.model == "zai/glm-4"
    assert resolution.provider_source == "env"
    assert resolution.model_source == "env"


def test_resolve_falls_back_from_invalid_configured_model(monkeypatch) -> None:
    monkeypatch.delenv("CLAWCODEX_PROVIDER", raising=False)
    monkeypatch.delenv("CLAWCODEX_MODEL", raising=False)
    monkeypatch.setattr("clawcodex_ext.cli.model_cmd.resolver.get_default_provider", lambda: "glm")
    monkeypatch.setattr(
        "clawcodex_ext.cli.model_cmd.resolver.get_provider_config",
        lambda provider: {"default_model": "not-valid-for-glm"},
    )

    resolution = resolve()

    assert resolution.provider == "zai"
    assert resolution.model == "not-valid-for-glm"
    assert resolution.provider_source == "user"
    assert resolution.model_source == "user-warn"


def test_resolve_inferrs_provider_from_cli_model(monkeypatch) -> None:
    monkeypatch.delenv("CLAWCODEX_PROVIDER", raising=False)
    monkeypatch.delenv("CLAWCODEX_MODEL", raising=False)
    monkeypatch.setattr("clawcodex_ext.cli.model_cmd.resolver.get_default_provider", lambda: "anthropic")

    resolution = resolve(cli_model="zai/glm-4")

    assert resolution.provider == "zai"
    assert resolution.model == "zai/glm-4"
    assert resolution.provider_source == "cli-model"
    assert resolution.model_source == "cli"


def test_resolve_trusts_explicit_model_newer_than_local_catalog() -> None:
    registry = ModelRegistry(
        provider_info={
            "openai-codex": {
                "label": "OpenAI Codex",
                "default_base_url": "https://example.test",
                "default_model": "gpt-5.5",
                "available_models": ["gpt-5.5"],
            }
        },
        discovery_hooks={},
    )

    resolution = resolve(
        cli_provider="openai-codex",
        cli_model="gpt-5.6-sol",
        registry=registry,
    )

    assert resolution.model == "gpt-5.6-sol"
    assert resolution.model_source == "cli"


def test_resolve_saved_model_uses_local_catalog_without_discovery(monkeypatch) -> None:
    discovery_calls: list[None] = []
    monkeypatch.delenv("CLAWCODEX_PROVIDER", raising=False)
    monkeypatch.delenv("CLAWCODEX_MODEL", raising=False)
    monkeypatch.setattr(
        "clawcodex_ext.cli.model_cmd.resolver.get_default_provider",
        lambda: "openai-codex",
    )
    monkeypatch.setattr(
        "clawcodex_ext.cli.model_cmd.resolver.get_provider_config",
        lambda provider: {"default_model": "gpt-5.5"},
    )
    registry = ModelRegistry(
        provider_info={
            "openai-codex": {
                "label": "OpenAI Codex",
                "default_base_url": "https://example.test",
                "default_model": "gpt-5.5",
                "available_models": ["gpt-5.5"],
            }
        },
        discovery_hooks={"openai-codex": [lambda: discovery_calls.append(None) or ["gpt-live-model"]]},
    )

    resolution = resolve(registry=registry)

    assert resolution.provider == "openai-codex"
    assert resolution.model == "gpt-5.5"
    assert resolution.model_source == "user"
    assert discovery_calls == []
