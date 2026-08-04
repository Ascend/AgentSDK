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

"""Resolve effective provider/model from CLI, env, config, and defaults."""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path

from clawcodex_ext.cli.model_cmd.registry import ModelRegistry
from clawcodex_ext.cli.provider_cmd.errors import UnknownProviderError
from src.config import get_default_provider, get_provider_config


@dataclass(frozen=True)
class Resolution:
    provider: str
    model: str
    provider_source: str
    model_source: str


def resolve(
    *,
    cli_provider: str | None = None,
    cli_model: str | None = None,
    project_root: Path | None = None,
    registry: ModelRegistry | None = None,
) -> Resolution:
    del project_root
    registry = registry or ModelRegistry()

    env_provider = _nonempty(os.environ.get("CLAWCODEX_PROVIDER"))
    env_model = _nonempty(os.environ.get("CLAWCODEX_MODEL"))

    provider = _nonempty(cli_provider)
    provider_source = "cli" if provider else ""
    if provider is None and env_provider:
        provider = env_provider
        provider_source = "env"
    if provider is None and cli_model:
        try:
            provider = registry.infer_provider_for_model(cli_model, discover=False)
            provider_source = "cli-model"
        except Exception:  # nosec B110
            pass
    if provider is None and env_model:
        try:
            provider = registry.infer_provider_for_model(env_model, discover=False)
            provider_source = "env-model"
        except Exception:  # nosec B110
            pass
    if provider is None:
        provider = get_default_provider()
        provider_source = "user"

    provider_unknown = False
    try:
        provider = registry.validate_provider(provider)
    except UnknownProviderError:
        provider_unknown = True
        print(
            f"Warning: provider '{provider}' is not in the built-in list — proceeding anyway",
            file=sys.stderr,
        )

    model = _nonempty(cli_model)
    model_source = "cli" if model else ""
    if model is None and env_model:
        model = env_model
        model_source = "env"
    if model is None:
        try:
            provider_cfg = get_provider_config(provider) or {}
        except ValueError:
            provider_cfg = {}
        configured_model = _nonempty(provider_cfg.get("default_model"))
        if configured_model:
            if not provider_unknown:
                local_models = set(registry.configured_models(provider))
                configured_models = provider_cfg.get("models")
                if isinstance(configured_models, list):
                    local_models.update(model for model in configured_models if isinstance(model, str) and model)
                if configured_model in local_models:
                    model = configured_model
                    model_source = "user"
                else:
                    # Startup resolution must stay local.  A provider may
                    # expose newer models than the cached/configured catalog,
                    # so trust the saved value and let the provider validate it
                    # on the first request instead of blocking on discovery.
                    print(
                        f"Warning: model '{configured_model}' is not in the known list "
                        f"for provider '{provider}' — using it anyway (saved config)",
                        file=sys.stderr,
                    )
                    model = configured_model
                    model_source = "user-warn"
            else:
                # Unknown provider — trust the configured model as-is
                model = configured_model
                model_source = "user"
    if model is None and not provider_unknown:
        model = registry.provider_default_model(provider)
        model_source = "default"
    elif model is None and provider_unknown:
        # No configured model and unknown provider — use provider name as model
        model = provider
        model_source = "fallback"

    # Every remaining model source is already authoritative: explicit CLI/env
    # values are session choices, saved values were checked locally above, and
    # provider defaults come from this same registry.  Do not perform another
    # live catalog lookup as final validation.
    return Resolution(
        provider=provider,
        model=model,
        provider_source=provider_source,
        model_source=model_source,
    )


def _nonempty(value: str | None) -> str | None:
    if value is None:
        return None
    stripped = value.strip()
    return stripped or None
