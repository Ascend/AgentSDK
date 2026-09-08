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

"""Config helper for stability-gate tests."""

from __future__ import annotations

import json
import os
from pathlib import Path


def make_config(home_path: Path, provider: str = "anthropic") -> Path:
    """Create a fake .clawcodex/config.json at *home_path*.

    Returns the config file path.
    """
    config_dir = home_path / ".clawcodex"
    config_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "default_provider": provider,
        "providers": {
            provider: {
                "api_key": "fake-stability-gate-key",
                "base_url": "https://api.anthropic.com/v1",
                "default_model": "claude-sonnet-4-20250514",
            }
        },
    }
    config_file = config_dir / "config.json"
    config_file.write_text(json.dumps(payload), encoding="utf-8")
    return config_file


def redirect_global_config(config_file: Path):
    """Redirect the shared config chain at *config_file*'s directory.

    Returns an object with ``.stop()`` restoring the previous env values.
    """
    import src.config as config_module

    saved = {name: os.environ.get(name) for name in ("CLAWCODEX_CONFIG_DIR", "CLAWCODEX_HOME")}
    for name in saved:
        os.environ[name] = str(config_file.parent)
    config_module._default_manager = None

    class _Restore:
        def stop(self):
            for name, value in saved.items():
                if value is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = value

    return _Restore()


def cleanup_config():
    """Reset ConfigManager singleton."""
    import src.config as config_module

    config_module._default_manager = None
