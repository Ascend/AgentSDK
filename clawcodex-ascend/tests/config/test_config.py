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

"""Tests for configuration management (legacy compat tests).

Updated to work with the new ConfigManager-based config system (WS-6 rewrite).
"""

from __future__ import annotations

import unittest
from pathlib import Path
import tempfile
import os
import contextlib

import src.config as config_module
from src.config import (
    get_config_path,
    get_default_config,
    load_config,
    save_config,
    get_provider_config,
    set_api_key,
    set_default_provider,
    get_default_provider,
)
from src.utils.clawcodex_dirs import CONFIG_DIR_ENV, HOME_DIR_ENV


@contextlib.contextmanager
def _env_redirect(changes: dict[str, str | None]):
    """Set env entries for the block, restoring previous values afterwards."""
    saved: dict[str, str | None] = {}
    for name, value in changes.items():
        saved[name] = os.environ.get(name)
        if value is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = value
    try:
        yield
    finally:
        for name, value in saved.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def _patch_global_config(temp_dir):
    """Isolate the global config tier under ``temp_dir/.clawcodex``."""
    root = Path(temp_dir) / ".clawcodex"
    return _env_redirect({CONFIG_DIR_ENV: str(root)})


def _reset_manager():
    """Reset the cached default ConfigManager."""
    config_module._default_manager = None


class TestConfigPath(unittest.TestCase):
    """Test configuration path functions."""

    def test_get_config_path(self):
        """Test getting config path returns the global config path."""
        path = get_config_path()
        self.assertTrue(str(path).endswith("config.json"))

    def test_config_path_is_in_home(self):
        """Default chain (no env overrides) resolves under ``~/.clawcodex``."""
        with _env_redirect({CONFIG_DIR_ENV: None, HOME_DIR_ENV: None}):
            path = get_config_path()
        self.assertIn(".clawcodex", str(path))
        self.assertTrue(str(path).startswith(str(Path.home())))

    def test_config_path_follows_env_chain(self):
        """``CLAWCODEX_CONFIG_DIR`` wins; ``CLAWCODEX_HOME`` is the fallback."""
        with tempfile.TemporaryDirectory() as config_dir, tempfile.TemporaryDirectory() as home_dir:
            with _env_redirect({CONFIG_DIR_ENV: config_dir, HOME_DIR_ENV: home_dir}):
                self.assertEqual(Path(get_config_path()), Path(config_dir) / "config.json")
            with _env_redirect({CONFIG_DIR_ENV: None, HOME_DIR_ENV: home_dir}):
                self.assertEqual(Path(get_config_path()), Path(home_dir) / "config.json")


class TestDefaultConfig(unittest.TestCase):
    """Test default configuration."""

    def test_get_default_config(self):
        """Test getting default config."""
        config = get_default_config()

        self.assertIn("default_provider", config)
        self.assertIn("providers", config)
        self.assertIn("anthropic", config["providers"])
        self.assertIn("openai", config["providers"])
        self.assertIn("zai", config["providers"])

    def test_default_provider_is_anthropic(self):
        """Test that default provider is Anthropic."""
        config = get_default_config()
        self.assertEqual(config["default_provider"], "anthropic")

    def test_default_models(self):
        """Test default models for providers."""
        config = get_default_config()
        self.assertEqual(config["providers"]["anthropic"]["default_model"], "claude-sonnet-4-6")
        self.assertEqual(config["providers"]["openai"]["default_model"], "gpt-5.4")
        self.assertEqual(config["providers"]["zai"]["default_model"], "GLM-5.1")


class TestLoadSaveConfig(unittest.TestCase):
    """Test loading and saving configuration."""

    def setUp(self):
        _reset_manager()

    def tearDown(self):
        _reset_manager()

    def test_save_and_load_config(self):
        """Test save and load roundtrip."""
        with tempfile.TemporaryDirectory() as temp_dir:
            with _patch_global_config(temp_dir):
                _reset_manager()
                config = {
                    "default_provider": "glm",
                    "providers": {
                        "glm": {
                            "api_key": "test_key",
                            "base_url": "https://example.com",
                            "default_model": "glm-4",
                        }
                    },
                }

                save_config(config)
                _reset_manager()
                loaded = load_config()

                self.assertEqual(loaded["default_provider"], "glm")
                self.assertEqual(loaded["providers"]["glm"]["api_key"], "test_key")

    def test_load_config_returns_dict(self):
        """Test that loading config returns a valid dict."""
        with tempfile.TemporaryDirectory() as temp_dir:
            with _patch_global_config(temp_dir):
                _reset_manager()
                config = load_config()
                self.assertIsInstance(config, dict)

    @unittest.skipIf(os.name == "nt", "POSIX file permission semantics differ on Windows")
    def test_config_file_permissions_restricted_on_save(self):
        """Test that saved config uses owner-only permissions on POSIX systems."""
        with tempfile.TemporaryDirectory() as temp_dir:
            with _patch_global_config(temp_dir):
                _reset_manager()
                config_path = Path(temp_dir) / ".clawcodex" / "config.json"
                save_config(get_default_config())
                mode = config_path.stat().st_mode & 0o777
                self.assertEqual(mode, 0o600)


class TestProviderConfig(unittest.TestCase):
    """Test provider-specific configuration."""

    def setUp(self):
        _reset_manager()

    def tearDown(self):
        _reset_manager()

    def test_get_provider_config(self):
        """Test getting provider config."""
        with tempfile.TemporaryDirectory() as temp_dir:
            with _patch_global_config(temp_dir):
                _reset_manager()
                glm_config = get_provider_config("glm")

                self.assertIn("api_key", glm_config)
                self.assertIn("base_url", glm_config)
                self.assertIn("default_model", glm_config)

    def test_get_unknown_provider(self):
        """Test getting unknown provider."""
        with tempfile.TemporaryDirectory() as temp_dir:
            with _patch_global_config(temp_dir):
                _reset_manager()
                with self.assertRaises(ValueError) as context:
                    get_provider_config("unknown")

                self.assertIn("Unknown provider", str(context.exception))


class TestSetAPIKey(unittest.TestCase):
    """Test setting API keys."""

    def setUp(self):
        _reset_manager()

    def tearDown(self):
        _reset_manager()

    def test_set_api_key(self):
        """Test setting API key for provider."""
        with tempfile.TemporaryDirectory() as temp_dir:
            with _patch_global_config(temp_dir):
                _reset_manager()
                set_api_key("glm", "new_api_key")

                _reset_manager()
                config = load_config()
                self.assertEqual(config["providers"]["glm"]["api_key"], "new_api_key")

    def test_set_api_key_with_options(self):
        """Test setting API key with base URL and model."""
        with tempfile.TemporaryDirectory() as temp_dir:
            with _patch_global_config(temp_dir):
                _reset_manager()
                set_api_key(
                    "glm",
                    "new_api_key",
                    base_url="https://custom.url",
                    default_model="custom-model",
                )

                _reset_manager()
                config = load_config()
                self.assertEqual(config["providers"]["glm"]["api_key"], "new_api_key")
                self.assertEqual(config["providers"]["glm"]["base_url"], "https://custom.url")
                self.assertEqual(config["providers"]["glm"]["default_model"], "custom-model")


class TestDefaultProvider(unittest.TestCase):
    """Test default provider management."""

    def setUp(self):
        _reset_manager()

    def tearDown(self):
        _reset_manager()

    def test_set_default_provider(self):
        """Test setting default provider."""
        with tempfile.TemporaryDirectory() as temp_dir:
            with _patch_global_config(temp_dir):
                _reset_manager()
                set_default_provider("openai")

                _reset_manager()
                provider = get_default_provider()
                self.assertEqual(provider, "openai")

    def test_get_default_provider(self):
        """Test getting default provider."""
        with tempfile.TemporaryDirectory() as temp_dir:
            with _patch_global_config(temp_dir):
                _reset_manager()
                provider = get_default_provider()
                self.assertEqual(provider, "anthropic")


if __name__ == "__main__":
    unittest.main()
