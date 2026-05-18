#!/usr/bin/env python3
# coding=utf-8
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

import sys
import types
import unittest
import importlib
from unittest.mock import patch, MagicMock


def _build_fake_torch_modules():
    """Create fake torch modules to avoid real torch import."""
    fake_torch = types.ModuleType("torch")
    fake_torch.__dict__["__version__"] = "0.0.fake"
    fake_torch_nn = types.ModuleType("torch.nn")
    fake_torch_nn_func = types.ModuleType("torch.nn.functional")
    fake_torch.nn = fake_torch_nn
    fake_torch.nn.functional = fake_torch_nn_func
    return {
        "torch": fake_torch,
        "torch.nn": fake_torch_nn,
        "torch.nn.functional": fake_torch_nn_func,
    }


def _build_fake_loggers_module():
    """Create a fake loggers module that provides Loggers class."""
    fake_loggers_mod = types.ModuleType("aura.base.log.loggers")

    class FakeLoggers:
        def __init__(self, *args, **kwargs):
            pass

        def get_logger(self):
            return MagicMock()

    fake_loggers_mod.Loggers = FakeLoggers
    return fake_loggers_mod


class TestChatProxy(unittest.IsolatedAsyncioTestCase):
    """Unit tests for chat_proxy module - fully isolated, no global pollution."""

    def setUp(self):
        # Clear module cache to force a clean reload for each test
        module_path = "aura.runner.agent_service.chat_proxy"
        if module_path in sys.modules:
            del sys.modules[module_path]

        # Build fake dependencies
        fake_torch_mods = _build_fake_torch_modules()
        fake_loggers_mod = _build_fake_loggers_module()
        mock_modules = {
            "ray": MagicMock(),
            "aura.base.log.loggers": fake_loggers_mod,
            **fake_torch_mods,
        }

        # Import the module under test with mocked dependencies
        with patch.dict(sys.modules, mock_modules):
            import aura.runner.agent_service.chat_proxy as mod

            self.chat_proxy = importlib.reload(mod)

        # Remove from sys.modules to avoid cross-test pollution
        if module_path in sys.modules:
            del sys.modules[module_path]

        # Reset internal global flag (already False after reload, but explicit)
        self.chat_proxy._PATCHED = False

        # Prepare real openai reference for patching
        import openai

        self.AsyncOpenAI = openai.AsyncOpenAI
        self._orig_init = self.AsyncOpenAI.__init__

    def tearDown(self):
        # Restore original AsyncOpenAI.__init__
        self.AsyncOpenAI.__init__ = self._orig_init
        # Reset module global flag to prevent cross-test interference
        if hasattr(self, 'chat_proxy'):
            self.chat_proxy._PATCHED = False

    # ------------------------------------------------------------
    # Tests for patch_async_openai_global
    # ------------------------------------------------------------
    def test_patch_async_openai_global_sets_flag_and_replaces_init(self):
        """Global patching should set _PATCHED and replace AsyncOpenAI.__init__."""
        original_init = self.AsyncOpenAI.__init__
        self.chat_proxy.patch_async_openai_global({"model": "test_model"})
        self.assertTrue(self.chat_proxy._PATCHED)
        self.assertNotEqual(self.AsyncOpenAI.__init__, original_init)

    def test_patch_async_openai_global_idempotent(self):
        """Calling patch_async_openai_global twice should not break idempotency."""
        original_init = self.AsyncOpenAI.__init__
        self.chat_proxy._PATCHED = True
        self.chat_proxy.patch_async_openai_global({"model": "test_model"})
        self.assertEqual(self.AsyncOpenAI.__init__, original_init)

    def test_patch_async_openai_global_handles_missing_completions_attr(self):
        """If client has no 'completions' attribute, patching should skip safely."""
        with patch("openai.AsyncOpenAI.__init__", return_value=None):
            self.chat_proxy.patch_async_openai_global({"model": "test_model"})
            client = self.AsyncOpenAI()
            # completions not set
            self.AsyncOpenAI.__init__(client)
            self.assertTrue(self.chat_proxy._PATCHED)

    def test_patch_async_openai_global_handles_missing_create_attr(self):
        """If completions has no 'create' attribute, patching should skip safely."""
        with patch("openai.AsyncOpenAI.__init__", return_value=None):
            self.chat_proxy.patch_async_openai_global({"model": "test_model"})
            client = self.AsyncOpenAI()
            client.completions = MagicMock()
            del client.completions.create
            self.AsyncOpenAI.__init__(client)
            self.assertTrue(self.chat_proxy._PATCHED)


if __name__ == "__main__":
    unittest.main()
