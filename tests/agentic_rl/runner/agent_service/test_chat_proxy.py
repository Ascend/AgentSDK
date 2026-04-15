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

import os
import sys
import types
import unittest
import asyncio
import importlib
from unittest.mock import patch, MagicMock, AsyncMock


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
    fake_loggers_mod = types.ModuleType("agentic_rl.base.log.loggers")

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
        module_path = "agentic_rl.runner.agent_service.chat_proxy"
        if module_path in sys.modules:
            del sys.modules[module_path]

        # Build fake dependencies
        fake_torch_mods = _build_fake_torch_modules()
        fake_loggers_mod = _build_fake_loggers_module()
        mock_modules = {
            "ray": MagicMock(),
            "agentic_rl.base.log.loggers": fake_loggers_mod,
            **fake_torch_mods,
        }

        # Import the module under test with mocked dependencies
        with patch.dict(sys.modules, mock_modules):
            import agentic_rl.runner.agent_service.chat_proxy as mod
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
    # Tests for make_proxy_create
    # ------------------------------------------------------------
    def test_make_proxy_create_returns_callable(self):
        """make_proxy_create should return a callable."""
        proxy_func = self.chat_proxy.make_proxy_create(
            AsyncMock(),
            {"model": "test_model"},
        )
        self.assertTrue(callable(proxy_func))

    async def test_make_proxy_create_calls_infer_router_create_once(self):
        """The returned proxy should call InferRouter.create exactly once."""
        proxy_func = self.chat_proxy.make_proxy_create(
            AsyncMock(),
            {"model": "test_model"},
        )
        with patch.object(self.chat_proxy, 'Completion') as mock_completion_cls:
            mock_completion_cls.return_value = MagicMock()
            with patch.object(self.chat_proxy, 'InferRouter') as mock_router_cls:
                mock_router_instance = MagicMock()
                mock_router_cls.create = AsyncMock(return_value=mock_router_instance)
                mock_router_instance.completions = AsyncMock(return_value={"ok": True})
                await proxy_func(prompt="hello", max_tokens=10)
                mock_router_cls.create.assert_called_once()

    async def test_make_proxy_create_merges_infer_service_params(self):
        """Infer service parameters should be merged into completions kwargs."""
        infer_service_params = {"endpoint": "test_endpoint", "model": "test_model"}
        proxy_func = self.chat_proxy.make_proxy_create(AsyncMock(), infer_service_params)
        with patch.object(self.chat_proxy, 'Completion') as mock_completion_cls:
            mock_completion_cls.return_value = MagicMock()
            with patch.object(self.chat_proxy, 'InferRouter') as mock_router_cls:
                mock_router_instance = MagicMock()
                mock_router_cls.create = AsyncMock(return_value=mock_router_instance)
                mock_router_instance.completions = AsyncMock(return_value={"ok": True})
                await proxy_func(prompt="hi", max_tokens=20)
                expected_kwargs = {
                    "prompt": "hi",
                    "max_tokens": 20,
                    "endpoint": "test_endpoint",
                    "model": "test_model",
                }
                mock_router_instance.completions.assert_called_once_with(expected_kwargs)

    async def test_make_proxy_create_infer_params_override_user_params(self):
        """Infer service parameters should take precedence over user-provided ones."""
        infer_service_params = {"model": "proxy_model", "temperature": 0.7}
        proxy_func = self.chat_proxy.make_proxy_create(AsyncMock(), infer_service_params)
        with patch.object(self.chat_proxy, 'Completion') as mock_completion_cls:
            mock_completion_cls.return_value = MagicMock()
            with patch.object(self.chat_proxy, 'InferRouter') as mock_router_cls:
                mock_router_instance = MagicMock()
                mock_router_cls.create = AsyncMock(return_value=mock_router_instance)
                mock_router_instance.completions = AsyncMock(return_value={"ok": True})
                await proxy_func(
                    prompt="hello",
                    model="client_model",
                    temperature=0.1,
                    max_tokens=10,
                )
                called_kwargs = mock_router_instance.completions.call_args[0][0]
                self.assertEqual(called_kwargs["model"], "proxy_model")
                self.assertEqual(called_kwargs["temperature"], 0.7)
                self.assertEqual(called_kwargs["max_tokens"], 10)

    async def test_make_proxy_create_wraps_completion_object(self):
        """The proxy should instantiate a Completion object from the response."""
        proxy_func = self.chat_proxy.make_proxy_create(AsyncMock(), {"model": "test_model"})
        with patch.object(self.chat_proxy, 'Completion') as mock_completion_cls:
            mock_completion_obj = MagicMock()
            mock_completion_cls.return_value = mock_completion_obj
            with patch.object(self.chat_proxy, 'InferRouter') as mock_router_cls:
                mock_router_instance = MagicMock()
                mock_router_cls.create = AsyncMock(return_value=mock_router_instance)
                mock_resp = {"id": "abc", "model": "test_model"}
                mock_router_instance.completions = AsyncMock(return_value=mock_resp)
                result = await proxy_func(prompt="hello")
                self.assertEqual(result, mock_completion_obj)
                mock_completion_cls.assert_called_once_with(**mock_resp)

    async def test_make_proxy_create_backend_exception_propagates(self):
        """Exceptions from InferRouter.completions should bubble up."""
        proxy_func = self.chat_proxy.make_proxy_create(AsyncMock(), {"model": "test_model"})
        with patch.object(self.chat_proxy, 'InferRouter') as mock_router_cls:
            mock_router_instance = MagicMock()
            mock_router_cls.create = AsyncMock(return_value=mock_router_instance)
            mock_router_instance.completions = AsyncMock(side_effect=RuntimeError("backend error"))
            with self.assertRaises(RuntimeError) as ctx:
                await proxy_func(prompt="hello")
            self.assertIn("backend error", str(ctx.exception))

    async def test_make_proxy_create_completion_constructor_exception(self):
        """If Completion constructor fails, exception should propagate."""
        proxy_func = self.chat_proxy.make_proxy_create(AsyncMock(), {"model": "test_model"})
        with patch.object(self.chat_proxy, 'Completion') as mock_completion_cls:
            mock_completion_cls.side_effect = ValueError("Completion validation error")
            with patch.object(self.chat_proxy, 'InferRouter') as mock_router_cls:
                mock_router_instance = MagicMock()
                mock_router_cls.create = AsyncMock(return_value=mock_router_instance)
                mock_router_instance.completions = AsyncMock(return_value={"id": "x"})
                with self.assertRaises(ValueError) as ctx:
                    await proxy_func(prompt="hello")
                self.assertIn("Completion validation error", str(ctx.exception))

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

    def test_patch_async_openai_global_replaces_completions_create(self):
        """After patching, client.completions.create should be the proxy."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test_key"}):
            with patch.object(self.chat_proxy, 'make_proxy_create') as mock_make_proxy:
                mock_proxy = MagicMock(name="proxy_func")
                mock_make_proxy.return_value = mock_proxy
                self.chat_proxy.patch_async_openai_global({"model": "test_model"})
                client = object.__new__(self.AsyncOpenAI)
                client.completions = MagicMock()
                client.completions.create = MagicMock(name="real_create")
                real_create = client.completions.create
                self.AsyncOpenAI.__init__(client)
                mock_make_proxy.assert_called_once_with(real_create, {"model": "test_model"})
                self.assertIs(client.completions.create, mock_proxy)

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

    def test_patch_async_openai_global_does_not_call_make_proxy_when_completions_invalid(self):
        """If completions is None, make_proxy_create should not be called."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test_key"}):
            with patch.object(self.chat_proxy, 'make_proxy_create') as mock_make_proxy:
                self.chat_proxy.patch_async_openai_global({"model": "test_model"})
                client = object.__new__(self.AsyncOpenAI)
                client.completions = None
                self.AsyncOpenAI.__init__(client)
                mock_make_proxy.assert_not_called()


if __name__ == "__main__":
    unittest.main()