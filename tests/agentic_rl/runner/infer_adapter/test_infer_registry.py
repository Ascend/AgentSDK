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

import unittest
import sys
import importlib
from unittest.mock import patch, MagicMock


# ============================================================
# Fake module builder to avoid real dependencies
# ============================================================

# List of modules that will be replaced with MagicMock
MOCK_MODULES = [
    "ray", "ray.state", "ray.runtime_context", "vllm",
    "vllm.engine.arg_utils", "vllm.entrypoints.openai.protocol",
    "vllm.entrypoints.openai.serving_chat", "vllm.entrypoints.openai.serving_completion",
    "vllm.entrypoints.openai.serving_models", "vllm.v1.engine.async_llm",
    "vllm.v1.executor.abstract", "vllm.config", "vllm.executor",
    "agentic_rl.base.log.loggers", "agentic_rl.runner.infer_adapter.async_server",
    "agentic_rl.runner.scheduler.workload", "agentic_rl.runner.scheduler.load_stat",
]


def _build_fake_modules():
    """Create a dict of MagicMock objects for all modules in MOCK_MODULES."""
    return {mod: MagicMock(name=f"fake_{mod}") for mod in MOCK_MODULES}


def _reload_infer_registry_module():
    """
    Reload the infer_registry module after fake modules are installed,
    ensuring all imports use the mocks.
    """
    mod_name = "agentic_rl.runner.infer_adapter.infer_registry"
    if mod_name in sys.modules:
        del sys.modules[mod_name]
    return importlib.import_module(mod_name)


# ============================================================
# Base Test Class with module patching
# ============================================================

class BaseInferRegistryTest(unittest.TestCase):
    """Base class that patches sys.modules and reloads the target module."""

    def setUp(self):
        self.module_patcher = patch.dict(sys.modules, _build_fake_modules())
        self.module_patcher.start()
        self.infer_registry_mod = _reload_infer_registry_module()

    def tearDown(self):
        self.module_patcher.stop()


# ============================================================
# Tests for InferBackendRegistry
# ============================================================

class TestInferBackendRegistry(BaseInferRegistryTest):
    """Unit tests for InferBackendRegistry class (pure logic)."""

    def test_register_and_get_class(self):
        """Verify register stores a class and get_class retrieves it."""
        InferBackendRegistry = self.infer_registry_mod.InferBackendRegistry

        reg = InferBackendRegistry()
        self.assertEqual(reg._registry, {})

        cls_a = object()
        cls_b = object()

        reg.register("backend_a", cls_a)
        self.assertIs(reg.get_class("backend_a"), cls_a)

        # Registering again with same key overwrites
        reg.register("backend_a", cls_b)
        self.assertIs(reg.get_class("backend_a"), cls_b)

        # Non-existent key returns None
        self.assertIsNone(reg.get_class("not_exist"))


# ============================================================
# Tests for async_server_class function
# ============================================================

class TestAsyncServerClass(BaseInferRegistryTest):
    """Unit tests for async_server_class function."""

    def setUp(self):
        super().setUp()

        # Save original registry state to restore later
        self._backup_registry_dict = self.infer_registry_mod.registry._registry.copy()

        self.fake_vllm_cls = MagicMock(name="FakeVLLMServer")
        self.fake_vllm_pd_cls = MagicMock(name="FakeVLLMServerPDSep")

        # Reset registry to a known clean state with only our fakes
        self.infer_registry_mod.registry._registry.clear()
        self.infer_registry_mod.registry.register("vllm", self.fake_vllm_cls)
        self.infer_registry_mod.registry.register("vllm_pd", self.fake_vllm_pd_cls)

    def tearDown(self):
        # Restore original registry state
        self.infer_registry_mod.registry._registry.clear()
        self.infer_registry_mod.registry._registry.update(self._backup_registry_dict)
        super().tearDown()

    def test_async_server_class_vllm_normal_mode(self):
        """When PD separation is disabled, 'vllm' returns the standard vLLM server class."""
        async_server_class = self.infer_registry_mod.async_server_class

        with patch.object(self.infer_registry_mod, "is_pd_separate", return_value=False):
            result = async_server_class("vllm")
            self.assertIs(result, self.fake_vllm_cls)

    def test_async_server_class_vllm_pd_separate_mode(self):
        """When PD separation is enabled, 'vllm' returns the PD-aware server class."""
        async_server_class = self.infer_registry_mod.async_server_class

        with patch.object(self.infer_registry_mod, "is_pd_separate", return_value=True):
            result = async_server_class("vllm")
            self.assertIs(result, self.fake_vllm_pd_cls)

    def test_async_server_class_vllm_pd_explicit(self):
        """Explicitly requesting 'vllm_pd' returns the PD server class regardless of mode."""
        async_server_class = self.infer_registry_mod.async_server_class

        with patch.object(self.infer_registry_mod, "is_pd_separate", return_value=False):
            result = async_server_class("vllm_pd")
            self.assertIs(result, self.fake_vllm_pd_cls)

    def test_async_server_class_other_backend_pd_separate_returns_none(self):
        """
        For backends other than 'vllm', PD mode does not trigger substitution.
        Returns None if the backend is not registered.
        """
        async_server_class = self.infer_registry_mod.async_server_class

        with patch.object(self.infer_registry_mod, "is_pd_separate", return_value=True):
            result = async_server_class("other_backend")
            self.assertIsNone(result)

    def test_async_server_class_unknown_backend_returns_none(self):
        """If the requested backend is not registered, return None."""
        async_server_class = self.infer_registry_mod.async_server_class

        with patch.object(self.infer_registry_mod, "is_pd_separate", return_value=False):
            result = async_server_class("unknown_backend")
            self.assertIsNone(result)


# ============================================================
# Test global registry instance
# ============================================================

class TestRegistryGlobalState(BaseInferRegistryTest):
    """
    Verify that the module-level registry exists and is an instance of InferBackendRegistry.
    Does not rely on real server imports.
    """

    def test_registry_is_instance(self):
        """Ensure the global 'registry' variable is an instance of InferBackendRegistry."""
        registry = self.infer_registry_mod.registry
        InferBackendRegistry = self.infer_registry_mod.InferBackendRegistry

        self.assertIsInstance(registry, InferBackendRegistry)


if __name__ == "__main__":
    unittest.main()