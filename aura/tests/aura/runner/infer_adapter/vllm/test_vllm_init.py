#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# -------------------------------------------------------------------------
#  This file is part of the AgentSDK project.
# Copyright (c) 2026 Huawei Technologies Co.,Ltd.
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
import sys
import types
from unittest.mock import patch
import pytest


@pytest.fixture
def fake_patch_init_env():
    """Prepare fake sub-packages and parent packages for patch/__init__.py"""
    # Fake target sub-packages (will be imported by __init__.py)
    fake_0_10_2 = types.ModuleType("aura.runner.infer_adapter.vllm.patch.patch_0_10_2")
    fake_0_9_1 = types.ModuleType("aura.runner.infer_adapter.vllm.patch.patch_0_9_1")

    sys.modules[fake_0_10_2.__name__] = fake_0_10_2
    sys.modules[fake_0_9_1.__name__] = fake_0_9_1

    # Ensure parent packages exist with correct __path__, but NOT the target "patch" package itself
    import os
    import aura
    base_path = aura.__path__[0]
    parent_pkgs = {
        "aura.runner": os.path.join(base_path, "runner"),
        "aura.runner.infer_adapter": os.path.join(base_path, "runner/infer_adapter"),
        "aura.runner.infer_adapter.vllm": os.path.join(base_path, "runner/infer_adapter/vllm"),
        # Deliberately omit "aura.runner.infer_adapter.vllm.patch" so that the real __init__.py is executed
    }
    for mod_name, path in parent_pkgs.items():
        if mod_name not in sys.modules:
            m = types.ModuleType(mod_name)
            m.__path__ = [path]
            sys.modules[mod_name] = m

    yield {
        "fake_0_10_2": fake_0_10_2,
        "fake_0_9_1": fake_0_9_1,
    }

    # Cleanup to avoid polluting other tests
    patch_prefix = "aura.runner.infer_adapter.vllm.patch"
    for mod_name in list(sys.modules.keys()):
        if mod_name.startswith(patch_prefix):
            del sys.modules[mod_name]
    for mod_name in parent_pkgs:
        if mod_name in sys.modules and isinstance(sys.modules[mod_name], types.ModuleType):
            del sys.modules[mod_name]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
class TestPatchInit:
    def test_no_version_imports_nothing(self, fake_patch_init_env):
        """When VLLM_VERSION is absent, no sub-packages are imported."""
        patch_mod_name = "aura.runner.infer_adapter.vllm.patch"
        if patch_mod_name in sys.modules:
            del sys.modules[patch_mod_name]

        with patch('os.getenv', return_value=None):
            import aura.runner.infer_adapter.vllm.patch as patch_pkg

        assert not hasattr(patch_pkg, "patch_0_10_2")
        assert not hasattr(patch_pkg, "patch_0_9_1")

    def test_unknown_version_imports_nothing(self, fake_patch_init_env):
        """When VLLM_VERSION is unknown (e.g., '0.11.0'), nothing is imported."""
        patch_mod_name = "aura.runner.infer_adapter.vllm.patch"
        if patch_mod_name in sys.modules:
            del sys.modules[patch_mod_name]

        with patch('os.getenv', return_value='0.11.0'):
            import aura.runner.infer_adapter.vllm.patch as patch_pkg

        assert not hasattr(patch_pkg, "patch_0_10_2")
        assert not hasattr(patch_pkg, "patch_0_9_1")
