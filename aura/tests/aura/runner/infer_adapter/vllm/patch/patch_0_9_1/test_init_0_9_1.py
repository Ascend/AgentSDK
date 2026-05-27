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
def fake_init_env():
    """Prepare fake parent packages and sub-modules for patch_0_9_1/__init__.py"""
    import os
    import aura as _aura
    real_aura_path = _aura.__path__
    base_path = real_aura_path[0] if real_aura_path else "."

    # Parent packages (up to "patch", NOT "patch_0_9_1")
    packages = {
        "aura": real_aura_path,
        "aura.runner": os.path.join(base_path, "runner"),
        "aura.runner.infer_adapter": os.path.join(base_path, "runner/infer_adapter"),
        "aura.runner.infer_adapter.vllm": os.path.join(base_path, "runner/infer_adapter/vllm"),
        "aura.runner.infer_adapter.vllm.patch": os.path.join(base_path, "runner/infer_adapter/vllm/patch"),
    }

    for mod_name, path in packages.items():
        if mod_name not in sys.modules:
            mod = types.ModuleType(mod_name)
            mod.__path__ = [path] if isinstance(path, str) else path
            sys.modules[mod_name] = mod

    # Fake sub-modules listed in __init__.py
    sub_modules = [
        "patch_worker_v1",
        "patch_camem",
        "patch_attention",
        "patch_attention_v1",
    ]
    for name in sub_modules:
        full_name = f"aura.runner.infer_adapter.vllm.patch.patch_0_9_1.{name}"
        sys.modules[full_name] = types.ModuleType(full_name)

    yield {"sub_modules": sub_modules}

    # Cleanup
    prefix = "aura.runner.infer_adapter.vllm.patch.patch_0_9_1"
    for key in list(sys.modules.keys()):
        if key.startswith(prefix):
            del sys.modules[key]
    for mod_name in packages:
        if mod_name in sys.modules and isinstance(sys.modules[mod_name], types.ModuleType):
            del sys.modules[mod_name]


def test_init_imports_all_submodules(fake_init_env):
    """Verify that __init__.py imports the four expected sub-modules."""
    import aura.runner.infer_adapter.vllm.patch.patch_0_9_1 as pkg

    expected = fake_init_env["sub_modules"]
    for attr in expected:
        assert hasattr(pkg, attr), f"Missing attribute {attr}"
        assert isinstance(getattr(pkg, attr), types.ModuleType)
