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
    import os
    import aura as _aura
    real_aura_path = _aura.__path__
    base_path = real_aura_path[0] if real_aura_path else "."

    # Prepare fake sub-modules (the ones __init__.py imports)
    sub_modules = [
        "patch_acl_graph",
        "patch_camem",
        "patch_model_runner_v1",
        "patch_sampler",
        "patch_serving_completion",
        "patch_vllm_sampler",
        "patch_worker_v1",
    ]
    for name in sub_modules:
        full_name = f"aura.runner.infer_adapter.vllm.patch.patch_0_11_0.{name}"
        sys.modules[full_name] = types.ModuleType(full_name)

    # Ensure parent packages exist with correct __path__
    packages = {
        "aura.runner": "runner",
        "aura.runner.infer_adapter": "runner/infer_adapter",
        "aura.runner.infer_adapter.vllm": "runner/infer_adapter/vllm",
        "aura.runner.infer_adapter.vllm.patch": "runner/infer_adapter/vllm/patch",
    }
    for mod_name, rel_path in packages.items():
        if mod_name not in sys.modules:
            mod = types.ModuleType(mod_name)
            mod.__path__ = [os.path.join(base_path, rel_path)]
            sys.modules[mod_name] = mod

    yield

    # Cleanup after test to avoid affecting other tests
    pkg_prefix = "aura.runner.infer_adapter.vllm.patch.patch_0_11_0"
    for name in list(sys.modules.keys()):
        if name.startswith(pkg_prefix):
            del sys.modules[name]
    for mod_name in packages:
        if mod_name in sys.modules:
            del sys.modules[mod_name]


def test_init_imports_all_submodules(fake_init_env):
    """Verify that __init__.py executes and exposes all expected sub-modules."""
    import aura.runner.infer_adapter.vllm.patch.patch_0_11_0 as pkg

    expected = [
        "patch_acl_graph",
        "patch_camem",
        "patch_model_runner_v1",
        "patch_sampler",
        "patch_serving_completion",
        "patch_vllm_sampler",
        "patch_worker_v1",
    ]
    for attr in expected:
        assert hasattr(pkg, attr), f"Missing attribute {attr}"
        assert isinstance(getattr(pkg, attr), types.ModuleType)
