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
    """Create fake package tree and sub-modules without injecting the target package."""
    import os
    import aura as _aura
    real_aura_path = _aura.__path__
    base_path = real_aura_path[0] if real_aura_path else "."

    # Only inject parent packages (up to "patch"), not "patch_0_10_2" itself
    packages = {
        "aura": real_aura_path,
        "aura.runner": os.path.join(base_path, "runner"),
        "aura.runner.infer_adapter": os.path.join(base_path, "runner/infer_adapter"),
        "aura.runner.infer_adapter.vllm": os.path.join(base_path, "runner/infer_adapter/vllm"),
        "aura.runner.infer_adapter.vllm.patch": os.path.join(base_path, "runner/infer_adapter/vllm/patch"),
        # Do NOT add patch_0_10_2 here, or the real __init__.py won't be executed.
    }

    # Fake sub-modules that __init__.py will import
    sub_modules = [
        "patch_worker_v1",
        "patch_camem",
        "patch_schedule_config",
        "patch_model_runner_v1",
        "patch_qwen3_moe",
        "patch_scheduler",
        "patch_attention_mask",
        "patch_attention_v1",
        "patch_vllm_qwen3_moe",
        "patch_serving_completion",
        "patch_acl_graph",
        "patch_base",
        "patch_llmdatadist_c_mgr_connector",
        "patch_multiproc_executor",
        "patch_abstract",
        "patch_sampler",
        "patch_vllm_sampler",
    ]
    for name in sub_modules:
        full_name = f"aura.runner.infer_adapter.vllm.patch.patch_0_10_2.{name}"
        sys.modules[full_name] = types.ModuleType(full_name)

    # Inject parent packages (with correct __path__ so Python can locate the target package)
    for mod_name, path in packages.items():
        if mod_name not in sys.modules:
            mod = types.ModuleType(mod_name)
            mod.__path__ = [path] if isinstance(path, str) else path
            sys.modules[mod_name] = mod

    yield {
        "sub_modules": sub_modules,
    }

    # Cleanup
    prefix = "aura.runner.infer_adapter.vllm.patch.patch_0_10_2"
    for key in list(sys.modules.keys()):
        if key.startswith(prefix):
            del sys.modules[key]
    for mod_name in list(packages.keys()):
        if mod_name in sys.modules and isinstance(sys.modules[mod_name], types.ModuleType):
            del sys.modules[mod_name]


def test_init_imports_all_submodules(fake_init_env):
    """Verify that __init__.py imports all expected sub-modules."""
    import aura.runner.infer_adapter.vllm.patch.patch_0_10_2 as pkg

    expected = fake_init_env["sub_modules"]
    for attr in expected:
        assert hasattr(pkg, attr), f"Missing attribute {attr}"
        assert isinstance(getattr(pkg, attr), types.ModuleType)
