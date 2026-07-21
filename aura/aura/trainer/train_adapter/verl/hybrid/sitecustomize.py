# -*- coding: utf-8 -*-
# SPDX-License-Identifier: Apache-2.0 OR MulanPSL-2.0
#
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
#
"""NPU verl hybrid startup hooks (auto-imported via PYTHONPATH).

Patches applied when ``hybrid/`` is on PYTHONPATH:
  1. NPU set_device physical-to-logical remap for verl FSDP WorkerDict.
  2. torch.compile no-op for torch_npu / vLLM import paths.
  3. vLLM module-path shims expected by bundled verl.
"""
import builtins
import importlib
import os
import sys
import types

import torch


def _noop_compile(fn=None, /, *args, **kwargs):
    if fn is not None:
        return fn
    return lambda f: f


def _patch_compile():
    torch.compile = _noop_compile


_VLLM_EAGER_SHIMS = (
    "vllm.utils.argparse_utils",
    "vllm.lora.lora_model",
)

_VLLM_LAZY_SHIMS = (
    "vllm.entrypoints.openai.parser",
    "vllm.entrypoints.openai.parser.harmony_utils",
)

_VLLM_VERL_SHIMS = _VLLM_EAGER_SHIMS + _VLLM_LAZY_SHIMS

_PARSER_PKG = "vllm.entrypoints.openai.parser"
_HARMONY_SHIM = "vllm.entrypoints.openai.parser.harmony_utils"
_HARMONY_SRC = "vllm.entrypoints.harmony_utils"


def _ensure_openai_parser_package():
    importlib.import_module("vllm.entrypoints.openai")
    if _PARSER_PKG not in sys.modules:
        parser_mod = types.ModuleType(_PARSER_PKG)
        parser_mod.__path__ = []
        sys.modules[_PARSER_PKG] = parser_mod
    return sys.modules[_PARSER_PKG]


def _install_vllm_verl_shim(fullname: str):
    if fullname in sys.modules:
        return sys.modules[fullname]
    try:
        importlib.import_module("vllm")
    except ImportError:
        return None
    if fullname == "vllm.utils.argparse_utils":
        utils = importlib.import_module("vllm.utils")
        mod = types.ModuleType(fullname)
        mod.FlexibleArgumentParser = utils.FlexibleArgumentParser
        sys.modules[fullname] = mod
        return mod
    if fullname == "vllm.lora.lora_model":
        mod = importlib.import_module("vllm.lora.models")
        sys.modules[fullname] = mod
        return mod
    if fullname == _PARSER_PKG:
        return _ensure_openai_parser_package()
    if fullname == _HARMONY_SHIM:
        src = importlib.import_module(_HARMONY_SRC)
        _ensure_openai_parser_package()
        sys.modules[fullname] = src
        return src
    return None


def _ensure_vllm_verl_compat():
    installed = []
    for name in _VLLM_EAGER_SHIMS:
        try:
            if _install_vllm_verl_shim(name) is not None:
                installed.append(name.split(".")[-1])
        except Exception as exc:
            print(f"[vllm verl compat] WARN: shim {name} failed: {exc}, pid={os.getpid()}", flush=True)
    if installed:
        print(f"[vllm verl compat] shims installed: {', '.join(installed)}, pid={os.getpid()}", flush=True)


if not getattr(builtins, "_verl_hybrid_import_hook", False):
    builtins._verl_hybrid_import_hook = True
    _orig_import = builtins.__import__

    def _import_hook(name, globals=None, locals=None, fromlist=(), level=0):
        if isinstance(name, str) and name in _VLLM_VERL_SHIMS:
            shim = _install_vllm_verl_shim(name)
            if shim is not None:
                return shim
        mod = _orig_import(name, globals, locals, fromlist, level)
        if isinstance(name, str) and (name == "torch_npu" or name.startswith("torch_npu.")):
            _patch_compile()
        return mod

    builtins.__import__ = _import_hook

_ensure_vllm_verl_compat()

if hasattr(torch, "npu"):
    _patch_compile()
    try:
        importlib.import_module("torch_npu")
    except ImportError:
        pass
    _patch_compile()

    print(f"[NPU torch.compile] verl hybrid no-op patch installed, pid={os.getpid()}", flush=True)

    if not hasattr(torch.npu, "_verl_hybrid_npu_remap"):
        torch.npu._verl_hybrid_npu_remap = True
        _orig_npu = torch.npu.set_device
        _orig_cuda = torch.cuda.set_device

        def _remap(device):
            visible_list = [
                x.strip() for x in os.getenv("ASCEND_RT_VISIBLE_DEVICES", "").split(",") if x.strip()
            ]
            if isinstance(device, torch.device):
                dev_idx = device.index if device.index is not None else 0
            elif isinstance(device, str) and ":" in device:
                dev_idx = int(device.split(":")[-1])
            else:
                dev_idx = int(device)
            if visible_list and str(dev_idx) in visible_list:
                out = visible_list.index(str(dev_idx))
            elif visible_list and 0 <= dev_idx < len(visible_list):
                out = dev_idx
            else:
                out = dev_idx
            if out != dev_idx:
                print(
                    f"[NPU device remap] set_device({device}) -> {out}, "
                    f"ASCEND_RT_VISIBLE_DEVICES={os.getenv('ASCEND_RT_VISIBLE_DEVICES', '')}, pid={os.getpid()}",
                    flush=True,
                )
            return out

        torch.npu.set_device = lambda d: _orig_npu(_remap(d))
        torch.cuda.set_device = lambda d: _orig_cuda(_remap(d))
        print(
            f"[NPU device remap] verl hybrid patch installed, "
            f"ASCEND_RT_VISIBLE_DEVICES={os.getenv('ASCEND_RT_VISIBLE_DEVICES', '')}, pid={os.getpid()}",
            flush=True,
        )
