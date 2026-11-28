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
"""Exercise native vLLM launch flags without importing vLLM or starting services."""
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[2]
SERVE = ROOT / "scripts/infer/vllm/vllm_serve.sh"


def launch(eager, pd=0, role="prefill", opt="false"):
    bash = shutil.which("bash")
    if sys.platform == "win32":
        candidate = Path(os.environ.get("ProgramFiles", "C:/Program Files")) / "Git/bin/bash.exe"
        if candidate.is_file():
            bash = str(candidate)
    if bash is None:
        pytest.skip("Bash is required for launch tests")
    # The shell function captures argv: no installed vLLM, model or NPU is used.
    shell = r'''
vllm() { printf '__ARGS__\0'; printf '%s\0' "$@"; }
source "$1" --host 127.0.0.1 --port 20012 --master_addr 127.0.0.1 \
    --local_node_rank 0 --role "$2" --engine_id 0
'''
    env = dict(os.environ, VLLM_VERSION="0.22.1", VLLM_ENFORCE_EAGER=eager,
               USE_PD=str(pd), USE_VLLM_OPT=opt, MODEL_PATH="/models/Qwen3-4B",
               SERVED_MODEL_NAME="Qwen3-4B", TENSOR_PARALLEL_SIZE="2",
               PREFILL_TENSOR_PARALLEL_SIZE="2", DECODE_TENSOR_PARALLEL_SIZE="2",
               DATA_PARALLEL_SIZE="1", DATA_PARALLEL_SIZE_LOCAL="1", KV_BACKEND="llmdatadist",
               DEFAULT_SOCKET_IFNAME="eth0", CUDAGRAPH_CAPTURE_SIZES="[1, 2]",
               CUDAGRAPH_MODE="NONE", ENABLE_SP="false", PYTHONPATH="/original")
    return subprocess.run([bash, "--noprofile", "--norc", "-c", shell, "test", SERVE.as_posix(), role],
                          env=env, text=True, capture_output=True, timeout=15)


@pytest.mark.parametrize("eager", ["false", "true"])
@pytest.mark.parametrize("pd,role", [(0, "prefill"), (1, "prefill"), (1, "decode")])
def test_native_eager_flag_without_rpc_injection(eager, pd, role):
    result = launch(eager, pd, role)
    assert result.returncode == 0, result.stderr
    args = result.stdout.split("__ARGS__\0", 1)[1].rstrip("\0").split("\0")
    assert args[:2] == ["serve", "/models/Qwen3-4B"]
    # PD prefill keeps the upstream eager default; others follow the YAML flag.
    assert args.count("--enforce-eager") == int(eager == "true" or (pd == 1 and role == "prefill"))
    assert args[args.index("--tensor-parallel-size") + 1] == "2"
    if "--compilation_config" in args:
        assert json.loads(args[args.index("--compilation_config") + 1])["cudagraph_mode"] == "NONE"


def test_invalid_eager_value_fails_before_launch():
    result = launch("invalid")
    assert result.returncode != 0
    assert "Invalid VLLM_ENFORCE_EAGER" in result.stdout
    assert "__ARGS__" not in result.stdout


def test_optimized_launch_keeps_upstream_path():
    result = launch("false", opt="true")
    assert result.returncode == 0, result.stderr
    args = result.stdout.split("__ARGS__\0", 1)[1].rstrip("\0").split("\0")
    assert "--async-scheduling" in args
    assert "--enforce-eager" not in args


def test_runtime_options():
    spec = importlib.util.spec_from_file_location(
        "runtime_options", ROOT / "scripts/infer/vllm/read_runtime_options.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.runtime_options({}, 8) == (8, "false")
    assert module.runtime_options({"npu_num_per_node": 2, "enforce_eager": True}, 8) == (2, "true")
    for config in ({"npu_num_per_node": 0}, {"enforce_eager": "true"},
                   {"enforce_eager": True, "use_vllm_opt": True}):
        with pytest.raises(ValueError):
            module.runtime_options(config, 8)


def test_parser_applies_runtime_options_before_topology():
    source = (ROOT / "scripts/infer/vllm/parse_infer_config.sh").read_text(encoding="utf-8")
    assert source.index("read_runtime_options.py") < source.index("PREFILL_DATA_PARALLEL_SIZE_LOCAL=")
    assert "export NPU_NUM_PER_NODE VLLM_ENFORCE_EAGER" in source
