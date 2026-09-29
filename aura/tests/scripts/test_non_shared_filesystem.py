#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
-------------------------------------------------------------------------
This file is part of the AgentSDK project.
Copyright (c) 2026 Huawei Technologies Co.,Ltd.

AgentSDK is licensed under Mulan PSL v2.
You can use this software according to the terms and conditions of the Mulan PSL v2.
You may obtain a copy of Mulan PSL v2 at:

        http://license.coscl.org.cn/MulanPSL2

THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
See the Mulan PSL v2 for more details.
-------------------------------------------------------------------------
"""

import shutil
import subprocess
from pathlib import Path


REPO_ROOT = next(
    parent
    for parent in Path(__file__).resolve().parents
    if (parent / "aura" / "scripts" / "base" / "utils.sh").exists()
)
UTILS_SCRIPT = REPO_ROOT / "aura" / "scripts" / "base" / "utils.sh"
ENVS_SCRIPT = REPO_ROOT / "aura" / "scripts" / "base" / "envs.sh"
INFER_CONFIG_SCRIPT = (
    REPO_ROOT / "aura" / "scripts" / "infer" / "vllm" / "parse_infer_config.sh"
)


def run_bash(script: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", "-c", script],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )


def prepare_local_code_root(path: Path) -> Path:
    aura_root = path / "aura"
    shutil.copytree(REPO_ROOT / "aura" / "scripts", aura_root / "scripts")
    infer_config_dir = aura_root / "configs" / "infer"
    infer_config_dir.mkdir(parents=True)
    shutil.copy2(
        REPO_ROOT / "aura" / "configs" / "infer" / "vllm_infer_i16_qwen3_4b.yaml",
        infer_config_dir,
    )
    return aura_root


def filesystem_mode_script(environment: str) -> str:
    return f"""
npu-smi() {{ :; }}
source '{UTILS_SCRIPT}'
unset VC_TASK_HOSTS VC_WORKER_HOSTS IS_SHARED_FILESYSTEM
{environment}
get_is_config_shared_filesystem
printf '%s|%s' "$IS_SHARED_FILESYSTEM" "$VC_WORKER_HOSTS"
"""


def test_auto_mode_selects_shared_filesystem_for_vc_task_hosts():
    result = run_bash(filesystem_mode_script("export VC_TASK_HOSTS=10.0.0.1,10.0.0.2"))

    assert result.returncode == 0, result.stderr
    assert result.stdout.endswith("1|10.0.0.1,10.0.0.2")


def test_auto_mode_selects_non_shared_filesystem_for_vc_worker_hosts():
    result = run_bash(
        filesystem_mode_script("export VC_WORKER_HOSTS=10.0.0.1,10.0.0.2")
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.endswith("0|10.0.0.1,10.0.0.2")


def test_explicit_non_shared_mode_overrides_vc_task_hosts():
    result = run_bash(
        filesystem_mode_script(
            "export VC_TASK_HOSTS=10.0.0.1,10.0.0.2\nexport IS_SHARED_FILESYSTEM=0"
        )
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.endswith("0|10.0.0.1,10.0.0.2")


def test_invalid_filesystem_mode_is_rejected():
    result = run_bash(
        filesystem_mode_script(
            "export VC_WORKER_HOSTS=10.0.0.1,10.0.0.2\n"
            "export IS_SHARED_FILESYSTEM=invalid"
        )
    )

    assert result.returncode == 1
    assert "invalid IS_SHARED_FILESYSTEM" in result.stdout


def test_training_node_generates_local_infer_config(tmp_path):
    result = run_bash(
        f"""
npu-smi() {{ :; }}
source '{ENVS_SCRIPT}'
export INFER_CONF_NAME=vllm_infer_i16_qwen3_4b
export VC_WORKER_HOSTS=10.0.0.1,10.0.0.2
export VC_TASK_INDEX=1
export MASTER_TRAIN_INDEX=1
export IS_SHARED_FILESYSTEM=0
source '{INFER_CONFIG_SCRIPT}'
infer_dir='{tmp_path}'
get_infer_configs
cat "$infer_dir/conf_for_train/config_done"
cat "$infer_dir/conf_for_train/prefill_server_list"
cat "$infer_dir/conf_for_train/tensor_parallel_size"
"""
    )

    assert result.returncode == 0, result.stderr
    assert "done" in result.stdout
    assert '"http://10.0.0.1:20012"' in result.stdout
    assert result.stdout.rstrip().endswith("8")


def test_shared_mode_training_node_waits_for_infer_generated_config(tmp_path):
    result = run_bash(
        f"""
npu-smi() {{ :; }}
source '{ENVS_SCRIPT}'
export INFER_CONF_NAME=vllm_infer_i16_qwen3_4b
export VC_WORKER_HOSTS=10.0.0.1,10.0.0.2
export VC_TASK_INDEX=1
export MASTER_TRAIN_INDEX=1
export IS_SHARED_FILESYSTEM=1
source '{INFER_CONFIG_SCRIPT}'
infer_dir='{tmp_path}'
get_infer_configs
test ! -e "$infer_dir/conf_for_train/config_done"
"""
    )

    assert result.returncode == 0, result.stderr


def test_nodes_with_different_code_roots_generate_matching_configs(tmp_path):
    infer_root = prepare_local_code_root(tmp_path / "infer-node")
    train_root = prepare_local_code_root(tmp_path / "train-node")

    def generate_config(
        aura_root: Path, task_index: int
    ) -> subprocess.CompletedProcess[str]:
        return run_bash(
            f"""
npu-smi() {{ :; }}
source '{aura_root}/scripts/base/envs.sh'
export INFER_CONF_NAME=vllm_infer_i16_qwen3_4b
export VC_WORKER_HOSTS=10.0.0.1,10.0.0.2
export VC_TASK_INDEX={task_index}
export MASTER_TRAIN_INDEX=1
export IS_SHARED_FILESYSTEM=0
source '{aura_root}/scripts/infer/vllm/parse_infer_config.sh'
get_infer_configs
cat '{aura_root}/scripts/infer/conf_for_train/prefill_server_list'
cat '{aura_root}/scripts/infer/conf_for_train/tensor_parallel_size'
"""
        )

    infer_result = generate_config(infer_root, 0)
    train_result = generate_config(train_root, 1)

    assert infer_result.returncode == 0, infer_result.stderr
    assert train_result.returncode == 0, train_result.stderr
    infer_config = infer_root / "scripts" / "infer" / "conf_for_train"
    train_config = train_root / "scripts" / "infer" / "conf_for_train"
    config_names = (
        "prefill_server_list",
        "decode_server_list",
        "tensor_parallel_size",
        "data_parallel_size",
        "enable_expert_parallel",
        "vllm_version",
    )
    for name in config_names:
        assert (infer_config / name).read_text() == (train_config / name).read_text()
