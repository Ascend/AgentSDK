#!/usr/bin/env python3
# coding=utf-8
# -------------------------------------------------------------------------
# This file is part of the AgentSDK project.
# Copyright (c) 2026 Huawei Technologies Co.,Ltd.
#
# AgentSDK is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of the License at:
#
#          http://license.coscl.org.cn/MulanPSL2
#
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
# -------------------------------------------------------------------------

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import yaml


TRAIN_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "train" / "start_verl_train_cluster.sh"
UPDATE_SCRIPT = TRAIN_SCRIPT.with_name("update_infer_server_config.py")


@pytest.mark.parametrize("work_mode", ["one_step_off", "hybrid"])
@pytest.mark.parametrize("role", ["master", "worker"])
@pytest.mark.parametrize("card_type,visible,expected_npus", [
    (2, "", 8), (2, "0", 1), (3, "", 16), (3, "0,1", 2),
])
def test_ray_start_preserves_cpu_detection_metrics_and_visible_npus(work_mode, role, card_type, visible, expected_npus):
    bash = shutil.which("bash")
    if sys.platform == "win32":
        git_bash = Path(os.environ.get("ProgramFiles", "C:/Program Files")) / "Git/bin/bash.exe"
        if git_bash.is_file():
            bash = str(git_bash)
    if bash is None:
        pytest.skip("Bash is required to exercise the training launcher")
    script = TRAIN_SCRIPT.read_text(encoding="utf-8")
    resource_setup = script.split('[ "$CARD_TYPE" = "${A3_CARD}" ]', 1)[1].split("function start_ray_master()", 1)[0]
    resource_setup = '[ "$CARD_TYPE" = "${A3_CARD}" ]' + resource_setup
    # Extract only resource selection and Ray launch functions: never source
    # the launcher, which would stop/start real services and patch dependencies.
    functions = script.split("function start_ray_master()", 1)[1].split("function start_ray_cluster()", 1)[0]
    shell = r'''
log_info() { :; }
log_error() { echo "$*" >&2; }
sleep() { :; }
ray() {
    if [[ "$1" == "status" ]]; then return 1; fi
    printf '%s\0' "$@"
}
'''
    shell += resource_setup + "\nfunction start_ray_master()" + functions
    shell += "\nstart_ray_" + role + "\n"
    env = dict(os.environ, CARD_TYPE=str(card_type), A3_CARD="3", WORK_MODE=work_mode,
               ASCEND_RT_VISIBLE_DEVICES=visible, VC_TASK_INDEX="0", MASTER_TRAIN_HOST="10.0.0.1",
               AURA_RAY_NUM_CPUS="4")
    result = subprocess.run([bash, "--noprofile", "--norc"], input=shell.encode("utf-8"),
                            env=env, capture_output=True, timeout=15)
    # The worker exits when its mocked Ray head disappears, as in production.
    assert result.returncode == (0 if role == "master" else 1), result.stderr.decode()
    args = result.stdout.decode().rstrip("\0").split("\0")
    assert args[0] == "start"
    if card_type == 3:
        assert args[args.index("--num-cpus") + 1] == "192"
    else:
        assert not any(arg.startswith("--num-cpus") for arg in args)
    assert not any(arg.startswith("--system-config") for arg in args)
    resources = next(arg.split("=", 1)[1] for arg in args if arg.startswith("--resources="))
    assert json.loads(resources) == {"NPU": expected_npus}
    assert ("--head" in args) == (role == "master")


def test_external_inference_config_uses_structural_updater_and_fails_fast():
    script = TRAIN_SCRIPT.read_text(encoding="utf-8")

    assert 'python3 "${train_dir}/update_infer_server_config.py"' in script
    assert "wait_for_infer_cluster_ready || exit 1" in script
    assert 's|prefill_server_list:.*|prefill_server_list:' not in script


def test_infer_server_update_preserves_yaml_and_training_parallelism(tmp_path):
    config_path = tmp_path / "train.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "verl_conf": {
                    "actor_rollout_ref": {
                        "rollout": {
                            "tensor_model_parallel_size": 4,
                            "data_parallel_size": 2,
                        }
                    }
                },
                "infer_instances": [
                    {
                        "executor_kwargs": {
                            "engine_kwargs": {
                                "chat_server": "http://old:8080",
                                "prefill_server_list": ["http://old:20012"],
                                "decode_server_list": [],
                                "tensor_parallel_size": 8,
                                "data_parallel_size": 2,
                                "enable_expert_parallel": False,
                            }
                        }
                    }
                ],
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            str(UPDATE_SCRIPT),
            "--config-file",
            str(config_path),
            "--rollout-host",
            "80.48.37.134",
            "--prefill-server-list",
            '"http://80.48.37.134:20012"',
            "--decode-server-list",
            "",
            "--tensor-parallel-size",
            "2",
            "--data-parallel-size",
            "1",
            "--enable-expert-parallel",
            "false",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    updated = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    rollout = updated["verl_conf"]["actor_rollout_ref"]["rollout"]
    engine = updated["infer_instances"][0]["executor_kwargs"]["engine_kwargs"]

    assert rollout == {"tensor_model_parallel_size": 4, "data_parallel_size": 2}
    assert engine == {
        "chat_server": "http://80.48.37.134:8080",
        "prefill_server_list": ["http://80.48.37.134:20012"],
        "decode_server_list": [],
        "tensor_parallel_size": 2,
        "data_parallel_size": 1,
        "enable_expert_parallel": False,
    }
