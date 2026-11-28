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
"""Exercise launch selection and hybrid compatibility without starting services."""
from copy import deepcopy
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

from omegaconf import OmegaConf
import pytest
import yaml

from aura.trainer.config_validation import ConfigValidationError, ValidationContext
from aura.trainer.config_validation.launch import validate_launch_config
from aura.trainer.config_validation.rules.modes import ModeCompatibilityRule


ROOT = Path(__file__).resolve().parents[4]
CLI = ROOT / "scripts/base/validate_train_config.py"


@pytest.fixture(autouse=True)
def no_inherited_launch_mode(monkeypatch):
    monkeypatch.delenv("WORK_MODE", raising=False)
    monkeypatch.delenv("RL_TRAIN_BACKEND", raising=False)


def config(mode="hybrid", engine="verl"):
    return {"train_instances": [{"name": "train", "executor_kwargs": {
        "work_mode": mode, "train_engine": engine,
        "train_config": {"actor_rollout_ref": {"hybrid_engine": mode == "hybrid"}},
    }}]}


@pytest.mark.parametrize("engine,mode,launcher", [
    ("verl", "hybrid", "hybrid"), ("verl", "one_step_off", "one_step_off"),
    ("verl", "fully_async", "fully_async"), ("verl", "dummy_train", "one_step_off"),
    ("mindspeed_rl", "hybrid", "hybrid"), ("mindspeed_rl", "one_step_off", "one_step_off"),
    ("mindspeed_rl", "dummy_train", "one_step_off"), ("omni_rl", "hybrid", "hybrid"),
])
def test_registered_training_paths_have_compatible_launch_modes(engine, mode, launcher):
    assert validate_launch_config(config(mode, engine), launcher).valid


@pytest.mark.parametrize("launcher,mode", [
    ("hybrid", "one_step_off"), ("hybrid", "fully_async"), ("hybrid", "dummy_train"),
    ("one_step_off", "hybrid"), ("one_step_off", "fully_async"),
    ("fully_async", "hybrid"), ("fully_async", "one_step_off"),
])
def test_cross_file_conflict_is_rejected_even_when_strict_is_disabled(launcher, mode):
    value = config(mode)
    value["train_instances"][0]["executor_kwargs"]["train_config"]["config_validation"] = {"strict": False}
    with pytest.raises(ConfigValidationError) as error:
        validate_launch_config(value, launcher)
    issue, = error.value.report.issues
    assert issue.rule_id == "mode.launch_consistency"
    assert issue.path == "train_instances.0.executor_kwargs.work_mode"
    assert issue.value == {"WORK_MODE": launcher, "yaml": mode}


@pytest.mark.parametrize("launcher", ["", "invalid", True, []])
def test_invalid_launcher_mode_is_reported(launcher):
    with pytest.raises(ConfigValidationError):
        validate_launch_config(config(), launcher)


@pytest.mark.parametrize("mode", [None, 1, [], "invalid"])
def test_invalid_yaml_mode_is_reported(mode):
    with pytest.raises(ConfigValidationError) as error:
        validate_launch_config(config(mode))
    assert error.value.report.issues[0].path.endswith("work_mode")


def test_environment_mode_is_checked_and_direct_python_can_omit_it(monkeypatch):
    value = OmegaConf.create(config())
    assert validate_launch_config(value).valid
    monkeypatch.setenv("WORK_MODE", "one_step_off")
    with pytest.raises(ConfigValidationError):
        validate_launch_config(value)


def test_every_selected_training_instance_is_checked():
    value = config("one_step_off")
    value["train_instances"] += config("hybrid")["train_instances"]
    with pytest.raises(ConfigValidationError) as error:
        validate_launch_config(value, "fully_async")
    assert [issue.path for issue in error.value.report.issues] == [
        "train_instances.0.executor_kwargs.work_mode", "train_instances.1.executor_kwargs.work_mode",
    ]


def test_launcher_backend_must_match_yaml_engine(monkeypatch):
    assert validate_launch_config(config(engine="mindspeed_rl"), "hybrid", "mindspeed_rl").valid
    monkeypatch.setenv("RL_TRAIN_BACKEND", "verl")
    with pytest.raises(ConfigValidationError) as error:
        validate_launch_config(config(engine="mindspeed_rl"), "hybrid")
    assert error.value.report.issues[0].path == "train_instances.0.executor_kwargs.train_engine"


def test_mode_not_registered_by_the_selected_backend_is_rejected():
    with pytest.raises(ConfigValidationError) as error:
        validate_launch_config(config("fully_async", "mindspeed_rl"), "fully_async")
    assert "does not register this work mode" in error.value.report.issues[0].message
    assert len(error.value.report.issues) == 1


@pytest.mark.parametrize("mode,value", [("hybrid", False), ("one_step_off", True), ("fully_async", True),
                                        ("hybrid", 1), ("one_step_off", "false")])
def test_hybrid_engine_must_match_the_executed_path(mode, value):
    primary = config(mode)
    primary["train_instances"][0]["executor_kwargs"]["train_config"]["actor_rollout_ref"]["hybrid_engine"] = value
    with pytest.raises(ConfigValidationError) as error:
        validate_launch_config(primary, mode)
    assert error.value.report.issues[0].rule_id == "mode.hybrid_engine"


def test_missing_hybrid_engine_allows_native_default():
    primary = config()
    primary["train_instances"][0]["executor_kwargs"]["train_config"] = {}
    assert validate_launch_config(primary, "hybrid").valid


def test_unresolved_hybrid_engine_is_not_treated_as_a_missing_override():
    primary = config()
    primary["train_instances"][0]["executor_kwargs"]["train_config"]["actor_rollout_ref"]["hybrid_engine"] = "${missing}"
    with pytest.raises(ConfigValidationError) as error:
        validate_launch_config(OmegaConf.create(primary), "hybrid")
    assert error.value.report.issues[0].value == "<unresolved>"


def test_unused_invalid_container_value_stays_a_serializable_warning():
    context = ValidationContext(work_mode="hybrid", train_config=OmegaConf.create({
        "extras": {"data_loader": {"train_data_path": ["unused/path"]}},
    }))
    issue, = ModeCompatibilityRule().validate(context)
    assert json.loads(json.dumps(issue.to_dict()))["value"] == ["unused/path"]


def test_hybrid_shared_fields_and_inherited_verl_defaults_are_not_misreported():
    train = {"actor_rollout_ref": {"hybrid_engine": True}, "extras": {
        "agent_service": "math", "infer_service": "model", "traj_output_path": "/output",
        "validate_num_samples": 5, "data_loader": {"test_data_path": None},
    }}
    context = ValidationContext(train_config=train, work_mode="hybrid",
                                source_map={"train_config.extras.validate_num_samples": "verl_default"})
    assert ModeCompatibilityRule().validate(context) == []


def bash_path():
    bash = shutil.which("bash")
    if sys.platform == "win32":
        candidate = Path(os.environ.get("ProgramFiles", "C:/Program Files")) / "Git/bin/bash.exe"
        if candidate.is_file():
            bash = str(candidate)
    if bash is None:
        pytest.skip("Bash is needed to exercise the real launcher parser")
    return bash


@pytest.mark.parametrize("yaml_mode,resume_mode,success,engine", [
    ("hybrid", None, True, "verl"), ("one_step_off", None, False, "verl"),
    ("one_step_off", "hybrid", True, "verl"), ("hybrid", "one_step_off", False, "verl"),
    ("hybrid", None, False, "mindspeed_rl"),
])
def test_shell_parser_checks_selected_or_resume_yaml_before_service_start(tmp_path, yaml_mode, resume_mode, success, engine):
    root = tmp_path / "aura"
    (root / "configs/train").mkdir(parents=True)
    (root / "scripts/base").mkdir(parents=True)
    shutil.copyfile(CLI, root / "scripts/base/validate_train_config.py")
    (root / "configs/base.conf").write_text("work_mode=hybrid\ntrain_config_name=selected\n", encoding="utf-8")
    for name, mode in (("selected", yaml_mode), ("resumed", resume_mode)):
        if mode:
            (root / f"configs/train/{name}.yaml").write_text(yaml.safe_dump(config(mode, engine)), encoding="utf-8")
    source = (ROOT / "scripts/base/utils.sh").read_text(encoding="utf-8")
    functions = source[source.index("function get_conf_val()"):source.index("function parse_resume_conf()")]
    shell = 'log_error() { echo "$*" >&2; }\nlog_info() { :; }\n'
    shell += 'python3() { command "$TEST_PYTHON" "$@"; }\nroot_dir="$1"\n'
    shell += functions + '\nparse_train_conf verl\necho "__SERVICE_START__"\n'
    env = dict(os.environ, TEST_PYTHON=sys.executable.replace("\\", "/"),
               PYTHONPATH=str(ROOT), RESUME_TRAIN_CONF_NAME="resumed" if resume_mode else "", CONFIG_EXT=".yaml")
    result = subprocess.run([bash_path(), "--noprofile", "--norc", "-c", shell, "test", root.as_posix()],
                            env=env, capture_output=True, text=True, timeout=15)
    assert (result.returncode == 0) is success, result.stderr
    assert ("__SERVICE_START__" in result.stdout) is success
    if not success:
        assert ("Launcher and training YAML disagree" if engine == "verl" else "train_engine disagree") in result.stderr


@pytest.mark.parametrize("primary", sorted((ROOT / "configs/train").glob("verl_train_*.yaml")), ids=lambda p: p.stem)
def test_all_official_templates_have_consistent_mode_routing(primary):
    value = OmegaConf.load(primary)
    mode = value.train_instances[0].executor_kwargs.work_mode
    assert validate_launch_config(value, mode).valid


def test_shell_guard_imports_no_ray_or_accelerator_runtime(tmp_path):
    path = tmp_path / "train.yaml"
    path.write_text(yaml.safe_dump(config()), encoding="utf-8")
    code = '''import runpy,sys
before=set(sys.modules)
entry=runpy.run_path(sys.argv[1])
sys.argv=[sys.argv[1],"--config-file",sys.argv[2],"--work-mode","hybrid"]
assert entry["main"]()==0
loaded=set(sys.modules)-before
assert not any(name.split(".")[0] in {"ray","torch","torch_npu","vllm"} for name in loaded)
'''
    result = subprocess.run([sys.executable, "-c", code, str(CLI), str(path)],
                            capture_output=True, text=True, timeout=15)
    assert result.returncode == 0, result.stderr
