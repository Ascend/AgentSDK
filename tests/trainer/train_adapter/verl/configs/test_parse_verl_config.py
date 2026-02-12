#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
-------------------------------------------------------------------------
This file is part of the AgentSDK project.
Copyright (c) 2025 Huawei Technologies Co.,Ltd.

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

import pytest  # type: ignore[import-not-found]
 
from agentic_rl.trainer.train_adapter.verl.configs.parse_verl_config import VerlConfigParser
from agentic_rl.trainer.train_adapter.verl.configs.default_config import DEFAULT_CONFIG
 

@pytest.fixture(autouse=True)
def _disable_file_checks(monkeypatch):
    # Avoid touching the filesystem in Pydantic path validators.
    monkeypatch.setattr(
        "agentic_rl.trainer.train_adapter.schema.FileCheck.check_data_path_is_valid",
        lambda *_args, **_kwargs: None,
    )
 
 
def _make_valid_config(*, dataset_additional_keys=...):
    cfg = {
        "tokenizer_name_or_path": "/path/to/tokenizer",
        "model_name": "llama",
        "agent_name": "my_agent",
        "agent_engine_wrapper_path": "/path/to/wrapper",
        "train_backend": "verl",
        "verl": {
            "train_files": "/path/to/train",
            "val_files": "/path/to/val",
        },
    }
    if dataset_additional_keys is not ...:
        cfg["dataset_additional_keys"] = dataset_additional_keys
    return cfg
 
 
def test_set_nested_value_creates_structure_and_sets_leaf():
    data = {}
    VerlConfigParser._set_nested_value(data, "a.b.c", 1)
    assert data == {"a": {"b": {"c": 1}}}
 
    VerlConfigParser._set_nested_value(data, "a.b.c", 2)
    assert data.get("a", {}).get("b", {}).get("c") == 2

    VerlConfigParser._set_nested_value(data, "a.b.d", 3)
    assert data.get("a", {}).get("b", {}).get("d") == 3
 
 
@pytest.mark.parametrize("dataset_additional_keys", [None, []])
def test_process_config_defaults_dataset_additional_keys(dataset_additional_keys):
    cfg = _make_valid_config(dataset_additional_keys=dataset_additional_keys)
    if dataset_additional_keys is None:
        cfg.pop("dataset_additional_keys", None)
 
    result = VerlConfigParser(cfg)._process_input_config()
    assert result["input_config"]["dataset_additional_keys"] == ["labels"]
 
 
def test_process_config_preserves_dataset_additional_keys_when_nonempty():
    keys = ["labels", "extra"]
    result = VerlConfigParser(_make_valid_config(dataset_additional_keys=keys))._process_input_config()
    assert result["input_config"]["dataset_additional_keys"] == keys
 
 
def test_process_config_builds_expected_verl_config_mapping():
    cfg = _make_valid_config(dataset_additional_keys=["labels"])
    cfg.update(
        {
            "lr": 2e-6,
            "kl_coef": 0.123,
            "max_prompt_length": 111,
            "rollout_n": 3,
            "top_k": 7,
            "temperature": 0.9,
        }
    )
    cfg["verl"].update(
        {
            "train_batch_size": 9,
            "val_batch_size": 10,
            "dataloader_num_workers": 4,
            "max_response_length": 222,
        }
    )
 
    result = VerlConfigParser(cfg)._process_input_config()
 
    agentic = result["agentic_rl_config"]
    assert agentic.agent_name == "my_agent"
    assert agentic.agent_engine_wrapper_path == "/path/to/wrapper"
    assert agentic.train_backend == "verl"
 
    verl = result["verl_config"]
    assert verl["actor_rollout_ref"]["actor"]["optim"]["lr"] == 2e-6
    assert verl["algorithm"]["kl_ctrl"]["kl_coef"] == 0.123
    assert verl["data"]["max_prompt_length"] == 111
    assert verl["data"]["max_response_length"] == 222
    assert verl["actor_rollout_ref"]["rollout"]["n"] == 3
    assert verl["actor_rollout_ref"]["rollout"]["top_k"] == 7
    assert verl["actor_rollout_ref"]["rollout"]["temperature"] == 0.9
 
    assert verl["data"]["train_batch_size"] == 9
    assert verl["data"]["val_batch_size"] == 10
    assert verl["data"]["dataloader_num_workers"] == 4
 
    assert verl["data"]["train_files"] == "/path/to/train"
    assert verl["actor_rollout_ref"]["model"]["path"] == "/path/to/tokenizer"


def test_all_mapping_keys_are_applied():
    cfg = _make_valid_config(dataset_additional_keys=["labels"])
    cfg.update({
        "lr": 1e-6,
        "weight_decay": 0.01,
        "entropy_coeff": 0.01,
        "clip_ratio": 0.3,
        "dtype": "float16",
        "enforce_eager": False,
        "rollout_n": 2,
        "max_model_len": 4096,
        "temperature": 0.8,
        "top_k": 42,
        "top_p": 0.95,
        "gpu_memory_utilization": 0.66,
        "max_num_seqs": 16,
        "max_num_batched_tokens": 7777,
        "use_kl_in_reward": True,
        "gamma": 0.99,
        "lam": 0.95,
        "kl_ctrl_type": "adaptive",
        "kl_penalty": "mse",
        "kl_coef": 0.2,
        "kl_horizon": 999,
        "num_gpus_per_node": 8,
        "max_prompt_length": 128,
    })

    cfg["verl"].update({
        "adv_estimator": "grpo",
        "ppo_epochs": 2,
        "ppo_max_token_len_per_gpu": 123,
        "ppo_mini_batch_size": 4,
        "train_batch_size": 3,
        "val_batch_size": 4,
        "dataloader_num_workers": 5,
        
        "max_response_length": 256,
        "total_epochs": 7,
        "save_freq": 11,
        "project_name": "proj",
        "experiment_name": "exp",
        "nnodes": 2,
    })

    result = VerlConfigParser(cfg)._process_input_config()
    verl = result["verl_config"]

    assert verl["actor_rollout_ref"]["actor"]["optim"]["lr"] == 1e-6
    assert verl["actor_rollout_ref"]["actor"]["optim"]["weight_decay"] == 0.01
    assert verl["actor_rollout_ref"]["actor"]["ppo_epochs"] == 2
    assert verl["actor_rollout_ref"]["actor"]["ppo_mini_batch_size"] == 4
    assert verl["actor_rollout_ref"]["actor"]["ppo_max_token_len_per_gpu"] == 123
    assert verl["actor_rollout_ref"]["actor"]["entropy_coeff"] == 0.01
    assert verl["actor_rollout_ref"]["actor"]["clip_ratio"] == 0.3

    assert verl["actor_rollout_ref"]["rollout"]["dtype"] == "float16"
    assert verl["actor_rollout_ref"]["rollout"]["enforce_eager"] is False
    assert verl["actor_rollout_ref"]["rollout"]["n"] == 2
    assert verl["actor_rollout_ref"]["rollout"]["max_model_len"] == 4096
    assert verl["actor_rollout_ref"]["rollout"]["temperature"] == 0.8
    assert verl["actor_rollout_ref"]["rollout"]["top_k"] == 42
    assert verl["actor_rollout_ref"]["rollout"]["top_p"] == 0.95
    assert verl["actor_rollout_ref"]["rollout"]["gpu_memory_utilization"] == 0.66
    assert verl["actor_rollout_ref"]["rollout"]["max_num_seqs"] == 16
    assert verl["actor_rollout_ref"]["rollout"]["max_num_batched_tokens"] == 7777

    assert verl["algorithm"]["use_kl_in_reward"] is True
    assert verl["algorithm"]["gamma"] == 0.99
    assert verl["algorithm"]["lam"] == 0.95
    assert verl["algorithm"]["adv_estimator"] == "grpo"
    assert verl["algorithm"]["kl_ctrl"]["type"] == "adaptive"
    assert verl["algorithm"]["kl_penalty"] == "mse"
    assert verl["algorithm"]["kl_ctrl"]["kl_coef"] == 0.2
    assert verl["algorithm"]["kl_ctrl"]["horizon"] == 999

    assert verl["data"]["train_batch_size"] == 3
    assert verl["data"]["val_batch_size"] == 4
    assert verl["data"]["dataloader_num_workers"] == 5
    assert verl["data"]["max_prompt_length"] == 128
    assert verl["data"]["max_response_length"] == 256
 
 
def test_process_config_raises_on_missing_verl_section():
    cfg = _make_valid_config()
    cfg.pop("verl")
 
    with pytest.raises(ValueError, match="verl config section is required"):
        VerlConfigParser(cfg).process_config()