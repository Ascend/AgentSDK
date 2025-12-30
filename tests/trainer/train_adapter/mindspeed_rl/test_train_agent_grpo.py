#!/usr/bin/env python3
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
import sys
from unittest.mock import patch, MagicMock

import pytest
from ray.exceptions import RayError

from agentic_rl.configs.agentic_rl_config import AgenticRLConfig
from test_agent_grpo_trainer import MockRuleReward, MockRayActorGroup
from workers.test_agent_hybrid_worker import (MockRLConfig,
                                              MockMegatronConfig,
                                              MockBaseTokenizer,
                                              MockGenerateConfig)


class MockTrainer:

    def __init__(self, *args, **kwargs):
        pass

    def fit(self, *args, **kwargs):
        pass


class TestTrainAgentGrpo:

    @pytest.fixture(scope="class")
    def patch_modules(self):
        with patch.dict(sys.modules, {
            "cli": MagicMock(),
            "cli.train_grpo": MagicMock(),
            "mindspeed_rl": MagicMock(),
            "mindspeed_rl.datasets": MagicMock(),
            "mindspeed_rl.datasets.dataloader": MagicMock(),
            "mindspeed_rl.datasets.build_dataset": MagicMock(),
            "mindspeed_rl.datasets.prompt_dataset": MagicMock(),
            "mindspeed_rl.trainer.utils.transfer_dock": MagicMock(),
            "mindspeed_rl.utils": MagicMock(),
            "mindspeed_rl.utils.utils": MagicMock(),
            "mindspeed_rl.utils.tokenizer": MagicMock(),
            "mindspeed_rl.workers": MagicMock(),
            "mindspeed_rl.workers.rule_reward": MagicMock(),
            "mindspeed_rl.workers.scheduler": MagicMock(),
            "mindspeed_rl.workers.scheduler.launcher": MagicMock(),
            "agentic_rl.trainer.train_adapter.mindspeed_rl.configs": MagicMock(),
            "agentic_rl.trainer.train_adapter.mindspeed_rl.configs.parse_config": MagicMock(),
            "agentic_rl.trainer.train_adapter.mindspeed_rl.agent_grpo_trainer": MagicMock(),
            "agentic_rl.trainer.train_adapter.mindspeed_rl.workers.integrated_worker": MagicMock(),
        }):
            yield

    @pytest.fixture(scope="class")
    def patch_target(self, patch_modules):
        with patch("ray.remote") as mock_remote, \
                patch("ray.get") as mock_ray_get, \
                patch("mindspeed_rl.MegatronConfig", MockMegatronConfig), \
                patch("mindspeed_rl.RLConfig", MockRLConfig), \
                patch("mindspeed_rl.GenerateConfig", MockGenerateConfig), \
                patch("mindspeed_rl.utils.tokenizer.BaseTokenizer", MockBaseTokenizer), \
                patch("mindspeed_rl.workers.rule_reward.RuleReward", MockRuleReward), \
                patch("mindspeed_rl.workers.scheduler.launcher.RayActorGroup", MockRayActorGroup), \
                patch("agentic_rl.runner.infer_adapter.vllm.patch.apply_patch"), \
                patch("agentic_rl.trainer.train_adapter.mindspeed_rl.patch.apply_patch"), \
                patch("agentic_rl.base.utils.file_utils.FileCheck.check_data_path_is_valid"):

            def fake_ray_get(*args):
                return args

            mock_ray_get.side_effect = fake_ray_get

            def fake_remote(*args, **kwargs):
                if len(args) == 1 and callable(args[0]) and not kwargs:
                    obj = args[0]
                    obj.remote = obj
                    return obj
                else:
                    def decorator(obj):
                        obj.remote = obj
                        return obj

                    return decorator

            mock_remote.side_effect = fake_remote

            yield

    @pytest.fixture()
    def train_patches(self, patch_target):
        with patch("mindspeed_rl.utils.get_tokenizer"), \
                patch("ray.util.placement_group"), \
                patch("mindspeed_rl.datasets.dataloader.PromptDataLoader"), \
                patch("mindspeed_rl.datasets.build_dataset.build_train_valid_test_datasets") as mock_build_dataset, \
                patch("agentic_rl.trainer.train_adapter.mindspeed_rl.configs.parse_config."
                      "ConfigParser") as mock_config_parser_class, \
                patch("agentic_rl.trainer.train_adapter.mindspeed_rl.agent_grpo_trainer."
                      "AgentGRPOTrainer", MockTrainer):
            parser = mock_config_parser_class.return_value

            parser.process_config.return_value = {
                "agentic_rl_config": AgenticRLConfig(),
                "actor_config": MockMegatronConfig(),
                "reward_config": MockMegatronConfig(),
                "rl_config": MockRLConfig(),
                "generate_config": MockGenerateConfig(),
            }

            mock_build_dataset.return_value = MagicMock(), None, None

            yield

    def test_train_fit_failed_with_tokenizer(self, train_patches):

        from agentic_rl.trainer.train_adapter.mindspeed_rl.train_agent_grpo import train

        with patch("agentic_rl.trainer.train_adapter.mindspeed_rl.train_agent_grpo."
                   "get_tokenizer") as mock_get_tokenizer:
            mock_get_tokenizer.side_effect = ValueError("test")
            with pytest.raises(ValueError):
                train({})

    def test_train_fit_failed_with_create_worker(self, train_patches):
        from agentic_rl.trainer.train_adapter.mindspeed_rl.train_agent_grpo import train

        with patch("agentic_rl.trainer.train_adapter.mindspeed_rl.train_agent_grpo."
                   "RayActorGroup") as mock_ray_actor_group:
            mock_ray_actor_group.side_effect = AttributeError("error")
            with pytest.raises(AttributeError):
                train({})

            mock_ray_actor_group.side_effect = ValueError("error")
            with pytest.raises(RuntimeError):
                train({})

        with patch("agentic_rl.trainer.train_adapter.mindspeed_rl.train_agent_grpo."
                   "ConfigParser") as mock_config_parser_class:
            parser = mock_config_parser_class.return_value
            parser.process_config.return_value = {
                "agentic_rl_config": AgenticRLConfig(),
                "actor_config": MockMegatronConfig(),
                "reward_config": MockMegatronConfig(),
                "rl_config": MockRLConfig(use_integrated_worker=False),
                "generate_config": MockGenerateConfig(),
            }

            with pytest.raises(ValueError):
                train({})

    def test_train_fit_failed_with_reward_worker(self, train_patches):
        from agentic_rl.trainer.train_adapter.mindspeed_rl.train_agent_grpo import train

        with patch("agentic_rl.trainer.train_adapter.mindspeed_rl.train_agent_grpo."
                   "RuleReward.options") as mock_rule_reward_options:
            mock_rule_reward_options.side_effect = RayError("error")
            with pytest.raises(RayError):
                train({})

            mock_rule_reward_options.side_effect = AttributeError("error")
            with pytest.raises(RuntimeError):
                train({})

        with patch("agentic_rl.trainer.train_adapter.mindspeed_rl.train_agent_grpo."
                   "ConfigParser") as mock_config_parser_class:
            parser = mock_config_parser_class.return_value
            parser.process_config.return_value = {
                "agentic_rl_config": AgenticRLConfig(),
                "actor_config": MockMegatronConfig(),
                "reward_config": MockMegatronConfig(),
                "rl_config": MockRLConfig(rule_reward=False),
                "generate_config": MockGenerateConfig(),
            }

            with pytest.raises(ValueError):
                train({})

    def test_train_fit_failed_with_process_datasets(self, train_patches):
        from agentic_rl.trainer.train_adapter.mindspeed_rl.train_agent_grpo import train

        with patch("agentic_rl.trainer.train_adapter.mindspeed_rl.train_agent_grpo."
                   "build_train_valid_test_datasets") as mock_build_dataset:
            mock_build_dataset.side_effect = ValueError("test")
            with pytest.raises(ValueError):
                train({})

            mock_build_dataset.side_effect = AttributeError("test")
            with pytest.raises(RuntimeError):
                train({})

        with patch("agentic_rl.trainer.train_adapter.mindspeed_rl.train_agent_grpo."
                   "PromptDataLoader") as mock_data_loader_class:
            mock_data_loader_class.side_effect = AttributeError("test")
            with pytest.raises(AttributeError):
                train({})

            mock_data_loader_class.side_effect = ValueError("test")
            with pytest.raises(RuntimeError):
                train({})

    def test_train_fit_failed_with_create_trainer(self, train_patches):
        from agentic_rl.trainer.train_adapter.mindspeed_rl.train_agent_grpo import train

        with patch("agentic_rl.trainer.train_adapter.mindspeed_rl.train_agent_grpo."
                   "AgentGRPOTrainer") as mock_trainer_class:
            mock_trainer_class.side_effect = AttributeError("test")
            with pytest.raises(AttributeError):
                train({})

            mock_trainer_class.side_effect = ZeroDivisionError("test")
            with pytest.raises(RuntimeError):
                train({})

    def test_train_fit_failed_with_fit(self, train_patches):
        from agentic_rl.trainer.train_adapter.mindspeed_rl.train_agent_grpo import train

        with patch("agentic_rl.trainer.train_adapter.mindspeed_rl.train_agent_grpo."
                   "AgentGRPOTrainer.fit") as mock_fit:
            mock_fit.side_effect = AttributeError("test")
            with pytest.raises(AttributeError):
                train({})

            mock_fit.side_effect = FileExistsError("test")
            with pytest.raises(RuntimeError):
                train({})

    def test_train_fit_success(self, train_patches):
        from agentic_rl.trainer.train_adapter.mindspeed_rl.train_agent_grpo import train
        train({})
