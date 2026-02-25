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
import os
from unittest.mock import patch, MagicMock
import pytest
from transformers import PreTrainedTokenizerBase

from agentic_rl.configs.agentic_rl_config import AgenticRLConfig, GenConfig


class MockBaseTokenizer(PreTrainedTokenizerBase):
    def __init__(self):
        self.pad_token_id = 0


class MockConfig:
    def __init__(self):
        self.actor_rollout_ref = MagicMock()
        self.trainer = MagicMock()
        self.algorithm = MagicMock()
        self.critic = MagicMock()
        self.reward_model = MagicMock()
        self.actor_rollout_ref.model = MagicMock()
        self.actor_rollout_ref.model.path = "test/path"
        self.actor_rollout_ref.actor = MagicMock()
        self.actor_rollout_ref.actor.strategy = "fsdp"
        self.critic.strategy = "fsdp"
        self.trainer.get = MagicMock(return_value="auto")
        self.trainer.n_gpus_per_node = 1
        self.trainer.nnodes = 1
        self.algorithm.use_kl_in_reward = False
        self.actor_rollout_ref.actor.use_kl_loss = False
        self.reward_model.get = MagicMock(return_value={})
        self.data = {}


class MockVerlConfigParser:
    def __init__(self, config):
        pass

    def process_config(self):
        return AgenticRLConfig(), {}, MockConfig(), GenConfig(tokenizer_name_or_path="/path/to/tokenizer")


class MockRewardManager:
    def __init__(self):
        pass


class MockAgentGRPOTrainer:
    def __init__(self, *args, **kwargs):
        pass

    def init_workers(self):
        pass

    def fit(self):
        pass

    def shutdown(self):
        pass


class TestTrainAgentGrpo:
    @pytest.fixture(scope="class")
    def patch_modules(self):
        with patch.dict(
            sys.modules,
            {
                "verl": MagicMock(),
                "verl.utils": MagicMock(),
                "verl.utils.hf_tokenizer": MagicMock(),
                "verl.trainer": MagicMock(),
                "verl.trainer.ppo": MagicMock(),
                "verl.trainer.ppo.ray_trainer": MagicMock(),
                "verl.trainer.ppo.reward": MagicMock(),
                "verl.single_controller": MagicMock(),
                "verl.single_controller.ray": MagicMock(),
                "verl.workers": MagicMock(),
                "verl.workers.fsdp_workers": MagicMock(),
                "agentic_rl.trainer.train_adapter.verl.patch": MagicMock(),
                "agentic_rl.trainer.train_adapter.verl.patch.verl_vllm_model_patch": MagicMock(),
                "agentic_rl.trainer.train_adapter.verl.configs": MagicMock(),
                "agentic_rl.trainer.train_adapter.verl.configs.parse_verl_config": MagicMock(),
                "agentic_rl.trainer.train_adapter.verl.agent_grpo_trainer": MagicMock(),
            },
        ):
            yield

    @pytest.fixture(scope="class")
    def patch_target(self, patch_modules):
        with (
            patch("ray.remote") as mock_remote,
            patch("ray.get") as mock_ray_get,
            patch("verl.utils.hf_tokenizer", return_value=MockBaseTokenizer()),
            patch("verl.trainer.ppo.reward.load_reward_manager", return_value=MockRewardManager()),
            patch("verl.single_controller.ray.RayWorkerGroup", MagicMock()),
            patch("verl.workers.fsdp_workers.ActorRolloutRefWorker", MagicMock()),
            patch("verl.workers.fsdp_workers.AsyncActorRolloutRefWorker", MagicMock()),
            patch("verl.trainer.ppo.ray_trainer.ResourcePoolManager", MagicMock()),
            patch("verl.trainer.ppo.ray_trainer.Role", MagicMock()),
            patch("agentic_rl.trainer.train_adapter.verl.patch.verl_vllm_model_patch.apply_vllm_model_patch"),
            patch(
                "agentic_rl.trainer.train_adapter.verl.configs.parse_verl_config.VerlConfigParser", MockVerlConfigParser
            ),
            patch("agentic_rl.trainer.train_adapter.verl.agent_grpo_trainer.AgentGRPOTrainer", MockAgentGRPOTrainer),
            patch("os.path.exists", return_value=True),
            patch("os.path.isfile", return_value=True),
            patch("os.path.isdir", return_value=False),
            patch("os.stat", return_value=MagicMock(st_mode=0o640, st_uid=os.getuid(), st_gid=os.getgid())),
        ):

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

    def test_create_tokenizer_success(self, patch_target):
        from agentic_rl.trainer.train_adapter.verl.train_agent_grpo import _create_tokenizer

        config = MockConfig()
        tokenizer = _create_tokenizer(config)
        assert tokenizer is not None

    def test_create_tokenizer_failure(self, patch_target):
        from agentic_rl.trainer.train_adapter.verl.train_agent_grpo import _create_tokenizer

        config = MockConfig()
        with patch("verl.utils.hf_tokenizer", side_effect=ValueError("Tokenization error")):
            with pytest.raises(ValueError, match="Tokenization error"):
                _create_tokenizer(config)

    def test_define_worker_classes_success(self, patch_target):
        from agentic_rl.trainer.train_adapter.verl.train_agent_grpo import _define_worker_classes

        config = MockConfig()
        role_worker_mapping, ray_worker_group_cls = _define_worker_classes(config)
        assert isinstance(role_worker_mapping, dict)
        assert ray_worker_group_cls is not None

    def test_define_worker_classes_invalid_actor_strategy(self, patch_target):
        from agentic_rl.trainer.train_adapter.verl.train_agent_grpo import _define_worker_classes

        config = MockConfig()
        config.actor_rollout_ref.actor.strategy = "invalid"
        with pytest.raises(ValueError, match="actor strategy invalid is not supported"):
            _define_worker_classes(config)

    def test_define_worker_classes_invalid_critic_strategy(self, patch_target):
        from agentic_rl.trainer.train_adapter.verl.train_agent_grpo import _define_worker_classes

        config = MockConfig()
        config.critic.strategy = "invalid"
        with pytest.raises(ValueError, match="critic strategy invalid is not supported"):
            _define_worker_classes(config)

    def test_define_worker_classes_invalid_use_legacy_worker_impl(self, patch_target):
        from agentic_rl.trainer.train_adapter.verl.train_agent_grpo import _define_worker_classes

        config = MockConfig()
        config.trainer.get = MagicMock(return_value="invalid")
        with pytest.raises(ValueError, match="Invalid use_legacy_worker_impl: invalid"):
            _define_worker_classes(config)

    def test_create_resource_pool_manager(self, patch_target):
        from agentic_rl.trainer.train_adapter.verl.train_agent_grpo import _create_resource_pool_manager

        config = MockConfig()
        rpm = _create_resource_pool_manager(config)
        assert rpm is not None

    def test_train_success(self, patch_target):
        from agentic_rl.trainer.train_adapter.verl.train_agent_grpo import train

        train({})

    def test_train_config_parsing_error(self, patch_target):
        from agentic_rl.trainer.train_adapter.verl.train_agent_grpo import train

        with patch(
            "agentic_rl.trainer.train_adapter.verl.configs.parse_verl_config.VerlConfigParser.process_config",
            side_effect=ValueError("Config parsing error"),
        ):
            with pytest.raises(RuntimeError, match="Configuration or initialization error: Config parsing error"):
                train({})

    def test_train_tokenizer_creation_error(self, patch_target):
        from agentic_rl.trainer.train_adapter.verl.train_agent_grpo import train

        with patch(
            "agentic_rl.trainer.train_adapter.verl.train_agent_grpo._create_tokenizer",
            side_effect=ValueError("Tokenizer error"),
        ):
            with pytest.raises(RuntimeError, match="Configuration or initialization error: Tokenizer error"):
                train({})

    def test_train_worker_creation_error(self, patch_target):
        from agentic_rl.trainer.train_adapter.verl.train_agent_grpo import train

        with patch(
            "agentic_rl.trainer.train_adapter.verl.train_agent_grpo._define_worker_classes",
            side_effect=ValueError("Worker error"),
        ):
            with pytest.raises(RuntimeError, match="Configuration or initialization error: Worker error"):
                train({})

    def test_train_resource_pool_error(self, patch_target):
        from agentic_rl.trainer.train_adapter.verl.train_agent_grpo import train

        with patch(
            "agentic_rl.trainer.train_adapter.verl.train_agent_grpo._create_resource_pool_manager",
            side_effect=ValueError("Resource pool error"),
        ):
            with pytest.raises(RuntimeError, match="Configuration or initialization error: Resource pool error"):
                train({})

    def test_train_reward_manager_error(self, patch_target):
        from agentic_rl.trainer.train_adapter.verl.train_agent_grpo import train

        with patch("verl.trainer.ppo.reward.load_reward_manager", side_effect=ValueError("Reward manager error")):
            with pytest.raises(RuntimeError, match="Configuration or initialization error: Reward manager error"):
                train({})

    def test_train_os_error(self, patch_target):
        from agentic_rl.trainer.train_adapter.verl.train_agent_grpo import train

        with patch(
            "agentic_rl.trainer.train_adapter.verl.train_agent_grpo._create_tokenizer", side_effect=OSError("OS error")
        ):
            with pytest.raises(RuntimeError, match="OS error: OS error"):
                train({})

    def test_train_unexpected_error(self, patch_target):
        from agentic_rl.trainer.train_adapter.verl.train_agent_grpo import train

        with patch(
            "agentic_rl.trainer.train_adapter.verl.train_agent_grpo._create_tokenizer",
            side_effect=Exception("Unexpected error"),
        ):
            with pytest.raises(RuntimeError, match="Unexpected error: Unexpected error"):
                train({})

    def test_train_trainer_initialization_error(self, patch_target):
        from agentic_rl.trainer.train_adapter.verl.train_agent_grpo import train

        with patch(
            "agentic_rl.trainer.train_adapter.verl.agent_grpo_trainer.AgentGRPOTrainer",
            side_effect=AttributeError("Trainer init error"),
        ):
            with pytest.raises(RuntimeError, match="Trainer initialization error: Trainer init error"):
                train({})

    def test_train_trainer_fit_error(self, patch_target):
        from agentic_rl.trainer.train_adapter.verl.train_agent_grpo import train

        with patch(
            "agentic_rl.trainer.train_adapter.verl.agent_grpo_trainer.AgentGRPOTrainer.fit",
            side_effect=Exception("Fit error"),
        ):
            with pytest.raises(RuntimeError, match="Unexpected error during trainer fit: Fit error"):
                train({})
