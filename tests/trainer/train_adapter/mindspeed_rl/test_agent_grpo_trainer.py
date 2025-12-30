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

from workers.test_agent_hybrid_worker import (MockRLConfig,
                                              MockMegatronConfig,
                                              MockBaseTokenizer,
                                              MockGenerateConfig)


class MockRayGRPOTrainer:
    def __init__(self, actor_worker, ref_worker, reward_list, **kwargs):
        self.actor_worker = actor_worker
        self.ref_worker = ref_worker
        self.reward_list = reward_list
        self.blocking = False
        self.train_iters = 5
        self.n_samples_per_prompt = 2
        self.dataset_additional_keys = []
        self.skip_actor_log_prob = False
        self.save_interval = 5
        self.guarantee_order = False
        self.kl_ctrl = None

        self.transfer_dock_init()

    def save_checkpoint(self, iteration):
        pass

    def compute_advantage(self, *args, **kwargs):
        pass

    def transfer_dock_init(self):
        class TransferDock:
            def __init__(self):
                self.clear = MagicMock()
                self.put_experience = MagicMock()

        self.transfer_dock = TransferDock()


class MockRayActorGroup:
    def __init__(self, *args, **kwargs):
        self.actor_handlers = []

    def initialize(self):
        return self

    def get_iteration(self):
        return 4

    def update(self, kl_ctrl, skip_actor_log_prob=False):
        pass

    def compute_log_prob(self, blocking=False):
        pass

    def compute_ref_log_prob(self, blocking=False):
        pass

    def wait_all_ref_objs_run_over(self):
        pass

    def get_consumed_train_samples(self):
        return 0


class MockRuleReward:
    @staticmethod
    def options(*args, **kwargs):
        return MockRuleReward(*args, **kwargs)

    def __init__(self, *args, **kwargs):
        self.initialize = MagicMock()

    def remote(self, *args, **kwargs):
        return self


class MockRolloutWorker:
    def __init__(self, *args, **kwargs):
        self.wait_init_finished = MagicMock()
        self.init_data_manager = MagicMock()
        self.generate_sequences = MagicMock()

    @classmethod
    def remote(cls, *args, **kwargs):
        return MockRolloutWorker(*args, **kwargs)


class TestAgentGRPO:

    @pytest.fixture(scope="class")
    def patch_modules(self):
        with patch.dict(sys.modules, {
            "mindspeed_rl": MagicMock(),
            "mindspeed_rl.trainer": MagicMock(),
            "mindspeed_rl.trainer.utils": MagicMock(),
            "mindspeed_rl.trainer.utils.transfer_dock": MagicMock(),
            "mindspeed_rl.utils": MagicMock(),
            "mindspeed_rl.utils.pad_process": MagicMock(),
            "mindspeed_rl.utils.tokenizer": MagicMock(),
            "tensordict": MagicMock(),
            "agentic_rl.trainer.rollout.rollout_worker": MagicMock()
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
                patch("mindspeed_rl.RayActorGroup", MockRayActorGroup), \
                patch("mindspeed_rl.RuleReward", MockRuleReward), \
                patch("mindspeed_rl.RayGRPOTrainer", MockRayGRPOTrainer), \
                patch("tensordict.TensorDict"), \
                patch("agentic_rl.trainer.rollout.rollout_worker.RolloutWorker", MockRolloutWorker), \
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

    @pytest.fixture
    def agent_grpo_trainer(self, patch_target):
        from mindspeed_rl import RLConfig, MegatronConfig, GenerateConfig, RayActorGroup, RuleReward
        from mindspeed_rl.utils.tokenizer import BaseTokenizer

        from agentic_rl.configs.agentic_rl_config import AgenticRLConfig
        from agentic_rl.trainer.train_adapter.mindspeed_rl.agent_grpo_trainer import AgentGRPOTrainer

        rl_config = RLConfig()
        megatron_config = MegatronConfig()
        generate_config = GenerateConfig()
        agentic_rl_config = AgenticRLConfig()

        actor_worker = RayActorGroup()
        reference_worker = RayActorGroup()
        reward_list = [RuleReward()]

        tokenizer = BaseTokenizer()

        with patch("agentic_rl.trainer.train_adapter.mindspeed_rl.agent_grpo_trainer."
                   "put_prompts_experience") as mock_put_prompts_experience, \
                patch("agentic_rl.trainer.train_adapter.mindspeed_rl.agent_grpo_trainer."
                      "ray.actor.ActorHandle", MockRuleReward):
            trainer = AgentGRPOTrainer(rl_config=rl_config,
                                       actor_config=megatron_config,
                                       generate_config=generate_config,
                                       agentic_rl_config=agentic_rl_config,
                                       actor_worker=actor_worker,
                                       reference_worker=reference_worker,
                                       reward_list=reward_list,
                                       tokenizer=tokenizer)

            mock_put_prompts_experience.return_value = {}, []

            yield (trainer,
                   {"rl_config": rl_config,
                    "actor_config": megatron_config,
                    "generate_config": generate_config,
                    "agentic_rl_config": agentic_rl_config,
                    "actor_worker": actor_worker,
                    "reference_worker": reference_worker,
                    "reward_list": reward_list,
                    "tokenizer": tokenizer},
                   {"mock_put_prompts_experience": mock_put_prompts_experience})

    def test_init_success_with_no_error(self, agent_grpo_trainer):
        worker, _, _ = agent_grpo_trainer

        assert isinstance(worker.rollout_worker, MockRolloutWorker)

    def test_init_failed_with_params_check(self, agent_grpo_trainer):
        _, targets, patches = agent_grpo_trainer

        from agentic_rl.trainer.train_adapter.mindspeed_rl.agent_grpo_trainer import AgentGRPOTrainer

        params = targets.copy()
        params["rl_config"] = 1
        with pytest.raises(ValueError, match="rl_config must not be none or is not an instance of"):
            AgentGRPOTrainer(**params)

        params = targets.copy()
        params["actor_config"] = 1
        with pytest.raises(ValueError, match="actor_config must not be none or is not an instance of"):
            AgentGRPOTrainer(**params)

        params = targets.copy()
        params["generate_config"] = 1
        with pytest.raises(ValueError, match="generate_config must not be none or is not an instance of"):
            AgentGRPOTrainer(**params)

        params = targets.copy()
        params["agentic_rl_config"] = 1
        with pytest.raises(ValueError, match="agentic_rl_config must not be none or is not an instance of"):
            AgentGRPOTrainer(**params)

        params = targets.copy()
        params["actor_worker"] = 1
        with pytest.raises(ValueError, match="actor_worker must not be none or is not an instance of"):
            AgentGRPOTrainer(**params)

        params = targets.copy()
        params["reference_worker"] = 1
        with pytest.raises(ValueError, match="reference_worker must not be none or is not an instance of"):
            AgentGRPOTrainer(**params)

        params = targets.copy()
        params["tokenizer"] = 1
        with pytest.raises(ValueError, match="tokenizer must not be none or is not an instance of"):
            AgentGRPOTrainer(**params)

        params = targets.copy()
        params["reward_list"] = 1
        with pytest.raises(ValueError, match="reward_list must be a list"):
            AgentGRPOTrainer(**params)

        params["reward_list"] = [1]
        with pytest.raises(ValueError, match="reward in reward_list must be an instance of ray.actor.ActorHandle"):
            AgentGRPOTrainer(**params)

    def test_init_failed_create_rollout_worker_failed(self, agent_grpo_trainer):
        _, targets, patches = agent_grpo_trainer

        from agentic_rl.trainer.train_adapter.mindspeed_rl.agent_grpo_trainer import AgentGRPOTrainer

        with patch("agentic_rl.trainer.train_adapter.mindspeed_rl.agent_grpo_trainer.RolloutWorker") as mock_rollout, \
                patch("agentic_rl.trainer.train_adapter.mindspeed_rl.agent_grpo_trainer."
                      "AgentGRPOTrainer._convert_generate_config"):
            mock_rollout.remote.side_effect = AttributeError("error")
            with pytest.raises(AttributeError):
                AgentGRPOTrainer(**targets)

            mock_rollout.remote.side_effect = RayError("error")
            with pytest.raises(RuntimeError):
                AgentGRPOTrainer(**targets)

            mock_rollout.remote.side_effect = Exception("error")
            with pytest.raises(RuntimeError):
                AgentGRPOTrainer(**targets)

    def test_init_failed_with_init_sharding_manager(self, agent_grpo_trainer):
        _, targets, patches = agent_grpo_trainer

        actor_handler = MagicMock()
        targets["reference_worker"].actor_handlers.append(actor_handler)

        from agentic_rl.trainer.train_adapter.mindspeed_rl.agent_grpo_trainer import AgentGRPOTrainer

        actor_handler.init_sharding_manager.remote.side_effect = RayError("error")
        with pytest.raises(RuntimeError):
            AgentGRPOTrainer(**targets)

        actor_handler.init_sharding_manager.remote.side_effect = AttributeError("error")
        with pytest.raises(RuntimeError):
            AgentGRPOTrainer(**targets)

    def test_init_failed_with_base_class(self, agent_grpo_trainer):
        _, targets, patches = agent_grpo_trainer
        from agentic_rl.trainer.train_adapter.mindspeed_rl.agent_grpo_trainer import AgentGRPOTrainer

        with patch("agentic_rl.trainer.train_adapter.mindspeed_rl.agent_grpo_trainer."
                   "RayGRPOTrainer.transfer_dock_init") as mock_base_dock_init:
            mock_base_dock_init.side_effect = RayError("error")
            with pytest.raises(RuntimeError):
                AgentGRPOTrainer(**targets)

            mock_base_dock_init.side_effect = Exception("error")
            with pytest.raises(RuntimeError):
                AgentGRPOTrainer(**targets)

    def test_fit_failed_with_get_iteration(self, agent_grpo_trainer):
        trainer, _, _ = agent_grpo_trainer

        with patch("agentic_rl.trainer.train_adapter.mindspeed_rl.agent_grpo_trainer."
                   "RayActorGroup.get_iteration") as mock_get_iteration:
            mock_get_iteration.side_effect = AttributeError("error")
            with pytest.raises(AttributeError):
                trainer.fit(iter([1]))

            mock_get_iteration.side_effect = Exception("error")
            with pytest.raises(RuntimeError):
                trainer.fit(iter([1]))

    def test_fit_failed_with_data_iters(self, agent_grpo_trainer):
        trainer, _, _ = agent_grpo_trainer

        trainer.fit(iter([]))

    def test_fit_failed_with_put_prompts_experience(self, agent_grpo_trainer):
        trainer, _, patches = agent_grpo_trainer

        mock_put_prompts_experience = patches["mock_put_prompts_experience"]

        mock_put_prompts_experience.side_effect = RayError("error")
        with pytest.raises(RuntimeError):
            trainer.fit(iter([1]))

        mock_put_prompts_experience.side_effect = AttributeError("error")
        with pytest.raises(RuntimeError):
            trainer.fit(iter([1]))

    def test_fit_failed_with_generate_sequences(self, agent_grpo_trainer):
        trainer, _, _ = agent_grpo_trainer

        trainer.rollout_worker.generate_sequences.remote.side_effect = RayError("error")
        with pytest.raises(RuntimeError):
            trainer.fit(iter([1]))

        trainer.rollout_worker.generate_sequences.remote.side_effect = AttributeError("error")
        with pytest.raises(RuntimeError):
            trainer.fit(iter([1]))

    def test_fit_failed_with_compute_advantage(self, agent_grpo_trainer):
        trainer, _, _ = agent_grpo_trainer

        with patch("agentic_rl.trainer.train_adapter.mindspeed_rl.agent_grpo_trainer."
                   "RayGRPOTrainer.compute_advantage") as mock_compute_advantage:
            mock_compute_advantage.side_effect = RayError("error")
            with pytest.raises(RuntimeError):
                trainer.fit(iter([1]))

            mock_compute_advantage.side_effect = AttributeError("error")
            with pytest.raises(RuntimeError):
                trainer.fit(iter([1]))

    def test_fit_failed_with_compute_ref_log_prob(self, agent_grpo_trainer):
        trainer, _, _ = agent_grpo_trainer

        with patch("agentic_rl.trainer.train_adapter.mindspeed_rl.agent_grpo_trainer."
                   "RayActorGroup.compute_ref_log_prob") as mock_compute_ref_log_prob:
            mock_compute_ref_log_prob.side_effect = RayError("error")
            with pytest.raises(RuntimeError):
                trainer.fit(iter([1]))

            mock_compute_ref_log_prob.side_effect = AttributeError("error")
            with pytest.raises(RuntimeError):
                trainer.fit(iter([1]))

    def test_fit_failed_with_compute_log_prob(self, agent_grpo_trainer):
        trainer, _, _ = agent_grpo_trainer

        with patch("agentic_rl.trainer.train_adapter.mindspeed_rl.agent_grpo_trainer."
                   "RayActorGroup.compute_log_prob") as mock_compute_log_prob:
            mock_compute_log_prob.side_effect = RayError("error")
            with pytest.raises(RuntimeError):
                trainer.fit(iter([1]))

            mock_compute_log_prob.side_effect = AttributeError("error")
            with pytest.raises(RuntimeError):
                trainer.fit(iter([1]))

    def test_fit_failed_update(self, agent_grpo_trainer):
        trainer, _, _ = agent_grpo_trainer

        with patch("agentic_rl.trainer.train_adapter.mindspeed_rl.agent_grpo_trainer."
                   "RayActorGroup.update") as mock_update:
            mock_update.side_effect = RayError("error")
            with pytest.raises(RuntimeError):
                trainer.fit(iter([1]))

            mock_update.side_effect = AttributeError("error")
            with pytest.raises(RuntimeError):
                trainer.fit(iter([1]))

    def test_fit_failed_save_checkpoint(self, agent_grpo_trainer):
        trainer, _, _ = agent_grpo_trainer

        with patch("agentic_rl.trainer.train_adapter.mindspeed_rl.agent_grpo_trainer."
                   "RayGRPOTrainer.save_checkpoint") as mock_save_checkpoint:
            mock_save_checkpoint.side_effect = RayError("error")
            with pytest.raises(RuntimeError):
                trainer.fit(iter([1]))

            mock_save_checkpoint.side_effect = AttributeError("error")
            with pytest.raises(RuntimeError):
                trainer.fit(iter([1]))

    def test_fit_failed_with_check_path(self, agent_grpo_trainer):
        trainer, _, _ = agent_grpo_trainer

        with patch("agentic_rl.base.utils.file_utils.FileCheck.check_data_path_is_valid") as mock_check_path:
            mock_check_path.side_effect = ValueError("error")
            with pytest.raises(ValueError):
                trainer.fit(iter([1]))

            mock_check_path.side_effect = AttributeError("error")
            with pytest.raises(RuntimeError):
                trainer.fit(iter([1]))

    def test_fit_success_with_no_error(self, agent_grpo_trainer):
        trainer, _, _ = agent_grpo_trainer

        trainer.fit(iter([1]))
