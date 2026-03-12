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

import sys
from unittest.mock import patch, MagicMock
import pytest
import numpy as np
import torch
from transformers import PreTrainedTokenizerBase
from ray.exceptions import RayError
from agentic_rl.configs.agentic_rl_config import AgenticRLConfig, GenConfig


class MockConfig:
    def __init__(self):
        self.actor_rollout_ref = MagicMock()
        self.trainer = MagicMock()
        self.algorithm = MagicMock()
        self.actor_rollout_ref.hybrid_engine = True
        self.actor_rollout_ref.rollout = MagicMock()
        self.actor_rollout_ref.rollout.mode = "async"
        self.actor_rollout_ref.rollout.n = 2
        self.actor_rollout_ref.rollout.actor = MagicMock()
        self.actor_rollout_ref.rollout.actor.loss_agg_mode = "mean"
        self.trainer.total_epochs = 1
        self.trainer.save_freq = 4
        self.trainer.critic_warmup = 0
        self.trainer.total_training_steps = 10
        self.algorithm.adv_estimator = "gae"
        self.algorithm.gamma = 0.99
        self.algorithm.lam = 0.95
        self.algorithm.norm_adv_by_std_in_grpo = True


class MockBaseTokenizer(PreTrainedTokenizerBase):
    def __init__(self):
        self.pad_token_id = 0


class MockRayWorkerGroup:
    def __init__(self, *args, **kwargs):
        self.world_size = 0

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

    def update_actor(self, batch):
        pass


class MockResourcePoolManager:
    def __init__(self, resource_pool_spec=None, mapping=None):
        self.resource_pool_spec = resource_pool_spec
        self.mapping = mapping

    def initialize(self):
        return self

    def get_n_gpus(self):
        return 4


class FakeTensorDict(dict):
    def __init__(self, data):
        super().__init__(data)
        self.is_locked = False


class MockTensorDict(dict):
    def __init__(self, data):
        super().__init__(data)
        for k, v in data.items():
            setattr(self, k, v)
        self.is_locked = False

    def unlock_(self):
        self.is_locked = False


class MockDataProto:
    def __init__(self):
        data = {
            "prompts": np.array(["test prompt"], dtype=object),
            "input_ids": torch.tensor([[1, 2, 3]], dtype=torch.long),
            "attention_mask": torch.tensor([[1, 1, 0]], dtype=torch.long),
            "position_ids": torch.tensor([[0, 1, 2]], dtype=torch.long),
            "response_mask": torch.tensor([[0, 1, 1]], dtype=torch.bool),
            "entropys": torch.tensor([[0.1, 0.2, 0.3]], dtype=torch.float32),
            "token_level_scores": torch.tensor([[0.5, 0.6, 0.7]], dtype=torch.float32),
            "is_last_step": torch.tensor([True, False, False]),
            "is_pad_step": torch.tensor([False, False, False]),
        }

        self.batch = MockTensorDict(data)
        self.non_tensor_batch = {
            "uid": ["test"],
            "is_last_step": [True],
            "is_pad_step": [False],
        }
        self.meta_info = {}
        self.is_locked = False

    @classmethod
    def from_single_dict(cls, batch_dict):
        return cls()

    def repeat(self, repeat_times, interleave=True):
        return MockDataProto()

    def pop(self, batch_keys):
        for key in batch_keys:
            self.batch.pop(key, None)

    def union(self, other):
        return self


class MockRolloutWorker:
    def __init__(self, *args, **kwargs):
        self.wait_init_finished = MagicMock()
        self.init_data_manager = MagicMock()
        self.generate_sequences_verl = MagicMock()

    @classmethod
    def remote(cls, *args, **kwargs):
        return MockRolloutWorker(*args, **kwargs)


class MockCriticWG:
    def __init__(self):
        self.world_size = 1


class MockRefPolicyWG:
    def __init__(self):
        self.world_size = 1


class MockRMWG:
    def __init__(self):
        self.world_size = 1


class MockActorRolloutWG:
    def __init__(self):
        self.world_size = 1


class MockDataLoader:
    def __init__(self, num_batches=1):
        self.num_batches = num_batches
        self.count = 0

    def __iter__(self):
        self.count = 0
        return self

    def __next__(self):
        if self.count < self.num_batches:
            self.count += 1
            return {"prompts": ["test prompt"]}
        else:
            raise StopIteration


class MockRewardFn:
    def __call__(self, *args, **kwargs):
        return self

    def remote(self, *args, **kwargs):
        return self


class MockRayPPOTrainer:
    def __init__(self, **kwargs):
        self.global_steps = 0
        self.use_critic = False
        self.use_rm = False
        self.use_reference_policy = False
        self.hybrid_engine = True
        self.actor_rollout_wg = MockActorRolloutWG()
        self.actor_wg = MockRayWorkerGroup()
        self.rollout_wg = MockRayWorkerGroup()
        self.critic_wg = MockCriticWG()
        self._compute_rollout_probs_diff = MagicMock()
        self.config = MockConfig()
        self.tokenizer = MockBaseTokenizer()
        self.tokenizer.return_value = {
            "input_ids": torch.tensor([[1, 2, 3]]),
            "attention_mask": torch.tensor([[1, 1, 1]]),
        }

        self.role_worker_mapping = {"role1": MockRayWorkerGroup()}
        self.resource_pool_manager = MockResourcePoolManager()
        self.reward_fn = MockRewardFn()
        self.val_reward_fn = MockRewardFn()
        self.train_dataloader = MockDataLoader(num_batches=1)
        self.total_training_steps = 10

    def _load_checkpoint(self):
        pass

    def _balance_batch(self, batch, metrics):
        pass

    def init_workers(self):
        pass

    def _validate_agent(self):
        return {}

    def _save_checkpoint(self):
        pass


class TestAgentGRPOTrainer:
    @pytest.fixture(scope="class")
    def patch_modules(self):
        with patch.dict(
            sys.modules,
            {
                "verl": MagicMock(),
                "verl.utils": MagicMock(),
                "verl.utils.metric": MagicMock(),
                "verl.experimental": MagicMock(),
                "verl.protocol": MagicMock(),
                "verl.trainer": MagicMock(),
                "verl.trainer.ppo": MagicMock(),
                "verl.trainer.ppo.core_algos": MagicMock(),
                "verl.trainer.ppo.ray_trainer": MagicMock(),
                "verl.trainer.ppo.metric_utils": MagicMock(),
            },
        ):
            yield

    @pytest.fixture(scope="class")
    def patch_target(self, patch_modules):
        with (
            patch("ray.remote") as mock_remote,
            patch("ray.get") as mock_ray_get,
            patch("verl.DataProto", MockDataProto),
            patch("verl.utils.hf_tokenizer", MockBaseTokenizer),
            patch("agentic_rl.base.utils.file_utils.FileCheck.check_data_path_is_valid"),
            patch("verl.trainer.ppo.ray_trainer.RayPPOTrainer", MockRayPPOTrainer),
            patch("verl.trainer.ppo.ray_trainer.compute_data_metrics"),
            patch("verl.trainer.ppo.ray_trainer.compute_timing_metrics"),
            patch("verl.trainer.ppo.ray_trainer.marked_timer"),
            patch("verl.trainer.ppo.ray_trainer.reduce_metrics"),
            patch("verl.trainer.ppo.ray_trainer.Role"),
            patch("verl.trainer.ppo.ray_trainer.WorkerType"),
            patch("verl.trainer.ppo.ray_trainer.ResourcePoolManager", MockResourcePoolManager),
            patch("agentic_rl.trainer.train_adapter.verl.patch.apply_patch"),
            patch("verl.trainer.ppo.ray_trainer.RayWorkerGroup", MockRayWorkerGroup),
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

    @pytest.fixture
    def agent_grpo_trainer(self, patch_target):
        from agentic_rl.trainer.train_adapter.verl.agent_grpo_trainer import AgentGRPOTrainer

        config = MockConfig()
        tokenizer = MockBaseTokenizer()
        role_worker_mapping = {"role1": MockRayWorkerGroup()}
        resource_pool_manager = MockResourcePoolManager()
        ray_worker_group_cls = MockRayWorkerGroup
        reward_fn = MockRewardFn()
        val_reward_fn = MockRewardFn()
        tokenizer_path = "/path/to/tokenizer"
        dataset_additional_keys = ["key1", "key2"]
        generate_config = GenConfig()
        agentic_rl_config = AgenticRLConfig()

        with patch("verl.trainer.ppo.ray_trainer.RayPPOTrainer"):
            trainer = AgentGRPOTrainer(
                config=config,
                tokenizer=tokenizer,
                role_worker_mapping=role_worker_mapping,
                resource_pool_manager=resource_pool_manager,
                ray_worker_group_cls=ray_worker_group_cls,
                reward_fn=reward_fn,
                val_reward_fn=val_reward_fn,
                tokenizer_path=tokenizer_path,
                dataset_additional_keys=dataset_additional_keys,
                generate_config=generate_config,
                agentic_rl_config=agentic_rl_config,
            )

            yield (
                trainer,
                {
                    "config": config,
                    "tokenizer": tokenizer,
                    "role_worker_mapping": role_worker_mapping,
                    "resource_pool_manager": resource_pool_manager,
                    "ray_worker_group_cls": ray_worker_group_cls,
                    "reward_fn": reward_fn,
                    "val_reward_fn": val_reward_fn,
                    "tokenizer_path": tokenizer_path,
                    "dataset_additional_keys": dataset_additional_keys,
                    "generate_config": generate_config,
                    "agentic_rl_config": agentic_rl_config,
                },
            )

    def test_init_success_with_no_error(self, agent_grpo_trainer):
        worker, _ = agent_grpo_trainer
        assert worker is not None

    def test_check_args_valid_signature(self, agent_grpo_trainer):
        from agentic_rl.trainer.train_adapter.verl.agent_grpo_trainer import AgentGRPOTrainer

        _, targets = agent_grpo_trainer
        agentic_configs = (targets["generate_config"], targets["agentic_rl_config"])
        AgentGRPOTrainer._check_args(
            targets["tokenizer"],
            targets["role_worker_mapping"],
            targets["resource_pool_manager"],
            targets["ray_worker_group_cls"],
            agentic_configs,
        )
 
    def test_check_args_invalid_agentic_configs(self, agent_grpo_trainer):
        """_check_args raises when agentic_configs tuple has invalid generate_config or agentic_rl_config."""
        from agentic_rl.trainer.train_adapter.verl.agent_grpo_trainer import AgentGRPOTrainer

        _, targets = agent_grpo_trainer
        with pytest.raises(ValueError, match="generate_config must be an instance of GenConfig"):
            AgentGRPOTrainer._check_args(
                targets["tokenizer"],
                targets["role_worker_mapping"],
                targets["resource_pool_manager"],
                targets["ray_worker_group_cls"],
                (1, AgenticRLConfig()),  # invalid generate_config
            )
        with pytest.raises(ValueError, match="agentic_rl_config must be an instance of AgenticRLConfig"):
            AgentGRPOTrainer._check_args(
                targets["tokenizer"],
                targets["role_worker_mapping"],
                targets["resource_pool_manager"],
                targets["ray_worker_group_cls"],
                (GenConfig(), 1),  # invalid agentic_rl_config
            )

    def test_init_failed_with_params_check(self, agent_grpo_trainer):
        from agentic_rl.trainer.train_adapter.verl.agent_grpo_trainer import AgentGRPOTrainer

        _, targets = agent_grpo_trainer

        params = targets.copy()
        params["tokenizer"] = 1
        with pytest.raises(ValueError, match="tokenizer must be an instance of PreTrainedTokenizerBase"):
            AgentGRPOTrainer(**params)

        params = targets.copy()
        params["role_worker_mapping"] = 1
        with pytest.raises(ValueError, match="role_worker_mapping must be a dict"):
            AgentGRPOTrainer(**params)

        params = targets.copy()
        params["resource_pool_manager"] = 1
        with pytest.raises(ValueError, match="resource_pool_manager must be an instance of ResourcePoolManager"):
            AgentGRPOTrainer(**params)

        params = targets.copy()
        params["ray_worker_group_cls"] = type("1")
        with pytest.raises(ValueError, match="ray_worker_group_cls must be a subclass of RayWorkerGroup"):
            AgentGRPOTrainer(**params)

        params = targets.copy()
        params["generate_config"] = 1
        with pytest.raises(ValueError, match="generate_config must be an instance of GenConfig"):
            AgentGRPOTrainer(**params)

        params = targets.copy()
        params["agentic_rl_config"] = 1
        with pytest.raises(ValueError, match="agentic_rl_config must be an instance of AgenticRLConfig"):
            AgentGRPOTrainer(**params)

    def test_init_failed_with_rollout_worker(self, agent_grpo_trainer):
        _, targets = agent_grpo_trainer
        import agentic_rl.trainer.train_adapter.verl.agent_grpo_trainer as trainer_mod

        with patch.object(trainer_mod, "RolloutWorker") as MockRW:
            MockRW.remote.side_effect = AttributeError("init rollout worker missing attr")
            trainer = trainer_mod.AgentGRPOTrainer(**targets)

            with pytest.raises(AttributeError, match="init rollout worker missing attr"):
                trainer.init_workers()

    def test_prepare_batch_success(self, agent_grpo_trainer):
        trainer, _ = agent_grpo_trainer

        batch_dict = {
            "input_ids": [[1, 2, 3]],
            "attention_mask": [[1, 1, 0]],
            "position_ids": [[0, 1, 2]],
            "is_last_step": [True],
            "is_pad_step": [False],
        }
        result = trainer._prepare_batch(batch_dict)

        from verl import DataProto

        assert isinstance(result, DataProto)
        assert "input_ids" not in result.batch
        assert "attention_mask" not in result.batch
        assert "position_ids" not in result.batch
        assert "is_last_step" in result.non_tensor_batch
        assert "is_pad_step" in result.non_tensor_batch
        assert "uid" in result.non_tensor_batch
        assert len(result.non_tensor_batch["uid"]) == 1
        assert isinstance(result.non_tensor_batch["uid"][0], str)

    def test_prepare_batch_failed(self, agent_grpo_trainer):
        trainer, _ = agent_grpo_trainer
        with patch("verl.DataProto.from_single_dict") as mock_from_single_dict:
            mock_from_single_dict.side_effect = Exception("error")
            with pytest.raises(Exception):
                trainer._prepare_batch({"prompts": ["test prompt"]})

    def test_default_console_metrics_returns_expected_list(self, patch_target):
        from agentic_rl.trainer.train_adapter.verl.agent_grpo_trainer import AgentGRPOTrainer

        result = AgentGRPOTrainer._default_console_metrics()
        assert isinstance(result, list)
        assert len(result) == 15
        assert "actor/entropy" in result
        assert "actor/grad_norm" in result
        assert "actor/lr" in result
        assert "traj/steps_mean" in result
        assert "batch/solve_all" in result
        assert "batch/solve_partial" in result
        assert "batch/solve_none" in result
        assert "perf/max_memory_allocated_gb" in result
        assert "timing_s/update_actor" in result

    def test_format_metrics_for_console_empty_metrics(self, agent_grpo_trainer):
        trainer, _ = agent_grpo_trainer
        result = trainer._format_metrics_for_console({})
        assert result == ""

    def test_format_metrics_for_console_float_normal_range(self, agent_grpo_trainer):
        trainer, _ = agent_grpo_trainer
        metrics = {"actor/entropy": 1.2345, "actor/lr": 5.0}
        result = trainer._format_metrics_for_console(metrics)
        assert "actor/entropy=1.2345" in result
        assert "actor/lr=5.0000" in result

    def test_format_metrics_for_console_float_small_scientific(self, agent_grpo_trainer):
        trainer, _ = agent_grpo_trainer
        metrics = {"actor/entropy": 1e-5}
        result = trainer._format_metrics_for_console(metrics)
        assert "1.00e-05" in result or "1e-05" in result

    def test_format_metrics_for_console_float_large_scientific(self, agent_grpo_trainer):
        trainer, _ = agent_grpo_trainer
        metrics = {"actor/entropy": 15000.0}
        result = trainer._format_metrics_for_console(metrics)
        assert "1.50e+04" in result or "15000" in result

    def test_format_metrics_for_console_non_float_values(self, agent_grpo_trainer):
        trainer, _ = agent_grpo_trainer
        metrics = {"batch/solve_all": 42}
        result = trainer._format_metrics_for_console(metrics)
        assert "batch/solve_all=42" in result

    def test_format_metrics_for_console_only_in_console_metrics(self, agent_grpo_trainer):
        trainer, _ = agent_grpo_trainer
        metrics = {"actor/entropy": 0.5, "unknown/key": 1.0}
        result = trainer._format_metrics_for_console(metrics)
        assert "actor/entropy" in result
        assert "unknown/key" not in result

    def test_format_metrics_for_console_custom_console_metrics(self, patch_target):
        from agentic_rl.trainer.train_adapter.verl.agent_grpo_trainer import AgentGRPOTrainer

        with patch("verl.trainer.ppo.ray_trainer.RayPPOTrainer"):
            trainer = AgentGRPOTrainer(
                config=MockConfig(),
                tokenizer=MockBaseTokenizer(),
                role_worker_mapping={"role1": MockRayWorkerGroup()},
                resource_pool_manager=MockResourcePoolManager(),
                ray_worker_group_cls=MockRayWorkerGroup,
                reward_fn=MockRewardFn(),
                val_reward_fn=MockRewardFn(),
                tokenizer_path="/path/to/tokenizer",
                dataset_additional_keys=[],
                generate_config=GenConfig(),
                agentic_rl_config=AgenticRLConfig(),
                console_metrics=["custom/a", "custom/b"],
            )
        result = trainer._format_metrics_for_console({"custom/a": 1.0, "custom/b": 2.0})
        assert "custom/a=1.0000" in result
        assert "custom/b=2.0000" in result

    def test_log_metrics_to_tensorboard_no_writer(self, agent_grpo_trainer):
        trainer, _ = agent_grpo_trainer
        trainer.tensorboard_writer = None
        trainer._log_metrics_to_tensorboard({"actor/entropy": 0.5}, step=1)
        # No exception; no writer to call

    def test_log_metrics_to_tensorboard_calls_add_scalar(self, agent_grpo_trainer):
        trainer, _ = agent_grpo_trainer
        mock_writer = MagicMock()
        trainer.tensorboard_writer = mock_writer
        trainer.tensorboard_flush_interval = 10
        metrics = {"actor/entropy": 0.5, "actor/lr": 1e-4}
        trainer._log_metrics_to_tensorboard(metrics, step=1)
        assert mock_writer.add_scalar.call_count == 2
        mock_writer.add_scalar.assert_any_call("train/actor/entropy", 0.5, 1)
        mock_writer.add_scalar.assert_any_call("train/actor/lr", 1e-4, 1)

    def test_log_metrics_to_tensorboard_skips_nan_and_inf(self, agent_grpo_trainer):
        trainer, _ = agent_grpo_trainer
        mock_writer = MagicMock()
        trainer.tensorboard_writer = mock_writer
        trainer.tensorboard_flush_interval = 10
        metrics = {"valid": 1.0, "nan_val": float("nan"), "inf_val": float("inf")}
        trainer._log_metrics_to_tensorboard(metrics, step=1)
        mock_writer.add_scalar.assert_called_once_with("train/valid", 1.0, 1)

    def test_log_metrics_to_tensorboard_skips_non_numeric(self, agent_grpo_trainer):
        trainer, _ = agent_grpo_trainer
        mock_writer = MagicMock()
        trainer.tensorboard_writer = mock_writer
        trainer.tensorboard_flush_interval = 10
        metrics = {"num": 1.0, "str_val": "hello", "list_val": [1, 2]}
        trainer._log_metrics_to_tensorboard(metrics, step=1)
        mock_writer.add_scalar.assert_called_once_with("train/num", 1.0, 1)

    def test_log_metrics_to_tensorboard_flushes_on_interval(self, agent_grpo_trainer):
        trainer, _ = agent_grpo_trainer
        mock_writer = MagicMock()
        trainer.tensorboard_writer = mock_writer
        trainer.tensorboard_flush_interval = 5
        with patch.object(trainer, "_flush_tensorboard") as mock_flush:
            trainer._log_metrics_to_tensorboard({"a": 1.0}, step=10)
            mock_writer.add_scalar.assert_called_once_with("train/a", 1.0, 10)
            mock_flush.assert_called_once()

    def test_log_metrics_to_tensorboard_no_flush_when_not_interval(self, agent_grpo_trainer):
        trainer, _ = agent_grpo_trainer
        mock_writer = MagicMock()
        trainer.tensorboard_writer = mock_writer
        trainer.tensorboard_flush_interval = 5
        with patch.object(trainer, "_flush_tensorboard") as mock_flush:
            trainer._log_metrics_to_tensorboard({"a": 1.0}, step=3)
            mock_flush.assert_not_called()

    def test_flush_tensorboard_no_writer(self, agent_grpo_trainer):
        trainer, _ = agent_grpo_trainer
        trainer.tensorboard_writer = None
        trainer._flush_tensorboard()
        # No exception

    def test_flush_tensorboard_calls_flush(self, agent_grpo_trainer):
        trainer, _ = agent_grpo_trainer
        mock_writer = MagicMock()
        trainer.tensorboard_writer = mock_writer
        trainer._flush_tensorboard()
        mock_writer.flush.assert_called_once()

    def test_flush_tensorboard_handles_oserror(self, agent_grpo_trainer):
        trainer, _ = agent_grpo_trainer
        mock_writer = MagicMock()
        mock_writer.flush.side_effect = OSError("disk full")
        trainer.tensorboard_writer = mock_writer
        trainer._flush_tensorboard()
        mock_writer.flush.assert_called_once()

    def test_flush_tensorboard_handles_runtime_error(self, agent_grpo_trainer):
        trainer, _ = agent_grpo_trainer
        mock_writer = MagicMock()
        mock_writer.flush.side_effect = RuntimeError("writer closed")
        trainer.tensorboard_writer = mock_writer
        trainer._flush_tensorboard()
        mock_writer.flush.assert_called_once()

    def test_pad_dataproto_to_world_size_success(self, agent_grpo_trainer):
        trainer, _ = agent_grpo_trainer

        trainer.use_critic = False
        trainer.use_reference_policy = False
        trainer.use_rm = False
        trainer.hybrid_engine = True
        trainer.actor_rollout_wg = MockActorRolloutWG()
        batch = MockDataProto()

        with patch("agentic_rl.trainer.train_adapter.verl.agent_grpo_trainer.pad_dataproto_to_divisor") as mock_pad:
            mock_pad.return_value = (batch, 0)
            result = trainer._pad_dataproto_to_world_size(batch)
            assert result is not None

    def test_pad_dataproto_to_world_size_no_world_sizes(self, agent_grpo_trainer):
        trainer, _ = agent_grpo_trainer
        trainer.use_critic = False
        trainer.use_reference_policy = False
        trainer.use_rm = False
        trainer.hybrid_engine = False
        trainer.actor_wg = MockRayWorkerGroup()
        trainer.critic_wg = MockRayWorkerGroup()
        trainer.rollout_wg = MockRayWorkerGroup()
        trainer.ref_policy_wg = MockRayWorkerGroup()

        batch = MockDataProto()

        with patch("agentic_rl.trainer.train_adapter.verl.agent_grpo_trainer.pad_dataproto_to_divisor") as mock_pad:
            mock_pad.return_value = (batch, 2)
            result = trainer._pad_dataproto_to_world_size(batch)
            assert result is batch

    def test_reject_low_reward_sequences_success(self, agent_grpo_trainer):
        trainer, _ = agent_grpo_trainer
        batch = MockDataProto()
        reward_tensor = torch.tensor([[0.5, 0.6, 0.7]])
        metrics = {}

        trainer._reject_low_reward_sequences(batch, reward_tensor, metrics)
        assert "batch/solve_none" in metrics
        assert "batch/solve_all" in metrics
        assert "batch/solve_partial" in metrics

    def test_compute_rewards_and_advantages_success(self, agent_grpo_trainer):
        trainer, _ = agent_grpo_trainer
        trainer.use_rm = False
        trainer.use_reference_policy = False
        trainer.use_critic = False
        trainer.hybrid_engine = True
        trainer.actor_rollout_wg = MockRayWorkerGroup()
        trainer.rm_wg = MockRMWG()
        trainer.ref_policy_wg = MockRefPolicyWG()
        trainer.critic_wg = MockCriticWG()

        batch = MockDataProto()
        metrics = {}
        timing_raw = {}

        with (
            patch(
                "agentic_rl.trainer.train_adapter.verl.agent_grpo_trainer.compute_advantage"
            ) as mock_compute_advantage,
            patch("agentic_rl.trainer.train_adapter.verl.agent_grpo_trainer.agg_loss") as mock_agg_loss,
            patch("agentic_rl.trainer.train_adapter.verl.agent_grpo_trainer.reduce_metrics") as mock_reduce_metrics,
        ):
            mock_compute_advantage.return_value = batch
            mock_agg_loss.return_value = torch.tensor(0.1)
            
            mock_reduce_metrics.return_value = {}

            mock_result = MagicMock()
            mock_result.batch = {"entropys": [0.5, 0.6, 0.7]}
            trainer.actor_rollout_wg.compute_log_prob = MagicMock(return_value=mock_result)

            trainer._compute_rewards_and_advantages(batch, metrics, timing_raw)
            assert "actor/entropy" in metrics

    def test_update_actor_and_critic_success(self, agent_grpo_trainer):
        trainer, _ = agent_grpo_trainer
        trainer.use_critic = False
        trainer.use_reference_policy = False
        trainer.config.trainer.critic_warmup = 0
        trainer.global_steps = 1
        # Ensure worker health check passes by providing a mock rollout worker
        trainer.rollout_worker = MagicMock()
        trainer.rollout_worker.wait_init_finished.remote.return_value = None
        trainer.actor_rollout_wg = MagicMock()
        batch = MockDataProto()
        metrics = {}
        timing_raw = {}

        mock_actor_output = MagicMock()
        mock_actor_output.meta_info = {"metrics": {"actor_loss": 0.5, "actor_entropy": 1.2}}
        trainer.actor_rollout_wg.update_actor.return_value = mock_actor_output

        with (
            patch("agentic_rl.trainer.train_adapter.verl.agent_grpo_trainer.reduce_metrics") as mock_reduce,
            patch("agentic_rl.trainer.train_adapter.verl.agent_grpo_trainer.marked_timer") as mock_timer,
            patch("ray.get") as mock_ray_get,
        ):

            def side_effect_function(x):
                return x

            mock_reduce.side_effect = side_effect_function
            mock_timer.return_value.__enter__ = MagicMock()
            mock_timer.return_value.__exit__ = MagicMock()
            # Simulate successful worker health check; accept timeout kwarg
            def fake_ray_get(obj, timeout=None):
                return obj
            mock_ray_get.side_effect = fake_ray_get

            trainer._update_actor_and_critic(batch, metrics, timing_raw)
            trainer.actor_rollout_wg.update_actor.assert_called_once_with(batch)
            mock_reduce.assert_called_once_with({"actor_loss": 0.5, "actor_entropy": 1.2})
            assert metrics == {"actor_loss": 0.5, "actor_entropy": 1.2}

    def test_train_step_success(self, agent_grpo_trainer):
        trainer, _ = agent_grpo_trainer
        trainer.rollout_worker = MockRolloutWorker()
        trainer.use_critic = False
        trainer.use_reference_policy = False
        trainer.hybrid_engine = True
        trainer.actor_rollout_wg = MockRayWorkerGroup()
        trainer.rm_wg = MockRMWG()
        trainer.ref_policy_wg = MockRefPolicyWG()
        trainer.critic_wg = MockCriticWG()
        trainer._pad_dataproto_to_world_size = MagicMock(return_value=MockDataProto())
        trainer._balance_batch = MagicMock()

        with (
            patch("ray.get") as mock_ray_get,
            patch("verl.trainer.ppo.ray_trainer.compute_data_metrics") as mock_compute_data_metrics,
            patch("verl.trainer.ppo.ray_trainer.compute_timing_metrics") as mock_compute_timing_metrics,
            patch("verl.trainer.ppo.ray_trainer.marked_timer") as mock_marked_timer,
            patch("agentic_rl.trainer.train_adapter.verl.agent_grpo_trainer.agg_loss") as mock_agg_loss,
        ):
            mock_ray_get.return_value = (MockDataProto(), {})
            mock_compute_data_metrics.return_value = {}
            mock_compute_timing_metrics.return_value = {}
            mock_marked_timer.return_value.__enter__ = MagicMock()
            mock_marked_timer.return_value.__exit__ = MagicMock()
            mock_agg_loss.return_value = torch.tensor(0.1)

            trainer.actor_rollout_wg = MagicMock()
            mock_result = MagicMock()
            mock_result.batch = {"entropys": [0.5, 0.6, 0.7]}
            trainer.actor_rollout_wg.compute_log_prob = MagicMock(return_value=mock_result)
            trainer.actor_rollout_wg.update_actor.return_value = MagicMock(
                meta_info={"metrics": {"loss": 0.5, "entropy": 1.2}}
            )

            batch = MockDataProto()
            result = trainer._train_step(batch)
            assert isinstance(result, dict)

    def test_train_step_failed_with_ray_error(self, agent_grpo_trainer):
        trainer, _ = agent_grpo_trainer
        trainer.rollout_worker = MockRolloutWorker()

        with patch("ray.get") as mock_ray_get:
            mock_ray_get.side_effect = RayError("error")

            batch = MockDataProto()
            with pytest.raises(RayError):
                trainer._train_step(batch)

    def test_train_step_failed_with_general_error(self, agent_grpo_trainer):
        trainer, _ = agent_grpo_trainer
        trainer.rollout_worker = MockRolloutWorker()

        with patch("ray.get") as mock_ray_get:
            mock_ray_get.side_effect = Exception("error")

            batch = MockDataProto()
            with pytest.raises(Exception):
                trainer._train_step(batch)

    def test_fit_success_with_no_error(self, agent_grpo_trainer):
        trainer, _ = agent_grpo_trainer
        trainer.train_dataloader = MockDataLoader(num_batches=1)
        trainer._load_checkpoint = MagicMock()
        trainer._save_checkpoint = MagicMock()
        trainer._validate_agent = MagicMock(return_value={})
        trainer._prepare_batch = MagicMock(return_value=MockDataProto())
        trainer._train_step = MagicMock(return_value={})

        trainer.fit()

    def test_fit_failed_with_load_checkpoint(self, agent_grpo_trainer):
        trainer, _ = agent_grpo_trainer

        trainer.train_dataloader = MockDataLoader(num_batches=1)
        trainer._load_checkpoint = MagicMock(side_effect=RuntimeError("error"))

        with pytest.raises(RuntimeError):
            trainer.fit()

    def test_fit_failed_with_train_step(self, agent_grpo_trainer):
        trainer, _ = agent_grpo_trainer

        trainer.train_dataloader = MockDataLoader(num_batches=1)
        trainer._load_checkpoint = MagicMock()
        trainer._save_checkpoint = MagicMock()
        trainer._validate_agent = MagicMock(return_value={})
        trainer._prepare_batch = MagicMock(return_value=MockDataProto())
        trainer._train_step = MagicMock(side_effect=Exception("error"))

        with pytest.raises(Exception):
            trainer.fit()

    def test_shutdown_success(self, agent_grpo_trainer):
        trainer, _ = agent_grpo_trainer
        trainer.shutdown()
        assert True
