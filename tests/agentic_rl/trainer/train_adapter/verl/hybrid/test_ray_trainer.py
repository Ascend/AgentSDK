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
import sys
from unittest.mock import MagicMock, patch

import pytest
import torch


class MockRayPPOTrainer:
    def __init__(self, *args, **kwargs):
        pass


# 测试用例类
class TestHybridTrainer:
    """测试HybridTrainer类"""

    @pytest.fixture(scope="class")
    def patch_modules(self):
        with patch.dict(sys.modules, {
            "verl": MagicMock(),
            "verl.utils": MagicMock(),
            "verl.utils.tracking": MagicMock(),
            "verl.utils.checkpoint": MagicMock(),
            "verl.utils.checkpoint.checkpoint_manager": MagicMock(),
            "verl.utils.debug": MagicMock(),
            "verl.utils.metric": MagicMock(),
            "verl.utils.rollout_skip": MagicMock(),
            "verl.trainer": MagicMock(),
            "verl.trainer.ppo": MagicMock(),
            "verl.trainer.ppo.reward": MagicMock(),
            "verl.trainer.ppo.utils": MagicMock(),
            "verl.trainer.ppo.core_algos": MagicMock(),
            "verl.trainer.ppo.metric_utils": MagicMock(),
            "verl.trainer.ppo.ray_trainer": MagicMock(),
            "verl.trainer.ppo.rollout_corr_helper": MagicMock(),
            "verl.utils.hf_tokenizer": MagicMock(),
            "verl.experimental": MagicMock(),
            "verl.experimental.dataset": MagicMock(),
            "verl.experimental.dataset.sampler": MagicMock(),
            "verl.DataProto": MagicMock(),
            "verl.single_controller": MagicMock(),
            "verl.single_controller.ray": MagicMock(),

            "ray": MagicMock(),
        }):
            # 在fixture内部导入，确保patch生效
            yield

    @pytest.fixture(scope="function")
    def mock_config(self):
        """创建模拟的配置对象"""
        config = MagicMock()

        # 配置trainer相关参数
        config.trainer = MagicMock()
        config.trainer.project_name = "test_project"
        config.trainer.experiment_name = "test_experiment"
        config.trainer.logger = "test_logger"
        config.trainer.total_epochs = 1
        config.trainer.save_freq = 1000
        config.trainer.test_freq = 100
        config.trainer.val_before_train = False
        config.trainer.val_only = False
        config.trainer.balance_batch = False
        config.trainer.critic_warmup = 0
        config.trainer.esi_redundant_time = 3600
        config.trainer.rollout_data_dir = None
        config.trainer.get.side_effect = lambda k, default=None: getattr(config.trainer, k, default)

        # 配置actor_rollout_ref相关参数
        config.actor_rollout_ref = MagicMock()
        config.actor_rollout_ref.rollout = MagicMock()
        config.actor_rollout_ref.rollout.n = 1
        config.actor_rollout_ref.rollout.temperature = 0.7
        config.actor_rollout_ref.rollout.skip_rollout = False
        config.actor_rollout_ref.actor = MagicMock()
        config.actor_rollout_ref.actor.policy_loss = MagicMock()
        config.actor_rollout_ref.actor.loss_agg_mode = "mean"
        config.actor_rollout_ref.actor.loss_scale_factor = 1.0

        # 配置algorithm相关参数
        config.algorithm = MagicMock()
        config.algorithm.adv_estimator = "GAE"
        config.algorithm.gamma = 0.99
        config.algorithm.lam = 0.95
        config.algorithm.use_kl_in_reward = False
        config.algorithm.kl_penalty = 0.0
        config.algorithm.norm_adv_by_std_in_grpo = True
        config.algorithm.rollout_correction = None
        config.algorithm.get.side_effect = lambda k, default=None: getattr(config.algorithm, k, default)

        # 配置reward_model相关参数
        config.reward_model = MagicMock()
        config.reward_model.launch_reward_fn_async = False

        # 配置global_profiler相关参数
        config.global_profiler = MagicMock()
        config.global_profiler.steps = None
        config.global_profiler.profile_continuous_steps = False

        return config

    @pytest.fixture(scope="function")
    def mock_data_proto(self):
        """创建模拟的DataProto对象"""
        data_proto = MagicMock()
        data_proto.batch = {

        }
        data_proto.meta_info = {
            "timing": {
                "step": 123456
            },
            "metrics": {}
        }
        data_proto.non_tensor_batch = {}

        # 模拟方法
        data_proto.from_single_dict.return_value = data_proto
        data_proto.repeat.return_value = data_proto
        data_proto.union.return_value = data_proto
        data_proto.pop.return_value = data_proto

        return data_proto

    @pytest.fixture(scope="function")
    def mock_trainer(self, mock_config, patch_modules):
        """创建模拟的HybridTrainer对象"""
        with patch("verl.trainer.ppo.ray_trainer.RayPPOTrainer", MockRayPPOTrainer):
            from agentic_rl.trainer.train_adapter.verl.hybrid.ray_trainer import HybridTrainer

            trainer = HybridTrainer()
            trainer.config = mock_config
            trainer.total_training_steps = 10
            trainer.global_steps = 0
            trainer.train_dataloader = MagicMock()
            trainer.val_reward_fn = None
            trainer.use_rm = False
            trainer.use_reward_loop = False
            trainer.use_critic = True
            trainer.use_reference_policy = False
            trainer.async_rollout_mode = False
            trainer.kl_ctrl_in_reward = MagicMock()
            trainer.resource_pool_manager = MagicMock()
            trainer.resource_pool_manager.get_n_gpus.return_value = 1
            trainer.tokenizer = MagicMock()
            trainer.train_dataset = MagicMock()

            # 模拟组件
            trainer.actor_rollout_wg = MagicMock()
            trainer.async_rollout_manager = MagicMock()
            trainer.rm_wg = MagicMock()
            trainer.reward_loop_manager = MagicMock()
            trainer.reward_fn = MagicMock()

            yield trainer

    def test_fit_basic_flow(self, mock_trainer, mock_data_proto):
        """测试fit方法的基本流程"""
        with patch("agentic_rl.trainer.train_adapter.verl.hybrid.ray_trainer.DataProto") as mock_data_proto_class, \
                patch("agentic_rl.trainer.train_adapter.verl.hybrid.ray_trainer.uuid.uuid4") as mock_uuid, \
                patch("omegaconf.OmegaConf") as mock_omega_conf, \
                patch("agentic_rl.trainer.train_adapter.verl.hybrid.ray_trainer."
                      "AbstractCurriculumSampler", MagicMock), \
                patch("agentic_rl.trainer.train_adapter.verl.hybrid.ray_trainer.apply_kl_penalty") as mock_apply_kl:
            # 设置mock
            mock_trainer._load_checkpoint = MagicMock()
            mock_trainer._get_gen_batch = MagicMock(return_value=mock_data_proto)
            mock_trainer._start_profiling = MagicMock()
            mock_trainer._stop_profiling = MagicMock()
            mock_trainer._save_checkpoint = MagicMock()
            mock_trainer._compute_or_extract_reward = MagicMock(return_value=(torch.tensor([0.1, 0.2]), {}))
            mock_trainer._compute_old_log_prob = MagicMock(return_value=(mock_data_proto, 0.5))
            mock_trainer._compute_values = MagicMock(return_value=mock_data_proto)
            mock_trainer._update_critic = MagicMock(return_value=mock_data_proto)
            mock_trainer._update_actor = MagicMock(return_value=mock_data_proto)
            mock_trainer._log_rollout_data = MagicMock()
            mock_trainer.actor_rollout_wg.generate_sequences.return_value = mock_data_proto
            mock_trainer.async_rollout_manager.generate_sequences.return_value = mock_data_proto

            mock_omega_conf.to_container.return_value = mock_trainer.config

            mock_trainer.use_rm = True
            mock_trainer.config.algorithm.use_kl_in_reward = True
            mock_apply_kl.return_value = mock_data_proto, {}

            # 模拟训练数据加载器
            mock_batch_dict = {"input_ids": [1, 2, 3], "attention_mask": [1, 1, 1], "labels": [1, 2, 3]}
            mock_trainer.train_dataloader.__iter__.return_value = [mock_batch_dict]
            mock_trainer.train_dataloader.__len__.return_value = 1

            # 模拟DataProto
            mock_data_proto_class.from_single_dict.return_value = mock_data_proto

            # 设置batch的响应掩码
            mock_data_proto.batch = {"input_ids": torch.tensor([[1, 2, 3], [4, 5, 6]]),
                                     "attention_mask": torch.tensor([[1, 1, 1], [1, 1, 1]]),
                                     "labels": torch.tensor([[1, 2, 3], [4, 5, 6]]),
                                     "old_log_probs": torch.tensor([[1, 2, 3], [4, 5, 6]]),
                                     "entropys": torch.tensor([[1, 2, 3], [4, 5, 6]])}

            # 运行fit方法
            mock_trainer.fit()

            # 验证关键方法被调用
            mock_trainer._load_checkpoint.assert_called_once()
            mock_trainer._get_gen_batch.assert_called_once()
            mock_trainer._compute_or_extract_reward.assert_called()
            mock_trainer._compute_old_log_prob.assert_called()
            mock_trainer._compute_values.assert_called()
            mock_trainer._update_critic.assert_called()
            mock_trainer._update_actor.assert_called()

    def test_fit_validate_only(self, mock_trainer, mock_data_proto):
        """测试fit方法只进行验证"""
        # 设置mock
        with patch("omegaconf.OmegaConf") as mock_omega_conf:
            mock_trainer._load_checkpoint = MagicMock()

            mock_omega_conf.to_container.return_value = mock_trainer.config

            # 模拟训练数据加载器
            mock_batch_dict = {"input_ids": [1, 2, 3], "attention_mask": [1, 1, 1], "labels": [1, 2, 3]}
            mock_trainer.train_dataloader.__iter__.return_value = [mock_batch_dict]
            mock_trainer.train_dataloader.__len__.return_value = 1

            # 模拟验证设置
            mock_trainer._validate = MagicMock()
            mock_trainer._validate.return_value = 1  # not None
            mock_trainer.config.trainer.val_before_train = True
            mock_trainer.config.trainer.val_only = True
            mock_trainer.val_reward_fn = MagicMock()

            # 运行fit方法
            mock_trainer.fit()

            # 验证关键方法被调用
            mock_trainer._validate.assert_called_once()

    def test_fit_async_rollout_mode(self, mock_trainer, mock_data_proto):
        """测试fit方法的异步rollout模式"""
        with patch("agentic_rl.trainer.train_adapter.verl.hybrid.ray_trainer.DataProto") as mock_data_proto_class, \
                patch("agentic_rl.trainer.train_adapter.verl.hybrid.ray_trainer.uuid.uuid4") as mock_uuid, \
                patch("omegaconf.OmegaConf") as mock_omega_conf, \
                patch("agentic_rl.trainer.train_adapter.verl.hybrid.ray_trainer.AbstractCurriculumSampler", MagicMock):
            # 设置异步模式
            mock_trainer.async_rollout_mode = True

            # 设置mock
            mock_trainer._load_checkpoint = MagicMock()
            mock_trainer._get_gen_batch = MagicMock(return_value=mock_data_proto)
            mock_trainer._start_profiling = MagicMock()
            mock_trainer._stop_profiling = MagicMock()
            mock_trainer._save_checkpoint = MagicMock()
            mock_trainer._compute_or_extract_reward = MagicMock(return_value=(torch.tensor([0.1, 0.2]), {}))
            mock_trainer._compute_old_log_prob = MagicMock(return_value=(mock_data_proto, 0.5))
            mock_trainer._compute_values = MagicMock(return_value=mock_data_proto)
            mock_trainer._update_critic = MagicMock(return_value=mock_data_proto)
            mock_trainer._update_actor = MagicMock(return_value=mock_data_proto)
            mock_trainer._log_rollout_data = MagicMock()
            mock_trainer.actor_rollout_wg.generate_sequences.return_value = mock_data_proto
            mock_trainer.async_rollout_manager.generate_sequences.return_value = mock_data_proto

            mock_omega_conf.to_container.return_value = mock_trainer.config

            # 模拟训练数据加载器
            mock_batch_dict = {"input_ids": [1, 2, 3], "attention_mask": [1, 1, 1], "labels": [1, 2, 3]}
            mock_trainer.train_dataloader.__iter__.return_value = [mock_batch_dict]
            mock_trainer.train_dataloader.__len__.return_value = 1

            # 模拟DataProto
            mock_data_proto_class.from_single_dict.return_value = mock_data_proto

            # 设置batch的响应掩码
            mock_data_proto.batch = {"response_mask": torch.tensor([[1, 1, 0], [1, 0, 0]]),
                                     "input_ids": torch.tensor([[1, 2, 3], [4, 5, 6]]),
                                     "attention_mask": torch.tensor([[1, 1, 1], [1, 1, 1]]),
                                     "labels": torch.tensor([[1, 2, 3], [4, 5, 6]]),
                                     "old_log_probs": torch.tensor([[1, 2, 3], [4, 5, 6]]),
                                     "entropys": torch.tensor([[1, 2, 3], [4, 5, 6]])}

            # 运行fit方法
            mock_trainer.fit()

            # 验证异步rollout manager被调用
            mock_trainer.async_rollout_manager.generate_sequences.assert_called()

    def test_fit_async_reward_mode(self, mock_trainer, mock_data_proto):
        """测试fit方法的异步奖励计算模式"""
        with patch("agentic_rl.trainer.train_adapter.verl.hybrid.ray_trainer.DataProto") as mock_data_proto_class, \
                patch("agentic_rl.trainer.train_adapter.verl.hybrid.ray_trainer.uuid.uuid4") as mock_uuid, \
                patch("omegaconf.OmegaConf") as mock_omega_conf, \
                patch("agentic_rl.trainer.train_adapter.verl.hybrid.ray_trainer."
                      "AbstractCurriculumSampler", MagicMock), \
                patch("agentic_rl.trainer.train_adapter.verl.hybrid.ray_trainer."
                      "compute_reward_async") as mock_compute_reward_async, \
                patch("ray.get") as mock_ray_get:
            # 设置异步奖励模式
            mock_trainer.config.reward_model.launch_reward_fn_async = True

            # 设置mock
            mock_trainer._load_checkpoint = MagicMock()
            mock_trainer._get_gen_batch = MagicMock(return_value=mock_data_proto)
            mock_trainer._start_profiling = MagicMock()
            mock_trainer._stop_profiling = MagicMock()
            mock_trainer._save_checkpoint = MagicMock()
            mock_trainer._compute_old_log_prob = MagicMock(return_value=(mock_data_proto, 0.5))
            mock_trainer._compute_values = MagicMock(return_value=mock_data_proto)
            mock_trainer._update_critic = MagicMock(return_value=mock_data_proto)
            mock_trainer._update_actor = MagicMock(return_value=mock_data_proto)
            mock_trainer._log_rollout_data = MagicMock()
            mock_trainer.actor_rollout_wg.generate_sequences.return_value = mock_data_proto
            mock_trainer.async_rollout_manager.generate_sequences.return_value = mock_data_proto

            # 模拟异步奖励计算
            mock_future = MagicMock()
            mock_compute_reward_async.remote.return_value = mock_future
            mock_ray_get.return_value = (torch.tensor([0.1, 0.2]), {})

            # 模拟训练数据加载器
            mock_batch_dict = {"input_ids": [1, 2, 3], "attention_mask": [1, 1, 1], "labels": [1, 2, 3]}
            mock_trainer.train_dataloader.__iter__.return_value = [mock_batch_dict]
            mock_trainer.train_dataloader.__len__.return_value = 1

            # 模拟DataProto
            mock_data_proto_class.from_single_dict.return_value = mock_data_proto

            # 设置batch的响应掩码
            mock_data_proto.batch = {"response_mask": torch.tensor([[1, 1, 0], [1, 0, 0]]),
                                     "input_ids": torch.tensor([[1, 2, 3], [4, 5, 6]]),
                                     "attention_mask": torch.tensor([[1, 1, 1], [1, 1, 1]]),
                                     "labels": torch.tensor([[1, 2, 3], [4, 5, 6]]),
                                     "old_log_probs": torch.tensor([[1, 2, 3], [4, 5, 6]]),
                                     "entropys": torch.tensor([[1, 2, 3], [4, 5, 6]])}

            # 运行fit方法
            mock_trainer.fit()

            # 验证异步奖励计算被调用
            mock_compute_reward_async.remote.assert_called()
            mock_ray_get.assert_called_with(mock_future)

    def test_fit_bypass_recomputing_logprobs(self, mock_trainer, mock_data_proto):
        """测试fit方法的bypass recomputing logprobs模式"""
        with patch("agentic_rl.trainer.train_adapter.verl.hybrid.ray_trainer.DataProto") as mock_data_proto_class, \
                patch("agentic_rl.trainer.train_adapter.verl.hybrid.ray_trainer.uuid.uuid4") as mock_uuid, \
                patch("omegaconf.OmegaConf") as mock_omega_conf, \
                patch("agentic_rl.trainer.train_adapter.verl.hybrid.ray_trainer."
                      "AbstractCurriculumSampler", MagicMock), \
                patch('verl.trainer.ppo.rollout_corr_helper.apply_bypass_mode'):
            # 设置bypass模式
            mock_rollout_corr_config = MagicMock()
            mock_rollout_corr_config.get.return_value = True
            mock_trainer.config.algorithm.rollout_correction = mock_rollout_corr_config

            # 设置mock
            mock_trainer._load_checkpoint = MagicMock()
            mock_trainer._get_gen_batch = MagicMock(return_value=mock_data_proto)
            mock_trainer._start_profiling = MagicMock()
            mock_trainer._stop_profiling = MagicMock()
            mock_trainer._save_checkpoint = MagicMock()
            mock_trainer._compute_or_extract_reward = MagicMock(return_value=(torch.tensor([0.1, 0.2]), {}))
            mock_trainer._compute_old_log_prob = MagicMock(return_value=(mock_data_proto, 0.5))
            mock_trainer._compute_values = MagicMock(return_value=mock_data_proto)
            mock_trainer._update_critic = MagicMock(return_value=mock_data_proto)
            mock_trainer._update_actor = MagicMock(return_value=mock_data_proto)
            mock_trainer._log_rollout_data = MagicMock()
            mock_trainer.actor_rollout_wg.generate_sequences.return_value = mock_data_proto
            mock_trainer.async_rollout_manager.generate_sequences.return_value = mock_data_proto

            # 模拟训练数据加载器
            mock_batch_dict = {"input_ids": [1, 2, 3], "attention_mask": [1, 1, 1], "labels": [1, 2, 3]}
            mock_trainer.train_dataloader.__iter__.return_value = [mock_batch_dict]
            mock_trainer.train_dataloader.__len__.return_value = 1

            # 模拟DataProto
            mock_data_proto_class.from_single_dict.return_value = mock_data_proto

            # 设置batch的响应掩码和old_log_probs
            mock_data_proto.batch = {"response_mask": torch.tensor([[1, 1, 0], [1, 0, 0]]),
                                     "input_ids": torch.tensor([[1, 2, 3], [4, 5, 6]]),
                                     "attention_mask": torch.tensor([[1, 1, 1], [1, 1, 1]]),
                                     "labels": torch.tensor([[1, 2, 3], [4, 5, 6]]),
                                     "old_log_probs": torch.tensor([[1, 2, 3], [4, 5, 6]]),
                                     "entropys": torch.tensor([[1, 2, 3], [4, 5, 6]])}

            # 运行fit方法
            mock_trainer.fit()

            # 验证_compute_old_log_prob没有被调用（因为使用了bypass模式）
            mock_trainer._compute_old_log_prob.assert_not_called()

    def test_fit_with_estimator_remax(self, mock_trainer, mock_data_proto):
        """测试fit方法的使用REMAX计算优势值"""
        with patch("agentic_rl.trainer.train_adapter.verl.hybrid.ray_trainer.DataProto") as mock_data_proto_class, \
                patch("agentic_rl.trainer.train_adapter.verl.hybrid.ray_trainer.uuid.uuid4") as mock_uuid, \
                patch("omegaconf.OmegaConf") as mock_omega_conf, \
                patch("agentic_rl.trainer.train_adapter.verl.hybrid.ray_trainer."
                      "AbstractCurriculumSampler", MagicMock), \
                patch("agentic_rl.trainer.train_adapter.verl.hybrid.ray_trainer."
                      "AdvantageEstimator") as mock_estimator_type:
            mock_estimator_type.REMAX = "REMAX"
            mock_trainer.config.algorithm.adv_estimator = "REMAX"

            # 设置mock
            mock_trainer._load_checkpoint = MagicMock()
            mock_trainer._get_gen_batch = MagicMock(return_value=mock_data_proto)
            mock_trainer._start_profiling = MagicMock()
            mock_trainer._stop_profiling = MagicMock()
            mock_trainer._save_checkpoint = MagicMock()
            mock_trainer._compute_or_extract_reward = MagicMock(return_value=(torch.tensor([0.1, 0.2]), {}))
            mock_trainer._compute_old_log_prob = MagicMock(return_value=(mock_data_proto, 0.5))
            mock_trainer._compute_values = MagicMock(return_value=mock_data_proto)
            mock_trainer._update_critic = MagicMock(return_value=mock_data_proto)
            mock_trainer._update_actor = MagicMock(return_value=mock_data_proto)
            mock_trainer._log_rollout_data = MagicMock()
            mock_trainer.actor_rollout_wg.generate_sequences.return_value = mock_data_proto
            mock_trainer.async_rollout_manager.generate_sequences.return_value = mock_data_proto

            mock_omega_conf.to_container.return_value = mock_trainer.config

            # 模拟训练数据加载器
            mock_batch_dict = {"input_ids": [1, 2, 3], "attention_mask": [1, 1, 1], "labels": [1, 2, 3]}
            mock_trainer.train_dataloader.__iter__.return_value = [mock_batch_dict]
            mock_trainer.train_dataloader.__len__.return_value = 1

            # 模拟DataProto
            mock_data_proto_class.from_single_dict.return_value = mock_data_proto

            # 设置batch的响应掩码
            mock_data_proto.batch = {"input_ids": torch.tensor([[1, 2, 3], [4, 5, 6]]),
                                     "attention_mask": torch.tensor([[1, 1, 1], [1, 1, 1]]),
                                     "labels": torch.tensor([[1, 2, 3], [4, 5, 6]]),
                                     "old_log_probs": torch.tensor([[1, 2, 3], [4, 5, 6]]),
                                     "entropys": torch.tensor([[1, 2, 3], [4, 5, 6]])}

            # 模拟最后一个step
            mock_trainer.total_training_steps = 1

            # 运行fit方法
            mock_trainer.fit()
