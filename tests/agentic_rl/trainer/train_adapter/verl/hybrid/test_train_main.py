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
import pytest
import importlib
from unittest.mock import MagicMock, patch


class TestHybridTrainMain:
    """
    Tests for verl hybrid train_main module.
    Covers:
      - HybridTaskRunner.run
      - start_train function
    """

    def setup_method(self):
        """
        Patch sys.modules and set up mocks before each test.
        """
        self.original_modules = sys.modules.copy()

        # Basic external modules mocks
        mock_ray = MagicMock()
        mock_ray.remote = lambda cls: cls

        mock_omega = MagicMock()
        mock_omega.to_container.return_value = {}
        mock_omega.resolve = MagicMock()

        mock_socket = MagicMock()
        mock_socket.gethostname.return_value = "test-host"

        mock_os = MagicMock()
        mock_os.getpid.return_value = 12345

        mock_fs = MagicMock()
        mock_fs.copy_to_local.return_value = "/local/path"

        mock_dataset = MagicMock()
        mock_dataset.collate_fn = MagicMock()

        mock_hf_processor = MagicMock()
        mock_hf_tokenizer = MagicMock()

        mock_config = MagicMock()
        mock_config.validate_config = MagicMock()

        # Utils mocks
        mock_utils = MagicMock()
        mock_utils.fs = mock_fs
        mock_utils.dataset = MagicMock(rl_dataset=mock_dataset)
        mock_utils.hf_processor = mock_hf_processor
        mock_utils.hf_tokenizer = mock_hf_tokenizer
        mock_utils.config = mock_config

        # PPO reward / utils mocks
        mock_reward = MagicMock()
        mock_reward.load_reward_manager = MagicMock()

        mock_ppo_utils = MagicMock()
        mock_ppo_utils.need_critic.return_value = True
        mock_ppo_utils.need_reference_policy.return_value = True

        # FakeTaskRunner for mocking TaskRunner behavior
        class FakeTaskRunner:
            add_actor_rollout_worker = MagicMock(return_value=(MagicMock(), MagicMock()))
            add_critic_worker = MagicMock()
            add_reward_model_worker = MagicMock()
            add_ref_policy_worker = MagicMock()
            init_resource_pool_mgr = MagicMock()
            create_rl_dataset = MagicMock()
            create_rl_sampler = MagicMock()

        # Mocked PPO main functions
        def fake_run_ppo(*args, **kwargs):
            return None

        mock_main_ppo = MagicMock()
        mock_main_ppo.TaskRunner = FakeTaskRunner
        mock_main_ppo.create_rl_dataset = FakeTaskRunner.create_rl_dataset
        mock_main_ppo.create_rl_sampler = FakeTaskRunner.create_rl_sampler
        mock_main_ppo.run_ppo = fake_run_ppo

        # Trainer mock
        mock_trainer = MagicMock()
        mock_trainer.main_ppo = mock_main_ppo
        mock_trainer.ppo = MagicMock(reward=mock_reward, utils=mock_ppo_utils)

        # Verl module mock
        mock_verl = MagicMock()
        mock_verl.trainer = mock_trainer
        mock_verl.utils = mock_utils

        # HybridTrainer mock
        mock_hybrid_trainer = MagicMock()
        mock_hybrid_trainer.init_workers = MagicMock()
        mock_hybrid_trainer.fit = MagicMock()

        # Patch sys.modules
        self.mock_modules = {
            "ray": mock_ray,
            "omegaconf": MagicMock(OmegaConf=mock_omega),
            "socket": mock_socket,
            "os": mock_os,
            "verl": mock_verl,
            "verl.trainer": mock_trainer,
            "verl.trainer.main_ppo": mock_main_ppo,
            "verl.trainer.ppo": mock_trainer.ppo,
            "verl.trainer.ppo.reward": mock_reward,
            "verl.trainer.ppo.utils": mock_ppo_utils,
            "verl.utils": mock_utils,
            "verl.utils.fs": mock_fs,
            "verl.utils.dataset": mock_utils.dataset,
            "verl.utils.dataset.rl_dataset": mock_dataset,
            "verl.utils.config": mock_config,
            "verl.utils.hf_processor": mock_hf_processor,
            "verl.utils.hf_tokenizer": mock_hf_tokenizer,
            "agentic_rl.trainer.train_adapter.verl.hybrid.ray_trainer": MagicMock(
                HybridTrainer=MagicMock(return_value=mock_hybrid_trainer)
            ),
        }

        self.module_patcher = patch.dict(sys.modules, self.mock_modules)
        self.module_patcher.start()

        # Import the module under test inside patched context
        import agentic_rl.trainer.train_adapter.verl.hybrid.train_main as tm
        importlib.reload(tm)

        self.HybridTaskRunner = tm.HybridTaskRunner
        self.start_train = tm.start_train

    def teardown_method(self):
        """
        Stop all patches and restore original sys.modules.
        """
        patch.stopall()
        sys.modules.clear()
        sys.modules.update(self.original_modules)

    def test_run_full(self):
        """
        Test running HybridTaskRunner.run with a sample config.
        """
        from omegaconf import OmegaConf

        config_dict = {
            "actor_rollout_ref": {
                "model": {"path": "hdfs://test/path", "use_shm": True}
            },
            "data": {
                "train_files": ["t1"],
                "val_files": ["v1"],
                "trust_remote_code": True,
            },
            "reward_model": {"reward_kwargs": {"p": 1}},
        }

        config = OmegaConf.create(config_dict)
        runner = self.HybridTaskRunner()
        runner.role_worker_mapping = {"actor": MagicMock()}

        runner.run(config)

    def test_start_train(self):
        """
        Test start_train function executes without errors.
        """
        self.start_train("local", {"key": "value"})