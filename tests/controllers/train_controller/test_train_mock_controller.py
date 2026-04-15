#!/usr/bin/env python3
# -*- coding: utf-8 -*-

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

import sys
import unittest
from unittest.mock import MagicMock, patch

# ---------------------------------------------------------------------------
# Module-level mocks
# ---------------------------------------------------------------------------
mock_random = MagicMock()
mock_torch = MagicMock()

# We must mock the parent class TrainController so that super().__init__ is harmless.
mock_train_controller_module = MagicMock()
mock_logger = MagicMock()


class MockTrainController:
    """Stand-in for the real TrainController base class."""
    def __init__(self, *args, **kwargs):
        # The real TrainMockController passes 7 positional args to super
        self.global_batch_size = 4
        self.n_samples_per_prompt = 2


mock_train_controller_module.TrainController = MockTrainController
mock_train_controller_module.logger = mock_logger

mock_loggers_module = MagicMock()
mock_loggers_module.Loggers.return_value.get_logger.return_value = mock_logger

with patch.dict('sys.modules', {
    'random': mock_random,
    'torch': mock_torch,
    'torch.distributed': mock_torch.distributed,
    'ray': MagicMock(),
    'requests': MagicMock(),
    'fastapi': MagicMock(),
    'agentic_rl.base.log.loggers': mock_loggers_module,
    'agentic_rl.controllers.train_controller.train_controller': mock_train_controller_module,
    'agentic_rl.controllers.utils.controller_config': MagicMock(),
    'agentic_rl.controllers.utils.http_status': MagicMock(),
    'agentic_rl.controllers.utils.utils': MagicMock(),
}):
    from agentic_rl.controllers.train_controller.train_mock_controller import TrainMockController


class TestTrainMockController(unittest.TestCase):
    """Tests for TrainMockController -- a mock variant of TrainController."""

    def setUp(self):
        mock_logger.reset_mock()
        mock_torch.reset_mock()
        mock_random.reset_mock()

        self.mock_rl_config = MagicMock()
        self.mock_rl_config.mock_prompt_mean = 50
        self.mock_rl_config.mock_prompt_gap = 20
        self.mock_rl_config.mock_response_mean = 100
        self.mock_rl_config.mock_response_gap = 40
        self.mock_rl_config.mock_eos_token_id = 2

        self.controller = TrainMockController(
            actor_worker=MagicMock(),
            actor_config=MagicMock(),
            rl_config=self.mock_rl_config,
            generate_config=MagicMock(),
            initialize_rollout_dataloader=MagicMock(),
            consumed_train_samples=0,
            data_optimized=False,
        )
        # Ensure the inherited attributes are set
        self.controller.global_batch_size = 4
        self.controller.n_samples_per_prompt = 2

    def tearDown(self):
        mock_torch.reset_mock()
        mock_random.reset_mock()
        mock_logger.reset_mock()

    # ---- __init__ -----------------------------------------------------------

    def test_init_sets_rl_config(self):
        self.assertIs(self.controller.rl_config, self.mock_rl_config)

    def test_init_inherits_global_batch_size(self):
        self.assertEqual(self.controller.global_batch_size, 4)

    # ---- wait_for_rollout_unit_ready (no-op) --------------------------------

    def test_wait_for_rollout_unit_ready_is_noop(self):
        self.controller.wait_for_rollout_unit_ready()
        mock_logger.info.assert_called()

    # ---- update_rollout_weights (no-op) -------------------------------------

    def test_update_rollout_weights_is_noop(self):
        self.controller.update_rollout_weights(1)
        mock_logger.info.assert_called()

    # ---- send_initial_batch_groups_to_rollout (no-op) -----------------------

    def test_send_initial_batch_groups_is_noop(self):
        self.controller.send_initial_batch_groups_to_rollout()
        mock_logger.info.assert_called()

    # ---- unlock_rollout_unit (no-op) ----------------------------------------

    def test_unlock_rollout_unit_is_noop(self):
        self.controller.unlock_rollout_unit()
        mock_logger.info.assert_called()

    # ---- get_next_training_batch (mock data generation) ---------------------

    def test_get_next_training_batch_returns_outputs_and_metric(self):
        # Set up mock_random.random to return a deterministic value
        mock_random.random.return_value = 0.5

        # Set up mock_torch to return MagicMock tensors
        mock_tensor = MagicMock()
        mock_torch.randint.return_value = mock_tensor
        mock_torch.tensor.return_value = mock_tensor
        mock_torch.cat.return_value = mock_tensor
        mock_torch.nn.utils.rnn.pad_sequence.return_value = mock_tensor
        mock_torch.zeros_like.return_value = mock_tensor
        mock_torch.ones_like.return_value = mock_tensor
        mock_torch.float32 = "float32"

        outputs, metric = self.controller.get_next_training_batch()

        self.assertIsInstance(outputs, dict)
        self.assertIn('responses', outputs)
        self.assertIn('input_ids', outputs)
        self.assertIn('response_length', outputs)
        self.assertIn('prompt_length', outputs)
        self.assertIn('rm_scores', outputs)
        self.assertIn('token_level_rewards', outputs)
        self.assertIn('response_mask', outputs)

    def test_get_next_training_batch_metric_keys(self):
        mock_random.random.return_value = 0.5
        mock_tensor = MagicMock()
        mock_torch.randint.return_value = mock_tensor
        mock_torch.tensor.return_value = mock_tensor
        mock_torch.cat.return_value = mock_tensor
        mock_torch.nn.utils.rnn.pad_sequence.return_value = mock_tensor
        mock_torch.zeros_like.return_value = mock_tensor
        mock_torch.ones_like.return_value = mock_tensor
        mock_torch.float32 = "float32"

        _, metric = self.controller.get_next_training_batch()

        self.assertIn('rollout_cost', metric)
        self.assertIn('resharding_to_infer', metric)
        self.assertEqual(metric['rollout_cost'], 1)
        self.assertEqual(metric['resharding_to_infer'], 1)

    def test_get_next_training_batch_calls_torch_operations(self):
        mock_random.random.return_value = 0.5
        mock_tensor = MagicMock()
        mock_torch.randint.return_value = mock_tensor
        mock_torch.tensor.return_value = mock_tensor
        mock_torch.cat.return_value = mock_tensor
        mock_torch.nn.utils.rnn.pad_sequence.return_value = mock_tensor
        mock_torch.zeros_like.return_value = mock_tensor
        mock_torch.ones_like.return_value = mock_tensor
        mock_torch.float32 = "float32"

        self.controller.get_next_training_batch()

        # Should call randint for prompt_ids (one per global_batch_size=4)
        self.assertTrue(mock_torch.randint.called)
        # Should call pad_sequence for response_batch
        mock_torch.nn.utils.rnn.pad_sequence.assert_called_once()
        # Should call zeros_like and ones_like
        mock_torch.zeros_like.assert_called_once()
        mock_torch.ones_like.assert_called_once()


if __name__ == '__main__':
    unittest.main()
