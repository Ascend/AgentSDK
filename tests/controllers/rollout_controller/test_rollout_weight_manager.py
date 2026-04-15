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
import os
import unittest
from unittest.mock import MagicMock, patch, call

# ---------------------------------------------------------------------------
# Module-level mocks
# ---------------------------------------------------------------------------
mock_ray = MagicMock()
mock_ray.remote = MagicMock(side_effect=lambda cls: cls)

mock_transformers = MagicMock()
mock_auto_config = MagicMock()
mock_transformers.AutoConfig = mock_auto_config

mock_loggers_module = MagicMock()
mock_logger = MagicMock()
mock_loggers_module.Loggers.return_value.get_logger.return_value = mock_logger

mock_globals_module = MagicMock()
mock_globals_module.ROLLOUT_WEIGHTS_PREFIX = "/rollout"

with patch.dict('sys.modules', {
    'ray': mock_ray,
    'torch': MagicMock(),
    'torch.distributed': MagicMock(),
    'transformers': mock_transformers,
    'agentic_rl.base.log.loggers': mock_loggers_module,
    'agentic_rl.base.utils.globals': mock_globals_module,
}):
    from agentic_rl.controllers.rollout_controller.rollout_weight_manager import (
        RolloutWeightManager, MAX_RETAIN_WEIGHTS_VERSION, PATH_ITER_PATTERN
    )
    import agentic_rl.controllers.rollout_controller.rollout_weight_manager as _rwm_mod


class TestRolloutWeightManager(unittest.TestCase):
    """Tests for RolloutWeightManager covering init, version control, weight updates."""

    def _make_manager(self, **overrides):
        defaults = dict(
            weight_save_dir="/tmp/weights",
            tokenizer_name_or_path="/models/tokenizer",
            trust_remote_code=False,
            infer_tensor_parallel_size=4,
            train_tensor_parallel_size=4,
            infer_expert_parallel_size=1,
            enable_version_control=False,
            use_on_policy=True,
            model_name="test_model",
        )
        defaults.update(overrides)
        with patch('os.makedirs'), \
             patch('os.getenv', return_value='false'), \
             patch.object(_rwm_mod, 'AutoConfig', mock_auto_config):
            return RolloutWeightManager(**defaults)

    def setUp(self):
        mock_auto_config.from_pretrained.return_value = MagicMock()
        mock_logger.reset_mock()
        mock_ray.reset_mock()
        self.mgr = self._make_manager()

    def tearDown(self):
        mock_ray.reset_mock()
        mock_logger.reset_mock()
        mock_auto_config.reset_mock()

    # ---- __init__ -----------------------------------------------------------

    def test_init_sets_inference_save_path(self):
        self.assertEqual(self.mgr.inference_save_path, "/tmp/weights/rollout")

    def test_init_weights_version_zero(self):
        self.assertEqual(self.mgr.weights_version, 0)

    def test_init_model_name(self):
        self.assertEqual(self.mgr.model_name, "test_model")

    def test_init_model_path(self):
        self.assertEqual(self.mgr.model_path, "/models/tokenizer")

    def test_init_calls_auto_config(self):
        mock_auto_config.from_pretrained.assert_called_with(
            "/models/tokenizer",
            trust_remote_code=False,
        )

    def test_init_on_policy_true(self):
        self.assertTrue(self.mgr.use_on_policy)

    def test_init_one_step_off_ep_mode_false(self):
        self.assertFalse(self.mgr.one_step_off_ep_mode)

    def test_init_infer_tp_non_ep_mode(self):
        # When one_step_off_ep_mode is False and train_tp == infer_tp == 4:
        # infer_tp = int(4 * (4/4)) = 4
        self.assertEqual(self.mgr.infer_tp, 4)

    def test_init_head_dim_scale_non_ep_mode(self):
        # head_dim_scale = infer_tp // train_tp = 4 // 4 = 1
        self.assertEqual(self.mgr.head_dim_scale, 1)

    # ---- get_weights_version ------------------------------------------------

    def test_get_weights_version_initial(self):
        self.assertEqual(self.mgr.get_weights_version(), 0)

    def test_get_weights_version_after_update(self):
        self.mgr.weights_version = 5
        self.assertEqual(self.mgr.get_weights_version(), 5)

    # ---- update_max_version -------------------------------------------------

    def test_update_max_version_adds(self):
        self.mgr.max_possible_version = 0
        self.mgr.update_max_version(3)
        self.assertEqual(self.mgr.max_possible_version, 3)

    def test_update_max_version_accumulates(self):
        self.mgr.max_possible_version = 2
        self.mgr.update_max_version(1)
        self.assertEqual(self.mgr.max_possible_version, 3)

    # ---- clean_old_weights --------------------------------------------------

    def test_clean_old_weights_skip_when_version_low(self):
        self.mgr.weights_version = 1
        with patch('os.listdir') as mock_listdir:
            self.mgr.clean_old_weights()
            mock_listdir.assert_not_called()

    def test_clean_old_weights_removes_expired(self):
        self.mgr.weights_version = 5
        with patch('os.listdir', return_value=["weights_1", "weights_2", "weights_3", "weights_4", "weights_5"]), \
             patch('os.path.isdir', return_value=True), \
             patch('os.path.join', side_effect=lambda *a: "/".join(a)), \
             patch('shutil.rmtree') as mock_rmtree:
            self.mgr.clean_old_weights()
            # MAX_RETAIN_WEIGHTS_VERSION=2, version=5, so remove weights_1 and weights_2
            self.assertEqual(mock_rmtree.call_count, 2)

    def test_clean_old_weights_ignores_non_matching_entries(self):
        self.mgr.weights_version = 5
        with patch('os.listdir', return_value=["not_a_weight", "readme.txt"]), \
             patch('shutil.rmtree') as mock_rmtree:
            self.mgr.clean_old_weights()
            mock_rmtree.assert_not_called()

    # ---- _should_weights_update ---------------------------------------------

    def test_should_weights_update_input_outdated(self):
        self.mgr.weights_version = 5
        # weight_iter=3 means input_weight_version=4, which is <= 5
        result = self.mgr._should_weights_update(3)
        self.assertFalse(result)

    def test_should_weights_update_on_policy_always_true(self):
        self.mgr.weights_version = 0
        self.mgr.use_on_policy = True
        result = self.mgr._should_weights_update(0)
        self.assertTrue(result)

    def test_should_weights_update_off_policy_no_version_control(self):
        self.mgr.weights_version = 0
        self.mgr.use_on_policy = False
        self.mgr.enable_version_control = False
        result = self.mgr._should_weights_update(0)
        self.assertTrue(result)

    def test_should_weights_update_off_policy_version_control_match(self):
        self.mgr.weights_version = 0
        self.mgr.use_on_policy = False
        self.mgr.enable_version_control = True
        self.mgr.max_possible_version = 2
        # weight_iter=0, input_weight_version=1, required=2-1=1 => match
        result = self.mgr._should_weights_update(0)
        self.assertTrue(result)

    def test_should_weights_update_off_policy_version_control_no_match(self):
        self.mgr.weights_version = 0
        self.mgr.use_on_policy = False
        self.mgr.enable_version_control = True
        self.mgr.max_possible_version = 5
        # weight_iter=0, input_weight_version=1, required=5-1=4 => no match
        result = self.mgr._should_weights_update(0)
        self.assertFalse(result)

    # ---- _do_weights_update -------------------------------------------------

    def test_do_weights_update_moves_files(self):
        with patch('os.path.exists', return_value=False), \
             patch('os.makedirs'), \
             patch('os.listdir', return_value=["shard_0.pt", "shard_1.pt"]), \
             patch('os.path.isfile', return_value=True), \
             patch('os.path.join', side_effect=lambda *a: "/".join(a)), \
             patch('shutil.move') as mock_move, \
             patch('shutil.rmtree'):
            self.mgr._do_weights_update("/src/iter_0000001", 1)
            self.assertEqual(mock_move.call_count, 2)
            # After update, weights_version should be 2 (weight_iter + 1)
            self.assertEqual(self.mgr.weights_version, 2)

    def test_do_weights_update_removes_existing_dst(self):
        with patch('os.path.exists', return_value=True), \
             patch('os.makedirs'), \
             patch('os.listdir', return_value=[]), \
             patch('os.path.isfile', return_value=False), \
             patch('os.path.join', side_effect=lambda *a: "/".join(a)), \
             patch('shutil.rmtree') as mock_rmtree:
            self.mgr._do_weights_update("/src/iter_0000003", 3)
            mock_rmtree.assert_called_once()
            self.assertEqual(self.mgr.weights_version, 4)


    # ---- sync_weights_update ------------------------------------------------

    def test_sync_weights_update_full_flow(self):
        self.mgr.weights_version = 0
        self.mgr.use_on_policy = True
        with patch.object(self.mgr, 'clean_old_weights') as mock_clean, \
             patch.object(self.mgr, '_should_weights_update', return_value=True) as mock_should, \
             patch.object(self.mgr, '_do_weights_update') as mock_do:
            self.mgr.sync_weights_update("/weights/iter_0000005")
            mock_clean.assert_called_once()
            mock_should.assert_called_once_with(5)
            mock_do.assert_called_once_with("/weights/iter_0000005", 5)

    def test_sync_weights_update_skips_when_should_returns_false(self):
        self.mgr.weights_version = 10
        with patch.object(self.mgr, 'clean_old_weights'), \
             patch.object(self.mgr, '_should_weights_update', return_value=False), \
             patch.object(self.mgr, '_do_weights_update') as mock_do:
            self.mgr.sync_weights_update("/weights/iter_0000002")
            mock_do.assert_not_called()

    # ---- init_done ----------------------------------------------------------

    def test_init_done(self):
        self.mgr.init_done()

    # ---- constants ----------------------------------------------------------

    def test_max_retain_weights_version(self):
        self.assertEqual(MAX_RETAIN_WEIGHTS_VERSION, 2)

    def test_path_iter_pattern(self):
        import re
        match = re.search(PATH_ITER_PATTERN, "/weights/iter_0000042")
        self.assertIsNotNone(match)
        self.assertEqual(int(match.group(1)), 42)


if __name__ == '__main__':
    unittest.main()
