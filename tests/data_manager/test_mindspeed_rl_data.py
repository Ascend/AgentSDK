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
# Module-level mocks  (BEFORE importing the code under test)
# ---------------------------------------------------------------------------
mock_ray = MagicMock()
mock_torch = MagicMock()

mock_loggers_module = MagicMock()
mock_loggers_module.Loggers.return_value.get_logger.return_value = MagicMock()

mock_data_transform_module = MagicMock()
mock_data_transform_module.padding_dict_to_tensor_dict = MagicMock(side_effect=lambda x: x)

# utils.py imports ray, requests, torch and Loggers
mock_requests = MagicMock()
mock_controller_utils_module = MagicMock()
mock_controller_utils_module.DEFAULT_SLEEP_TIME = 2
mock_controller_utils_module.MAX_TIMEOUT = 1800

with patch.dict(sys.modules, {
    'ray': mock_ray,
    'torch': mock_torch,
    'torch.distributed': mock_torch.distributed,
    'requests': mock_requests,
    'agentic_rl.base.log.loggers': mock_loggers_module,
    'agentic_rl.data_manager.data_transform': mock_data_transform_module,
    'agentic_rl.controllers.utils.utils': mock_controller_utils_module,
}):
    from agentic_rl.data_manager.mindspeed_rl_data import MindSpeedRLDataManager
    import agentic_rl.data_manager.mindspeed_rl_data as _msrl_data_mod


class TestMindSpeedRLDataManager(unittest.TestCase):
    """Tests for MindSpeedRLDataManager covering all public methods."""

    def setUp(self):
        self.dm = MindSpeedRLDataManager()
        self.mock_remote_dm = MagicMock()
        self.dm.data_manager = self.mock_remote_dm

    def tearDown(self):
        mock_ray.reset_mock()
        mock_torch.reset_mock()
        mock_loggers_module.reset_mock()
        mock_data_transform_module.reset_mock()
        # Restore side_effect after reset
        mock_data_transform_module.padding_dict_to_tensor_dict = MagicMock(side_effect=lambda x: x)

    # ---- __init__ -----------------------------------------------------------

    def test_init_data_manager_is_none(self):
        """data_manager is None after construction."""
        fresh = MindSpeedRLDataManager()
        self.assertIsNone(fresh.data_manager)

    # ---- sync_init_data_manager --------------------------------------------

    def test_sync_init_data_manager(self):
        """sync_init_data_manager stores the remote data manager."""
        remote_dm = MagicMock()
        fresh = MindSpeedRLDataManager()
        fresh.sync_init_data_manager(remote_dm)
        self.assertIs(fresh.data_manager, remote_dm)

    # ---- all_consumed -------------------------------------------------------

    def test_all_consumed_returns_zero_when_all_consumed(self):
        """all_consumed returns 0 when remote returns True (all consumed)."""
        with patch.object(_msrl_data_mod, 'ray') as patched_ray:
            patched_ray.get.return_value = True
            result = self.dm.all_consumed("train")
            self.assertEqual(result, 0)
            self.mock_remote_dm.all_consumed.remote.assert_called_once_with("train")

    def test_all_consumed_returns_one_when_not_consumed(self):
        """all_consumed returns 1 when remote returns False (not all consumed)."""
        with patch.object(_msrl_data_mod, 'ray') as patched_ray:
            patched_ray.get.return_value = False
            result = self.dm.all_consumed("train")
            self.assertEqual(result, 1)

    # ---- get_data -----------------------------------------------------------

    def test_get_data_with_valid_index(self):
        """get_data returns batch and index when index is truthy."""
        with patch.object(_msrl_data_mod, 'ray') as patched_ray:
            batch = MagicMock()
            batch.keys.return_value = ["input_ids"]
            patched_ray.get.return_value = (batch, [0, 1])

            result_batch, result_idx = self.dm.get_data("train", ["input_ids"], 32)
            self.assertIs(result_batch, batch)
            self.assertEqual(result_idx, [0, 1])
            self.mock_remote_dm.get_experience.remote.assert_called_once_with(
                "train", ["input_ids"], 32, get_n_samples=True
            )

    def test_get_data_with_empty_index(self):
        """get_data returns ({}, []) when index is empty."""
        with patch.object(_msrl_data_mod, 'ray') as patched_ray:
            batch = MagicMock()
            batch.keys.return_value = ["input_ids"]
            patched_ray.get.return_value = (batch, [])

            result_batch, result_idx = self.dm.get_data("train", None, 16)
            self.assertEqual(result_batch, {})
            self.assertEqual(result_idx, [])

    # ---- put_data -----------------------------------------------------------

    def test_put_data_calls_cpu_on_tensor_values(self):
        """put_data calls .cpu() on non-list values and passes through list values."""
        with patch.object(_msrl_data_mod, 'padding_dict_to_tensor_dict') as patched_padding:
            patched_padding.side_effect = lambda x: x

            tensor_val = MagicMock()
            tensor_val.cpu.return_value = "cpu_tensor"
            list_val = [1, 2, 3]

            self.dm.put_data({"t": tensor_val, "l": list_val}, [0])

            tensor_val.cpu.assert_called_once()
            patched_padding.assert_called_once()
            self.mock_remote_dm.put_experience.remote.assert_called_once()

    def test_put_data_passes_to_padding_and_remote(self):
        """put_data calls padding_dict_to_tensor_dict then remote put_experience."""
        with patch.object(_msrl_data_mod, 'padding_dict_to_tensor_dict') as patched_padding:
            val = MagicMock()
            val.cpu.return_value = "cpu_val"
            patched_padding.return_value = {"padded": "data"}

            self.dm.put_data({"key": val}, [5])

            patched_padding.assert_called_once()
            self.mock_remote_dm.put_experience.remote.assert_called_once_with(
                data_dict={"padded": "data"}, indexes=[5]
            )

    # ---- put_experience -----------------------------------------------------

    def test_put_experience(self):
        """put_experience calls remote put_experience directly."""
        self.dm.put_experience({"x": 1}, [2])
        self.mock_remote_dm.put_experience.remote.assert_called_once_with(
            data_dict={"x": 1}, indexes=[2]
        )

    # ---- update_metrics -----------------------------------------------------

    def test_update_metrics(self):
        """update_metrics calls ray.get on remote update_metrics."""
        with patch.object(_msrl_data_mod, 'ray') as patched_ray:
            self.dm.update_metrics("loss", 0.5, True)
            self.mock_remote_dm.update_metrics.remote.assert_called_once_with("loss", 0.5, cumulate=True)
            patched_ray.get.assert_called()


if __name__ == '__main__':
    unittest.main()
