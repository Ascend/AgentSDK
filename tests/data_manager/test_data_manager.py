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
#
# data_manager.py imports data_registry which in turn imports all three
# concrete DataManager classes and their transitive dependencies.
# ---------------------------------------------------------------------------
mock_ray = MagicMock()
mock_torch = MagicMock()
mock_np = MagicMock()
mock_verl = MagicMock()
mock_requests = MagicMock()
mock_tensordict = MagicMock()

mock_loggers_module = MagicMock()
mock_loggers_module.Loggers.return_value.get_logger.return_value = MagicMock()

mock_base_utils_module = MagicMock()
mock_base_utils_module.singleton = lambda cls: cls

mock_rollout_queue_module = MagicMock()
mock_rollout_client_module = MagicMock()
mock_controller_config_module = MagicMock()
mock_http_status_module = MagicMock()

mock_controller_utils_module = MagicMock()
mock_controller_utils_module.DEFAULT_SLEEP_TIME = 2
mock_controller_utils_module.READ_TIMEOUT = 600
mock_controller_utils_module.DEFAULT_URL_METHOD = "http"
mock_controller_utils_module.MAX_TIMEOUT = 1800

mock_data_transform_module = MagicMock()

# Mock for mindspeed_rl transitive deps
mock_mindspeed_rl = MagicMock()
mock_mindspeed_rl_utils = MagicMock()
mock_mindspeed_rl_utils_utils = MagicMock()
mock_mindspeed_rl_utils_utils.mstx_timer_decorator = lambda fn: fn

with patch.dict(sys.modules, {
    'ray': mock_ray,
    'ray.util': MagicMock(),
    'ray.util.scheduling_strategies': MagicMock(),
    'torch': mock_torch,
    'torch.distributed': mock_torch.distributed,
    'torch.nn': mock_torch.nn,
    'torch.nn.utils': mock_torch.nn.utils,
    'torch.nn.utils.rnn': mock_torch.nn.utils.rnn,
    'numpy': mock_np,
    'verl': mock_verl,
    'requests': mock_requests,
    'tensordict': mock_tensordict,
    'mindspeed_rl': mock_mindspeed_rl,
    'mindspeed_rl.utils': mock_mindspeed_rl_utils,
    'mindspeed_rl.utils.utils': mock_mindspeed_rl_utils_utils,
    'agentic_rl.base.log.loggers': mock_loggers_module,
    'agentic_rl.base.utils.utils': mock_base_utils_module,
    'agentic_rl.controllers.rollout_controller.rollout_queue': mock_rollout_queue_module,
    'agentic_rl.controllers.rollout_controller.rollout_client': mock_rollout_client_module,
    'agentic_rl.controllers.utils.controller_config': mock_controller_config_module,
    'agentic_rl.controllers.utils.http_status': mock_http_status_module,
    'agentic_rl.controllers.utils.utils': mock_controller_utils_module,
    'agentic_rl.data_manager.data_transform': mock_data_transform_module,
}):
    from agentic_rl.data_manager.data_manager import DataManager
    import agentic_rl.data_manager.data_manager as _data_manager_mod


class TestDataManager(unittest.TestCase):
    """Tests for the DataManager facade class."""

    def setUp(self):
        """Create a DataManager backed by a mock instance."""
        self.mock_instance = MagicMock()
        # Bypass __init__ to avoid data_manager_class resolution issues in CI
        self.dm = DataManager.__new__(DataManager)
        self.dm.data_manager_instance = self.mock_instance

    def tearDown(self):
        mock_ray.reset_mock()
        mock_torch.reset_mock()

    # ---- __init__ -----------------------------------------------------------

    def test_init_sets_instance(self):
        """DataManager stores the data_manager_instance from the registry."""
        self.assertIs(self.dm.data_manager_instance, self.mock_instance)

    # ---- sync_init_data_manager --------------------------------------------

    def test_sync_init_data_manager(self):
        """sync_init_data_manager delegates to the instance."""
        remote_dm = MagicMock()
        self.dm.sync_init_data_manager(remote_dm)
        self.mock_instance.sync_init_data_manager.assert_called_once_with(remote_dm)

    # ---- all_consumed -------------------------------------------------------

    def test_all_consumed(self):
        """all_consumed delegates and returns the instance result."""
        self.mock_instance.all_consumed.return_value = 1
        result = self.dm.all_consumed("train")
        self.assertEqual(result, 1)
        self.mock_instance.all_consumed.assert_called_once_with("train")

    # ---- get_data -----------------------------------------------------------

    def test_get_data(self):
        """get_data delegates all arguments to the instance."""
        self.mock_instance.get_data.return_value = ({"ids": [1]}, [0])
        batch, idx = self.dm.get_data("train", ["ids"], 32, True)
        self.assertEqual(batch, {"ids": [1]})
        self.assertEqual(idx, [0])
        self.mock_instance.get_data.assert_called_once_with("train", ["ids"], 32, True)

    # ---- put_data -----------------------------------------------------------

    def test_put_data_without_metric(self):
        """put_data delegates to the instance without metric."""
        self.dm.put_data({"out": 1}, [0])
        self.mock_instance.put_data.assert_called_once_with({"out": 1}, [0], None)

    def test_put_data_with_metric(self):
        """put_data delegates to the instance with metric."""
        self.dm.put_data({"out": 1}, [0], metric={"loss": 0.5})
        self.mock_instance.put_data.assert_called_once_with({"out": 1}, [0], {"loss": 0.5})

    # ---- put_experience -----------------------------------------------------

    def test_put_experience(self):
        """put_experience delegates to the instance."""
        self.dm.put_experience({"x": 1}, [2])
        self.mock_instance.put_experience.assert_called_once_with({"x": 1}, [2])

    # ---- update_metrics -----------------------------------------------------

    def test_update_metrics(self):
        """update_metrics delegates to the instance."""
        self.dm.update_metrics("loss", 0.5, True)
        self.mock_instance.update_metrics.assert_called_once_with("loss", 0.5, True)

    # ---- set_pad_token_id ---------------------------------------------------

    def test_set_pad_token_id_when_supported(self):
        """set_pad_token_id calls instance method when it exists."""
        self.dm.set_pad_token_id(42)
        self.mock_instance.set_pad_token_id.assert_called_once_with(42)

    def test_set_pad_token_id_when_not_supported(self):
        """set_pad_token_id is a no-op when instance lacks the method."""
        del self.mock_instance.set_pad_token_id
        # Should not raise
        self.dm.set_pad_token_id(42)

    # ---- set_pad_token_id_from_tokenizer ------------------------------------

    def test_set_pad_token_id_from_tokenizer_uses_pad_token_id(self):
        """When tokenizer.pad_token_id is set, use it."""
        tokenizer = MagicMock()
        tokenizer.pad_token_id = 100
        tokenizer.eos_token_id = 200

        result = self.dm.set_pad_token_id_from_tokenizer(tokenizer)
        self.assertEqual(result, 100)
        self.mock_instance.set_pad_token_id.assert_called_once_with(100)

    def test_set_pad_token_id_from_tokenizer_uses_eos_token_id(self):
        """When pad_token_id is None, fall back to eos_token_id."""
        tokenizer = MagicMock()
        tokenizer.pad_token_id = None
        tokenizer.eos_token_id = 200

        result = self.dm.set_pad_token_id_from_tokenizer(tokenizer)
        self.assertEqual(result, 200)

    def test_set_pad_token_id_from_tokenizer_defaults_to_zero(self):
        """When both pad_token_id and eos_token_id are None, default to 0."""
        tokenizer = MagicMock()
        tokenizer.pad_token_id = None
        tokenizer.eos_token_id = None

        result = self.dm.set_pad_token_id_from_tokenizer(tokenizer)
        self.assertEqual(result, 0)

    # ---- get_pad_token_id_info ----------------------------------------------

    def test_get_pad_token_id_info_with_attribute(self):
        """get_pad_token_id_info returns _pad_token_id when present."""
        self.mock_instance._pad_token_id = 42
        result = self.dm.get_pad_token_id_info()
        self.assertEqual(result, 42)

    def test_get_pad_token_id_info_without_attribute(self):
        """get_pad_token_id_info returns None when _pad_token_id is absent."""
        # Remove the attribute from the mock
        del self.mock_instance._pad_token_id
        result = self.dm.get_pad_token_id_info()
        self.assertIsNone(result)


if __name__ == '__main__':
    unittest.main()
