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

import unittest
from unittest.mock import MagicMock, patch

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

with patch.dict(
    'sys.modules',
    {
        'ray': mock_ray,
        'torch': MagicMock(),
        'torch.distributed': MagicMock(),
        'transformers': mock_transformers,
        'aura.base.log.loggers': mock_loggers_module,
        'aura.base.utils.globals': mock_globals_module,
    },
):
    from aura.controllers.rollout_controller.rollout_weight_manager import RolloutWeightManager
    import aura.controllers.rollout_controller.rollout_weight_manager as _rwm_mod


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
        with (
            patch('os.makedirs'),
            patch('os.getenv', return_value='false'),
            patch.object(_rwm_mod, 'AutoConfig', mock_auto_config),
        ):
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


if __name__ == '__main__':
    unittest.main()
