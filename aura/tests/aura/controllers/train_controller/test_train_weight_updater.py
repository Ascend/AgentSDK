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
# train_weight_updater.py imports:
#   - time, dataclasses (stdlib -- no mock needed)
#   - ray  (used as @ray.remote decorator on WeightUpdateActor)
#   - aura.base.log.loggers.Loggers  (which imports torch, torch.distributed)
# ---------------------------------------------------------------------------
mock_ray = MagicMock()
mock_torch = MagicMock()

# Make @ray.remote a passthrough decorator so the class is importable
mock_ray.remote = MagicMock(side_effect=lambda cls: cls)

mock_loggers_module = MagicMock()
mock_loggers_module.Loggers.return_value.get_logger.return_value = MagicMock()

with patch.dict(
    sys.modules,
    {
        'ray': mock_ray,
        'torch': mock_torch,
        'torch.distributed': mock_torch.distributed,
        'aura.base.log.loggers': mock_loggers_module,
    },
):
    from aura.controllers.train_controller.train_weight_updater import (
        WeightUpdateActor,
    )


class TestWeightUpdateActor(unittest.TestCase):
    """Tests for WeightUpdateActor covering all methods."""

    def setUp(self):
        self.dispatch_actor = MagicMock()
        self.handler_a = MagicMock()
        self.handler_b = MagicMock()
        self.actor = WeightUpdateActor(
            dispatch_actor=self.dispatch_actor,
            actor_handlers=[self.handler_a, self.handler_b],
        )

    def tearDown(self):
        mock_ray.reset_mock()
        mock_torch.reset_mock()
        mock_loggers_module.reset_mock()


if __name__ == '__main__':
    unittest.main()
