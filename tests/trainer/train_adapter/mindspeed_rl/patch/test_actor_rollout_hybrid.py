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
import unittest
from unittest.mock import patch, MagicMock

import pytest


class TestActorRolloutHybrid:
    @pytest.fixture(scope="class")
    def patch_modules(self):
        with patch.dict(sys.modules, {
            "mindspeed_rl": MagicMock(),
            "mindspeed_rl.utils": MagicMock(),
            "mindspeed_rl.utils.utils": MagicMock(),
            "mindspeed_rl.utils.compute": MagicMock(),
        }):
            yield

    def test_update_mini_batch_size_with_stepwise_advantage(self, patch_modules):
        from agentic_rl.trainer.train_adapter.mindspeed_rl.patch.actor_rollout_hybrid import update_mini_batch_size
        obj = MagicMock()
        obj.train_actor = MagicMock()
        original_n_samples_per_prompt = 10
        new_samples_per_prompt = 20
        update_mini_batch_size(obj, original_n_samples_per_prompt, new_samples_per_prompt, True)
        obj.train_actor.update_mini_batch_size.assert_called_once_with(
            original_n_samples_per_prompt, new_samples_per_prompt, True
        )

    def test_update_mini_batch_size_without_stepwise_advantage(self, patch_modules):
        from agentic_rl.trainer.train_adapter.mindspeed_rl.patch.actor_rollout_hybrid import update_mini_batch_size
        obj = MagicMock()
        obj.train_actor = MagicMock()
        original_n_samples_per_prompt = 10
        new_samples_per_prompt = 20
        update_mini_batch_size(obj, original_n_samples_per_prompt, new_samples_per_prompt, False)
        obj.train_actor.update_mini_batch_size.assert_called_once_with(
            original_n_samples_per_prompt, new_samples_per_prompt, False
        )


if __name__ == '__main__':
    unittest.main()
