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
import unittest
from unittest.mock import MagicMock


class TestActorRolloutHybrid(unittest.TestCase):
    def test_update_mini_batch_size(self):
        """Test the update_mini_batch_size function"""
        from aura.trainer.train_adapter.mindspeed_rl.patch.actor_rollout_hybrid import update_mini_batch_size

        # Create a mock self
        class MockSelf:
            def __init__(self):
                self.train_actor = MagicMock()

        mock_self = MockSelf()

        # Call the function
        update_mini_batch_size(mock_self, 8, 16, True)

        # Verify it was called correctly
        mock_self.train_actor.update_mini_batch_size.assert_called_once_with(8, 16, True)


if __name__ == '__main__':
    unittest.main()
