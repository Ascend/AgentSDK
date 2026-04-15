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
import unittest
from unittest import mock
from unittest.mock import patch


class TestRolloutWorkerUtils(unittest.TestCase):
    def setUp(self):
        """Set up test environment"""
        self.original_modules = {}
        for module_name in ['mindspeed_rl', 'mindspeed_rl.utils', 'mindspeed_rl.utils.utils', 'verl', 'uvicorn', 'ray']:
            if module_name in sys.modules:
                self.original_modules[module_name] = sys.modules[module_name]
        
        # mock mindspeed_rl, verl, uvicorn and ray
        self.mock_mindspeed_rl = mock.MagicMock()
        self.mock_mindspeed_rl_utils = mock.MagicMock()
        self.mock_mindspeed_rl_utils.utils = mock.MagicMock()
        self.mock_verl = mock.MagicMock()
        self.mock_uvicorn = mock.MagicMock()
        self.mock_ray = mock.MagicMock()
        
        class MockRayRemote:
            def __init__(self, *args, **kwargs):
                self.args = args
                self.kwargs = kwargs

            def __call__(self, cls_or_func):
                if isinstance(cls_or_func, type):
                    class WrappedClass(cls_or_func):
                        pass
                    WrappedClass.__ray_decorator_args__ = self.args
                    WrappedClass.__ray_decorator_kwargs__ = self.kwargs
                    return WrappedClass
                else:
                    cls_or_func.__ray_decorator_args__ = self.args
                    cls_or_func.__ray_decorator_kwargs__ = self.kwargs
                    return cls_or_func

        self.mock_ray.remote = MockRayRemote()
        self.mock_ray.get = mock.MagicMock(return_value=0)
        
        # Replace modules with mocks to avoid import errors
        sys.modules['mindspeed_rl'] = self.mock_mindspeed_rl
        sys.modules['mindspeed_rl.utils'] = self.mock_mindspeed_rl_utils
        sys.modules['mindspeed_rl.utils.utils'] = self.mock_mindspeed_rl_utils.utils
        sys.modules['verl'] = self.mock_verl
        sys.modules['uvicorn'] = self.mock_uvicorn
        sys.modules['ray'] = self.mock_ray
        
        # Import test objects
        global get_least_common_multiple, generate_dummy_trajectory, parse_messages
        global _stat_rollout_metrics, clean_traj_groups, get_all_prompt_ids, RolloutWorker
        from agentic_rl.trainer.rollout.rollout_worker import (
            get_least_common_multiple,
            generate_dummy_trajectory,
            parse_messages,
            _stat_rollout_metrics,
            clean_traj_groups,
            get_all_prompt_ids,
            RolloutWorker
        )
    
    def tearDown(self):
        """Clean up test environment"""
        # Restore original modules
        for module_name, module in self.original_modules.items():
            sys.modules[module_name] = module
        # Delete mock modules
        mock_modules = ['mindspeed_rl', 'mindspeed_rl.utils', 'mindspeed_rl.utils.utils', 'verl', 'uvicorn', 'ray']
        for module_name in mock_modules:
            if module_name in sys.modules and module_name not in self.original_modules:
                del sys.modules[module_name]
        # Clean up global variables
        if 'get_least_common_multiple' in globals():
            del globals()['get_least_common_multiple']
        if 'generate_dummy_trajectory' in globals():
            del globals()['generate_dummy_trajectory']
        if 'parse_messages' in globals():
            del globals()['parse_messages']
        if '_stat_rollout_metrics' in globals():
            del globals()['_stat_rollout_metrics']
        if 'clean_traj_groups' in globals():
            del globals()['clean_traj_groups']
        if 'get_all_prompt_ids' in globals():
            del globals()['get_all_prompt_ids']
        if 'RolloutWorker' in globals():
            del globals()['RolloutWorker']
    def test_get_least_common_multiple(self):
        """Test get_least_common_multiple function"""
        self.assertEqual(get_least_common_multiple(4, 6), 12)
        self.assertEqual(get_least_common_multiple(3, 5), 15)
        self.assertEqual(get_least_common_multiple(2, 8), 8)
        self.assertEqual(get_least_common_multiple(1, 10), 10)

    def test_generate_dummy_trajectory(self):
        """Test generate_dummy_trajectory function"""
        trajectory = generate_dummy_trajectory(123)
        self.assertIsInstance(trajectory, dict)
        self.assertEqual(trajectory["prompt_id"], "123")
        # Since torch.Tensor is a mock object, we only check if the return values exist
        self.assertIn("prompt_tokens", trajectory)
        self.assertIn("response_tokens", trajectory)
        self.assertIn("response_masks", trajectory)
        self.assertEqual(trajectory["trajectory_reward"], 0.0)
        self.assertIsInstance(trajectory["chat_completions"], list)
        self.assertIsInstance(trajectory["trajectory"], dict)
        self.assertIsInstance(trajectory["metrics"], dict)

    def test_parse_messages_qwen(self):
        """Test parse_messages function handling qwen format"""
        prompt = "<|im_start|>system\nYou are a helpful assistant.<|im_end|><|im_start|>user\nHello!<|im_end|>"
        messages = parse_messages(prompt, model_name="qwen")
        self.assertEqual(len(messages), 2)
        self.assertEqual(messages[0]["role"], "system")
        self.assertEqual(messages[0]["content"], "You are a helpful assistant.")
        self.assertEqual(messages[1]["role"], "user")
        self.assertEqual(messages[1]["content"], "Hello!")

    def test_parse_messages_deepseek(self):
        """Test parse_messages function handling deepseek format"""
        prompt = "<｜system｜>You are a helpful assistant.<｜user｜>Hello!"
        messages = parse_messages(prompt, model_name="deepseek")
        self.assertEqual(len(messages), 2)
        self.assertEqual(messages[0]["role"], "system")
        self.assertEqual(messages[0]["content"], "You are a helpful assistant.")
        self.assertEqual(messages[1]["role"], "user")
        self.assertEqual(messages[1]["content"], "Hello!")

    def test_parse_messages_unsupported(self):
        """Test parse_messages function handling unsupported model"""
        prompt = "Hello!"
        with self.assertRaises(NotImplementedError):
            parse_messages(prompt, model_name="unsupported")

    def test_stat_rollout_metrics(self):
        """Test _stat_rollout_metrics function"""
        rollout_cost = 1.5
        resharding_to_infer = 0.5
        metrics = {
            "traj/res_reward_mean": 0.8,
            "traj/toolcall_reward_mean": 0.2,
            "traj/env_time_mean": 1.0
        }
        rollout_metrics = _stat_rollout_metrics(rollout_cost, resharding_to_infer, metrics)
        self.assertEqual(rollout_metrics["rollout_cost"], 1.5)
        self.assertEqual(rollout_metrics["resharding_to_infer"], 0.5)
        self.assertEqual(rollout_metrics["res_reward_mean"], 0.8)
        self.assertEqual(rollout_metrics["toolcall_reward_mean"], 0.2)

    def test_clean_traj_groups(self):
        """Test clean_traj_groups function"""
        traj_groups = {
            "1": [
                {"prompt_id": "1", "content": "traj1"},
                {"prompt_id": "1", "content": "traj2"}
            ],
            "2": [
                {"prompt_id": "2", "content": "traj3"}
            ]
        }
        all_prompt_ids = {"1", "2"}
        trajectories = [
            {"prompt_id": "1", "content": "traj1"},
            {"prompt_id": "2", "content": "traj3"}
        ]
        # Use patch to mock the behavior of clean_traj_groups function
        with patch('agentic_rl.trainer.rollout.rollout_worker.clean_traj_groups') as mock_clean_traj_groups:
            # Import the original function
            from agentic_rl.trainer.rollout.rollout_worker import clean_traj_groups
            # Call the original function
            clean_traj_groups(traj_groups, all_prompt_ids, trajectories)
            # Verify if the function is called correctly
            mock_clean_traj_groups.assert_called_once_with(traj_groups, all_prompt_ids, trajectories)

    def test_get_all_prompt_ids(self):
        """Test get_all_prompt_ids function"""
        class MockTask:
            def __init__(self, prompt_id):
                self.prompt_id = prompt_id
        tasks = [MockTask(1), MockTask(2), MockTask(1)]
        prompt_ids = get_all_prompt_ids(tasks)
        self.assertEqual(prompt_ids, {1, 2})

    def test_clean_traj_groups_with_nonexistent_trajectory(self):
        """Test clean_traj_groups function with nonexistent trajectory"""
        traj_groups = {
            "1": [
                {"prompt_id": "1", "content": "traj1"}
            ]
        }
        all_prompt_ids = {"1"}
        trajectories = [
            {"prompt_id": "1", "content": "nonexistent"}  # This trajectory is not in traj_groups
        ]

        # Import the original function
        from agentic_rl.trainer.rollout.rollout_worker import clean_traj_groups
        # Call the function
        clean_traj_groups(traj_groups, all_prompt_ids, trajectories)

        # Verify the traj_groups and all_prompt_ids are unchanged
        self.assertEqual(len(traj_groups["1"]), 1)
        self.assertEqual(all_prompt_ids, {"1"})


if __name__ == '__main__':
    unittest.main()
