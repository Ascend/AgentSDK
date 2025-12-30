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
import asyncio
import os
import unittest
from functools import wraps
from unittest.mock import patch, Mock, MagicMock

import torch

from agentic_rl.configs.agentic_rl_config import AgenticRLConfig
from agentic_rl.configs.agentic_rl_config import GenConfig
from agentic_rl.runner.agent_engine_wrapper.base import Trajectory


def dummy_compile(*compile_args, **compile_kwargs):
    def decorate(fn):
        def wrapper(*args, **kwargs):
            return fn(*args, **kwargs)

        return wrapper

    return decorate


torch.jit.script = dummy_compile
torch.compile = dummy_compile


def mock_ray_remote(*args, **kwargs):
    def decorator(cls):
        @wraps(cls)
        def wrapper(*w_args, **w_kwargs):
            return cls(*w_args, **w_kwargs)

        return wrapper

    if len(args) == 1 and not kwargs and callable(args[0]):
        return decorator(args[0])

    return decorator


with patch('ray.remote', mock_ray_remote):
    from agentic_rl.trainer.rollout.rollout_worker import parse_qwen_messages, RolloutWorker


class TestRolloutWorker(unittest.TestCase):
    @patch('agentic_rl.base.utils.file_utils.FileCheck.check_data_path_is_valid')
    @patch('agentic_rl.trainer.rollout.rollout_worker.AutoTokenizer.from_pretrained')
    @patch('agentic_rl.trainer.rollout.rollout_worker.DataManager')
    @patch('agentic_rl.trainer.rollout.rollout_worker.AsyncServerManager')
    @patch('agentic_rl.trainer.rollout.rollout_worker.RunnerWorker')
    def setUp(self, mock_runner_worker, mock_async_server_manager, mock_data_manager, mock_tokenizer,
              mock_is_file_valid):
        mock_is_file_valid.return_value = True
        mock_tokenizer.return_value = MagicMock()
        mock_data_manager.return_value = MagicMock()
        mock_async_server_manager.return_value = MagicMock()
        mock_runner_worker.return_value = MagicMock()

        self.mock_is_file_valid = mock_is_file_valid
        self.mock_tokenizer = mock_tokenizer
        self.mock_data_manager = mock_data_manager
        self.mock_async_server_manager = mock_async_server_manager
        self.mock_runner_worker = mock_runner_worker
        self.generate_config = GenConfig(tokenizer_name_or_path=os.path.dirname(__file__))

        self.agentic_rl_config = AgenticRLConfig()

        self.remove_padding_tensor_dict_to_dict = MagicMock()
        self.remove_padding_and_split_to_list = MagicMock()

        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)

        self.worker = RolloutWorker(
            n_parallel_agents=8,
            max_prompt_length=8192,
            actor_rollout_dispatch_size=0,
            simplify_think_content=False,
            tokenizer_name_or_path='tokenizer',
            dataset_additional_keys=None,
            generate_config=self.generate_config,
            agentic_rl_config=self.agentic_rl_config,
            worker_group=None,
            remove_padding_tensor_dict_to_dict=self.remove_padding_tensor_dict_to_dict,
            remove_padding_and_split_to_list=self.remove_padding_and_split_to_list,
        )

    def test_non_string_input(self):
        with self.assertRaises(TypeError):
            parse_qwen_messages(123)

    def test_input_too_long(self):
        with self.assertRaises(ValueError):
            parse_qwen_messages("a" * 100001)

    def test_no_im_tags(self):
        self.assertEqual(parse_qwen_messages("Hello, world!"), [])

    def test_valid_input(self):
        prompt = "<|im_start|>system\nHello, world!\n<|im_end|>\n<|im_start|>user\nHow are you?\n<|im_end|>\n<|im_start|>assistant\nI'm fine, thank you.\n<|im_end|>"
        expected_output = [
            {"role": "system", "content": "Hello, world!"},
            {"role": "user", "content": "How are you?"},
            {"role": "assistant", "content": "I'm fine, thank you."}
        ]
        self.assertEqual(parse_qwen_messages(prompt), expected_output)

    def test_invalid_role(self):
        prompt = "<|im_start|>invalid\nHello, world!\n<|im_end|>"
        self.assertEqual(parse_qwen_messages(prompt), [])

    def test_init(self):
        self.assertEqual(self.worker.actor_rollout_dispatch_size, 0)
        self.assertEqual(self.worker.tokenizer_name_or_path, 'tokenizer')
        self.assertEqual(self.worker.generate_config, self.generate_config)
        self.assertEqual(self.worker.agentic_rl_config, self.agentic_rl_config)
        self.assertIsNone(self.worker.parallel_state)
        self.assertEqual(self.worker.tokenizer, self.mock_tokenizer.return_value)
        self.assertEqual(self.worker.iteration, 0)
        self.assertIsNone(self.worker.dataset_additional_keys)
        self.assertEqual(self.worker.data_manager, self.mock_data_manager.return_value)
        self.assertEqual(self.worker.remove_padding_tensor_dict_to_dict, self.remove_padding_tensor_dict_to_dict)
        self.assertEqual(self.worker.remove_padding_and_split_to_list, self.remove_padding_and_split_to_list)
        self.assertEqual(self.worker.rollout_engine, self.mock_async_server_manager.return_value)
        self.assertTrue(hasattr(self.worker, 'runner_worker'))

    def test_init_without_tokenizer_name_or_path(self):
        with self.assertRaises(ValueError):
            RolloutWorker(
                n_parallel_agents=8,
                max_prompt_length=8192,
                actor_rollout_dispatch_size=0,
                simplify_think_content=False,
                tokenizer_name_or_path=None,
                dataset_additional_keys=None,
                generate_config=GenConfig(),
                agentic_rl_config=AgenticRLConfig(),
                worker_group=None,
                remove_padding_tensor_dict_to_dict={},
                remove_padding_and_split_to_list={},
            )

    def test_init_without_generate_config(self):
        with self.assertRaises(ValueError):
            RolloutWorker(
                n_parallel_agents=8,
                max_prompt_length=8192,
                actor_rollout_dispatch_size=0,
                simplify_think_content=False,
                tokenizer_name_or_path='tokenizer',
                dataset_additional_keys=None,
                generate_config=None,
                agentic_rl_config=AgenticRLConfig(),
                worker_group=None,
                remove_padding_tensor_dict_to_dict={},
                remove_padding_and_split_to_list={},
            )

    def test_init_without_agentic_rl_config(self):
        with self.assertRaises(ValueError):
            RolloutWorker(
                n_parallel_agents=8,
                max_prompt_length=8192,
                actor_rollout_dispatch_size=0,
                simplify_think_content=False,
                tokenizer_name_or_path='tokenizer',
                dataset_additional_keys=None,
                generate_config=GenConfig(),
                agentic_rl_config=None,
                worker_group=None,
                remove_padding_tensor_dict_to_dict={},
                remove_padding_and_split_to_list={},
            )

    def test_init_without_remove_padding_tensor_dict_to_dict(self):
        with self.assertRaises(ValueError):
            RolloutWorker(
                n_parallel_agents=8,
                max_prompt_length=8192,
                actor_rollout_dispatch_size=0,
                simplify_think_content=False,
                tokenizer_name_or_path='tokenizer',
                dataset_additional_keys=None,
                generate_config=GenConfig(),
                agentic_rl_config=AgenticRLConfig(),
                worker_group=None,
                remove_padding_tensor_dict_to_dict=None,
                remove_padding_and_split_to_list={},
            )

    def test_init_without_remove_padding_and_split_to_list(self):
        with self.assertRaises(ValueError):
            RolloutWorker(
                n_parallel_agents=8,
                max_prompt_length=8192,
                actor_rollout_dispatch_size=0,
                simplify_think_content=False,
                tokenizer_name_or_path='tokenizer',
                dataset_additional_keys=None,
                generate_config=GenConfig(),
                agentic_rl_config=AgenticRLConfig(),
                worker_group=None,
                remove_padding_tensor_dict_to_dict={},
                remove_padding_and_split_to_list=None,
            )

    def test_validate_param_correct_type(self):
        """Test _validate_param with correct type"""
        self.assertIsNone(self.worker._validate_param(5, 'test_param', int))

    def test_validate_param_incorrect_type(self):
        """Test _validate_param with incorrect type"""
        with self.assertRaises(ValueError) as context:
            self.worker._validate_param('5', 'test_param', int)
        self.assertEqual(str(context.exception), "test_param: 5 type error, should int.")

    def test_validate_param_with_min_val(self):
        """Test _validate_param with min_val"""
        with self.assertRaises(ValueError) as context:
            self.worker._validate_param(3, 'test_param', int, min_val=5)
        self.assertEqual(str(context.exception), "test_param: 3, should be ≥ 5.")

    def test_validate_param_with_max_val(self):
        """Test _validate_param with max_val"""
        with self.assertRaises(ValueError) as context:
            self.worker._validate_param(7, 'test_param', int, max_val=5)
        self.assertEqual(str(context.exception), "test_param: 7, should be ≤ 5.")

    @patch('agentic_rl.base.utils.file_utils.FileCheck.check_data_path_is_valid')
    def test_validate_param_with_path_check(self, mock_check):
        """Test _validate_param with path_check"""
        mock_check.side_effect = TypeError
        with self.assertRaises(ValueError) as context:
            self.worker._validate_param('invalid_path', 'test_param', str, path_check=True)
        self.assertEqual(str(context.exception), "test_param: invalid_path is invalid, should be valid path.")

    def test_init_data_manager(self):
        self.data_manager = Mock()
        self.worker.init_data_manager(self.data_manager)
        self.worker.data_manager.sync_init_data_manager.assert_called_once_with(self.data_manager)

    def test_get_batch_data_no_data(self):
        self.worker.data_manager.all_consumed.return_value = 0
        result = self.loop.run_until_complete(self.worker._get_batch_data('stage', ['column'], 1))
        self.assertEqual(result, (None, None))

    def test_get_batch_data_with_data_no_index(self):
        self.worker.data_manager.all_consumed.side_effect = [1, 0]
        self.worker.data_manager.get_data.return_value = ({'data': 'value'}, None)
        result = self.loop.run_until_complete(self.worker._get_batch_data('stage', ['column'], 1))
        self.assertEqual(result, (None, None))

    def test_get_batch_data_with_data_and_index(self):
        self.worker.data_manager.all_consumed.side_effect = [1, 0]
        self.worker.data_manager.get_data.return_value = ({'data': 'value'}, 'index')
        result = self.loop.run_until_complete(self.worker._get_batch_data('stage', ['column'], 1))
        self.assertEqual(result, ({'data': 'value'}, 'index'))

    @patch('agentic_rl.trainer.rollout.rollout_worker.parse_qwen_messages')
    @patch('agentic_rl.trainer.rollout.rollout_worker.AutoTokenizer')
    def test_preprocess_batch_data(self, mock_tokenizer, mock_parse_qwen_messages):
        mock_tokenizer.decode.side_effect = lambda x: f"decoded_{x}"
        self.worker.tokenizer = mock_tokenizer
        self.worker.remove_padding_tensor_dict_to_dict = Mock()
        self.worker.remove_padding_tensor_dict_to_dict.side_effect = lambda x: x

        mock_parse_qwen_messages.side_effect = lambda x, max_length=100: [{'role': 'user', 'content': x}]

        batch_data = {'prompts': ['Test prompt 1']}
        expected_problems = ['decoded_Test prompt 1']
        self.assertEqual(self.worker._preprocess_batch_data(batch_data), expected_problems)

        # Test case 2: Multiple prompts with user role
        batch_data = {'prompts': ['Test prompt 1', 'Test prompt 2']}
        expected_problems = ['decoded_Test prompt 1', 'decoded_Test prompt 2']
        self.assertEqual(self.worker._preprocess_batch_data(batch_data), expected_problems)

        # Test case 3: Prompts with different roles
        batch_data = {'prompts': ['User prompt']}
        expected_problems = ['decoded_User prompt']
        self.assertEqual(self.worker._preprocess_batch_data(batch_data), expected_problems)

        # Test case 4: No user role in prompts
        mock_parse_qwen_messages.side_effect = lambda x, max_length=100: [{'role': 'assistant', 'content': x}]
        batch_data = {'prompts': ['Assistant response']}
        expected_problems = []
        self.assertEqual(self.worker._preprocess_batch_data(batch_data), expected_problems)

    def test_generate_tasks(self):
        self.worker.dataset_additional_keys = ["question", "labels", "other"]
        self.worker.tokenizer.decode.side_effect = lambda x: f"decoded_{x}"
        batch_data = {
            "question": torch.tensor([1, 2, 3]),
            "labels": torch.tensor([4, 5, 6]),
            "other": torch.tensor([7, 8, 9])
        }
        problems = ["problem1", "problem2", "problem3"]
        index = [0, 1, 2]

        tasks = self.worker._generate_tasks(batch_data, problems, index)

        self.assertEqual(len(tasks), 3)
        for i, task in enumerate(tasks):
            self.assertEqual(task["id"], index[i])
            self.assertEqual(task["question"], f"decoded_{batch_data['question'][i]}")
            self.assertEqual(task["ground_truth"], f"decoded_{batch_data['labels'][i]}")
            self.assertEqual(task["other"], f"decoded_{batch_data['other'][i]}")

    def test_generate_tasks_index_error(self):
        self.worker.dataset_additional_keys = ["question", "labels"]
        self.worker.tokenizer.decode.side_effect = lambda x: f"decoded_{x}"
        batch_data = {
            "question": [1, 2, 3],
            "labels": [4, 5]
        }
        problems = ["question1", "question2", "question3"]
        index = [0, 1, 2]
        with self.assertRaises(IndexError):
            self.worker._generate_tasks(batch_data, problems, index)

    @patch('ray.get')
    def test_generate_trajectories_success(self, mock_ray_get):
        self.worker.rollout_engine = MagicMock()
        self.worker.runner_worker = MagicMock()
        generate_mock = MagicMock()
        self.worker.runner_worker.generate_agent_trajectories_async.remote = generate_mock
        traj = [
            Trajectory(prompt_tokens=torch.tensor([]), response_tokens=torch.tensor([4, 5, 6]),
                       response_masks=torch.tensor([1, 1, 1]), trajectory_reward=1.0,
                       chat_completions=[{"role": "assistant", "content": "test"}],
                       metrics={"steps": 1, "reward_time": 2.0, "env_time": 3.0, "llm_time": 4.0, "total_time": 0.08}),
        ]
        mock_ray_get.return_value = traj
        tasks = ['task1', 'task2']
        result = self.loop.run_until_complete(self.worker._generate_trajectories(tasks))
        self.worker.rollout_engine.wake_up.assert_called_once()
        self.worker.rollout_engine.sleep.assert_called_once()
        generate_mock.assert_called_once_with(tasks)
        self.assertEqual(result, traj)

    @patch('ray.get')
    def test_generate_trajectories_exception(self, mock_ray_get):
        self.worker.rollout_engine = MagicMock()
        self.worker.runner_worker = MagicMock()
        generate_mock = MagicMock()
        self.worker.runner_worker.generate_agent_trajectories_async.remote = generate_mock
        mock_ray_get.side_effect = Exception('Test Exception')
        tasks = ['task1', 'task2']
        with self.assertRaises(RuntimeError) as context:
            self.loop.run_until_complete(self.worker._generate_trajectories(tasks))
        self.worker.rollout_engine.wake_up.assert_called_once()
        self.worker.rollout_engine.sleep.assert_not_called()
        generate_mock.assert_called_once_with(tasks)
        self.assertIn('Unexpected error during get response from API: Test Exception', str(context.exception))

    @patch('ray.get')
    def test_generate_trajectories_return_with_exception(self, mock_ray_get):
        self.worker.rollout_engine = MagicMock()
        self.worker.runner_worker = MagicMock()
        generate_mock = MagicMock()
        self.worker.runner_worker.generate_agent_trajectories_async.remote = generate_mock
        mock_ray_get.side_effect = ["return_error"]
        tasks = ['task1', 'task2']
        with self.assertRaises(RuntimeError) as context:
            self.loop.run_until_complete(self.worker._generate_trajectories(tasks))
        self.worker.rollout_engine.wake_up.assert_called_once()
        self.worker.rollout_engine.sleep.assert_not_called()
        generate_mock.assert_called_once_with(tasks)
        self.assertIn('Trajectories must be a list of Trajectory objects', str(context.exception))

    def test_process_trajectories(self):
        self.mock_tokenizer.pad_token_id = 0
        self.mock_tokenizer.eos_token_id = 1
        self.worker._transform_agent_trajectories = Mock()
        self.worker.remove_padding_and_split_to_list = Mock()

        self.worker._transform_agent_trajectories.return_value = (
            {
                'responses': torch.tensor([[1, 2, 3], [4, 5, 6]]),
                'input_ids': torch.tensor([[7, 8, 9], [10, 11, 12]]),
                'prompt_length': torch.tensor([3, 3]),
                'token_level_scores': torch.tensor([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]),
                'traj_mask': torch.tensor([[1, 1, 1], [1, 1, 1]])
            },
            {'metric1': 1.0, 'metric2': 2.0}
        )
        self.worker.remove_padding_and_split_to_list.return_value = [[1, 2, 3], [4, 5, 6]]

        trajectories = [
            Trajectory(idx=1, prompt_tokens=torch.tensor([1, 2, 3]), response_tokens=torch.tensor([4, 5, 6]),
                       response_masks=torch.tensor([1, 1, 1]), trajectory_reward=1.0,
                       chat_completions=[{"role": "assistant", "content": "test"}],
                       metrics={"steps": 1, "reward_time": 2.0, "env_time": 3.0, "llm_time": 4.0, "total_time": 0.08}),

            Trajectory(idx=0, prompt_tokens=torch.tensor([7, 8, 9]), response_tokens=torch.tensor([10, 11, 12]),
                       response_masks=torch.tensor([1, 1, 1]), trajectory_reward=2.0,
                       chat_completions=[{"role": "assistant", "content": "test"}],
                       metrics={"steps": 1, "reward_time": 2.0, "env_time": 3.0, "llm_time": 4.0, "total_time": 0.08}),
        ]

        outputs, metrics = self.worker._process_trajectories(trajectories)
        self.assertEqual(metrics, {'metric1': 1.0, 'metric2': 2.0})
        self.assertEqual(outputs['responses'], [[1, 2, 3], [4, 5, 6]])
        self.assertEqual(outputs['input_ids'].tolist(), [[7, 8, 9], [10, 11, 12]])
        self.assertEqual(outputs['response_length'], [torch.tensor([3]), torch.tensor([3])])
        self.assertEqual(outputs['prompt_length'].tolist(), [3, 3])
        self.assertTrue(torch.equal(outputs['rm_scores'][0], torch.tensor([0.1, 0.2, 0.3])))
        self.assertTrue(torch.equal(outputs['rm_scores'][1], torch.tensor([0.4, 0.5, 0.6])))
        self.assertTrue(torch.equal(outputs['token_level_rewards'][0], torch.tensor([0.1, 0.2, 0.3])))
        self.assertTrue(torch.equal(outputs['token_level_rewards'][1], torch.tensor([0.4, 0.5, 0.6])))
        self.assertEqual(outputs['response_mask'].tolist(), [[1, 1, 1], [1, 1, 1]])

    @patch('time.time', return_value=12345)
    def test_generate_sequences(self, mock_time):
        # Mock the return values of the private methods
        self.worker.dataset_additional_keys = []
        self.worker.actor_rollout_dispatch_size = 10
        self.worker.iteration = 0

        batch_data_mock = [
            ({'prompts': ['prompt1', 'prompt2'], 'prompt_length': [5, 6]}, [1, 2]),
            (None, None)
        ]

        async def mock_get_batch_data(*args, **kwargs):
            return batch_data_mock.pop(0)

        async def mock_generate_trajectories(tasks):
            return [MagicMock(spec=Trajectory), MagicMock(spec=Trajectory)]

        self.worker._get_batch_data = MagicMock(side_effect=mock_get_batch_data)
        self.worker._preprocess_batch_data = MagicMock()
        self.worker._generate_tasks = MagicMock()
        self.worker._generate_trajectories = MagicMock(side_effect=mock_generate_trajectories)
        self.worker._process_trajectories = MagicMock()

        self.worker._get_batch_data.return_value = (
            {'prompts': ['prompt1', 'prompt2'], 'prompt_length': [5, 6]}, [1, 2])
        self.worker._preprocess_batch_data.return_value = ['problem1', 'problem2']
        self.worker._generate_tasks.return_value = ['task1', 'task2']
        traj = [MagicMock(spec=Trajectory), MagicMock(spec=Trajectory)]
        self.worker._generate_trajectories.return_value = traj
        self.worker._process_trajectories.return_value = (
            ['output1', 'output2'], {'res_reward': 10, 'toolcall_reward': 20})

        self.loop.run_until_complete(self.worker.generate_sequences())

        self.assertEqual(self.worker.iteration, 1)
        self.worker.data_manager.put_data.assert_called_once_with(['output1', 'output2'], [1, 2])
        self.worker.data_manager.update_metrics.assert_any_call('res_reward', value=[10.0], cumulate=True)
        self.worker.data_manager.update_metrics.assert_any_call('toolcall_reward', value=[20.0], cumulate=True)

    def test_transform_agent_trajectories(self):
        self.worker.tokenizer = MagicMock()
        self.worker.tokenizer.pad_token_id = 0
        self.worker._extract_trajectory_data = MagicMock()
        self.worker.run_trajectories_perf_metric = MagicMock()
        self.worker._pad_sequences = MagicMock()
        self.worker._create_input_ids = MagicMock()

        # Mock the return values of the dependent methods
        self.worker._extract_trajectory_data.return_value = (
            ['initial_tokens'], ['response_tokens'], ['masks'], [1.0], ['chat_completions']
        )
        self.worker.run_trajectories_perf_metric.return_value = {'metric': 1.0}
        self.worker._pad_sequences.return_value = torch.tensor([[1, 2, 3]])
        self.worker._create_input_ids.return_value = (torch.tensor([1, 2, 3]), [3])

        trajectories = [{'trajectory': 'data'}]

        # Call the method to test
        result, metrics = self.worker._transform_agent_trajectories(trajectories)

        # Assert the expected results
        self.worker._extract_trajectory_data.assert_called_once_with(trajectories)
        self.worker.run_trajectories_perf_metric.assert_called_once_with(trajectories)
        self.worker._pad_sequences.assert_any_call(['initial_tokens'], left_pad=True)
        self.worker._pad_sequences.assert_any_call(['response_tokens'], left_pad=False)
        self.worker._pad_sequences.assert_any_call(['masks'], left_pad=False)
        self.worker._create_input_ids.assert_called_once_with(['initial_tokens'], ['response_tokens'])

        self.assertIsInstance(result, dict)
        self.assertIn('input_ids', result)
        self.assertIn('prompt_length', result)
        self.assertIn('attention_mask', result)
        self.assertIn('position_ids', result)
        self.assertIn('responses', result)
        self.assertIn('prompts', result)
        self.assertIn('token_level_scores', result)
        self.assertIn('traj_mask', result)

        self.assertIsInstance(metrics, dict)
        self.assertIn('metric', metrics)

    def test_transform_agent_trajectories_response_batch_dim_error(self):
        self.worker.tokenizer = MagicMock()
        self.worker.tokenizer.pad_token_id = 0
        self.worker._extract_trajectory_data = MagicMock()
        self.worker.run_trajectories_perf_metric = MagicMock()
        self.worker._pad_sequences = MagicMock()
        self.worker._create_input_ids = MagicMock()

        trajectories = [{}]
        all_initial_tokens = [torch.tensor([1, 2, 3])]
        all_response_tokens = [torch.tensor([4])]
        all_masks = [torch.tensor([1, 1, 1])]
        traj_scores = [1.0]
        chat_completions = [{}]
        self.worker._extract_trajectory_data.return_value = all_initial_tokens, all_response_tokens, all_masks, traj_scores, chat_completions
        self.worker.run_trajectories_perf_metric.return_value = {}
        self.worker._pad_sequences.return_value = torch.tensor([1, 2, 3])
        self.worker._create_input_ids.return_value = [torch.tensor([1, 2, 3])], [3]
        self.worker.tokenizer.pad_token_id = 0

        with self.assertRaises(ValueError):
            self.worker._transform_agent_trajectories(trajectories)

    def test_extract_trajectory_data(self):
        traj = [
            Trajectory(prompt_tokens=torch.tensor([1, 2, 3]), response_tokens=torch.tensor([4, 5, 6]),
                       response_masks=torch.tensor([1, 1, 1]), trajectory_reward=1.0,
                       chat_completions=[{"role": "assistant", "content": "test"}],
                       metrics={"steps": 1, "reward_time": 2.0, "env_time": 3.0, "llm_time": 4.0, "total_time": 0.08}),

            Trajectory(prompt_tokens=torch.tensor([7, 8, 9]), response_tokens=torch.tensor([10, 11, 12]),
                       response_masks=torch.tensor([1, 1, 1]), trajectory_reward=2.0,
                       chat_completions=[{"role": "assistant", "content": "test"}],
                       metrics={"steps": 1, "reward_time": 2.0, "env_time": 3.0, "llm_time": 4.0, "total_time": 0.08}),
        ]

        all_initial_tokens, all_response_tokens, all_masks, traj_scores, chat_completions = self.worker._extract_trajectory_data(
            traj)

        self.assertTrue(torch.equal(all_initial_tokens[0], torch.tensor([1, 2, 3])))
        self.assertTrue(torch.equal(all_initial_tokens[1], torch.tensor([7, 8, 9])))

        self.assertTrue(torch.equal(all_response_tokens[0], torch.tensor([4, 5, 6])))
        self.assertTrue(torch.equal(all_response_tokens[1], torch.tensor([10, 11, 12])))

        self.assertTrue(torch.equal(all_masks[0], torch.tensor([1, 1, 1])))
        self.assertTrue(torch.equal(all_masks[1], torch.tensor([1, 1, 1])))

        self.assertEqual(traj_scores, [1.0, 2.0])
        self.assertEqual(chat_completions,
                         [[{"role": "assistant", "content": "test"}], [{"role": "assistant", "content": "test"}]])

    def test_extract_trajectory_data_with_empty_prompt_tokens(self):
        traj = [
            Trajectory(prompt_tokens=torch.tensor([]), response_tokens=torch.tensor([4, 5, 6]),
                       response_masks=torch.tensor([1, 1, 1]), trajectory_reward=1.0,
                       chat_completions=[{"role": "assistant", "content": "test"}],
                       metrics={"steps": 1, "reward_time": 2.0, "env_time": 3.0, "llm_time": 4.0, "total_time": 0.08}),
        ]

        with self.assertRaises(ValueError):
            self.worker._extract_trajectory_data(traj)

    def test_extract_trajectory_data_with_empty_response_tokens(self):
        traj = [
            Trajectory(prompt_tokens=torch.tensor([1, 2, 3]), response_tokens=torch.tensor([]),
                       response_masks=torch.tensor([1, 1, 1]), trajectory_reward=1.0,
                       chat_completions=[{"role": "assistant", "content": "test"}],
                       metrics={"steps": 1, "reward_time": 2.0, "env_time": 3.0, "llm_time": 4.0, "total_time": 0.08}),
        ]

        with self.assertRaises(ValueError):
            self.worker._extract_trajectory_data(traj)

    def test_pad_sequences_left_pad_false(self):
        mock_tokenizer = MagicMock()
        mock_tokenizer.pad_token_id = 6
        self.worker.tokenizer = mock_tokenizer
        sequences = [torch.tensor([1, 2, 3]), torch.tensor([4, 5])]
        padded_sequences = self.worker._pad_sequences(sequences, left_pad=False)
        expected_sequences = torch.tensor([[1, 2, 3], [4, 5, mock_tokenizer.pad_token_id]])
        self.assertTrue(torch.equal(padded_sequences, expected_sequences))

    def test_pad_sequences_left_pad_true(self):
        mock_tokenizer = MagicMock()
        mock_tokenizer.pad_token_id = 6
        self.worker.tokenizer = mock_tokenizer
        sequences = [torch.tensor([1, 2, 3]), torch.tensor([4, 5])]
        padded_sequences = self.worker._pad_sequences(sequences, left_pad=True)
        expected_sequences = torch.tensor([[1, 2, 3], [mock_tokenizer.pad_token_id, 4, 5]])
        self.assertTrue(torch.equal(padded_sequences, expected_sequences))

    def test_create_input_ids(self):
        all_initial_tokens = [torch.tensor([1, 2, 3]), torch.tensor([4, 5, 6])]
        all_response_tokens = [torch.tensor([7, 8, 9]), torch.tensor([10, 11, 12])]
        input_ids_list, prompt_length_list = self.worker._create_input_ids(all_initial_tokens, all_response_tokens)

        self.assertTrue(torch.equal(input_ids_list[0], torch.tensor([1, 2, 3, 7, 8, 9])))
        self.assertTrue(torch.equal(input_ids_list[1], torch.tensor([4, 5, 6, 10, 11, 12])))
        self.assertTrue(torch.equal(prompt_length_list[0], torch.tensor([3])))
        self.assertTrue(torch.equal(prompt_length_list[1], torch.tensor([3])))

    def test_create_input_ids_empty(self):
        all_initial_tokens = []
        all_response_tokens = []
        input_ids_list, prompt_length_list = self.worker._create_input_ids(all_initial_tokens, all_response_tokens)
        self.assertEqual(input_ids_list, [])
        self.assertEqual(prompt_length_list, [])

    def test_create_input_ids_single_token(self):
        all_initial_tokens = [torch.tensor([1])]
        all_response_tokens = [torch.tensor([2])]
        input_ids_list, prompt_length_list = self.worker._create_input_ids(all_initial_tokens, all_response_tokens)
        self.assertTrue(torch.allclose(input_ids_list[0], torch.tensor([1, 2])), msg="张量值不匹配")
        self.assertEqual(prompt_length_list, [torch.tensor([1])])

    def test_run_trajectories_perf_metric_normal_input(self):
        """测试正常输入情况"""
        trajectories = [
            Trajectory(
                prompt_tokens=torch.rand(10),
                response_tokens=torch.rand(10),
                response_masks=torch.rand(10),
                idx=0,
                trajectory_reward=1.1,
                chat_completions=[],
                metrics={
                    "steps": 3,
                    "reward_time": 0.1,
                    "env_time": 0.1,
                    "llm_time": 0.2,
                    "total_time": 0.08,
                }
            ),
            Trajectory(
                prompt_tokens=torch.rand(10),
                response_tokens=torch.rand(10),
                response_masks=torch.rand(10),
                idx=1,
                trajectory_reward=1.1,
                chat_completions=[],
                metrics={
                    "steps": 3,
                    "reward_time": 0.1,
                    "env_time": 0.1,
                    "llm_time": 0.2,
                    "total_time": 0.08,
                }
            )
        ]

        result = self.worker.run_trajectories_perf_metric(trajectories)

        # 验证非数组类型指标
        self.assertAlmostEqual(result["traj/reward_time_mean"], 0.1)

    def test_run_trajectories_perf_metric_empty_trajectories(self):
        with self.assertRaises(ValueError) as context:
            self.worker.run_trajectories_perf_metric([])
        self.assertEqual(str(context.exception), "Parameter trajectories cannot be empty")

    def test_run_trajectories_perf_metric_invalid_input_type(self):
        with self.assertRaises(TypeError) as context:
            self.worker.run_trajectories_perf_metric("123")
        self.assertEqual(str(context.exception), "Parameter trajectories must be a not empty list")


if __name__ == "__main__":
    unittest.main()
