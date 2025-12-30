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

import importlib
import sys
import unittest
from unittest import mock
from unittest.mock import MagicMock, Mock

import torch
from agentic_rl.configs.agentic_rl_config import AgenticRLConfig
from agentic_rl.runner.agent_engine_wrapper.base import Trajectory


class TestRunnerWorkerInit(unittest.TestCase):
    """Test cases for RunnerWorker.__init__ method."""

    @mock.patch('agentic_rl.runner.runner_worker.ray.remote', side_effect=lambda cls: cls)
    @mock.patch('agentic_rl.base.utils.class_loader.load_subclasses_from_file')
    @mock.patch('agentic_rl.runner.runner_worker.AutoTokenizer.from_pretrained')
    @mock.patch('agentic_rl.runner.runner_worker.FileCheck.check_data_path_is_valid')
    def test_init_success(self, mock_filecheck, mock_tokenizer, mock_load_subclasses, mock_ray):
        """Test successful initialization with all valid parameters."""
        # Reload module to apply the mocked ray.remote decorator
        if 'agentic_rl.runner.runner_worker' in sys.modules:
            importlib.reload(sys.modules['agentic_rl.runner.runner_worker'])
        from agentic_rl.runner.runner_worker import RunnerWorker

        # Arrange
        mock_tokenizer_instance = MagicMock()
        mock_tokenizer.return_value = mock_tokenizer_instance

        mock_engine_wrapper_instance = MagicMock()
        mock_engine_wrapper_class = Mock(return_value=mock_engine_wrapper_instance)
        mock_load_subclasses.return_value = mock_engine_wrapper_class

        # Create mock address with completions attribute
        mock_address = MagicMock()
        mock_address.completions.remote = MagicMock()

        config = AgenticRLConfig(agent_name="test_agent", max_steps=3)

        # Act
        worker = RunnerWorker(
            tokenizer_name_or_path="/path/to/tokenizer",
            sampling_params={"temperature": 0.8},
            max_prompt_length=4096,
            max_model_len=8192,
            n_parallel_agents=4,
            agent_engine_wrapper_path="/path/to/wrapper.py",
            addresses=[mock_address],
            agentic_rl_config=config,
        )

        # Assert
        self.assertEqual(worker.agentic_rl_config, config)
        self.assertEqual(worker.tokenizer, mock_tokenizer_instance)
        self.assertEqual(worker.agent_executor_wrapper, mock_engine_wrapper_instance)

        mock_filecheck.assert_any_call("/path/to/tokenizer")
        mock_filecheck.assert_any_call("/path/to/wrapper.py")
        mock_tokenizer.assert_called_once_with("/path/to/tokenizer", local_files_only=True, weights_only=True)
        mock_load_subclasses.assert_called_once()
        
        # Get the call arguments
        call_args = mock_engine_wrapper_class.call_args
        self.assertEqual(call_args.kwargs['agent_name'], "test_agent")
        self.assertEqual(call_args.kwargs['tokenizer'], mock_tokenizer_instance)
        self.assertEqual(call_args.kwargs['sampling_params'], {"temperature": 0.8})
        self.assertEqual(call_args.kwargs['max_prompt_length'], 4096)
        self.assertEqual(call_args.kwargs['max_response_length'], 4096)
        self.assertEqual(call_args.kwargs['n_parallel_agents'], 4)
        self.assertEqual(call_args.kwargs['max_steps'], 3)
        mock_engine_wrapper_instance.initialize.assert_called_once()

    @mock.patch('agentic_rl.runner.runner_worker.ray.remote', side_effect=lambda cls: cls)
    @mock.patch('agentic_rl.runner.runner_worker.FileCheck.check_data_path_is_valid')
    def test_init_filecheck_failure_tokenizer_path(self, mock_filecheck, mock_ray):
        """Test FileCheck failure for tokenizer_name_or_path."""
        # Reload module to apply the mocked ray.remote decorator
        if 'agentic_rl.runner.runner_worker' in sys.modules:
            importlib.reload(sys.modules['agentic_rl.runner.runner_worker'])
        from agentic_rl.runner.runner_worker import RunnerWorker

        # Arrange
        mock_filecheck.side_effect = ValueError("Invalid tokenizer path")
        
        # Create mock address with completions attribute
        mock_address = MagicMock()
        mock_address.completions.remote = MagicMock()
        
        config = AgenticRLConfig(agent_name="test_agent", max_steps=3)

        # Act & Assert
        with self.assertRaises(ValueError) as context:
            RunnerWorker(
                tokenizer_name_or_path="/invalid/path",
                agent_engine_wrapper_path="/path/to/wrapper.py",
                addresses=[mock_address],
                agentic_rl_config=config,
            )
        self.assertIn("Invalid tokenizer path", str(context.exception))

    @mock.patch('agentic_rl.runner.runner_worker.ray.remote', side_effect=lambda cls: cls)
    @mock.patch('agentic_rl.runner.runner_worker.AutoTokenizer.from_pretrained')
    @mock.patch('agentic_rl.runner.runner_worker.FileCheck.check_data_path_is_valid')
    def test_init_filecheck_failure_wrapper_path(self, mock_filecheck, mock_tokenizer, mock_ray):
        """Test FileCheck failure for agent_engine_wrapper_path."""
        # Reload module to apply the mocked ray.remote decorator
        if 'agentic_rl.runner.runner_worker' in sys.modules:
            importlib.reload(sys.modules['agentic_rl.runner.runner_worker'])
        from agentic_rl.runner.runner_worker import RunnerWorker

        # Arrange
        def filecheck_side_effect(path):
            if "wrapper" in path:
                raise ValueError("Invalid wrapper path")
        
        mock_filecheck.side_effect = filecheck_side_effect
        
        # Create mock address with completions attribute
        mock_address = MagicMock()
        mock_address.completions.remote = MagicMock()
        
        config = AgenticRLConfig(agent_name="test_agent", max_steps=3)

        # Act & Assert
        with self.assertRaises(ValueError) as context:
            RunnerWorker(
                tokenizer_name_or_path="/path/to/tokenizer",
                agent_engine_wrapper_path="/invalid/wrapper.py",
                addresses=[mock_address],
                agentic_rl_config=config,
            )
        self.assertIn("Invalid wrapper path", str(context.exception))

    @mock.patch('agentic_rl.runner.runner_worker.ray.remote', side_effect=lambda cls: cls)
    @mock.patch('agentic_rl.runner.runner_worker.AutoTokenizer.from_pretrained')
    @mock.patch('agentic_rl.runner.runner_worker.FileCheck.check_data_path_is_valid')
    def test_init_tokenizer_oserror(self, mock_filecheck, mock_tokenizer, mock_ray):
        """Test OSError during tokenizer loading."""
        # Reload module to apply the mocked ray.remote decorator
        if 'agentic_rl.runner.runner_worker' in sys.modules:
            importlib.reload(sys.modules['agentic_rl.runner.runner_worker'])
        from agentic_rl.runner.runner_worker import RunnerWorker

        # Arrange
        mock_tokenizer.side_effect = OSError("Failed to load tokenizer")
        
        # Create mock address with completions attribute
        mock_address = MagicMock()
        mock_address.completions.remote = MagicMock()
        
        config = AgenticRLConfig(agent_name="test_agent", max_steps=3)

        # Act & Assert
        with self.assertRaises(OSError) as context:
            RunnerWorker(
                tokenizer_name_or_path="/path/to/tokenizer",
                agent_engine_wrapper_path="/path/to/wrapper.py",
                addresses=[mock_address],
                agentic_rl_config=config,
            )
        self.assertIn("Failed to load tokenizer", str(context.exception))

    @mock.patch('agentic_rl.runner.runner_worker.ray.remote', side_effect=lambda cls: cls)
    @mock.patch('agentic_rl.runner.runner_worker.AutoTokenizer.from_pretrained')
    @mock.patch('agentic_rl.runner.runner_worker.FileCheck.check_data_path_is_valid')
    def test_init_tokenizer_valueerror(self, mock_filecheck, mock_tokenizer, mock_ray):
        """Test ValueError during tokenizer loading."""
        # Reload module to apply the mocked ray.remote decorator
        if 'agentic_rl.runner.runner_worker' in sys.modules:
            importlib.reload(sys.modules['agentic_rl.runner.runner_worker'])
        from agentic_rl.runner.runner_worker import RunnerWorker

        # Arrange
        mock_tokenizer.side_effect = ValueError("Invalid tokenizer configuration")
        
        # Create mock address with completions attribute
        mock_address = MagicMock()
        mock_address.completions.remote = MagicMock()
        
        config = AgenticRLConfig(agent_name="test_agent", max_steps=3)

        # Act & Assert
        with self.assertRaises(ValueError) as context:
            RunnerWorker(
                tokenizer_name_or_path="/path/to/tokenizer",
                agent_engine_wrapper_path="/path/to/wrapper.py",
                addresses=[mock_address],
                agentic_rl_config=config,
            )
        self.assertIn("Invalid tokenizer configuration", str(context.exception))

    @mock.patch('agentic_rl.runner.runner_worker.ray.remote', side_effect=lambda cls: cls)
    @mock.patch('agentic_rl.runner.runner_worker.AutoTokenizer.from_pretrained')
    @mock.patch('agentic_rl.runner.runner_worker.FileCheck.check_data_path_is_valid')
    def test_init_tokenizer_generic_exception(self, mock_filecheck, mock_tokenizer, mock_ray):
        """Test generic Exception during tokenizer loading."""
        # Reload module to apply the mocked ray.remote decorator
        if 'agentic_rl.runner.runner_worker' in sys.modules:
            importlib.reload(sys.modules['agentic_rl.runner.runner_worker'])
        from agentic_rl.runner.runner_worker import RunnerWorker

        # Arrange
        mock_tokenizer.side_effect = RuntimeError("Unexpected tokenizer error")
        
        # Create mock address with completions attribute
        mock_address = MagicMock()
        mock_address.completions.remote = MagicMock()
        
        config = AgenticRLConfig(agent_name="test_agent", max_steps=3)

        # Act & Assert
        with self.assertRaises(RuntimeError) as context:
            RunnerWorker(
                tokenizer_name_or_path="/path/to/tokenizer",
                agent_engine_wrapper_path="/path/to/wrapper.py",
                addresses=[mock_address],
                agentic_rl_config=config,
            )
        self.assertIn("Unexpected tokenizer error", str(context.exception))

    @mock.patch('agentic_rl.runner.runner_worker.ray.remote', side_effect=lambda cls: cls)
    @mock.patch('agentic_rl.base.utils.class_loader.load_subclasses_from_file')
    @mock.patch('agentic_rl.runner.runner_worker.AutoTokenizer.from_pretrained')
    @mock.patch('agentic_rl.runner.runner_worker.FileCheck.check_data_path_is_valid')
    def test_init_engine_wrapper_import_error(self, mock_filecheck, mock_tokenizer, mock_load_subclasses, mock_ray):
        """Test ImportError during engine wrapper class loading."""
        # Reload module to apply the mocked ray.remote decorator
        if 'agentic_rl.runner.runner_worker' in sys.modules:
            importlib.reload(sys.modules['agentic_rl.runner.runner_worker'])
        from agentic_rl.runner.runner_worker import RunnerWorker

        # Arrange
        mock_tokenizer.return_value = MagicMock()
        mock_load_subclasses.side_effect = ImportError("Failed to import wrapper module")
        
        # Create mock address with completions attribute
        mock_address = MagicMock()
        mock_address.completions.remote = MagicMock()
        
        config = AgenticRLConfig(agent_name="test_agent", max_steps=3)

        # Act & Assert
        with self.assertRaises(ImportError) as context:
            RunnerWorker(
                tokenizer_name_or_path="/path/to/tokenizer",
                agent_engine_wrapper_path="/path/to/wrapper.py",
                addresses=[mock_address],
                agentic_rl_config=config,
            )
        self.assertIn("Failed to import wrapper module", str(context.exception))

    @mock.patch('agentic_rl.runner.runner_worker.ray.remote', side_effect=lambda cls: cls)
    @mock.patch('agentic_rl.base.utils.class_loader.load_subclasses_from_file')
    @mock.patch('agentic_rl.runner.runner_worker.AutoTokenizer.from_pretrained')
    @mock.patch('agentic_rl.runner.runner_worker.FileCheck.check_data_path_is_valid')
    def test_init_engine_wrapper_generic_exception(self, mock_filecheck, mock_tokenizer, mock_load_subclasses, mock_ray):
        """Test generic Exception during engine wrapper class loading."""
        # Reload module to apply the mocked ray.remote decorator
        if 'agentic_rl.runner.runner_worker' in sys.modules:
            importlib.reload(sys.modules['agentic_rl.runner.runner_worker'])
        from agentic_rl.runner.runner_worker import RunnerWorker

        # Arrange
        mock_tokenizer.return_value = MagicMock()
        mock_load_subclasses.side_effect = RuntimeError("Unexpected loading error")
        
        # Create mock address with completions attribute
        mock_address = MagicMock()
        mock_address.completions.remote = MagicMock()
        
        config = AgenticRLConfig(agent_name="test_agent", max_steps=3)

        # Act & Assert
        with self.assertRaises(RuntimeError) as context:
            RunnerWorker(
                tokenizer_name_or_path="/path/to/tokenizer",
                agent_engine_wrapper_path="/path/to/wrapper.py",
                addresses=[mock_address],
                agentic_rl_config=config,
            )
        self.assertIn("Unexpected loading error", str(context.exception))

    @mock.patch('agentic_rl.runner.runner_worker.ray.remote', side_effect=lambda cls: cls)
    @mock.patch('agentic_rl.base.utils.class_loader.load_subclasses_from_file')
    @mock.patch('agentic_rl.runner.runner_worker.AutoTokenizer.from_pretrained')
    @mock.patch('agentic_rl.runner.runner_worker.FileCheck.check_data_path_is_valid')
    def test_init_engine_wrapper_init_typeerror(self, mock_filecheck, mock_tokenizer, mock_load_subclasses, mock_ray):
        """Test TypeError during engine wrapper initialization."""
        # Reload module to apply the mocked ray.remote decorator
        if 'agentic_rl.runner.runner_worker' in sys.modules:
            importlib.reload(sys.modules['agentic_rl.runner.runner_worker'])
        from agentic_rl.runner.runner_worker import RunnerWorker

        # Arrange
        mock_tokenizer.return_value = MagicMock()
        mock_engine_wrapper_class = Mock(side_effect=TypeError("Invalid argument type"))
        mock_load_subclasses.return_value = mock_engine_wrapper_class
        
        # Create mock address with completions attribute
        mock_address = MagicMock()
        mock_address.completions.remote = MagicMock()
        
        config = AgenticRLConfig(agent_name="test_agent", max_steps=3)

        # Act & Assert
        with self.assertRaises(TypeError) as context:
            RunnerWorker(
                tokenizer_name_or_path="/path/to/tokenizer",
                agent_engine_wrapper_path="/path/to/wrapper.py",
                addresses=[mock_address],
                agentic_rl_config=config,
            )
        self.assertIn("Invalid argument type", str(context.exception))

    @mock.patch('agentic_rl.runner.runner_worker.ray.remote', side_effect=lambda cls: cls)
    @mock.patch('agentic_rl.base.utils.class_loader.load_subclasses_from_file')
    @mock.patch('agentic_rl.runner.runner_worker.AutoTokenizer.from_pretrained')
    @mock.patch('agentic_rl.runner.runner_worker.FileCheck.check_data_path_is_valid')
    def test_init_engine_wrapper_init_valueerror(self, mock_filecheck, mock_tokenizer, mock_load_subclasses, mock_ray):
        """Test ValueError during engine wrapper initialization."""
        # Reload module to apply the mocked ray.remote decorator
        if 'agentic_rl.runner.runner_worker' in sys.modules:
            importlib.reload(sys.modules['agentic_rl.runner.runner_worker'])
        from agentic_rl.runner.runner_worker import RunnerWorker

        # Arrange
        mock_tokenizer.return_value = MagicMock()
        mock_engine_wrapper_class = Mock(side_effect=ValueError("Invalid configuration"))
        mock_load_subclasses.return_value = mock_engine_wrapper_class
        
        # Create mock address with completions attribute
        mock_address = MagicMock()
        mock_address.completions.remote = MagicMock()
        
        config = AgenticRLConfig(agent_name="test_agent", max_steps=3)

        # Act & Assert
        with self.assertRaises(ValueError) as context:
            RunnerWorker(
                tokenizer_name_or_path="/path/to/tokenizer",
                agent_engine_wrapper_path="/path/to/wrapper.py",
                addresses=[mock_address],
                agentic_rl_config=config,
            )
        self.assertIn("Invalid configuration", str(context.exception))

    @mock.patch('agentic_rl.runner.runner_worker.ray.remote', side_effect=lambda cls: cls)
    @mock.patch('agentic_rl.base.utils.class_loader.load_subclasses_from_file')
    @mock.patch('agentic_rl.runner.runner_worker.AutoTokenizer.from_pretrained')
    @mock.patch('agentic_rl.runner.runner_worker.FileCheck.check_data_path_is_valid')
    def test_init_engine_wrapper_init_generic_exception(self, mock_filecheck, mock_tokenizer, mock_load_subclasses, mock_ray):
        """Test generic Exception during engine wrapper initialization."""
        # Reload module to apply the mocked ray.remote decorator
        if 'agentic_rl.runner.runner_worker' in sys.modules:
            importlib.reload(sys.modules['agentic_rl.runner.runner_worker'])
        from agentic_rl.runner.runner_worker import RunnerWorker

        # Arrange
        mock_tokenizer.return_value = MagicMock()
        mock_engine_wrapper_class = Mock(side_effect=RuntimeError("Unexpected init error"))
        mock_load_subclasses.return_value = mock_engine_wrapper_class
        
        # Create mock address with completions attribute
        mock_address = MagicMock()
        mock_address.completions.remote = MagicMock()
        
        config = AgenticRLConfig(agent_name="test_agent", max_steps=3)

        # Act & Assert
        with self.assertRaises(RuntimeError) as context:
            RunnerWorker(
                tokenizer_name_or_path="/path/to/tokenizer",
                agent_engine_wrapper_path="/path/to/wrapper.py",
                addresses=[mock_address],
                agentic_rl_config=config,
            )
        self.assertIn("Unexpected init error", str(context.exception))

    @mock.patch('agentic_rl.runner.runner_worker.ray.remote', side_effect=lambda cls: cls)
    @mock.patch('agentic_rl.runner.runner_worker.FileCheck.check_data_path_is_valid')
    def test_init_invalid_sampling_params(self, mock_filecheck, mock_ray):
        """Test invalid sampling_params (not dict/None)."""
        # Reload module to apply the mocked ray.remote decorator
        if 'agentic_rl.runner.runner_worker' in sys.modules:
            importlib.reload(sys.modules['agentic_rl.runner.runner_worker'])
        from agentic_rl.runner.runner_worker import RunnerWorker

        # Arrange
        # Create mock address with completions attribute
        mock_address = MagicMock()
        mock_address.completions.remote = MagicMock()
        
        config = AgenticRLConfig(agent_name="test_agent", max_steps=3)

        # Act & Assert
        with self.assertRaises(ValueError) as context:
            RunnerWorker(
                tokenizer_name_or_path="/path/to/tokenizer",
                sampling_params="invalid",  # Should be dict or None
                agent_engine_wrapper_path="/path/to/wrapper.py",
                addresses=[mock_address],
                agentic_rl_config=config,
            )
        self.assertIn("sampling_params must be a dictionary", str(context.exception))

    @mock.patch('agentic_rl.runner.runner_worker.ray.remote', side_effect=lambda cls: cls)
    @mock.patch('agentic_rl.runner.runner_worker.FileCheck.check_data_path_is_valid')
    def test_init_invalid_max_prompt_length_zero(self, mock_filecheck, mock_ray):
        """Test invalid max_prompt_length (zero)."""
        # Reload module to apply the mocked ray.remote decorator
        if 'agentic_rl.runner.runner_worker' in sys.modules:
            importlib.reload(sys.modules['agentic_rl.runner.runner_worker'])
        from agentic_rl.runner.runner_worker import RunnerWorker

        # Arrange
        # Create mock address with completions attribute
        mock_address = MagicMock()
        mock_address.completions.remote = MagicMock()
        
        config = AgenticRLConfig(agent_name="test_agent", max_steps=3)

        # Act & Assert
        with self.assertRaises(ValueError) as context:
            RunnerWorker(
                tokenizer_name_or_path="/path/to/tokenizer",
                max_prompt_length=0,  # Must be positive
                agent_engine_wrapper_path="/path/to/wrapper.py",
                addresses=[mock_address],
                agentic_rl_config=config,
            )
        self.assertIn("max_prompt_length must be an positive integer", str(context.exception))

    @mock.patch('agentic_rl.runner.runner_worker.ray.remote', side_effect=lambda cls: cls)
    @mock.patch('agentic_rl.runner.runner_worker.FileCheck.check_data_path_is_valid')
    def test_init_invalid_max_prompt_length_negative(self, mock_filecheck, mock_ray):
        """Test invalid max_prompt_length (negative)."""
        # Reload module to apply the mocked ray.remote decorator
        if 'agentic_rl.runner.runner_worker' in sys.modules:
            importlib.reload(sys.modules['agentic_rl.runner.runner_worker'])
        from agentic_rl.runner.runner_worker import RunnerWorker

        # Arrange
        # Create mock address with completions attribute
        mock_address = MagicMock()
        mock_address.completions.remote = MagicMock()
        
        config = AgenticRLConfig(agent_name="test_agent", max_steps=3)

        # Act & Assert
        with self.assertRaises(ValueError) as context:
            RunnerWorker(
                tokenizer_name_or_path="/path/to/tokenizer",
                max_prompt_length=-100,  # Must be positive
                agent_engine_wrapper_path="/path/to/wrapper.py",
                addresses=[mock_address],
                agentic_rl_config=config,
            )
        self.assertIn("max_prompt_length must be an positive integer", str(context.exception))

    @mock.patch('agentic_rl.runner.runner_worker.ray.remote', side_effect=lambda cls: cls)
    @mock.patch('agentic_rl.runner.runner_worker.FileCheck.check_data_path_is_valid')
    def test_init_invalid_max_model_len_zero(self, mock_filecheck, mock_ray):
        """Test invalid max_model_len (zero)."""
        # Reload module to apply the mocked ray.remote decorator
        if 'agentic_rl.runner.runner_worker' in sys.modules:
            importlib.reload(sys.modules['agentic_rl.runner.runner_worker'])
        from agentic_rl.runner.runner_worker import RunnerWorker

        # Arrange
        # Create mock address with completions attribute
        mock_address = MagicMock()
        mock_address.completions.remote = MagicMock()
        
        config = AgenticRLConfig(agent_name="test_agent", max_steps=3)

        # Act & Assert
        with self.assertRaises(ValueError) as context:
            RunnerWorker(
                tokenizer_name_or_path="/path/to/tokenizer",
                max_model_len=0,  # Must be positive
                agent_engine_wrapper_path="/path/to/wrapper.py",
                addresses=[mock_address],
                agentic_rl_config=config,
            )
        self.assertIn("max_model_len must be an positive integer", str(context.exception))

    @mock.patch('agentic_rl.runner.runner_worker.ray.remote', side_effect=lambda cls: cls)
    @mock.patch('agentic_rl.runner.runner_worker.FileCheck.check_data_path_is_valid')
    def test_init_invalid_max_model_len_negative(self, mock_filecheck, mock_ray):
        """Test invalid max_model_len (negative)."""
        # Reload module to apply the mocked ray.remote decorator
        if 'agentic_rl.runner.runner_worker' in sys.modules:
            importlib.reload(sys.modules['agentic_rl.runner.runner_worker'])
        from agentic_rl.runner.runner_worker import RunnerWorker

        # Arrange
        # Create mock address with completions attribute
        mock_address = MagicMock()
        mock_address.completions.remote = MagicMock()
        
        config = AgenticRLConfig(agent_name="test_agent", max_steps=3)

        # Act & Assert
        with self.assertRaises(ValueError) as context:
            RunnerWorker(
                tokenizer_name_or_path="/path/to/tokenizer",
                max_model_len=-100,  # Must be positive
                agent_engine_wrapper_path="/path/to/wrapper.py",
                addresses=[mock_address],
                agentic_rl_config=config,
            )
        self.assertIn("max_model_len must be an positive integer", str(context.exception))

    @mock.patch('agentic_rl.runner.runner_worker.ray.remote', side_effect=lambda cls: cls)
    @mock.patch('agentic_rl.runner.runner_worker.FileCheck.check_data_path_is_valid')
    def test_init_invalid_n_parallel_agents_zero(self, mock_filecheck, mock_ray):
        """Test invalid n_parallel_agents (zero)."""
        # Reload module to apply the mocked ray.remote decorator
        if 'agentic_rl.runner.runner_worker' in sys.modules:
            importlib.reload(sys.modules['agentic_rl.runner.runner_worker'])
        from agentic_rl.runner.runner_worker import RunnerWorker

        # Arrange
        # Create mock address with completions attribute
        mock_address = MagicMock()
        mock_address.completions.remote = MagicMock()
        
        config = AgenticRLConfig(agent_name="test_agent", max_steps=3)

        # Act & Assert
        with self.assertRaises(ValueError) as context:
            RunnerWorker(
                tokenizer_name_or_path="/path/to/tokenizer",
                n_parallel_agents=0,  # Must be in range [1, 100]
                agent_engine_wrapper_path="/path/to/wrapper.py",
                addresses=[mock_address],
                agentic_rl_config=config,
            )
        self.assertIn("n_parallel_agents must be in range [1, 100]", str(context.exception))

    @mock.patch('agentic_rl.runner.runner_worker.ray.remote', side_effect=lambda cls: cls)
    @mock.patch('agentic_rl.runner.runner_worker.FileCheck.check_data_path_is_valid')
    def test_init_invalid_n_parallel_agents_over_limit(self, mock_filecheck, mock_ray):
        """Test invalid n_parallel_agents (over 100)."""
        # Reload module to apply the mocked ray.remote decorator
        if 'agentic_rl.runner.runner_worker' in sys.modules:
            importlib.reload(sys.modules['agentic_rl.runner.runner_worker'])
        from agentic_rl.runner.runner_worker import RunnerWorker

        # Arrange
        # Create mock address with completions attribute
        mock_address = MagicMock()
        mock_address.completions.remote = MagicMock()
        
        config = AgenticRLConfig(agent_name="test_agent", max_steps=3)

        # Act & Assert
        with self.assertRaises(ValueError) as context:
            RunnerWorker(
                tokenizer_name_or_path="/path/to/tokenizer",
                n_parallel_agents=101,  # Must be in range [1, 100]
                agent_engine_wrapper_path="/path/to/wrapper.py",
                addresses=[mock_address],
                agentic_rl_config=config,
            )
        self.assertIn("n_parallel_agents must be in range [1, 100]", str(context.exception))

    @mock.patch('agentic_rl.runner.runner_worker.ray.remote', side_effect=lambda cls: cls)
    @mock.patch('agentic_rl.runner.runner_worker.FileCheck.check_data_path_is_valid')
    def test_init_invalid_agentic_rl_config(self, mock_filecheck, mock_ray):
        """Test invalid agentic_rl_config (not AgenticRLConfig)."""
        # Reload module to apply the mocked ray.remote decorator
        if 'agentic_rl.runner.runner_worker' in sys.modules:
            importlib.reload(sys.modules['agentic_rl.runner.runner_worker'])
        from agentic_rl.runner.runner_worker import RunnerWorker

        # Arrange
        # Create mock address with completions attribute
        mock_address = MagicMock()
        mock_address.completions.remote = MagicMock()
        
        # Act & Assert
        with self.assertRaises(ValueError) as context:
            RunnerWorker(
                tokenizer_name_or_path="/path/to/tokenizer",
                agent_engine_wrapper_path="/path/to/wrapper.py",
                addresses=[mock_address],
                agentic_rl_config={"invalid": "config"},  # Must be AgenticRLConfig instance
            )
        self.assertIn("agentic_rl_config must be an AgenticRLConfig", str(context.exception))


class TestRunnerWorkerGenerateAgentTrajectoriesAsync(unittest.TestCase):
    """Test cases for RunnerWorker.generate_agent_trajectories_async method."""

    def _create_mock_trajectory(self, idx=0):
        """Helper to create a valid Trajectory object for testing."""
        # Create a Trajectory instance bypassing __init__ to avoid validation
        traj = object.__new__(Trajectory)
        traj.prompt_tokens = torch.tensor([1, 2, 3])
        traj.response_tokens = torch.tensor([4, 5, 6])
        traj.response_masks = torch.tensor([1, 1, 1])
        traj.idx = idx
        traj.trajectory_reward = 1.0
        traj.chat_completions = [{"role": "user", "content": "test"}]
        traj.metrics = {
            'steps': 1,
            'reward_time': 0.1,
            'env_time': 0.2,
            'llm_time': 0.3,
            'total_time': 0.6
        }
        return traj

    def _create_mock_worker(self):
        """Helper to create a mock RunnerWorker instance with mocked dependencies."""
        with mock.patch('agentic_rl.runner.runner_worker.ray.remote', side_effect=lambda cls: cls), \
             mock.patch('agentic_rl.base.utils.class_loader.load_subclasses_from_file'), \
             mock.patch('agentic_rl.runner.runner_worker.AutoTokenizer.from_pretrained'), \
             mock.patch('agentic_rl.runner.runner_worker.FileCheck.check_data_path_is_valid'):
            
            # Reload module to apply the mocked ray.remote decorator
            if 'agentic_rl.runner.runner_worker' in sys.modules:
                importlib.reload(sys.modules['agentic_rl.runner.runner_worker'])
            from agentic_rl.runner.runner_worker import RunnerWorker
            
            with mock.patch('agentic_rl.base.utils.class_loader.load_subclasses_from_file') as mock_load:
                mock_engine_wrapper_instance = MagicMock()
                mock_engine_wrapper_class = Mock(return_value=mock_engine_wrapper_instance)
                mock_load.return_value = mock_engine_wrapper_class
                
                # Create mock address with completions attribute
                mock_address = MagicMock()
                mock_address.completions.remote = MagicMock()
                
                config = AgenticRLConfig(agent_name="test_agent", max_steps=3)
                worker = RunnerWorker(
                    tokenizer_name_or_path="/path/to/tokenizer",
                    agent_engine_wrapper_path="/path/to/wrapper.py",
                    addresses=[mock_address],
                    agentic_rl_config=config,
                )
                
                return worker

    def test_generate_agent_trajectories_async_success(self):
        """Test successful generation with valid tasks."""
        # Arrange
        worker = self._create_mock_worker()
        tasks = [{"task_id": 1, "prompt": "test prompt"}]
        expected_result = [self._create_mock_trajectory(idx=0)]
        worker.agent_executor_wrapper.generate_agent_trajectories_async.return_value = expected_result

        # Act
        result = worker.generate_agent_trajectories_async(tasks)

        # Assert
        self.assertEqual(result, expected_result)
        worker.agent_executor_wrapper.generate_agent_trajectories_async.assert_called_once_with(tasks)

    def test_generate_agent_trajectories_async_tasks_not_list(self):
        """Test tasks parameter not a list (TypeError)."""
        # Arrange
        worker = self._create_mock_worker()

        # Act & Assert
        with self.assertRaises(TypeError) as context:
            worker.generate_agent_trajectories_async(tasks="not a list")
        self.assertIn("tasks must be a list", str(context.exception))

    def test_generate_agent_trajectories_async_empty_tasks(self):
        """Test empty tasks list (ValueError)."""
        # Arrange
        worker = self._create_mock_worker()

        # Act & Assert
        with self.assertRaises(ValueError) as context:
            worker.generate_agent_trajectories_async(tasks=[])
        self.assertIn("tasks list cannot be empty", str(context.exception))

    def test_generate_agent_trajectories_async_tasks_non_dict_item(self):
        """Test tasks containing non-dict item (TypeError)."""
        # Arrange
        worker = self._create_mock_worker()
        tasks = [{"task_id": 1}, "not a dict", {"task_id": 2}]

        # Act & Assert
        with self.assertRaises(TypeError) as context:
            worker.generate_agent_trajectories_async(tasks=tasks)
        self.assertIn("task must be a dictionary", str(context.exception))

    def test_generate_agent_trajectories_async_runtime_error(self):
        """Test RuntimeError during trajectory generation."""
        # Arrange
        worker = self._create_mock_worker()
        tasks = [{"task_id": 1, "prompt": "test"}]
        worker.agent_executor_wrapper.generate_agent_trajectories_async.side_effect = RuntimeError("Agent execution failed")

        # Act & Assert
        with self.assertRaises(RuntimeError) as context:
            worker.generate_agent_trajectories_async(tasks=tasks)
        self.assertIn("Agent execution failed", str(context.exception))

    def test_generate_agent_trajectories_async_generic_exception(self):
        """Test generic Exception during trajectory generation."""
        # Arrange
        worker = self._create_mock_worker()
        tasks = [{"task_id": 1, "prompt": "test"}]
        worker.agent_executor_wrapper.generate_agent_trajectories_async.side_effect = Exception("Unexpected error")

        # Act & Assert
        with self.assertRaises(Exception) as context:
            worker.generate_agent_trajectories_async(tasks=tasks)
        self.assertIn("Unexpected error", str(context.exception))


if __name__ == '__main__':
    unittest.main()
