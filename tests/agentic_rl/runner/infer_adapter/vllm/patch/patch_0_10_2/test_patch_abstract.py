#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# -------------------------------------------------------------------------
#  This file is part of the AgentSDK project.
# Copyright (c) 2026 Huawei Technologies Co.,Ltd.
#
# AgentSDK is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#
#           http://license.coscl.org.cn/MulanPSL2
#
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
# -------------------------------------------------------------------------

import importlib
import importlib.util
import os
import sys
import unittest
from unittest.mock import MagicMock, patch


class TestPatchAbstract(unittest.TestCase):
    """Test patch_abstract.py module"""

    @classmethod
    def setUpClass(cls):
        """Set up test environment for the entire test class"""
        cls._setup_mocks()
        cls._import_module_under_test()

    @classmethod
    def tearDownClass(cls):
        """Clean up test environment for the entire test class"""
        cls._cleanup_mocks()

    @classmethod
    def _setup_mocks(cls):
        """Setup mock objects for vllm and loguru"""
        cls.mock_vllm = MagicMock()
        cls.mock_vllm.v1 = MagicMock()
        cls.mock_vllm.v1.executor = MagicMock()
        cls.mock_vllm_v1_executor_abstract = MagicMock()
        cls.mock_vllm.v1.executor.abstract = cls.mock_vllm_v1_executor_abstract

        cls.mock_vllm_v1_executor_abstract.Executor = MagicMock()

        cls.mock_loguru = MagicMock()
        cls.mock_logger = MagicMock()
        cls.mock_logger.info = MagicMock()
        cls.mock_loguru.logger = cls.mock_logger

        cls.modules_patcher = patch.dict('sys.modules', {
            'vllm': cls.mock_vllm,
            'vllm.v1': cls.mock_vllm.v1,
            'vllm.v1.executor': cls.mock_vllm.v1.executor,
            'vllm.v1.executor.abstract': cls.mock_vllm_v1_executor_abstract,
            'loguru': cls.mock_loguru,
        })
        cls.modules_patcher.start()

    @classmethod
    def _import_module_under_test(cls):
        """Import the module under test after mocks are set up"""
        test_file_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.abspath(os.path.join(test_file_dir, '..', '..', '..', '..', '..', '..', '..'))
        sys.path.append(project_root)

        spec = importlib.util.spec_from_file_location(
            'patch_abstract',
            os.path.join(project_root, 'agentic_rl', 'runner', 'infer_adapter', 'vllm', 'patch', 'patch_0_10_2',
                         'patch_abstract.py')
        )
        cls.patch_abstract = importlib.util.module_from_spec(spec)
        sys.modules['patch_abstract'] = cls.patch_abstract
        spec.loader.exec_module(cls.patch_abstract)

    @classmethod
    def _cleanup_mocks(cls):
        """Clean up mock patches"""
        cls.modules_patcher.stop()

    def setUp(self):
        """Set up test environment"""
        self.mock_logger.info.reset_mock()

        self.mock_executor = MagicMock()
        self.mock_executor.collective_rpc = MagicMock()

    def test_execute_dummy_batch_when_sleeping(self):
        """Test execute_dummy_batch_patch when executor is sleeping"""
        self.mock_executor.is_sleeping = True

        self.patch_abstract.execute_dummy_batch_patch(self.mock_executor)

        self.mock_logger.info.assert_called_once_with("Engine is currently sleeping, skipping dummy batch execution.")

        self.mock_executor.collective_rpc.assert_not_called()

    def test_execute_dummy_batch_when_not_sleeping(self):
        """Test execute_dummy_batch_patch when executor is not sleeping"""
        self.mock_executor.is_sleeping = False

        self.patch_abstract.execute_dummy_batch_patch(self.mock_executor)

        self.mock_logger.info.assert_not_called()

        self.mock_executor.collective_rpc.assert_called_once_with("execute_dummy_batch")


if __name__ == '__main__':
    unittest.main()
