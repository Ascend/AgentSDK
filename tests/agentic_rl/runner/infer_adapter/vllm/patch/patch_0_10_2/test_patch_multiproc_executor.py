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

import unittest
from unittest.mock import patch, MagicMock
import sys
import os
import importlib
import importlib.util


class TestPatchMultiprocExecutor(unittest.TestCase):
    """Test patch_multiproc_executor.py module"""

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
        """Setup mock objects for vllm, loguru, agentic_rl"""
        cls.mock_vllm = MagicMock()

        cls.mock_vllm_utils = MagicMock()
        cls.mock_mp_context = MagicMock()
        cls.mock_process = MagicMock()
        cls.mock_process.name = "test_process"
        cls.mock_mp_context.current_process.return_value = cls.mock_process
        cls.mock_vllm_utils.get_mp_context.return_value = cls.mock_mp_context

        cls.mock_vllm_config = MagicMock()
        cls.mock_vllm_config.VllmConfig = MagicMock()

        cls.mock_shm_broadcast = MagicMock()
        cls.mock_shm_broadcast.Handle = MagicMock()

        cls.mock_loguru = MagicMock()

        class MockWorkerProc:
            def __init__(self, vllm_config, local_rank, rank, distributed_init_method, input_shm_handle):
                self.vllm_config = vllm_config
                self.local_rank = local_rank
                self.rank = rank
                self.distributed_init_method = distributed_init_method
                self.input_shm_handle = input_shm_handle
                self.original_init_called = True

            def shutdown(self):
                self.shutdown_called = True

        class MockMultiprocExecutor:
            def execute_dummy_batch(self):
                self.execute_dummy_batch_called = True

        cls.mock_multiproc_executor = MagicMock()
        cls.mock_multiproc_executor.WorkerProc = MockWorkerProc
        cls.mock_multiproc_executor.MultiprocExecutor = MockMultiprocExecutor

        cls.mock_vllm_execute_stat = MagicMock()
        cls.mock_vllm_execute_stat.vllm_output_statics = MagicMock()

        cls.modules_patcher = patch.dict('sys.modules', {
            'vllm': cls.mock_vllm,
            'vllm.utils': cls.mock_vllm_utils,
            'vllm.config': cls.mock_vllm_config,
            'vllm.distributed.device_communicators.shm_broadcast': cls.mock_shm_broadcast,
            'vllm.v1.executor.multiproc_executor': cls.mock_multiproc_executor,
            'loguru': cls.mock_loguru,
            'agentic_rl.runner.infer_adapter.vllm.patch.comm.vllm_execute_stat': cls.mock_vllm_execute_stat,
        })
        cls.modules_patcher.start()

    @classmethod
    def _import_module_under_test(cls):
        """Import the module under test after mocks are set up"""
        test_file_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.abspath(os.path.join(test_file_dir, '..', '..', '..', '..', '..', '..', '..'))
        sys.path.append(project_root)

        spec = importlib.util.spec_from_file_location(
            'patch_multiproc_executor',
            os.path.join(project_root, 'agentic_rl', 'runner', 'infer_adapter', 'vllm', 'patch', 'patch_0_10_2', 'patch_multiproc_executor.py')
        )
        cls.patch_multiproc_executor = importlib.util.module_from_spec(spec)
        sys.modules['patch_multiproc_executor'] = cls.patch_multiproc_executor
        spec.loader.exec_module(cls.patch_multiproc_executor)

    @classmethod
    def _cleanup_mocks(cls):
        """Clean up mock patches"""
        cls.modules_patcher.stop()

    def setUp(self):
        """Set up test environment"""
        self.mock_vllm_execute_stat.vllm_output_statics.reset_mock()
        self.mock_loguru.logger.info.reset_mock()

        self.mock_vllm_config_instance = MagicMock()
        self.mock_input_shm_handle = MagicMock()

    def test_worker_proc_init(self):
        """Test worker_proc_init function"""
        worker_proc = self.mock_multiproc_executor.WorkerProc(
            self.mock_vllm_config_instance,
            0,
            0,
            "tcp://localhost:1234",
            self.mock_input_shm_handle
        )

        self.assertTrue(worker_proc.original_init_called)
        self.mock_vllm_execute_stat.vllm_output_statics.set_process_name.assert_called_once_with("test_process")

    def test_shutdown_patch(self):
        """Test shutdown_patch function"""
        worker_proc = self.mock_multiproc_executor.WorkerProc(
            self.mock_vllm_config_instance,
            0,
            0,
            "tcp://localhost:1234",
            self.mock_input_shm_handle
        )

        worker_proc.shutdown()

        self.assertTrue(worker_proc.shutdown_called)
        self.mock_vllm_execute_stat.vllm_output_statics.write_stats_tofile.assert_called_once()

    def test_execute_dummy_batch_patch_sleeping(self):
        """Test execute_dummy_batch_patch function when engine is sleeping"""
        multiproc_executor = self.mock_multiproc_executor.MultiprocExecutor()

        multiproc_executor.is_sleeping = True
        multiproc_executor.output_rank = 0
        multiproc_executor.collective_rpc = MagicMock()

        multiproc_executor.execute_dummy_batch()

        self.mock_loguru.logger.info.assert_called_once_with("Engine is currently sleeping, skipping dummy batch execution.")
        multiproc_executor.collective_rpc.assert_not_called()

    def test_execute_dummy_batch_patch_not_sleeping(self):
        """Test execute_dummy_batch_patch function when engine is not sleeping"""
        multiproc_executor = self.mock_multiproc_executor.MultiprocExecutor()

        multiproc_executor.is_sleeping = False
        multiproc_executor.output_rank = 0
        multiproc_executor.collective_rpc = MagicMock()

        multiproc_executor.execute_dummy_batch()

        self.mock_loguru.logger.info.assert_not_called()
        multiproc_executor.collective_rpc.assert_called_once_with("execute_dummy_batch", unique_reply_rank=0)


if __name__ == '__main__':
    unittest.main()
