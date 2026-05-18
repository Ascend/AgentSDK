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


class MockRequest:
    def __init__(self, request_id, num_prompt_tokens=10):
        self.request_id = request_id
        self.num_computed_tokens = 0
        self.num_cached_tokens = 0
        self.num_prompt_tokens = num_prompt_tokens
        self.num_output_tokens = 0
        self.has_encoder_inputs = False
        self.output_token_ids = []
        self.record_event = MagicMock()

    def append_output_token_ids(self, token_id):
        self.output_token_ids.append(token_id)
        self.num_output_tokens += 1


class MockScheduler:
    def __init__(self):
        self.requests = {}
        self.waiting = MagicMock()
        self.log_stats = False
        self.max_model_len = 100
        self.kv_cache_manager = MagicMock()


class TestPatchScheduler(unittest.TestCase):
    """Test patch_scheduler.py module"""

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
        """Setup mock objects for vllm and vllm_ascend"""
        cls.mock_scheduler_stat = MagicMock()
        cls.mock_scheduler_stat.RequestStats = MagicMock()

        mock_vllm = MagicMock()
        mock_vllm_config = MagicMock()
        mock_vllm_multimodal = MagicMock()
        mock_vllm_multimodal.MULTIMODAL_REGISTRY = MagicMock()
        mock_vllm_multimodal.MultiModalRegistry = MagicMock()
        mock_vllm_core_sched_output = MagicMock()
        mock_vllm_core_sched_utils = MagicMock()
        mock_vllm_engine = MagicMock()
        mock_vllm_engine.EngineCoreEventType = MagicMock()
        mock_vllm_engine.EngineCoreEventType.QUEUED = "QUEUED"
        mock_vllm_kv_cache = MagicMock()
        mock_vllm_request = MagicMock()
        mock_vllm_structured_output = MagicMock()
        mock_vllm_core_sched_scheduler = MagicMock()

        class MockSchedulerClass:
            def __init__(self, *args, **kwargs):
                pass

        mock_vllm_core_sched_scheduler.Scheduler = MockSchedulerClass

        cls.mock_vllm_core_sched_utils = mock_vllm_core_sched_utils
        cls.mock_vllm_core_sched_utils.check_stop = MagicMock(return_value=False)
        cls.mock_vllm_core_sched_utils.remove_all = MagicMock()

        cls.mock_vllm_engine = mock_vllm_engine

        cls.modules_patcher = patch.dict(
            'sys.modules',
            {
                'vllm': mock_vllm,
                'vllm.config': mock_vllm_config,
                'vllm.multimodal': mock_vllm_multimodal,
                'vllm.v1': MagicMock(),
                'vllm.v1.core': MagicMock(),
                'vllm.v1.core.sched': MagicMock(),
                'vllm.v1.core.sched.output': mock_vllm_core_sched_output,
                'vllm.v1.core.sched.utils': mock_vllm_core_sched_utils,
                'vllm.v1.engine': mock_vllm_engine,
                'vllm.v1.kv_cache_interface': mock_vllm_kv_cache,
                'vllm.v1.request': mock_vllm_request,
                'vllm.v1.structured_output': mock_vllm_structured_output,
                'vllm.v1.core.sched.scheduler': mock_vllm_core_sched_scheduler,
                'aura.runner.infer_adapter.vllm.patch.comm.scheduler_stat': cls.mock_scheduler_stat,
                'vllm_ascend': MagicMock(),
                'vllm_ascend.patch': MagicMock(),
                'vllm_ascend.patch.platform': MagicMock(),
                'vllm_ascend.patch.worker': MagicMock(),
            },
        )
        cls.modules_patcher.start()

    @classmethod
    def _import_module_under_test(cls):
        """Import the module under test after mocks are set up"""
        test_file_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.abspath(os.path.join(test_file_dir, '..', '..', '..', '..', '..', '..', '..'))
        sys.path.append(project_root)

        spec = importlib.util.spec_from_file_location(
            'patch_scheduler',
            os.path.join(
                project_root, 'aura', 'runner', 'infer_adapter', 'vllm', 'patch', 'patch_0_10_2', 'patch_scheduler.py'
            ),
        )
        cls.patch_scheduler = importlib.util.module_from_spec(spec)
        sys.modules['patch_scheduler'] = cls.patch_scheduler
        spec.loader.exec_module(cls.patch_scheduler)

    @classmethod
    def _cleanup_mocks(cls):
        """Clean up mock patches"""
        cls.modules_patcher.stop()

    def setUp(self):
        """Set up test environment"""
        self.scheduler = MockScheduler()
        self.scheduler._free_encoder_inputs = MagicMock()

        self.request1 = MockRequest("req1")
        self.request2 = MockRequest("req2", num_prompt_tokens=20)
        self.request2.num_cached_tokens = 20
        self.request2.num_computed_tokens = 20

        self.scheduler_output = MagicMock()
        self.scheduler_output.num_scheduled_tokens = {"req1": 1, "req2": 1}

        self.vllm_config = MagicMock()
        self.kv_cache_config = MagicMock()
        self.structured_output_manager = MagicMock()
        self.mm_registry = MagicMock()

        self.mock_scheduler_stat.RequestStats.reset_mock()
        self.mock_vllm_core_sched_utils.check_stop.reset_mock()
        self.mock_vllm_core_sched_utils.check_stop.return_value = False


if __name__ == '__main__':
    unittest.main()
