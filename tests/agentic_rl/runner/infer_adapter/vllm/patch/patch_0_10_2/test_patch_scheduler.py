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

        cls.modules_patcher = patch.dict('sys.modules', {
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
            'agentic_rl.runner.infer_adapter.vllm.patch.comm.scheduler_stat': cls.mock_scheduler_stat,
            'vllm_ascend': MagicMock(),
            'vllm_ascend.patch': MagicMock(),
            'vllm_ascend.patch.platform': MagicMock(),
            'vllm_ascend.patch.worker': MagicMock(),
        })
        cls.modules_patcher.start()

    @classmethod
    def _import_module_under_test(cls):
        """Import the module under test after mocks are set up"""
        test_file_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.abspath(os.path.join(test_file_dir, '..', '..', '..', '..', '..', '..', '..'))
        sys.path.append(project_root)

        spec = importlib.util.spec_from_file_location(
            'patch_scheduler',
            os.path.join(project_root, 'agentic_rl', 'runner', 'infer_adapter', 'vllm', 'patch', 'patch_0_10_2',
                         'patch_scheduler.py')
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

    def test_scheduler_init(self):
        """Test scheduler_init function"""
        mock_scheduler = MagicMock()

        self.patch_scheduler.scheduler_init(
            mock_scheduler,
            self.vllm_config,
            self.kv_cache_config,
            self.structured_output_manager,
            self.mm_registry,
            include_finished_set=True,
            log_stats=True
        )

        self.assertTrue(hasattr(mock_scheduler, 'req_stats'))
        self.mock_scheduler_stat.RequestStats.assert_called_once()

    def test_update_after_schedule_patch(self):
        """Test update_after_schedule_patch function"""
        self.scheduler.requests["req1"] = self.request1
        self.scheduler.requests["req2"] = self.request2

        self.scheduler.req_stats = MagicMock()

        self.patch_scheduler.update_after_schedule_patch(self.scheduler, self.scheduler_output)

        self.assertEqual(self.request1.num_computed_tokens, 1)
        self.assertEqual(self.request2.num_computed_tokens, 21)

        self.scheduler.req_stats.stat_schedule.assert_any_call("req1")
        self.scheduler.req_stats.stat_schedule.assert_any_call("req2")
        self.assertEqual(self.scheduler.req_stats.stat_schedule.call_count, 2)

        self.scheduler._free_encoder_inputs.assert_not_called()

    def test_update_after_schedule_patch_with_encoder_inputs(self):
        """Test update_after_schedule_patch with encoder inputs"""
        scheduler_output = MagicMock()
        scheduler_output.num_scheduled_tokens = {"req1": 1}

        self.request1.has_encoder_inputs = True
        self.scheduler.requests["req1"] = self.request1

        self.scheduler.req_stats = MagicMock()

        self.patch_scheduler.update_after_schedule_patch(self.scheduler, scheduler_output)

        self.scheduler._free_encoder_inputs.assert_called_once_with(self.request1)

    def test_update_request_with_output_patch_first_token(self):
        """Test update_request_with_output_patch with first output token"""
        self.scheduler.req_stats = MagicMock()

        new_token_ids = [42]
        result_ids, stopped = self.patch_scheduler.update_request_with_output_patch(self.scheduler, self.request1, new_token_ids)

        self.assertEqual(result_ids, [42])
        self.assertEqual(self.request1.num_output_tokens, 1)
        self.assertEqual(self.request1.output_token_ids, [42])
        self.assertFalse(stopped)

        self.scheduler.req_stats.stat_prefill_done.assert_called_once_with("req1")

        self.scheduler.req_stats.stat_finish.assert_not_called()

    def test_update_request_with_output_patch_stopped(self):
        """Test update_request_with_output_patch when request is stopped"""
        self.mock_vllm_core_sched_utils.check_stop.return_value = True

        self.scheduler.req_stats = MagicMock()

        new_token_ids = [42, 100, 200]
        result_ids, stopped = self.patch_scheduler.update_request_with_output_patch(self.scheduler, self.request1, new_token_ids)

        self.assertEqual(result_ids, [42])
        self.assertTrue(stopped)
        self.assertEqual(self.request1.num_output_tokens, 1)

        self.scheduler.req_stats.stat_prefill_done.assert_called_once_with("req1")

        self.scheduler.req_stats.stat_finish.assert_called_once_with("req1", 10, 1)

    def test_add_request_patch(self):
        """Test add_request_patch function"""
        self.scheduler.req_stats = MagicMock()

        self.patch_scheduler.add_request_patch(self.scheduler, self.request1)

        self.scheduler.waiting.add_request.assert_called_once_with(self.request1)
        self.assertIn("req1", self.scheduler.requests)
        self.assertEqual(self.scheduler.requests["req1"], self.request1)

        self.scheduler.req_stats.stat_add.assert_called_once_with("req1")

        self.request1.record_event.assert_not_called()

    def test_add_request_patch_with_log_stats(self):
        """Test add_request_patch function with log_stats=True"""
        self.scheduler.req_stats = MagicMock()
        self.scheduler.log_stats = True

        self.patch_scheduler.add_request_patch(self.scheduler, self.request1)

        self.request1.record_event.assert_called_once_with(self.mock_vllm_engine.EngineCoreEventType.QUEUED)

    def test_reset_prefix_cache_patch(self):
        """Test reset_prefix_cache_patch function"""
        self.scheduler.req_stats = MagicMock()

        self.scheduler.kv_cache_manager.reset_prefix_cache.return_value = True

        result = self.patch_scheduler.reset_prefix_cache_patch(self.scheduler)

        self.scheduler.req_stats.print.assert_called_once()
        self.scheduler.kv_cache_manager.reset_prefix_cache.assert_called_once()
        self.assertTrue(result)


if __name__ == '__main__':
    unittest.main()
