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

import importlib.util
import os
import sys
import unittest
from unittest.mock import patch, MagicMock


class TestStatTimeUtil(unittest.TestCase):
    """Test the StatTimeUtil class"""

    @classmethod
    def setUpClass(cls):
        cls._setup_mocks()
        cls._import_module_under_test()

    @classmethod
    def tearDownClass(cls):
        cls._cleanup_mocks()

    @classmethod
    def _setup_mocks(cls):
        cls.mock_torch = MagicMock()
        cls.mock_torch.npu = MagicMock()
        cls.mock_torch.npu.synchronize = MagicMock()

        cls.mock_socket = MagicMock()
        cls.mock_socket.gethostname.return_value = "test-host"
        cls.mock_socket.gethostbyname.return_value = "127.0.0.1"

        cls.mock_pd = MagicMock()
        cls.mock_df = MagicMock()
        cls.mock_pd.DataFrame.return_value = cls.mock_df
        cls.mock_df.set_index.return_value = cls.mock_df
        cls.mock_df.transpose.return_value = cls.mock_df
        cls.mock_df.reset_index.return_value = cls.mock_df
        cls.mock_df.rename.return_value = cls.mock_df
        cls.mock_df.to_csv = MagicMock()

        cls.mock_vllm_logger = MagicMock()
        cls.mock_vllm_logger.warn = MagicMock()

        cls.mock_vllm = MagicMock()
        cls.mock_vllm_logger_module = MagicMock()
        cls.mock_vllm_logger_module.logger = cls.mock_vllm_logger

        cls.modules_patcher = patch.dict('sys.modules', {
            'torch': cls.mock_torch,
            'socket': cls.mock_socket,
            'pandas': cls.mock_pd,
            'vllm': cls.mock_vllm,
            'vllm.logger': cls.mock_vllm_logger_module,
        })
        cls.modules_patcher.start()

    @classmethod
    def _import_module_under_test(cls):
        test_file_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.abspath(os.path.join(test_file_dir, '..', '..', '..', '..', '..', '..', '..'))
        module_path = os.path.join(project_root, 'agentic_rl', 'runner', 'infer_adapter', 'vllm', 'patch', 'comm', 'vllm_execute_stat.py')
        spec = importlib.util.spec_from_file_location('vllm_execute_stat', module_path)
        cls.stat_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.stat_module)
        cls.StatTimeUtil = cls.stat_module.StatTimeUtil
        cls.StatPhase = cls.stat_module.StatPhase

    @classmethod
    def _cleanup_mocks(cls):
        cls.modules_patcher.stop()

    def setUp(self):
        self.time_util = self.StatTimeUtil()

    @patch('time.time', return_value=123456789.0)
    def test_init(self, mock_time):
        """Test initialization"""
        time_util = self.StatTimeUtil()
        mock_time.assert_called_once()
        self.assertEqual(time_util.last_time, 123456789.0)

    @patch('time.time')
    def test_get_duration_with_npu(self, mock_time):
        """Test time measurement with NPU synchronization"""
        self.time_util.last_time = 123456789.0
        mock_time.return_value = 123456790.0

        duration = self.time_util.get_duration(is_npu_exist=True)

        self.mock_torch.npu.synchronize.assert_called_once()
        self.assertEqual(duration, 1000.0)
        self.assertEqual(self.time_util.last_time, 123456790.0)

    @patch('time.time')
    def test_get_duration_without_npu(self, mock_time):
        """Test time measurement without NPU synchronization"""
        self.mock_torch.npu.synchronize.reset_mock()
        self.time_util.last_time = 123456789.0
        mock_time.return_value = 123456789.5

        duration = self.time_util.get_duration(is_npu_exist=False)

        self.assertEqual(duration, 500.0)
        self.assertEqual(self.time_util.last_time, 123456789.5)


class TestStatPhase(unittest.TestCase):
    """Test the StatPhase enum"""

    @classmethod
    def setUpClass(cls):
        cls._setup_mocks()
        cls._import_module_under_test()

    @classmethod
    def tearDownClass(cls):
        cls._cleanup_mocks()

    @classmethod
    def _setup_mocks(cls):
        cls.mock_torch = MagicMock()
        cls.mock_torch.npu = MagicMock()
        cls.mock_torch.npu.synchronize = MagicMock()

        cls.mock_socket = MagicMock()
        cls.mock_socket.gethostname.return_value = "test-host"
        cls.mock_socket.gethostbyname.return_value = "127.0.0.1"

        cls.mock_pd = MagicMock()
        cls.mock_df = MagicMock()
        cls.mock_pd.DataFrame.return_value = cls.mock_df
        cls.mock_df.set_index.return_value = cls.mock_df
        cls.mock_df.transpose.return_value = cls.mock_df
        cls.mock_df.reset_index.return_value = cls.mock_df
        cls.mock_df.rename.return_value = cls.mock_df
        cls.mock_df.to_csv = MagicMock()

        cls.mock_vllm_logger = MagicMock()
        cls.mock_vllm_logger.warn = MagicMock()

        cls.mock_vllm = MagicMock()
        cls.mock_vllm_logger_module = MagicMock()
        cls.mock_vllm_logger_module.logger = cls.mock_vllm_logger

        cls.modules_patcher = patch.dict('sys.modules', {
            'torch': cls.mock_torch,
            'socket': cls.mock_socket,
            'pandas': cls.mock_pd,
            'vllm': cls.mock_vllm,
            'vllm.logger': cls.mock_vllm_logger_module,
        })
        cls.modules_patcher.start()

    @classmethod
    def _import_module_under_test(cls):
        test_file_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.abspath(os.path.join(test_file_dir, '..', '..', '..', '..', '..', '..', '..'))
        module_path = os.path.join(project_root, 'agentic_rl', 'runner', 'infer_adapter', 'vllm', 'patch', 'comm', 'vllm_execute_stat.py')
        spec = importlib.util.spec_from_file_location('vllm_execute_stat', module_path)
        cls.stat_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.stat_module)
        cls.StatPhase = cls.stat_module.StatPhase

    @classmethod
    def _cleanup_mocks(cls):
        cls.modules_patcher.stop()

    def test_enum_values(self):
        """Test that enum values are correctly generated"""
        phase_values = [phase.value for phase in self.StatPhase]
        expected_values = list(range(0, len(self.StatPhase)))
        self.assertEqual(phase_values, expected_values)

    def test_enum_names(self):
        """Test that enum names are correct"""
        expected_names = [
            'step_start_time', 'step_finished_time',
            'prepare_input_time', 'aclgraph_dispatcher_time', 'forward_time',
            'kvconnectoroutput_time', 'post_process_time', 'pop_captured_sync_time',
            'step_total_time', 'step_inter_time',
            'forward_init_metadata_time', 'forward_embedding_time',
            'forward_alllayers_time', 'forward_last_norm_time',
            'forward_metadata_unpadding_time',
            'post_process_compute_logits_time', 'post_process_sampler_time',
            'post_process_other_time',
            'with_prefill', 'attn_state', 'batch_num', 'num_actual_tokens',
            'seq_lens', 'is_dummy_run', 'is_profiling'
        ]

        phase_names = [phase.name for phase in self.StatPhase]
        for name in expected_names:
            self.assertIn(name, phase_names)


class TestVllmOutputStatics(unittest.TestCase):
    """Test the _VllmOutputStatics class"""

    @classmethod
    def setUpClass(cls):
        cls._setup_mocks()

    @classmethod
    def tearDownClass(cls):
        cls._cleanup_mocks()

    @classmethod
    def _setup_mocks(cls):
        cls.mock_torch = MagicMock()
        cls.mock_torch.npu = MagicMock()
        cls.mock_torch.npu.synchronize = MagicMock()

        cls.mock_socket = MagicMock()
        cls.mock_socket.gethostname.return_value = "test-host"
        cls.mock_socket.gethostbyname.return_value = "127.0.0.1"

        cls.mock_pd = MagicMock()
        cls.mock_df = MagicMock()
        cls.mock_pd.DataFrame.return_value = cls.mock_df
        cls.mock_df.set_index.return_value = cls.mock_df
        cls.mock_df.transpose.return_value = cls.mock_df
        cls.mock_df.reset_index.return_value = cls.mock_df
        cls.mock_df.rename.return_value = cls.mock_df
        cls.mock_df.to_csv = MagicMock()

        cls.mock_vllm_logger = MagicMock()
        cls.mock_vllm_logger.warn = MagicMock()

        cls.mock_vllm = MagicMock()
        cls.mock_vllm_logger_module = MagicMock()
        cls.mock_vllm_logger_module.logger = cls.mock_vllm_logger

        cls.modules_patcher = patch.dict('sys.modules', {
            'torch': cls.mock_torch,
            'socket': cls.mock_socket,
            'pandas': cls.mock_pd,
            'vllm': cls.mock_vllm,
            'vllm.logger': cls.mock_vllm_logger_module,
        })
        cls.modules_patcher.start()

    @classmethod
    def _cleanup_mocks(cls):
        cls.modules_patcher.stop()

    def _import_module_with_env(self):
        test_file_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.abspath(os.path.join(test_file_dir, '..', '..', '..', '..', '..', '..', '..'))
        module_path = os.path.join(project_root, 'agentic_rl', 'runner', 'infer_adapter', 'vllm', 'patch', 'comm', 'vllm_execute_stat.py')
        spec = importlib.util.spec_from_file_location('vllm_execute_stat', module_path)
        stat_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(stat_module)
        return stat_module

    def setUp(self):
        self.original_enable_vllm_stat = os.environ.get('ENABLE_VLLM_STAT')
        self.original_vllm_stat_suffix = os.environ.get('VLLM_STAT_SVAE_PATH_SUFFIX')

        os.environ['ENABLE_VLLM_STAT'] = 'true'
        os.environ['VLLM_STAT_SVAE_PATH_SUFFIX'] = 'test_suffix'

        stat_module = self._import_module_with_env()
        self.statics = stat_module._VllmOutputStatics()
        self.StatPhase = stat_module.StatPhase

    def tearDown(self):
        if self.original_enable_vllm_stat is None:
            if 'ENABLE_VLLM_STAT' in os.environ:
                del os.environ['ENABLE_VLLM_STAT']
        else:
            os.environ['ENABLE_VLLM_STAT'] = self.original_enable_vllm_stat

        if self.original_vllm_stat_suffix is None:
            if 'VLLM_STAT_SVAE_PATH_SUFFIX' in os.environ:
                del os.environ['VLLM_STAT_SVAE_PATH_SUFFIX']
        else:
            os.environ['VLLM_STAT_SVAE_PATH_SUFFIX'] = self.original_vllm_stat_suffix

    def test_init(self):
        """Test initialization"""
        self.assertEqual(len(self.statics.stats), 1)
        self.assertIn('title', self.statics.stats)
        self.assertEqual(self.statics.last_step_finish_time, 0)
        self.assertEqual(self.statics.step_start_time, 0)
        self.assertEqual(self.statics.local_ip, "127.0.0.1")
        self.assertIn("127.0.0.1", self.statics.process_name)
        self.assertIn("IntegratedWorker", self.statics.process_name)
        self.assertEqual(self.statics.cur_requestid_stepid, "")
        self.assertEqual(self.statics.base_path, "logs/vllm_statistic")

    def test_set_process_name(self):
        """Test setting the process name"""
        new_process_name = "TestWorker"
        self.statics.set_process_name(new_process_name)

        self.assertIn("127.0.0.1", self.statics.process_name)
        self.assertIn(new_process_name, self.statics.process_name)
        self.assertIn("pid=", self.statics.process_name)

    def test_set_cur_requestid_stepid(self):
        """Test setting request ID and step ID"""
        requestid_stepid = "req_123/step_456"
        start_time = 123456789.0

        self.statics.set_cur_requestid_stepid(requestid_stepid, start_time)

        self.assertEqual(self.statics.step_start_time, start_time)
        self.assertIn("127.0.0.1", self.statics.cur_requestid_stepid)
        self.assertIn("IntegratedWorker", self.statics.cur_requestid_stepid)
        self.assertIn(requestid_stepid, self.statics.cur_requestid_stepid)

        self.assertIn(self.statics.cur_requestid_stepid, self.statics.stats)
        self.assertEqual(len(self.statics.stats[self.statics.cur_requestid_stepid]), len(self.StatPhase))
        self.assertEqual(self.statics.stats[self.statics.cur_requestid_stepid][self.StatPhase.is_profiling.value],
                         False)
        self.assertEqual(self.statics.stats[self.statics.cur_requestid_stepid][self.StatPhase.is_dummy_run.value],
                         False)
        self.assertEqual(self.statics.stats[self.statics.cur_requestid_stepid][self.StatPhase.step_start_time.value],
                         start_time)

    def test_set_cur_requestid_stepid_with_inter_time(self):
        """Test setting request ID and step ID (with inter-step time)"""
        self.statics.last_step_finish_time = 123456788.0
        requestid_stepid = "req_123/step_456"
        start_time = 123456789.0

        self.statics.set_cur_requestid_stepid(requestid_stepid, start_time)

        self.assertEqual(self.statics.stats[self.statics.cur_requestid_stepid][self.StatPhase.step_inter_time.value],
                         1000.0)

    def test_set_step_finish_time(self):
        """Test setting step finish time"""
        self.statics.set_cur_requestid_stepid("req_123/step_456", 123456789.0)

        finish_time = 123456790.0
        self.statics.set_step_finish_time(finish_time)

        self.assertEqual(self.statics.last_step_finish_time, finish_time)
        self.assertEqual(self.statics.stats[self.statics.cur_requestid_stepid][self.StatPhase.step_total_time.value],
                         1000.0)
        self.assertEqual(self.statics.stats[self.statics.cur_requestid_stepid][self.StatPhase.step_finished_time.value],
                         finish_time)

    def test_add_stat(self):
        """Test adding statistics data"""
        self.statics.set_cur_requestid_stepid("req_123/step_456", 123456789.0)

        duration = 123.45
        self.statics.add_stat(self.StatPhase.forward_time, duration)

        self.assertEqual(self.statics.stats[self.statics.cur_requestid_stepid][self.StatPhase.forward_time.value],
                         duration)

    def test_add_stat_new_request(self):
        """Test adding statistics for a new request"""
        duration = 123.45
        self.statics.add_stat(self.StatPhase.forward_time, duration)

        self.assertIn(self.statics.cur_requestid_stepid, self.statics.stats)
        self.assertEqual(self.statics.stats[self.statics.cur_requestid_stepid][self.StatPhase.forward_time.value],
                         duration)

    def test_set_stat(self):
        """Test setting statistics data"""
        self.statics.set_cur_requestid_stepid("req_123/step_456", 123456789.0)

        value = 42
        self.statics.set_stat(self.StatPhase.batch_num, value)

        self.assertEqual(self.statics.stats[self.statics.cur_requestid_stepid][self.StatPhase.batch_num.value], value)

    def test_set_stat_new_request(self):
        """Test setting statistics for a new request"""
        value = 42
        self.statics.set_stat(self.StatPhase.batch_num, value)

        self.assertIn(self.statics.cur_requestid_stepid, self.statics.stats)
        self.assertEqual(self.statics.stats[self.statics.cur_requestid_stepid][self.StatPhase.batch_num.value], value)

    @patch('builtins.print')
    def test_print_stats(self, mock_print):
        """Test printing statistics"""
        self.statics.set_cur_requestid_stepid("req_123/step_456", 123456789.0)

        self.statics.print_stats()

        mock_print.assert_called()

    @patch('builtins.print')
    def test_print_one_stats(self, mock_print):
        """Test printing statistics for a single request"""
        self.statics.set_cur_requestid_stepid("req_123/step_456", 123456789.0)

        self.statics.print_one_stats()

        mock_print.assert_called_once()
        self.assertIn("_VllmOutputStatics cur_request-id_step-id:", mock_print.call_args[0][0])

    @patch('os.makedirs')
    @patch('os.path.exists', return_value=False)
    def test_write_stats_tofile(self, mock_exists, mock_makedirs):
        """Test writing statistics to file"""
        mock_date = MagicMock()
        mock_date.strftime.side_effect = ["20230402", "2023-04-02 12:34:56"]

        with patch('datetime.datetime') as mock_datetime_class:
            mock_datetime_class.now.return_value = mock_date

            stat_module = self._import_module_with_env()
            statics = stat_module._VllmOutputStatics()
            StatPhase = stat_module.StatPhase

            statics.set_cur_requestid_stepid("req_123/step_456", 123456789.0)
            statics.set_step_finish_time(123456790.0)

            statics.write_stats_tofile()

            mock_exists.assert_called_once()
            exists_call_args = mock_exists.call_args[0][0]
            self.assertIn('20230402', exists_call_args)
            self.assertIn('test_suffix', exists_call_args)
            self.assertIn('logs', exists_call_args)
            self.assertIn('vllm_statistic', exists_call_args)

            mock_makedirs.assert_called_once()
            makedirs_call_args = mock_makedirs.call_args[0][0]
            self.assertIn('20230402', makedirs_call_args)
            self.assertIn('test_suffix', makedirs_call_args)
            self.assertIn('logs', makedirs_call_args)
            self.assertIn('vllm_statistic', makedirs_call_args)
            self.assertEqual(mock_makedirs.call_args[1]['exist_ok'], True)

            self.mock_pd.DataFrame.assert_called_once()
            self.mock_df.set_index.assert_called_once_with('title')
            self.mock_df.transpose.assert_called_once()
            self.mock_df.reset_index.assert_called_once()
            self.mock_df.rename.assert_called_once_with(columns={'index': 'title'})
            self.mock_df.to_csv.assert_called_once()

    def test_clear(self):
        """Test clearing statistics"""
        self.statics.set_cur_requestid_stepid("req_123/step_456", 123456789.0)
        self.statics.set_step_finish_time(123456790.0)

        self.statics.clear()

        self.assertEqual(len(self.statics.stats), 1)
        self.assertIn('title', self.statics.stats)
        self.assertEqual(self.statics.last_step_finish_time, 0)
        self.assertEqual(self.statics.step_start_time, 0)
        self.assertEqual(self.statics.cur_requestid_stepid, "")

    @patch.dict(os.environ, {"ENABLE_VLLM_STAT": "false"})
    def test_disabled_operations(self):
        """Test operations when statistics are disabled"""
        stat_module = self._import_module_with_env()

        self.assertFalse(stat_module.is_vllm_statistic)

        statics = stat_module._VllmOutputStatics()
        StatPhase = stat_module.StatPhase

        statics.set_cur_requestid_stepid("req_123/step_456", 123456789.0)
        statics.set_step_finish_time(123456790.0)
        statics.add_stat(StatPhase.forward_time, 123.45)
        statics.set_stat(StatPhase.batch_num, 42)

        self.assertIn("title", statics.stats)
        self.assertIn(statics.cur_requestid_stepid, statics.stats)

        self.assertEqual(statics.stats[statics.cur_requestid_stepid][StatPhase.forward_time.value], 123.45)
        self.assertEqual(statics.stats[statics.cur_requestid_stepid][StatPhase.batch_num.value], 42)

        with patch.object(stat_module, 'pd') as mock_pd_module:
            statics.write_stats_tofile()
            mock_pd_module.DataFrame.assert_not_called()

        with patch('builtins.print') as mock_print:
            statics.print_stats()
            mock_print.assert_not_called()


class TestGlobalVariables(unittest.TestCase):
    """Test global variables and singleton instance"""

    @classmethod
    def setUpClass(cls):
        cls._setup_mocks()

    @classmethod
    def tearDownClass(cls):
        cls._cleanup_mocks()

    @classmethod
    def _setup_mocks(cls):
        cls.mock_torch = MagicMock()
        cls.mock_torch.npu = MagicMock()
        cls.mock_torch.npu.synchronize = MagicMock()

        cls.mock_socket = MagicMock()
        cls.mock_socket.gethostname.return_value = "test-host"
        cls.mock_socket.gethostbyname.return_value = "127.0.0.1"

        cls.mock_pd = MagicMock()
        cls.mock_df = MagicMock()
        cls.mock_pd.DataFrame.return_value = cls.mock_df
        cls.mock_df.set_index.return_value = cls.mock_df
        cls.mock_df.transpose.return_value = cls.mock_df
        cls.mock_df.reset_index.return_value = cls.mock_df
        cls.mock_df.rename.return_value = cls.mock_df
        cls.mock_df.to_csv = MagicMock()

        cls.mock_vllm_logger = MagicMock()
        cls.mock_vllm_logger.warn = MagicMock()

        cls.mock_vllm = MagicMock()
        cls.mock_vllm_logger_module = MagicMock()
        cls.mock_vllm_logger_module.logger = cls.mock_vllm_logger

        cls.modules_patcher = patch.dict('sys.modules', {
            'torch': cls.mock_torch,
            'socket': cls.mock_socket,
            'pandas': cls.mock_pd,
            'vllm': cls.mock_vllm,
            'vllm.logger': cls.mock_vllm_logger_module,
        })
        cls.modules_patcher.start()

    @classmethod
    def _cleanup_mocks(cls):
        cls.modules_patcher.stop()

    def _import_module_with_env(self):
        test_file_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.abspath(os.path.join(test_file_dir, '..', '..', '..', '..', '..', '..', '..'))
        module_path = os.path.join(project_root, 'agentic_rl', 'runner', 'infer_adapter', 'vllm', 'patch', 'comm', 'vllm_execute_stat.py')
        spec = importlib.util.spec_from_file_location('vllm_execute_stat', module_path)
        stat_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(stat_module)
        return stat_module

    def test_is_vllm_statistic_default(self):
        """Test default statistics status"""
        if 'ENABLE_VLLM_STAT' in os.environ:
            del os.environ['ENABLE_VLLM_STAT']

        stat_module = self._import_module_with_env()

        self.assertFalse(stat_module.is_vllm_statistic)

    @patch.dict(os.environ, {"ENABLE_VLLM_STAT": "true"})
    def test_is_vllm_statistic_enabled(self):
        """Test statistics enabled status"""
        stat_module = self._import_module_with_env()

        self.assertTrue(stat_module.is_vllm_statistic)

    @patch.dict(os.environ, {"ENABLE_VLLM_STAT": "false"})
    def test_is_vllm_statistic_disabled(self):
        """Test statistics disabled status"""
        stat_module = self._import_module_with_env()

        self.assertFalse(stat_module.is_vllm_statistic)

    def test_vllm_stat_save_path_suffix_default(self):
        """Test default save path suffix"""
        if 'VLLM_STAT_SVAE_PATH_SUFFIX' in os.environ:
            del os.environ['VLLM_STAT_SVAE_PATH_SUFFIX']

        stat_module = self._import_module_with_env()

        self.assertEqual(stat_module.vllm_stat_save_path_suffix, " ")

    @patch.dict(os.environ, {"VLLM_STAT_SVAE_PATH_SUFFIX": "test_suffix"})
    def test_vllm_stat_save_path_suffix_custom(self):
        """Test custom save path suffix"""
        stat_module = self._import_module_with_env()

        self.assertEqual(stat_module.vllm_stat_save_path_suffix, "test_suffix")

    def test_vllm_output_statics_singleton(self):
        """Test singleton instance"""
        stat_module = self._import_module_with_env()

        instance1 = stat_module.vllm_output_statics
        instance2 = stat_module.vllm_output_statics

        self.assertIs(instance1, instance2)


if __name__ == '__main__':
    unittest.main()
