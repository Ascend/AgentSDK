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


class TestPatchLLMDataDistCMgrConnector(unittest.TestCase):
    """Test patch_llmdatadist_c_mgr_connector.py module"""

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
        """Setup mock objects for llm_datadist, msgspec, zmq, vllm, vllm_ascend"""
        cls.mock_msg_encoder = MagicMock()
        cls.mock_msg_encoder.encode.return_value = b"encoded-message"

        cls.mock_msg_decoder = MagicMock()

        cls.mock_sock = MagicMock()

        cls.mock_llm_datadist = MagicMock()
        cls.mock_llm_datadist.BlocksCacheKey = MagicMock()
        cls.mock_llm_datadist.LLMException = Exception

        cls.mock_msgspec = MagicMock()
        cls.mock_msgspec.msgpack.Encoder = MagicMock(return_value=cls.mock_msg_encoder)
        cls.mock_msgspec.msgpack.Decoder = MagicMock(return_value=cls.mock_msg_decoder)

        cls.mock_zmq = MagicMock()

        cls.mock_vllm = MagicMock()
        cls.mock_vllm.utils = MagicMock()
        cls.mock_vllm.utils.get_ip = MagicMock(return_value="127.0.0.1")
        cls.mock_vllm.utils.logger = MagicMock()

        cls.mock_vllm_ascend = MagicMock()
        cls.mock_vllm_ascend.envs = MagicMock()
        cls.mock_vllm_ascend.envs.VLLM_ASCEND_LLMDD_RPC_PORT = 5555
        cls.mock_vllm_ascend.envs.VLLM_ASCEND_LLMDD_RPC_IP = "127.0.0.1"
        cls.mock_vllm_ascend.distributed = MagicMock()
        cls.mock_vllm_ascend.distributed.llmdatadist_c_mgr_connector = MagicMock()

        class MockLLMDataDistCMgrEvent:
            ReqForMetadata = "ReqForMetadata"
            ReqForFinished = "ReqForFinished"

            def __new__(cls, value):
                return value

        cls.mock_vllm_ascend.distributed.llmdatadist_c_mgr_connector.LLMDataDistCMgrEvent = MockLLMDataDistCMgrEvent

        class MockLLMDataDistCMgrAgentMetadata:
            def __init__(self, **kwargs):
                self.__dict__.update(kwargs)

        cls.mock_vllm_ascend.distributed.llmdatadist_c_mgr_connector.LLMDataDistCMgrAgentMetadata = MockLLMDataDistCMgrAgentMetadata

        class MockLLMDataDistCMgrConnectorWorker:
            pass

        cls.mock_vllm_ascend.distributed.llmdatadist_c_mgr_connector.LLMDataDistCMgrConnectorWorker = MockLLMDataDistCMgrConnectorWorker

        def mock_zmq_ctx(socket_type, url):
            mock_ctx = MagicMock()
            mock_ctx.__enter__.return_value = cls.mock_sock
            return mock_ctx

        cls.mock_vllm_ascend.distributed.llmdatadist_c_mgr_connector.zmq_ctx = mock_zmq_ctx

        cls.modules_patcher = patch.dict('sys.modules', {
            'llm_datadist': cls.mock_llm_datadist,
            'msgspec': cls.mock_msgspec,
            'zmq': cls.mock_zmq,
            'vllm': cls.mock_vllm,
            'vllm.utils': cls.mock_vllm.utils,
            'vllm_ascend': cls.mock_vllm_ascend,
            'vllm_ascend.envs': cls.mock_vllm_ascend.envs,
            'vllm_ascend.distributed': cls.mock_vllm_ascend.distributed,
            'vllm_ascend.distributed.llmdatadist_c_mgr_connector': cls.mock_vllm_ascend.distributed.llmdatadist_c_mgr_connector,
        })
        cls.modules_patcher.start()

    @classmethod
    def _import_module_under_test(cls):
        """Import the module under test after mocks are set up"""
        test_file_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.abspath(os.path.join(test_file_dir, '..', '..', '..', '..', '..', '..', '..'))
        sys.path.append(project_root)

        spec = importlib.util.spec_from_file_location(
            'patch_llmdatadist_c_mgr_connector',
            os.path.join(project_root, 'agentic_rl', 'runner', 'infer_adapter', 'vllm', 'patch', 'patch_0_10_2', 'patch_llmdatadist_c_mgr_connector.py')
        )
        cls.patch_llmdatadist_c_mgr_connector = importlib.util.module_from_spec(spec)
        sys.modules['patch_llmdatadist_c_mgr_connector'] = cls.patch_llmdatadist_c_mgr_connector
        spec.loader.exec_module(cls.patch_llmdatadist_c_mgr_connector)

    @classmethod
    def _cleanup_mocks(cls):
        """Clean up mock patches"""
        cls.modules_patcher.stop()

    def setUp(self):
        """Set up test environment"""
        self.mock_msg_encoder.encode.reset_mock()
        self.mock_msg_decoder.decode.reset_mock()
        self.mock_sock.reset_mock()
        self.mock_sock.send_multipart.reset_mock()
        self.mock_sock.send.reset_mock()
        self.mock_sock.recv.reset_mock()

        self.mock_self = MagicMock()
        self.mock_self.local_agent_metadata = MagicMock()
        self.mock_self.local_agent_metadata.cluster_id = "test-cluster-123"
        self.mock_self.local_dp_rank = 0
        self.mock_self.tp_rank = 0
        self.mock_self.tp_size = 4
        self.mock_self.thread_lock = MagicMock()
        self.mock_self.finished_reqs = set()
        self.mock_self.linked_cluster = {}
        self.mock_self.use_mla = False
        self.mock_self.cache_manager = MagicMock()
        self.mock_self.cache = MagicMock()
        self.mock_self.connect_to_remote_agent = MagicMock(return_value="remote-cluster-456")
        self.mock_self.add_remote_agent = MagicMock()

        self.mock_event = MagicMock()

    def test_listen_for_agent_metadata_req_patch_with_metadata_request(self):
        """Test listen_for_agent_metadata_req_patch with ReqForMetadata event"""
        self.mock_msg_decoder.decode.return_value = ("ReqForMetadata", {"cluster_id": "remote-cluster-456"})

        def mock_recv_multipart():
            self.mock_event.set.assert_called_once()
            yield [b"client-id", b"", b"request-message"]
            raise KeyboardInterrupt

        self.mock_sock.recv_multipart = mock_recv_multipart().__next__

        try:
            self.patch_llmdatadist_c_mgr_connector.listen_for_agent_metadata_req_patch(self.mock_self, self.mock_event)
        except KeyboardInterrupt:
            pass

        self.mock_event.set.assert_called_once()
        self.mock_msg_encoder.encode.assert_called_once_with(self.mock_self.local_agent_metadata)
        self.mock_sock.send_multipart.assert_called_once_with((b"client-id", b"", b"encoded-message"))
        self.mock_self.add_remote_agent.assert_called_once()

    def test_listen_for_agent_metadata_req_patch_with_finished_request(self):
        """Test listen_for_agent_metadata_req_patch with ReqForFinished event"""
        self.mock_msg_decoder.decode.return_value = ("ReqForFinished", ["request-123"])

        def mock_recv_multipart():
            self.mock_event.set.assert_called_once()
            yield [b"client-id", b"", b"request-message"]
            raise KeyboardInterrupt

        self.mock_sock.recv_multipart = mock_recv_multipart().__next__

        try:
            self.patch_llmdatadist_c_mgr_connector.listen_for_agent_metadata_req_patch(self.mock_self, self.mock_event)
        except KeyboardInterrupt:
            pass

        self.mock_event.set.assert_called_once()
        self.mock_self.thread_lock.__enter__.assert_called_once()
        self.mock_self.thread_lock.__exit__.assert_called_once()
        self.assertIn("request-123", self.mock_self.finished_reqs)
        self.mock_sock.send_multipart.assert_called_once_with((b"client-id", b"", b"receiving decode finished"))

    def test_listen_for_agent_metadata_req_patch_with_unexpected_event(self):
        """Test listen_for_agent_metadata_req_patch with unexpected event"""
        self.mock_msg_decoder.decode.return_value = ("UnexpectedEvent", {})

        def mock_recv_multipart():
            self.mock_event.set.assert_called_once()
            yield [b"client-id", b"", b"request-message"]

        self.mock_sock.recv_multipart = mock_recv_multipart().__next__

        with self.assertRaises(RuntimeError):
            self.patch_llmdatadist_c_mgr_connector.listen_for_agent_metadata_req_patch(self.mock_self, self.mock_event)

        self.mock_event.set.assert_called_once()

    def test_listen_for_agent_metadata_req_patch_with_unrecognized_data(self):
        """Test listen_for_agent_metadata_req_patch with unrecognized data"""
        self.mock_msg_decoder.decode.return_value = ("ReqForMetadata", {"some_other_key": "value"})

        def mock_recv_multipart():
            self.mock_event.set.assert_called_once()
            yield [b"client-id", b"", b"request-message"]
            raise KeyboardInterrupt

        self.mock_sock.recv_multipart = mock_recv_multipart().__next__

        try:
            self.patch_llmdatadist_c_mgr_connector.listen_for_agent_metadata_req_patch(self.mock_self, self.mock_event)
        except KeyboardInterrupt:
            pass

        self.mock_event.set.assert_called_once()
        self.mock_sock.send_multipart.assert_not_called()

    def test_send_finish_to_remote_success(self):
        """Test send_finish_to_remote with successful send"""
        self.mock_sock.send = MagicMock()
        self.mock_sock.recv.return_value = b"response"

        self.patch_llmdatadist_c_mgr_connector.send_finish_to_remote(self.mock_self, "192.168.1.100", [5555, 5556], "request-123")

        self.assertEqual(self.mock_msg_encoder.encode.call_count, 2)
        self.assertEqual(self.mock_sock.send.call_count, 2)
        self.assertEqual(self.mock_sock.recv.call_count, 2)

    def test_send_finish_to_remote_failure(self):
        """Test send_finish_to_remote with failure"""
        self.mock_sock.send = MagicMock(side_effect=Exception("Connection error"))

        self.patch_llmdatadist_c_mgr_connector.send_finish_to_remote(self.mock_self, "192.168.1.100", [5555], "request-123")

        self.mock_msg_encoder.encode.assert_called_once()
        self.mock_sock.send.assert_called_once()
        self.mock_sock.recv.assert_not_called()

    def test_read_blocks_patch_no_mla_success(self):
        """Test _read_blocks_patch without MLA (mixed precision)"""
        self.mock_self.use_mla = False

        self.patch_llmdatadist_c_mgr_connector._read_blocks_patch(
            self.mock_self,
            [0, 1, 2],
            [10, 11, 12],
            "192.168.1.100",
            5555,
            "engine-1",
            "request-123",
            "4"
        )

        self.mock_self.connect_to_remote_agent.assert_called_once_with("192.168.1.100", 5555)
        self.mock_self.cache_manager.pull_blocks.assert_called_once()
        self.mock_self.add_remote_agent.assert_not_called()
        self.assertIn("request-123", self.mock_self.finished_reqs)

    def test_read_blocks_patch_with_mla_success(self):
        """Test _read_blocks_patch with MLA (mixed precision)"""
        self.mock_self.use_mla = True
        self.mock_self.cache = [MagicMock(), MagicMock()]

        self.patch_llmdatadist_c_mgr_connector._read_blocks_patch(
            self.mock_self,
            [0, 1, 2],
            [10, 11, 12],
            "192.168.1.100",
            5555,
            "engine-1",
            "request-123",
            "4"
        )

        self.mock_self.connect_to_remote_agent.assert_called_once_with("192.168.1.100", 5555)
        self.assertEqual(self.mock_self.cache_manager.pull_blocks.call_count, 2)
        self.assertIn("request-123", self.mock_self.finished_reqs)

    def test_read_blocks_patch_with_more_remote_blocks(self):
        """Test _read_blocks_patch with more remote blocks than local blocks"""
        self.mock_self.use_mla = False

        self.patch_llmdatadist_c_mgr_connector._read_blocks_patch(
            self.mock_self,
            [0, 1],
            [10, 11, 12, 13],
            "192.168.1.100",
            5555,
            "engine-1",
            "request-123",
            "4"
        )

        self.mock_self.connect_to_remote_agent.assert_called_once()
        self.mock_self.cache_manager.pull_blocks.assert_called_once()
        call_args = self.mock_self.cache_manager.pull_blocks.call_args
        self.assertEqual(call_args[0][3], [0, 1])
        self.assertEqual(call_args[0][2], [12, 13])

    def test_read_blocks_patch_with_no_local_blocks(self):
        """Test _read_blocks_patch with no local blocks"""
        self.mock_self.use_mla = False

        self.patch_llmdatadist_c_mgr_connector._read_blocks_patch(
            self.mock_self,
            [],
            [10, 11, 12],
            "192.168.1.100",
            5555,
            "engine-1",
            "request-123",
            "4"
        )

        self.mock_self.connect_to_remote_agent.assert_called_once_with("192.168.1.100", 5555)
        self.mock_self.cache_manager.pull_blocks.assert_not_called()
        self.assertNotIn("request-123", self.mock_self.finished_reqs)

    def test_read_blocks_patch_with_pull_blocks_type_error(self):
        """Test _read_blocks_patch with TypeError during pull_blocks"""
        self.mock_self.use_mla = False
        self.mock_self.cache_manager.pull_blocks.side_effect = TypeError("Type error")

        with self.assertRaises(RuntimeError):
            self.patch_llmdatadist_c_mgr_connector._read_blocks_patch(
                self.mock_self,
                [0, 1, 2],
                [10, 11, 12],
                "192.168.1.100",
                5555,
                "engine-1",
                "request-123",
                "4"
            )

    def test_read_blocks_patch_with_pull_blocks_value_error(self):
        """Test _read_blocks_patch with ValueError during pull_blocks"""
        self.mock_self.use_mla = False
        self.mock_self.cache_manager.pull_blocks.side_effect = ValueError("Value error")

        with self.assertRaises(RuntimeError):
            self.patch_llmdatadist_c_mgr_connector._read_blocks_patch(
                self.mock_self,
                [0, 1, 2],
                [10, 11, 12],
                "192.168.1.100",
                5555,
                "engine-1",
                "request-123",
                "4"
            )

    def test_read_blocks_patch_with_llm_exception(self):
        """Test _read_blocks_patch with LLMException during pull_blocks"""
        self.mock_self.use_mla = False
        self.mock_self.cache_manager.pull_blocks.side_effect = self.mock_llm_datadist.LLMException("LLM error")

        with self.assertRaises(RuntimeError):
            self.patch_llmdatadist_c_mgr_connector._read_blocks_patch(
                self.mock_self,
                [0, 1, 2],
                [10, 11, 12],
                "192.168.1.100",
                5555,
                "engine-1",
                "request-123",
                "4"
            )

    def test_patch_applied(self):
        """Test that the patches are correctly applied"""
        listen_for_agent_metadata_req_patch = self.patch_llmdatadist_c_mgr_connector.listen_for_agent_metadata_req_patch
        _read_blocks_patch = self.patch_llmdatadist_c_mgr_connector._read_blocks_patch

        from vllm_ascend.distributed.llmdatadist_c_mgr_connector import LLMDataDistCMgrConnectorWorker

        self.assertEqual(LLMDataDistCMgrConnectorWorker.listen_for_agent_metadata_req, listen_for_agent_metadata_req_patch)
        self.assertEqual(LLMDataDistCMgrConnectorWorker._read_blocks, _read_blocks_patch)


if __name__ == '__main__':
    unittest.main()
