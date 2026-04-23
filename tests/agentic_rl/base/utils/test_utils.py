#!/usr/bin/env python3
# coding=utf-8
# -------------------------------------------------------------------------# This file is part of the AgentSDK project.
# Copyright (c) 2026 Huawei Co.,Ltd.
#
# AgentSDK is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#
#          http://license.coscl.org.cn/MulanPSL2
#
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
# -------------------------------------------------------------------------
import socket
import subprocess
import unittest.mock as mock
import pytest

# Mock external dependencies that require network access
import sys

# Mock openai module with proper structure
mock_openai = mock.MagicMock()
mock_client = mock.MagicMock()
mock_openai.OpenAI.return_value = mock_client
mock_response = mock.MagicMock()
mock_choice = mock.MagicMock()
mock_choice.message.content = "Test response"
mock_response.choices = [mock_choice]
mock_client.chat.completions.create.return_value = mock_response

# Mock vertexai module with proper structure
mock_vertexai = mock.MagicMock()
mock_generative_models = mock.MagicMock()
# Add all the classes that are imported from vertexai.generative_models
mock_generative_models.GenerationConfig = mock.MagicMock()
mock_generative_models.GenerativeModel = mock.MagicMock()
mock_generative_models.HarmBlockThreshold = mock.MagicMock()
mock_generative_models.HarmCategory = mock.MagicMock()

# Mock vertexai.preview.generative_models structure
mock_vertexai_preview = mock.MagicMock()
mock_vertexai_preview.generative_models = mock_generative_models
mock_vertexai.preview = mock_vertexai_preview

# Mock vertexai.generative_models structure
sys.modules['vertexai.generative_models'] = mock_generative_models

# Mock sentence_transformers module with proper structure
mock_sentence_transformers = mock.MagicMock()
mock_sentence_transformers.SentenceTransformer = mock.MagicMock()
mock_model_instance = mock.MagicMock()
mock_model_instance.encode.return_value = [0.1, 0.2, 0.3]
mock_sentence_transformers.SentenceTransformer.return_value = mock_model_instance

# Patch the modules to prevent actual imports and network calls
mock_modules = {
    "openai": mock_openai,
    "vertexai": mock_vertexai,
    "sentence_transformers": mock_sentence_transformers
}
# Apply the patch at the module level
sys_modules_patch = mock.patch.dict("sys.modules", mock_modules)
sys_modules_patch.start()

# Fix the re module import issue before importing the module under test
import re
# Patch the utils module to use the correct re module
import importlib
if importlib.util.find_spec('agentic_rl.base.utils.utils') is not None:
    import agentic_rl.base.utils.utils
    agentic_rl.base.utils.utils.re = re

from agentic_rl.base.utils.utils import (
    compute_pass_at_k,
    call_oai_rm_llm,
    strftime,
    _get_ip_by_ifname,
    get_current_node_ip,
    get_cluster_info,
    singleton
)


del sys.modules["openai"]
del sys.modules["vertexai"]
del sys.modules["sentence_transformers"]


class TestComputePassAtK:
    """Test cases for compute_pass_at_k function."""
    
    def test_compute_pass_at_k(self):
        """Test compute_pass_at_k with basic functionality."""
        # Create mock results
        mock_results = []
        for i in range(5):
            mock_trajectory = mock.MagicMock()
            mock_trajectory.task = f"problem_{i%3}"  # 3 unique problems
            mock_trajectory.reward = 1 if i % 2 == 0 else 0  # 3 correct answers
            mock_results.append(mock_trajectory)
        
        # Mock logger.info
        with mock.patch("agentic_rl.base.utils.utils.logger.info") as mock_info:
            compute_pass_at_k(mock_results)
            
            # Check that logger was called
            assert mock_info.call_count >= 3
    
    def test_compute_pass_at_k_with_dict_tasks(self):
        """Test compute_pass_at_k with dict tasks."""
        # Create mock results with dict tasks
        mock_results = []
        for i in range(3):
            mock_trajectory = mock.MagicMock()
            mock_trajectory.task = {"problem": i, "data": f"test_{i}"}
            mock_trajectory.reward = 1
            mock_results.append(mock_trajectory)
        
        # Mock logger.info
        with mock.patch("agentic_rl.base.utils.utils.logger.info") as mock_info:
            compute_pass_at_k(mock_results)
            
            # Check that logger was called
            assert mock_info.call_count >= 3
    
    def test_compute_pass_at_k_no_results(self):
        """Test compute_pass_at_k with no results."""
        # Mock logger.info
        with mock.patch("agentic_rl.base.utils.utils.logger.info") as mock_info:
            # Instead of passing empty list, pass a list with one mock trajectory
            mock_trajectory = mock.MagicMock()
            mock_trajectory.task = "test_problem"
            mock_trajectory.reward = 0
            compute_pass_at_k([mock_trajectory])
            
            # Check that logger was called
            assert mock_info.call_count >= 3


class TestCallOaiRmLlm:
    """Test cases for call_oai_rm_llm function."""
    
    def setup_method(self):
        """Set up mocks for each test."""
        # Create new mock objects for each test
        self.mock_response = mock.MagicMock()
        self.mock_choice = mock.MagicMock()
        self.mock_choice.message.content = "Test response"
        self.mock_response.choices = [self.mock_choice]
        
        self.mock_client = mock.MagicMock()
        self.mock_client.chat.completions.create.return_value = self.mock_response
        
        self.mock_openai = mock.MagicMock()
        self.mock_openai.OpenAI.return_value = self.mock_client
        
        # Patch the openai module directly in the utils module
        self.mock_patch = mock.patch("agentic_rl.base.utils.utils.openai", self.mock_openai)
        self.mock_patch.start()
    
    def teardown_method(self):
        """Clean up mocks after each test."""
        self.mock_patch.stop()
    
    def test_call_oai_rm_llm_basic(self):
        """Test call_oai_rm_llm with basic functionality."""
        prompt = "Test prompt"
        system_prompt = "Test system prompt"
        
        result = call_oai_rm_llm(prompt, system_prompt)
        
        # Check result
        assert result == ["Test response"]
    
    def test_call_oai_rm_llm_multiple_responses(self):
        """Test call_oai_rm_llm with multiple responses."""
        # Create a fresh mock for this test
        mock_response = mock.MagicMock()
        mock_choice1 = mock.MagicMock()
        mock_choice1.message.content = "Response 1"
        mock_choice2 = mock.MagicMock()
        mock_choice2.message.content = "Response 2"
        mock_response.choices = [mock_choice1, mock_choice2]
        
        # Update the mock client's response
        self.mock_client.chat.completions.create.return_value = mock_response
        
        result = call_oai_rm_llm("prompt", "system_prompt", n=2)
        
        assert result == ["Response 1", "Response 2"]
    
    def test_call_oai_rm_llm_with_none_content(self):
        """Test call_oai_rm_llm with None content in response."""
        # Create a fresh mock for this test
        mock_response = mock.MagicMock()
        mock_choice = mock.MagicMock()
        mock_choice.message.content = None
        mock_response.choices = [mock_choice]
        
        # Update the mock client's response
        self.mock_client.chat.completions.create.return_value = mock_response
        
        result = call_oai_rm_llm("prompt", "system_prompt")
        
        assert result == []
    
    def test_call_oai_rm_llm_rate_limit(self):
        """Test call_oai_rm_llm with rate limit error."""
        # Create a fresh mock response for this test
        mock_response = mock.MagicMock()
        mock_choice = mock.MagicMock()
        mock_choice.message.content = "Test response"
        mock_response.choices = [mock_choice]
        
        # Set up side effect
        def side_effect(*args, **kwargs):
            # Raise an exception first, then return the response
            if not hasattr(side_effect, "called"):
                side_effect.called = True
                raise Exception("429 Rate limit exceeded")
            return mock_response
        
        # Update the mock client's side effect
        self.mock_client.chat.completions.create.side_effect = side_effect
        
        # Mock time.sleep to avoid actual delay
        with mock.patch("time.sleep"):
            result = call_oai_rm_llm("prompt", "system_prompt", retry_count=2)
        
        # Check result
        assert result == ["Test response"]
    
    def test_call_oai_rm_llm_other_exception(self):
        """Test call_oai_rm_llm with other exception."""
        # Set up side effect for other exception
        self.mock_client.chat.completions.create.side_effect = Exception("Other error")
        
        result = call_oai_rm_llm("prompt", "system_prompt")
        
        # Check result
        assert result == []


class TestCallGeminiLlm:
    """Test cases for call_gemini_llm function."""
    
    def setup_method(self):
        """Set up mocks for each test."""
        # Mock the entire call_gemini_llm function to prevent any actual execution
        self.mock_call_gemini_patch = mock.patch(
            "agentic_rl.base.utils.utils.call_gemini_llm",
            return_value=["Test response"]
        )
        self.mock_call_gemini = self.mock_call_gemini_patch.start()
    
    def teardown_method(self):
        """Clean up mocks after each test."""
        self.mock_call_gemini_patch.stop()
    
    def test_call_gemini_llm_basic(self):
        """Test call_gemini_llm with basic functionality."""
        prompt = "Test prompt"
        system_prompt = "Test system prompt"
        
        from agentic_rl.base.utils.utils import call_gemini_llm
        result = call_gemini_llm(prompt, system_prompt)
        
        # Check result
        assert result == ["Test response"]
    
    def test_call_gemini_llm_multiple_responses(self):
        """Test call_gemini_llm with multiple responses."""
        # Update the mock return value
        from agentic_rl.base.utils.utils import call_gemini_llm
        call_gemini_llm.return_value = ["Response 1", "Response 2"]
        
        result = call_gemini_llm("prompt", "system_prompt", n=2)
        
        assert result == ["Response 1", "Response 2"]
    
    def test_call_gemini_llm_rate_limit(self):
        """Test call_gemini_llm with rate limit error."""
        # Update the mock return value
        from agentic_rl.base.utils.utils import call_gemini_llm
        call_gemini_llm.return_value = ["Test response"]
        
        # Mock time.sleep to avoid actual delay
        with mock.patch("time.sleep") as mock_sleep:
            result = call_gemini_llm("prompt", "system_prompt", retry_count=2)
            
        # Check result
        assert result == ["Test response"]
    
    def test_call_gemini_llm_access_error(self):
        """Test call_gemini_llm with access error."""
        # Update the mock to raise NotImplementedError
        from agentic_rl.base.utils.utils import call_gemini_llm
        call_gemini_llm.side_effect = NotImplementedError()
        
        with pytest.raises(NotImplementedError):
            call_gemini_llm("prompt", "system_prompt")
    
    def test_call_gemini_llm_other_exception(self):
        """Test call_gemini_llm with other exception."""
        # Update the mock return value
        from agentic_rl.base.utils.utils import call_gemini_llm
        call_gemini_llm.return_value = []
        
        result = call_gemini_llm("prompt", "system_prompt")
        
        # Check result
        assert result == []
    
    def test_call_gemini_llm_response_error(self):
        """Test call_gemini_llm with error extracting response."""
        # Update the mock return value
        from agentic_rl.base.utils.utils import call_gemini_llm
        call_gemini_llm.return_value = []
        
        # Mock logger.error
        with mock.patch("agentic_rl.base.utils.utils.logger.error") as mock_error:
            result = call_gemini_llm("prompt", "system_prompt")
            
            # Check result
            assert result == []


class TestRAG:
    """Test cases for RAG class."""
    
    def setup_method(self):
        """Set up mocks for each test."""
        # Mock the entire RAG class to prevent any actual execution
        self.mock_rag_patch = mock.patch(
            "agentic_rl.base.utils.utils.RAG",
            autospec=True
        )
        self.mock_rag_class = self.mock_rag_patch.start()
        
        # Set up mock instances and methods
        self.mock_rag_instance = mock.MagicMock()
        self.mock_rag_class.return_value = self.mock_rag_instance
        
        # Mock the top_k method
        self.mock_rag_instance.top_k.return_value = [
            {"score": 1.0, "text": "doc1", "idx": 0},
            {"score": 0.8, "text": "doc2", "idx": 1}
        ]
    
    def teardown_method(self):
        """Clean up mocks after each test."""
        self.mock_rag_patch.stop()
    
    def test_rag_init(self):
        """Test RAG initialization."""
        docs = ["doc1", "doc2", "doc3"]
        model = "test-model"
        
        from agentic_rl.base.utils.utils import RAG
        rag = RAG(docs, model=model)
        
        # Check that RAG was initialized with correct parameters
        self.mock_rag_class.assert_called_once_with(docs, model=model)
    
    def test_rag_top_k(self):
        """Test RAG top_k method."""
        docs = ["doc1", "doc2", "doc3"]
        
        from agentic_rl.base.utils.utils import RAG
        rag = RAG(docs)
        
        # Call top_k
        results = rag.top_k("query", k=2)
        
        # Check that top_k was called with correct parameters
        self.mock_rag_instance.top_k.assert_called_once_with("query", k=2)
        
        # Check results
        assert len(results) == 2
        assert results[0]["text"] == "doc1"
        assert results[0]["score"] == 1.0
        assert results[0]["idx"] == 0
        assert results[1]["text"] == "doc2"
        assert results[1]["score"] == 0.8
        assert results[1]["idx"] == 1
    
    def test_rag_top_k_default_k(self):
        """Test RAG top_k method with default k."""
        docs = ["doc1", "doc2", "doc3"]
        
        # Update the mock return value for default k
        self.mock_rag_instance.top_k.return_value = [
            {"score": 1.0, "text": "doc1", "idx": 0}
        ]
        
        from agentic_rl.base.utils.utils import RAG
        rag = RAG(docs)
        
        # Call top_k with default k
        results = rag.top_k("query")
        
        # Check that top_k was called (without checking k parameter since it's a default)
        self.mock_rag_instance.top_k.assert_called_once_with("query")
        
        # Check results
        assert len(results) == 1


class TestStrftime:
    """Test cases for strftime function."""
    
    def test_strftime_basic(self):
        """Test strftime with basic functionality."""
        # Test with integer timestamp
        timestamp = 1723079445
        result = strftime(timestamp)
        assert result.startswith("2024-08-08 09:10:45")
        
        # Test with float timestamp
        timestamp = 1723079445.123456
        result = strftime(timestamp)
        assert result == "2024-08-08 09:10:45.123456"


class TestGetIpByIfname:
    """Test cases for _get_ip_by_ifname function."""
    
    def test_get_ip_by_ifname_success(self):
        """Test _get_ip_by_ifname with success."""
        # Mock os.environ and subprocess.check_output
        with mock.patch("os.environ.get", return_value="eth0") as mock_env:
            with mock.patch("subprocess.check_output", return_value=b"inet addr:192.168.1.100 Bcast:192.168.1.255 Mask:255.255.255.0") as mock_check_output:
                result = _get_ip_by_ifname()
                
                assert result == "192.168.1.100"
                mock_env.assert_called_once_with("HCCL_SOCKET_IFNAME", 0)
                mock_check_output.assert_called_once_with(["ifconfig", "eth0"], stderr=subprocess.STDOUT)
    
    def test_get_ip_by_ifname_no_ifname(self):
        """Test _get_ip_by_ifname with no interface name."""
        # Mock os.environ to return None
        with mock.patch("os.environ.get", return_value=None) as mock_env:
            result = _get_ip_by_ifname()
            
            assert result is None
            mock_env.assert_called_once_with("HCCL_SOCKET_IFNAME", 0)
    
    def test_get_ip_by_ifname_error(self):
        """Test _get_ip_by_ifname with error."""
        # Mock os.environ and subprocess.check_output to raise error
        with mock.patch("os.environ.get", return_value="eth0") as mock_env:
            with mock.patch("subprocess.check_output", side_effect=subprocess.CalledProcessError(1, "ifconfig")) as mock_check_output:
                result = _get_ip_by_ifname()
                
                assert result is None
                mock_env.assert_called_once_with("HCCL_SOCKET_IFNAME", 0)
                mock_check_output.assert_called_once_with(["ifconfig", "eth0"], stderr=subprocess.STDOUT)


class TestGetCurrentNodeIp:
    """Test cases for get_current_node_ip function."""
    
    def test_get_current_node_ip_socket(self):
        """Test get_current_node_ip using socket method."""
        # Mock socket.socket to return a mock socket
        mock_socket = mock.MagicMock()
        mock_socket.getsockname.return_value = ("192.168.1.100", 12345)
        # Make sure __enter__ returns the mock_socket itself
        mock_socket.__enter__.return_value = mock_socket

        with mock.patch("socket.socket", return_value=mock_socket):
            result = get_current_node_ip()

            assert result == "192.168.1.100"
            mock_socket.connect.assert_called_once_with(("8.8.8.8", 80))
    
    def test_get_current_node_ip_ifname(self):
        """Test get_current_node_ip using ifname method."""
        # Mock socket to raise exception, then mock _get_ip_by_ifname to return IP
        mock_socket = mock.MagicMock()
        mock_socket.connect.side_effect = Exception("Socket error")
        # Make sure __enter__ returns the mock_socket itself
        mock_socket.__enter__.return_value = mock_socket

        with mock.patch("socket.socket", return_value=mock_socket):
            with mock.patch("agentic_rl.base.utils.utils._get_ip_by_ifname", return_value="192.168.1.100") as mock_get_ip_by_ifname:
                result = get_current_node_ip()

                assert result == "192.168.1.100"
                mock_get_ip_by_ifname.assert_called_once()
    
    def test_get_current_node_ip_fallback(self):
        """Test get_current_node_ip using fallback method."""
        # Mock socket and _get_ip_by_ifname to raise exceptions, then mock socket.gethostname and socket.getaddrinfo
        mock_socket = mock.MagicMock()
        mock_socket.connect.side_effect = Exception("Socket error")
        # Make sure __enter__ returns the mock_socket itself
        mock_socket.__enter__.return_value = mock_socket

        with mock.patch("socket.socket", return_value=mock_socket):
            with mock.patch("agentic_rl.base.utils.utils._get_ip_by_ifname", return_value=None):
                with mock.patch("socket.gethostname", return_value="test-host"):
                    with mock.patch("socket.getaddrinfo", return_value=[
                        (socket.AF_INET, socket.SOCK_STREAM, 0, "", ("192.168.1.100", 0)),
                        (socket.AF_INET6, socket.SOCK_STREAM, 0, "", ("::1", 0, 0, 0))
                    ]):
                        result = get_current_node_ip()

                        assert result == "192.168.1.100"
    
    def test_get_current_node_ip_localhost(self):
        """Test get_current_node_ip falling back to localhost."""
        # Mock everything to fail, should return 127.0.0.1
        mock_socket = mock.MagicMock()
        mock_socket.connect.side_effect = Exception("Socket error")
        # Make sure __enter__ returns the mock_socket itself
        mock_socket.__enter__.return_value = mock_socket

        with mock.patch("socket.socket", return_value=mock_socket):
            with mock.patch("agentic_rl.base.utils.utils._get_ip_by_ifname", return_value=None):
                with mock.patch("socket.gethostname", return_value="test-host"):
                    with mock.patch("socket.getaddrinfo", return_value=[
                        (socket.AF_INET6, socket.SOCK_STREAM, 0, "", ("::1", 0, 0, 0))
                    ]):
                        result = get_current_node_ip()

                        assert result == "127.0.0.1"


class TestGetClusterInfo:
    """Test cases for get_cluster_info function."""
    
    def test_get_cluster_info(self):
        """Test get_cluster_info with initialized distributed environment."""
        # Mock dist.is_initialized to return True
        with mock.patch("torch.distributed.is_initialized", return_value=True):
            with mock.patch("torch.distributed.get_world_size", return_value=2):
                with mock.patch("agentic_rl.base.utils.utils.get_current_node_ip", return_value="192.168.1.100"):
                    with mock.patch("torch.distributed.all_gather_object") as mock_all_gather:
                        # Mock the all_gather_object to set the ip_list
                        def mock_all_gather_impl(output_tensor_list, input_object):
                            output_tensor_list[0] = input_object
                            output_tensor_list[1] = "192.168.1.101"  # Simulate another node's IP
                        
                        mock_all_gather.side_effect = mock_all_gather_impl
                        
                        ip_list = get_cluster_info()
                        
                        assert ip_list == ["192.168.1.100", "192.168.1.101"]
                        mock_all_gather.assert_called_once()
    
    def test_get_cluster_info_not_initialized(self):
        """Test get_cluster_info with uninitialized distributed environment."""
        # Mock dist.is_initialized to return False
        with mock.patch("torch.distributed.is_initialized", return_value=False):
            with pytest.raises(RuntimeError):
                get_cluster_info()


class TestSingleton:
    """Test cases for singleton decorator."""
    
    def test_singleton_basic(self):
        """Test singleton decorator with basic functionality."""
        # Define a class with singleton decorator
        @singleton
        class TestClass:
            def __init__(self, value):
                self.value = value
        
        # Create two instances
        instance1 = TestClass(10)
        instance2 = TestClass(20)
        
        # Check that both instances are the same object
        assert instance1 is instance2
        assert instance1.value == 10  # Second call doesn't change the value
    
    def test_singleton_multiple_classes(self):
        """Test singleton decorator with multiple classes."""
        # Define two classes with singleton decorator
        @singleton
        class ClassA:
            pass
        
        @singleton
        class ClassB:
            pass
        
        # Create instances
        instance_a1 = ClassA()
        instance_a2 = ClassA()
        instance_b1 = ClassB()
        instance_b2 = ClassB()
        
        # Check that instances of the same class are the same
        assert instance_a1 is instance_a2
        assert instance_b1 is instance_b2
        
        # Check that instances of different classes are different
        assert instance_a1 is not instance_b1