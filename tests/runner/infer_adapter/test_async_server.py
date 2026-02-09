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
import types
from types import ModuleType
import pytest

from agentic_rl.configs.agentic_rl_config import GenConfig, AgenticRLConfig


def _import_after_mocking(monkeypatch, module_name, preinstall):
    """Helper to install mock modules and import target module cleanly."""
    for fullname, module in preinstall.items():
        monkeypatch.setitem(sys.modules, fullname, module)
    monkeypatch.delitem(sys.modules, module_name, raising=False)
    return importlib.import_module(module_name)


@pytest.fixture
def async_server_mod(monkeypatch):
    """Fixture providing the async_server module with mocked dependencies."""
    # Mock Ray
    ray_mod = ModuleType("ray")
    ray_mod.get = lambda x, **kwargs: [(i, f"node_{i}") for i in range(8)]
    ray_mod.kill = lambda actor: None
    
    def _remote(obj=None, **kwargs):
        return (lambda x: x) if obj is None else obj
    
    ray_mod.remote = _remote
    
    # Mock Ray utilities and exceptions
    util_mod = ModuleType("ray.util")
    strategies_mod = ModuleType("ray.util.scheduling_strategies")
    
    class NodeAffinitySchedulingStrategy:
        def __init__(self, node_id: str, soft: bool):
            self.node_id = node_id
            self.soft = soft
    
    strategies_mod.NodeAffinitySchedulingStrategy = NodeAffinitySchedulingStrategy
    util_mod.scheduling_strategies = strategies_mod
    ray_mod.util = util_mod
    
    rex_mod = ModuleType("ray.exceptions")
    
    class RayError(Exception):
        pass
    
    class RayTaskError(RayError):
        pass
    
    class RayActorError(RayError):
        pass
    
    rex_mod.RayError = RayError
    rex_mod.RayTaskError = RayTaskError
    rex_mod.RayActorError = RayActorError
    
    # Mock mindspeed_rl
    class RayActorGroup:
        pass
    
    rg_mod = ModuleType("mindspeed_rl.workers.scheduler.launcher")
    rg_mod.RayActorGroup = RayActorGroup
    
    # Mock infer_registry
    ir_mod = ModuleType("agentic_rl.runner.infer_adapter.infer_registry")
    ir_mod.async_server_class = lambda infer_backend: None
    
    # Mock transformers
    transformers_mod = ModuleType("transformers")
    
    class AutoTokenizer:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            return object()
    
    transformers_mod.AutoTokenizer = AutoTokenizer
    
    preinstall = {
        "ray": ray_mod,
        "ray.exceptions": rex_mod,
        "ray.util": util_mod,
        "ray.util.scheduling_strategies": strategies_mod,
        "mindspeed_rl": ModuleType("mindspeed_rl"),
        "mindspeed_rl.config_cls": ModuleType("mindspeed_rl.config_cls"),
        "mindspeed_rl.workers": ModuleType("mindspeed_rl.workers"),
        "mindspeed_rl.workers.scheduler": ModuleType("mindspeed_rl.workers.scheduler"),
        "mindspeed_rl.workers.scheduler.launcher": rg_mod,
        "agentic_rl.runner.infer_adapter.infer_registry": ir_mod,
        "transformers": transformers_mod,
    }
    
    mod = _import_after_mocking(monkeypatch, "agentic_rl.runner.infer_adapter.async_server", preinstall)
    yield mod


@pytest.fixture
def valid_inputs(async_server_mod, monkeypatch):
    """Fixture providing valid initialization inputs for AsyncServerManager."""
    # Mock FileCheck to allow GenConfig initialization
    from unittest.mock import MagicMock
    mock_file_check = MagicMock()
    mock_file_check.check_data_path_is_valid = MagicMock(return_value=None)
    monkeypatch.setattr("agentic_rl.configs.agentic_rl_config.FileCheck", mock_file_check)
    
    RayActorGroup = sys.modules["mindspeed_rl.workers.scheduler.launcher"].RayActorGroup
    
    cfg = GenConfig()
    cfg.infer_tensor_parallel_size = 2
    
    agent_cfg = AgenticRLConfig()
    agent_cfg.infer_backend = "vllm"
    
    wg = RayActorGroup()
    wg.num_npus = 8
    wg.execute_async_command = lambda cmd: object()
    
    return cfg, agent_cfg, wg


@pytest.fixture
def server_stub():
    """Fixture providing a mock server instance."""
    class _Remote:
        def remote(self, *args, **kwargs):
            return object()
    
    class _Actor:
        def __init__(self):
            self.__ray_ready__ = _Remote()
            self.init_engine = _Remote()
            self.wake_up = _Remote()
            self.sleep = _Remote()
            self.get_server_address = _Remote()
    
    class _Opts:
        def remote(self, *args, **kwargs):
            return _Actor()
    
    class _ServerClass:
        def options(self, **kwargs):
            return _Opts()
    
    return _ServerClass()


class TestAsyncServerManager:
    """Test suite for AsyncServerManager class."""

    def test_init_success(self, async_server_mod, valid_inputs, server_stub, monkeypatch):
        """Test successful AsyncServerManager initialization."""
        cfg, agent_cfg, wg = valid_inputs
        monkeypatch.setenv("VLLM_DP_SIZE", "1")
        monkeypatch.setattr(
            async_server_mod, "FileCheck",
            types.SimpleNamespace(check_data_path_is_valid=lambda _: None)
        )
        monkeypatch.setattr(async_server_mod, "async_server_class", lambda infer_backend: server_stub)

        manager = async_server_mod.AsyncServerManager(cfg, agent_cfg, "/path/to/tokenizer", wg)
        
        assert manager.rollout_infer_backend == "vllm"
        assert manager.rollout_tp_size == 2
        assert manager.rollout_dp_size == 4
        assert len(manager.async_servers) == 4
        assert len(manager.server_addresses) == 4

    @pytest.mark.parametrize("invalid_param,error_match", [
        ("config", "config must be a GenConfig"),
        ("agentic_rl_config", "agentic_rl_config must be a AgenticRLConfig"),
        ("worker_group", "worker_group must be a RayActorGroup"),
    ])
    def test_init_invalid_params(
        self, async_server_mod, valid_inputs, invalid_param, error_match
    ):
        """Test initialization with invalid parameters."""
        cfg, agent_cfg, wg = valid_inputs
        
        if invalid_param == "config":
            cfg = object()
        elif invalid_param == "agentic_rl_config":
            agent_cfg = types.SimpleNamespace(infer_backend="vllm")
        elif invalid_param == "worker_group":
            wg = types.SimpleNamespace(num_npus=8)
        
        with pytest.raises(ValueError, match=error_match):
            async_server_mod.AsyncServerManager(cfg, agent_cfg, "/path/to/tokenizer", wg)

    def test_init_invalid_tokenizer_path(self, async_server_mod, valid_inputs, monkeypatch):
        """Test initialization with invalid tokenizer path."""
        cfg, agent_cfg, wg = valid_inputs
        
        def raise_error(_):
            raise ValueError("Invalid path")

        monkeypatch.setattr(
            async_server_mod, "FileCheck",
            types.SimpleNamespace(check_data_path_is_valid=raise_error)
        )
        
        with pytest.raises(ValueError, match="Invalid path"):
            async_server_mod.AsyncServerManager(cfg, agent_cfg, "/invalid/path", wg)

    def test_init_ray_error_during_worker_info_retrieval(
        self, async_server_mod, valid_inputs, monkeypatch
    ):
        """Test initialization failure when Ray worker info retrieval fails."""
        cfg, agent_cfg, wg = valid_inputs
        RayError = sys.modules["ray.exceptions"].RayError
        ray_mod = sys.modules["ray"]
        
        ray_mod.get = lambda _, **kwargs: (_ for _ in ()).throw(RayError("Failed to get worker info"))
        
        monkeypatch.setenv("VLLM_DP_SIZE", "1")
        monkeypatch.setattr(
            async_server_mod, "FileCheck",
            types.SimpleNamespace(check_data_path_is_valid=lambda _: None)
        )
        
        with pytest.raises(RuntimeError, match="Failed to retrieve worker info"):
            async_server_mod.AsyncServerManager(cfg, agent_cfg, "/path/to/tokenizer", wg)

    def test_init_unsupported_inference_backend(
        self, async_server_mod, valid_inputs, monkeypatch
    ):
        """Test initialization with unsupported inference backend."""
        cfg, agent_cfg, wg = valid_inputs
        
        monkeypatch.setenv("VLLM_DP_SIZE", "1")
        monkeypatch.setattr(
            async_server_mod, "FileCheck",
            types.SimpleNamespace(check_data_path_is_valid=lambda _: None)
        )
        monkeypatch.setattr(async_server_mod, "async_server_class", lambda infer_backend: None)
        
        with pytest.raises(ValueError, match="Unsupported inference backend"):
            async_server_mod.AsyncServerManager(cfg, agent_cfg, "/path/to/tokenizer", wg)

    def test_init_dp_size_calculation_error(
        self, async_server_mod, valid_inputs, monkeypatch
    ):
        """Test initialization failure when DP size calculation fails."""
        cfg, agent_cfg, wg = valid_inputs
        wg.num_npus = None  # This will cause calculation to fail
        
        monkeypatch.setenv("VLLM_DP_SIZE", "1")
        monkeypatch.setattr(
            async_server_mod, "FileCheck",
            types.SimpleNamespace(check_data_path_is_valid=lambda _: None)
        )
        
        with pytest.raises(RuntimeError, match="Failed to calculate rollout_dp_size"):
            async_server_mod.AsyncServerManager(cfg, agent_cfg, "/path/to/tokenizer", wg)

    def test_init_engine_initialization_failure(
        self, async_server_mod, valid_inputs, server_stub, monkeypatch
    ):
        """Test initialization failure when engine initialization fails."""
        cfg, agent_cfg, wg = valid_inputs
        RayError = sys.modules["ray.exceptions"].RayError
        ray_mod = sys.modules["ray"]
        
        # Track ray.get calls to fail on engine init
        original_get = ray_mod.get
        call_count = [0]
        
        def mock_get(x, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                # First call for worker info
                return original_get(x, **kwargs)
            elif call_count[0] <= 9:
                # Health check calls for 4 server instances (rollout_dp_size=4) + 4 get_server_address
                return original_get(x, **kwargs)
            else:
                # Engine init call - fail here
                raise RayError("Engine initialization failed")
        
        ray_mod.get = mock_get
        
        monkeypatch.setenv("VLLM_DP_SIZE", "1")
        monkeypatch.setattr(
            async_server_mod, "FileCheck",
            types.SimpleNamespace(check_data_path_is_valid=lambda _: None)
        )
        monkeypatch.setattr(async_server_mod, "async_server_class", lambda infer_backend: server_stub)
        
        with pytest.raises(RuntimeError, match="Failed to initialize AsyncLLM engines"):
            async_server_mod.AsyncServerManager(cfg, agent_cfg, "/path/to/tokenizer", wg)

    def test_start_server_instances_max_retries_exceeded(
        self, async_server_mod, valid_inputs, monkeypatch
    ):
        """Test initialization failure when max server start retries are exceeded."""
        cfg, agent_cfg, wg = valid_inputs
        RayError = sys.modules["ray.exceptions"].RayError
        
        # Mock server that always fails health check
        class _FailingRemote:
            def remote(self, *args, **kwargs):
                raise RayError("Health check failed")
        
        class _FailingActor:
            def __init__(self):
                self.__ray_ready__ = _FailingRemote()
                self.init_engine = _FailingRemote()
        
        class _FailingOpts:
            def remote(self, *args, **kwargs):
                return _FailingActor()
        
        class _FailingServerClass:
            def options(self, **kwargs):
                return _FailingOpts()
        
        monkeypatch.setenv("VLLM_DP_SIZE", "1")
        monkeypatch.setattr(
            async_server_mod, "FileCheck",
            types.SimpleNamespace(check_data_path_is_valid=lambda _: None)
        )
        monkeypatch.setattr(async_server_mod, "async_server_class", lambda infer_backend: _FailingServerClass())
        
        with pytest.raises(RuntimeError, match="Failed to start server instances after"):
            async_server_mod.AsyncServerManager(cfg, agent_cfg, "/path/to/tokenizer", wg)

    def test_wake_up_success(self, async_server_mod, valid_inputs, server_stub, monkeypatch):
        """Test successful wake_up of all server instances."""
        cfg, agent_cfg, wg = valid_inputs

        monkeypatch.setenv("VLLM_DP_SIZE", "1")
        monkeypatch.setattr(
            async_server_mod, "FileCheck",
            types.SimpleNamespace(check_data_path_is_valid=lambda _: None)
        )
        monkeypatch.setattr(async_server_mod, "async_server_class", lambda infer_backend: server_stub)

        manager = async_server_mod.AsyncServerManager(cfg, agent_cfg, "/path/to/tokenizer", wg)
        manager.wake_up()  # Should complete without error

    def test_wake_up_failure(self, async_server_mod, valid_inputs, server_stub, monkeypatch):
        """Test wake_up failure with RayError."""
        cfg, agent_cfg, wg = valid_inputs
        RayError = sys.modules["ray.exceptions"].RayError
        ray_mod = sys.modules["ray"]

        original_get = ray_mod.get
        call_count = [0]

        def mock_get(x, **kwargs):
            call_count[0] += 1
            if call_count[0] <= 9:
                # Calls during init: worker info (1) + health checks (4) + get_server_address(4)
                return original_get(x, **kwargs)
            elif call_count[0] == 10:
                # Engine init call
                return original_get(x, **kwargs)
            else:
                # Wake up call - fail here
                raise RayError("Wake up failed")

        ray_mod.get = mock_get

        monkeypatch.setenv("VLLM_DP_SIZE", "1")
        monkeypatch.setattr(
            async_server_mod, "FileCheck",
            types.SimpleNamespace(check_data_path_is_valid=lambda _: None)
        )
        monkeypatch.setattr(async_server_mod, "async_server_class", lambda infer_backend: server_stub)

        manager = async_server_mod.AsyncServerManager(cfg, agent_cfg, "/path/to/tokenizer", wg)

        with pytest.raises(RuntimeError, match="Failed to wake up server instances"):
            manager.wake_up()

    def test_sleep_success(self, async_server_mod, valid_inputs, server_stub, monkeypatch):
        """Test successful sleep of all server instances."""
        cfg, agent_cfg, wg = valid_inputs

        monkeypatch.setenv("VLLM_DP_SIZE", "1")
        monkeypatch.setattr(
            async_server_mod, "FileCheck",
            types.SimpleNamespace(check_data_path_is_valid=lambda _: None)
        )
        monkeypatch.setattr(async_server_mod, "async_server_class", lambda infer_backend: server_stub)

        manager = async_server_mod.AsyncServerManager(cfg, agent_cfg, "/path/to/tokenizer", wg)
        manager.sleep()  # Should complete without error

    def test_sleep_failure(self, async_server_mod, valid_inputs, server_stub, monkeypatch):
        """Test sleep failure with RayError."""
        cfg, agent_cfg, wg = valid_inputs
        RayError = sys.modules["ray.exceptions"].RayError
        ray_mod = sys.modules["ray"]

        original_get = ray_mod.get
        call_count = [0]

        def mock_get(x, **kwargs):
            call_count[0] += 1
            if call_count[0] <= 10:
                # First calls for init
                return original_get(x, **kwargs)
            else:
                # Sleep call
                raise RayError("Sleep failed")

        ray_mod.get = mock_get

        monkeypatch.setenv("VLLM_DP_SIZE", "1")
        monkeypatch.setattr(
            async_server_mod, "FileCheck",
            types.SimpleNamespace(check_data_path_is_valid=lambda _: None)
        )
        monkeypatch.setattr(async_server_mod, "async_server_class", lambda infer_backend: server_stub)

        manager = async_server_mod.AsyncServerManager(cfg, agent_cfg, "/path/to/tokenizer", wg)

        with pytest.raises(RuntimeError, match="Failed to put server instances to sleep"):
            manager.sleep()

    def test_cleanup_servers(self, async_server_mod, valid_inputs, server_stub, monkeypatch):
        """Test cleanup_servers successfully kills all servers."""
        cfg, agent_cfg, wg = valid_inputs
        ray_mod = sys.modules["ray"]

        killed_actors = []
        ray_mod.kill = lambda actor: killed_actors.append(actor)

        monkeypatch.setenv("VLLM_DP_SIZE", "1")
        monkeypatch.setattr(
            async_server_mod, "FileCheck",
            types.SimpleNamespace(check_data_path_is_valid=lambda _: None)
        )
        monkeypatch.setattr(async_server_mod, "async_server_class", lambda infer_backend: server_stub)

        manager = async_server_mod.AsyncServerManager(cfg, agent_cfg, "/path/to/tokenizer", wg)
        initial_servers = [s for s in manager.async_servers if s is not None]

        manager._cleanup_servers()

        assert len(killed_actors) == len(initial_servers)
        assert all(s is None for s in manager.async_servers)
        assert all(s is None for s in manager.server_addresses)

    def test_cleanup_servers_kill_failure(self, async_server_mod, valid_inputs, server_stub, monkeypatch):
        """Test cleanup_servers handles ray.kill failures gracefully."""
        cfg, agent_cfg, wg = valid_inputs
        ray_mod = sys.modules["ray"]

        def failing_kill(actor):
            raise Exception("Kill failed")

        ray_mod.kill = failing_kill

        monkeypatch.setenv("VLLM_DP_SIZE", "1")
        monkeypatch.setattr(
            async_server_mod, "FileCheck",
            types.SimpleNamespace(check_data_path_is_valid=lambda _: None)
        )
        monkeypatch.setattr(async_server_mod, "async_server_class", lambda infer_backend: server_stub)

        manager = async_server_mod.AsyncServerManager(cfg, agent_cfg, "/path/to/tokenizer", wg)

        # Should not raise exception despite kill failures
        manager._cleanup_servers()

        assert all(s is None for s in manager.async_servers)
        assert all(s is None for s in manager.server_addresses)