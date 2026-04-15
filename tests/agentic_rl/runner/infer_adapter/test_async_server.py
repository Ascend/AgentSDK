#!/usr/bin/env python3
# coding=utf-8
# -------------------------------------------------------------------------
# This file is part of the AgentSDK project.
# Copyright (c) 2026 Huawei Technologies Co.,Ltd.
#
# AgentSDK is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#
#          http://license.coscl.org.cn/MulanPSL2
#
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
# -------------------------------------------------------------------------

import asyncio
import importlib
import os
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock, AsyncMock, mock_open, patch

import pytest

FAKE_IP = "127.0.0.1"


# -----------------------------
# Helper: import after mocking
# -----------------------------
def _import_after_mocking(monkeypatch, module_name, preinstall):
    """Inject preinstalled fake modules, then reload and import target module."""
    for fullname, module in preinstall.items():
        monkeypatch.setitem(sys.modules, fullname, module)
    monkeypatch.delitem(sys.modules, module_name, raising=False)
    return importlib.import_module(module_name)


def _create_task(coro):
    """Mock asyncio.create_task to avoid actual task creation."""
    coro.close()
    return MagicMock()


def _remote(obj=None, **_kwargs):
    """Mock ray.remote decorator."""
    return (lambda x: x) if obj is None else obj


# -----------------------------
# Fake runtime context for Ray
# -----------------------------
class _RuntimeContext:
    """Fake Ray runtime context."""
    def __init__(self, node_id="node_x", namespace="ns"):
        self._node_id = node_id
        self.namespace = namespace

    def get_node_id(self):
        return self._node_id


# -----------------------------
# Fixture: async_server module with all dependencies mocked
# -----------------------------
@pytest.fixture
def async_server_mod(monkeypatch):
    """
    Import agentic_rl.runner.infer_adapter.async_server
    with all external dependencies (ray, fastapi, uvicorn, etc.) mocked.
    """
    # ---- ray module stub ----
    ray_mod = ModuleType("ray")
    ray_mod.remote = _remote
    ray_mod.get_runtime_context = lambda: _RuntimeContext("node_abc")
    ray_mod.get = MagicMock()
    ray_mod.kill = MagicMock()
    ray_mod.is_initialized = MagicMock(return_value=True)
    ray_mod.nodes = MagicMock(return_value=[])

    # ray.exceptions
    ray_ex_mod = ModuleType("ray.exceptions")

    class RayTaskError(Exception):
        pass

    ray_ex_mod.RayTaskError = RayTaskError
    ray_mod.exceptions = ray_ex_mod

    # ray._private.services.get_node_ip_address
    ray_mod._private = SimpleNamespace()
    ray_mod._private.services = SimpleNamespace()
    ray_mod._private.services.get_node_ip_address = MagicMock(return_value=FAKE_IP)

    # ray.util.scheduling_strategies.NodeAffinitySchedulingStrategy
    ray_util_mod = ModuleType("ray.util")
    ray_sched_mod = ModuleType("ray.util.scheduling_strategies")

    class NodeAffinitySchedulingStrategy:
        def __init__(self, node_id=None, soft=False):
            self.node_id = node_id
            self.soft = soft

    ray_sched_mod.NodeAffinitySchedulingStrategy = NodeAffinitySchedulingStrategy
    ray_util_mod.scheduling_strategies = ray_sched_mod
    ray_mod.util = ray_util_mod

    # ---- fastapi stub ----
    fastapi_mod = ModuleType("fastapi")

    class _Router:
        def __init__(self):
            self.add_api_route = MagicMock()

    class _FastAPI:
        def __init__(self, lifespan=None, *args, **kwargs):
            self.lifespan = lifespan
            self.router = _Router()

    fastapi_mod.FastAPI = _FastAPI

    # ---- uvicorn stub ----
    uvicorn_mod = ModuleType("uvicorn")

    class _Config:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

    class _Server:
        def __init__(self, config):
            self.config = config
            self.should_exit = False
            self.serve = AsyncMock(return_value=None)

    uvicorn_mod.Config = _Config
    uvicorn_mod.Server = _Server

    # ---- starlette.requests stub ----
    starlette_req_mod = ModuleType("starlette.requests")

    class Request:
        pass

    starlette_req_mod.Request = Request

    # ---- logger stub ----
    loggers_mod = ModuleType("agentic_rl.base.log.loggers")

    class Loggers:
        def __init__(self, *_args, **_kwargs):
            pass

        def get_logger(self):
            return MagicMock()

    loggers_mod.Loggers = Loggers

    # ---- work_mode stub ----
    work_mode_mod = ModuleType("agentic_rl.base.utils.work_mode")
    work_mode_mod.get_work_mode = MagicMock(return_value="separate")

    # ---- globals stub ----
    globals_mod = ModuleType("agentic_rl.base.utils.globals")
    globals_mod.is_pd_separate = MagicMock(return_value=False)

    # ---- InferRouter stub ----
    infer_router_mod = ModuleType("agentic_rl.runner.infer_router")

    # ---- InferRegistry stub ----
    infer_registry_mod = ModuleType("agentic_rl.runner.infer_adapter.infer_registry")
    infer_registry_mod.async_server_class = MagicMock()

    class InferRouter:
        @classmethod
        async def create(cls):
            return AsyncMock()

    infer_router_mod.InferRouter = InferRouter

    # patch asyncio.create_task to avoid starting server in __init__
    monkeypatch.setattr(asyncio, "create_task", _create_task)

    preinstall = {
        "ray": ray_mod,
        "ray.exceptions": ray_ex_mod,
        "ray.util": ray_util_mod,
        "ray.util.scheduling_strategies": ray_sched_mod,
        "fastapi": fastapi_mod,
        "uvicorn": uvicorn_mod,
        "starlette.requests": starlette_req_mod,
        "agentic_rl.base.log.loggers": loggers_mod,
        "agentic_rl.base.utils.work_mode": work_mode_mod,
        "agentic_rl.base.utils.globals": globals_mod,
        "agentic_rl.runner.infer_router": infer_router_mod,
        "agentic_rl.runner.infer_adapter.infer_registry": infer_registry_mod,
    }

    mod = _import_after_mocking(
        monkeypatch,
        "agentic_rl.runner.infer_adapter.async_server",
        preinstall,
    )
    mod._infer_registry_mod = infer_registry_mod
    return mod


# -----------------------------
# Dummy server for AsyncServerBase tests
# -----------------------------
@pytest.fixture
def dummy_server(async_server_mod):
    """Create a dummy server that implements all abstract methods."""
    class Dummy(async_server_mod.AsyncServerBase):
        async def chat_completion(self, raw_request):
            return {"ok": True}

        async def completions(self, raw_request):
            return {"ok": True}

        async def init_engine(self):
            return None

        async def wake_up(self):
            return None

        async def sleep(self):
            return None

        async def get_workload(self, raw_request):
            return {"workload": 0}

        async def cancel_requests(self, raw_request):
            return {"cancelled": True}

    return Dummy()


# -----------------------------
# Tests for AsyncServerBase
# -----------------------------
class TestAsyncServerBase:
    """Test basic functionality of AsyncServerBase."""

    def test_init(self, dummy_server):
        """Verify initialization sets address and server_ready event."""
        assert dummy_server.address == FAKE_IP
        assert dummy_server.port is None
        assert dummy_server.server_ready is not None

    def test_get_server_address(self, dummy_server):
        """get_server_address waits for server_ready and returns address:port."""
        dummy_server.port = 8080
        dummy_server.server_ready.wait = AsyncMock()

        addr = asyncio.run(dummy_server.get_server_address())

        assert addr == f"{FAKE_IP}:8080"
        dummy_server.server_ready.wait.assert_awaited_once()

    def test_start_fastapi_server(self, dummy_server, async_server_mod, monkeypatch):
        """_start_fastapi_server assigns a free port and starts uvicorn."""
        monkeypatch.setattr(async_server_mod, "_get_free_port", lambda: 9090)

        asyncio.run(dummy_server._start_fastapi_server())

        assert dummy_server.port == 9090
        assert async_server_mod.uvicorn.Server is not None


# -----------------------------
# Tests for helper functions
# -----------------------------
class TestHelperFunctions:
    """Test _get_free_port and _write_ranktable_on_node."""

    @patch("agentic_rl.runner.infer_adapter.async_server.socket.socket")
    def test_get_free_port(self, mock_socket, async_server_mod):
        """_get_free_port should bind to port 0 and return the assigned port."""
        sock_obj = MagicMock()
        mock_socket.return_value.__enter__.return_value = sock_obj
        sock_obj.getsockname.return_value = ("0.0.0.0", 12345)

        port = async_server_mod._get_free_port()

        sock_obj.bind.assert_called_once_with(("", 0))
        assert port == 12345

    def test_write_ranktable_on_node_with_dir(self, async_server_mod):
        """_write_ranktable_on_node creates directory if needed and writes JSON."""
        fn = async_server_mod._write_ranktable_on_node
        if hasattr(fn, "_function"):
            fn = fn._function

        m = mock_open()
        with patch("builtins.open", m):
            with patch("os.makedirs") as mk:
                ret = fn({"a": 1}, "/tmp/test_ranktable.json")

        mk.assert_called_once_with("/tmp", exist_ok=True)
        assert ret == "node_abc"

    def test_write_ranktable_on_node_without_dir(self, async_server_mod):
        """When path has no directory, os.makedirs is not called."""
        fn = async_server_mod._write_ranktable_on_node
        if hasattr(fn, "_function"):
            fn = fn._function

        m = mock_open()
        with patch("builtins.open", m):
            with patch("os.makedirs") as mk:
                ret = fn({"a": 1}, "ranktable.json")

        mk.assert_not_called()
        assert ret == "node_abc"


# -----------------------------
# Tests for AsyncServerProxyManager
# -----------------------------
class TestAsyncServerProxyManager:
    """Test proxy manager that delegates to InferRouter."""

    def test_init_separate_mode(self, async_server_mod, monkeypatch):
        """Initialization reads VLLM_DP_SIZE and sets flags."""
        monkeypatch.setenv("VLLM_DP_SIZE", "2")

        mgr = async_server_mod.AsyncServerProxyManager(
            tokenizer_name_or_path="tok",
            worker_group=MagicMock(),
            infer_service="infer",
        )

        assert mgr.weight_offloaded is True
        assert mgr.dp_size == 2
        assert mgr.rollout_tp_size == 0
        assert mgr.rollout_dp_size == 0

    def test_wake_up(self, async_server_mod):
        """wake_up calls infer_router.wake_up and clears weight_offloaded flag."""
        mgr = async_server_mod.AsyncServerProxyManager(
            tokenizer_name_or_path="tok",
            worker_group=MagicMock(),
            infer_service="infer",
        )
        mgr.infer_router = AsyncMock()
        mgr.weight_offloaded = True

        asyncio.run(mgr.wake_up())

        mgr.infer_router.wake_up.assert_awaited_once_with(model_name="infer")
        assert mgr.weight_offloaded is False

    def test_sleep(self, async_server_mod):
        """sleep calls infer_router.sleep and sets weight_offloaded flag."""
        mgr = async_server_mod.AsyncServerProxyManager(
            tokenizer_name_or_path="tok",
            worker_group=MagicMock(),
            infer_service="infer",
        )
        mgr.infer_router = AsyncMock()
        mgr.weight_offloaded = False

        asyncio.run(mgr.sleep())

        mgr.infer_router.sleep.assert_awaited_once_with(model_name="infer")
        assert mgr.weight_offloaded is True

    def test_update_weights(self, async_server_mod):
        """update_weights forwards to infer_router.update_weights."""
        mgr = async_server_mod.AsyncServerProxyManager(
            tokenizer_name_or_path="tok",
            worker_group=MagicMock(),
            infer_service="infer",
        )
        mgr.infer_router = AsyncMock()

        asyncio.run(mgr.update_weights("/path/to/weights"))

        mgr.infer_router.update_weights.assert_awaited_once()

    def test_proxy_manager_init_calls_infer_router(self, async_server_mod, monkeypatch):
        """init() creates InferRouter and initializes it with per-rank kwargs."""
        mgr = async_server_mod.AsyncServerProxyManager(
            tokenizer_name_or_path="tok",
            worker_group=None,
            infer_service="infer",
        )
        mgr.rollout_dp_size = 2  # Force dp size for test

        fake_router = AsyncMock()
        async_server_mod.InferRouter.create = AsyncMock(return_value=fake_router)

        asyncio.run(mgr.init())

        async_server_mod.InferRouter.create.assert_awaited_once()
        fake_router.init.assert_awaited_once_with("infer")

        # wake_up kwargs_list must contain two ranks
        args, kwargs = fake_router.wake_up.call_args
        assert kwargs["model_name"] == "infer"
        assert len(kwargs["kwargs_list"]) == 2
        assert kwargs["kwargs_list"][0]["vllm_dp_rank"] == 0
        assert kwargs["kwargs_list"][1]["vllm_dp_rank"] == 1


# -----------------------------
# Tests for AsyncServerManager
# -----------------------------
class TestAsyncServerManager:
    """Test the Ray-based distributed server manager."""

    def _make_mgr(self, async_server_mod):
        """Helper to create a minimal AsyncServerManager instance."""
        mgr = async_server_mod.AsyncServerManager.__new__(async_server_mod.AsyncServerManager)
        mgr.weight_offloaded = True
        mgr.config = SimpleNamespace(infer_backend="vllm", infer_tensor_parallel_size=2)
        mgr.worker_group = SimpleNamespace(num_npus=4)
        mgr.rollout_tp_size = 2
        mgr.rollout_dp_size = 2
        mgr.async_servers = []
        mgr.server_addresses = []
        return mgr

    def test_get_mapping_id_to_ip(self, async_server_mod):
        """get_mapping_id_to_ip returns node ID to IP mapping from Ray."""
        mgr = self._make_mgr(async_server_mod)

        async_server_mod.ray.is_initialized.return_value = True
        async_server_mod.ray.nodes.return_value = [
            {"NodeID": "nodeA", "NodeManagerAddress": "10.0.0.1"},
            {"NodeID": "nodeB", "NodeManagerAddress": "10.0.0.2"},
        ]

        mapping = mgr.get_mapping_id_to_ip()
        assert mapping["nodeA"] == "10.0.0.1"
        assert mapping["nodeB"] == "10.0.0.2"

    def test_get_mapping_id_to_ip_ray_not_init(self, async_server_mod):
        """If Ray not initialized, raise RuntimeError."""
        mgr = self._make_mgr(async_server_mod)

        async_server_mod.ray.is_initialized.return_value = False

        with pytest.raises(RuntimeError):
            mgr.get_mapping_id_to_ip()

    def test_update_ranktable_pd_disabled(self, async_server_mod, monkeypatch):
        """When PD separation disabled, update_ranktable does nothing."""
        mgr = self._make_mgr(async_server_mod)

        monkeypatch.setattr(async_server_mod, "is_pd_separate", lambda: False)

        mgr.get_mapping_id_to_ip = MagicMock()
        mgr.rewrite_ranktable_to_all_nodes = MagicMock()

        mgr.update_ranktable_from_workers_info({0: "nodeA"})

        mgr.get_mapping_id_to_ip.assert_not_called()
        mgr.rewrite_ranktable_to_all_nodes.assert_not_called()

    def test_update_ranktable_missing_env_raises(self, async_server_mod, monkeypatch):
        """If PD enabled but RANKTABLE env var missing, raise ValueError."""
        mgr = self._make_mgr(async_server_mod)

        monkeypatch.setattr(async_server_mod, "is_pd_separate", lambda: True)

        monkeypatch.setenv("P_INSTANCE_NUM_DEVICE", "1")
        monkeypatch.setenv("D_INSTANCE_NUM_DEVICE", "0")

        monkeypatch.delenv("DISAGGREGATED_PREFILL_RANK_TABLE_PATH", raising=False)

        mgr.get_mapping_id_to_ip = MagicMock(return_value={"nodeA": "10.0.0.1"})

        with pytest.raises(ValueError, match="DISAGGREGATED_PREFILL_RANK_TABLE_PATH must be set"):
            mgr.update_ranktable_from_workers_info({0: "nodeA"})

    def test_update_ranktable_file_not_found(self, async_server_mod, monkeypatch):
        """If ranktable file does not exist, raise FileNotFoundError."""
        mgr = self._make_mgr(async_server_mod)

        monkeypatch.setattr(async_server_mod, "is_pd_separate", lambda: True)

        monkeypatch.setenv("P_INSTANCE_NUM_DEVICE", "1")
        monkeypatch.setenv("D_INSTANCE_NUM_DEVICE", "0")

        monkeypatch.setenv("DISAGGREGATED_PREFILL_RANK_TABLE_PATH", "/tmp/ranktable.json")

        mgr.get_mapping_id_to_ip = MagicMock(return_value={"nodeA": "10.0.0.1"})

        with patch("os.path.exists", return_value=False):
            with pytest.raises(FileNotFoundError, match="Ranktable file must exist but not found"):
                mgr.update_ranktable_from_workers_info({0: "nodeA"})

    def test_update_ranktable_node_mismatch_raises(self, async_server_mod, monkeypatch):
        """If node IPs in ranktable do not match actual worker IPs, raise ValueError."""
        mgr = self._make_mgr(async_server_mod)

        monkeypatch.setattr(async_server_mod, "is_pd_separate", lambda: True)

        monkeypatch.setenv("P_INSTANCE_NUM_DEVICE", "1")
        monkeypatch.setenv("D_INSTANCE_NUM_DEVICE", "1")
        monkeypatch.setenv("DISAGGREGATED_PREFILL_RANK_TABLE_PATH", "/tmp/ranktable.json")

        mgr.get_mapping_id_to_ip = MagicMock(return_value={"nodeA": "10.0.0.1"})

        fake_ranktable = {
            "prefill_device_list": [{"server_id": "10.0.0.2", "cluster_id": "1"}],
            "decode_device_list": [{"server_id": "10.0.0.3", "cluster_id": "2"}],
        }

        with patch("os.path.exists", return_value=True):
            with patch("builtins.open", mock_open(read_data="{}")):
                with patch("json.load", return_value=fake_ranktable):
                    with pytest.raises(ValueError, match="Node mismatch"):
                        mgr.update_ranktable_from_workers_info({0: "nodeA", 1: "nodeA"})

    def test_rewrite_ranktable_to_all_nodes(self, async_server_mod):
        """rewrite_ranktable_to_all_nodes calls remote writer on all worker nodes."""
        mgr = self._make_mgr(async_server_mod)

        # mock _write_ranktable_on_node.options().remote()
        fake_writer = MagicMock()
        fake_writer.options.return_value.remote.side_effect = ["futA", "futB"]

        async_server_mod._write_ranktable_on_node = fake_writer
        async_server_mod.ray.get.return_value = ["nodeA", "nodeB"]

        workers_info = {0: "nodeA", 1: "nodeB"}
        ranktable = {"prefill_device_list": [], "decode_device_list": []}

        mgr.rewrite_ranktable_to_all_nodes(workers_info, ranktable, "/tmp/ranktable.json")

        assert fake_writer.options.call_count == 2
        async_server_mod.ray.get.assert_called_once()

    def test_manager_wake_up(self, async_server_mod):
        """wake_up calls wake_up.remote on all async servers and waits."""
        mgr = async_server_mod.AsyncServerManager.__new__(async_server_mod.AsyncServerManager)
        mgr.async_servers = [MagicMock(), MagicMock()]
        mgr.weight_offloaded = True

        mgr.async_servers[0].wake_up.remote.return_value = "fut0"
        mgr.async_servers[1].wake_up.remote.return_value = "fut1"

        async_server_mod.ray.get.return_value = [None, None]

        asyncio.run(mgr.wake_up())

        async_server_mod.ray.get.assert_called_once_with(["fut0", "fut1"])
        assert mgr.weight_offloaded is False

    def test_manager_sleep(self, async_server_mod):
        """sleep calls sleep.remote on all async servers and waits."""
        mgr = async_server_mod.AsyncServerManager.__new__(async_server_mod.AsyncServerManager)
        mgr.async_servers = [MagicMock(), MagicMock()]
        mgr.weight_offloaded = False

        mgr.async_servers[0].sleep.remote.return_value = "fut0"
        mgr.async_servers[1].sleep.remote.return_value = "fut1"

        async_server_mod.ray.get.return_value = [None, None]

        asyncio.run(mgr.sleep())

        async_server_mod.ray.get.assert_called_once_with(["fut0", "fut1"])
        assert mgr.weight_offloaded is True

    def test_manager_reset_prefix_cache(self, async_server_mod):
        """reset_prefix_cache calls reset_prefix_cache.remote on all servers."""
        mgr = async_server_mod.AsyncServerManager.__new__(async_server_mod.AsyncServerManager)
        mgr.async_servers = [MagicMock(), MagicMock()]

        mgr.async_servers[0].reset_prefix_cache.remote.return_value = "fut0"
        mgr.async_servers[1].reset_prefix_cache.remote.return_value = "fut1"

        async_server_mod.ray.get.return_value = [None, None]

        asyncio.run(mgr.reset_prefix_cache())

        async_server_mod.ray.get.assert_called_once_with(["fut0", "fut1"])

    def test_manager_update_weights(self, async_server_mod):
        """update_weights calls collective_rpc.remote with 'update_weights' method."""
        mgr = async_server_mod.AsyncServerManager.__new__(async_server_mod.AsyncServerManager)
        mgr.async_servers = [MagicMock(), MagicMock()]

        mgr.async_servers[0].collective_rpc.remote = AsyncMock()
        mgr.async_servers[1].collective_rpc.remote = AsyncMock()

        asyncio.run(mgr.update_weights("/tmp/weights"))

        mgr.async_servers[0].collective_rpc.remote.assert_awaited_once_with("update_weights", args="/tmp/weights")
        mgr.async_servers[1].collective_rpc.remote.assert_awaited_once_with("update_weights", args="/tmp/weights")

    def test_manager_vllm_statistics(self, async_server_mod):
        """vllm_statistics calls collective_rpc.remote with 'vllm_statistics'."""
        mgr = async_server_mod.AsyncServerManager.__new__(async_server_mod.AsyncServerManager)
        mgr.async_servers = [MagicMock(), MagicMock()]

        mgr.async_servers[0].collective_rpc.remote = AsyncMock()
        mgr.async_servers[1].collective_rpc.remote = AsyncMock()

        asyncio.run(mgr.vllm_statistics())

        mgr.async_servers[0].collective_rpc.remote.assert_awaited_once_with("vllm_statistics")
        mgr.async_servers[1].collective_rpc.remote.assert_awaited_once_with("vllm_statistics")


# -----------------------------
# Tests for AsyncServerManager __init__ heavy flow
# -----------------------------
class TestAsyncServerManagerInitFlow:
    """Test the complex initialization flow of AsyncServerManager."""

    def test_init_happy_path(self, async_server_mod, monkeypatch):
        """Normal initialization: creates Ray actors, initializes engines, builds rank table."""
        monkeypatch.setenv("VLLM_DP_SIZE", "1")

        config = SimpleNamespace(
            infer_backend="vllm",
            infer_tensor_parallel_size=2,
        )

        worker_group = MagicMock()
        worker_group.num_npus = 4
        worker_group.execute_async_command.return_value = "worker_info_future"

        def fake_ray_get(arg):
            if arg == "worker_info_future":
                return [("0", "nodeA"), ("1", "nodeB"), ("2", "nodeC"), ("3", "nodeD")]
            if arg in ("addr_future0", "addr_future1"):
                return "10.0.0.1:8000" if arg == "addr_future0" else "10.0.0.2:8001"
            if isinstance(arg, list):
                return [None for _ in arg]
            return None

        async_server_mod.ray.get.side_effect = fake_ray_get

        fake_server_class = MagicMock(name="FakeServerClass")
        infer_registry_mod = async_server_mod._infer_registry_mod
        infer_registry_mod.async_server_class.return_value = fake_server_class

        actor0 = MagicMock()
        actor0.get_server_address.remote.return_value = "addr_future0"
        actor0.init_engine.remote.return_value = "init_future0"

        actor1 = MagicMock()
        actor1.get_server_address.remote.return_value = "addr_future1"
        actor1.init_engine.remote.return_value = "init_future1"

        remote_builder = MagicMock()
        remote_builder.options.return_value.remote.side_effect = [actor0, actor1]

        async_server_mod.ray.remote = MagicMock(return_value=remote_builder)

        monkeypatch.setattr(
            async_server_mod.AsyncServerManager,
            "update_ranktable_from_workers_info",
            MagicMock(),
        )

        mgr = async_server_mod.AsyncServerManager(config, "tok", worker_group)

        assert mgr.rollout_dp_size == 2
        assert mgr.server_addresses == ["10.0.0.1:8000", "10.0.0.2:8001"]
        assert mgr.async_servers == [actor0, actor1]

        async_server_mod.AsyncServerManager.update_ranktable_from_workers_info.assert_called_once()
        assert async_server_mod.ray.remote.call_count == 2
        assert remote_builder.options.call_count == 2

    def test_init_pd_separate_adds_infer_mode(self, async_server_mod, monkeypatch):
        """When PD separation is enabled, server addresses include 'prefill'/'decode' prefix."""
        monkeypatch.setenv("VLLM_DP_SIZE", "1")
        monkeypatch.setattr(async_server_mod, "is_pd_separate", lambda: True)

        config = SimpleNamespace(
            infer_backend="vllm",
            infer_tensor_parallel_size=2,
        )

        worker_group = MagicMock()
        worker_group.num_npus = 4
        worker_group.execute_async_command.return_value = "worker_info_future"

        def fake_ray_get(arg):
            if arg == "worker_info_future":
                return [("0", "nodeA"), ("1", "nodeB"), ("2", "nodeC"), ("3", "nodeD")]
            if arg == "addr_future0":
                return "10.0.0.1:8000"
            if arg == "addr_future1":
                return "10.0.0.2:8001"
            if arg == "infer_mode_future0":
                return "prefill"
            if arg == "infer_mode_future1":
                return "decode"
            if isinstance(arg, list):
                return [None for _ in arg]
            return None

        async_server_mod.ray.get.side_effect = fake_ray_get

        fake_server_class = MagicMock(name="FakeServerClass")
        infer_registry_mod = async_server_mod._infer_registry_mod
        infer_registry_mod.async_server_class.return_value = fake_server_class

        actor0 = MagicMock()
        actor0.get_server_address.remote.return_value = "addr_future0"
        actor0.get_infer_mode.remote.return_value = "infer_mode_future0"
        actor0.init_engine.remote.return_value = "init_future0"

        actor1 = MagicMock()
        actor1.get_server_address.remote.return_value = "addr_future1"
        actor1.get_infer_mode.remote.return_value = "infer_mode_future1"
        actor1.init_engine.remote.return_value = "init_future1"

        remote_builder = MagicMock()
        remote_builder.options.return_value.remote.side_effect = [actor0, actor1]

        async_server_mod.ray.remote = MagicMock(return_value=remote_builder)

        monkeypatch.setattr(
            async_server_mod.AsyncServerManager,
            "update_ranktable_from_workers_info",
            MagicMock(),
        )

        mgr = async_server_mod.AsyncServerManager(config, "tok", worker_group)

        assert mgr.server_addresses == [
            "prefill-10.0.0.1:8000",
            "decode-10.0.0.2:8001",
        ]

    def test_init_restart_on_address_in_use(self, async_server_mod, monkeypatch):
        """If an actor fails with address already in use, it is killed and restarted."""
        monkeypatch.setenv("VLLM_DP_SIZE", "1")

        config = SimpleNamespace(
            infer_backend="vllm",
            infer_tensor_parallel_size=2,
        )

        worker_group = MagicMock()
        worker_group.num_npus = 4
        worker_group.execute_async_command.return_value = "worker_info_future"

        RayTaskError = async_server_mod.ray.exceptions.RayTaskError

        # Three actors: first rank0 fails, rank1 succeeds; retry for rank0 succeeds
        actor0 = MagicMock()
        actor0.get_server_address.remote.return_value = "addr_future0"
        actor0.init_engine.remote.return_value = "init_future0"

        actor1 = MagicMock()
        actor1.get_server_address.remote.return_value = "addr_future1"
        actor1.init_engine.remote.return_value = "init_future1"

        actor0_retry = MagicMock()
        actor0_retry.get_server_address.remote.return_value = "addr_future0_retry"
        actor0_retry.init_engine.remote.return_value = "init_future0_retry"

        remote_builder = MagicMock()
        remote_builder.options.return_value.remote.side_effect = [actor0, actor1, actor0_retry]

        async_server_mod.ray.remote = MagicMock(return_value=remote_builder)

        def fake_ray_get(arg):
            if arg == "worker_info_future":
                return [("0", "nodeA"), ("1", "nodeB"), ("2", "nodeC"), ("3", "nodeD")]

            # First get for rank0 fails with address in use
            if arg == "addr_future0":
                raise RayTaskError("address in use")

            # rank1 succeeds
            if arg == "addr_future1":
                return "10.0.0.2:8001"

            # retry for rank0 succeeds
            if arg == "addr_future0_retry":
                return "10.0.0.1:8000"

            # init_engine list
            if isinstance(arg, list):
                return [None for _ in arg]

            return None

        async_server_mod.ray.get.side_effect = fake_ray_get

        monkeypatch.setattr(
            async_server_mod.AsyncServerManager,
            "update_ranktable_from_workers_info",
            MagicMock(),
        )

        mgr = async_server_mod.AsyncServerManager(config, "tok", worker_group)

        assert mgr.server_addresses == ["10.0.0.1:8000", "10.0.0.2:8001"]

        # ray.kill must be called on the failed actor0
        async_server_mod.ray.kill.assert_called_once_with(actor0)

    def test_init_engine_called_for_all_servers(self, async_server_mod, monkeypatch):
        """
        Ensure init_engine is called on all created actors and ray.get waits for them.
        """
        monkeypatch.setenv("VLLM_DP_SIZE", "1")

        config = SimpleNamespace(
            infer_backend="vllm",
            infer_tensor_parallel_size=2,
        )

        worker_group = MagicMock()
        worker_group.num_npus = 4
        worker_group.execute_async_command.return_value = "worker_info_future"

        def fake_ray_get(arg):
            if arg == "worker_info_future":
                return [("0", "nodeA"), ("1", "nodeB"), ("2", "nodeC"), ("3", "nodeD")]
            if arg == "addr_future":
                return "10.0.0.1:8000"
            if isinstance(arg, list):
                return [None for _ in arg]
            return None

        async_server_mod.ray.get.side_effect = fake_ray_get

        fake_server_class = MagicMock(name="FakeServerClass")
        infer_registry_mod = async_server_mod._infer_registry_mod
        infer_registry_mod.async_server_class.return_value = fake_server_class

        fake_actor = MagicMock()
        fake_actor.get_server_address.remote.return_value = "addr_future"
        fake_actor.init_engine.remote.return_value = "init_future"

        remote_builder = MagicMock()
        remote_builder.options.return_value.remote.return_value = fake_actor

        async_server_mod.ray.remote = MagicMock(return_value=remote_builder)

        monkeypatch.setattr(async_server_mod.AsyncServerManager,
                            "update_ranktable_from_workers_info",
                            MagicMock())

        _ = async_server_mod.AsyncServerManager(config, "tok", worker_group)

        # The last ray.get call should be on the list of init_engine futures
        last_call_args = async_server_mod.ray.get.call_args_list[-1][0][0]
        assert isinstance(last_call_args, list)
        assert len(last_call_args) == 2
        assert async_server_mod.ray.remote.call_count == 2
        assert remote_builder.options.call_count == 2


if __name__ == "__main__":
    pytest.main([__file__])