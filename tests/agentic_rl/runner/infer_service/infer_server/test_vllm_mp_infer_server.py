#!/usr/bin/env python3
# -*- coding: utf-8 -*-
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
import json
import os
import pytest
import sys
import threading
from unittest.mock import patch, MagicMock, AsyncMock


# =========================
# Helper Functions Tests
# =========================
class TestHelperFunctions:
    """Tests for helper functions in vllm_mp_infer_server."""

    def setup_method(self):
        """Setup method to import helper functions before each test."""
        from agentic_rl.runner.infer_service.infer_server.vllm_mp_infer_server import (
            print_log, start_cmd, map_dict_to_cli_args, start_slave, start_master
        )
        self.print_log = print_log
        self.start_cmd = start_cmd
        self.map_dict_to_cli_args = map_dict_to_cli_args
        self.start_slave = start_slave
        self.start_master = start_master

    @patch('agentic_rl.runner.infer_service.infer_server.vllm_mp_infer_server.logger')
    @patch('time.sleep')
    def test_print_log(self, mock_sleep, mock_logger):
        def stop(_):
            raise KeyboardInterrupt()

        mock_sleep.side_effect = stop

        out_buffer = ["line 1", "line 2", "Application startup complete.", "line 4"]
        key_event = threading.Event()

        with pytest.raises(KeyboardInterrupt):
            self.print_log(out_buffer, "Application startup complete.", key_event)

        assert mock_logger.info.call_count == 4
        assert key_event.is_set()

    @patch('os.getpgid', return_value=1234)
    @patch('agentic_rl.runner.infer_service.infer_server.vllm_mp_infer_server.subprocess.Popen')
    @patch('agentic_rl.runner.infer_service.infer_server.vllm_mp_infer_server.logger')
    @patch('agentic_rl.runner.infer_service.infer_server.vllm_mp_infer_server.atexit')
    @patch('os.killpg')
    @patch.dict('os.environ', {'ASCEND_RT_VISIBLE_DEVICES': '3,1,2'})
    def test_start_cmd(self, mock_killpg, mock_atexit, mock_logger, mock_popen, mock_getpgid):
        mock_process = MagicMock()
        mock_process.pid = 1234
        mock_process.poll.return_value = None
        mock_process.stdout.readline.side_effect = [b'output\n', b'']
        mock_popen.return_value = mock_process

        out_buffer = []
        self.start_cmd(["cmd"], out_buffer)

        assert os.environ['ASCEND_RT_VISIBLE_DEVICES'] == "1,2,3"
        assert os.environ['VLLM_SERVER_DEV_MODE'] == "1"
        assert out_buffer == ["output\n"]

        cleanup = mock_atexit.register.call_args[0][0]
        cleanup()
        mock_killpg.assert_called()

    @patch('os.getpgid', side_effect=ProcessLookupError("Process not found"))
    @patch('agentic_rl.runner.infer_service.infer_server.vllm_mp_infer_server.subprocess.Popen')
    @patch('agentic_rl.runner.infer_service.infer_server.vllm_mp_infer_server.logger')
    @patch('agentic_rl.runner.infer_service.infer_server.vllm_mp_infer_server.atexit')
    @patch.dict('os.environ', {'ASCEND_RT_VISIBLE_DEVICES': '3,1,2'})
    def test_start_cmd_cleanup_process_lookup_error(self, mock_atexit, mock_logger, mock_popen, mock_getpgid):
        mock_process = MagicMock()
        mock_process.pid = 1234
        mock_process.poll.return_value = None
        mock_process.stdout.readline.side_effect = [b'output\n', b'']
        mock_popen.return_value = mock_process

        out_buffer = []
        self.start_cmd(["cmd"], out_buffer)

        cleanup = mock_atexit.register.call_args[0][0]
        cleanup()

        mock_getpgid.assert_called_with(1234)

    @patch('agentic_rl.runner.infer_service.infer_server.vllm_mp_infer_server.subprocess.Popen')
    @patch('agentic_rl.runner.infer_service.infer_server.vllm_mp_infer_server.logger')
    @patch.dict('os.environ', {'ASCEND_RT_VISIBLE_DEVICES': '3,1,2'})
    def test_start_cmd_keyboard_interrupt(self, mock_logger, mock_popen):
        mock_process = MagicMock()
        mock_process.pid = 1234
        mock_process.poll.return_value = None
        mock_process.stdout.readline.side_effect = KeyboardInterrupt()
        mock_popen.return_value = mock_process

        out_buffer = []
        self.start_cmd(["cmd"], out_buffer)

        mock_logger.error.assert_called_once_with("User interrupted.")

    @patch('agentic_rl.runner.infer_service.infer_server.vllm_mp_infer_server.subprocess.Popen')
    @patch('agentic_rl.runner.infer_service.infer_server.vllm_mp_infer_server.logger')
    @patch.dict('os.environ', {'ASCEND_RT_VISIBLE_DEVICES': '3,1,2'})
    def test_start_cmd_exception(self, mock_logger, mock_popen):
        mock_popen.side_effect = Exception("启动失败")

        out_buffer = []
        self.start_cmd(["cmd"], out_buffer)

        mock_logger.error.assert_called_once()
        assert "启动失败" in mock_logger.error.call_args[0][0]

    def test_map_dict_to_cli_args(self):
        params = {
            "a": 1,
            "b": True,
            "c": False,
            "d": None
        }

        result = self.map_dict_to_cli_args(params)
        assert result == ["--a", "1", "--b"]

        assert self.map_dict_to_cli_args({}) == []
        assert self.map_dict_to_cli_args({"a": None}) == []

    @patch('agentic_rl.runner.infer_service.infer_server.vllm_mp_infer_server.threading.Thread')
    @patch('agentic_rl.runner.infer_service.infer_server.vllm_mp_infer_server.map_dict_to_cli_args')
    def test_start_slave(self, mock_map, mock_thread):
        mock_map.return_value = ["cmd"]

        kwargs = {"model": "m", "data_parallel_size_local": 2}
        self.start_slave(1, "addr", kwargs)

        assert kwargs["data_parallel_start_rank"] == 2
        assert "model" not in kwargs

        assert mock_thread.call_count == 2

    @patch('agentic_rl.runner.infer_service.infer_server.vllm_mp_infer_server.threading.Event')
    @patch('agentic_rl.runner.infer_service.infer_server.vllm_mp_infer_server.threading.Thread')
    @patch('agentic_rl.runner.infer_service.infer_server.vllm_mp_infer_server.map_dict_to_cli_args')
    def test_start_master(self, mock_map, mock_thread, mock_event):
        mock_event_instance = MagicMock()
        mock_event.return_value = mock_event_instance

        kwargs = {"model": "m"}
        self.start_master("addr", kwargs)

        mock_event_instance.wait.assert_called_once()


# =========================
# RemoteMPVLLMInferServer Tests
# =========================
@pytest.mark.asyncio
class TestRemoteMPVLLMInferServer:
    """Tests for RemoteMPVLLMInferServer class."""

    def setup_method(self):
        """Setup method to import RemoteMPVLLMInferServer before each test."""
        from agentic_rl.runner.infer_service.infer_server.vllm_mp_infer_server import RemoteMPVLLMInferServer
        self.RemoteMPVLLMInferServer = RemoteMPVLLMInferServer

    @patch('agentic_rl.runner.infer_service.infer_server.vllm_mp_infer_server.start_master')
    async def test_init_master(self, mock_master):
        with (
            patch.object(self.RemoteMPVLLMInferServer, "init_environment", new_callable=AsyncMock),
            patch.object(self.RemoteMPVLLMInferServer, "get_local_addr", new_callable=AsyncMock, return_value="ip"),
        ):
            server = self.RemoteMPVLLMInferServer()
            await server.init_server(0, "ip", "model", model="m")

        mock_master.assert_called_once()

    @patch('agentic_rl.runner.infer_service.infer_server.vllm_mp_infer_server.start_slave')
    async def test_init_slave(self, mock_slave):
        with (
            patch.object(self.RemoteMPVLLMInferServer, "init_environment", new_callable=AsyncMock),
            patch.object(self.RemoteMPVLLMInferServer, "get_local_addr", new_callable=AsyncMock, return_value="ip2"),
        ):
            server = self.RemoteMPVLLMInferServer()
            await server.init_server(1, "ip", "model", model="m")

        mock_slave.assert_called_once()

    @patch.dict("os.environ", {}, clear=True)
    async def test_init_env(self):
        with (
            patch.object(self.RemoteMPVLLMInferServer, "get_local_addr", new_callable=AsyncMock, return_value="ip"),
            patch.object(self.RemoteMPVLLMInferServer, "get_nic_name", new_callable=AsyncMock, return_value="eth0"),
        ):
            server = self.RemoteMPVLLMInferServer()
            await server.init_environment()

        assert os.environ["HCCL_IF_IP"] == "ip"
        assert os.environ["GLOO_SOCKET_IFNAME"] == "eth0"

    @patch("psutil.net_if_addrs")
    async def test_get_nic(self, mock_net):
        mock_snic = MagicMock()
        mock_snic.family = 2
        mock_snic.address = "ip"

        mock_net.return_value = {"eth0": [mock_snic]}

        assert await self.RemoteMPVLLMInferServer.get_nic_name("ip") == "eth0"

    @patch("psutil.net_if_addrs")
    async def test_get_nic_not_found(self, mock_net):
        mock_snic = MagicMock()
        mock_snic.family = 2
        mock_snic.address = "other_ip"

        mock_net.return_value = {"eth0": [mock_snic]}

        result = await self.RemoteMPVLLMInferServer.get_nic_name("ip")
        assert result == "Unknown"

    @patch("socket.socket")
    async def test_get_local_addr(self, mock_socket):
        mock_sock = MagicMock()
        mock_sock.getsockname.return_value = ("ip", 1)
        mock_socket.return_value = mock_sock

        ip = await self.RemoteMPVLLMInferServer.get_local_addr()
        assert ip == "ip"

    @patch("socket.socket")
    @patch("agentic_rl.runner.infer_service.infer_server.vllm_mp_infer_server.logger")
    async def test_get_local_addr_exception(self, mock_logger, mock_socket):
        mock_socket.side_effect = Exception("Connection failed")

        ip = await self.RemoteMPVLLMInferServer.get_local_addr()
        assert ip == "127.0.0.1"
        mock_logger.error.assert_called_once()


# =========================
# VLLMMPInferServer Tests
# =========================
class TestVLLMMPInferServer:
    """Tests for VLLMMPInferServer class."""

    def setup_method(self):
        """Setup method to import VLLMMPInferServer before each test."""
        from agentic_rl.runner.infer_service.infer_server.vllm_mp_infer_server import VLLMMPInferServer
        self.VLLMMPInferServer = VLLMMPInferServer

    def _setup_mocks(self, mock_pg, mock_remote, mock_openai, mock_ray_get):
        pg = MagicMock()
        pg.bundle_specs = [{}]
        mock_pg.return_value = pg

        actor = MagicMock()

        def remote(*a, **k):
            return MagicMock(
                options=MagicMock(
                    return_value=MagicMock(
                        remote=MagicMock(return_value=actor)
                    )
                )
            )

        mock_remote.side_effect = remote
        mock_ray_get.return_value = "ip"

        client = MagicMock()
        client.completions.create = AsyncMock()
        client.chat.completions.create = AsyncMock()

        mock_openai.return_value = client
        return client

    @patch("ray.get")
    @patch("ray.util.get_current_placement_group")
    @patch("ray.remote")
    @patch("agentic_rl.runner.infer_service.infer_server.vllm_mp_infer_server.AsyncOpenAI")
    def test_init(self, mock_openai, mock_remote, mock_pg, mock_ray_get):
        client = self._setup_mocks(mock_pg, mock_remote, mock_openai, mock_ray_get)

        s = self.VLLMMPInferServer("m", model="m", port=1, distributed_executor_backend="mp")

        assert s.vllm_server_addr == "http://ip:1"
        assert client is not None

    def test_init_invalid_backend(self):
        with pytest.raises(RuntimeError, match="distributed_executor_backend must be mp"):
            self.VLLMMPInferServer("m", model="m", port=1, distributed_executor_backend="invalid")

    @patch("ray.util.get_current_placement_group")
    def test_init_no_placement_group(self, mock_pg):
        mock_pg.return_value = None

        with pytest.raises(RuntimeError, match="Current actor is not running in a Placement Group"):
            self.VLLMMPInferServer("m", model="m", port=1)

    @patch("ray.get")
    @patch("ray.util.get_current_placement_group")
    @patch("ray.remote")
    @patch("agentic_rl.runner.infer_service.infer_server.vllm_mp_infer_server.AsyncOpenAI")
    @pytest.mark.asyncio
    async def test_completions(self, mock_openai, mock_remote, mock_pg, mock_ray_get):
        client = self._setup_mocks(mock_pg, mock_remote, mock_openai, mock_ray_get)

        s = self.VLLMMPInferServer("m", model="m", port=1, distributed_executor_backend="mp")

        mock_resp = MagicMock()
        mock_resp.model_dump.return_value = {"ok": 1}
        client.completions.create.return_value = mock_resp

        data = {"prompt": "a", "stream": True}
        res = await s.completions(data)

        args = client.completions.create.call_args[1]

        assert "stream" not in args
        assert args["model"] == "m"
        assert res == {"ok": 1}

    @patch("ray.get")
    @patch("ray.util.get_current_placement_group")
    @patch("ray.remote")
    @patch("agentic_rl.runner.infer_service.infer_server.vllm_mp_infer_server.AsyncOpenAI")
    @pytest.mark.asyncio
    async def test_completions_with_extra_headers(self, mock_openai, mock_remote, mock_pg, mock_ray_get):
        client = self._setup_mocks(mock_pg, mock_remote, mock_openai, mock_ray_get)

        s = self.VLLMMPInferServer("m", model="m", port=1, distributed_executor_backend="mp")

        mock_resp = MagicMock()
        mock_resp.model_dump.return_value = {"ok": 1}
        client.completions.create.return_value = mock_resp

        data = {"prompt": "a", "stream": True, "extra_headers": {"X-Custom": "value"}}
        res = await s.completions(data)

        args = client.completions.create.call_args[1]

        assert "extra_headers" not in args
        assert res == {"ok": 1}

    @patch("ray.get")
    @patch("ray.util.get_current_placement_group")
    @patch("ray.remote")
    @patch("agentic_rl.runner.infer_service.infer_server.vllm_mp_infer_server.AsyncOpenAI")
    @pytest.mark.asyncio
    async def test_chat_completions(self, mock_openai, mock_remote, mock_pg, mock_ray_get):
        client = self._setup_mocks(mock_pg, mock_remote, mock_openai, mock_ray_get)

        s = self.VLLMMPInferServer("m", model="m", port=1, distributed_executor_backend="mp")

        mock_resp = MagicMock()
        mock_resp.model_dump.return_value = {"ok": 1}
        client.chat.completions.create.return_value = mock_resp

        data = {"messages": [{"role": "user", "content": "hi"}], "stream": True}
        res = await s.chat_completions(data)

        args = client.chat.completions.create.call_args[1]

        assert "stream" not in args
        assert args["model"] == "m"
        assert res == {"ok": 1}

    @patch("ray.get")
    @patch("ray.util.get_current_placement_group")
    @patch("ray.remote")
    @patch("agentic_rl.runner.infer_service.infer_server.vllm_mp_infer_server.AsyncOpenAI")
    @pytest.mark.asyncio
    async def test_chat_completions_with_extra_headers(self, mock_openai, mock_remote, mock_pg, mock_ray_get):
        client = self._setup_mocks(mock_pg, mock_remote, mock_openai, mock_ray_get)

        s = self.VLLMMPInferServer("m", model="m", port=1, distributed_executor_backend="mp")

        mock_resp = MagicMock()
        mock_resp.model_dump.return_value = {"ok": 1}
        client.chat.completions.create.return_value = mock_resp

        data = {"messages": [{"role": "user", "content": "hi"}], "extra_headers": {"X-Custom": "value"}}
        res = await s.chat_completions(data)

        args = client.chat.completions.create.call_args[1]

        assert "extra_headers" not in args
        assert res == {"ok": 1}

    @patch("ray.get")
    @patch("ray.util.get_current_placement_group")
    @patch("ray.remote")
    @patch("agentic_rl.runner.infer_service.infer_server.vllm_mp_infer_server.AsyncOpenAI")
    @pytest.mark.asyncio
    async def test_stream(self, mock_openai, mock_remote, mock_pg, mock_ray_get):
        client = self._setup_mocks(mock_pg, mock_remote, mock_openai, mock_ray_get)

        s = self.VLLMMPInferServer("m", model="m", port=1, distributed_executor_backend="mp")

        async def gen():
            m = MagicMock()
            m.model_dump.return_value = {"a": 1}
            yield m

        client.chat.completions.create.return_value = gen()

        chunks = []
        async for c in s.stream_chat_completions({"messages": []}):
            chunks.append(json.loads(c))

        assert chunks == [{"a": 1}]

    @patch("ray.get")
    @patch("ray.util.get_current_placement_group")
    @patch("ray.remote")
    @patch("agentic_rl.runner.infer_service.infer_server.vllm_mp_infer_server.AsyncOpenAI")
    @patch("agentic_rl.runner.infer_service.infer_server.vllm_mp_infer_server.requests")
    @pytest.mark.asyncio
    async def test_rpc(self, mock_req, mock_openai, mock_remote, mock_pg, mock_ray_get):
        client = self._setup_mocks(mock_pg, mock_remote, mock_openai, mock_ray_get)

        s = self.VLLMMPInferServer("m", model="m", port=1, distributed_executor_backend="mp")

        resp = MagicMock()
        resp.status_code = 200
        resp.json.return_value = {"ok": 1}
        mock_req.post.return_value = resp

        r = await s.collective_rpc("m")

        assert r == {"ok": 1}
        assert client is not None

    @patch("ray.get")
    @patch("ray.util.get_current_placement_group")
    @patch("ray.remote")
    @patch("agentic_rl.runner.infer_service.infer_server.vllm_mp_infer_server.AsyncOpenAI")
    @patch("agentic_rl.runner.infer_service.infer_server.vllm_mp_infer_server.requests")
    @pytest.mark.asyncio
    async def test_rpc_non_200_status(self, mock_req, mock_openai, mock_remote, mock_pg, mock_ray_get):
        client = self._setup_mocks(mock_pg, mock_remote, mock_openai, mock_ray_get)

        s = self.VLLMMPInferServer("m", model="m", port=1, distributed_executor_backend="mp")

        resp = MagicMock()
        resp.status_code = 500
        resp.json.return_value = {"error": "internal error"}
        mock_req.post.return_value = resp

        r = await s.collective_rpc("m")

        assert r is None
        assert client is not None

    @patch("ray.get")
    @patch("ray.util.get_current_placement_group")
    @patch("ray.remote")
    @patch("agentic_rl.runner.infer_service.infer_server.vllm_mp_infer_server.AsyncOpenAI")
    @patch("agentic_rl.runner.infer_service.infer_server.vllm_mp_infer_server.requests")
    @pytest.mark.asyncio
    async def test_rpc_exception(self, mock_req, mock_openai, mock_remote, mock_pg, mock_ray_get):
        client = self._setup_mocks(mock_pg, mock_remote, mock_openai, mock_ray_get)

        s = self.VLLMMPInferServer("m", model="m", port=1, distributed_executor_backend="mp")

        mock_req.post.side_effect = Exception("err")

        with pytest.raises(Exception):
            await s.collective_rpc("m")
        assert client is not None
