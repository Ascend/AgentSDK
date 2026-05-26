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
import sys
import types
import io
import json
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch, ANY
import pytest
from fastapi import UploadFile


# ---------------------------------------------------------------------------
# Fixture: fake module tree for train_server
# ---------------------------------------------------------------------------
@pytest.fixture
def fake_train_server_env():
    """Construct isolated fake modules and return the module under test."""

    # ---- fake time ----
    fake_time = types.ModuleType("time")
    fake_time.time = MagicMock(return_value=1000.0)

    # ---- fake ray ----
    fake_ray = types.ModuleType("ray")
    fake_ray.get_actor = MagicMock()

    # ---- fake torch ----
    fake_torch = types.ModuleType("torch")
    fake_torch.save = MagicMock()

    # ---- fake fastapi ----
    fake_fastapi = types.ModuleType("fastapi")
    class FakeAPIRouter:
        def __init__(self, *args, **kwargs):
            self._routes = {}

        def post(self, path: str):
            def decorator(func):
                self._routes[path] = func
                return func
            return decorator

        def get(self, path: str):
            def decorator(func):
                self._routes[path] = func
                return func
            return decorator

    fake_fastapi.APIRouter = FakeAPIRouter

    class FakeUploadFile:
        pass

    fake_fastapi.UploadFile = FakeUploadFile
    fake_fastapi.File = MagicMock()
    fake_fastapi.Form = MagicMock()

    # ---- fake starlette.responses ----
    fake_starlette = types.ModuleType("starlette")
    fake_starlette.__path__ = []
    fake_starlette_responses = types.ModuleType("starlette.responses")
    fake_starlette_responses.Response = MagicMock
    fake_starlette.responses = fake_starlette_responses

    # ---- fake aura loggers ----
    fake_loggers_mod = types.ModuleType("aura.base.log.loggers")
    mock_logger = MagicMock()
    fake_loggers_mod.Loggers = MagicMock(
        return_value=MagicMock(get_logger=MagicMock(return_value=mock_logger))
    )

    # ---- fake TrainQueue ----
    fake_train_queue_mod = types.ModuleType(
        "aura.controllers.train_controller.train_queue"
    )
    fake_train_queue_mod.TrainQueue = MagicMock()

    # ---- fake msg_handler ----
    fake_msg_handler_mod = types.ModuleType(
        "aura.controllers.utils.msg_handler"
    )
    fake_msg_handler_mod.deserialize_and_split = MagicMock(return_value="outputs")

    # ---- aura packages to locate the real file ----
    import os
    import aura as _aura
    base = _aura.__path__[0] if _aura.__path__ else "."
    fake_aura = types.ModuleType("aura")
    fake_aura.__path__ = _aura.__path__
    fake_aura_base = types.ModuleType("aura.base")
    fake_aura_base.__path__ = []
    fake_aura_base_log = types.ModuleType("aura.base.log")
    fake_aura_base_log.__path__ = []
    fake_aura_controllers = types.ModuleType("aura.controllers")
    fake_aura_controllers.__path__ = []
    fake_aura_controllers_train_controller = types.ModuleType(
        "aura.controllers.train_controller"
    )
    fake_aura_controllers_train_controller.__path__ = [
        os.path.join(base, "controllers/train_controller")
    ]
    fake_aura_controllers_utils = types.ModuleType("aura.controllers.utils")
    fake_aura_controllers_utils.__path__ = []

    fakes = {
        "time": fake_time,
        "ray": fake_ray,
        "torch": fake_torch,
        "fastapi": fake_fastapi,
        "starlette": fake_starlette,
        "starlette.responses": fake_starlette_responses,
        "aura.base.log.loggers": fake_loggers_mod,
        "aura.controllers.train_controller.train_queue": fake_train_queue_mod,
        "aura.controllers.utils.msg_handler": fake_msg_handler_mod,
        "aura": fake_aura,
        "aura.base": fake_aura_base,
        "aura.base.log": fake_aura_base_log,
        "aura.controllers": fake_aura_controllers,
        "aura.controllers.train_controller": fake_aura_controllers_train_controller,
        "aura.controllers.utils": fake_aura_controllers_utils,
    }

    target = "aura.controllers.train_controller.train_server"
    if target in sys.modules:
        del sys.modules[target]

    with patch.dict(sys.modules, fakes):
        import aura.controllers.train_controller.train_server as mod
        yield {
            "mod": mod,
            "fake_ray": fake_ray,
            "fake_torch": fake_torch,
            "fake_time": fake_time,
            "mock_logger": mock_logger,
            "fake_train_queue_mod": fake_train_queue_mod,
            "fake_msg_handler_mod": fake_msg_handler_mod,
            "fake_fastapi": fake_fastapi,
            "fake_starlette_responses": fake_starlette_responses,
            "FakeUploadFile": FakeUploadFile,
        }

    if target in sys.modules:
        del sys.modules[target]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def make_train_server(
    fake_ray=None,
    fake_train_queue_mod=None,
    max_queue_size=10,
    global_batch_size=4,
    n_samples_per_prompt=2,
    dispatch_actor_side_effect=None,
):
    """Create a TrainServer instance with optional ray actor override."""
    if fake_ray and dispatch_actor_side_effect:
        fake_ray.get_actor.side_effect = dispatch_actor_side_effect
    from aura.controllers.train_controller.train_server import TrainServer

    if fake_ray:
        fake_ray.get_actor.reset_mock()
    if fake_train_queue_mod:
        fake_train_queue_mod.TrainQueue.reset_mock()

    return TrainServer(max_queue_size, global_batch_size, n_samples_per_prompt)


def make_upload_file_mock(content_bytes: bytes = b"test", upload_file_cls=None):
    """Create a mock UploadFile with async read and close."""
    if upload_file_cls is None:
        upload_file_cls = FakeUploadFile
    mock_file = MagicMock(spec=upload_file_cls)
    mock_file.read = AsyncMock(side_effect=[content_bytes, b""])
    mock_file.close = AsyncMock()
    return mock_file



# ---------------------------------------------------------------------------
# Tests for __init__
# ---------------------------------------------------------------------------
class TestInit:
    def test_init_success(self, fake_train_server_env):
        """TrainServer initialises correctly when ray actor exists."""
        fake_ray = fake_train_server_env["fake_ray"]
        mock_actor = MagicMock()
        fake_ray.get_actor.return_value = mock_actor

        server = make_train_server(fake_ray=fake_ray)

        assert server.dispatch_actor is mock_actor
        assert server.queue_instance is not None
        fake_train_server_env["fake_train_queue_mod"].TrainQueue.assert_called_once_with(
            max_queue_size=10, global_batch_size=4, n_samples_per_prompt=2
        )
        assert isinstance(server.router, fake_train_server_env["fake_fastapi"].APIRouter)
        assert len(server.router._routes) == 4

    def test_init_ray_failure(self, fake_train_server_env):
        """When ray.get_actor raises, error is logged but server is still created without dispatch_actor."""
        fake_ray = fake_train_server_env["fake_ray"]
        fake_ray.get_actor.side_effect = ValueError("actor not found")

        server = make_train_server(fake_ray=fake_ray, dispatch_actor_side_effect=None)
        # dispatch_actor should not be assigned; queue_instance should not exist either
        assert not hasattr(server, "dispatch_actor")
        assert not hasattr(server, "queue_instance")
        fake_train_server_env["mock_logger"].error.assert_called_once()


# ---------------------------------------------------------------------------
# Tests for put_minibatch_to_queue
# ---------------------------------------------------------------------------
class TestPutMinibatchToQueue:
    def test_queue_not_full(self, fake_train_server_env):
        """When queue is not full, send_batch_groups is called."""
        server = make_train_server(fake_ray=fake_train_server_env["fake_ray"])
        server.queue_instance = MagicMock()
        server.queue_instance.add_minibatch.return_value = True
        server.dispatch_actor = MagicMock()

        server.put_minibatch_to_queue("outputs", "metric")
        server.dispatch_actor.send_batch_groups.remote.assert_called_once_with(2)
        server.dispatch_actor.lock_rollout_unit.assert_not_called()

    def test_queue_full(self, fake_train_server_env):
        """When queue is full, lock_rollout_unit is called."""
        server = make_train_server(fake_ray=fake_train_server_env["fake_ray"])
        server.queue_instance = MagicMock()
        server.queue_instance.add_minibatch.return_value = False
        server.dispatch_actor = MagicMock()

        server.put_minibatch_to_queue("outputs", "metric")
        server.dispatch_actor.lock_rollout_unit.assert_called_once()
        server.dispatch_actor.send_batch_groups.remote.assert_not_called()


# ---------------------------------------------------------------------------
# Tests for receive_minibatch
# ---------------------------------------------------------------------------
class TestReceiveMinibatch:
    @pytest.mark.asyncio
    async def test_receive_and_process(self, fake_train_server_env):
        """receive_minibatch reads the file, deserializes, builds metric and puts to queue."""
        server = make_train_server(fake_ray=fake_train_server_env["fake_ray"])
        server.put_minibatch_to_queue = MagicMock()

        content = b"some binary data"
        mock_file = make_upload_file_mock(content, upload_file_cls=fake_train_server_env["FakeUploadFile"])

        rollout_cost = 1.5
        resharding_to_infer = 0.3
        toolcall_reward_mean = 0.8
        toolcall_reward_min = 0.5
        toolcall_reward_max = 0.9
        res_reward_mean = 1.0
        res_reward_min = 0.2
        res_reward_max = 1.5

        result = await server.receive_minibatch(
            file=mock_file,
            rollout_cost=rollout_cost,
            resharding_to_infer=resharding_to_infer,
            toolcall_reward_mean=toolcall_reward_mean,
            toolcall_reward_min=toolcall_reward_min,
            toolcall_reward_max=toolcall_reward_max,
            res_reward_mean=res_reward_mean,
            res_reward_min=res_reward_min,
            res_reward_max=res_reward_max,
        )

        assert result == {"status": "ok"}
        fake_msg = fake_train_server_env["fake_msg_handler_mod"]
        fake_msg.deserialize_and_split.assert_called_once()
        args, _ = fake_msg.deserialize_and_split.call_args
        assert isinstance(args[0], io.BytesIO)

        server.put_minibatch_to_queue.assert_called_once()
        _, metric = server.put_minibatch_to_queue.call_args[0]
        assert metric["rollout_cost"] == rollout_cost
        assert metric["toolcall_reward_mean"] == toolcall_reward_mean
        assert metric["res_reward_max"] == res_reward_max

        mock_file.close.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_receive_minibatch_with_large_file(self, fake_train_server_env):
        """File is read in chunks until empty."""
        server = make_train_server(fake_ray=fake_train_server_env["fake_ray"])
        server.put_minibatch_to_queue = MagicMock()
        chunk1 = b"a" * 512 * 1024
        chunk2 = b"b" * 100
        mock_file = make_upload_file_mock(b"", upload_file_cls=fake_train_server_env["FakeUploadFile"])
        mock_file.read = AsyncMock(side_effect=[chunk1, chunk2, b""])
        mock_file.close = AsyncMock()
        await server.receive_minibatch(
            file=mock_file,
            rollout_cost=0,
            resharding_to_infer=0,
            toolcall_reward_mean=0,
            toolcall_reward_min=0,
            toolcall_reward_max=0,
            res_reward_mean=0,
            res_reward_min=0,
            res_reward_max=0,
        )
        assert mock_file.read.call_count == 3
        fake_train_server_env["fake_msg_handler_mod"].deserialize_and_split.assert_called_once()


# ---------------------------------------------------------------------------
# Tests for is_batch_ready
# ---------------------------------------------------------------------------
class TestIsBatchReady:
    @pytest.mark.asyncio
    async def test_ready(self, fake_train_server_env):
        """Returns True when queue size > 0."""
        server = make_train_server(fake_ray=fake_train_server_env["fake_ray"])
        server.queue_instance = MagicMock()
        server.queue_instance.size.return_value = 3
        result = await server.is_batch_ready()
        assert result == {"is_ready": True}

    @pytest.mark.asyncio
    async def test_not_ready(self, fake_train_server_env):
        """Returns False when queue size == 0."""
        server = make_train_server(fake_ray=fake_train_server_env["fake_ray"])
        server.queue_instance = MagicMock()
        server.queue_instance.size.return_value = 0
        result = await server.is_batch_ready()
        assert result == {"is_ready": False}


# ---------------------------------------------------------------------------
# Tests for pop_minibatch
# ---------------------------------------------------------------------------
class TestPopMinibatch:
    @pytest.mark.asyncio
    async def test_pop_returns_response(self, fake_train_server_env):
        """pop_minibatch returns a Response with binary data and headers."""
        mod = fake_train_server_env["mod"]
        server = make_train_server(fake_ray=fake_train_server_env["fake_ray"])
        server.queue_instance = MagicMock()
        server.queue_instance.pop_batch.return_value = ("outputs", {"key": "val"})

        fake_torch = fake_train_server_env["fake_torch"]
        with patch.object(mod, "Response") as mock_resp:
            result = await server.pop_minibatch()

            fake_torch.save.assert_called_once_with("outputs", ANY)
            mock_resp.assert_called_once()
            call_kwargs = mock_resp.call_args.kwargs
            assert "content" in call_kwargs
            assert "headers" in call_kwargs
            headers = call_kwargs["headers"]
            assert headers["Content-Type"] == "application/octet-stream"
            assert "X-Metrics-Metadata" in headers
            metadata = json.loads(headers["X-Metrics-Metadata"])
            assert metadata == {"key": "val"}


# ---------------------------------------------------------------------------
# Tests for is_ready
# ---------------------------------------------------------------------------
class TestIsReady:
    @pytest.mark.asyncio
    async def test_is_ready_notifies_actor(self, fake_train_server_env):
        """is_ready calls dispatch_actor.set_rollout_unit_ready.remote()."""
        server = make_train_server(fake_ray=fake_train_server_env["fake_ray"])
        server.dispatch_actor = MagicMock()
        server.dispatch_actor.set_rollout_unit_ready.remote = AsyncMock()

        result = await server.is_ready()
        assert result == {"status": "ok"}
        server.dispatch_actor.set_rollout_unit_ready.remote.assert_called_once()
