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
import asyncio
import numpy as np
import os
import pytest
import sys
from unittest.mock import patch, MagicMock, AsyncMock


class TestRepeatInterleave:

    def test_repeat_interleave_torch(self):
        import torch
        from agentic_rl.runner.scheduler.router import _repeat_interleave

        tensor = torch.tensor([1, 2, 3])
        result = _repeat_interleave(tensor, 2)
        expected = torch.tensor([1, 1, 2, 2, 3, 3])
        assert torch.equal(result, expected)

    def test_repeat_interleave_numpy(self):
        from agentic_rl.runner.scheduler.router import _repeat_interleave

        arr = np.array([1, 2, 3])
        result = _repeat_interleave(arr, 2)
        expected = np.array([1, 1, 2, 2, 3, 3])
        np.testing.assert_array_equal(result, expected)


class TestPollCompletionsOpenaiStream:

    @pytest.fixture(autouse=True, scope="function")
    def mock_dependencies(self, monkeypatch):
        """Mock all external dependencies for poll_completions_openai_stream tests."""
        mock_openai = MagicMock()
        mock_openai.RateLimitError = Exception

        mock_types = MagicMock()
        mock_types.chat = MagicMock()
        mock_types.chat.ChatCompletionChunk = MagicMock

        mock_scheduler = MagicMock()
        mock_scheduler.SchedulerFactory = MagicMock()

        mock_misc = MagicMock()
        mock_misc.app_stats = MagicMock()

        mock_globals = MagicMock()
        mock_globals.is_pd_separate = MagicMock(return_value=False)

        monkeypatch.setitem(sys.modules, "openai", mock_openai)
        monkeypatch.setitem(sys.modules, "openai.types", mock_types)
        monkeypatch.setitem(sys.modules, "openai.types.chat", mock_types.chat)
        monkeypatch.setitem(sys.modules, "agentic_rl.runner.scheduler.req_scheduler", mock_scheduler)
        monkeypatch.setitem(sys.modules, "agentic_rl.base.misc.misc", mock_misc)
        monkeypatch.setitem(sys.modules, "agentic_rl.base.utils.globals", mock_globals)
        monkeypatch.delitem(sys.modules, "agentic_rl.runner.scheduler.router", raising=False)

        from agentic_rl.runner.scheduler.router import poll_completions_openai_stream

        self.poll_completions_openai_stream = poll_completions_openai_stream

        yield {
            "openai": mock_openai,
            "scheduler": mock_scheduler,
            "misc": mock_misc,
            "globals": mock_globals,
        }

    @pytest.mark.asyncio
    async def test_poll_completions_openai_stream_success(self, mock_dependencies):
        mock_chunk1 = MagicMock()
        mock_chunk1.choices = [MagicMock()]
        mock_chunk1.choices[0].delta.content = "Hello"
        mock_chunk1.model_dump_json = MagicMock(return_value='{"test": 1}')

        mock_chunk2 = MagicMock()
        mock_chunk2.choices = [MagicMock()]
        mock_chunk2.choices[0].delta.content = " World"
        mock_chunk2.model_dump_json = MagicMock(return_value='{"test": 2}')

        async def mock_stream():
            yield mock_chunk1
            yield mock_chunk2

        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_stream())

        with patch('openai.AsyncOpenAI', return_value=mock_client):
            with patch('agentic_rl.runner.scheduler.router.logger'):
                result = await self.poll_completions_openai_stream(
                    address="192.168.1.1:8080-0",
                    prompt=[{"role": "user", "content": "test"}],
                    model="test-model",
                    max_tokens=100
                )

        assert result == "Hello World"

    @pytest.mark.asyncio
    async def test_poll_completions_openai_stream_with_queue(self, mock_dependencies):
        mock_chunk = MagicMock()
        mock_chunk.choices = [MagicMock()]
        mock_chunk.choices[0].delta.content = "test"
        mock_chunk.model_dump_json = MagicMock(return_value='{"test": 1}')

        async def mock_stream():
            yield mock_chunk

        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_stream())

        stream_queue = asyncio.Queue()

        with patch('openai.AsyncOpenAI', return_value=mock_client):
            with patch('agentic_rl.runner.scheduler.router.logger'):
                result = await self.poll_completions_openai_stream(
                    address="192.168.1.1:8080-0",
                    prompt=[{"role": "user", "content": "test"}],
                    model="test-model",
                    max_tokens=100,
                    stream_queue=stream_queue
                )

        assert result == "test"
        assert not stream_queue.empty()

    @pytest.mark.asyncio
    async def test_poll_completions_openai_stream_with_meta_info(self, mock_dependencies):
        mock_chunk = MagicMock()
        mock_chunk.choices = [MagicMock()]
        mock_chunk.choices[0].delta.content = "test"
        mock_chunk.model_dump_json = MagicMock(return_value='{"test": 1}')

        async def mock_stream():
            yield mock_chunk

        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_stream())

        with patch('openai.AsyncOpenAI', return_value=mock_client):
            with patch('agentic_rl.runner.scheduler.router.logger'):
                result = await self.poll_completions_openai_stream(
                    address="192.168.1.1:8080-0",
                    prompt=[{"role": "user", "content": "test"}],
                    model="test-model",
                    max_tokens=100,
                    meta_info={"key": "value"}
                )

        assert result == "test"

    @pytest.mark.asyncio
    async def test_poll_completions_openai_stream_with_extra_headers(self, mock_dependencies):
        mock_chunk = MagicMock()
        mock_chunk.choices = [MagicMock()]
        mock_chunk.choices[0].delta.content = "test"
        mock_chunk.model_dump_json = MagicMock(return_value='{"test": 1}')

        async def mock_stream():
            yield mock_chunk

        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_stream())

        with patch('openai.AsyncOpenAI', return_value=mock_client):
            with patch('agentic_rl.runner.scheduler.router.logger'):
                result = await self.poll_completions_openai_stream(
                    address="192.168.1.1:8080-0",
                    prompt=[{"role": "user", "content": "test"}],
                    model="test-model",
                    max_tokens=100,
                    extra_headers={"X-Custom": "value"}
                )

        assert result == "test"


class TestPollCompletionsOpenai:

    @pytest.fixture(autouse=True, scope="function")
    def mock_dependencies(self, monkeypatch):
        """Mock all external dependencies for poll_completions_openai tests."""
        mock_openai = MagicMock()
        mock_openai.AsyncOpenAI = MagicMock

        mock_scheduler = MagicMock()
        mock_scheduler.SchedulerFactory = MagicMock()

        mock_misc = MagicMock()
        mock_misc.app_stats = MagicMock()

        mock_globals = MagicMock()
        mock_globals.is_pd_separate = MagicMock(return_value=False)

        monkeypatch.setitem(sys.modules, "openai", mock_openai)
        monkeypatch.setitem(sys.modules, "agentic_rl.runner.scheduler.req_scheduler", mock_scheduler)
        monkeypatch.setitem(sys.modules, "agentic_rl.base.misc.misc", mock_misc)
        monkeypatch.setitem(sys.modules, "agentic_rl.base.utils.globals", mock_globals)
        monkeypatch.delitem(sys.modules, "agentic_rl.runner.scheduler.router", raising=False)

        from agentic_rl.runner.scheduler.router import poll_completions_openai

        self.poll_completions_openai = poll_completions_openai

        yield {
            "openai": mock_openai,
            "scheduler": mock_scheduler,
            "misc": mock_misc,
            "globals": mock_globals,
        }

    @pytest.mark.asyncio
    async def test_poll_completions_openai_success(self, mock_dependencies):
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].text = "test response"
        mock_response.choices[0].logprobs.token_logprobs = [0.1, 0.2]
        mock_response.choices[0].token_ids = [1, 2]
        mock_response.choices[0].prompt_token_ids = [0]

        mock_client = MagicMock()
        mock_client.completions.create = AsyncMock(return_value=mock_response)

        with patch('openai.AsyncOpenAI', return_value=mock_client):
            with patch('agentic_rl.runner.scheduler.router.logger'):
                result = await self.poll_completions_openai(
                    dp_address="192.168.1.1:8080-0",
                    prompt="test prompt",
                    model="test-model",
                    max_tokens=100,
                    request_id="test-request"
                )

        assert result["message"] == "test response"

    @pytest.mark.asyncio
    async def test_poll_completions_openai_with_meta_info(self, mock_dependencies):
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].text = "test response"
        mock_response.choices[0].logprobs.token_logprobs = []
        mock_response.choices[0].token_ids = []
        mock_response.choices[0].prompt_token_ids = []

        mock_client = MagicMock()
        mock_client.completions.create = AsyncMock(return_value=mock_response)

        with patch('openai.AsyncOpenAI', return_value=mock_client):
            with patch('agentic_rl.runner.scheduler.router.logger'):
                result = await self.poll_completions_openai(
                    dp_address="192.168.1.1:8080-0",
                    prompt="test prompt",
                    model="test-model",
                    max_tokens=100,
                    request_id="test-request",
                    meta_info={"key": "value"}
                )

        assert result["message"] == "test response"


class TestRouter:

    @pytest.fixture(autouse=True, scope="function")
    def mock_dependencies(self, monkeypatch):
        """Mock all external dependencies for Router tests."""
        mock_openai = MagicMock()
        mock_scheduler = MagicMock()
        mock_misc = MagicMock()
        mock_globals = MagicMock()
        mock_globals.is_pd_separate = MagicMock(return_value=False)

        monkeypatch.setitem(sys.modules, "openai", mock_openai)
        monkeypatch.setitem(sys.modules, "agentic_rl.runner.scheduler.req_scheduler", mock_scheduler)
        monkeypatch.setitem(sys.modules, "agentic_rl.base.misc.misc", mock_misc)
        monkeypatch.setitem(sys.modules, "agentic_rl.base.utils.globals", mock_globals)
        monkeypatch.delitem(sys.modules, "agentic_rl.runner.scheduler.router", raising=False)

        from agentic_rl.runner.scheduler.router import Router

        self.Router = Router

        mock_tokenizer = MagicMock()
        mock_tokenizer.pad_token_id = 0
        mock_tokenizer.eos_token_id = 1

        self.router = Router(
            tokenizer_name_or_path="/path/to/model",
            tokenizer=mock_tokenizer,
            addresses=["192.168.1.1:8080", "192.168.1.2:8080"],
            model_name="test-model"
        )

        yield {
            "openai": mock_openai,
            "scheduler": mock_scheduler,
            "misc": mock_misc,
            "globals": mock_globals,
        }

    def test_init(self, mock_dependencies):
        assert self.router.dp_size == 1
        assert len(self.router.addresses) == 2
        assert self.router.model_name == "test-model"

    def test_cal_request_id(self, mock_dependencies):
        result = self.Router.cal_request_id("app-1", 0)
        assert result == "app-1--0"

    @pytest.mark.asyncio
    async def test_chat(self, mock_dependencies):
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].text = "test response"
        mock_response.choices[0].logprobs.token_logprobs = []
        mock_response.choices[0].token_ids = []
        mock_response.choices[0].prompt_token_ids = []

        mock_client = MagicMock()
        mock_client.completions.create = AsyncMock(return_value=mock_response)

        with patch('openai.AsyncOpenAI', return_value=mock_client):
            with patch('agentic_rl.runner.scheduler.router.logger'):
                with patch('agentic_rl.runner.scheduler.router.poll_completions_openai', AsyncMock(return_value={"message": "test"})):
                    result = await self.router.chat(
                        prompt="test prompt",
                        application_id="app-1",
                        default_simpling={"temperature": 0.7},
                        step_idx=0
                    )

        assert result is not None

    @pytest.mark.asyncio
    async def test_stop(self, mock_dependencies):
        await self.router.stop()

    def test_reset(self, mock_dependencies):
        self.router.reset()

    @pytest.mark.asyncio
    async def test_cancel_request(self, mock_dependencies):
        await self.router.cancel_request("app-1")


class TestRouterPDSep:

    @pytest.fixture(autouse=True, scope="function")
    def mock_dependencies(self, monkeypatch):
        """Mock all external dependencies for RouterPDSep tests."""
        mock_openai = MagicMock()
        mock_scheduler = MagicMock()

        class MockSchedulerFactory:
            @classmethod
            def get_scheduler(cls, addresses, dp_size, workload_inf, role=""):
                mock_scheduler = MagicMock()
                mock_scheduler.schedule = AsyncMock(return_value="192.168.1.1:8080-0")
                mock_scheduler.release = AsyncMock()
                return mock_scheduler

        mock_scheduler.SchedulerFactory = MockSchedulerFactory

        mock_misc = MagicMock()
        mock_globals = MagicMock()
        mock_globals.is_pd_separate = MagicMock(return_value=True)

        monkeypatch.setitem(sys.modules, "openai", mock_openai)
        monkeypatch.setitem(sys.modules, "agentic_rl.runner.scheduler.req_scheduler", mock_scheduler)
        monkeypatch.setitem(sys.modules, "agentic_rl.base.misc.misc", mock_misc)
        monkeypatch.setitem(sys.modules, "agentic_rl.base.utils.globals", mock_globals)
        monkeypatch.delitem(sys.modules, "agentic_rl.runner.scheduler.router", raising=False)

        from agentic_rl.runner.scheduler.router import RouterPDSep

        self.RouterPDSep = RouterPDSep

        mock_tokenizer = MagicMock()
        mock_tokenizer.pad_token_id = 0
        mock_tokenizer.eos_token_id = 1

        with patch('time.sleep'):
            self.router_pd = RouterPDSep(
                tokenizer_name_or_path="/path/to/model",
                tokenizer=mock_tokenizer,
                addresses=[
                    "prefill-192.168.1.1:8080",
                    "decode-192.168.1.2:8080"
                ],
                model_name="test-model"
            )

        yield {
            "openai": mock_openai,
            "scheduler": mock_scheduler,
            "misc": mock_misc,
            "globals": mock_globals,
        }

    def test_init(self, mock_dependencies):
        assert self.router_pd.dp_size == 1
        assert self.router_pd.model_name == "test-model"

    def test_get_pd_addresses(self, mock_dependencies):
        addresses = [
            "prefill-192.168.1.1:8080",
            "prefill-192.168.1.2:8080",
            "decode-192.168.1.3:8080",
            "decode-192.168.1.4:8080"
        ]
        prefill_addrs, decode_addrs = self.router_pd.get_pd_addresses(addresses)

        assert len(prefill_addrs) == 2
        assert len(decode_addrs) == 2
        assert "192.168.1.1:8080" in prefill_addrs
        assert "192.168.1.3:8080" in decode_addrs

    def test_cal_request_id(self, mock_dependencies):
        result = self.RouterPDSep.cal_request_id("app-1", 0)
        assert result == "app-1--0"

    @pytest.mark.asyncio
    async def test_chat_with_prefill(self, mock_dependencies):
        mock_response = MagicMock()
        mock_response.kv_transfer_params = {"key": "value"}

        with patch('agentic_rl.runner.scheduler.router.poll_completions_openai', AsyncMock(return_value=mock_response)):
            with patch('agentic_rl.runner.scheduler.router.logger'):
                result = await self.router_pd.chat_with_prefill(
                    prompt="test prompt",
                    application_id="app-1",
                    default_sampling={"temperature": 0.7},
                    step_idx=0
                )

        assert result is not None

    @pytest.mark.asyncio
    async def test_stop(self, mock_dependencies):
        await self.router_pd.stop()

    def test_reset(self, mock_dependencies):
        self.router_pd.reset()

    @pytest.mark.asyncio
    async def test_cancel_request(self, mock_dependencies):
        await self.router_pd.cancel_request("app-1")

    @pytest.mark.asyncio
    async def test_chat(self, mock_dependencies):
        mock_prefill_response = MagicMock()
        mock_prefill_response.kv_transfer_params = {"key": "value"}

        mock_decode_chunks = []
        mock_decode_content = []

        async def mock_decode_stream():
            for i in range(3):
                chunk = MagicMock()
                chunk.choices = [MagicMock()]
                chunk.choices[0].delta.content = f"chunk{i}"
                chunk.model_dump_json = MagicMock(return_value=f'{{"chunk": {i}}}')
                mock_decode_chunks.append(chunk)
                mock_decode_content.append(f"chunk{i}")
                yield chunk

        with patch('agentic_rl.runner.scheduler.router.poll_completions_openai', AsyncMock(return_value=mock_prefill_response)):
            with patch('agentic_rl.runner.scheduler.router.poll_completions_openai_stream') as mock_stream:
                mock_stream.return_value = mock_decode_stream()

                with patch('agentic_rl.runner.scheduler.router.logger'):
                    result = await self.router_pd.chat(
                        prompt="test prompt",
                        application_id="app-1",
                        default_sampling={"temperature": 0.0},
                        step_idx=0
                    )

        assert result is not None

    @pytest.mark.asyncio
    async def test_chat_prefill_failed(self, mock_dependencies):
        with patch('agentic_rl.runner.scheduler.router.poll_completions_openai', AsyncMock(return_value=None)):
            with patch('agentic_rl.runner.scheduler.router.logger'):
                result = await self.router_pd.chat(
                    prompt="test prompt",
                    application_id="app-1",
                    default_sampling={"temperature": 1.0},
                    step_idx=1
                )

        assert result is None


class TestRouterCreate:

    @pytest.fixture(autouse=True, scope="function")
    def mock_dependencies(self, monkeypatch):
        """Mock all external dependencies for Router create tests."""
        mock_openai = MagicMock()
        mock_scheduler = MagicMock()
        mock_misc = MagicMock()
        mock_globals = MagicMock()
        mock_globals.is_pd_separate = MagicMock(return_value=False)

        monkeypatch.setitem(sys.modules, "openai", mock_openai)
        monkeypatch.setitem(sys.modules, "agentic_rl.runner.scheduler.req_scheduler", mock_scheduler)
        monkeypatch.setitem(sys.modules, "agentic_rl.base.misc.misc", mock_misc)
        monkeypatch.setitem(sys.modules, "agentic_rl.base.utils.globals", mock_globals)
        monkeypatch.delitem(sys.modules, "agentic_rl.runner.scheduler.router", raising=False)

        from agentic_rl.runner.scheduler.router import Router

        self.Router = Router
        self.Router._router = None

        yield {
            "openai": mock_openai,
            "scheduler": mock_scheduler,
            "misc": mock_misc,
            "globals": mock_globals,
        }

    def test_create_first_time(self, mock_dependencies):
        mock_tokenizer = MagicMock()
        mock_tokenizer.pad_token_id = 0
        mock_tokenizer.eos_token_id = 1

        router = self.Router.create(
            tokenizer_name_or_path="/path/to/model",
            tokenizer=mock_tokenizer,
            addresses=["192.168.1.1:8080"]
        )

        assert router is not None
        assert self.Router._router is router

    def test_create_singleton(self, mock_dependencies):
        mock_tokenizer = MagicMock()
        mock_tokenizer.pad_token_id = 0
        mock_tokenizer.eos_token_id = 1

        router1 = self.Router.create(
            tokenizer_name_or_path="/path/to/model",
            tokenizer=mock_tokenizer,
            addresses=["192.168.1.1:8080"]
        )

        router2 = self.Router.create(
            tokenizer_name_or_path="/path/to/model",
            tokenizer=mock_tokenizer,
            addresses=["192.168.1.2:8080"]
        )

        assert router1 is router2

    def test_create_with_none_addresses(self, mock_dependencies):
        mock_tokenizer = MagicMock()

        router = self.Router.create(
            tokenizer_name_or_path="/path/to/model",
            tokenizer=mock_tokenizer,
            addresses=None
        )

        assert router is None
