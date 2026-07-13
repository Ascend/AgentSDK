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
import os
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
import pytest

# ---------------------------------------------------------------------------
# Fixture: fake module tree for router
# ---------------------------------------------------------------------------
@pytest.fixture
def fake_env():
    """Build isolated fake modules and return the module under test."""

    # ---- fake numpy ----
    fake_numpy = types.ModuleType("numpy")
    fake_numpy.ndarray = type("ndarray", (), {})
    fake_numpy.repeat = MagicMock(name="numpy.repeat")

    # ---- fake torch ----
    fake_torch = types.ModuleType("torch")
    # A concrete Tensor-like class so isinstance checks pass and we can add methods
    class FakeTensor:
        repeat_interleave = MagicMock(name="Tensor.repeat_interleave")
    fake_torch.Tensor = FakeTensor
    fake_torch.nn = types.ModuleType("torch.nn")

    # ---- fake openai ----
    fake_openai = types.ModuleType("openai")
    fake_openai.RateLimitError = type("RateLimitError", (Exception,), {})
    fake_openai.AsyncOpenAI = MagicMock()
    fake_openai.types = types.ModuleType("openai.types")
    fake_openai.types.chat = types.ModuleType("openai.types.chat")
    fake_openai.types.chat.ChatCompletionChunk = MagicMock()
    fake_openai.types.completion = types.ModuleType("openai.types.completion")
    fake_openai.types.completion.Completion = MagicMock()

    # ---- fake httpx ----
    fake_httpx = types.ModuleType("httpx")
    mock_http_client = MagicMock()
    mock_http_client.aclose = AsyncMock()
    fake_httpx.AsyncClient = MagicMock(return_value=mock_http_client)

    # ---- aura packages ----
    import aura as _aura
    base = _aura.__path__[0] if _aura.__path__ else "."
    fake_aura = types.ModuleType("aura")
    fake_aura.__path__ = _aura.__path__
    fake_aura_base = types.ModuleType("aura.base")
    fake_aura_base.__path__ = []
    # aura.base.log.loggers
    fake_aura_base_log = types.ModuleType("aura.base.log")
    fake_aura_base_log.__path__ = [os.path.join(base, "base/log")]
    fake_logger_instance = MagicMock()
    fake_loggers_mod = types.ModuleType("aura.base.log.loggers")
    fake_loggers_mod.Loggers = MagicMock()
    fake_loggers_mod.Loggers.return_value.get_logger.return_value = fake_logger_instance
    # aura.base.misc.misc
    fake_aura_base_misc = types.ModuleType("aura.base.misc")
    fake_aura_base_misc.__path__ = [os.path.join(base, "base/misc")]
    fake_misc_mod = types.ModuleType("aura.base.misc.misc")
    fake_misc_mod.app_stats = MagicMock()
    # aura.base.utils.globals
    fake_aura_base_utils = types.ModuleType("aura.base.utils")
    fake_aura_base_utils.__path__ = [os.path.join(base, "base/utils")]
    fake_globals_mod = types.ModuleType("aura.base.utils.globals")
    fake_globals_mod.is_pd_separate = MagicMock(return_value=False)  # default
    # aura.runner.scheduler.req_scheduler
    fake_aura_runner = types.ModuleType("aura.runner")
    fake_aura_runner.__path__ = [os.path.join(base, "runner")]
    fake_aura_runner_scheduler = types.ModuleType("aura.runner.scheduler")
    fake_aura_runner_scheduler.__path__ = [os.path.join(base, "runner/scheduler")]
    fake_req_scheduler_mod = types.ModuleType("aura.runner.scheduler.req_scheduler")
    fake_req_scheduler_mod.SchedulerFactory = MagicMock()
    mock_scheduler = MagicMock()
    mock_scheduler.schedule = AsyncMock(return_value="addr-0")
    mock_scheduler.release = AsyncMock()
    mock_scheduler.cancel_requests = AsyncMock()
    mock_scheduler.cancel_request = AsyncMock()
    mock_scheduler.reset = MagicMock()
    mock_scheduler.on_ins_address_updated = MagicMock()
    fake_req_scheduler_mod.SchedulerFactory.get_scheduler.return_value = mock_scheduler

    fakes = {
        "numpy": fake_numpy,
        "torch": fake_torch,
        "openai": fake_openai,
        "httpx": fake_httpx,
        "openai.types": fake_openai.types,
        "openai.types.chat": fake_openai.types.chat,
        "openai.types.completion": fake_openai.types.completion,
        "aura": fake_aura,
        "aura.base": fake_aura_base,
        "aura.base.log": fake_aura_base_log,
        "aura.base.log.loggers": fake_loggers_mod,
        "aura.base.misc": fake_aura_base_misc,
        "aura.base.misc.misc": fake_misc_mod,
        "aura.base.utils": fake_aura_base_utils,
        "aura.base.utils.globals": fake_globals_mod,
        "aura.runner": fake_aura_runner,
        "aura.runner.scheduler": fake_aura_runner_scheduler,
        "aura.runner.scheduler.req_scheduler": fake_req_scheduler_mod,
    }

    target = "aura.runner.scheduler.router"
    if target in sys.modules:
        del sys.modules[target]

    with patch.dict(sys.modules, fakes):
        import aura.runner.scheduler.router as mod
        # Patch logger and app_stats in the module for easy access
        mod.logger = fake_logger_instance
        mod.app_stats = fake_misc_mod.app_stats
        yield {
            "mod": mod,
            "fake_numpy": fake_numpy,
            "fake_torch": fake_torch,
            "fake_openai": fake_openai,
            "fake_httpx": fake_httpx,
            "mock_http_client": mock_http_client,
            "mock_scheduler": mock_scheduler,
            "fake_logger": fake_logger_instance,
            "fake_app_stats": fake_misc_mod.app_stats,
            "fake_is_pd_separate": fake_globals_mod.is_pd_separate,
            "fake_SchedulerFactory": fake_req_scheduler_mod.SchedulerFactory,
        }

    if target in sys.modules:
        del sys.modules[target]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def create_mock_stream_chunk(content_text):
    """Create a mock stream chunk with given content."""
    chunk = MagicMock()
    chunk.choices = [MagicMock()]
    chunk.choices[0].delta = MagicMock()
    chunk.choices[0].delta.content = content_text
    chunk.model_dump_json.return_value = f'{{"content": "{content_text}"}}'
    return chunk


async def async_iter(*items):
    for item in items:
        yield item


# ---------------------------------------------------------------------------
# Tests for _repeat_interleave
# ---------------------------------------------------------------------------
class TestRepeatInterleave:
    def test_torch_tensor(self, fake_env):
        mod = fake_env["mod"]
        # FakeTensor already has repeat_interleave as a MagicMock
        fake_tensor = mod.torch.Tensor()
        result = mod._repeat_interleave(fake_tensor, 3)
        fake_tensor.repeat_interleave.assert_called_once_with(3, dim=0)
        assert result == fake_tensor.repeat_interleave.return_value

    def test_numpy_array(self, fake_env):
        mod = fake_env["mod"]
        fake_numpy = fake_env["fake_numpy"]
        fake_arr = fake_numpy.ndarray()
        result = mod._repeat_interleave(fake_arr, 2)
        fake_numpy.repeat.assert_called_once_with(fake_arr, 2, axis=0)
        assert result == fake_numpy.repeat.return_value


# ---------------------------------------------------------------------------
# Tests for poll_completions_openai_stream
# ---------------------------------------------------------------------------
class TestPollCompletionsOpenaiStream:
    @pytest.mark.asyncio
    async def test_success_no_stream_queue(self, fake_env):
        mod = fake_env["mod"]
        fake_openai = fake_env["fake_openai"]
        address = "http://1.2.3.4/v1-0"
        prompt = [{"role": "user", "content": "hello"}]
        model = "test-model"
        max_tokens = 100

        mock_client = AsyncMock()
        mock_client.completions.create = AsyncMock()
        chunk1 = create_mock_stream_chunk("Hello")
        chunk2 = create_mock_stream_chunk(" World")
        mock_client.completions.create.return_value = async_iter(chunk1, chunk2)
        fake_openai.AsyncOpenAI.return_value = mock_client

        result = await mod.poll_completions_openai_stream(
            address, stream_queue=None, prompt=prompt, model=model, max_tokens=max_tokens
        )
        assert result == "Hello World"
        fake_openai.AsyncOpenAI.assert_called_once_with(
            base_url="http://1.2.3.4/v1",
            api_key="EMPTY",
            http_client=fake_env["mock_http_client"],
        )
        fake_env["fake_httpx"].AsyncClient.assert_called_once_with(trust_env=False)
        fake_env["mock_http_client"].aclose.assert_awaited_once()
        mock_client.completions.create.assert_called_once_with(
            messages=prompt, model=model, timeout=3600, stream=True, max_tokens=max_tokens
        )

    @pytest.mark.asyncio
    async def test_with_stream_queue(self, fake_env):
        mod = fake_env["mod"]
        fake_openai = fake_env["fake_openai"]
        stream_queue = MagicMock()
        address = "http://1.2.3.4/v1-0"
        prompt = [{"role": "user", "content": "hi"}]
        model = "test-model"
        max_tokens = 50

        mock_client = AsyncMock()
        mock_client.completions.create = AsyncMock()
        chunk = create_mock_stream_chunk("Hi")
        mock_client.completions.create.return_value = async_iter(chunk)
        fake_openai.AsyncOpenAI.return_value = mock_client

        result = await mod.poll_completions_openai_stream(
            address, stream_queue=stream_queue, prompt=prompt, model=model, max_tokens=max_tokens
        )
        assert result == "Hi"
        stream_queue.put_nowait.assert_called_once()
        args = stream_queue.put_nowait.call_args[0][0]
        assert args["event"] == "raw_response_event"
        assert "response" in args["data"]

    @pytest.mark.asyncio
    async def test_rate_limit_retries_then_succeeds(self, fake_env):
        mod = fake_env["mod"]
        fake_openai = fake_env["fake_openai"]
        address = "http://1.2.3.4/v1-0"
        prompt = [{"role": "user", "content": "test"}]

        mock_client = AsyncMock()
        mock_client.completions.create = AsyncMock()
        mock_client.completions.create.side_effect = [
            fake_openai.RateLimitError("limit"),
            fake_openai.RateLimitError("limit"),
            async_iter(create_mock_stream_chunk("ok")),
        ]
        fake_openai.AsyncOpenAI.return_value = mock_client

        with patch.object(mod, "asyncio") as mock_asyncio:
            mock_asyncio.sleep = AsyncMock()
            result = await mod.poll_completions_openai_stream(
                address, stream_queue=None, prompt=prompt, model="m", max_tokens=10
            )
        assert result == "ok"
        assert mock_asyncio.sleep.call_count == 2

    @pytest.mark.asyncio
    async def test_rate_limit_exhausted(self, fake_env):
        mod = fake_env["mod"]
        fake_openai = fake_env["fake_openai"]
        address = "http://1.2.3.4/v1-0"
        prompt = [{"role": "user", "content": "test"}]

        mock_client = AsyncMock()
        mock_client.completions.create = AsyncMock(side_effect=fake_openai.RateLimitError("limit"))
        fake_openai.AsyncOpenAI.return_value = mock_client

        with patch.object(mod, "asyncio") as mock_asyncio:
            mock_asyncio.sleep = AsyncMock()
            result = await mod.poll_completions_openai_stream(
                address, stream_queue=None, prompt=prompt, model="m", max_tokens=10
            )
        assert "retries exhausted" in result
        assert mock_asyncio.sleep.call_count == 2

    @pytest.mark.asyncio
    async def test_general_exception(self, fake_env):
        mod = fake_env["mod"]
        fake_openai = fake_env["fake_openai"]
        address = "http://1.2.3.4/v1-0"

        mock_client = AsyncMock()
        mock_client.completions.create = AsyncMock(side_effect=Exception("boom"))
        fake_openai.AsyncOpenAI.return_value = mock_client

        result = await mod.poll_completions_openai_stream(
            address, stream_queue=None, prompt=[], model="m", max_tokens=10
        )
        assert "Error processing content" in result

    @pytest.mark.asyncio
    async def test_removes_meta_info_and_extra_headers(self, fake_env):
        mod = fake_env["mod"]
        fake_openai = fake_env["fake_openai"]
        address = "http://1.2.3.4/v1-0"
        mock_client = AsyncMock()
        mock_client.completions.create = AsyncMock(return_value=async_iter())
        fake_openai.AsyncOpenAI.return_value = mock_client

        await mod.poll_completions_openai_stream(
            address,
            stream_queue=None,
            prompt=[{"role": "user", "content": "x"}],
            model="m",
            max_tokens=10,
            meta_info="should_be_removed",
            extra_headers={"h": "v"},
        )
        call_args = mock_client.completions.create.call_args[1]
        assert "meta_info" not in call_args
        assert "extra_headers" not in call_args


# ---------------------------------------------------------------------------
# Tests for poll_completions_openai
# ---------------------------------------------------------------------------
class TestPollCompletionsOpenai:
    @pytest.mark.asyncio
    async def test_success_non_pd_separate(self, fake_env):
        mod = fake_env["mod"]
        fake_is_pd_separate = fake_env["fake_is_pd_separate"]
        fake_is_pd_separate.return_value = False
        fake_openai = fake_env["fake_openai"]
        address = "1.2.3.4:8000-0"
        prompt = "hello prompt"
        model = "test-model"
        max_tokens = 50

        mock_client = AsyncMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].text = "response_text"
        mock_response.choices[0].logprobs = MagicMock()
        mock_response.choices[0].logprobs.token_logprobs = [-0.1, -0.2]
        mock_response.choices[0].token_ids = [1, 2]
        mock_response.choices[0].prompt_token_ids = [3, 4]
        mock_client.completions.create = AsyncMock(return_value=mock_response)
        fake_openai.AsyncOpenAI.return_value = mock_client

        result = await mod.poll_completions_openai(
            address, stream_queue=None, role="h",
            request_id="rid", prompt=prompt, model=model, max_tokens=max_tokens
        )
        assert result["message"] == "response_text"
        assert result["logprobs"] == [-0.1, -0.2]
        assert result["response_tokens"] == [1, 2]
        assert result["prompt_tokens"] == [3, 4]
        call_args = mock_client.completions.create.call_args[1]
        assert call_args["extra_headers"]["X-Request-Id"].startswith("rid-h-")
        assert call_args["extra_headers"]["X-Dp-Rank"] == "0"
        fake_env["mock_http_client"].aclose.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_success_pd_separate_returns_full_response(self, fake_env):
        mod = fake_env["mod"]
        fake_is_pd_separate = fake_env["fake_is_pd_separate"]
        fake_is_pd_separate.return_value = True
        fake_openai = fake_env["fake_openai"]
        address = "1.2.3.4:8000-0"
        prompt = "prompt"
        model = "m"
        max_tokens = 10

        mock_client = AsyncMock()
        mock_response = MagicMock()
        mock_client.completions.create = AsyncMock(return_value=mock_response)
        fake_openai.AsyncOpenAI.return_value = mock_client

        result = await mod.poll_completions_openai(
            address, stream_queue=None, role="h",
            request_id="rid", prompt=prompt, model=model, max_tokens=max_tokens
        )
        assert result is mock_response

    @pytest.mark.asyncio
    async def test_retry_logic(self, fake_env):
        mod = fake_env["mod"]
        fake_openai = fake_env["fake_openai"]
        address = "1.2.3.4:8000-0"
        prompt = "prompt"

        mock_client = AsyncMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].text = "ok"
        mock_client.completions.create = AsyncMock(side_effect=[
            Exception("fail1"),
            mock_response,
        ])
        fake_openai.AsyncOpenAI.return_value = mock_client
        fake_is_pd_separate = fake_env["fake_is_pd_separate"]
        fake_is_pd_separate.return_value = False

        result = await mod.poll_completions_openai(
            address, stream_queue=None, role="h",
            request_id="rid", prompt=prompt, model="m", max_tokens=10
        )
        assert result["message"] == "ok"
        assert mock_client.completions.create.call_count == 2

    @pytest.mark.asyncio
    async def test_retries_exhausted(self, fake_env):
        mod = fake_env["mod"]
        fake_openai = fake_env["fake_openai"]
        address = "1.2.3.4:8000-0"

        mock_client = AsyncMock()
        mock_client.completions.create = AsyncMock(side_effect=Exception("fail"))
        fake_openai.AsyncOpenAI.return_value = mock_client

        result = await mod.poll_completions_openai(
            address, stream_queue=None, role="h",
            request_id="rid", prompt="p", model="m", max_tokens=10
        )
        assert "retries exhausted" in result

    @pytest.mark.asyncio
    async def test_address_without_dash(self, fake_env):
        mod = fake_env["mod"]
        fake_is_pd_separate = fake_env["fake_is_pd_separate"]
        fake_is_pd_separate.return_value = False
        fake_openai = fake_env["fake_openai"]
        address = "1.2.3.4:8000"  # no dash
        mock_client = AsyncMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].text = "ok"
        mock_client.completions.create = AsyncMock(return_value=mock_response)
        fake_openai.AsyncOpenAI.return_value = mock_client

        result = await mod.poll_completions_openai(
            address, stream_queue=None, role="h",
            request_id="rid", prompt="p", model="m", max_tokens=10
        )
        call_args = mock_client.completions.create.call_args[1]
        assert call_args["extra_headers"]["X-Dp-Rank"] == 0  # integer
        assert call_args["extra_headers"]["X-Request-Id"].startswith("rid-h-")


# ---------------------------------------------------------------------------
# Tests for poll_chat_completions_openai
# ---------------------------------------------------------------------------
class TestPollChatCompletionsOpenai:
    @pytest.mark.asyncio
    async def test_success_non_pd_separate(self, fake_env):
        mod = fake_env["mod"]
        fake_is_pd_separate = fake_env["fake_is_pd_separate"]
        fake_is_pd_separate.return_value = False
        fake_openai = fake_env["fake_openai"]
        address = "1.2.3.4:8000-0"
        messages = [{"role": "user", "content": "hi"}]
        model = "chat-model"
        max_tokens = 100

        mock_client = AsyncMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "chat response"
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
        fake_openai.AsyncOpenAI.return_value = mock_client

        result = await mod.poll_chat_completions_openai(
            address, stream_queue=None, role="h",
            request_id="rid", prompt=messages, model=model, max_tokens=max_tokens
        )
        assert result == "chat response"
        call_args = mock_client.chat.completions.create.call_args[1]
        assert call_args["extra_headers"]["X-Request-Id"].startswith("rid-h-")
        assert call_args["extra_headers"]["X-Dp-Rank"] == "0"

    @pytest.mark.asyncio
    async def test_pd_separate_returns_full_response(self, fake_env):
        mod = fake_env["mod"]
        fake_is_pd_separate = fake_env["fake_is_pd_separate"]
        fake_is_pd_separate.return_value = True
        fake_openai = fake_env["fake_openai"]
        address = "1.2.3.4:8000-0"
        messages = [{"role": "user", "content": "hi"}]

        mock_client = AsyncMock()
        mock_response = MagicMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
        fake_openai.AsyncOpenAI.return_value = mock_client

        result = await mod.poll_chat_completions_openai(
            address, stream_queue=None, role="h",
            request_id="rid", prompt=messages, model="m", max_tokens=10
        )
        assert result is mock_response

    @pytest.mark.asyncio
    async def test_retry_logic(self, fake_env):
        mod = fake_env["mod"]
        fake_openai = fake_env["fake_openai"]
        address = "1.2.3.4:8000-0"

        mock_client = AsyncMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "ok after retry"
        mock_client.chat.completions.create = AsyncMock(side_effect=[
            Exception("fail"),
            mock_response,
        ])
        fake_openai.AsyncOpenAI.return_value = mock_client

        result = await mod.poll_chat_completions_openai(
            address, stream_queue=None, role="h",
            request_id="rid", prompt=[{}], model="m", max_tokens=10
        )
        assert result == "ok after retry"


# ---------------------------------------------------------------------------
# Tests for Router class
# ---------------------------------------------------------------------------
class TestRouter:
    def test_init(self, fake_env):
        mod = fake_env["mod"]
        tokenizer = MagicMock()
        tokenizer.pad_token_id = 0
        tokenizer.eos_token_id = 1
        with patch.dict(os.environ, {"VLLM_DP_SIZE": "4"}, clear=True):
            router = mod.Router(
                tokenizer_name_or_path="/path/to/model",
                tokenizer=tokenizer,
                addresses=["addr1", "addr2"],
                token_in_token_out=True,
                model_name="custom_name"
            )
        assert router.addresses == ["addr1", "addr2"]
        assert router.dp_size == 4
        assert router.counter == 0
        assert router.tokenizer is tokenizer
        assert router.pad_token_id == 0
        assert router.eos_token_id == 1
        assert router.model_path == "/path/to/model"
        assert router.model_name == "custom_name"
        assert router.token_in_token_out is True

    def test_init_default_model_name(self, fake_env):
        mod = fake_env["mod"]
        tokenizer = MagicMock()
        router = mod.Router(
            tokenizer_name_or_path="/a/b/c",
            tokenizer=tokenizer,
            addresses=[],
            token_in_token_out=False
        )
        assert router.model_name == "b/c"

    def test_create_singleton_router(self, fake_env):
        mod = fake_env["mod"]
        fake_is_pd_separate = fake_env["fake_is_pd_separate"]
        fake_is_pd_separate.return_value = False
        mod.Router._router = None
        router1 = mod.Router.create(
            tokenizer_name_or_path="/a/b",
            tokenizer=MagicMock(),
            addresses=["addr"]
        )
        router2 = mod.Router.create(
            tokenizer_name_or_path="/a/b",
            tokenizer=MagicMock(),
            addresses=["addr"]
        )
        assert router1 is router2
        assert isinstance(router1, mod.Router)

    def test_create_pd_separate_router(self, fake_env):
        mod = fake_env["mod"]
        fake_is_pd_separate = fake_env["fake_is_pd_separate"]
        fake_is_pd_separate.return_value = True
        mod.Router._router = None
        router = mod.Router.create(
            tokenizer_name_or_path="/a/b",
            tokenizer=MagicMock(),
            addresses=["addr"]
        )
        assert isinstance(router, mod.RouterPDSep)

    def test_create_none_addresses(self, fake_env):
        mod = fake_env["mod"]
        mod.Router._router = None
        router = mod.Router.create(
            tokenizer_name_or_path="/a/b",
            tokenizer=MagicMock(),
            addresses=None
        )
        assert router is None

    def test_cal_request_id(self, fake_env):
        mod = fake_env["mod"]
        rid = mod.Router.cal_request_id("app123", 5)
        assert rid == "app123--5"

    def test_update_address(self, fake_env):
        mod = fake_env["mod"]
        router = mod.Router("/p", MagicMock(), ["old"])
        router.update_address(["new"])
        assert router.addresses == ["new"]
        router.scheduler.on_ins_address_updated.assert_called_once_with(["new"], 1)

    def test_update_address_empty_warns_and_skips(self, fake_env):
        mod = fake_env["mod"]
        router = mod.Router("/p", MagicMock(), ["old"])
        router.update_address([])
        assert router.addresses == ["old"]
        router.scheduler.on_ins_address_updated.assert_not_called()
        fake_env["fake_logger"].warning.assert_called_once()

    @pytest.mark.asyncio
    async def test_chat_token_in_token_out_no_stream_queue(self, fake_env):
        mod = fake_env["mod"]
        tokenizer = MagicMock()
        addresses = ["addr-0"]
        router = mod.Router("/p", tokenizer, addresses, token_in_token_out=True)
        with patch.object(mod, "poll_completions_openai", new=AsyncMock(return_value="response1")) as mock_poll_comp, \
             patch.object(mod, "poll_chat_completions_openai") as mock_poll_chat, \
             patch.object(mod, "poll_completions_openai_stream") as mock_poll_stream, \
             patch.object(mod, "app_stats") as mock_app_stats, \
             patch.object(mod, "time") as mock_time:
            mock_time.time.return_value = 12345
            result = await router.chat(
                prompt="hello",
                application_id="app1",
                default_simpling={"temperature": 0.7},
                stream_queue=None,
                step_idx=0,
                extra_arg="extra"
            )
        assert result == "response1"
        mock_poll_comp.assert_called_once()
        mock_poll_chat.assert_not_called()
        mock_poll_stream.assert_not_called()
        mock_app_stats.stat_route.assert_called_once()

    @pytest.mark.asyncio
    async def test_chat_token_in_token_out_with_stream_queue(self, fake_env):
        mod = fake_env["mod"]
        router = mod.Router("/p", MagicMock(), ["addr-0"], token_in_token_out=True)
        stream_queue = MagicMock()
        with patch.object(mod, "poll_completions_openai_stream", new=AsyncMock(return_value="streamed")) as mock_stream, \
             patch.object(mod, "app_stats"):
            result = await router.chat(
                prompt="hello",
                application_id="app1",
                default_simpling={},
                stream_queue=stream_queue,
                step_idx=0
            )
        assert result == "streamed"
        mock_stream.assert_called_once()

    @pytest.mark.asyncio
    async def test_chat_not_token_in_token_out_no_stream_queue(self, fake_env):
        mod = fake_env["mod"]
        router = mod.Router("/p", MagicMock(), ["addr-0"], token_in_token_out=False)
        with patch.object(mod, "poll_chat_completions_openai", new=AsyncMock(return_value="chat_resp")) as mock_chat, \
             patch.object(mod, "poll_completions_openai") as mock_comp, \
             patch.object(mod, "app_stats"):
            result = await router.chat(
                prompt="hello",
                application_id="app1",
                default_simpling={},
                stream_queue=None,
                step_idx=0
            )
        assert result == "chat_resp"
        mock_chat.assert_called_once()
        mock_comp.assert_not_called()

    @pytest.mark.asyncio
    async def test_chat_address_none(self, fake_env):
        mod = fake_env["mod"]
        router = mod.Router("/p", MagicMock(), [None], token_in_token_out=True)
        router.scheduler.schedule.return_value = None
        result = await router.chat(prompt="", application_id="", default_simpling={}, stream_queue=None, step_idx=0)
        assert result is None

    @pytest.mark.asyncio
    async def test_stop_reset_cancel(self, fake_env):
        router = fake_env["mod"].Router("/p", MagicMock(), [])
        await router.stop()
        router.reset()
        await router.cancel_request("app")
        router.scheduler.cancel_requests.assert_awaited_once()
        router.scheduler.reset.assert_called_once()
        router.scheduler.cancel_request.assert_awaited_once_with("app")


# ---------------------------------------------------------------------------
# Tests for RouterPDSep
# ---------------------------------------------------------------------------
class TestRouterPDSep:
    def test_get_pd_addresses(self, fake_env):
        mod = fake_env["mod"]
        router = mod.RouterPDSep.__new__(mod.RouterPDSep)
        addresses = [
            "prefill-192.168.1.1:8000",
            "prefill-192.168.1.2:8000",
            "decode-192.168.1.3:9000",
            "decode-192.168.1.4:9000",
            "other-192.168.1.5",
        ]
        p, d = router.get_pd_addresses(addresses)
        assert p == ["192.168.1.1:8000", "192.168.1.2:8000"]
        assert d == ["192.168.1.3:9000", "192.168.1.4:9000"]

    def test_init(self, fake_env):
        mod = fake_env["mod"]
        fake_SchedulerFactory = fake_env["fake_SchedulerFactory"]
        mock_p_scheduler = AsyncMock()
        mock_d_scheduler = AsyncMock()
        fake_SchedulerFactory.get_scheduler.side_effect = [mock_p_scheduler, mock_d_scheduler]
        tokenizer = MagicMock()
        tokenizer.pad_token_id = 0
        tokenizer.eos_token_id = 1
        addresses = ["prefill-1.1.1.1:80", "decode-2.2.2.2:80"]
        with patch.dict(os.environ, {"VLLM_DP_SIZE": "2"}, clear=True):
            router = mod.RouterPDSep(
                tokenizer_name_or_path="/x/y",
                tokenizer=tokenizer,
                addresses=addresses,
                model_name="custom_model"
            )
        assert router.dp_size == 2
        assert router.tokenizer is tokenizer
        assert router.pad_token_id == 0
        assert router.eos_token_id == 1
        assert router.model_path == "/x/y"
        assert router.model_name == "custom_model"
        fake_SchedulerFactory.get_scheduler.assert_any_call(["1.1.1.1:80"], 2, None, role="prefill")
        fake_SchedulerFactory.get_scheduler.assert_any_call(["2.2.2.2:80"], 2, None, role="decode")

    @pytest.mark.asyncio
    async def test_chat_with_prefill_success(self, fake_env):
        mod = fake_env["mod"]
        router = mod.RouterPDSep.__new__(mod.RouterPDSep)
        router.p_scheduler = AsyncMock()
        router.p_scheduler.schedule.return_value = "prefill_addr"
        router.p_scheduler.release = AsyncMock()
        router.model_path = "/a/b"
        router.cal_request_id = MagicMock(return_value="req-id")

        mock_prefill_response = MagicMock()
        with patch.object(mod, "poll_completions_openai", new=AsyncMock(return_value=mock_prefill_response)) as mock_poll, \
             patch.object(mod, "time") as mock_time:
            mock_time.time.side_effect = [100, 101, 102, 103]
            result = await router.chat_with_prefill(
                prompt="prompt",
                application_id="app",
                default_sampling={"temperature": 0.5},
                step_idx=3
            )
        resp, sched_time, prefill_time = result
        assert resp is mock_prefill_response
        assert sched_time == 1
        assert prefill_time == 2   # 103 - 101 = 2
        router.p_scheduler.release.assert_called_once_with("prefill_addr", "app", "req-id")

    @pytest.mark.asyncio
    async def test_chat_with_prefill_schedule_failure(self, fake_env):
        mod = fake_env["mod"]
        router = mod.RouterPDSep.__new__(mod.RouterPDSep)
        router.p_scheduler = AsyncMock()
        router.p_scheduler.schedule.return_value = None
        result = await router.chat_with_prefill(
            prompt="p", application_id="a", default_sampling={}, step_idx=0
        )
        assert result is None

    @pytest.mark.asyncio
    async def test_chat_full_flow(self, fake_env):
        mod = fake_env["mod"]
        router = mod.RouterPDSep.__new__(mod.RouterPDSep)
        router.chat_with_prefill = AsyncMock(return_value=(
            MagicMock(kv_transfer_params={"key": "val"}),
            0.1, 0.2
        ))
        router.d_scheduler = AsyncMock()
        router.d_scheduler.schedule.return_value = "decode_addr"
        router.d_scheduler.release = AsyncMock()
        router.cal_request_id = MagicMock(return_value="req-id")
        router.model_path = "/a/b"
        router.model_name = "my_model"

        mock_decode_response = MagicMock()
        mock_decode_response.choices = [MagicMock()]
        mock_decode_response.choices[0].message.content = "final answer"
        with patch.object(mod, "poll_completions_openai", new=AsyncMock(return_value=mock_decode_response)) as mock_poll, \
             patch.object(mod, "time") as mock_time:
            # time.time() calls: chat() has 4 calls (sched_start_time, decode_start_time, end_time, and in logger.info)
            mock_time.time.side_effect = [200, 201, 202, 203]
            result = await router.chat(
                prompt="p",
                application_id="app",
                default_sampling={},
                stream_queue=None,
                step_idx=1
            )
        assert result == "final answer"
        mock_poll.assert_called_once()
        call_kwargs = mock_poll.call_args[1]
        assert call_kwargs["kv_transfer_params"] == {"key": "val"}
        assert call_kwargs["role"] == 'd'
        router.d_scheduler.release.assert_called_once_with("decode_addr", "app", "req-id")

    @pytest.mark.asyncio
    async def test_chat_with_stream_queue(self, fake_env):
        mod = fake_env["mod"]
        router = mod.RouterPDSep.__new__(mod.RouterPDSep)
        router.chat_with_prefill = AsyncMock(return_value=(
            MagicMock(kv_transfer_params={}),
            0.1, 0.2
        ))
        router.d_scheduler = AsyncMock()
        router.d_scheduler.schedule.return_value = "decode_addr"
        router.d_scheduler.release = AsyncMock()
        router.model_path = "/a/b"
        router.model_name = "stream_model"  # added model_name
        stream_queue = MagicMock()
        with patch.object(mod, "poll_completions_openai_stream", new=AsyncMock(return_value="stream_out")) as mock_stream, \
             patch.object(mod, "time") as mock_time:
            mock_time.time.side_effect = [100, 101, 102, 103]
            result = await router.chat(
                prompt="p",
                application_id="app",
                default_sampling={},
                stream_queue=stream_queue,
                step_idx=2
            )
        assert result == "stream_out"
        mock_stream.assert_called_once()
        assert mock_stream.call_args[1]["stream_queue"] is stream_queue
        assert "kv_transfer_params" in mock_stream.call_args[1]

    @pytest.mark.asyncio
    async def test_chat_prefill_returns_none_response(self, fake_env):
        """If prefill returns a tuple with None as first element, chat returns None."""
        mod = fake_env["mod"]
        router = mod.RouterPDSep.__new__(mod.RouterPDSep)
        # chat_with_prefill returns (None, 0, 0) to simulate failure after schedule
        router.chat_with_prefill = AsyncMock(return_value=(None, 0, 0))
        result = await router.chat(
            prompt="p",
            application_id="app",
            default_sampling={},
            stream_queue=None,
            step_idx=0
        )
        assert result is None

    @pytest.mark.asyncio
    async def test_chat_decode_schedule_failure(self, fake_env):
        mod = fake_env["mod"]
        router = mod.RouterPDSep.__new__(mod.RouterPDSep)
        router.chat_with_prefill = AsyncMock(return_value=(
            MagicMock(kv_transfer_params={}),
            0.1, 0.2
        ))
        router.d_scheduler = AsyncMock()
        router.d_scheduler.schedule.return_value = None
        result = await router.chat(
            prompt="p",
            application_id="app",
            default_sampling={},
            stream_queue=None,
            step_idx=0
        )
        assert result is None
