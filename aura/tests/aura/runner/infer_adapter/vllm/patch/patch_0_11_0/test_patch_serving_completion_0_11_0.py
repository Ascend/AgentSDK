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
import sys
import types
from unittest.mock import AsyncMock, MagicMock, patch
import pytest
import asyncio


# ---------------------------------------------------------------------------
# Fixture: fake module tree for patch_serving_completion
# ---------------------------------------------------------------------------
@pytest.fixture
def fake_serving_completion_env():
    # ---- Fake jinja2 ----
    fake_jinja2 = types.ModuleType("jinja2")
    class FakeTemplateError(Exception):
        pass
    fake_jinja2.TemplateError = FakeTemplateError

    # ---- Fake fastapi ----
    fake_fastapi = types.ModuleType("fastapi")
    fake_fastapi.Request = MagicMock

    # ---- Fake typing_extensions ----
    fake_typing_extensions = types.ModuleType("typing_extensions")
    fake_typing_extensions.assert_never = MagicMock()

    # ---- Fake vllm.entrypoints.openai.protocol ----
    fake_vllm_ep_openai_protocol = types.ModuleType("vllm.entrypoints.openai.protocol")
    fake_vllm_ep_openai_protocol.CompletionRequest = MagicMock
    fake_vllm_ep_openai_protocol.CompletionResponse = MagicMock
    fake_vllm_ep_openai_protocol.ErrorResponse = MagicMock
    fake_vllm_ep_openai_protocol.RequestResponseMetadata = MagicMock

    # ---- Fake vllm.logger ----
    fake_vllm_logger = types.ModuleType("vllm.logger")
    fake_vllm_logger.init_logger = MagicMock(return_value=MagicMock())

    # ---- Fake vllm.entrypoints.utils ----
    fake_vllm_ep_utils = types.ModuleType("vllm.entrypoints.utils")
    fake_vllm_ep_utils.get_max_tokens = MagicMock(return_value=100)

    # ---- Fake vllm.inputs.data ----
    fake_vllm_inputs_data = types.ModuleType("vllm.inputs.data")
    fake_vllm_inputs_data.is_embeds_prompt = lambda p: "prompt_embeds" in p
    fake_vllm_inputs_data.is_tokens_prompt = lambda p: "prompt_token_ids" in p
    fake_vllm_inputs_data.EmbedsPrompt = dict
    fake_vllm_inputs_data.TokensPrompt = dict

    # ---- Fake vllm.outputs ----
    fake_vllm_outputs = types.ModuleType("vllm.outputs")
    fake_vllm_outputs.RequestOutput = MagicMock

    # ---- Fake vllm.sampling_params (real classes for isinstance) ----
    fake_vllm_sampling_params = types.ModuleType("vllm.sampling_params")
    class FakeBeamSearchParams:
        pass
    class FakeSamplingParams:
        pass
    fake_vllm_sampling_params.BeamSearchParams = FakeBeamSearchParams
    fake_vllm_sampling_params.SamplingParams = FakeSamplingParams

    # ---- Fake vllm.utils ----
    fake_vllm_utils = types.ModuleType("vllm.utils")
    fake_vllm_utils.merge_async_iterators = MagicMock()

    # ---- Fake vllm.entrypoints.openai.serving_completion ----
    fake_vllm_serving_comp = types.ModuleType("vllm.entrypoints.openai.serving_completion")
    class FakeOpenAIServingCompletion:
        pass
    fake_vllm_serving_comp.OpenAIServingCompletion = FakeOpenAIServingCompletion

    # ---- Fake vllm packages (with __path__ to avoid import errors) ----
    fake_vllm = types.ModuleType("vllm")
    fake_vllm.__path__ = []
    fake_vllm_entrypoints = types.ModuleType("vllm.entrypoints")
    fake_vllm_entrypoints.__path__ = []
    fake_vllm_ep_openai = types.ModuleType("vllm.entrypoints.openai")
    fake_vllm_ep_openai.__path__ = []
    fake_vllm_inputs = types.ModuleType("vllm.inputs")
    fake_vllm_inputs.__path__ = []
    fake_vllm_sampling = types.ModuleType("vllm.sampling")
    fake_vllm_sampling.__path__ = []
    fake_vllm_distributed = types.ModuleType("vllm.distributed")
    fake_vllm_distributed.__path__ = []

    # ---- Aura packages ----
    import os as _os
    import aura as _aura
    real_aura_path = _aura.__path__
    base_path = real_aura_path[0] if real_aura_path else "."

    fake_aura = types.ModuleType("aura")
    fake_aura.__path__ = real_aura_path
    fake_aura_runner = types.ModuleType("aura.runner")
    fake_aura_runner.__path__ = [_os.path.join(base_path, "runner")]
    fake_aura_runner_infer_adapter = types.ModuleType("aura.runner.infer_adapter")
    fake_aura_runner_infer_adapter.__path__ = [_os.path.join(base_path, "runner/infer_adapter")]
    fake_vllm_pkg = types.ModuleType("aura.runner.infer_adapter.vllm")
    fake_vllm_pkg.__path__ = [_os.path.join(base_path, "runner/infer_adapter/vllm")]
    fake_patch_pkg = types.ModuleType("aura.runner.infer_adapter.vllm.patch")
    fake_patch_pkg.__path__ = [_os.path.join(base_path, "runner/infer_adapter/vllm/patch")]
    fake_0_11_0_pkg = types.ModuleType("aura.runner.infer_adapter.vllm.patch.patch_0_11_0")
    fake_0_11_0_pkg.__path__ = [_os.path.join(base_path, "runner/infer_adapter/vllm/patch/patch_0_11_0")]

    all_fakes = {
        "jinja2": fake_jinja2,
        "fastapi": fake_fastapi,
        "typing_extensions": fake_typing_extensions,
        "vllm": fake_vllm,
        "vllm.entrypoints": fake_vllm_entrypoints,
        "vllm.entrypoints.openai": fake_vllm_ep_openai,
        "vllm.entrypoints.openai.protocol": fake_vllm_ep_openai_protocol,
        "vllm.logger": fake_vllm_logger,
        "vllm.entrypoints.utils": fake_vllm_ep_utils,
        "vllm.inputs": fake_vllm_inputs,
        "vllm.inputs.data": fake_vllm_inputs_data,
        "vllm.outputs": fake_vllm_outputs,
        "vllm.sampling": fake_vllm_sampling,
        "vllm.sampling_params": fake_vllm_sampling_params,
        "vllm.utils": fake_vllm_utils,
        "vllm.entrypoints.openai.serving_completion": fake_vllm_serving_comp,
        "vllm.distributed": fake_vllm_distributed,
        "aura": fake_aura,
        "aura.runner": fake_aura_runner,
        "aura.runner.infer_adapter": fake_aura_runner_infer_adapter,
        "aura.runner.infer_adapter.vllm": fake_vllm_pkg,
        "aura.runner.infer_adapter.vllm.patch": fake_patch_pkg,
        "aura.runner.infer_adapter.vllm.patch.patch_0_11_0": fake_0_11_0_pkg,
    }
    for name, mod in all_fakes.items():
        sys.modules[name] = mod

    # Ensure fresh import of the module under test
    target_module = "aura.runner.infer_adapter.vllm.patch.patch_0_11_0.patch_serving_completion"
    if target_module in sys.modules:
        del sys.modules[target_module]

    yield {
        "jinja2": fake_jinja2,
        "fastapi": fake_fastapi,
        "typing_extensions": fake_typing_extensions,
        "vllm_ep_openai_protocol": fake_vllm_ep_openai_protocol,
        "vllm_logger": fake_vllm_logger,
        "vllm_ep_utils": fake_vllm_ep_utils,
        "vllm_inputs_data": fake_vllm_inputs_data,
        "vllm_outputs": fake_vllm_outputs,
        "vllm_sampling_params": fake_vllm_sampling_params,
        "vllm_utils": fake_vllm_utils,
        "vllm_serving_comp": fake_vllm_serving_comp,
        "openai_serving_cls": fake_vllm_serving_comp.OpenAIServingCompletion,
        "BeamSearchParams": FakeBeamSearchParams,
        "SamplingParams": FakeSamplingParams,
    }

    # Cleanup
    for name in list(all_fakes.keys()):
        if name in sys.modules:
            del sys.modules[name]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def make_raw_request(dp_rank=None):
    raw = MagicMock()
    raw.headers = MagicMock()
    raw.headers.get.return_value = dp_rank
    raw.state.request_metadata = None
    return raw


def make_self_mock(fake_env):
    self = MagicMock()
    self._check_model = AsyncMock(return_value=None)
    self.engine_client = MagicMock()
    self.engine_client.errored = False
    self.engine_client.dead_error = Exception("dead")
    self.engine_client.get_tokenizer = AsyncMock(return_value="tokenizer")
    self.engine_client.beam_search = MagicMock()
    self.engine_client.generate = MagicMock()
    self.create_error_response = MagicMock(side_effect=lambda msg: f"error:{msg}")
    self._base_request_id = MagicMock(return_value="base_id")
    self._maybe_get_adapters = MagicMock(return_value=None)
    self.model_config = MagicMock()
    self.model_config.skip_tokenizer_init = False
    self.model_config.logits_processor_pattern = None
    self._get_renderer = MagicMock()
    self._build_render_config = MagicMock(return_value={})
    self.default_sampling_params = {}
    self.max_model_len = 2048
    self._log_inputs = MagicMock()
    self._get_trace_headers = AsyncMock(return_value={})
    self.models = MagicMock()
    self.models.model_name = MagicMock(return_value="test-model")
    self.completion_stream_generator = MagicMock()
    self.request_output_to_completion_response = MagicMock()
    self.enable_force_include_usage = False
    return self


def make_renderer_mock():
    renderer = MagicMock()
    renderer.render_prompt_and_embeds = AsyncMock()
    return renderer


def make_request(fake_env, **kwargs):
    """Create a mock CompletionRequest with proper return types for to_* methods."""
    req = MagicMock()
    req.suffix = None
    req.echo = False
    req.prompt_embeds = None
    req.prompt_logprobs = None
    req.prompt = "test prompt"
    req.request_id = None
    req.use_beam_search = False
    req.best_of = None
    req.n = 1
    req.stream = False
    req.priority = 0
    # Use real classes for isinstance checks
    req.to_beam_search_params = MagicMock(return_value=fake_env["BeamSearchParams"]())
    req.to_sampling_params = MagicMock(return_value=fake_env["SamplingParams"]())
    for k, v in kwargs.items():
        setattr(req, k, v)
    return req


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
class TestCreateCompletionPatch:
    @pytest.fixture(autouse=True)
    def setup_method(self, fake_serving_completion_env):
        self.fake_env = fake_serving_completion_env
        import aura.runner.infer_adapter.vllm.patch.patch_0_11_0.patch_serving_completion as mod
        self.mod = mod
        self.patch_func = mod.create_completion_patch

    # ---- 1. _check_model error ----
    @pytest.mark.asyncio
    async def test_check_model_error(self):
        self_mock = make_self_mock(self.fake_env)
        self_mock._check_model.return_value = "error_resp"
        req = make_request(self.fake_env)
        raw = make_raw_request()
        result = await self.patch_func(self_mock, req, raw)
        assert result == "error_resp"

    # ---- 2. Engine errored ----
    @pytest.mark.asyncio
    async def test_engine_client_errored(self):
        self_mock = make_self_mock(self.fake_env)
        self_mock.engine_client.errored = True
        req = make_request(self.fake_env)
        raw = make_raw_request()
        with pytest.raises(Exception, match="dead"):
            await self.patch_func(self_mock, req, raw)

    # ---- 3. suffix not supported ----
    @pytest.mark.asyncio
    async def test_suffix_not_supported(self):
        self_mock = make_self_mock(self.fake_env)
        req = make_request(self.fake_env, suffix="some suffix")
        raw = make_raw_request()
        result = await self.patch_func(self_mock, req, raw)
        assert "suffix is not currently supported" in result

    # ---- 4. echo with prompt embeds ----
    @pytest.mark.asyncio
    async def test_echo_with_prompt_embeds(self):
        self_mock = make_self_mock(self.fake_env)
        req = make_request(self.fake_env, echo=True, prompt_embeds=[1,2,3])
        raw = make_raw_request()
        result = await self.patch_func(self_mock, req, raw)
        assert "Echo is unsupported with prompt embeds" in result

    # ---- 5. prompt_logprobs with prompt embeds ----
    @pytest.mark.asyncio
    async def test_prompt_logprobs_with_prompt_embeds(self):
        self_mock = make_self_mock(self.fake_env)
        req = make_request(self.fake_env, prompt_logprobs=1, prompt_embeds=[1,2,3])
        raw = make_raw_request()
        result = await self.patch_func(self_mock, req, raw)
        assert "prompt_logprobs is not compatible with prompt embeds" in result

    # ---- 6. Render exceptions ----
    @pytest.mark.asyncio
    async def test_render_value_error(self):
        self_mock = make_self_mock(self.fake_env)
        renderer = make_renderer_mock()
        renderer.render_prompt_and_embeds.side_effect = ValueError("val err")
        self_mock._get_renderer.return_value = renderer
        req = make_request(self.fake_env)
        raw = make_raw_request()
        result = await self.patch_func(self_mock, req, raw)
        assert "val err" in result

    @pytest.mark.asyncio
    async def test_render_type_error(self):
        self_mock = make_self_mock(self.fake_env)
        renderer = make_renderer_mock()
        renderer.render_prompt_and_embeds.side_effect = TypeError("type err")
        self_mock._get_renderer.return_value = renderer
        req = make_request(self.fake_env)
        raw = make_raw_request()
        result = await self.patch_func(self_mock, req, raw)
        assert "type err" in result

    @pytest.mark.asyncio
    async def test_render_runtime_error(self):
        self_mock = make_self_mock(self.fake_env)
        renderer = make_renderer_mock()
        renderer.render_prompt_and_embeds.side_effect = RuntimeError("runtime err")
        self_mock._get_renderer.return_value = renderer
        req = make_request(self.fake_env)
        raw = make_raw_request()
        result = await self.patch_func(self_mock, req, raw)
        assert "runtime err" in result

    @pytest.mark.asyncio
    async def test_render_jinja2_template_error(self):
        self_mock = make_self_mock(self.fake_env)
        renderer = make_renderer_mock()
        renderer.render_prompt_and_embeds.side_effect = self.fake_env["jinja2"].TemplateError("jinja err")
        self_mock._get_renderer.return_value = renderer
        req = make_request(self.fake_env)
        raw = make_raw_request()
        result = await self.patch_func(self_mock, req, raw)
        assert "jinja err" in result

    # ---- 7. Tokenizer skip ----
    @pytest.mark.asyncio
    async def test_skip_tokenizer_init(self):
        self_mock = make_self_mock(self.fake_env)
        self_mock.model_config.skip_tokenizer_init = True
        renderer = make_renderer_mock()
        renderer.render_prompt_and_embeds.return_value = [{"prompt_token_ids": [1,2,3]}]
        self_mock._get_renderer.return_value = renderer
        req = make_request(self.fake_env, stream=True)  # avoid non-streaming path
        raw = make_raw_request()
        self_mock.completion_stream_generator.return_value = "stream_done"
        result = await self.patch_func(self_mock, req, raw)
        self_mock.engine_client.get_tokenizer.assert_not_called()

    # ---- 8. Beam search generator ----
    @pytest.mark.asyncio
    async def test_beam_search_generator(self):
        self_mock = make_self_mock(self.fake_env)
        renderer = make_renderer_mock()
        renderer.render_prompt_and_embeds.return_value = [{"prompt_token_ids": [1]}]
        self_mock._get_renderer.return_value = renderer
        req = make_request(self.fake_env, use_beam_search=True, stream=False)
        raw = make_raw_request()

        final_res = MagicMock()
        async def fake_agen():
            yield 0, final_res
        self.fake_env["vllm_utils"].merge_async_iterators.return_value = fake_agen()

        mock_response = MagicMock()
        self_mock.request_output_to_completion_response.return_value = mock_response

        await self.patch_func(self_mock, req, raw)

        self_mock.engine_client.beam_search.assert_called_once()
        self_mock.engine_client.generate.assert_not_called()

    # ---- 9. DP rank from header ----
    @pytest.mark.asyncio
    async def test_dp_rank_from_header(self):
        self_mock = make_self_mock(self.fake_env)
        renderer = make_renderer_mock()
        renderer.render_prompt_and_embeds.return_value = [{"prompt_token_ids": [1]}]
        self_mock._get_renderer.return_value = renderer
        raw = make_raw_request(dp_rank="2")
        req = make_request(self.fake_env, stream=True)
        self_mock.completion_stream_generator.return_value = "stream_done"
        await self.patch_func(self_mock, req, raw)
        self_mock.engine_client.generate.assert_called_once()
        kwargs = self_mock.engine_client.generate.call_args.kwargs
        assert kwargs["data_parallel_rank"] == 2

    # ---- 10. Multiple prompts ----
    @pytest.mark.asyncio
    async def test_multiple_engine_prompts(self):
        self_mock = make_self_mock(self.fake_env)
        renderer = make_renderer_mock()
        renderer.render_prompt_and_embeds.return_value = [
            {"prompt_token_ids": [1]},
            {"prompt_token_ids": [2]},
        ]
        self_mock._get_renderer.return_value = renderer
        req = make_request(self.fake_env, stream=True)
        raw = make_raw_request()
        self_mock.completion_stream_generator.return_value = "stream_done"
        await self.patch_func(self_mock, req, raw)
        assert self_mock.engine_client.generate.call_count == 2

    # ---- 11. Streaming response ----
    @pytest.mark.asyncio
    async def test_streaming_response(self):
        self_mock = make_self_mock(self.fake_env)
        renderer = make_renderer_mock()
        renderer.render_prompt_and_embeds.return_value = [{"prompt_token_ids": [1]}]
        self_mock._get_renderer.return_value = renderer
        req = make_request(self.fake_env, stream=True)
        raw = make_raw_request()
        self_mock.completion_stream_generator.return_value = "stream_result"
        result = await self.patch_func(self_mock, req, raw)
        assert result == "stream_result"

    # ---- 12. Non-streaming but request stream (fake stream generator) ----
    @pytest.mark.asyncio
    async def test_non_streaming_but_request_stream(self):
        self_mock = make_self_mock(self.fake_env)
        renderer = make_renderer_mock()
        renderer.render_prompt_and_embeds.return_value = [{"prompt_token_ids": [1]}]
        self_mock._get_renderer.return_value = renderer
        req = make_request(self.fake_env, stream=True, best_of=2, n=1)  # stream forced off
        raw = make_raw_request()
        final_res = MagicMock()
        final_res.prompt = None
        async def fake_agen():
            yield 0, final_res
        self.fake_env["vllm_utils"].merge_async_iterators.return_value = fake_agen()
        mock_resp = MagicMock()
        mock_resp.model_dump_json.return_value = '{"a":1}'
        self_mock.request_output_to_completion_response.return_value = mock_resp
        result = await self.patch_func(self_mock, req, raw)
        assert hasattr(result, '__aiter__')
        self_mock.completion_stream_generator.assert_not_called()

    # ---- 13. final_res None raises ----
    @pytest.mark.asyncio
    async def test_final_res_none_raises(self):
        self_mock = make_self_mock(self.fake_env)
        renderer = make_renderer_mock()
        renderer.render_prompt_and_embeds.return_value = [{"prompt_token_ids": [1]}]
        self_mock._get_renderer.return_value = renderer
        req = make_request(self.fake_env)
        raw = make_raw_request()
        async def fake_agen():
            yield 0, None
        self.fake_env["vllm_utils"].merge_async_iterators.return_value = fake_agen()
        with pytest.raises(RuntimeError, match="Failed to get response"):
            await self.patch_func(self_mock, req, raw)

    # ---- 14. CancelledError ----
    @pytest.mark.asyncio
    async def test_cancelled_error(self):
        self_mock = make_self_mock(self.fake_env)
        renderer = make_renderer_mock()
        renderer.render_prompt_and_embeds.return_value = [{"prompt_token_ids": [1]}]
        self_mock._get_renderer.return_value = renderer
        req = make_request(self.fake_env)
        raw = make_raw_request()
        async def fake_agen():
            raise asyncio.CancelledError()
            yield
        self.fake_env["vllm_utils"].merge_async_iterators.return_value = fake_agen()
        result = await self.patch_func(self_mock, req, raw)
        assert "Client disconnected" in result

    # ---- 15. ValueError in generator loop ----
    @pytest.mark.asyncio
    async def test_value_error_in_generator(self):
        self_mock = make_self_mock(self.fake_env)
        renderer = make_renderer_mock()
        renderer.render_prompt_and_embeds.return_value = [{"prompt_token_ids": [1]}]
        self_mock._get_renderer.return_value = renderer
        req = make_request(self.fake_env)
        raw = make_raw_request()
        async def fake_agen():
            raise ValueError("val err in loop")
            yield
        self.fake_env["vllm_utils"].merge_async_iterators.return_value = fake_agen()
        result = await self.patch_func(self_mock, req, raw)
        assert "val err in loop" in result

    # ---- 16. ValueError during scheduling ----
    @pytest.mark.asyncio
    async def test_value_error_during_scheduling(self):
        self_mock = make_self_mock(self.fake_env)
        renderer = make_renderer_mock()
        renderer.render_prompt_and_embeds.return_value = [{"prompt_token_ids": [1]}]
        self_mock._get_renderer.return_value = renderer
        # Make generate throw ValueError
        self_mock.engine_client.generate.side_effect = ValueError("sched err")
        req = make_request(self.fake_env)  # will use generate, not beam_search
        raw = make_raw_request()
        result = await self.patch_func(self_mock, req, raw)
        assert "sched err" in result
