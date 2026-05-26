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
from unittest.mock import MagicMock, patch
import pytest


# ---------------------------------------------------------------------------
# Fixture: fake module tree for patch_attention (0.9.1)
# ---------------------------------------------------------------------------
@pytest.fixture
def fake_attention_env():
    fake_torch = types.ModuleType("torch")
    fake_torch.int32 = "int32"
    fake_torch.int = "int"
    fake_torch.float32 = "float32"
    fake_torch.Tensor = MagicMock
    fake_torch.max = MagicMock()
    fake_torch.sum = MagicMock(return_value=10)
    fake_torch.zeros = MagicMock(return_value=MagicMock())
    fake_torch.index_select = MagicMock(return_value=MagicMock())
    fake_torch.cat = MagicMock(return_value=MagicMock())

    fake_vllm = types.ModuleType("vllm")
    fake_vllm.__path__ = []
    fake_vllm_attention = types.ModuleType("vllm.attention")
    fake_vllm_attention.__path__ = []
    fake_vllm_attention_backends = types.ModuleType("vllm.attention.backends")
    fake_vllm_attention_backends.__path__ = []
    fake_vllm_attention_backends_utils = types.ModuleType("vllm.attention.backends.utils")
    fake_vllm_attention_backends_utils.PAD_SLOT_ID = -1

    fake_vllm_utils = types.ModuleType("vllm.utils")
    fake_vllm_utils.async_tensor_h2d = MagicMock(return_value=MagicMock())
    fake_vllm_utils.make_tensor_with_pad = MagicMock(return_value=MagicMock())

    fake_vllm_ascend = types.ModuleType("vllm_ascend")
    fake_vllm_ascend.__path__ = []
    fake_vllm_ascend_attention = types.ModuleType("vllm_ascend.attention")
    fake_vllm_ascend_attention.__path__ = []
    fake_vllm_ascend_attention_attention = types.ModuleType("vllm_ascend.attention.attention")

    class AscendMetadata:
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)

    class AscendMetadataBuilder:
        _attn_mask_builder = MagicMock()
        _attn_mask_builder.get_attn_mask = MagicMock()

    class AttentionMaskBuilder:
        pass

    fake_vllm_ascend_attention_attention.AscendMetadataBuilder = AscendMetadataBuilder
    fake_vllm_ascend_attention_attention.AscendMetadata = AscendMetadata
    fake_vllm_ascend_attention_attention.AttentionMaskBuilder = AttentionMaskBuilder

    import os
    import aura as _aura
    real_aura_path = _aura.__path__
    base_path = real_aura_path[0] if real_aura_path else "."

    fake_aura = types.ModuleType("aura")
    fake_aura.__path__ = real_aura_path
    fake_aura_runner = types.ModuleType("aura.runner")
    fake_aura_runner.__path__ = [os.path.join(base_path, "runner")]
    fake_aura_runner_infer_adapter = types.ModuleType("aura.runner.infer_adapter")
    fake_aura_runner_infer_adapter.__path__ = [os.path.join(base_path, "runner/infer_adapter")]
    fake_vllm_pkg = types.ModuleType("aura.runner.infer_adapter.vllm")
    fake_vllm_pkg.__path__ = [os.path.join(base_path, "runner/infer_adapter/vllm")]
    fake_patch_pkg = types.ModuleType("aura.runner.infer_adapter.vllm.patch")
    fake_patch_pkg.__path__ = [os.path.join(base_path, "runner/infer_adapter/vllm/patch")]
    fake_0_9_1_pkg = types.ModuleType("aura.runner.infer_adapter.vllm.patch.patch_0_9_1")
    fake_0_9_1_pkg.__path__ = [os.path.join(base_path, "runner/infer_adapter/vllm/patch/patch_0_9_1")]

    fakes = {
        "torch": fake_torch,
        "vllm": fake_vllm,
        "vllm.attention": fake_vllm_attention,
        "vllm.attention.backends": fake_vllm_attention_backends,
        "vllm.attention.backends.utils": fake_vllm_attention_backends_utils,
        "vllm.utils": fake_vllm_utils,
        "vllm_ascend": fake_vllm_ascend,
        "vllm_ascend.attention": fake_vllm_ascend_attention,
        "vllm_ascend.attention.attention": fake_vllm_ascend_attention_attention,
        "aura": fake_aura,
        "aura.runner": fake_aura_runner,
        "aura.runner.infer_adapter": fake_aura_runner_infer_adapter,
        "aura.runner.infer_adapter.vllm": fake_vllm_pkg,
        "aura.runner.infer_adapter.vllm.patch": fake_patch_pkg,
        "aura.runner.infer_adapter.vllm.patch.patch_0_9_1": fake_0_9_1_pkg,
    }

    fake_vllm.__path__ = []
    fake_vllm_ascend.__path__ = []

    yield {
        "fakes": fakes,
        "torch": fake_torch,
        "PAD_SLOT_ID": fake_vllm_attention_backends_utils.PAD_SLOT_ID,
        "async_tensor_h2d": fake_vllm_utils.async_tensor_h2d,
        "make_tensor_with_pad": fake_vllm_utils.make_tensor_with_pad,
        "AscendMetadataBuilder": AscendMetadataBuilder,
        "AscendMetadata": AscendMetadata,
        "AttentionMaskBuilder": AttentionMaskBuilder,
    }


def import_module(fake_attention_env):
    module_name = "aura.runner.infer_adapter.vllm.patch.patch_0_9_1.patch_attention"
    if module_name in sys.modules:
        del sys.modules[module_name]
    with patch.dict(sys.modules, fake_attention_env["fakes"]):
        import aura.runner.infer_adapter.vllm.patch.patch_0_9_1.patch_attention as mod
    return mod


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def make_attn_mask_builder_mock():
    self = MagicMock()
    self._seq_len_cached = 100
    cache = MagicMock()
    cache.numel.return_value = 2
    row0 = MagicMock()
    row0.__getitem__.return_value = 1
    cache.__getitem__.return_value = row0
    self.attn_mask_cache = cache
    self.splitfuse_mask_value = -65504
    self.update_attn_cache = MagicMock()
    self.get_attn_mask = MagicMock(return_value=MagicMock())
    return self

def make_metadata_builder_mock():
    self = MagicMock()
    self.input_builder = MagicMock()
    self.input_builder.inter_data_list = []
    self.input_builder.chunked_prefill_enabled = False
    self._add_seq_group = MagicMock()
    self.runner = MagicMock()
    self.runner.device = "npu:0"
    self.runner.model_config = MagicMock()
    self.runner.model_config.dtype = "float16"
    self.runner.pin_memory = True
    self.prefill_seq_lens = [10]
    self.curr_seq_lens = [20]
    self.num_decode_tokens = 2
    self.num_prefills = 1
    self.num_prefill_tokens = 8
    self.slot_mapping = [1, 2, 3]
    self.block_tables = [[1], [2]]
    self.context_lens = [0]
    self.multimodal_placeholder_maps = {}
    self._get_graph_runner_block_tables = MagicMock(return_value=MagicMock())
    return self


def make_attn_mask_mock(numel_value=0, first_row_second_elem=0):
    """Create a mock tensor returned by get_attn_mask with configurable numel and [0][1]."""
    mask = MagicMock()
    mask.numel.return_value = numel_value
    row0 = MagicMock()
    row0.__getitem__.return_value = first_row_second_elem
    mask.__getitem__.return_value = row0   # mask[0] returns row0
    return mask


# ===========================================================================
# Tests
# ===========================================================================
class TestGetSplitfuseAttnMask:
    @pytest.fixture(autouse=True)
    def setup(self, fake_attention_env):
        self.env = fake_attention_env
        self.mod = import_module(fake_attention_env)

    def test_cached_valid(self):
        self_mock = make_attn_mask_builder_mock()
        self_mock._seq_len_cached = 50
        seq_lens = [30, 20]        # max_seq_len = 30
        query_lens = [10, 10]
        position = MagicMock()

        self.env["torch"].index_select.return_value = MagicMock()
        result = self.mod.get_splitfuse_attn_mask_patch(
            self_mock, seq_lens, query_lens, position, "float16", "npu:0"
        )

        # max_seq_len=30 is passed to update_attn_cache
        self_mock.update_attn_cache.assert_called_once_with(30, "float16", "npu:0")
        self_mock.get_attn_mask.assert_called_once_with(30, "float16", "npu:0")
        self.env["torch"].index_select.assert_called()
        assert result is not None

    def test_cached_invalid(self):
        self_mock = make_attn_mask_builder_mock()
        self_mock._seq_len_cached = 60
        self_mock.attn_mask_cache.numel.return_value = 0    # trigger else branch
        seq_lens = [40]           # max_seq_len = 40
        query_lens = [30]
        position = MagicMock()

        result = self.mod.get_splitfuse_attn_mask_patch(
            self_mock, seq_lens, query_lens, position, "float16", "npu:0"
        )

        self_mock.update_attn_cache.assert_called_once_with(40, "float16", "npu:0")
        self_mock.get_attn_mask.assert_not_called()
        self.env["torch"].index_select.assert_called_once_with(
            self_mock.attn_mask_cache, dim=0, index=position
        )
        assert result is not None

    def test_large_seq_normal(self):
        self_mock = make_attn_mask_builder_mock()
        self_mock._seq_len_cached = 10
        seq_lens = [50, 30]
        query_lens = [10, 5]
        position = None

        result = self.mod.get_splitfuse_attn_mask_patch(
            self_mock, seq_lens, query_lens, position, "float16", "npu:0"
        )

        self.env["torch"].zeros.assert_called()
        assert result is not None

    def test_large_seq_negative_context_raises(self):
        self_mock = make_attn_mask_builder_mock()
        self_mock._seq_len_cached = 5
        seq_lens = [30, 20]
        query_lens = [35, 5]
        position = None

        with pytest.raises(ValueError, match="Context length .* cannot be negative"):
            self.mod.get_splitfuse_attn_mask_patch(
                self_mock, seq_lens, query_lens, position, "float16", "npu:0"
            )


class TestBuildPatch:
    @pytest.fixture(autouse=True)
    def setup(self, fake_attention_env):
        self.env = fake_attention_env
        self.mod = import_module(fake_attention_env)
        # Create a default mock for get_attn_mask that avoids MagicMock comparisons
        self.env["AscendMetadataBuilder"]._attn_mask_builder.get_attn_mask.return_value = make_attn_mask_mock(0, 0)

    def test_prefill_with_block_tables_and_chunk(self):
        attn_mask = make_attn_mask_mock(numel_value=2, first_row_second_elem=1)
        self.env["AscendMetadataBuilder"]._attn_mask_builder.get_attn_mask.return_value = attn_mask

        self_mock = make_metadata_builder_mock()
        self_mock.num_prefills = 1
        self_mock.num_decode_tokens = 2
        self_mock.input_builder.chunked_prefill_enabled = True
        self_mock.block_tables = [[1]]
        self_mock.context_lens = [0, 0]
        seq_lens = [30, 20]
        query_lens = [10, 5]

        result = self.mod.build_patch(self_mock, seq_lens, query_lens, -1)
        assert isinstance(result, self.env["AscendMetadata"])
        self.env["make_tensor_with_pad"].assert_called()
        self.env["torch"].cat.assert_called()

    def test_prefill_no_block_tables(self):
        self_mock = make_metadata_builder_mock()
        self_mock.num_prefills = 1
        self_mock.block_tables = []
        seq_lens = [30]
        query_lens = [20]

        result = self.mod.build_patch(self_mock, seq_lens, query_lens, -1)
        self.env["AscendMetadataBuilder"]._attn_mask_builder.get_attn_mask.assert_called()
        assert isinstance(result, self.env["AscendMetadata"])

    def test_prefill_no_decode_chunked_disabled(self):
        self_mock = make_metadata_builder_mock()
        self_mock.num_prefills = 1
        self_mock.num_decode_tokens = 0
        self_mock.input_builder.chunked_prefill_enabled = False
        self_mock.block_tables = [[1]]
        seq_lens = [30]
        query_lens = [20]

        result = self.mod.build_patch(self_mock, seq_lens, query_lens, -1)
        self.env["AscendMetadataBuilder"]._attn_mask_builder.get_attn_mask.assert_called_with(
            128, "float16", "npu:0"
        )
        assert isinstance(result, self.env["AscendMetadata"])

    def test_decode_only_npu_graph(self):
        self_mock = make_metadata_builder_mock()
        self_mock.num_prefills = 0
        self_mock.num_decode_tokens = 3
        seq_lens = [20]
        query_lens = [1]
        graph_pad_size = 10

        result = self.mod.build_patch(self_mock, seq_lens, query_lens, graph_pad_size)
        assert len(self_mock.slot_mapping) == 3 + graph_pad_size
        self_mock._get_graph_runner_block_tables.assert_called()
        assert isinstance(result, self.env["AscendMetadata"])

    def test_num_prefills_zero_no_npu_graph(self):
        self_mock = make_metadata_builder_mock()
        self_mock.num_prefills = 0
        seq_lens = [20]
        query_lens = [1]

        result = self.mod.build_patch(self_mock, seq_lens, query_lens, -1)
        self.env["make_tensor_with_pad"].assert_called()
        assert self_mock.attn_mask is None
        assert self_mock.compress_mask is None
        assert self_mock.chunk_mask is None
        assert isinstance(result, self.env["AscendMetadata"])

    def test_max_query_len_invalid(self):
        self_mock = make_metadata_builder_mock()
        seq_lens = [20]
        query_lens = [0]
        with pytest.raises(ValueError, match="Maximum query length must be positive"):
            self.mod.build_patch(self_mock, seq_lens, query_lens, -1)

    def test_device_none_raises(self):
        self_mock = make_metadata_builder_mock()
        self_mock.runner.device = None
        seq_lens = [30]
        query_lens = [10]
        with pytest.raises(RuntimeError, match="Device is not initialized"):
            self.mod.build_patch(self_mock, seq_lens, query_lens, -1)


class TestAssignments:
    @pytest.fixture(autouse=True)
    def setup(self, fake_attention_env):
        self.env = fake_attention_env
        self.mod = import_module(fake_attention_env)

    def test_assignments_done(self):
        AttentionMaskBuilder = self.env["AttentionMaskBuilder"]
        AscendMetadataBuilder = self.env["AscendMetadataBuilder"]
        assert AttentionMaskBuilder.get_splitfuse_attn_mask is self.mod.get_splitfuse_attn_mask_patch
        assert AscendMetadataBuilder.build is self.mod.build_patch
