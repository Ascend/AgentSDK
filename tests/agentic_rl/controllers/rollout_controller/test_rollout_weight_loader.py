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
#        http://license.coscl.org.cn/MulanPSL2
#
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
# -------------------------------------------------------------------------

import sys
import importlib.util
from unittest.mock import MagicMock
import os
import pytest
import tempfile
from unittest.mock import patch, AsyncMock
import torch
import re

# Mock ray before importing the module under test
mock_ray = MagicMock()
mock_ray.remote = lambda func_or_class: func_or_class
mock_ray.get = MagicMock(return_value=None)
mock_ray.get_actor = MagicMock()
mock_ray.kill = MagicMock()
mock_ray.is_initialized = MagicMock(return_value=True)
mock_ray.available_resources = MagicMock(return_value={"CPU": 8})

sys.modules['ray'] = mock_ray
sys.modules['ray.util'] = MagicMock()
sys.modules['ray.util.placement_group'] = MagicMock()
sys.modules['ray.util.scheduling_strategies'] = MagicMock()
sys.modules['ray.exceptions'] = MagicMock()

# Mock safetensors before importing the module under test
mock_safetensors = MagicMock()
mock_safetensors.__spec__ = importlib.util.spec_from_loader('safetensors', loader=None)
sys.modules['safetensors'] = mock_safetensors

mock_safetensors_torch = MagicMock()
mock_safetensors_torch.__spec__ = importlib.util.spec_from_loader('safetensors.torch', loader=None)
sys.modules['safetensors.torch'] = mock_safetensors_torch

from agentic_rl.controllers.rollout_controller.rollout_weight_loader import (
    _norm,
    cat_dim0,
    cat_dim1,
    _fast_cat,
    _ceil_div,
    _chunk_evenly,
    _parse_pp_tp_ep,
    LaunchedGroup,
    ParamsAssembler,
)


class TestNorm:
    """Test cases for _norm function."""

    def test_norm_removes_digit_segments(self):
        """Test that _norm removes digit segments."""
        assert _norm("layer.0.weight") == "layer.weight"
        assert _norm("layer.123.weight") == "layer.weight"

    def test_norm_multiple_segments(self):
        """Test _norm with multiple digit segments."""
        # _norm only removes one digit segment at a time
        assert _norm("layer.0.1.2.weight") == "layer.1.weight"

    def test_norm_no_digits(self):
        """Test _norm with no digit segments."""
        assert _norm("layer.weight") == "layer.weight"

    def test_norm_empty_string(self):
        """Test _norm with empty string."""
        assert _norm("") == ""


class TestCatDim0:
    """Test cases for cat_dim0 function."""

    def test_cat_dim0_single_tensor(self):
        """Test cat_dim0 with single tensor."""
        t = torch.randn(3, 4)
        result = cat_dim0([t])
        assert torch.equal(result, t)

    def test_cat_dim0_multiple_tensors(self):
        """Test cat_dim0 with multiple tensors."""
        t1 = torch.randn(2, 4)
        t2 = torch.randn(3, 4)
        t3 = torch.randn(1, 4)
        result = cat_dim0([t1, t2, t3])
        assert result.shape == (6, 4)
        assert torch.equal(result[:2, :], t1)
        assert torch.equal(result[2:5, :], t2)
        assert torch.equal(result[5:, :], t3)

    def test_cat_dim0_preserves_dtype(self):
        """Test cat_dim0 preserves dtype."""
        t1 = torch.randn(2, 4, dtype=torch.float32)
        t2 = torch.randn(3, 4, dtype=torch.float32)
        result = cat_dim0([t1, t2])
        assert result.dtype == torch.float32


class TestCatDim1:
    """Test cases for cat_dim1 function."""

    def test_cat_dim1_single_tensor(self):
        """Test cat_dim1 with single tensor."""
        t = torch.randn(3, 4)
        result = cat_dim1([t])
        assert torch.equal(result, t)

    def test_cat_dim1_multiple_tensors(self):
        """Test cat_dim1 with multiple tensors."""
        t1 = torch.randn(3, 2)
        t2 = torch.randn(3, 3)
        t3 = torch.randn(3, 1)
        result = cat_dim1([t1, t2, t3])
        assert result.shape == (3, 6)
        assert torch.equal(result[:, :2], t1)
        assert torch.equal(result[:, 2:5], t2)
        assert torch.equal(result[:, 5:], t3)

    def test_cat_dim1_preserves_dtype(self):
        """Test cat_dim1 preserves dtype."""
        t1 = torch.randn(3, 2, dtype=torch.float64)
        t2 = torch.randn(3, 3, dtype=torch.float64)
        result = cat_dim1([t1, t2])
        assert result.dtype == torch.float64


class TestFastCat:
    """Test cases for _fast_cat function."""

    def test_fast_cat_empty_list(self):
        """Test _fast_cat with empty list raises error."""
        with pytest.raises(ValueError, match="fast_cat received an empty tensor list"):
            _fast_cat([], dim=0)

    def test_fast_cat_single_tensor(self):
        """Test _fast_cat with single tensor."""
        t = torch.randn(3, 4)
        result = _fast_cat([t], dim=0)
        assert torch.equal(result, t)

    def test_fast_cat_dim0(self):
        """Test _fast_cat along dim 0."""
        t1 = torch.randn(2, 4)
        t2 = torch.randn(3, 4)
        result = _fast_cat([t1, t2], dim=0)
        assert result.shape == (5, 4)

    def test_fast_cat_dim1(self):
        """Test _fast_cat along dim 1."""
        t1 = torch.randn(3, 2)
        t2 = torch.randn(3, 3)
        result = _fast_cat([t1, t2], dim=1)
        assert result.shape == (3, 5)


class TestCeilDiv:
    """Test cases for _ceil_div function."""

    def test_ceil_div_exact(self):
        """Test _ceil_div with exact division."""
        assert _ceil_div(10, 5) == 2
        assert _ceil_div(20, 4) == 5

    def test_ceil_div_remainder(self):
        """Test _ceil_div with remainder."""
        assert _ceil_div(11, 5) == 3
        assert _ceil_div(13, 4) == 4

    def test_ceil_div_one(self):
        """Test _ceil_div resulting in 1."""
        assert _ceil_div(3, 10) == 1
        assert _ceil_div(1, 1) == 1


class TestChunkEvenly:
    """Test cases for _chunk_evenly function."""

    def test_chunk_evenly_empty(self):
        """Test _chunk_evenly with empty list."""
        assert _chunk_evenly([], 3) == []

    def test_chunk_evenly_single_worker(self):
        """Test _chunk_evenly with single worker."""
        items = [1, 2, 3, 4, 5]
        result = _chunk_evenly(items, 1)
        assert result == [[1, 2, 3, 4, 5]]

    def test_chunk_evenly_multiple_workers(self):
        """Test _chunk_evenly with multiple workers."""
        items = [1, 2, 3, 4, 5, 6]
        result = _chunk_evenly(items, 3)
        assert len(result) == 3
        assert result[0] == [1, 2]
        assert result[1] == [3, 4]
        assert result[2] == [5, 6]

    def test_chunk_evenly_more_workers_than_items(self):
        """Test _chunk_evenly with more workers than items."""
        items = [1, 2, 3]
        result = _chunk_evenly(items, 5)
        assert len(result) == 3

    def test_chunk_evenly_uneven(self):
        """Test _chunk_evenly with uneven distribution."""
        items = [1, 2, 3, 4, 5]
        result = _chunk_evenly(items, 2)
        assert len(result) == 2
        assert result[0] == [1, 2, 3]
        assert result[1] == [4, 5]


class TestParsePpTpEp:
    """Test cases for _parse_pp_tp_ep function."""

    def test_parse_with_ep(self):
        """Test _parse_pp_tp_ep with ep."""
        pp, tp, ep = _parse_pp_tp_ep("/path/pp0_tp1_ep2.safetensors")
        assert pp == 0
        assert tp == 1
        assert ep == 2

    def test_parse_without_ep(self):
        """Test _parse_pp_tp_ep without ep."""
        pp, tp, ep = _parse_pp_tp_ep("/path/pp1_tp2.safetensors")
        assert pp == 1
        assert tp == 2
        assert ep == 0

    def test_parse_complex_path(self):
        """Test _parse_pp_tp_ep with complex path."""
        pp, tp, ep = _parse_pp_tp_ep("/long/path/pp3_tp4_ep5.safetensors")
        assert pp == 3
        assert tp == 4
        assert ep == 5

    def test_parse_invalid_path(self):
        """Test _parse_pp_tp_ep with invalid path."""
        with pytest.raises(ValueError, match="Cannot parse"):
            _parse_pp_tp_ep("/path/invalid.safetensors")


class TestLaunchedGroup:
    """Test cases for LaunchedGroup dataclass."""

    def test_init(self):
        """Test LaunchedGroup initialization."""
        refs = [MagicMock()]
        pg = MagicMock()
        lg = LaunchedGroup(refs=refs, pg=pg, flatten=True)
        assert lg.refs == refs
        assert lg.pg == pg
        assert lg.flatten is True

    def test_get_simple(self):
        """Test LaunchedGroup.get with simple result."""
        mock_ref = MagicMock()
        with patch('agentic_rl.controllers.rollout_controller.rollout_weight_loader.ray.get') as mock_ray_get:
            mock_ray_get.return_value = ["result1", "result2"]
            lg = LaunchedGroup(refs=[mock_ref], pg=None, flatten=False)
            result = lg.get()
            assert result == ["result1", "result2"]

    def test_get_flatten(self):
        """Test LaunchedGroup.get with flatten."""
        mock_ref = MagicMock()
        with patch('agentic_rl.controllers.rollout_controller.rollout_weight_loader.ray.get') as mock_ray_get:
            mock_ray_get.return_value = [["a", "b"], ["c", "d"]]
            lg = LaunchedGroup(refs=[mock_ref], pg=None, flatten=True)
            result = lg.get()
            assert result == ["a", "b", "c", "d"]


class TestParamsAssembler:
    """Test cases for ParamsAssembler class."""

    def test_init(self):
        """Test ParamsAssembler initialization."""
        assembler = ParamsAssembler(infer_tp=2, infer_dp=4)
        assert assembler.infer_tp == 2
        assert assembler.infer_dp == 4

    def test_init_defaults(self):
        """Test ParamsAssembler initialization with defaults."""
        assembler = ParamsAssembler()
        assert assembler.infer_tp == 1
        assert assembler.infer_dp == 1

    def test_Teff_property(self):
        """Test _Teff property."""
        assembler = ParamsAssembler(infer_tp=2, infer_dp=3)
        assert assembler._Teff == 6

    def test_cast_if_needed_no_cast(self):
        """Test _cast_if_needed when no cast needed."""
        assembler = ParamsAssembler(target_dtype=torch.float32)
        t = torch.randn(3, 4, dtype=torch.float32)
        result = assembler._cast_if_needed(t)
        assert result.dtype == torch.float32

    def test_cast_if_needed_with_cast(self):
        """Test _cast_if_needed when cast needed."""
        assembler = ParamsAssembler(target_dtype=torch.float16)
        t = torch.randn(3, 4, dtype=torch.float32)
        result = assembler._cast_if_needed(t)
        assert result.dtype == torch.float16

    def test_cast_if_needed_no_target(self):
        """Test _cast_if_needed with no target dtype."""
        assembler = ParamsAssembler()
        t = torch.randn(3, 4, dtype=torch.float32)
        result = assembler._cast_if_needed(t)
        assert result.dtype == torch.float32

    def test_get_tp_split_axis(self):
        """Test get_tp_split_axis returns None."""
        assembler = ParamsAssembler()
        assert assembler.get_tp_split_axis("any.name") is None

    def test_is_fused_qkv_weight(self):
        """Test is_fused_qkv_weight returns False."""
        assembler = ParamsAssembler()
        assert assembler.is_fused_qkv_weight("any.name") is False

    def test_is_fused_qkv_bias(self):
        """Test is_fused_qkv_bias returns False."""
        assembler = ParamsAssembler()
        assert assembler.is_fused_qkv_bias("any.name") is False

    def test_is_w13(self):
        """Test is_w13 returns False."""
        assembler = ParamsAssembler()
        assert assembler.is_w13("any.name") is False

    def test_is_w2(self):
        """Test is_w2 returns False."""
        assembler = ParamsAssembler()
        assert assembler.is_w2("any.name") is False

    def test_get_train_tp_concat_axis_2d(self):
        """Test get_train_tp_concat_axis_2d returns None."""
        assembler = ParamsAssembler()
        assert assembler.get_train_tp_concat_axis_2d("any.name") is None

    def test_get_weight_3D_shape(self):
        """Test get_weight_3D_shape returns None."""
        assembler = ParamsAssembler()
        assert assembler.get_weight_3D_shape("any.name", [3, 4]) is None

    def test_get_weight_3D_permute(self):
        """Test get_weight_3D_permute returns None."""
        assembler = ParamsAssembler()
        assert assembler.get_weight_3D_permute("any.name") is None

    def test_unflatten_weight_no_change(self):
        """Test unflatten_weight when no change needed."""
        assembler = ParamsAssembler()
        t = torch.randn(3, 4)
        result = assembler.unflatten_weight("any.name", t)
        assert torch.equal(result, t)

    def test_write_file(self):
        """Test write_file method."""
        assembler = ParamsAssembler()
        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = os.path.join(tmpdir, "test.safetensors")
            tensors = {"weight": torch.randn(3, 4)}
            with patch('agentic_rl.controllers.rollout_controller.rollout_weight_loader.save_file') as mock_save:
                def create_file(data, path):
                    with open(path, 'wb') as f:
                        f.write(b"test")
                mock_save.side_effect = create_file
                assembler.write_file(tensors, out_path)
                mock_save.assert_called_once()

    def test_shard_w13_for_tp_grouped(self):
        """Test shard_w13_for_tp with grouped mode."""
        assembler = ParamsAssembler(infer_tp=2)
        assembler.w13_cols_mode = "grouped"
        t3d = torch.randn(4, 8, 16)
        result = assembler.shard_w13_for_tp(t3d, 0)
        assert result.shape == (4, 4, 16)

    def test_shard_w13_for_tp_interleaved(self):
        """Test shard_w13_for_tp with interleaved mode."""
        assembler = ParamsAssembler(infer_tp=2)
        assembler.w13_cols_mode = "interleaved"
        t3d = torch.randn(4, 8, 16)
        result = assembler.shard_w13_for_tp(t3d, 0)
        assert result.shape == (4, 4, 16)

    def test_shard_w2_for_tp_grouped(self):
        """Test shard_w2_for_tp with grouped mode."""
        assembler = ParamsAssembler(infer_tp=2)
        assembler.w2_rows_mode = "grouped"
        t3d = torch.randn(4, 16, 8)
        result = assembler.shard_w2_for_tp(t3d, 0)
        assert result.shape == (4, 16, 4)

    def test_shard_w2_for_tp_interleaved(self):
        """Test shard_w2_for_tp with interleaved mode."""
        assembler = ParamsAssembler(infer_tp=2)
        assembler.w2_rows_mode = "interleaved"
        t3d = torch.randn(4, 16, 8)
        result = assembler.shard_w2_for_tp(t3d, 0)
        assert result.shape == (4, 16, 4)


class TestQwen3MoEParamsAssembler:
    """Test cases for Qwen3MoEParamsAssembler class."""

    def test_init(self):
        """Test Qwen3MoEParamsAssembler initialization."""
        mock_config = MagicMock()
        mock_config.hidden_size = 4096
        mock_config.num_attention_heads = 32
        mock_config.num_key_value_heads = 8
        mock_config.head_dim = 128
        mock_config.num_experts = 8
        mock_config.num_hidden_layers = 32
        mock_config.vocab_size = 32000

        from agentic_rl.controllers.rollout_controller.rollout_weight_loader import Qwen3MoEParamsAssembler
        assembler = Qwen3MoEParamsAssembler(mock_config, infer_tp=2)

        assert assembler.hidden_size == 4096
        assert assembler.num_attention_heads == 32
        assert assembler.num_key_value_heads == 8
        assert assembler.head_dim == 128
        assert assembler.num_experts == 8
        assert assembler.num_layers == 32
        assert assembler.has_moe is True

    def test_is_fused_qkv_weight(self):
        """Test is_fused_qkv_weight for Qwen3MoE."""
        mock_config = MagicMock()
        mock_config.hidden_size = 4096
        mock_config.num_attention_heads = 32
        mock_config.num_key_value_heads = 8
        mock_config.head_dim = 128
        mock_config.num_experts = 8
        mock_config.num_hidden_layers = 32
        mock_config.vocab_size = 32000

        from agentic_rl.controllers.rollout_controller.rollout_weight_loader import Qwen3MoEParamsAssembler
        assembler = Qwen3MoEParamsAssembler(mock_config)

        assert assembler.is_fused_qkv_weight("model.layers.0.self_attn.qkv_proj.weight") is True
        assert assembler.is_fused_qkv_weight("model.layers.0.self_attn.q_proj.weight") is False

    def test_is_fused_qkv_bias(self):
        """Test is_fused_qkv_bias for Qwen3MoE."""
        mock_config = MagicMock()
        mock_config.hidden_size = 4096
        mock_config.num_attention_heads = 32
        mock_config.num_key_value_heads = 8
        mock_config.head_dim = 128
        mock_config.num_experts = 8
        mock_config.num_hidden_layers = 32
        mock_config.vocab_size = 32000

        from agentic_rl.controllers.rollout_controller.rollout_weight_loader import Qwen3MoEParamsAssembler
        assembler = Qwen3MoEParamsAssembler(mock_config)

        assert assembler.is_fused_qkv_bias("model.layers.0.self_attn.qkv_proj.bias") is True
        assert assembler.is_fused_qkv_bias("model.layers.0.self_attn.q_proj.bias") is False

    def test_is_w13(self):
        """Test is_w13 for Qwen3MoE."""
        mock_config = MagicMock()
        mock_config.hidden_size = 4096
        mock_config.num_attention_heads = 32
        mock_config.num_key_value_heads = 8
        mock_config.head_dim = 128
        mock_config.num_experts = 8
        mock_config.num_hidden_layers = 32
        mock_config.vocab_size = 32000

        from agentic_rl.controllers.rollout_controller.rollout_weight_loader import Qwen3MoEParamsAssembler
        assembler = Qwen3MoEParamsAssembler(mock_config)

        assert assembler.is_w13("model.layers.0.mlp.experts.w13_weight") is True
        assert assembler.is_w13("model.layers.0.mlp.experts.w2_weight") is False

    def test_is_w2(self):
        """Test is_w2 for Qwen3MoE."""
        mock_config = MagicMock()
        mock_config.hidden_size = 4096
        mock_config.num_attention_heads = 32
        mock_config.num_key_value_heads = 8
        mock_config.head_dim = 128
        mock_config.num_experts = 8
        mock_config.num_hidden_layers = 32
        mock_config.vocab_size = 32000

        from agentic_rl.controllers.rollout_controller.rollout_weight_loader import Qwen3MoEParamsAssembler
        assembler = Qwen3MoEParamsAssembler(mock_config)

        assert assembler.is_w2("model.layers.0.mlp.experts.w2_weight") is True
        assert assembler.is_w2("model.layers.0.mlp.experts.w13_weight") is False

    def test_get_tp_split_axis(self):
        """Test get_tp_split_axis for Qwen3MoE."""
        mock_config = MagicMock()
        mock_config.hidden_size = 4096
        mock_config.num_attention_heads = 32
        mock_config.num_key_value_heads = 8
        mock_config.head_dim = 128
        mock_config.num_experts = 8
        mock_config.num_hidden_layers = 32
        mock_config.vocab_size = 32000

        from agentic_rl.controllers.rollout_controller.rollout_weight_loader import Qwen3MoEParamsAssembler
        assembler = Qwen3MoEParamsAssembler(mock_config)

        assert assembler.get_tp_split_axis("model.layers.0.mlp.experts.w13_weight") == 0
        assert assembler.get_tp_split_axis("model.layers.0.mlp.experts.w2_weight") == 0
        assert assembler.get_tp_split_axis("lm_head.weight") == 0
        assert assembler.get_tp_split_axis("model.embed_tokens.weight") == 0
        assert assembler.get_tp_split_axis("model.layers.0.self_attn.o_proj.weight") == 1
        assert assembler.get_tp_split_axis("model.layers.0.self_attn.qkv_proj.weight") == 0

    def test_get_weight_3D_shape_w13(self):
        """Test get_weight_3D_shape for w13."""
        mock_config = MagicMock()
        mock_config.hidden_size = 4096
        mock_config.num_attention_heads = 32
        mock_config.num_key_value_heads = 8
        mock_config.head_dim = 128
        mock_config.num_experts = 8
        mock_config.num_hidden_layers = 32
        mock_config.vocab_size = 32000

        from agentic_rl.controllers.rollout_controller.rollout_weight_loader import Qwen3MoEParamsAssembler
        assembler = Qwen3MoEParamsAssembler(mock_config)

        result = assembler.get_weight_3D_shape("model.layers.0.mlp.experts.w13_weight", [4096, 32768])
        assert result == (4096, 8, 4096)

    def test_get_weight_3D_shape_w2(self):
        """Test get_weight_3D_shape for w2."""
        mock_config = MagicMock()
        mock_config.hidden_size = 4096
        mock_config.num_attention_heads = 32
        mock_config.num_key_value_heads = 8
        mock_config.head_dim = 128
        mock_config.num_experts = 8
        mock_config.num_hidden_layers = 32
        mock_config.vocab_size = 32000

        from agentic_rl.controllers.rollout_controller.rollout_weight_loader import Qwen3MoEParamsAssembler
        assembler = Qwen3MoEParamsAssembler(mock_config)

        result = assembler.get_weight_3D_shape("model.layers.0.mlp.experts.w2_weight", [32768, 4096])
        assert result == (8, 4096, 4096)

    def test_get_weight_3D_permute_w13(self):
        """Test get_weight_3D_permute for w13."""
        mock_config = MagicMock()
        mock_config.hidden_size = 4096
        mock_config.num_attention_heads = 32
        mock_config.num_key_value_heads = 8
        mock_config.head_dim = 128
        mock_config.num_experts = 8
        mock_config.num_hidden_layers = 32
        mock_config.vocab_size = 32000

        from agentic_rl.controllers.rollout_controller.rollout_weight_loader import Qwen3MoEParamsAssembler
        assembler = Qwen3MoEParamsAssembler(mock_config)

        result = assembler.get_weight_3D_permute("model.layers.0.mlp.experts.w13_weight")
        assert result == (1, 2, 0)

    def test_get_weight_3D_permute_w2(self):
        """Test get_weight_3D_permute for w2."""
        mock_config = MagicMock()
        mock_config.hidden_size = 4096
        mock_config.num_attention_heads = 32
        mock_config.num_key_value_heads = 8
        mock_config.head_dim = 128
        mock_config.num_experts = 8
        mock_config.num_hidden_layers = 32
        mock_config.vocab_size = 32000

        from agentic_rl.controllers.rollout_controller.rollout_weight_loader import Qwen3MoEParamsAssembler
        assembler = Qwen3MoEParamsAssembler(mock_config)

        result = assembler.get_weight_3D_permute("model.layers.0.mlp.experts.w2_weight")
        assert result == (0, 2, 1)


@pytest.fixture(autouse=True)
def ensure_mock():
    """Ensure ray mock exists and is properly configured before each test."""
    if 'ray' not in sys.modules:
        mock_ray = MagicMock()
        mock_ray.remote = lambda func_or_class: func_or_class
        mock_ray.get = MagicMock(return_value=None)
        mock_ray.get_actor = MagicMock()
        mock_ray.kill = MagicMock()
        mock_ray.is_initialized = MagicMock(return_value=True)
        mock_ray.available_resources = MagicMock(return_value={"CPU": 8})
        sys.modules['ray'] = mock_ray
        sys.modules['ray.util'] = MagicMock()
        sys.modules['ray.util.placement_group'] = MagicMock()
        sys.modules['ray.util.scheduling_strategies'] = MagicMock()
        sys.modules['ray.exceptions'] = MagicMock()
    else:
        # Reset the get mock to return None by default, but keep the same mock object
        sys.modules['ray'].get.reset_mock(return_value=True)
        sys.modules['ray'].get.return_value = None
    yield


@pytest.fixture(scope="module", autouse=True)
def cleanup_module():
    """Cleanup mock modules after all tests in this module."""
    yield
    modules_to_clean = ['ray', 'safetensors', 'safetensors.torch', 'ray.util', 
                       'ray.util.placement_group', 'ray.util.scheduling_strategies', 'ray.exceptions']
    for mod in modules_to_clean:
        if mod in sys.modules:
            del sys.modules[mod]
