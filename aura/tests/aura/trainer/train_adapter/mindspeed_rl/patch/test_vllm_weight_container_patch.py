#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
-------------------------------------------------------------------------
This file is part of the AgentSDK project.
Copyright (c) 2026 Huawei Technologies Co.,Ltd.

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

import pytest
from contextlib import ExitStack
from unittest.mock import MagicMock, patch

_PATCH_MODULE = 'aura.trainer.train_adapter.mindspeed_rl.patch.vllm_weight_container_patch'


_UNSET = object()


class TestVllmWeightContainerPatch:

    def _make_parallel_state(self, tp_world_size=1, ep_world_size=1, pp_rank=0,
                             pp_world_size=1, vpp_rank=0, vpp_world_size=1,
                             tp_and_ep_group=False):
        ps = MagicMock()
        ps.get_pipeline_model_parallel_rank.return_value = pp_rank
        ps.get_pipeline_model_parallel_group.return_value = MagicMock()
        ps.get_pipeline_model_parallel_world_size.return_value = pp_world_size
        ps.get_tensor_model_parallel_world_size.return_value = tp_world_size
        ps.get_tensor_model_parallel_group.return_value = MagicMock()
        ps.get_expert_model_parallel_world_size.return_value = ep_world_size
        ps.get_expert_model_parallel_group.return_value = MagicMock()
        ps._VIRTUAL_PIPELINE_MODEL_PARALLEL_RANK = vpp_rank
        ps._VIRTUAL_PIPELINE_MODEL_PARALLEL_WORLD_SIZE = vpp_world_size
        if tp_and_ep_group:
            ps.get_tensor_and_expert_parallel_group.return_value = MagicMock()
        return ps

    def _make_init_mocks(self, num_hidden_layers=24, **model_config_attrs):
        mock_self = MagicMock()
        mock_megatron_model = MagicMock()
        mock_vllm_model = MagicMock()
        mock_model_config = MagicMock()
        mock_model_config.num_hidden_layers = num_hidden_layers
        for k, v in model_config_attrs.items():
            setattr(mock_model_config, k, v)
        return mock_self, mock_megatron_model, mock_vllm_model, mock_model_config

    def test___init__(self):
        from aura.trainer.train_adapter.mindspeed_rl.patch.vllm_weight_container_patch import __init__

        mock_self, mock_megatron_model, mock_vllm_model, mock_model_config = self._make_init_mocks()
        mock_parallel_state = self._make_parallel_state()

        with ExitStack() as stack:
            mock_dist = stack.enter_context(patch(f'{_PATCH_MODULE}.dist'))
            mock_dist.get_world_size.return_value = 1
            mock_dist.get_rank.return_value = 0
            stack.enter_context(patch.object(mock_self, '_build_num_layer_list', return_value=[12, 12]))
            stack.enter_context(patch.object(mock_self, '_build_vpp_layer_list', return_value=[[0], [1]]))
            mock_validate = stack.enter_context(patch.object(mock_self, '_validate_parallel_config'))
            stack.enter_context(patch.object(mock_self, '_init_tensor_model_parallel_allgather_group'))
            stack.enter_context(patch.object(mock_self, '_init_pipeline_model_parallel_allgather_group'))
            stack.enter_context(patch.object(mock_self, '_init_tensor_model_parallel_split_group'))
            mock_init_weights = stack.enter_context(patch.object(mock_self, '_init_weight_buffers'))

            __init__(
                mock_self,
                megatron_model=mock_megatron_model,
                vllm_model=mock_vllm_model,
                model_config=mock_model_config,
                infer_tensor_parallel_size=1,
                infer_pipeline_parallel_size=1,
                infer_expert_parallel_size=1,
                num_layer_list="12,12",
                moe_tp_extend_ep=False,
                parallel_state=mock_parallel_state,
                weight_adaptor=MagicMock(),
                enable_validate=False,
                noop_layers=None,
            )

            assert mock_self.one_step_off_ep_mode == False
            assert mock_self._num_hidden_layers == 24
            mock_validate.assert_called_once()
            mock_init_weights.assert_called_once()

    def test___init__with_noop_layers(self):
        from aura.trainer.train_adapter.mindspeed_rl.patch.vllm_weight_container_patch import __init__

        mock_self, mock_megatron_model, mock_vllm_model, mock_model_config = self._make_init_mocks()
        mock_parallel_state = self._make_parallel_state()

        with ExitStack() as stack:
            mock_dist = stack.enter_context(patch(f'{_PATCH_MODULE}.dist'))
            mock_dist.get_world_size.return_value = 1
            mock_dist.get_rank.return_value = 0
            stack.enter_context(patch.object(mock_self, '_build_num_layer_list', return_value=[12, 12]))
            stack.enter_context(patch.object(mock_self, '_build_vpp_layer_list', return_value=[[0], [1]]))
            mock_build_map = stack.enter_context(patch.object(mock_self, '_build_global2local_map', return_value={0: 0}))
            stack.enter_context(patch.object(mock_self, '_validate_parallel_config'))
            stack.enter_context(patch.object(mock_self, '_init_tensor_model_parallel_allgather_group'))
            stack.enter_context(patch.object(mock_self, '_init_pipeline_model_parallel_allgather_group'))
            stack.enter_context(patch.object(mock_self, '_init_tensor_model_parallel_split_group'))
            stack.enter_context(patch.object(mock_self, '_init_weight_buffers'))

            __init__(
                mock_self,
                megatron_model=mock_megatron_model,
                vllm_model=mock_vllm_model,
                model_config=mock_model_config,
                infer_tensor_parallel_size=1,
                infer_pipeline_parallel_size=1,
                infer_expert_parallel_size=1,
                num_layer_list="12,12",
                moe_tp_extend_ep=False,
                parallel_state=mock_parallel_state,
                weight_adaptor=MagicMock(),
                enable_validate=False,
                noop_layers="0,1,2",
            )

            assert mock_self._noop_layers == [0, 1, 2]
            assert mock_self._num_hidden_layers == 27
            mock_build_map.assert_called_once()

    def test___init__with_experts(self):
        from aura.trainer.train_adapter.mindspeed_rl.patch.vllm_weight_container_patch import __init__

        mock_self, mock_megatron_model, mock_vllm_model, mock_model_config = self._make_init_mocks(n_routed_experts=8)
        mock_parallel_state = self._make_parallel_state(ep_world_size=4)

        with ExitStack() as stack:
            mock_dist = stack.enter_context(patch(f'{_PATCH_MODULE}.dist'))
            mock_dist.get_world_size.return_value = 1
            mock_dist.get_rank.return_value = 0
            stack.enter_context(patch.object(mock_self, '_build_num_layer_list', return_value=[12, 12]))
            stack.enter_context(patch.object(mock_self, '_build_vpp_layer_list', return_value=[[0], [1]]))
            stack.enter_context(patch.object(mock_self, '_validate_parallel_config'))
            stack.enter_context(patch.object(mock_self, '_init_tensor_model_parallel_allgather_group'))
            stack.enter_context(patch.object(mock_self, '_init_pipeline_model_parallel_allgather_group'))
            stack.enter_context(patch.object(mock_self, '_init_tensor_model_parallel_split_group'))
            stack.enter_context(patch.object(mock_self, '_init_weight_buffers'))

            __init__(
                mock_self,
                megatron_model=mock_megatron_model,
                vllm_model=mock_vllm_model,
                model_config=mock_model_config,
                infer_tensor_parallel_size=1,
                infer_pipeline_parallel_size=1,
                infer_expert_parallel_size=4,
                num_layer_list="12,12",
                moe_tp_extend_ep=False,
                parallel_state=mock_parallel_state,
                weight_adaptor=MagicMock(),
                enable_validate=False,
                noop_layers=None,
            )

            assert mock_self.num_experts == 8
            assert mock_self.num_local_experts == 2

    def test___init__with_moe_tp_extend_ep(self):
        from aura.trainer.train_adapter.mindspeed_rl.patch.vllm_weight_container_patch import __init__

        mock_self, mock_megatron_model, mock_vllm_model, mock_model_config = self._make_init_mocks(num_experts=8)
        mock_parallel_state = self._make_parallel_state(tp_world_size=2, ep_world_size=2, tp_and_ep_group=True)

        with ExitStack() as stack:
            mock_dist = stack.enter_context(patch(f'{_PATCH_MODULE}.dist'))
            mock_dist.get_world_size.return_value = 4
            mock_dist.get_rank.return_value = 0
            stack.enter_context(patch.object(mock_self, '_build_num_layer_list', return_value=[12, 12]))
            stack.enter_context(patch.object(mock_self, '_build_vpp_layer_list', return_value=[[0], [1]]))
            stack.enter_context(patch.object(mock_self, '_validate_parallel_config'))
            stack.enter_context(patch.object(mock_self, '_init_tensor_model_parallel_allgather_group'))
            stack.enter_context(patch.object(mock_self, '_init_pipeline_model_parallel_allgather_group'))
            stack.enter_context(patch.object(mock_self, '_init_tensor_model_parallel_split_group'))
            stack.enter_context(patch.object(mock_self, '_init_weight_buffers'))

            __init__(
                mock_self,
                megatron_model=mock_megatron_model,
                vllm_model=mock_vllm_model,
                model_config=mock_model_config,
                infer_tensor_parallel_size=2,
                infer_pipeline_parallel_size=1,
                infer_expert_parallel_size=4,
                num_layer_list="12,12",
                moe_tp_extend_ep=True,
                parallel_state=mock_parallel_state,
                weight_adaptor=MagicMock(),
                enable_validate=False,
                noop_layers=None,
            )

            assert mock_self._ep_size == 4
            assert mock_self.moe_tp_extend_ep == True

    def test_validate_parallel_config_patch(self):
        from aura.trainer.train_adapter.mindspeed_rl.patch.vllm_weight_container_patch import _validate_parallel_config_patch

        mock_self = MagicMock()
        mock_self.one_step_off_ep_mode = False
        mock_self._infer_pp_size = 1
        mock_self._infer_ep_size = 4
        mock_self._ep_size = 2
        mock_self.moe_tp_extend_ep = True
        mock_self._pp_size = 1
        mock_self._tp_size = 1
        mock_self._infer_tp_size = 1
        mock_self._world_size = 4

        _validate_parallel_config_patch(mock_self)

    def test_validate_parallel_config_patch_ep_mode(self):
        from aura.trainer.train_adapter.mindspeed_rl.patch.vllm_weight_container_patch import _validate_parallel_config_patch

        mock_self = MagicMock()
        mock_self.one_step_off_ep_mode = True

        _validate_parallel_config_patch(mock_self)

    def test_validate_parallel_config_patch_infer_pp_not_1(self):
        from aura.trainer.train_adapter.mindspeed_rl.patch.vllm_weight_container_patch import _validate_parallel_config_patch

        mock_self = MagicMock()
        mock_self.one_step_off_ep_mode = False
        mock_self._infer_pp_size = 2

        with pytest.raises(ValueError, match="infer_pp_size != 1 not supported yet"):
            _validate_parallel_config_patch(mock_self)

    def test_validate_parallel_config_patch_ep_divisible(self):
        from aura.trainer.train_adapter.mindspeed_rl.patch.vllm_weight_container_patch import _validate_parallel_config_patch

        mock_self = MagicMock()
        mock_self.one_step_off_ep_mode = False
        mock_self._infer_pp_size = 1
        mock_self._infer_ep_size = 5
        mock_self._ep_size = 2

        with pytest.raises(ValueError, match="should be divisible"):
            _validate_parallel_config_patch(mock_self)

    def test_split_tp_params_patch_no_split(self):
        from aura.trainer.train_adapter.mindspeed_rl.patch.vllm_weight_container_patch import split_tp_params_patch

        mock_self = MagicMock()
        mock_self._infer_tp_size = 1
        mock_self._tp_size = 2

        mock_param = MagicMock()
        result = split_tp_params_patch(mock_self, mock_param, "test_param")

        assert result == mock_param

    def test_split_tp_params_patch_with_split(self):
        from aura.trainer.train_adapter.mindspeed_rl.patch.vllm_weight_container_patch import split_tp_params_patch

        mock_self = MagicMock()
        mock_self._infer_tp_size = 4
        mock_self._tp_size = 1
        mock_self._rank = 0

        mock_param = MagicMock()
        mock_param.size.return_value = [10, 10]

        with patch(f'{_PATCH_MODULE}.is_tensor_parallel_param', return_value=False):
            result = split_tp_params_patch(mock_self, mock_param, "test_param")

            assert result == mock_param

    def test_get_infer_params_patch_ep_mode(self):
        from aura.trainer.train_adapter.mindspeed_rl.patch.vllm_weight_container_patch import get_infer_params_patch

        mock_self = MagicMock()
        mock_self.one_step_off_ep_mode = True
        expected_result = {"param1": MagicMock()}
        mock_self._get_simple_ep_params.return_value = expected_result

        result = get_infer_params_patch(mock_self)

        assert result == expected_result
        mock_self._get_simple_ep_params.assert_called_once()

    def test_get_infer_params_patch_normal_mode(self):
        from aura.trainer.train_adapter.mindspeed_rl.patch.vllm_weight_container_patch import get_infer_params_patch

        mock_self = MagicMock()
        mock_self.one_step_off_ep_mode = False
        mock_self.moe_tp_extend_ep = False

        with patch(f'{_PATCH_MODULE}._build_infer_param_dict', return_value={"param1": MagicMock()}) as mock_build_dict:
            mock_self._get_all_params.return_value = {"raw_param": MagicMock()}

            result = get_infer_params_patch(mock_self)

            mock_self._update_weight_buffers_intra_pp.assert_called_once()
            mock_self._update_weight_buffers_inter_pp.assert_called_once()
            mock_build_dict.assert_called_once()

    def test_get_infer_params_patch_with_moe_tp_extend_ep(self):
        from aura.trainer.train_adapter.mindspeed_rl.patch.vllm_weight_container_patch import get_infer_params_patch

        mock_self = MagicMock()
        mock_self.one_step_off_ep_mode = False
        mock_self.moe_tp_extend_ep = True
        mock_self._infer_ep_size = 4
        mock_self._ep_size = 2

        with patch(f'{_PATCH_MODULE}._build_infer_param_dict', return_value={"param1": MagicMock()}) as mock_build_dict:
            mock_self._get_all_params.return_value = {"raw_param": MagicMock()}

            result = get_infer_params_patch(mock_self)

            mock_self._update_weight_buffers_ep.assert_called_once()
            mock_self._send_receive_experts.assert_called_once()

    def test_validate_parallel_config_patch_tp_size_validation(self):
        from aura.trainer.train_adapter.mindspeed_rl.patch.vllm_weight_container_patch import _validate_parallel_config_patch

        mock_self = MagicMock()
        mock_self.one_step_off_ep_mode = False
        mock_self._infer_pp_size = 1
        mock_self._infer_ep_size = 2
        mock_self._ep_size = 1
        mock_self.moe_tp_extend_ep = False
        mock_self._pp_size = 1
        mock_self._tp_size = 3
        mock_self._infer_tp_size = 2
        mock_self._world_size = 6

        with pytest.raises(ValueError, match="should be an integer multiple"):
            _validate_parallel_config_patch(mock_self)

    def test_validate_parallel_config_patch_tp_increase_validation(self):
        from aura.trainer.train_adapter.mindspeed_rl.patch.vllm_weight_container_patch import _validate_parallel_config_patch

        mock_self = MagicMock()
        mock_self.one_step_off_ep_mode = False
        mock_self._infer_pp_size = 1
        mock_self._infer_ep_size = 1
        mock_self._ep_size = 1
        mock_self.moe_tp_extend_ep = False
        mock_self._pp_size = 1
        mock_self._tp_size = 1
        mock_self._infer_tp_size = 4
        mock_self._world_size = 2

        with pytest.raises(ValueError, match="Do not support split"):
            _validate_parallel_config_patch(mock_self)

    def test_split_tp_params_patch_fake_tp(self):
        from aura.trainer.train_adapter.mindspeed_rl.patch.vllm_weight_container_patch import split_tp_params_patch

        mock_self = MagicMock()
        mock_self._infer_tp_size = 4
        mock_self._tp_size = 1

        mock_param = MagicMock()

        with patch(f'{_PATCH_MODULE}.is_fake_tp_param', return_value=True):
            result = split_tp_params_patch(mock_self, mock_param, "test_param")

            assert result == mock_param

    def test_split_tp_params_patch_linear_fc1(self):
        import torch
        from aura.trainer.train_adapter.mindspeed_rl.patch.vllm_weight_container_patch import split_tp_params_patch

        mock_self = MagicMock()
        mock_self._infer_tp_size = 2
        mock_self._tp_size = 1
        mock_self._rank = 0

        mock_param = torch.randn(8, 64)

        with patch(f'{_PATCH_MODULE}.is_tensor_parallel_param', return_value=True), \
             patch(f'{_PATCH_MODULE}.is_fake_tp_param', return_value=False), \
             patch(f'{_PATCH_MODULE}.get_tp_group', return_value=None):
            result = split_tp_params_patch(mock_self, mock_param, "linear_fc1.weight")

            assert result is not None

    def test_split_tp_params_patch_qkv(self):
        import torch
        from aura.trainer.train_adapter.mindspeed_rl.patch.vllm_weight_container_patch import split_tp_params_patch

        mock_self = MagicMock()
        mock_self._infer_tp_size = 4
        mock_self._tp_size = 1
        mock_self._rank = 0

        mock_param = torch.randn(12, 64)

        mock_model_config = MagicMock()
        mock_model_config.num_key_value_heads = 1
        mock_model_config.num_attention_heads = 4
        mock_self.model_config = mock_model_config

        with patch(f'{_PATCH_MODULE}.is_tensor_parallel_param', return_value=True), \
             patch(f'{_PATCH_MODULE}.is_fake_tp_param', return_value=False), \
             patch(f'{_PATCH_MODULE}.get_tp_group', return_value=None), \
             patch(f'{_PATCH_MODULE}.get_tensor_parallel_partition_dim', return_value=0):
            result = split_tp_params_patch(mock_self, mock_param, "qkv.weight")

            assert result is not None

    def test_split_tp_params_patch_with_tp_allgather(self):
        import torch
        from aura.trainer.train_adapter.mindspeed_rl.patch.vllm_weight_container_patch import split_tp_params_patch

        mock_self = MagicMock()
        mock_self._infer_tp_size = 4
        mock_self._tp_size = 2
        mock_self._rank = 0

        mock_param = torch.randn(8, 64)

        with patch(f'{_PATCH_MODULE}.is_tensor_parallel_param', return_value=True), \
             patch(f'{_PATCH_MODULE}.is_fake_tp_param', return_value=False), \
             patch(f'{_PATCH_MODULE}.get_tp_group', return_value=None), \
             patch(f'{_PATCH_MODULE}.get_tensor_parallel_partition_dim', return_value=0), \
             patch('torch.distributed.all_gather') as mock_all_gather:
            mock_all_gather.return_value = None
            result = split_tp_params_patch(mock_self, mock_param, "test.weight")

            assert result is not None
            mock_all_gather.assert_called_once()

    def test_collect_name_pairs_for_pp(self):
        from aura.trainer.train_adapter.mindspeed_rl.patch.vllm_weight_container_patch import _collect_name_pairs_for_pp

        mock_self = MagicMock()
        mock_self.weight_names_per_pp = [[["layer_0.weight", "layer_0.bias"]]]

        mock_weight_adaptor = MagicMock()
        mock_weight_adaptor.convert_weight_name_meta.return_value = [["layer_0.weight", "layer_0.bias"]]
        mock_weight_adaptor.global2local_layer.return_value = "layer_0"
        mock_weight_adaptor.replace_name_i2t.return_value = "layer_0.weight"
        mock_self.weight_adaptor = mock_weight_adaptor

        mock_self._vpp_layer_list = [[0]]
        mock_self._vpp_size = 1
        mock_self._global2local_map = None

        result = _collect_name_pairs_for_pp(mock_self, 0)

        assert len(result) > 0

    def _make_simple_ep_self(self, ep_size=1, num_experts=0, num_local_experts=_UNSET):
        mock_self = MagicMock()
        mock_self._vpp_size = 1
        mock_self._pp_rank = 0
        mock_self._ep_size = ep_size
        mock_self._tp_size = 1
        mock_self.moe_tp_extend_ep = False
        mock_self.parallel_state = MagicMock()
        mock_self.parallel_state.get_tensor_model_parallel_rank.return_value = 0
        mock_self.num_experts = num_experts
        if num_local_experts is not _UNSET:
            mock_self.num_local_experts = num_local_experts
        mock_self._vpp_rank = 0
        return mock_self

    def _make_megatron_model(self, params_dict=None, buffers_dict=None):
        mock_model = MagicMock()
        mock_model[0].named_buffers.return_value = buffers_dict or {}
        mock_model[0].named_parameters.return_value = params_dict or {}
        return mock_model

    def _make_weight_adaptor(self, adjusted_dict=None, converted_names=None):
        mock_adaptor = MagicMock()
        mock_adaptor.adjust_megatron_param_dict.return_value = adjusted_dict or {}
        mock_adaptor.convert_weight_name_meta.return_value = converted_names or []
        return mock_adaptor

    def test_get_simple_ep_params(self):
        import torch
        from aura.trainer.train_adapter.mindspeed_rl.patch.vllm_weight_container_patch import _get_simple_ep_params

        mock_self = self._make_simple_ep_self()
        mock_self._unwrap_megatron_model.return_value = self._make_megatron_model()
        mock_self.weight_adaptor = self._make_weight_adaptor()
        mock_self._collect_name_pairs_for_pp.return_value = []

        result = _get_simple_ep_params(mock_self)

        assert "__simple_ep_meta__" in result

    def test_get_simple_ep_params_with_experts(self):
        import torch
        from aura.trainer.train_adapter.mindspeed_rl.patch.vllm_weight_container_patch import _get_simple_ep_params

        mock_param = torch.randn(64, 32)
        mock_self = self._make_simple_ep_self(ep_size=2, num_experts=8, num_local_experts=4)
        mock_self._unwrap_megatron_model.return_value = self._make_megatron_model(
            params_dict={"mlp.experts.w13.weight": mock_param}
        )
        mock_self.weight_adaptor = self._make_weight_adaptor(
            adjusted_dict={"mlp.experts.w13.weight": mock_param},
            converted_names=[["model.layers.0.mlp.experts.w13.weight"]]
        )
        mock_self._collect_name_pairs_for_pp.return_value = [("model.layers.0.mlp.experts.w13.weight", 0, "mlp.experts.w13.weight")]

        with patch(f'{_PATCH_MODULE}.dist') as mock_dist:
            mock_dist.get_rank.return_value = 0
            result = _get_simple_ep_params(mock_self)

            assert "__simple_ep_meta__" in result

    def test_get_simple_ep_params_w2_expert(self):
        import torch
        from aura.trainer.train_adapter.mindspeed_rl.patch.vllm_weight_container_patch import _get_simple_ep_params

        mock_param = torch.randn(32, 64)
        mock_self = self._make_simple_ep_self(ep_size=2, num_experts=8, num_local_experts=4)
        mock_self._unwrap_megatron_model.return_value = self._make_megatron_model(
            params_dict={"mlp.experts.w2.weight": mock_param}
        )
        mock_self.weight_adaptor = self._make_weight_adaptor(
            adjusted_dict={"mlp.experts.w2.weight": mock_param},
            converted_names=[["model.layers.0.mlp.experts.w2.weight"]]
        )
        mock_self._collect_name_pairs_for_pp.return_value = [("model.layers.0.mlp.experts.w2.weight", 0, "mlp.experts.w2.weight")]

        with patch(f'{_PATCH_MODULE}.dist') as mock_dist:
            mock_dist.get_rank.return_value = 0
            result = _get_simple_ep_params(mock_self)

            assert "__simple_ep_meta__" in result
            assert "slices" in result["__simple_ep_meta__"]

    def test_get_simple_ep_params_fallback_num_local_experts(self):
        import torch
        from aura.trainer.train_adapter.mindspeed_rl.patch.vllm_weight_container_patch import _get_simple_ep_params

        mock_param = torch.randn(64, 32)
        mock_self = self._make_simple_ep_self(ep_size=2, num_experts=8, num_local_experts=None)
        mock_self._unwrap_megatron_model.return_value = self._make_megatron_model(
            params_dict={"mlp.experts.w13.weight": mock_param}
        )
        mock_self.weight_adaptor = self._make_weight_adaptor(
            adjusted_dict={"mlp.experts.w13.weight": mock_param},
            converted_names=[["model.layers.0.mlp.experts.w13.weight"]]
        )
        mock_self._collect_name_pairs_for_pp.return_value = [("model.layers.0.mlp.experts.w13.weight", 0, "mlp.experts.w13.weight")]

        with patch(f'{_PATCH_MODULE}.dist') as mock_dist:
            mock_dist.get_rank.return_value = 0
            result = _get_simple_ep_params(mock_self)

            assert "__simple_ep_meta__" in result

    def test_get_simple_ep_params_no_experts(self):
        import torch
        from aura.trainer.train_adapter.mindspeed_rl.patch.vllm_weight_container_patch import _get_simple_ep_params

        mock_param = torch.randn(64, 64)
        mock_self = self._make_simple_ep_self()
        mock_self._unwrap_megatron_model.return_value = self._make_megatron_model(
            params_dict={"attention.weight": mock_param}
        )
        mock_self.weight_adaptor = self._make_weight_adaptor(
            adjusted_dict={"attention.weight": mock_param},
            converted_names=[["model.layers.0.attention.weight"]]
        )
        mock_self._collect_name_pairs_for_pp.return_value = [("model.layers.0.attention.weight", 0, "attention.weight")]

        result = _get_simple_ep_params(mock_self)

        assert "__simple_ep_meta__" in result
        assert "model.layers.0.attention.weight" in result

    def test_split_tp_params_patch_non_tensor_parallel(self):
        from aura.trainer.train_adapter.mindspeed_rl.patch.vllm_weight_container_patch import split_tp_params_patch

        mock_self = MagicMock()
        mock_self._infer_tp_size = 4
        mock_self._tp_size = 1
        mock_self._rank = 0

        mock_param = MagicMock()

        with patch(f'{_PATCH_MODULE}.is_tensor_parallel_param', return_value=False), \
             patch(f'{_PATCH_MODULE}.is_fake_tp_param', return_value=False):
            result = split_tp_params_patch(mock_self, mock_param, "test_param")

            assert result == mock_param

    def _make_update_ep_self(self, infer_ep_size=1, ep_size=1):
        import torch
        mock_self = MagicMock()
        mock_self._pp_size = 1
        mock_self._pp_rank = 0
        mock_self._vpp_size = 1
        mock_self._rank = 0
        mock_self.experts_memory_expand_N = 1
        mock_self._infer_ep_size = infer_ep_size
        mock_self._ep_size = ep_size

        mock_buffer_data = MagicMock()
        mock_buffer_data.device = torch.device('cpu')
        mock_buffer = MagicMock()
        mock_buffer.data = mock_buffer_data

        mock_memory_buffers = {torch.float32: mock_buffer}
        mock_weight_buffers = {0: MagicMock()}
        mock_weight_buffers[0].memory_buffers = mock_memory_buffers
        mock_self.weight_buffers = mock_weight_buffers

        return mock_self

    def _make_experts_memory_buffer(self, tensor_indices, device='cpu'):
        import torch
        mock_data = MagicMock()
        mock_data.device = torch.device(device)
        mock_buffer = MagicMock()
        mock_buffer.data = mock_data
        mock_buffer.tensor_indices = tensor_indices
        mock_buffer.get_by_name.return_value = torch.randn(10, 10)
        return mock_buffer

    def test_update_weight_buffers_ep_patch(self):
        import torch
        from aura.trainer.train_adapter.mindspeed_rl.patch.vllm_weight_container_patch import _update_weight_buffers_ep_patch

        mock_self = self._make_update_ep_self()
        mock_self.weight_names_per_pp = [[["layer_0.weight"]]]

        mock_weight_adaptor = MagicMock()
        mock_weight_adaptor.get_weight_buffer_meta.return_value = {"layer_0.weight": {"dtype": torch.float32, "shape": torch.Size([10, 10])}}
        mock_weight_adaptor.convert_weight_name_meta.return_value = [["layer_0.weight"]]
        mock_weight_adaptor.global2local_layer.return_value = "layer_0"
        mock_weight_adaptor.replace_name_i2t.return_value = "layer_0.weight"
        mock_self.weight_adaptor = mock_weight_adaptor

        mock_self._unwrap_megatron_model.return_value = self._make_megatron_model(
            params_dict={"layer_0.weight": torch.randn(10, 10)}
        )

        mock_experts_memory_buffer = self._make_experts_memory_buffer(
            tensor_indices={"layer_0.weight": (0, torch.Size([10, 10]))}
        )

        with patch(f'{_PATCH_MODULE}.get_weight_buffer_meta_from_buffer', return_value={}), \
             patch(f'{_PATCH_MODULE}.build_experts_memory_buffer', return_value={torch.float32: mock_experts_memory_buffer}), \
             patch(f'{_PATCH_MODULE}.dist') as mock_dist, \
             patch(f'{_PATCH_MODULE}.broadcast_if_gpu') as mock_broadcast:
            mock_dist.get_global_rank.return_value = 0
            _update_weight_buffers_ep_patch(mock_self)

            mock_broadcast.assert_called_once()

    def test_update_weight_buffers_ep_patch_with_expert(self):
        import torch
        from aura.trainer.train_adapter.mindspeed_rl.patch.vllm_weight_container_patch import _update_weight_buffers_ep_patch

        mock_self = self._make_update_ep_self(infer_ep_size=2, ep_size=1)
        mock_self.weight_names_per_pp = [[["model.layers.0.mlp.experts.w13.weight"]]]

        mock_weight_adaptor = MagicMock()
        mock_weight_adaptor.get_weight_buffer_meta.return_value = {"model.layers.0.mlp.experts.w13.weight": {"dtype": torch.float32, "shape": torch.Size([64, 32])}}
        mock_weight_adaptor.convert_weight_name_meta.return_value = [["model.layers.0.mlp.experts.w13.weight"]]
        mock_weight_adaptor.global2local_layer.return_value = "model.layers.0.mlp.experts.w13"
        mock_weight_adaptor.replace_name_i2t.return_value = "mlp.experts.w13.weight"
        mock_self.weight_adaptor = mock_weight_adaptor

        mock_param = torch.randn(64, 32)
        mock_self._unwrap_megatron_model.return_value = self._make_megatron_model(
            params_dict={"mlp.experts.w13.weight": mock_param}
        )

        mock_experts_memory_buffer = self._make_experts_memory_buffer(
            tensor_indices={"model.layers.0.mlp.experts.w13.weight": (0, torch.Size([64, 32]))}
        )
        mock_experts_memory_buffer.get_by_name.return_value = torch.randn(64, 32)
        mock_experts_memory_buffer.copy_by_name = MagicMock()

        with patch(f'{_PATCH_MODULE}.get_weight_buffer_meta_from_buffer', return_value={"model.layers.0.mlp.experts.w13.weight": {"dtype": torch.float32, "shape": torch.Size([64, 32])}}), \
             patch(f'{_PATCH_MODULE}.build_experts_memory_buffer', return_value={torch.float32: mock_experts_memory_buffer}), \
             patch(f'{_PATCH_MODULE}.dist') as mock_dist, \
             patch(f'{_PATCH_MODULE}.broadcast_if_gpu') as mock_broadcast:
            mock_dist.get_global_rank.return_value = 0
            _update_weight_buffers_ep_patch(mock_self)

            mock_experts_memory_buffer.copy_by_name.assert_called_once()
            mock_broadcast.assert_called_once()
