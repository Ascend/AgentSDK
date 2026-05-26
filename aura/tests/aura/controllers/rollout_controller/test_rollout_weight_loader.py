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
import re
import time
import copy
from unittest.mock import MagicMock, patch, ANY
import pytest


# ---------------------------------------------------------------------------
# Fixture: fake module tree for rollout_weight_loader
# ---------------------------------------------------------------------------
@pytest.fixture
def fake_env():
    """Construct isolated fake modules for rollout_weight_loader."""

    # ---- fake torch ----
    fake_torch = types.ModuleType("torch")
    fake_torch.set_num_threads = MagicMock()
    fake_torch.set_num_interop_threads = MagicMock()
    fake_torch.dtype = MagicMock
    fake_torch.Tensor = MagicMock
    fake_torch.empty = MagicMock(return_value=MagicMock())
    fake_torch.cat = MagicMock(return_value=MagicMock())
    fake_torch.chunk = MagicMock(return_value=[MagicMock(), MagicMock()])
    fake_torch.bfloat16 = "bfloat16"
    fake_torch.float32 = "float32"
    fake_torch.long = "long"
    fake_torch.no_grad = MagicMock()
    fake_torch.no_grad.return_value.__enter__ = MagicMock()
    fake_torch.no_grad.return_value.__exit__ = MagicMock()

    # ---- fake ray ----
    fake_ray = types.ModuleType("ray")
    fake_ray.remote = lambda fn: fn
    fake_ray.get = MagicMock(return_value=[])
    fake_ray.available_resources = MagicMock(return_value={"CPU": 64})
    fake_ray.is_initialized = MagicMock(return_value=True)
    fake_ray.init = MagicMock()
    fake_ray.util = types.ModuleType("ray.util")
    fake_ray.util.placement_group = MagicMock()
    fake_ray.util.scheduling_strategies = types.ModuleType("ray.util.scheduling_strategies")
    fake_ray.util.scheduling_strategies.PlacementGroupSchedulingStrategy = MagicMock()
    fake_ray.exceptions = types.ModuleType("ray.exceptions")
    fake_ray.exceptions.GetTimeoutError = TimeoutError

    # ---- fake safetensors.torch ----
    fake_safetensors = types.ModuleType("safetensors")
    fake_safetensors_torch = types.ModuleType("safetensors.torch")
    fake_safetensors_torch.safe_open = MagicMock()
    fake_safetensors_torch.save_file = MagicMock()
    fake_safetensors.torch = fake_safetensors_torch

    # ---- fake glob ----
    fake_glob = types.ModuleType("glob")
    fake_glob.glob = MagicMock(return_value=[])

    # ---- fake multiprocessing ----
    fake_mp = types.ModuleType("multiprocessing")
    fake_mp.get_context = MagicMock(return_value=MagicMock())
    fake_mp.Process = MagicMock()
    fake_mp.Queue = MagicMock()

    # ---- fake threading ----
    fake_threading = types.ModuleType("threading")
    fake_threading.Lock = MagicMock(return_value=MagicMock())

    # ---- fake traceback ----
    fake_traceback = types.ModuleType("traceback")
    fake_traceback.print_exc = MagicMock()

    # ---- fake shutil ----
    fake_shutil = types.ModuleType("shutil")
    fake_shutil.rmtree = MagicMock()

    # ---- fake transformers ----
    fake_transformers = types.ModuleType("transformers")
    fake_transformers.AutoConfig = MagicMock()

    # ---- fake aura loggers ----
    fake_loggers_mod = types.ModuleType("aura.base.log.loggers")
    mock_logger = MagicMock()
    fake_loggers_mod.Loggers = MagicMock(
        return_value=MagicMock(get_logger=MagicMock(return_value=mock_logger))
    )

    # ---- aura packages to locate real file ----
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
    fake_aura_controllers_rollout = types.ModuleType("aura.controllers.rollout_controller")
    fake_aura_controllers_rollout.__path__ = [os.path.join(base, "controllers/rollout_controller")]

    fake_placement_group_mod = types.ModuleType("ray.util.placement_group")
    fake_placement_group_mod.placement_group = MagicMock()
    fake_placement_group_mod.remove_placement_group = MagicMock()
    fake_ray.util.placement_group = fake_placement_group_mod

    fakes = {
        "torch": fake_torch,
        "ray": fake_ray,
        "ray.util": fake_ray.util,
        "ray.util.placement_group": fake_ray.util.placement_group,
        "ray.util.scheduling_strategies": fake_ray.util.scheduling_strategies,
        "ray.exceptions": fake_ray.exceptions,
        "safetensors": fake_safetensors,
        "safetensors.torch": fake_safetensors_torch,
        "glob": fake_glob,
        "multiprocessing": fake_mp,
        "threading": fake_threading,
        "traceback": fake_traceback,
        "shutil": fake_shutil,
        "transformers": fake_transformers,
        "aura.base.log.loggers": fake_loggers_mod,
        "aura": fake_aura,
        "aura.base": fake_aura_base,
        "aura.base.log": fake_aura_base_log,
        "aura.controllers": fake_aura_controllers,
        "aura.controllers.rollout_controller": fake_aura_controllers_rollout,
        "ray.util.placement_group": fake_placement_group_mod,
    }

    target = "aura.controllers.rollout_controller.rollout_weight_loader"
    if target in sys.modules:
        del sys.modules[target]

    with patch.dict(sys.modules, fakes):
        import aura.controllers.rollout_controller.rollout_weight_loader as mod
        yield {
            "mod": mod,
            "mock_logger": mock_logger,
            "fake_torch": fake_torch,
            "fake_ray": fake_ray,
            "fake_glob": fake_glob,
            "fake_mp": fake_mp,
            "fake_shutil": fake_shutil,
            "fake_safetensors_torch": fake_safetensors_torch,
        }

    if target in sys.modules:
        del sys.modules[target]


# ---------------------------------------------------------------------------
# Helpers for constructing Qwen3MoEParamsAssembler without MagicMock comparisons
# ---------------------------------------------------------------------------
def make_hf_config():
    """Return a mock config with integer attributes to avoid MagicMock comparison errors."""
    cfg = MagicMock()
    cfg.hidden_size = 1024
    cfg.num_attention_heads = 16
    cfg.num_key_value_heads = 4
    cfg.head_dim = 128
    cfg.num_hidden_layers = 24
    cfg.num_experts = 8
    cfg.vocab_size = 32000
    del cfg.text_config
    return cfg


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
class TestUtilityFunctions:
    def test_norm_removes_digit_segments(self, fake_env):
        mod = fake_env["mod"]
        assert mod._norm("abc.0.def") == "abc.def"
        assert mod._norm("abc.123.def.456") == "abc.def.456"

    def test_cat_dim0_single(self, fake_env):
        mod = fake_env["mod"]
        t = MagicMock()
        t.shape = [10, 20]
        assert mod.cat_dim0([t]) is t

    def test_cat_dim0_multiple(self, fake_env):
        mod = fake_env["mod"]
        fake_torch = fake_env["fake_torch"]
        t1 = MagicMock(); t1.shape = [2, 5]; t1.dtype = "float"
        t2 = MagicMock(); t2.shape = [3, 5]; t2.dtype = "float"
        fake_torch.empty.return_value = MagicMock()
        result = mod.cat_dim0([t1, t2])
        assert result is not None

    def test_cat_dim1_multiple(self, fake_env):
        mod = fake_env["mod"]
        fake_torch = fake_env["fake_torch"]
        t1 = MagicMock(); t1.shape = [5, 2]
        t2 = MagicMock(); t2.shape = [5, 3]
        fake_torch.empty.return_value = MagicMock()
        result = mod.cat_dim1([t1, t2])
        assert result is not None

    def test_fast_cat_empty_raises(self, fake_env):
        mod = fake_env["mod"]
        with pytest.raises(ValueError):
            mod._fast_cat([], 0)

    def test_fast_cat_single(self, fake_env):
        mod = fake_env["mod"]
        t = MagicMock()
        assert mod._fast_cat([t], 0) is t

    def test_fast_cat_multiple(self, fake_env):
        mod = fake_env["mod"]
        fake_torch = fake_env["fake_torch"]
        fake_torch.cat.return_value = MagicMock()
        result = mod._fast_cat([MagicMock(), MagicMock()], 0)
        assert result is not None

    def test_ceil_div(self, fake_env):
        mod = fake_env["mod"]
        assert mod._ceil_div(5, 2) == 3
        assert mod._ceil_div(4, 2) == 2

    def test_chunk_evenly(self, fake_env):
        mod = fake_env["mod"]
        assert mod._chunk_evenly([1,2,3], 2) == [[1,2], [3]]
        assert mod._chunk_evenly([1], 5) == [[1]]

    def test_parse_pp_tp_ep_valid(self, fake_env):
        mod = fake_env["mod"]
        pp, tp, ep = mod._parse_pp_tp_ep("model_pp0_tp1_ep2.safetensors")
        assert pp == 0 and tp == 1 and ep == 2

    def test_parse_pp_tp_ep_no_ep(self, fake_env):
        mod = fake_env["mod"]
        pp, tp, ep = mod._parse_pp_tp_ep("model_pp0_tp1.safetensors")
        assert ep == 0

    def test_parse_pp_tp_ep_invalid(self, fake_env):
        mod = fake_env["mod"]
        with pytest.raises(ValueError):
            mod._parse_pp_tp_ep("invalid_name.safetensors")


class TestReaderCache:
    def test_get_and_cache(self, fake_env):
        mod = fake_env["mod"]
        safe_open_mock = fake_env["fake_safetensors_torch"].safe_open
        fake_file = MagicMock()
        safe_open_mock.return_value = fake_file
        rc = mod._ReaderCache(max_open=2)
        f, lock = rc.get("path1")
        assert f is fake_file
        assert len(rc._cache) == 1

    def test_lru_eviction(self, fake_env):
        mod = fake_env["mod"]
        safe_open_mock = fake_env["fake_safetensors_torch"].safe_open
        fake_file1 = MagicMock()
        fake_file2 = MagicMock()
        fake_file3 = MagicMock()
        safe_open_mock.side_effect = [fake_file1, fake_file2, fake_file3]
        rc = mod._ReaderCache(max_open=2)
        rc.get("path1")
        rc.get("path2")
        rc.get("path3")
        assert len(rc._cache) == 2
        assert "path1" not in rc._cache

    def test_load_tensor(self, fake_env):
        mod = fake_env["mod"]
        safe_open_mock = fake_env["fake_safetensors_torch"].safe_open
        fake_file = MagicMock()
        fake_file.get_tensor.return_value = MagicMock()
        safe_open_mock.return_value = fake_file
        rc = mod._ReaderCache()
        t = rc.load_tensor("path", "key")
        assert t is not None

    def test_read_meta(self, fake_env):
        mod = fake_env["mod"]
        safe_open_mock = fake_env["fake_safetensors_torch"].safe_open
        fake_file = MagicMock()
        fake_file.metadata.return_value = {"__simple_ep_meta__": '{"a":1}'}
        safe_open_mock.return_value = fake_file
        rc = mod._ReaderCache()
        meta = rc.read_meta("path")
        assert meta == {"a": 1}

    def test_close_all(self, fake_env):
        mod = fake_env["mod"]
        rc = mod._ReaderCache()
        safe_open_mock = fake_env["fake_safetensors_torch"].safe_open
        fake_file = MagicMock()
        safe_open_mock.return_value = fake_file
        rc.get("path")
        rc.close_all()
        fake_file.close.assert_called_once()


class TestLaunchedGroup:
    def test_get_flatten(self, fake_env):
        mod = fake_env["mod"]
        fake_ray = fake_env["fake_ray"]
        fake_ray.get.return_value = [[1,2], [3,4]]
        lg = mod.LaunchedGroup(refs=["r1", "r2"], pg=None, flatten=True)
        result = lg.get()
        assert result == [1,2,3,4]

    def test_get_no_flatten(self, fake_env):
        mod = fake_env["mod"]
        fake_ray = fake_env["fake_ray"]
        fake_ray.get.return_value = [1,2]
        lg = mod.LaunchedGroup(refs=["r"], pg=None, flatten=False)
        assert lg.get() == [1,2]

    def test_pg_removed_on_failure(self, fake_env):
        mod = fake_env["mod"]
        pg = MagicMock()
        fake_ray = fake_env["fake_ray"]
        fake_ray.get.side_effect = Exception("ray error")
        lg = mod.LaunchedGroup(refs=["r"], pg=pg)
        with pytest.raises(Exception, match="ray error"):
            lg.get()
        mod.ray.util.placement_group.remove_placement_group.assert_called_with(pg)


class TestPlanAndLaunch:
    def test_plan_num_cpus(self, fake_env):
        mod = fake_env["mod"]
        fake_ray = fake_env["fake_ray"]
        fake_ray.available_resources.return_value = {"CPU": 64}
        cpus = mod.plan_num_cpus(4, 16)
        assert cpus == 8  # fallback calculation

    def test_create_pg_with_fallback_success(self, fake_env):
        mod = fake_env["mod"]
        fake_ray = fake_env["fake_ray"]
        pg_mock = MagicMock()
        # Mock placement_group directly in the module to return pg_mock
        with patch.object(mod, "placement_group", return_value=pg_mock), \
             patch.object(mod.ray, "get", return_value=None):
            pg, final_cpus = mod.create_pg_with_fallback(workers=2, bundle_cpus=4)
        assert pg is pg_mock

    def test_create_pg_with_fallback_timeout_then_success(self, fake_env):
        mod = fake_env["mod"]
        fake_ray = fake_env["fake_ray"]
        pg1 = MagicMock()
        pg2 = MagicMock()
        # Simulate two calls: first returns pg1 but then times out, second returns pg2 successfully
        with patch.object(mod, "placement_group", side_effect=[pg1, pg2]), \
             patch.object(mod.ray, "get", side_effect=[mod.ray.exceptions.GetTimeoutError(), None]):
            pg, final_cpus = mod.create_pg_with_fallback(workers=2, bundle_cpus=8)
        assert pg is pg2

    def test_launch_chunked_with_pg_single_chunk(self, fake_env):
        mod = fake_env["mod"]
        with patch.object(mod, "plan_num_cpus", return_value=4), \
             patch.object(mod, "assemble_subset_worker", MagicMock()) as mock_worker:
            mock_worker.options.return_value.remote.return_value = "ref"
            lg = mod.launch_chunked_with_pg(
                items=[1,2,3],
                workers=1,
                ideal_cpus_per_worker=4,
                make_kwargs=lambda idx, chunk: {"data": chunk},
                use_pg_if_one=False,
            )
        assert lg.refs == ["ref"]

    def test_launch_chunked_with_pg_multiple(self, fake_env):
        mod = fake_env["mod"]
        with patch.object(mod, "create_pg_with_fallback", return_value=(MagicMock(), 4)), \
             patch.object(mod, "assemble_subset_worker", MagicMock()) as mock_worker:
            mock_worker.options.return_value.remote.return_value = "ref"
            lg = mod.launch_chunked_with_pg(
                items=[1,2,3,4],
                workers=2,
                ideal_cpus_per_worker=4,
                make_kwargs=lambda idx, chunk: {"data": chunk},
                use_pg_if_one=True,
            )
        assert lg.refs == ["ref", "ref"]


class TestParamsAssembler:
    def test_group_paths_by_pp_ep_tp(self, fake_env):
        mod = fake_env["mod"]
        paths = [
            "dir/pp0_tp0_ep0.safetensors",
            "dir/pp0_tp1_ep0.safetensors",
            "dir/pp0_tp0_ep1.safetensors",
        ]
        buckets = mod.ParamsAssembler.group_paths_by_pp_ep_tp(paths)
        assert 0 in buckets
        assert 0 in buckets[0]
        assert 1 in buckets[0]

    def test_cast_if_needed(self, fake_env):
        mod = fake_env["mod"]
        assembler = mod.ParamsAssembler(target_dtype=mod.torch.bfloat16)
        t = MagicMock()
        t.is_floating_point.return_value = True
        t.dtype = "float32"
        assembler._cast_if_needed(t)
        t.to.assert_called_with(mod.torch.bfloat16)

    def test_write_file(self, fake_env):
        mod = fake_env["mod"]
        assembler = mod.ParamsAssembler()
        tensors = {"a": MagicMock()}
        with patch("os.makedirs"), patch("os.replace"):
            assembler.write_file(tensors, "/tmp/out.safetensors")
        fake_env["fake_safetensors_torch"].save_file.assert_called_once()

    def test_shard_w13_for_tp_grouped(self, fake_env):
        mod = fake_env["mod"]
        assembler = mod.ParamsAssembler(infer_tp=2, infer_dp=1)
        assembler.w13_cols_mode = "grouped"
        t = MagicMock()
        t.shape = [8, 4, 16]
        assembler.shard_w13_for_tp(t, 0)
        # Just ensure it doesn't raise
        t.__getitem__.assert_called()

    def test_final_rank_split_replicates(self, fake_env):
        mod = fake_env["mod"]
        assembler = mod.ParamsAssembler(infer_tp=1, infer_dp=1)
        tensors = {"weight": MagicMock()}
        assembler.get_tp_split_axis = MagicMock(return_value=None)
        assembler.is_fused_qkv_weight = MagicMock(return_value=False)
        assembler.is_fused_qkv_bias = MagicMock(return_value=False)
        result = assembler.final_rank_split(tensors)
        assert len(result) == 1
        assert "weight" in result[0]

    def test_assemble_dir_basic(self, fake_env):
        mod = fake_env["mod"]
        assembler = mod.ParamsAssembler()
        rc = mod._ReaderCache(max_open=2)
        safe_open_mock = fake_env["fake_safetensors_torch"].safe_open
        fake_file = MagicMock()
        fake_file.keys.return_value = {"key1"}
        safe_open_mock.return_value = fake_file
        pp2ep_paths = {0: {0: [(0, "dummy.safetensors")]}}
        with patch.object(assembler, "_cast_if_needed", side_effect=lambda x: x), \
             patch.object(assembler, "get_train_tp_concat_axis_2d", return_value=None), \
             patch.object(assembler, "is_w13", return_value=False), \
             patch.object(assembler, "is_w2", return_value=False):
            tensors = assembler.assemble_dir(pp2ep_paths, rc)
        assert isinstance(tensors, dict)


class TestQwen3MoEParamsAssembler:
    def test_init_configs(self, fake_env):
        mod = fake_env["mod"]
        hf_config = make_hf_config()
        assembler = mod.Qwen3MoEParamsAssembler(hf_config)
        assert assembler.hidden_size == 1024
        assert assembler.num_attention_heads == 16

    def test_is_w13(self, fake_env):
        mod = fake_env["mod"]
        assembler = mod.Qwen3MoEParamsAssembler(make_hf_config())
        name = "model.layers.0.mlp.experts.w13_weight"
        assert assembler.is_w13(name)

    def test_is_w2(self, fake_env):
        mod = fake_env["mod"]
        assembler = mod.Qwen3MoEParamsAssembler(make_hf_config())
        name = "model.layers.0.mlp.experts.w2_weight"
        assert assembler.is_w2(name)

    def test_convert_w13_weights(self, fake_env):
        mod = fake_env["mod"]
        assembler = mod.Qwen3MoEParamsAssembler(make_hf_config())
        t = MagicMock()
        t.view.return_value.permute.return_value.contiguous.return_value = t
        result = assembler.convert_w13_weights(t, 2, 4)
        assert result is not None

    def test_get_train_tp_concat_axis_2d_expert(self, fake_env):
        mod = fake_env["mod"]
        assembler = mod.Qwen3MoEParamsAssembler(make_hf_config())
        assembler.has_moe = True
        assert assembler.get_train_tp_concat_axis_2d("model.layers.0.mlp.experts.w13_weight") == 1
        assert assembler.get_train_tp_concat_axis_2d("model.layers.0.mlp.experts.w2_weight") == 1

    def test_get_tp_split_axis_o_proj(self, fake_env):
        mod = fake_env["mod"]
        assembler = mod.Qwen3MoEParamsAssembler(make_hf_config())
        assert assembler.get_tp_split_axis("model.layers.0.self_attn.o_proj.weight") == 1


class TestRunDistributedQwen3Assemble:
    def test_single_process(self, fake_env):
        mod = fake_env["mod"]
        fake_glob = fake_env["fake_glob"]
        fake_glob.glob.return_value = ["model_pp0_tp0_ep0.safetensors"]

        with patch.object(mod.Qwen3MoEParamsAssembler, "assemble_split_write_core", return_value=[]) as mock_assemble, \
            patch("os.path.exists", return_value=False), \
            patch("os.makedirs"), \
            patch("shutil.rmtree"), \
            patch.object(mod, "list_all_keys", return_value=["k1"]), \
            patch.object(mod, "launch_chunked_with_pg", return_value=MagicMock()) as mock_launch:
            mock_launch.return_value.get.return_value = []

            result = mod.run_distributed_qwen3_assemble(
                train_save_path="/train",
                hf_config=make_hf_config(),
                infer_tp=1,
                weights_version=1,
                inference_save_path="/infer",
                num_workers=0,
                num_procs=0,
            )
            assert result is None
            mock_assemble.assert_called_once()

    def test_with_workers(self, fake_env):
        mod = fake_env["mod"]
        fake_glob = fake_env["fake_glob"]
        fake_glob.glob.return_value = ["model_pp0_tp0_ep0.safetensors"]

        with patch.object(mod, "list_all_keys", return_value=["k1"]), \
             patch.object(mod, "launch_chunked_with_pg", return_value=MagicMock()) as mock_launch, \
             patch("os.path.exists", return_value=False), \
             patch("os.makedirs"), \
             patch("shutil.rmtree"):
            mock_launch.return_value.get.return_value = []
            mod.run_distributed_qwen3_assemble(
                train_save_path="/train",
                hf_config=make_hf_config(),
                infer_tp=1,
                weights_version=2,
                inference_save_path="/infer",
                num_workers=2,
                num_procs=2,
            )
            mock_launch.assert_called_once()


class TestParamsAssemblerExtended:
    """Cover branches in ParamsAssembler and Qwen3MoEParamsAssembler not touched yet."""

    def test_reshape_qkv_fused_weight(self, fake_env):
        mod = fake_env["mod"]
        assembler = mod.Qwen3MoEParamsAssembler(make_hf_config())
        assembler.infer_tp = 2
        assembler.num_attention_heads = 16
        assembler.num_key_value_heads = 8
        assembler.head_dim = 64
        assembler.head_dim_scale = 1
        with patch.object(assembler, "is_fused_qkv_weight", return_value=True):
            # ng = 8/2=4, repeats=16/8=2, repeats+2=4, head_dim=64 => shape[0]=4*4*64=1024
            t = MagicMock()
            t.shape = [1024, 512]   # arbitrary hidden
            assembler.reshape_qkv_megatron_local("qkv", t)

    def test_reshape_qkv_fused_bias(self, fake_env):
        mod = fake_env["mod"]
        assembler = mod.Qwen3MoEParamsAssembler(make_hf_config())
        assembler.infer_tp = 2
        assembler.num_attention_heads = 16
        assembler.num_key_value_heads = 8
        assembler.head_dim = 64
        assembler.head_dim_scale = 1
        with patch.object(assembler, "is_fused_qkv_bias", return_value=True), \
             patch.object(assembler, "is_fused_qkv_weight", return_value=False):
            t = MagicMock()
            t.shape = [1024]   # 1D bias, same head_dim calculation: 1024/(4*4)=64
            assembler.reshape_qkv_megatron_local("bias", t)

    def test_get_weight_3D_shape_w13(self, fake_env):
        mod = fake_env["mod"]
        assembler = mod.Qwen3MoEParamsAssembler(make_hf_config())
        assembler.hidden_size = 1024
        assembler.num_experts = 8
        old_shape = [1024, 32]  # 32 = 8*4 (per=4)
        ret = assembler.get_weight_3D_shape("model.layers.0.mlp.experts.w13_weight", old_shape)
        assert ret == (1024, 8, 4)

    def test_get_weight_3D_shape_w2(self, fake_env):
        mod = fake_env["mod"]
        assembler = mod.Qwen3MoEParamsAssembler(make_hf_config())
        assembler.hidden_size = 1024
        assembler.num_experts = 8
        old_shape = [32, 1024]  # 32 = 8*4 (per=4)
        ret = assembler.get_weight_3D_shape("model.layers.0.mlp.experts.w2_weight", old_shape)
        assert ret == (8, 4, 1024)

    def test_get_weight_3D_permute(self, fake_env):
        mod = fake_env["mod"]
        assembler = mod.Qwen3MoEParamsAssembler(make_hf_config())
        assert assembler.get_weight_3D_permute("w13_weight") == (1,2,0)
        assert assembler.get_weight_3D_permute("w2_weight") == (0,2,1)
        assert assembler.get_weight_3D_permute("other") is None

    def test_shard_w13_for_tp_with_axi_grouped(self, fake_env):
        mod = fake_env["mod"]
        assembler = mod.ParamsAssembler(infer_tp=2, infer_dp=1)
        assembler.w13_cols_mode = "grouped"
        t = MagicMock()
        t.shape = [8, 4, 16]
        t.ndim = 3
        assembler.shard_w13_for_tp_with_axi(t, 0, split_axis=1)

    def test_shard_w13_for_tp_with_axi_interleaved(self, fake_env):
        mod = fake_env["mod"]
        assembler = mod.ParamsAssembler(infer_tp=2, infer_dp=1)
        assembler.w13_cols_mode = "interleaved"
        t = MagicMock()
        t.shape = [8, 4, 16]
        t.ndim = 3
        assembler.shard_w13_for_tp_with_axi(t, 1, split_axis=2)

    def test_shard_w2_for_tp_interleaved(self, fake_env):
        mod = fake_env["mod"]
        assembler = mod.ParamsAssembler(infer_tp=2, infer_dp=2)
        assembler.w2_rows_mode = "interleaved"
        t = MagicMock()
        t.shape = [8, 16, 4]  # dim 2 size 4 divisible by R=4? R = TP*DP=4, 4%4=0
        t.ndim = 3
        # split_axis=2, R=4, dim=4 -> splitted_dim=1
        assembler.shard_w2_for_tp(t, 1, split_axis=2)

    def test_final_rank_split_new_w13(self, fake_env):
        mod = fake_env["mod"]
        assembler = mod.Qwen3MoEParamsAssembler(make_hf_config())
        assembler.infer_tp = 2
        assembler.infer_dp = 1
        t = MagicMock()
        t.shape = [8, 4, 16]
        tensors = {"w13_weight": t}
        with patch.object(assembler, "is_w13", return_value=True), \
             patch.object(assembler, "is_w2", return_value=False):
            result = assembler.final_rank_split_new(tensors)
        assert len(result) == assembler._Teff

    def test_final_rank_split_new_w2_interleaved(self, fake_env):
        mod = fake_env["mod"]
        assembler = mod.Qwen3MoEParamsAssembler(make_hf_config())
        assembler.infer_tp = 2
        assembler.infer_dp = 1
        assembler.w2_rows_mode = "interleaved"
        t = MagicMock()
        t.shape = [8, 16, 4]  # dim 2 per_total? need to satisfy shard_w2_for_tp
        tensors = {"w2_weight": t}
        with patch.object(assembler, "is_w13", return_value=False), \
             patch.object(assembler, "is_w2", return_value=True):
            result = assembler.final_rank_split_new(tensors)
        assert len(result) == assembler._Teff

    def test_final_rank_split_has_moe_w2_grouped(self, fake_env):
        mod = fake_env["mod"]
        assembler = mod.Qwen3MoEParamsAssembler(make_hf_config())
        assembler.infer_tp = 2
        assembler.infer_dp = 1
        assembler.w2_rows_mode = "grouped"
        t = MagicMock()
        t.shape = [8, 4, 16]  # shape[1] = per_total must be divisible by T=2
        tensors = {"w2_weight": t}
        with patch.object(assembler, "is_w13", return_value=False), \
             patch.object(assembler, "is_w2", return_value=True):
            result = assembler.final_rank_split(tensors)
        assert len(result) == assembler.infer_tp

    def test_convert_w13_weights_computation(self, fake_env):
        mod = fake_env["mod"]
        assembler = mod.Qwen3MoEParamsAssembler(make_hf_config())
        t = MagicMock()
        # shape arbitrary, we just care that conversion runs
        assembler.convert_w13_weights(t, 2, 4)


class TestPlanAndLaunchMore:
    """Cover more branches of placement group logic."""

    def test_create_pg_with_fallback_timeout_exhausted(self, fake_env):
        mod = fake_env["mod"]
        fake_ray = fake_env["fake_ray"]
        # Simulate repeated timeouts until b < min_cpus
        with patch.object(mod, "placement_group", return_value=MagicMock()), \
             patch.object(mod.ray, "get", side_effect=mod.ray.exceptions.GetTimeoutError()), \
             patch.object(mod.ray.util.placement_group, "remove_placement_group"):
            with pytest.raises(RuntimeError, match="Could not place PG"):
                mod.create_pg_with_fallback(workers=2, bundle_cpus=8, min_cpus=1)

    def test_launch_chunked_with_pg_empty_items(self, fake_env):
        mod = fake_env["mod"]
        lg = mod.launch_chunked_with_pg(
            items=[],
            workers=2,
            ideal_cpus_per_worker=4,
            make_kwargs=lambda idx, chunk: {},
        )
        assert lg.refs == []

    def test_launch_chunked_with_pg_exception_during_create(self, fake_env):
        mod = fake_env["mod"]
        pg_mock = MagicMock()
        with patch.object(mod, "create_pg_with_fallback", return_value=(pg_mock, 4)), \
            patch.object(mod, "remove_placement_group") as mock_rm:
            def raising_make(idx, chunk):
                raise RuntimeError("boom")
            with pytest.raises(RuntimeError, match="boom"):
                mod.launch_chunked_with_pg(
                    items=[1,2,3,4],
                    workers=2,
                    ideal_cpus_per_worker=4,
                    make_kwargs=raising_make,
                    use_pg_if_one=True,
                )
            mock_rm.assert_called_with(pg_mock)

    def test_launched_group_pg_remove_exception(self, fake_env):
        mod = fake_env["mod"]
        fake_ray = fake_env["fake_ray"]
        pg = MagicMock()
        mod.ray.util.placement_group.remove_placement_group.side_effect = Exception("remove error")
        fake_ray.get.return_value = [[1,2]]  # flatten path
        lg = mod.LaunchedGroup(refs=["r"], pg=pg, flatten=True)
        # should not raise, just log warning
        result = lg.get()
        assert result == [1,2]


class TestReaderCacheExtended:
    """Cover LRU eviction and copy_on_read."""

    def test_lru_with_eviction_no_close(self, fake_env):
        mod = fake_env["mod"]
        safe_open_mock = fake_env["fake_safetensors_torch"].safe_open
        f1 = MagicMock()
        f2 = MagicMock()
        f3 = MagicMock()
        safe_open_mock.side_effect = [f1, f2, f3]
        rc = mod._ReaderCache(max_open=2)
        rc.get("a")
        rc.get("b")
        rc.get("c")  # should evict 'a'
        assert "a" not in rc._cache

    def test_load_tensor_copy_on_read(self, fake_env):
        mod = fake_env["mod"]
        safe_open_mock = fake_env["fake_safetensors_torch"].safe_open
        fake_file = MagicMock()
        fake_tensor = MagicMock()
        fake_file.get_tensor.return_value = fake_tensor
        safe_open_mock.return_value = fake_file
        rc = mod._ReaderCache(copy_on_read=True)
        result = rc.load_tensor("path", "key")
        # copy_on_read triggers .to(dtype, copy=True).contiguous()
        fake_tensor.to.assert_called_with(fake_tensor.dtype, copy=True)
        assert result is fake_tensor.to.return_value.contiguous.return_value

    def test_read_meta_none_metadata(self, fake_env):
        mod = fake_env["mod"]
        safe_open_mock = fake_env["fake_safetensors_torch"].safe_open
        fake_file = MagicMock()
        fake_file.metadata.return_value = None
        safe_open_mock.return_value = fake_file
        rc = mod._ReaderCache()
        meta = rc.read_meta("path")
        assert meta is None

    def test_read_meta_json_error(self, fake_env):
        mod = fake_env["mod"]
        safe_open_mock = fake_env["fake_safetensors_torch"].safe_open
        fake_file = MagicMock()
        fake_file.metadata.return_value = {"__simple_ep_meta__": "not json"}
        safe_open_mock.return_value = fake_file
        rc = mod._ReaderCache()
        meta = rc.read_meta("path")
        assert meta is None  # json.loads raises, caught


class TestAssembleSubsetWorker:
    """Test the Ray remote function directly (without real Ray)."""

    def test_single_process_path(self, fake_env):
        mod = fake_env["mod"]
        assembler = mod.Qwen3MoEParamsAssembler(make_hf_config())
        # Mock assemble_split_write_core to return a list
        with patch.object(assembler, "assemble_split_write_core", return_value=["file1"]) as mock_core:
            # Call assemble_subset_worker directly (it's a normal function now)
            result = mod.assemble_subset_worker(
                final_dir="/tmp",
                reduce_idx=0,
                names=["key1"],
                pp2ep_paths={0: {0: [(0, "dummy.safetensors")]}},
                assembler=assembler,
                num_procs=1,
            )
            assert result == ["file1"]
            mock_core.assert_called_once()

    def test_multiprocess_path(self, fake_env):
        mod = fake_env["mod"]
        assembler = mod.Qwen3MoEParamsAssembler(make_hf_config())

        fake_mp = fake_env["fake_mp"]
        queue_mock = MagicMock()
        queue_mock.get.side_effect = [
            (0, ["child_file"], None),
            (1, ["child_file2"], None)
        ]
        fake_mp.get_context.return_value.Queue.return_value = queue_mock

        process_mock = MagicMock()
        fake_mp.get_context.return_value.Process = process_mock

        with patch("os.environ.setdefault"):
            result = mod.assemble_subset_worker(
                final_dir="/tmp",
                reduce_idx=1,
                names=["key1", "key2", "key3"],
                pp2ep_paths={0: {0: [(0, "dummy.safetensors")]}},
                assembler=assembler,
                num_procs=2,
            )
        assert result == ["child_file", "child_file2"]


class TestListAllKeys:
    def test_list_all_keys(self, fake_env):
        mod = fake_env["mod"]
        safe_open_mock = fake_env["fake_safetensors_torch"].safe_open
        fake_file = MagicMock()
        fake_file.keys.return_value = {"a", "b"}
        safe_open_mock.return_value = fake_file
        rc = mod._ReaderCache()
        pp2ep_paths = {0: {0: [(0, "path.safetensors")]}}
        keys = mod.list_all_keys(pp2ep_paths, rc)
        assert keys == ["a", "b"]

class TestParamsAssemblerAdvanced:
    """Target deep branches in assemble_dir for experts and non-experts."""

    def test_assemble_dir_multi_ep_multi_tp_experts(self, fake_env):
        mod = fake_env["mod"]
        assembler = mod.Qwen3MoEParamsAssembler(make_hf_config())
        assembler.hidden_size = 32
        assembler.num_experts = 4
        assembler.infer_tp = 1
        assembler.infer_dp = 1

        rc = mod._ReaderCache(max_open=4)
        safe_open_mock = fake_env["fake_safetensors_torch"].safe_open
        fake_file = MagicMock()
        fake_file.keys.return_value = {"w13_weight", "w2_weight"}
        safe_open_mock.return_value = fake_file

        pp2ep_paths = {
            0: {
                0: [(0, "ep0_tp0.safetensors"), (1, "ep0_tp1.safetensors")],
                1: [(0, "ep1_tp0.safetensors"), (1, "ep1_tp1.safetensors")],
            }
        }

        def load_tensor_side_effect(path, key):
            t = MagicMock()
            t.shape = [2, 32]
            t.dtype = "float32"
            t.is_floating_point.return_value = True
            return t
        rc.load_tensor = MagicMock(side_effect=load_tensor_side_effect)

        def read_meta_side_effect(path):
            return {
                "slices": {
                    "w13_weight": {"offset": 0, "length": 2, "axis": 1},
                    "w2_weight": {"offset": 0, "length": 2, "axis": 0},
                },
                "num_experts_total": 4,
            }
        rc.read_meta = MagicMock(side_effect=read_meta_side_effect)

        with patch.object(assembler, "is_w13", return_value=True), \
             patch.object(assembler, "is_w2", return_value=False), \
             patch.object(assembler, "_cast_if_needed", side_effect=lambda x: x), \
             patch.object(assembler, "unflatten_weight", side_effect=lambda n, w, p, l: w):
            tensors = assembler.assemble_dir(pp2ep_paths, rc)
        assert "w13_weight" in tensors

    def test_assemble_dir_non_expert_multi_tp(self, fake_env):
        mod = fake_env["mod"]
        assembler = mod.Qwen3MoEParamsAssembler(make_hf_config())
        assembler.hidden_size = 64
        rc = mod._ReaderCache(max_open=4)
        safe_open_mock = fake_env["fake_safetensors_torch"].safe_open
        fake_file = MagicMock()
        fake_file.keys.return_value = {"lm_head.weight"}
        safe_open_mock.return_value = fake_file

        pp2ep_paths = {0: {0: [(0, "f0.safetensors"), (1, "f1.safetensors")]}}

        def load_tensor(path, key):
            t = MagicMock()
            t.shape = [32, 64]
            t.dtype = "float32"
            t.is_floating_point.return_value = True
            return t
        rc.load_tensor = MagicMock(side_effect=load_tensor)
        rc.read_meta = MagicMock(return_value={})

        with patch.object(assembler, "get_train_tp_concat_axis_2d", return_value=0), \
             patch.object(assembler, "_cast_if_needed", side_effect=lambda x: x):
            tensors = assembler.assemble_dir(pp2ep_paths, rc)
        assert "lm_head.weight" in tensors

    def test_assemble_dir_exception_handling(self, fake_env):
        mod = fake_env["mod"]
        assembler = mod.Qwen3MoEParamsAssembler(make_hf_config())
        rc = mod._ReaderCache(max_open=2)
        safe_open_mock = fake_env["fake_safetensors_torch"].safe_open
        fake_file = MagicMock()
        fake_file.keys.side_effect = Exception("keys failed")
        safe_open_mock.return_value = fake_file

        pp2ep_paths = {0: {0: [(0, "bad.safetensors")]}}
        with pytest.raises(ValueError):
            assembler.assemble_dir(pp2ep_paths, rc)

    def test_fast_cat_contiguous_call(self, fake_env):
        mod = fake_env["mod"]
        t1 = MagicMock()
        t2 = MagicMock()
        t1.contiguous.return_value = t1
        t2.contiguous.return_value = t2
        mod.torch.cat.return_value = MagicMock()
        result = mod._fast_cat([t1, t2], dim=0)
        assert result is not None

    def test_w2_interleave_rows_complex(self, fake_env):
        mod = fake_env["mod"]
        assembler = mod.ParamsAssembler(infer_tp=2, infer_dp=2)
        assembler.w2_rows_mode = "interleaved"
        t = MagicMock()
        t.shape = [32, 128]
        assembler._w2_interleave_rows(t, ln=4)

    def test_assemble_dir_single_tp_window(self, fake_env):
        """Test the single TP shard branch inside expert window handling."""
        mod = fake_env["mod"]
        assembler = mod.Qwen3MoEParamsAssembler(make_hf_config())
        assembler.hidden_size = 64
        assembler.num_experts = 2
        rc = mod._ReaderCache(max_open=2)
        safe_open_mock = fake_env["fake_safetensors_torch"].safe_open
        fake_file = MagicMock()
        fake_file.keys.return_value = {"w13_weight"}
        safe_open_mock.return_value = fake_file

        pp2ep_paths = {0: {0: [(0, "single.safetensors")]}}

        def load_tensor(path, key):
            t = MagicMock()
            t.shape = [64, 4]
            t.dtype = "float32"
            t.is_floating_point.return_value = True
            return t
        rc.load_tensor = MagicMock(side_effect=load_tensor)

        rc.read_meta = MagicMock(return_value={
            "slices": {"w13_weight": {"offset": 0, "length": 2, "axis": 1}},
            "num_experts_total": 2,
        })

        with patch.object(assembler, "is_w13", return_value=True), \
             patch.object(assembler, "is_w2", return_value=False), \
             patch.object(assembler, "_cast_if_needed", side_effect=lambda x: x), \
             patch.object(assembler, "unflatten_weight", side_effect=lambda n, w, p, l: w):
            tensors = assembler.assemble_dir(pp2ep_paths, rc)
        assert "w13_weight" in tensors
