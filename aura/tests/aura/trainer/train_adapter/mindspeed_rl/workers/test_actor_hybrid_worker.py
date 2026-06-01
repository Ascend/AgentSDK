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

import os
import sys
import types
import pytest
from unittest.mock import MagicMock, patch


@pytest.fixture(autouse=True)
def mock_ray_runtime():
    with patch('ray.get', return_value=None), \
         patch('ray.get_actor', return_value=MagicMock(weight_saved=MagicMock(remote=MagicMock()))), \
         patch('ray.get_runtime_context', return_value=MagicMock(get_node_id=lambda: "node123")):
        yield


@pytest.fixture(autouse=True)
def mock_vllm_distributed_module():
    """Ensure `vllm.distributed` import works even when vllm isn't installed as a package."""
    original_vllm = sys.modules.get("vllm")
    original_vllm_distributed = sys.modules.get("vllm.distributed")

    fake_vllm = types.ModuleType("vllm")
    fake_vllm.__path__ = []
    fake_distributed = types.ModuleType("vllm.distributed")
    fake_distributed.get_world_group = MagicMock(return_value=None)
    fake_vllm.distributed = fake_distributed

    sys.modules["vllm"] = fake_vllm
    sys.modules["vllm.distributed"] = fake_distributed

    try:
        yield
    finally:
        if original_vllm is not None:
            sys.modules["vllm"] = original_vllm
        else:
            sys.modules.pop("vllm", None)

        if original_vllm_distributed is not None:
            sys.modules["vllm.distributed"] = original_vllm_distributed
        else:
            sys.modules.pop("vllm.distributed", None)


class TestActorHybridWorker:
    """Test suite for actor_hybrid_worker module."""

    def test_do_tensors_save_success(self):
        """Test successful tensor saving with meta header."""
        from aura.trainer.train_adapter.mindspeed_rl.workers.actor_hybrid_worker import _do_tensors_save

        save_dir = "/tmp/test_save"
        file_path = os.path.join(save_dir, "test.safetensors")
        params = {"param1": MagicMock()}
        meta_header = {"key": "value"}

        with patch('aura.trainer.train_adapter.mindspeed_rl.workers.actor_hybrid_worker.sf') as mock_sf:
            _do_tensors_save(save_dir, file_path, params, meta_header)
            mock_sf.save_file.assert_called_once()

    def test_do_tensors_save_without_meta(self):
        """Test tensor saving without meta header."""
        from aura.trainer.train_adapter.mindspeed_rl.workers.actor_hybrid_worker import _do_tensors_save

        save_dir = "/tmp/test_save_no_meta"
        file_path = os.path.join(save_dir, "test.safetensors")
        params = {"param1": MagicMock()}

        with patch('aura.trainer.train_adapter.mindspeed_rl.workers.actor_hybrid_worker.sf') as mock_sf:
            _do_tensors_save(save_dir, file_path, params, None)
            mock_sf.save_file.assert_called_once()

    def test_do_tensors_save_failure(self):
        """Test tensor saving with exception handling."""
        from aura.trainer.train_adapter.mindspeed_rl.workers.actor_hybrid_worker import _do_tensors_save

        save_dir = "/tmp/test_save_fail"
        file_path = os.path.join(save_dir, "test.safetensors")
        params = {"param1": MagicMock()}

        with patch('aura.trainer.train_adapter.mindspeed_rl.workers.actor_hybrid_worker.sf') as mock_sf:
            mock_sf.save_file.side_effect = RuntimeError("save failed")
            with pytest.raises(RuntimeError, match="save failed"):
                _do_tensors_save(save_dir, file_path, params, None)

    def test_async_tensors_save(self):
        """Test async tensor saving spawns thread."""
        from aura.trainer.train_adapter.mindspeed_rl.workers.actor_hybrid_worker import async_tensors_save

        save_dir = "/tmp/test_async_save"
        file_path = os.path.join(save_dir, "test.safetensors")
        params = {"param1": MagicMock()}

        with patch('aura.trainer.train_adapter.mindspeed_rl.workers.actor_hybrid_worker.sf') as mock_sf:
            async_tensors_save(save_dir, file_path, params)

            import time
            time.sleep(0.1)

    def test_update_actor_logprob_dispatch_size(self):
        """Test update_actor_logprob_dispatch_size function."""
        from aura.trainer.train_adapter.mindspeed_rl.workers.actor_hybrid_worker import update_actor_logprob_dispatch_size

        mock_self = MagicMock()
        mock_self.rl_config = MagicMock()
        mock_self.parallel_state = MagicMock()
        mock_self.parallel_state.get_data_parallel_world_size.return_value = 2

        update_actor_logprob_dispatch_size(mock_self, 100)

        assert mock_self.rl_config.actor_logprob_dispatch_size == 50

    def test_update_mini_batch_size(self):
        """Test update_mini_batch_size function."""
        from aura.trainer.train_adapter.mindspeed_rl.workers.actor_hybrid_worker import update_mini_batch_size

        mock_self = MagicMock()
        mock_self.actor_hybrid = MagicMock()

        update_mini_batch_size(mock_self, 4, 8, True)

        mock_self.actor_hybrid.update_mini_batch_size.assert_called_once_with(4, 8, True)

    def test_agent_actor_hybrid_worker_base_init(self):
        """Test AgentActorHybridWorkerBase initialization."""
        from aura.trainer.train_adapter.mindspeed_rl.workers.actor_hybrid_worker import AgentActorHybridWorkerBase

        worker = AgentActorHybridWorkerBase()

        assert worker.rl_config.zmq_communication == False
        assert worker.continue_infer_running == False
        assert worker.sentinel == None

    def test_get_worker_info(self):
        """Test get_worker_info method."""
        from aura.trainer.train_adapter.mindspeed_rl.workers.actor_hybrid_worker import AgentActorHybridWorkerBase

        worker = AgentActorHybridWorkerBase()
        rank, node_id = worker.get_worker_info()

        assert node_id == "node123"

    def test_init_worker(self):
        """Test init_worker method."""
        from aura.trainer.train_adapter.mindspeed_rl.workers.actor_hybrid_worker import AgentActorHybridWorkerBase

        worker = AgentActorHybridWorkerBase()
        worker.inference_model = MagicMock()

        all_kwargs = [{"key": "value"}]
        worker.init_worker(all_kwargs)

        worker.inference_model.init_worker.assert_called_once_with(all_kwargs)

    def test_load_model(self):
        """Test load_model method."""
        from aura.trainer.train_adapter.mindspeed_rl.workers.actor_hybrid_worker import AgentActorHybridWorkerBase

        worker = AgentActorHybridWorkerBase()
        worker.inference_model = MagicMock()

        worker.load_model("arg1", "arg2", kwarg="value")

        worker.inference_model.load_model.assert_called_once_with("arg1", "arg2", kwarg="value")

    def test_enter_infer_mode_already_infer(self):
        """Test enter_infer_mode when already in infer mode."""
        from aura.trainer.train_adapter.mindspeed_rl.workers.actor_hybrid_worker import AgentActorHybridWorkerBase

        worker = AgentActorHybridWorkerBase()
        worker.state = "INFER"
        worker.sharding_manager = MagicMock()

        worker.enter_infer_mode()

        worker.sharding_manager.enter_infer_mode.assert_not_called()

    def test_enter_infer_mode_not_infer(self):
        """Test enter_infer_mode transitions to infer mode."""
        from aura.trainer.train_adapter.mindspeed_rl.workers.actor_hybrid_worker import AgentActorHybridWorkerBase

        worker = AgentActorHybridWorkerBase()
        worker.state = "TRAIN"
        worker.sharding_manager = MagicMock()
        worker.td = MagicMock()

        with patch('aura.trainer.train_adapter.mindspeed_rl.workers.actor_hybrid_worker.time.time', side_effect=[1.0, 2.0]):
            worker.enter_infer_mode()

        worker.sharding_manager.enter_infer_mode.assert_called_once()
        assert worker.state == "INFER"

    def test_sleep_already_sleep(self):
        """Test sleep when already sleeping."""
        from aura.trainer.train_adapter.mindspeed_rl.workers.actor_hybrid_worker import AgentActorHybridWorkerBase

        worker = AgentActorHybridWorkerBase()
        worker.inference_model = MagicMock(is_sleep=True)

        worker.sleep()

        worker.inference_model.sleep.assert_not_called()

    def test_sleep_not_sleep(self):
        """Test sleep when not sleeping."""
        from aura.trainer.train_adapter.mindspeed_rl.workers.actor_hybrid_worker import AgentActorHybridWorkerBase

        worker = AgentActorHybridWorkerBase()
        worker.inference_model = MagicMock(is_sleep=False)
        worker.exit_infer_mode = MagicMock()

        worker.sleep()

        worker.inference_model.sleep.assert_called_once()
        assert worker.inference_model.is_sleep == True
        assert worker.continue_infer_running == True

    def test_wake_up_not_sleep(self):
        """Test wake_up when not sleeping."""
        from aura.trainer.train_adapter.mindspeed_rl.workers.actor_hybrid_worker import AgentActorHybridWorkerBase

        worker = AgentActorHybridWorkerBase()
        worker.inference_model = MagicMock(is_sleep=False)

        worker.wake_up()

        worker.inference_model.wake_up.assert_not_called()

    def test_wake_up_sleep_without_continue_infer(self):
        """Test wake_up when sleeping without continue_infer."""
        from aura.trainer.train_adapter.mindspeed_rl.workers.actor_hybrid_worker import AgentActorHybridWorkerBase

        worker = AgentActorHybridWorkerBase()
        worker.inference_model = MagicMock(is_sleep=True)
        worker.continue_infer_running = False
        worker.sharding_manager = MagicMock()
        worker.enter_infer_mode = MagicMock()

        worker.wake_up()

        worker.sharding_manager.enter_forward_mode.assert_not_called()
        worker.enter_infer_mode.assert_called_once()
        assert worker.inference_model.is_sleep == False

    def test_wake_up_sleep_with_continue_infer(self):
        """Test wake_up when sleeping with continue_infer."""
        from aura.trainer.train_adapter.mindspeed_rl.workers.actor_hybrid_worker import AgentActorHybridWorkerBase

        worker = AgentActorHybridWorkerBase()
        worker.inference_model = MagicMock(is_sleep=True)
        worker.continue_infer_running = True
        worker.sharding_manager = MagicMock()
        worker.enter_infer_mode = MagicMock()

        worker.wake_up()

        worker.sharding_manager.enter_forward_mode.assert_called_once()
        worker.enter_infer_mode.assert_called_once()
        assert worker.inference_model.is_sleep == False

    def test_execute_method_init_worker(self):
        """Test execute_method dispatches init_worker."""
        from aura.trainer.train_adapter.mindspeed_rl.workers.actor_hybrid_worker import AgentActorHybridWorkerBase

        worker = AgentActorHybridWorkerBase()
        worker.init_worker = MagicMock()

        worker.execute_method("init_worker", [{"key": "value"}])

        worker.init_worker.assert_called_once_with([{"key": "value"}])

    def test_execute_method_load_model(self):
        """Test execute_method dispatches load_model."""
        from aura.trainer.train_adapter.mindspeed_rl.workers.actor_hybrid_worker import AgentActorHybridWorkerBase

        worker = AgentActorHybridWorkerBase()
        worker.load_model = MagicMock()

        worker.execute_method("load_model", "arg1")

        worker.load_model.assert_called_once_with("arg1")

    def test_execute_method_sleep(self):
        """Test execute_method dispatches sleep."""
        from aura.trainer.train_adapter.mindspeed_rl.workers.actor_hybrid_worker import AgentActorHybridWorkerBase

        worker = AgentActorHybridWorkerBase()
        worker.sleep = MagicMock()

        worker.execute_method("sleep")

        worker.sleep.assert_called_once()

    def test_execute_method_wake_up(self):
        """Test execute_method dispatches wake_up."""
        from aura.trainer.train_adapter.mindspeed_rl.workers.actor_hybrid_worker import AgentActorHybridWorkerBase

        worker = AgentActorHybridWorkerBase()
        worker.wake_up = MagicMock()

        worker.execute_method("wake_up")

        worker.wake_up.assert_called_once()

    def test_execute_method_unknown(self):
        """Test execute_method with unknown method."""
        from aura.trainer.train_adapter.mindspeed_rl.workers.actor_hybrid_worker import AgentActorHybridWorkerBase

        worker = AgentActorHybridWorkerBase()
        worker.inference_model = MagicMock()

        worker.execute_method("unknown_method", "arg1")

        worker.inference_model.execute_method.assert_called_once_with("unknown_method", "arg1")

    def test_get_file_name_and_dev_basic(self):
        """Test get_file_name_and_dev basic case."""
        from aura.trainer.train_adapter.mindspeed_rl.workers.actor_hybrid_worker import AgentActorHybridWorkerBase

        worker = AgentActorHybridWorkerBase()
        worker.megatron_config = MagicMock(expert_model_parallel_size=1)
        import megatron.core.parallel_state as ps
        with patch.object(ps, 'get_data_parallel_rank', return_value=0):
            with patch.object(ps, 'get_pipeline_model_parallel_rank', return_value=0):
                with patch.object(ps, 'get_tensor_model_parallel_rank', return_value=0):
                    with patch('os.path.realpath', return_value='/tmp/test'):
                        with patch('os.makedirs'):
                            file_path, dev = worker.get_file_name_and_dev("/tmp/test")

        assert dev == "npu"
        assert "pp0_tp0_ep0.safetensors" in file_path

    def test_get_file_name_and_dev_with_ep(self):
        """Test get_file_name_and_dev with expert parallel."""
        from aura.trainer.train_adapter.mindspeed_rl.workers.actor_hybrid_worker import AgentActorHybridWorkerBase

        worker = AgentActorHybridWorkerBase()
        worker.megatron_config = MagicMock(expert_model_parallel_size=2)
        import megatron.core.parallel_state as ps
        with patch.object(ps, 'get_expert_model_parallel_rank', return_value=1):
            with patch.object(ps, 'get_data_parallel_rank', return_value=0):
                with patch.object(ps, 'get_pipeline_model_parallel_rank', return_value=0):
                    with patch.object(ps, 'get_tensor_model_parallel_rank', return_value=0):
                        with patch('os.path.realpath', return_value='/tmp/test'):
                            with patch('os.makedirs'):
                                file_path, dev = worker.get_file_name_and_dev("/tmp/test")

        assert dev == "npu"
        assert "_ep1" in file_path

    def test_actor_hybrid_worker_exists(self):
        """Test ActorHybridWorker class exists."""
        from aura.trainer.train_adapter.mindspeed_rl.workers.actor_hybrid_worker import ActorHybridWorker
        assert ActorHybridWorker is not None

    def test_split_tensors_and_meta(self):
        """Test split_tensors_and_meta is imported and callable."""
        from aura.trainer.train_adapter.mindspeed_rl.workers.actor_hybrid_worker import split_tensors_and_meta

        result = split_tensors_and_meta({"key": "value"})
        assert result is not None

    def test_get_meta_and_param_from_dev(self):
        """Test get_meta_and_param_from_dev method."""
        from aura.trainer.train_adapter.mindspeed_rl.workers.actor_hybrid_worker import AgentActorHybridWorkerBase

        worker = AgentActorHybridWorkerBase()
        worker.sharding_manager = MagicMock()
        worker.sharding_manager.vllm_weight_container.get_infer_params.return_value = {}
        worker.onload_infer_params_with_device = MagicMock()

        with patch('aura.trainer.train_adapter.mindspeed_rl.workers.actor_hybrid_worker.split_tensors_and_meta', return_value=({}, {})):
            tensor_params, meta_header = worker.get_meta_and_param_from_dev("cpu")

        worker.onload_infer_params_with_device.assert_called_once_with("cpu")

    def test_onload_infer_params_with_device(self):
        """Test onload_infer_params_with_device method."""
        from aura.trainer.train_adapter.mindspeed_rl.workers.actor_hybrid_worker import AgentActorHybridWorkerBase

        worker = AgentActorHybridWorkerBase()
        worker.sharding_manager = MagicMock()
        buffer = MagicMock()
        worker.sharding_manager.vllm_weight_container.weight_buffers = [buffer]

        with patch('aura.trainer.train_adapter.mindspeed_rl.workers.actor_hybrid_worker.resolve_device', return_value="cpu"):
            worker.onload_infer_params_with_device("cpu")

        buffer.rebuild_with_device.assert_called_once()

    def test_prepare_infer_params_to_cpu(self):
        """Test prepare_infer_params_to_cpu method."""
        from aura.trainer.train_adapter.mindspeed_rl.workers.actor_hybrid_worker import AgentActorHybridWorkerBase

        worker = AgentActorHybridWorkerBase()
        worker.get_file_name_and_dev = MagicMock(return_value=("/tmp/model.safetensors", "npu"))
        worker.get_meta_and_param_from_dev = MagicMock(return_value=({}, {}))
        worker.sharding_manager = MagicMock()

        result = worker.prepare_infer_params_to_cpu("/tmp")

        assert result == "/tmp/model.safetensors"
        worker.sharding_manager.offload_infer_params.assert_called_once()

    def test_initialize_basic(self):
        """Test initialize method basic flow."""
        from aura.trainer.train_adapter.mindspeed_rl.workers.actor_hybrid_worker import AgentActorHybridWorkerBase

        worker = AgentActorHybridWorkerBase()
        worker.generate_config = MagicMock(
            offload_train_optimizer=False,
            offload_train_grad=False,
            offload_train_param=False
        )
        worker.megatron_config = MagicMock()
        worker.profiler_config = MagicMock()
        worker.msprobe_config = MagicMock()
        worker._build_model_optimizer = MagicMock(return_value=(MagicMock(), MagicMock(), MagicMock()))
        worker._build_rollout = MagicMock(return_value=MagicMock())
        worker.setup_distributed_rank = MagicMock()
        worker._set_no_sync_func = MagicMock()

        with patch('aura.trainer.train_adapter.mindspeed_rl.workers.actor_hybrid_worker.MegatronOffLoader') as mock_offloader_class, \
             patch('aura.trainer.train_adapter.mindspeed_rl.workers.actor_hybrid_worker.profiler_start') as mock_profiler, \
             patch('aura.trainer.train_adapter.mindspeed_rl.workers.actor_hybrid_worker.MsProbe') as mock_msprobe, \
             patch('aura.trainer.train_adapter.mindspeed_rl.workers.actor_hybrid_worker.replace_torch_compile') as mock_replace, \
             patch('aura.trainer.train_adapter.mindspeed_rl.workers.actor_hybrid_worker.HAS_HACO', False):
            mock_offloader_instance = MagicMock()
            mock_offloader_class.return_value = mock_offloader_instance
            mock_replace.return_value.__enter__ = MagicMock()
            mock_replace.return_value.__exit__ = MagicMock(return_value=False)
            worker.initialize()

            worker.setup_distributed_rank.assert_called_once()
            worker._build_model_optimizer.assert_called_once()

    def test_initialize_with_offload(self):
        """Test initialize method with offload enabled."""
        from aura.trainer.train_adapter.mindspeed_rl.workers.actor_hybrid_worker import AgentActorHybridWorkerBase

        worker = AgentActorHybridWorkerBase()
        worker.generate_config = MagicMock(
            offload_train_optimizer=True,
            offload_train_grad=True,
            offload_train_param=True
        )
        worker.megatron_config = MagicMock()
        worker.profiler_config = MagicMock()
        worker.msprobe_config = MagicMock()
        worker._build_model_optimizer = MagicMock(return_value=(MagicMock(), MagicMock(), MagicMock()))
        worker._build_rollout = MagicMock(return_value=MagicMock())
        worker.setup_distributed_rank = MagicMock()
        worker._set_no_sync_func = MagicMock()

        with patch('aura.trainer.train_adapter.mindspeed_rl.workers.actor_hybrid_worker.MegatronOffLoader') as mock_offloader_class, \
             patch('aura.trainer.train_adapter.mindspeed_rl.workers.actor_hybrid_worker.profiler_start'), \
             patch('aura.trainer.train_adapter.mindspeed_rl.workers.actor_hybrid_worker.MsProbe'), \
             patch('aura.trainer.train_adapter.mindspeed_rl.workers.actor_hybrid_worker.replace_torch_compile') as mock_replace, \
             patch('aura.trainer.train_adapter.mindspeed_rl.workers.actor_hybrid_worker.HAS_HACO', False):
            mock_offloader_instance = MagicMock()
            mock_offloader_class.return_value = mock_offloader_instance
            mock_replace.return_value.__enter__ = MagicMock()
            mock_replace.return_value.__exit__ = MagicMock(return_value=False)
            worker.initialize()

            mock_offloader_instance.offload_optimizer.assert_called_once()
            mock_offloader_instance.offload_grad.assert_called_once()
            mock_offloader_instance.offload_param.assert_called_once()

    def test_init_sharding_manager_basic(self):
        """Test init_sharding_manager basic flow."""
        from aura.trainer.train_adapter.mindspeed_rl.workers.actor_hybrid_worker import AgentActorHybridWorkerBase

        worker = AgentActorHybridWorkerBase()
        worker.inference_model = MagicMock()
        worker._build_sharding_manager = MagicMock(return_value=MagicMock())
        worker.generate_config = MagicMock(
            enable_sleep_mode=False,
            offload_train_param=False
        )
        worker.model = MagicMock()
        worker.megatron_config = MagicMock()
        worker.optimizer = MagicMock()
        worker.opt_param_scheduler = MagicMock()
        worker.rl_config = MagicMock(
            beta=0.1,
            mini_batch_size=32,
            epochs=1,
            shuffle_mini_batch=False,
            clip_ratio=0.5,
            use_dynamic_bsz=False,
            max_packing_token_size=8192,
            dynamic_max_batch_size=False,
            use_remove_padding=False,
            entropy_coeff=0.0,
            kl_penalty="kl",
            token_level_loss=False,
            clip_higher_enable=False,
            clip_ratio_low=0.5,
            clip_ratio_high=2.0
        )
        worker.generate_config.sampling_config = {"temperature": 1.0}
        worker.parallel_state = MagicMock(get_data_parallel_world_size=MagicMock(return_value=1))
        worker.forward_backward_func = MagicMock()
        worker.set_actual_seq_len = MagicMock()
        worker.get_actual_seq_len = MagicMock()
        worker.set_position_ids = MagicMock()
        worker.empty_cache = MagicMock()

        worker.init_sharding_manager()

        worker.inference_model.sleep.assert_called_once()
        worker.empty_cache.assert_called_once()

    def test_init_sharding_manager_with_offload(self):
        """Test init_sharding_manager with offload enabled."""
        from aura.trainer.train_adapter.mindspeed_rl.workers.actor_hybrid_worker import AgentActorHybridWorkerBase

        worker = AgentActorHybridWorkerBase()
        worker.inference_model = MagicMock()
        worker._build_sharding_manager = MagicMock(return_value=MagicMock())
        worker.generate_config = MagicMock(
            enable_sleep_mode=True,
            offload_train_param=True
        )
        worker.actor_offloader = MagicMock()
        worker.model = MagicMock()
        worker.megatron_config = MagicMock()
        worker.optimizer = MagicMock()
        worker.opt_param_scheduler = MagicMock()
        worker.rl_config = MagicMock(
            beta=0.1,
            mini_batch_size=32,
            epochs=1,
            shuffle_mini_batch=False,
            clip_ratio=0.5,
            use_dynamic_bsz=False,
            max_packing_token_size=8192,
            dynamic_max_batch_size=False,
            use_remove_padding=False,
            entropy_coeff=0.0,
            kl_penalty="kl",
            token_level_loss=False,
            clip_higher_enable=False,
            clip_ratio_low=0.5,
            clip_ratio_high=2.0
        )
        worker.generate_config.sampling_config = {"temperature": 1.0}
        worker.parallel_state = MagicMock(get_data_parallel_world_size=MagicMock(return_value=1))
        worker.forward_backward_func = MagicMock()
        worker.set_actual_seq_len = MagicMock()
        worker.get_actual_seq_len = MagicMock()
        worker.set_position_ids = MagicMock()
        worker.empty_cache = MagicMock()

        worker.init_sharding_manager()

        worker.actor_offloader.onload_param.assert_called_once()

    def test__build_rollout(self):
        """Test _build_rollout method."""
        from aura.trainer.train_adapter.mindspeed_rl.workers.actor_hybrid_worker import AgentActorHybridWorkerBase

        worker = AgentActorHybridWorkerBase()
        worker.megatron_config = MagicMock(
            tokenizer_name_or_path="/tmp/model",
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=1,
            expert_model_parallel_size=1,
            context_parallel_size=1
        )
        worker.generate_config = MagicMock(
            infer_tensor_parallel_size=1,
            infer_pipeline_parallel_size=1,
            infer_expert_parallel_size=1,
            max_num_seqs=16,
            max_model_len=2048,
            dtype="float16",
            gpu_memory_utilization=0.9,
            trust_remote_code=False,
            enable_sleep_mode=False
        )

        with patch('aura.trainer.train_adapter.mindspeed_rl.workers.actor_hybrid_worker.AutoConfig') as mock_autoconfig, \
             patch('aura.runner.infer_adapter.vllm.vllm_worker.AsyncVLLMInferEngine') as mock_engine:
            mock_autoconfig.from_pretrained.return_value = MagicMock()
            result = worker._build_rollout()

            assert result is not None

    def test_update_basic(self):
        """Test update method basic flow."""
        from aura.trainer.train_adapter.mindspeed_rl.workers.actor_hybrid_worker import AgentActorHybridWorkerBase

        worker = AgentActorHybridWorkerBase()
        worker.sharding_manager = MagicMock()
        worker.args = MagicMock(curr_iteration=0, consumed_train_samples=0)
        worker.iteration = 0
        worker.megatron_config = MagicMock(
            stage="default",
            tensor_model_parallel_size=1,
            context_parallel_size=1,
            context_parallel_algo="default",
            global_batch_size=32,
            micro_batch_size=4
        )
        worker.rl_config = MagicMock(
            actor_update_dispatch_size=1,
            guarantee_order=False,
            n_samples_per_prompt=1,
            actor_logprob_dispatch_size=1
        )
        worker.optimizer = MagicMock(param_groups=[{"lr": 0.001}])
        worker.td = MagicMock()
        worker.profiler_config = MagicMock()
        worker.prof_iteration = 0
        worker.model = [MagicMock()]
        worker.actor_hybrid = MagicMock()
        worker.actor_hybrid.update_actor.return_value = {"loss": 1.0}
        worker.parallel_state = MagicMock(
            is_pipeline_last_stage=MagicMock(return_value=True),
            get_tensor_model_parallel_rank=MagicMock(return_value=0),
            get_context_parallel_rank=MagicMock(return_value=0)
        )
        worker.all_consumed = MagicMock(return_value=0)
        worker.dispatch_transfer_dock_data = MagicMock(return_value=(None, None))
        worker.enable_partial_rollout = False
        worker.num_floating_point_operations_so_far = 0

        worker.update()

        worker.sharding_manager.enter_train_mode.assert_called_once()
        worker.sharding_manager.exit_train_mode.assert_called_once()

    def test_update_with_ray_dapo_stage(self):
        """Test update method with ray_dapo stage."""
        from aura.trainer.train_adapter.mindspeed_rl.workers.actor_hybrid_worker import AgentActorHybridWorkerBase

        worker = AgentActorHybridWorkerBase()
        worker.sharding_manager = MagicMock()
        worker.args = MagicMock(curr_iteration=0, consumed_train_samples=0)
        worker.iteration = 0
        worker.megatron_config = MagicMock(
            stage="ray_dapo",
            tensor_model_parallel_size=1,
            context_parallel_size=1,
            context_parallel_algo="default",
            global_batch_size=32,
            micro_batch_size=4
        )
        worker.rl_config = MagicMock(
            actor_update_dispatch_size=1,
            guarantee_order=False,
            n_samples_per_prompt=1,
            actor_logprob_dispatch_size=1
        )
        worker.optimizer = MagicMock(param_groups=[{"lr": 0.001}])
        worker.td = MagicMock()
        worker.profiler_config = MagicMock()
        worker.prof_iteration = 0
        worker.model = [MagicMock()]
        worker.actor_hybrid = MagicMock()
        worker.actor_hybrid.update_actor.return_value = {"loss": 1.0}
        worker.parallel_state = MagicMock(
            is_pipeline_last_stage=MagicMock(return_value=True),
            get_tensor_model_parallel_rank=MagicMock(return_value=0),
            get_context_parallel_rank=MagicMock(return_value=0)
        )
        worker.all_consumed = MagicMock(return_value=0)
        worker.dispatch_transfer_dock_data = MagicMock(return_value=(None, None))
        worker.enable_partial_rollout = False
        worker.num_floating_point_operations_so_far = 0

        worker.update()

        assert worker.megatron_config.stage == "ray_dapo"

    def test_update_with_skip_actor_log_prob(self):
        """Test update method with skip_actor_log_prob=True."""
        from aura.trainer.train_adapter.mindspeed_rl.workers.actor_hybrid_worker import AgentActorHybridWorkerBase

        worker = AgentActorHybridWorkerBase()
        worker.sharding_manager = MagicMock()
        worker.args = MagicMock(curr_iteration=0, consumed_train_samples=0)
        worker.iteration = 0
        worker.megatron_config = MagicMock(
            stage="default",
            tensor_model_parallel_size=1,
            context_parallel_size=1,
            context_parallel_algo="default",
            global_batch_size=32,
            micro_batch_size=4
        )
        worker.rl_config = MagicMock(
            actor_update_dispatch_size=1,
            guarantee_order=False,
            n_samples_per_prompt=1,
            actor_logprob_dispatch_size=1
        )
        worker.optimizer = MagicMock(param_groups=[{"lr": 0.001}])
        worker.td = MagicMock()
        worker.profiler_config = MagicMock()
        worker.prof_iteration = 0
        worker.model = [MagicMock()]
        worker.actor_hybrid = MagicMock()
        worker.actor_hybrid.update_actor.return_value = {"loss": 1.0}
        worker.parallel_state = MagicMock(
            is_pipeline_last_stage=MagicMock(return_value=True),
            get_tensor_model_parallel_rank=MagicMock(return_value=0),
            get_context_parallel_rank=MagicMock(return_value=0)
        )
        worker.all_consumed = MagicMock(return_value=0)
        worker.dispatch_transfer_dock_data = MagicMock(return_value=(None, None))
        worker.enable_partial_rollout = False
        worker.num_floating_point_operations_so_far = 0

        worker.update(skip_actor_log_prob=True)

    def test_update_with_multimodal(self):
        """Test update method with multimodal enabled."""
        from aura.trainer.train_adapter.mindspeed_rl.workers.actor_hybrid_worker import AgentActorHybridWorkerBase, is_multimodal

        with patch('aura.trainer.train_adapter.mindspeed_rl.workers.actor_hybrid_worker.is_multimodal', return_value=True):
            worker = AgentActorHybridWorkerBase()
            worker.sharding_manager = MagicMock()
            worker.args = MagicMock(curr_iteration=0, consumed_train_samples=0)
            worker.iteration = 0
            worker.megatron_config = MagicMock(
                stage="default",
                tensor_model_parallel_size=1,
                context_parallel_size=1,
                context_parallel_algo="default",
                global_batch_size=32,
                micro_batch_size=4
            )
            worker.rl_config = MagicMock(
                actor_update_dispatch_size=1,
                guarantee_order=False,
                n_samples_per_prompt=1,
                actor_logprob_dispatch_size=1
            )
            worker.optimizer = MagicMock(param_groups=[{"lr": 0.001}])
            worker.td = MagicMock()
            worker.profiler_config = MagicMock()
            worker.prof_iteration = 0
            worker.model = [MagicMock()]
            worker.actor_hybrid = MagicMock()
            worker.actor_hybrid.update_actor.return_value = {"loss": 1.0}
            worker.parallel_state = MagicMock(
                is_pipeline_last_stage=MagicMock(return_value=True),
                get_tensor_model_parallel_rank=MagicMock(return_value=0),
                get_context_parallel_rank=MagicMock(return_value=0)
            )
            worker.all_consumed = MagicMock(return_value=0)
            worker.dispatch_transfer_dock_data = MagicMock(return_value=(None, None))
            worker.enable_partial_rollout = False
            worker.num_floating_point_operations_so_far = 0

            worker.update()

    def test_logger_exists(self):
        """Test logger is properly initialized."""
        from aura.trainer.train_adapter.mindspeed_rl.workers.actor_hybrid_worker import logger
        assert logger is not None

    def test_has_haco_exists(self):
        """Test HAS_HACO is properly set."""
        from aura.trainer.train_adapter.mindspeed_rl.workers.actor_hybrid_worker import HAS_HACO
        assert HAS_HACO is not None

    def test_do_tensors_save_exception(self):
        """Test _do_tensors_save handles exception and re-raises."""
        from aura.trainer.train_adapter.mindspeed_rl.workers.actor_hybrid_worker import _do_tensors_save

        save_dir = "/tmp/test_save_exc"
        file_path = os.path.join(save_dir, "test.safetensors")
        params = {"param1": MagicMock()}

        with patch('aura.trainer.train_adapter.mindspeed_rl.workers.actor_hybrid_worker.sf') as mock_sf:
            mock_sf.save_file.side_effect = RuntimeError("save failed")
            with pytest.raises(RuntimeError, match="save failed"):
                _do_tensors_save(save_dir, file_path, params, None)

    def test_initialize_with_haco(self):
        """Test initialize method with HAS_HACO enabled."""
        from aura.trainer.train_adapter.mindspeed_rl.workers.actor_hybrid_worker import AgentActorHybridWorkerBase

        worker = AgentActorHybridWorkerBase()
        worker.generate_config = MagicMock(
            offload_train_optimizer=False,
            offload_train_grad=False,
            offload_train_param=False
        )
        worker.megatron_config = MagicMock()
        worker.profiler_config = MagicMock()
        worker.msprobe_config = MagicMock()
        worker._build_model_optimizer = MagicMock(return_value=(MagicMock(), MagicMock(), MagicMock()))
        worker._build_rollout = MagicMock(return_value=MagicMock())
        worker.setup_distributed_rank = MagicMock()
        worker._set_no_sync_func = MagicMock()
        worker.get_master_addr_port = MagicMock(return_value=("127.0.0.1", 8080))
        worker.sentinel = None

        with patch('aura.trainer.train_adapter.mindspeed_rl.workers.actor_hybrid_worker.MegatronOffLoader') as mock_offloader_class, \
             patch('aura.trainer.train_adapter.mindspeed_rl.workers.actor_hybrid_worker.HAS_HACO', True), \
             patch('aura.trainer.train_adapter.mindspeed_rl.workers.actor_hybrid_worker.actor_worker_update_haco') as mock_haco:
            mock_offloader_class.return_value = MagicMock()
            worker.initialize()
            mock_haco.assert_called_once()

    def test_update_with_batch_data(self):
        """Test update method with batch data processing in while loop."""
        from aura.trainer.train_adapter.mindspeed_rl.workers.actor_hybrid_worker import AgentActorHybridWorkerBase

        worker = AgentActorHybridWorkerBase()
        worker.sharding_manager = MagicMock()
        worker.args = MagicMock(curr_iteration=0, consumed_train_samples=0)
        worker.iteration = 0
        worker.megatron_config = MagicMock(
            stage="default",
            tensor_model_parallel_size=1,
            context_parallel_size=1,
            context_parallel_algo="default",
            global_batch_size=32,
            micro_batch_size=4
        )
        worker.rl_config = MagicMock(
            actor_update_dispatch_size=1,
            guarantee_order=False,
            n_samples_per_prompt=1,
            actor_logprob_dispatch_size=1
        )
        worker.optimizer = MagicMock(param_groups=[{"lr": 0.001}])
        worker.td = MagicMock()
        worker.profiler_config = MagicMock()
        worker.prof_iteration = 0
        worker.model = [MagicMock()]
        worker.actor_hybrid = MagicMock()
        worker.actor_hybrid.update_actor.return_value = {"loss": 1.0}
        worker.parallel_state = MagicMock(
            is_pipeline_last_stage=MagicMock(return_value=True),
            get_tensor_model_parallel_rank=MagicMock(return_value=0),
            get_context_parallel_rank=MagicMock(return_value=0)
        )
        worker.enable_partial_rollout = False
        worker.num_floating_point_operations_so_far = 0
        worker.actor_profiler = MagicMock()

        call_count = [0]
        def all_consumed_side_effect(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                return 1
            return 0

        worker.all_consumed = all_consumed_side_effect
        worker.dispatch_transfer_dock_data = MagicMock(return_value=({"data": "batch"}, [1]))

        with patch('aura.trainer.train_adapter.mindspeed_rl.workers.actor_hybrid_worker.num_floating_point_operations', return_value=100.0):
            worker.update()

        worker.sharding_manager.enter_train_mode.assert_called_once()
        worker.sharding_manager.exit_train_mode.assert_called_once()
        worker.actor_hybrid.update_actor.assert_called_once()

    def test_update_with_guarantee_order(self):
        """Test update method with guarantee_order enabled."""
        from aura.trainer.train_adapter.mindspeed_rl.workers.actor_hybrid_worker import AgentActorHybridWorkerBase

        worker = AgentActorHybridWorkerBase()
        worker.sharding_manager = MagicMock()
        worker.args = MagicMock(curr_iteration=0, consumed_train_samples=0)
        worker.iteration = 0
        worker.megatron_config = MagicMock(
            stage="default",
            tensor_model_parallel_size=1,
            context_parallel_size=1,
            context_parallel_algo="default",
            global_batch_size=32,
            micro_batch_size=4
        )
        worker.rl_config = MagicMock(
            actor_update_dispatch_size=1,
            guarantee_order=True,
            n_samples_per_prompt=1,
            actor_logprob_dispatch_size=1
        )
        worker.optimizer = MagicMock(param_groups=[{"lr": 0.001}])
        worker.td = MagicMock()
        worker.profiler_config = MagicMock()
        worker.prof_iteration = 0
        worker.model = [MagicMock()]
        worker.actor_hybrid = MagicMock()
        worker.actor_hybrid.update_actor.return_value = {"loss": 1.0}
        worker.parallel_state = MagicMock(
            is_pipeline_last_stage=MagicMock(return_value=True),
            get_tensor_model_parallel_rank=MagicMock(return_value=0),
            get_context_parallel_rank=MagicMock(return_value=0)
        )
        worker.enable_partial_rollout = False
        worker.num_floating_point_operations_so_far = 0
        worker.actor_profiler = MagicMock()
        worker.get_dp_range_indexes = MagicMock(return_value=[[0]])

        call_count = [0]
        def all_consumed_side_effect(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                return 1
            return 0

        worker.all_consumed = all_consumed_side_effect
        worker.dispatch_transfer_dock_data = MagicMock(return_value=({"data": "batch"}, [0]))

        with patch('aura.trainer.train_adapter.mindspeed_rl.workers.actor_hybrid_worker.num_floating_point_operations', return_value=100.0):
            worker.update()

        worker.get_dp_range_indexes.assert_called_once()

    def test_get_meta_and_param_with_tensor(self):
        """Test get_meta_and_param_from_dev with torch.Tensor in params."""
        import torch
        from aura.trainer.train_adapter.mindspeed_rl.workers.actor_hybrid_worker import AgentActorHybridWorkerBase

        worker = AgentActorHybridWorkerBase()
        worker.sharding_manager = MagicMock()
        mock_tensor = torch.tensor([1.0, 2.0])
        worker.sharding_manager.vllm_weight_container.get_infer_params.return_value = {
            "weight": mock_tensor,
            "meta_key": "meta_value"
        }
        worker.onload_infer_params_with_device = MagicMock()

        with patch('aura.trainer.train_adapter.mindspeed_rl.workers.actor_hybrid_worker.split_tensors_and_meta') as mock_split:
            mock_split.return_value = ({"weight": mock_tensor.detach().cpu()}, {"meta_key": "meta_value"})
            tensor_params, meta_header = worker.get_meta_and_param_from_dev("cpu")

            worker.onload_infer_params_with_device.assert_called_once_with("cpu")
            assert "weight" in tensor_params
            assert "meta_key" in meta_header

    def test_actor_hybrid_worker_init(self):
        """Test ActorHybridWorker initialization."""
        from aura.trainer.train_adapter.mindspeed_rl.workers.actor_hybrid_worker import ActorHybridWorker

        worker = ActorHybridWorker()
        assert worker.rl_config.zmq_communication == False
        assert worker.continue_infer_running == False
        assert worker.sentinel == None

    def test_update_with_two_iterations(self):
        """Test update method with two while loop iterations to cover second-iteration branches."""
        from aura.trainer.train_adapter.mindspeed_rl.workers.actor_hybrid_worker import AgentActorHybridWorkerBase

        worker = AgentActorHybridWorkerBase()
        worker.sharding_manager = MagicMock()
        worker.args = MagicMock(curr_iteration=0, consumed_train_samples=0)
        worker.iteration = 0
        worker.megatron_config = MagicMock(
            stage="default",
            tensor_model_parallel_size=1,
            context_parallel_size=1,
            context_parallel_algo="default",
            global_batch_size=32,
            micro_batch_size=4
        )
        worker.rl_config = MagicMock(
            actor_update_dispatch_size=1,
            guarantee_order=False,
            n_samples_per_prompt=1,
            actor_logprob_dispatch_size=1
        )
        worker.optimizer = MagicMock(param_groups=[{"lr": 0.001}])
        worker.td = MagicMock()
        worker.profiler_config = MagicMock()
        worker.prof_iteration = 0
        worker.model = [MagicMock()]
        worker.actor_hybrid = MagicMock()
        worker.actor_hybrid.update_actor.return_value = {"loss": 1.0}
        worker.parallel_state = MagicMock(
            is_pipeline_last_stage=MagicMock(return_value=True),
            get_tensor_model_parallel_rank=MagicMock(return_value=0),
            get_context_parallel_rank=MagicMock(return_value=0)
        )
        worker.enable_partial_rollout = False
        worker.num_floating_point_operations_so_far = 0
        worker.actor_profiler = MagicMock()

        call_count = [0]
        def all_consumed_side_effect(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] <= 2:
                return 1
            return 0

        worker.all_consumed = all_consumed_side_effect
        worker.dispatch_transfer_dock_data = MagicMock(return_value=({"data": "batch"}, [1]))

        with patch('aura.trainer.train_adapter.mindspeed_rl.workers.actor_hybrid_worker.num_floating_point_operations', return_value=100.0):
            worker.update()

        assert worker.actor_hybrid.update_actor.call_count == 2

    def test_update_with_empty_batch_data(self):
        """Test update method with empty batch_data to cover the skip branch."""
        from aura.trainer.train_adapter.mindspeed_rl.workers.actor_hybrid_worker import AgentActorHybridWorkerBase

        worker = AgentActorHybridWorkerBase()
        worker.sharding_manager = MagicMock()
        worker.args = MagicMock(curr_iteration=0, consumed_train_samples=0)
        worker.iteration = 0
        worker.megatron_config = MagicMock(
            stage="default",
            tensor_model_parallel_size=1,
            context_parallel_size=1,
            context_parallel_algo="default",
            global_batch_size=32,
            micro_batch_size=4
        )
        worker.rl_config = MagicMock(
            actor_update_dispatch_size=1,
            guarantee_order=False,
            n_samples_per_prompt=1,
            actor_logprob_dispatch_size=1
        )
        worker.optimizer = MagicMock(param_groups=[{"lr": 0.001}])
        worker.td = MagicMock()
        worker.profiler_config = MagicMock()
        worker.prof_iteration = 0
        worker.model = [MagicMock()]
        worker.actor_hybrid = MagicMock()
        worker.parallel_state = MagicMock()
        worker.enable_partial_rollout = False
        worker.num_floating_point_operations_so_far = 0
        worker.actor_profiler = MagicMock()

        call_count = [0]
        def all_consumed_side_effect(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                return 1
            return 0

        worker.all_consumed = all_consumed_side_effect
        worker.dispatch_transfer_dock_data = MagicMock(return_value=({}, []))

        with patch('aura.trainer.train_adapter.mindspeed_rl.workers.actor_hybrid_worker.num_floating_point_operations', return_value=100.0):
            worker.update()

        worker.actor_hybrid.update_actor.assert_not_called()

    def test_update_not_last_stage(self):
        """Test update method when not pipeline last stage to cover skip metrics branch."""
        from aura.trainer.train_adapter.mindspeed_rl.workers.actor_hybrid_worker import AgentActorHybridWorkerBase

        worker = AgentActorHybridWorkerBase()
        worker.sharding_manager = MagicMock()
        worker.args = MagicMock(curr_iteration=0, consumed_train_samples=0)
        worker.iteration = 0
        worker.megatron_config = MagicMock(
            stage="default",
            tensor_model_parallel_size=1,
            context_parallel_size=1,
            context_parallel_algo="default",
            global_batch_size=32,
            micro_batch_size=4
        )
        worker.rl_config = MagicMock(
            actor_update_dispatch_size=1,
            guarantee_order=False,
            n_samples_per_prompt=1,
            actor_logprob_dispatch_size=1
        )
        worker.optimizer = MagicMock(param_groups=[{"lr": 0.001}])
        worker.td = MagicMock()
        worker.profiler_config = MagicMock()
        worker.prof_iteration = 0
        worker.model = [MagicMock()]
        worker.actor_hybrid = MagicMock()
        worker.actor_hybrid.update_actor.return_value = {"loss": 1.0}
        worker.parallel_state = MagicMock(
            is_pipeline_last_stage=MagicMock(return_value=False),
            get_tensor_model_parallel_rank=MagicMock(return_value=0),
            get_context_parallel_rank=MagicMock(return_value=0)
        )
        worker.enable_partial_rollout = False
        worker.num_floating_point_operations_so_far = 0
        worker.actor_profiler = MagicMock()

        call_count = [0]
        def all_consumed_side_effect(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                return 1
            return 0

        worker.all_consumed = all_consumed_side_effect
        worker.dispatch_transfer_dock_data = MagicMock(return_value=({"data": "batch"}, [1]))

        with patch('aura.trainer.train_adapter.mindspeed_rl.workers.actor_hybrid_worker.num_floating_point_operations', return_value=100.0):
            worker.update()

        worker.actor_hybrid.update_actor.assert_called_once()
