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

import sys
import types
import importlib
import importlib.util
from unittest.mock import MagicMock, patch
import pytest


_POLLUTABLE_SYS_MODULE_KEYS = {
    'sentence_transformers', 'vllm', 'vllm.logger', 'vllm.model_executor',
    'vllm.model_executor.weight_utils', 'vllm.distributed', 'vllm.distributed.get_world_group',
    'vllm.distributed.parallel_state', 'vllm.transformers_utils', 'vllm.attention',
    'vllm.attention.backends', 'vllm.attention.backends.registry', 'acl', 'acl.rt', 'acl.rt.memcpy',
    'mindspeed_rl', 'mindspeed_rl.trainer', 'mindspeed_rl.trainer.utils',
    'mindspeed_rl.trainer.utils.transfer_dock', 'mindspeed_rl.trainer.utils.parallel_state',
    'mindspeed_rl.trainer.grpo_trainer_hybrid', 'mindspeed_rl.utils', 'mindspeed_rl.utils.pad_process',
    'mindspeed_rl.utils.seqlen_balancing', 'mindspeed_rl.utils.utils', 'mindspeed_rl.utils.compute',
    'mindspeed_rl.utils.context_parallel', 'mindspeed_rl.utils.remove_padding',
    'mindspeed_rl.utils.tokenizer', 'mindspeed_rl.utils.loggers', 'mindspeed_rl.datasets',
    'mindspeed_rl.datasets.base_dataset', 'mindspeed_rl.datasets.prompt_dataset',
    'mindspeed_rl.datasets.indexed_dataset', 'mindspeed_rl.datasets.build_dataset',
    'mindspeed_rl.datasets.dataloader', 'mindspeed_rl.workers', 'mindspeed_rl.workers.scheduler',
    'mindspeed_rl.workers.scheduler.launcher', 'mindspeed_rl.workers.actor_worker',
    'mindspeed_rl.workers.actor_hybrid_worker', 'mindspeed_rl.workers.resharding',
    'mindspeed_rl.workers.resharding.memory_buffer', 'mindspeed_rl.workers.resharding.vllm_weight_container',
    'mindspeed_rl.workers.resharding.megatron_sharding_manager',
    'mindspeed_rl.workers.resharding.megatron_off_loader', 'mindspeed_rl.workers.resharding.utils',
    'mindspeed_rl.workers.rule_reward', 'mindspeed_rl.workers.reward_woker',
    'mindspeed_rl.workers.reference_woker', 'mindspeed_rl.config_cls',
    'mindspeed_rl.config_cls.base_config', 'mindspeed_rl.config_cls.megatron_config',
    'mindspeed_rl.config_cls.rl_config', 'mindspeed_rl.config_cls.generate_config',
    'mindspeed_rl.config_cls.validate_config', 'mindspeed_rl.config_cls.mindstudio_config',
    'mindspeed_rl.train', 'mindspeed_rl.train.distributed_train', 'mindspeed_rl.models',
    'mindspeed_rl.models.loss', 'mindspeed_rl.models.loss.grpo_actor_loss_func',
    'mindspeed_rl.models.loss.logprob_computer', 'mindspeed_rl.models.base',
    'mindspeed_rl.models.base.base_training_engine', 'mindspeed_rl.models.actor_rollout_hybrid',
    'mindspeed_rl.models.reference', 'third_party', 'third_party.rl', 'ray.util',
    'ray.util.scheduling_strategies', 'ray.util.scheduling_strategies.PlacementGroupSchedulingStrategy',
    'transformers', 'datasets', 'safetensors', 'safetensors.torch', 'safetensors.torch.safe_open',
    'aura.base', 'aura.base.log', 'aura.base.log.loggers', 'aura.base.accuracy',
    'aura.base.accuracy.haco_tool', 'aura.base.analysis', 'aura.base.analysis.data_analysis',
    'aura.base.utils', 'aura.base.utils.utils', 'aura.base.utils.http_server',
    'aura.base.utils.globals', 'aura.base.utils.work_mode', 'aura.base.misc', 'aura.base.misc.misc',
    'aura.runner', 'aura.runner.infer_adapter', 'aura.runner.infer_adapter.vllm',
    'aura.runner.infer_adapter.vllm.extension',
    'aura.runner.infer_adapter.vllm.extension.custom_worker_extensions',
    'aura.runner.infer_adapter.vllm.vllm_worker', 'aura.runner.agent_router',
}



class MockModule:
    def __init__(self):
        self._attrs = {}
        self.__path__ = []  # Make it look like a package for submodule imports

    def __getattr__(self, name):
        if name in self._attrs:
            return self._attrs[name]
        # Create a new MockModule for submodules
        submodule = MockModule()
        self._attrs[name] = submodule
        # Register in sys.modules for proper import handling
        if hasattr(self, '__name__'):
            full_name = f"{self.__name__}.{name}"
            sys.modules.setdefault(full_name, submodule)
            submodule.__name__ = full_name
        return submodule

    def __call__(self, *args, **kwargs):
        return MagicMock()

    def __iter__(self):
        return iter([])

    def __getitem__(self, key):
        return MagicMock()

    def __len__(self):
        return 0


# Mock sentence_transformers at module level to avoid sklearn/pyarrow import issues on Windows
# Only mock sentence_transformers, keep sklearn and pyarrow as-is if installed
_mock_sentence_transformers = MockModule()
_mock_sentence_transformers._attrs['SentenceTransformer'] = MagicMock()
_mock_sentence_transformers._attrs['util'] = MockModule()
_mock_sentence_transformers.__name__ = "sentence_transformers"

sys.modules.setdefault('sentence_transformers', _mock_sentence_transformers)


class MockBaseConfig:
    def __init__(self, config_dict=None):
        if config_dict:
            for k, v in config_dict.items():
                setattr(self, k, v)

    def update(self, config_dict):
        if config_dict:
            for k, v in config_dict.items():
                setattr(self, k, v)


class MockGenerateConfig(MockBaseConfig):
    pass


class MockRLConfig(MockBaseConfig):
    max_packing_token_size = 8192


class MockMegatronConfig(MockBaseConfig):
    seq_length = 4096
    micro_batch_size = 4

    def __init__(self, training_config=None, model_config=None):
        super().__init__(training_config)
        if model_config:
            for k, v in model_config.items():
                setattr(self, k, v)


class MockRayGRPOTrainer:
    def __init__(self, *args, **kwargs):
        self.validate_n_samples = 1
        self.dataset_additional_keys = []
        self.rollout_worker = MagicMock()
        self.kwargs = kwargs

    def transfer_dock_init(self):
        pass


class MockMetric:
    pass


class MockBaseDataset:
    def __init__(self, dataset, dataset_type):
        self.dataset = dataset
        self.dataset_type = dataset_type

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]


_INSTALLED_MODULES = set()
_MODULES_THAT_MUST_BE_MOCKED = {
    'megatron', 'megatron.core', 'megatron.core.parallel_state',
    'mindspeed_rl', 'mindspeed_rl.trainer', 'mindspeed_rl.trainer.utils',
    'mindspeed_rl.trainer.utils.transfer_dock', 'mindspeed_rl.trainer.utils.parallel_state',
    'mindspeed_rl.trainer.grpo_trainer_hybrid',
    'mindspeed_rl.utils', 'mindspeed_rl.utils.pad_process',
    'mindspeed_rl.utils.seqlen_balancing', 'mindspeed_rl.utils.utils',
    'mindspeed_rl.utils.compute', 'mindspeed_rl.utils.context_parallel',
    'mindspeed_rl.utils.remove_padding', 'mindspeed_rl.utils.tokenizer',
    'mindspeed_rl.utils.loggers',
    'mindspeed.core.parallel_state', 'mindspeed.core.context_parallel.utils',
    'mindspeed_llm.core.transformer.dot_product_attention',
    'mindspeed_rl.datasets', 'mindspeed_rl.datasets.base_dataset',
    'mindspeed_rl.datasets.prompt_dataset', 'mindspeed_rl.datasets.indexed_dataset',
    'mindspeed_rl.datasets.build_dataset', 'mindspeed_rl.datasets.dataloader',
    'mindspeed_rl.workers', 'mindspeed_rl.workers.scheduler',
    'mindspeed_rl.workers.scheduler.launcher', 'mindspeed_rl.workers.actor_worker',
    'mindspeed_rl.workers.actor_hybrid_worker',
    'mindspeed_rl.workers.resharding', 'mindspeed_rl.workers.resharding.memory_buffer',
    'mindspeed_rl.workers.resharding.vllm_weight_container',
    'mindspeed_rl.workers.resharding.megatron_sharding_manager',
    'mindspeed_rl.workers.resharding.megatron_off_loader',
    'mindspeed_rl.workers.resharding.utils',
    'mindspeed_rl.workers.rule_reward', 'mindspeed_rl.workers.reward_woker',
    'mindspeed_rl.workers.reference_woker',
    'mindspeed_rl.config_cls', 'mindspeed_rl.config_cls.base_config',
    'mindspeed_rl.config_cls.megatron_config', 'mindspeed_rl.config_cls.rl_config',
    'mindspeed_rl.config_cls.generate_config', 'mindspeed_rl.config_cls.validate_config',
    'mindspeed_rl.config_cls.mindstudio_config',
    'mindspeed_rl.train', 'mindspeed_rl.train.distributed_train',
    'mindspeed_rl.models', 'mindspeed_rl.models.loss',
    'mindspeed_rl.models.loss.grpo_actor_loss_func',
    'mindspeed_rl.models.loss.logprob_computer',
    'mindspeed_rl.models.base', 'mindspeed_rl.models.base.base_training_engine',
    'mindspeed_rl.models.actor_rollout_hybrid', 'mindspeed_rl.models.reference',
    'third_party', 'third_party.rl',
    'vllm', 'vllm.logger', 'vllm.model_executor',
    'vllm.model_executor.weight_utils', 'vllm.distributed',
    'vllm.transformers_utils', 'vllm.attention', 'vllm.attention.backends',
    'vllm.attention.backends.registry',
    'acl', 'acl.rt',
    'verl',
}

_MODULES_NEVER_MOCK = {
    'ray', 'ray.util', 'ray.util.scheduling_strategies',
    'transformers', 'datasets', 'safetensors', 'safetensors.torch',
    'aura', 'aura.base', 'aura.base.log', 'aura.base.log.loggers',
    'aura.base.accuracy', 'aura.base.accuracy.haco_tool',
    'aura.base.analysis', 'aura.base.analysis.data_analysis',
    'aura.base.utils', 'aura.base.utils.utils',
    'aura.base.utils.http_server', 'aura.base.utils.globals', 'aura.base.utils.work_mode',
    'aura.base.misc', 'aura.base.misc.misc',
    'aura.runner', 'aura.runner.infer_adapter',
    'aura.runner.infer_adapter.vllm',
    'aura.runner.infer_adapter.vllm.extension',
    'aura.runner.infer_adapter.vllm.extension.custom_worker_extensions',
    'aura.runner.infer_adapter.vllm.vllm_worker',
    'aura.runner.agent_router',
}


def _is_module_installed(module_name):
    if module_name in sys.modules:
        return True
    try:
        spec = importlib.util.find_spec(module_name)
        return spec is not None
    except (ModuleNotFoundError, ValueError, ImportError):
        return False


for _module_name in _MODULES_THAT_MUST_BE_MOCKED:
    if _module_name not in _MODULES_NEVER_MOCK and not _is_module_installed(_module_name):
        mock_mod = MockModule()
        mock_mod.__name__ = _module_name
        sys.modules[_module_name] = mock_mod

for _module_name in _MODULES_NEVER_MOCK:
    if _module_name in _MODULES_THAT_MUST_BE_MOCKED:
        if _module_name not in sys.modules:
            try:
                __import__(_module_name)
            except ImportError:
                mock_mod = MockModule()
                mock_mod.__name__ = _module_name
                sys.modules[_module_name] = mock_mod

_VLLM_DIST_FORCE_MOCK_MODULES = [
    'vllm.distributed',
    'vllm.distributed.get_world_group',
    'vllm.distributed.parallel_state',
]
for _mod_name in _VLLM_DIST_FORCE_MOCK_MODULES:
    if _mod_name not in sys.modules:
        _mock = MockModule()
        _mock.__name__ = _mod_name
        _mock.__path__ = []
        sys.modules[_mod_name] = _mock

_VLLM_FORCE_MOCK_IF_NOT_PACKAGE = [
    'vllm', 'vllm.logger', 'vllm.model_executor',
    'vllm.model_executor.weight_utils', 'vllm.distributed',
    'vllm.transformers_utils', 'vllm.attention',
    'vllm.attention.backends', 'vllm.attention.backends.registry',
]


def _is_valid_package(name):
    mod = sys.modules.get(name)
    if mod is None:
        return False
    return hasattr(mod, '__path__') and isinstance(mod.__path__, list)


if not _is_valid_package('vllm'):
    # Force `vllm` to be a real package-like module so `import vllm.distributed`
    # won't fail with: "'vllm' is not a package".
    fake_vllm = types.ModuleType('vllm')
    fake_vllm.__path__ = []
    fake_vllm.__package__ = 'vllm'
    sys.modules['vllm'] = fake_vllm

    fake_distributed = types.ModuleType('vllm.distributed')
    fake_distributed.__package__ = 'vllm'
    fake_distributed.get_world_group = MagicMock(return_value=None)
    sys.modules['vllm.distributed'] = fake_distributed
    fake_vllm.distributed = fake_distributed

    for _mod_name in _VLLM_FORCE_MOCK_IF_NOT_PACKAGE:
        if _mod_name in ('vllm', 'vllm.distributed'):
            continue
        _mock = MockModule()
        _mock.__name__ = _mod_name
        _mock.__path__ = []
        sys.modules[_mod_name] = _mock


@pytest.fixture(autouse=True)
def ensure_vllm_package_shape_for_tests():
    """Keep `vllm` package-like during tests.

    Some tests/fixtures may overwrite `sys.modules['vllm']` with plain objects.
    When that happens, imports like `from vllm.distributed import get_world_group`
    fail with: `'vllm' is not a package`.
    """
    vllm_mod = sys.modules.get('vllm')
    if vllm_mod is None or not hasattr(vllm_mod, '__path__'):
        vllm_mod = types.ModuleType('vllm')
        vllm_mod.__path__ = []
        vllm_mod.__package__ = 'vllm'
        sys.modules['vllm'] = vllm_mod

    vllm_dist_mod = sys.modules.get('vllm.distributed')
    if vllm_dist_mod is None or not isinstance(vllm_dist_mod, types.ModuleType):
        vllm_dist_mod = types.ModuleType('vllm.distributed')
        vllm_dist_mod.__package__ = 'vllm'
        sys.modules['vllm.distributed'] = vllm_dist_mod

    if not hasattr(vllm_dist_mod, 'get_world_group'):
        vllm_dist_mod.get_world_group = MagicMock(return_value=None)

    if not hasattr(vllm_mod, 'distributed'):
        vllm_mod.distributed = vllm_dist_mod

    yield


@pytest.fixture(autouse=True)
def ensure_aura_config_cls_package_shape_for_tests():
    """Keep `aura...mindspeed_rl.config_cls` importable as package.

    Some tests may pollute `sys.modules` with non-package mocks at this path,
    causing: `'...config_cls' is not a package` during submodule imports.
    """
    pkg_name = 'aura.trainer.train_adapter.mindspeed_rl.config_cls'
    pkg_mod = sys.modules.get(pkg_name)

    if pkg_mod is not None:
        if not hasattr(pkg_mod, '__path__'):
            # Keep existing mocked attributes, but mark as a package so
            # submodule imports like `...config_cls.extend_generate` work.
            pkg_mod.__path__ = []
        if not getattr(pkg_mod, '__package__', None):
            pkg_mod.__package__ = pkg_name

    yield

@pytest.fixture(autouse=True)
def restore_polluted_sys_modules():
    """Snapshot and restore module-level mocks to avoid cross-test pollution."""
    snapshot = {name: sys.modules.get(name) for name in _POLLUTABLE_SYS_MODULE_KEYS}
    try:
        yield
    finally:
        for name, original in snapshot.items():
            if original is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = original


class MockRemoteObject:
    def __init__(self, fn):
        self._fn = fn
        self.remote = fn
        self.options = lambda **kw: self
    def __call__(self, *a, **kw):
        return self._fn(*a, **kw)
    def __getattr__(self, name):
        if hasattr(self._fn, name):
            return getattr(self._fn, name)
        return MagicMock()

def mock_ray_remote(fn=None, *a, **kw):
    if fn is not None:
        return MockRemoteObject(fn)
    def decorator(obj):
        return MockRemoteObject(obj)
    return decorator

# Use patch to safely mock ray.remote, this will be applied in the session fixture
try:
    import ray as _ray_module
    _original_ray_remote = _ray_module.remote
except ImportError:
    _original_ray_remote = None


@pytest.fixture(autouse=True, scope="session")
def mock_ray_remote_fixture():
    if _original_ray_remote is None:
        yield
        return

    import ray
    original = ray.remote
    ray.remote = mock_ray_remote
    try:
        yield
    finally:
        ray.remote = original


@pytest.fixture(autouse=True, scope="session")
def setup_mock_modules():

    msrl_module = sys.modules['mindspeed_rl']
    msrl_module._attrs['RayGRPOTrainer'] = MockRayGRPOTrainer
    msrl_module._attrs['GenerateConfig'] = MockGenerateConfig
    msrl_module._attrs['RLConfig'] = MockRLConfig
    msrl_module._attrs['MegatronConfig'] = MockMegatronConfig
    msrl_module._attrs['Metric'] = MockMetric

    sys.modules['mindspeed_rl.RayGRPOTrainer'] = MockRayGRPOTrainer
    sys.modules['mindspeed_rl.GenerateConfig'] = MockGenerateConfig
    sys.modules['mindspeed_rl.RLConfig'] = MockRLConfig
    sys.modules['mindspeed_rl.MegatronConfig'] = MockMegatronConfig
    sys.modules['mindspeed_rl.Metric'] = MockMetric

    config_cls_module = sys.modules['mindspeed_rl.config_cls']
    base_config_module = sys.modules['mindspeed_rl.config_cls.base_config']
    megatron_config_module = sys.modules['mindspeed_rl.config_cls.megatron_config']
    rl_config_module = sys.modules['mindspeed_rl.config_cls.rl_config']
    generate_config_module = sys.modules['mindspeed_rl.config_cls.generate_config']
    validate_config_module = sys.modules['mindspeed_rl.config_cls.validate_config']

    base_config_module._attrs['BaseConfig'] = MockBaseConfig
    megatron_config_module._attrs['MegatronConfig'] = MockMegatronConfig
    rl_config_module._attrs['RLConfig'] = MockRLConfig
    generate_config_module._attrs['GenerateConfig'] = MockGenerateConfig
    validate_config_module._attrs['validate_rl_args'] = MagicMock(return_value=None)

    config_cls_module._attrs['base_config'] = base_config_module
    config_cls_module._attrs['megatron_config'] = megatron_config_module
    config_cls_module._attrs['rl_config'] = rl_config_module
    config_cls_module._attrs['generate_config'] = generate_config_module
    config_cls_module._attrs['validate_config'] = validate_config_module
    config_cls_module._attrs['BaseConfig'] = MockBaseConfig
    config_cls_module._attrs['MegatronConfig'] = MockMegatronConfig
    config_cls_module._attrs['RLConfig'] = MockRLConfig
    config_cls_module._attrs['GenerateConfig'] = MockGenerateConfig

    sys.modules['mindspeed_rl.config_cls.base_config.BaseConfig'] = MockBaseConfig
    sys.modules['mindspeed_rl.config_cls.megatron_config.MegatronConfig'] = MockMegatronConfig
    sys.modules['mindspeed_rl.config_cls.rl_config.RLConfig'] = MockRLConfig
    sys.modules['mindspeed_rl.config_cls.generate_config.GenerateConfig'] = MockGenerateConfig
    sys.modules['mindspeed_rl.config_cls.validate_config.validate_rl_args'] = MagicMock(return_value=None)

    sys.modules['mindspeed_rl.workers.scheduler.launcher.RayActorGroup'] = MagicMock()
    sys.modules['mindspeed_rl.datasets.prompt_dataset.PromptDataset'] = MagicMock()
    sys.modules['mindspeed_rl.datasets.dataloader.PromptDataLoader'] = MagicMock()
    sys.modules['mindspeed_rl.datasets.indexed_dataset.get_packed_indexed_dataset'] = MagicMock()
    sys.modules['mindspeed_rl.datasets.build_dataset.build_train_valid_test_datasets'] = MagicMock()
    sys.modules['mindspeed_rl.train.distributed_train.distributed_train'] = MagicMock()
    sys.modules['mindspeed_rl.utils.seqlen_balancing.karmarkar_karp'] = MagicMock(return_value=[[0], [1], [2]])
    sys.modules['mindspeed_rl.utils.utils.generate_mask'] = MagicMock(return_value=MagicMock())
    sys.modules['mindspeed_rl.utils.compute.compute_log_probs'] = MagicMock(return_value=MagicMock())
    sys.modules['mindspeed_rl.utils.compute.vocab_parallel_entropy'] = MagicMock(return_value=MagicMock())
    sys.modules['mindspeed_rl.utils.compute.get_parallel_state'] = MagicMock(return_value=MagicMock())
    sys.modules['mindspeed_rl.utils.context_parallel.get_tensor_allgather_cp_without_pack'] = MagicMock()
    sys.modules['mindspeed_rl.utils.context_parallel.get_tensor_allgather_cp_with_pack'] = MagicMock()
    sys.modules['mindspeed_rl.utils.remove_padding.postprocess_packed_seqs'] = MagicMock()
    sys.modules['ray.util.scheduling_strategies.PlacementGroupSchedulingStrategy'] = MagicMock()
    sys.modules['mindspeed_rl.workers.resharding.megatron_sharding_manager.MegatronShardingManager'] = MagicMock()
    sys.modules['mindspeed_rl.models.loss.grpo_actor_loss_func.GRPOActorLossFunc'] = MagicMock()
    sys.modules['mindspeed_rl.trainer.utils.compute_utils'] = MagicMock()
    sys.modules['mindspeed_rl.models.base.base_training_engine.BaseTrainingEngine'] = MagicMock()
    sys.modules['mindspeed_rl.models.loss.logprob_computer.StandardLogProbComputer'] = MagicMock()
    sys.modules['mindspeed_rl.workers.resharding.memory_buffer.MemoryBuffer'] = MagicMock()
    sys.modules['mindspeed_rl.workers.resharding.memory_buffer.ModelWeightBuffer'] = MagicMock()
    sys.modules['mindspeed_rl.workers.resharding.memory_buffer.build_memory_buffer'] = MagicMock()
    sys.modules['mindspeed_rl.workers.resharding.memory_buffer.calc_padded_numel'] = MagicMock(return_value=100)
    sys.modules['mindspeed_rl.workers.resharding.memory_buffer.get_weight_buffer_meta_from_buffer'] = MagicMock()
    sys.modules['mindspeed_rl.workers.resharding.vllm_weight_container.MegatronStyleVllmWeightContainer'] = MagicMock()
    sys.modules['mindspeed_rl.models.actor_rollout_hybrid.ActorRolloutHybrid'] = MagicMock()
    sys.modules['mindspeed_rl.trainer.grpo_trainer_hybrid.RayGRPOTrainer'] = MockRayGRPOTrainer
    sys.modules['mindspeed_rl.models.loss.grpo_actor_loss_func'] = MockModule()
    sys.modules['mindspeed_rl.models.loss.grpo_actor_loss_func.GRPOActorLossFunc'] = MagicMock()
    sys.modules['mindspeed_rl.workers.resharding.utils.get_tensor_parallel_partition_dim'] = MagicMock(return_value=0)
    sys.modules['mindspeed_rl.workers.resharding.utils._build_infer_param_dict'] = MagicMock(return_value={})
    sys.modules['mindspeed_rl.workers.resharding.utils.is_tensor_parallel_param'] = MagicMock(return_value=False)
    sys.modules['mindspeed_rl.workers.resharding.utils.get_tp_group'] = MagicMock()
    sys.modules['mindspeed_rl.workers.resharding.utils.is_fake_tp_param'] = MagicMock(return_value=False)

    acl_module = MockModule()
    sys.modules['acl'] = acl_module
    sys.modules['acl.rt'] = MockModule()
    sys.modules['acl.rt.memcpy'] = MagicMock()

    sys.modules['vllm.distributed.get_world_group'] = MagicMock()

    if isinstance(sys.modules.get('safetensors'), MockModule):
        sys.modules['safetensors.torch.safe_open'] = MagicMock()

    if isinstance(sys.modules.get('transformers'), MockModule):
        transformers_module = sys.modules['transformers']
        transformers_module._attrs['AutoConfig'] = MagicMock()

    aura_base_log_loggers_module = sys.modules.get('aura.base.log.loggers')
    if aura_base_log_loggers_module and isinstance(aura_base_log_loggers_module, MockModule):
        mock_logger_instance = MagicMock()
        aura_base_log_loggers_module._attrs['Loggers'] = MagicMock(return_value=MagicMock(get_logger=MagicMock(return_value=mock_logger_instance)))

    aura_base_accuracy_haco_module = sys.modules.get('aura.base.accuracy.haco_tool')
    if aura_base_accuracy_haco_module and isinstance(aura_base_accuracy_haco_module, MockModule):
        aura_base_accuracy_haco_module._attrs['enable_haco'] = MagicMock(return_value=False)
        aura_base_accuracy_haco_module._attrs['actor_worker_update_haco'] = MagicMock()

    aura_base_analysis_module = sys.modules.get('aura.base.analysis.data_analysis')
    if aura_base_analysis_module and isinstance(aura_base_analysis_module, MockModule):
        aura_base_analysis_module._attrs['json_save_data'] = MagicMock()

    aura_runner_custom_extensions_module = sys.modules.get('aura.runner.infer_adapter.vllm.extension.custom_worker_extensions')
    if aura_runner_custom_extensions_module and isinstance(aura_runner_custom_extensions_module, MockModule):
        aura_runner_custom_extensions_module._attrs['resolve_device'] = MagicMock(return_value="cpu")
        aura_runner_custom_extensions_module._attrs['split_tensors_and_meta'] = MagicMock(return_value=({}, {}))

    aura_runner_vllm_worker_module = sys.modules.get('aura.runner.infer_adapter.vllm.vllm_worker')
    if aura_runner_vllm_worker_module is None:
        aura_runner_vllm_worker_module = MockModule()
        sys.modules['aura.runner.infer_adapter.vllm.vllm_worker'] = aura_runner_vllm_worker_module
    if isinstance(aura_runner_vllm_worker_module, MockModule):
        aura_runner_vllm_worker_module._attrs['AsyncVLLMInferEngine'] = MagicMock()

    tokenizer_module = sys.modules['mindspeed_rl.utils.tokenizer']
    tokenizer_module._attrs['BaseTokenizer'] = MagicMock()

    loggers_module = sys.modules['mindspeed_rl.utils.loggers']
    mock_logger = MagicMock()
    mock_logger.info = MagicMock()
    mock_logger.error = MagicMock()
    loggers_module._attrs['Loggers'] = MagicMock(return_value=mock_logger)

    base_dataset_module = sys.modules['mindspeed_rl.datasets.base_dataset']
    base_dataset_module._attrs['BaseDataset'] = MockBaseDataset


    class MockActorHybridWorkerBase:
        def __init__(self, *args, **kwargs):
            self.rl_config = MagicMock()
            self.distributed_optimizer = MagicMock()
            self.float16_optimizer_with_float16_params = MagicMock()
            self.state = "TRAIN"

        def __getattr__(self, name):
            return MagicMock()

    class MockActorState:
        INFER = "INFER"
        TRAIN = "TRAIN"

    def mock_is_multimodal():
        return False

    def mock_num_floating_point_operations(x):
        return 0

    actor_hybrid_worker_module = sys.modules['mindspeed_rl.workers.actor_hybrid_worker']
    actor_hybrid_worker_module._attrs['ActorHybridWorkerBase'] = MockActorHybridWorkerBase
    actor_hybrid_worker_module._attrs['ActorState'] = MockActorState
    actor_hybrid_worker_module._attrs['is_multimodal'] = mock_is_multimodal
    actor_hybrid_worker_module._attrs['num_floating_point_operations'] = mock_num_floating_point_operations

    class MockReferenceWorkerBase:
        pass

    class MockRewardWorkerBase:
        pass

    class MockAgentActorHybridWorkerBase:
        pass

    reference_woker_module = sys.modules['mindspeed_rl.workers.reference_woker']
    reference_woker_module._attrs['ReferenceWorkerBase'] = MockReferenceWorkerBase

    reward_woker_module = sys.modules['mindspeed_rl.workers.reward_woker']
    reward_woker_module._attrs['RewardWorkerBase'] = MockRewardWorkerBase

    actor_hybrid_worker_module._attrs['AgentActorHybridWorkerBase'] = MockAgentActorHybridWorkerBase

    utils_module = sys.modules['mindspeed_rl.utils.utils']
    utils_module._attrs['mstx_timer_decorator'] = lambda func: func

    vllm_module = sys.modules.get('vllm')
    if vllm_module and isinstance(vllm_module, MockModule):
        vllm_module._attrs['worker'] = MockModule()
        vllm_module._attrs['worker'].worker_base = MockModule()
        vllm_module._attrs['worker'].worker_base.WorkerWrapperBase = MagicMock()
        vllm_module._attrs['worker'].worker_base.set_current_vllm_config = MagicMock()

    yield


class MockAgentActorHybridWorkerBaseFull:
    zmq_communication = False
    continue_infer_running = False
    sentinel = None

    def __init__(self, *args, **kwargs):
        self.rl_config = MagicMock()
        self.continue_infer_running = False
        self.sentinel = None
        self.inference_model = MagicMock()
        self.sharding_manager = MagicMock()
        self.state = "TRAIN"
        self.td = MagicMock()
        self.megatron_config = MagicMock()
        self.generate_config = MagicMock()
        self.model = MagicMock()
        self.optimizer = MagicMock()
        self.opt_param_scheduler = MagicMock()
        self.actor_offloader = MagicMock()
        self.profiler_config = MagicMock()
        self.msprobe_config = MagicMock()
        self.actor_hybrid = MagicMock()
        self.parallel_state = MagicMock()
        self.forward_backward_func = MagicMock()
        self.args = MagicMock()
        self.iteration = 0
        self.prof_iteration = 0
        self.actor_profiler = MagicMock()
        self.num_floating_point_operations_so_far = 0
        self._build_sharding_manager = MagicMock(return_value=MagicMock())
        self.setup_distributed_rank = MagicMock()
        self._build_model_optimizer = MagicMock(return_value=(MagicMock(), MagicMock(), MagicMock()))
        self._set_no_sync_func = MagicMock()
        self.get_master_addr_port = MagicMock(return_value=("127.0.0.1", 12345))
        self.empty_cache = MagicMock()
        self.all_consumed = MagicMock(return_value=0)
        self.dispatch_transfer_dock_data = MagicMock(return_value=(None, None))
        self.get_dp_range_indexes = MagicMock(return_value=[])
        self.enable_partial_rollout = False
        self.set_actual_seq_len = MagicMock()
        self.get_actual_seq_len = MagicMock()
        self.set_position_ids = MagicMock()
        self.exit_infer_mode = MagicMock()
        self.compute_log_prob = MagicMock()

    def initialize(self):
        self.setup_distributed_rank()
        self.model, self.optimizer, self.opt_param_scheduler = self._build_model_optimizer()
        self._set_no_sync_func()
        from mindspeed_rl.workers.resharding.megatron_off_loader import MegatronOffLoader
        self.actor_offloader = MegatronOffLoader(
            self.model,
            self.optimizer,
            megatron_config=self.megatron_config,
            distributed_optimizer=self.distributed_optimizer if hasattr(self, 'distributed_optimizer') else MagicMock(),
            float16_optimizer_with_float16_params=self.float16_optimizer_with_float16_params if hasattr(self, 'float16_optimizer_with_float16_params') else MagicMock(),
        )

        if hasattr(self.generate_config, 'offload_train_optimizer') and self.generate_config.offload_train_optimizer:
            self.actor_offloader.offload_optimizer()
        if hasattr(self.generate_config, 'offload_train_grad') and self.generate_config.offload_train_grad:
            self.actor_offloader.offload_grad()
        if hasattr(self.generate_config, 'offload_train_param') and self.generate_config.offload_train_param:
            self.actor_offloader.offload_param()
        with patch('mindspeed_rl.utils.utils.replace_torch_compile'):
            self.inference_model = self._build_rollout()
        if self.sentinel is None:
            addr, port = self.get_master_addr_port()
            self.sentinel = MagicMock()

        self.actor_profiler = MagicMock()
        if hasattr(self, 'msprobe_config'):
            pass

    def get_worker_info(self):
        import os
        return os.getenv('RANK'), "node123"

    def init_worker(self, all_kwargs):
        self.inference_model.init_worker(all_kwargs)

    def load_model(self, *args, **kwargs):
        self.inference_model.load_model(*args, **kwargs)

    def enter_infer_mode(self):
        if self.state == "INFER":
            return
        import time
        start_time = time.time()
        self.sharding_manager.enter_infer_mode()
        self.state = "INFER"
        end_time = time.time()
        self.td.update_metrics.remote("timing/resharding_to_infer", value=[end_time - start_time], cumulate=True)

    def init_sharding_manager(self):
        self.inference_model.sleep()
        self.sharding_manager = self._build_sharding_manager()
        if hasattr(self.sharding_manager, 'enable_sleep_mode'):
            self.sharding_manager.enable_sleep_mode = self.generate_config.enable_sleep_mode if hasattr(self.generate_config, 'enable_sleep_mode') else False

        if hasattr(self.generate_config, 'offload_train_param') and self.generate_config.offload_train_param:
            self.actor_offloader.onload_param()

        self.actor_hybrid = MagicMock()
        self.empty_cache()

    def sleep(self, *args, **kwargs):
        if self.inference_model.is_sleep:
            return
        self.inference_model.sleep(*args, **kwargs)
        self.inference_model.is_sleep = True
        self.exit_infer_mode()
        self.continue_infer_running = True

    def wake_up(self, *args, **kwargs):
        if not self.inference_model.is_sleep:
            return
        if self.continue_infer_running:
            self.sharding_manager.enter_forward_mode()
        self.enter_infer_mode()
        self.inference_model.wake_up(*args, **kwargs)
        self.inference_model.is_sleep = False

    def execute_method(self, method, *args, **kwargs):
        dispatch = {
            "init_worker": self.init_worker,
            "load_model": self.load_model,
            "sleep": self.sleep,
            "wake_up": self.wake_up,
        }
        handler = dispatch.get(method)
        if handler is not None:
            return handler(*args, **kwargs)
        return self.inference_model.execute_method(method, *args, **kwargs)

    def _build_rollout(self):
        return MagicMock(
            tokenizer_name_or_path=self.megatron_config.tokenizer_name_or_path if hasattr(self.megatron_config, 'tokenizer_name_or_path') else "",
            train_tensor_parallel_size=self.megatron_config.tensor_model_parallel_size if hasattr(self.megatron_config, 'tensor_model_parallel_size') else 1,
            train_pipeline_parallel_size=self.megatron_config.pipeline_model_parallel_size if hasattr(self.megatron_config, 'pipeline_model_parallel_size') else 1,
            train_expert_parallel_size=self.megatron_config.expert_model_parallel_size if hasattr(self.megatron_config, 'expert_model_parallel_size') else 1,
            train_context_parallel_size=self.megatron_config.context_parallel_size if hasattr(self.megatron_config, 'context_parallel_size') else 1,
            infer_tensor_parallel_size=self.generate_config.infer_tensor_parallel_size if hasattr(self.generate_config, 'infer_tensor_parallel_size') else 1,
            infer_pipeline_parallel_size=self.generate_config.infer_pipeline_parallel_size if hasattr(self.generate_config, 'infer_pipeline_parallel_size') else 1,
            infer_expert_parallel_size=self.generate_config.infer_expert_parallel_size if hasattr(self.generate_config, 'infer_expert_parallel_size') else 1,
            max_num_seqs=self.generate_config.max_num_seqs if hasattr(self.generate_config, 'max_num_seqs') else 16,
            max_model_len=self.generate_config.max_model_len if hasattr(self.generate_config, 'max_model_len') else 2048,
            dtype=self.generate_config.dtype if hasattr(self.generate_config, 'dtype') else "float16",
            gpu_memory_utilization=self.generate_config.gpu_memory_utilization if hasattr(self.generate_config, 'gpu_memory_utilization') else 0.9,
            trust_remote_code=self.generate_config.trust_remote_code if hasattr(self.generate_config, 'trust_remote_code') else False,
            enable_sleep_mode=self.generate_config.enable_sleep_mode if hasattr(self.generate_config, 'enable_sleep_mode') else False,
        )

    def update(self, kl_ctrl=None, skip_actor_log_prob=False):
        import time
        import ray

        start_sharding_enter_train = time.time()
        self.sharding_manager.enter_train_mode()
        sharding_train_interval = time.time() - start_sharding_enter_train

        self.args.curr_iteration = self.iteration

        experience_consumer_stage = 'actor_train'

        if hasattr(self.megatron_config, 'stage') and self.megatron_config.stage == "ray_dapo":
            experience_columns = ['responses', 'advantages', 'old_log_prob', 'input_ids', 'response_length', 'prompt_length']
        else:
            experience_columns = ['responses', 'advantages', 'old_log_prob', 'ref_log_prob', 'input_ids', 'response_length', 'prompt_length']

        experience_columns.append("response_mask")

        is_multimodal = lambda: False
        if is_multimodal():
            experience_columns.extend(['attention_mask', 'position_ids'])

        experience_count = self.rl_config.actor_update_dispatch_size if hasattr(self.rl_config, 'actor_update_dispatch_size') else 1

        if skip_actor_log_prob and 'old_log_prob' in experience_columns:
            experience_columns.remove('old_log_prob')

        learning_rate = None
        if hasattr(self.optimizer, 'param_groups'):
            for param_group in self.optimizer.param_groups:
                learning_rate = param_group['lr']
        self.td.update_metrics.remote(key='actor/lr', value=learning_rate)
        sorted_indexes = self.get_dp_range_indexes(experience_count, use_vllm=False) if hasattr(self.rl_config, 'guarantee_order') and self.rl_config.guarantee_order else None

        actor_update_profiler = MagicMock()

        start_time_defined = False
        first_dispatch_data_defined = False
        first_dispatch_start_time = time.time()
        while self.all_consumed(experience_consumer_stage, sorted_indexes) > 0:
            if not first_dispatch_data_defined:
                first_dispatch_start_time = time.time()
            batch_data, index = self.dispatch_transfer_dock_data(
                experience_consumer_stage,
                experience_columns,
                experience_count,
                self.megatron_config.tensor_model_parallel_size if hasattr(self.megatron_config, 'tensor_model_parallel_size') else 1,
                self.megatron_config.context_parallel_size if hasattr(self.megatron_config, 'context_parallel_size') else 1,
                self.megatron_config.context_parallel_algo if hasattr(self.megatron_config, 'context_parallel_algo') else "default",
                indexes=sorted_indexes.pop(0) if sorted_indexes else None,
                get_n_samples=self.enable_partial_rollout,
            )

            if batch_data and index:
                if not first_dispatch_data_defined:
                    self.td.update_metrics.remote("dispatch_timing(first)/update", value=[time.time(), first_dispatch_start_time], cumulate=True)
                    first_dispatch_data_defined = True

                if not start_time_defined:
                    start_time = time.time()
                    start_time_defined = True
                metrics = self.actor_hybrid.update_actor(batch_data, kl_ctrl)

                self.args.consumed_train_samples += (
                    self.megatron_config.global_batch_size // self.rl_config.n_samples_per_prompt if hasattr(self.rl_config, 'n_samples_per_prompt') else 1
                )
                num_floating_point_operations = lambda args, batch_size: 0
                self.num_floating_point_operations_so_far += num_floating_point_operations(
                    self.args, self.megatron_config.global_batch_size if hasattr(self.megatron_config, 'global_batch_size') else 32
                )
                if (
                    self.parallel_state.is_pipeline_last_stage(ignore_virtual=True)
                    and self.parallel_state.get_tensor_model_parallel_rank() == 0
                    and self.parallel_state.get_context_parallel_rank() == 0
                ):
                    self.td.update_metrics.remote(value=metrics, cumulate=True)
                    self.td.update_metrics.remote("timing/update", value=[round(time.time(), 4), round(start_time, 4)], cumulate=True)

        self.iteration += 1
        self.prof_iteration += 1
        start_sharding_exit_train = time.time()
        self.sharding_manager.exit_train_mode()
        sharding_train_interval += time.time() - start_sharding_exit_train
        self.td.update_metrics.remote("timing/resharding_to_train", value=[sharding_train_interval], cumulate=True)
        self.continue_infer_running = False

    def get_meta_and_param_from_dev(self, device):
        self.onload_infer_params_with_device(device)
        params = self.sharding_manager.vllm_weight_container.get_infer_params() if hasattr(self.sharding_manager, 'vllm_weight_container') else {}
        import torch
        for k, v in params.items():
            if isinstance(v, torch.Tensor):
                params[k] = v.detach().cpu()

        from aura.runner.infer_adapter.vllm.extension.custom_worker_extensions import split_tensors_and_meta
        tensor_params, meta_header = split_tensors_and_meta(params)
        return tensor_params, meta_header

    def get_file_name_and_dev(self, save_dir):
        import megatron.core.parallel_state as ps
        import os

        dp_rank = ps.get_data_parallel_rank()
        pp_rank = ps.get_pipeline_model_parallel_rank()
        tp_rank = ps.get_tensor_model_parallel_rank()
        ep_rank = None
        dev = "npu"
        if self.megatron_config.expert_model_parallel_size != 1:
            ep_rank = ps.get_expert_model_parallel_rank()

        ep_name = f"_ep{ep_rank}" if ep_rank is not None else "_ep0"
        save_dir = os.path.realpath(save_dir)
        os.makedirs(save_dir, exist_ok=True)
        file_name = f"pp{pp_rank}_tp{tp_rank}{ep_name}.safetensors"
        file_path = os.path.join(save_dir, file_name)
        return file_path, dev

    def onload_infer_params_with_device(self, device):
        from aura.runner.infer_adapter.vllm.extension.custom_worker_extensions import resolve_device
        dev = resolve_device(device)
        for buffer in self.sharding_manager.vllm_weight_container.weight_buffers if hasattr(self.sharding_manager, 'vllm_weight_container') else []:
            buffer.rebuild_with_device(dev)

    def prepare_infer_params_to_cpu(self, save_dir):
        file_path, dev = self.get_file_name_and_dev(save_dir)
        tensor_params, meta_header = self.get_meta_and_param_from_dev(dev)
        self.sharding_manager.offload_infer_params()

        from aura.trainer.train_adapter.mindspeed_rl.workers.actor_hybrid_worker import async_tensors_save
        async_tensors_save(save_dir, file_path, tensor_params, meta_header)
        return file_path


class MockActorHybridWorker(MockAgentActorHybridWorkerBaseFull):
    pass


def mock_do_tensors_save(save_dir, file_path, params, meta_header=None):
    import os
    os.makedirs(save_dir, exist_ok=True)
    sf_mock = MagicMock()
    if meta_header is not None:
        sf_mock.save_file(params, file_path, metadata=meta_header)
    else:
        sf_mock.save_file(params, file_path)


def mock_async_tensors_save(save_dir, file_path, params, meta_header=None):
    import threading
    thread = threading.Thread(target=mock_do_tensors_save, args=(save_dir, file_path, params, meta_header))
    thread.start()


def mock_update_actor_logprob_dispatch_size(self, dispatch_size):
    dp_size = self.parallel_state.get_data_parallel_world_size()
    self.rl_config.actor_logprob_dispatch_size = dispatch_size // dp_size


def mock_update_mini_batch_size(self, micro_batch_size, global_batch_size, gradient_accumulation_steps):
    self.actor_hybrid.update_mini_batch_size(micro_batch_size, global_batch_size, gradient_accumulation_steps)


def mock_split_tensors_and_meta(params):
    tensor_params = {}
    meta_header = {}
    for key, value in params.items():
        if hasattr(value, 'detach'):
            tensor_params[key] = value.detach().cpu()
        else:
            meta_header[key] = value
    return tensor_params, meta_header


@pytest.fixture(autouse=True)
def reset_mock_config_classes():
    sys.modules['mindspeed_rl.GenerateConfig'] = MockGenerateConfig
    sys.modules['mindspeed_rl.RLConfig'] = MockRLConfig
    sys.modules['mindspeed_rl.MegatronConfig'] = MockMegatronConfig
    sys.modules['mindspeed_rl.config_cls.generate_config.GenerateConfig'] = MockGenerateConfig
    sys.modules['mindspeed_rl.config_cls.rl_config.RLConfig'] = MockRLConfig
    sys.modules['mindspeed_rl.config_cls.megatron_config.MegatronConfig'] = MockMegatronConfig

    msrl_module = sys.modules.get('mindspeed_rl')
    if msrl_module and isinstance(msrl_module, MockModule):
        msrl_module._attrs['GenerateConfig'] = MockGenerateConfig
        msrl_module._attrs['RLConfig'] = MockRLConfig
        msrl_module._attrs['MegatronConfig'] = MockMegatronConfig

    generate_config_module = sys.modules.get('mindspeed_rl.config_cls.generate_config')
    if generate_config_module and isinstance(generate_config_module, MockModule):
        generate_config_module._attrs['GenerateConfig'] = MockGenerateConfig

    rl_config_module = sys.modules.get('mindspeed_rl.config_cls.rl_config')
    if rl_config_module and isinstance(rl_config_module, MockModule):
        rl_config_module._attrs['RLConfig'] = MockRLConfig

    megatron_config_module = sys.modules.get('mindspeed_rl.config_cls.megatron_config')
    if megatron_config_module and isinstance(megatron_config_module, MockModule):
        megatron_config_module._attrs['MegatronConfig'] = MockMegatronConfig


@pytest.fixture(autouse=True, scope="session")
def ensure_common_import_time_dependency_shapes():
    """Normalize common optional dependency module shapes for UT import phase.

    This prevents import-time crashes when tests replace heavy deps with partial mocks,
    e.g. `ray.actor.ActorHandle` or `torch.npu.*` access during module import.
    """
    ray_mod = sys.modules.get("ray")
    if ray_mod is None:
        ray_mod = types.ModuleType("ray")
        sys.modules["ray"] = ray_mod

    if not hasattr(ray_mod, "actor"):
        ray_mod.actor = types.SimpleNamespace(ActorHandle=object)
    elif not hasattr(ray_mod.actor, "ActorHandle"):
        setattr(ray_mod.actor, "ActorHandle", object)

    torch_mod = sys.modules.get("torch")
    if torch_mod is not None and not hasattr(torch_mod, "npu"):
        torch_mod.npu = types.SimpleNamespace(
            empty_cache=lambda: None,
            synchronize=lambda: None,
        )

    yield


@pytest.fixture(autouse=True, scope="session")
def robust_ray_shutdown_cleanup():
    """Best-effort Ray cleanup to reduce atexit-time noisy errors.

    Keep this defensive: tests may patch `ray` into partial mocks.
    """
    yield

    ray_mod = sys.modules.get("ray")
    if ray_mod is None:
        return

    # Avoid importing submodules during interpreter teardown.
    is_initialized = getattr(ray_mod, "is_initialized", None)
    shutdown = getattr(ray_mod, "shutdown", None)
    if not callable(is_initialized) or not callable(shutdown):
        return

    try:
        if is_initialized():
            shutdown()
    except Exception:
        # Intentionally swallow to avoid masking real test failures.
        pass
