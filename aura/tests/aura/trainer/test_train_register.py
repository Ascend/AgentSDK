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
import importlib
import pytest
from unittest.mock import MagicMock, patch


class TestTrainRegistry:
    """
    Tests for the TrainRegistry class.

    Goals:
    - Prevent sys.modules contamination across pytest files
    - Ensure registry initialization executes correctly
    - Ensure coverage metrics are accurate
    """

    def setup_method(self):
        """
        Runs before each test: setup mocks
        """
        # Create mock objects
        self.mock_ray = MagicMock()
        self.mock_ray.remote = lambda cls_func: cls_func
        self.mock_ray.get_actor = MagicMock()
        self.mock_ray.get = MagicMock()
        self.mock_ray.util = MagicMock()
        self.mock_ray.util.scheduling_strategies = MagicMock()
        self.mock_ray.util.scheduling_strategies.PlacementGroupSchedulingStrategy = MagicMock()

        self.mock_loggers_module = MagicMock()
        self.mock_logger = MagicMock()
        self.mock_loggers_module.Loggers.return_value.get_logger.return_value = self.mock_logger

        # Create comprehensive mock mindspeed_rl module hierarchy
        mock_mindspeed_rl = MagicMock()
        mock_mindspeed_rl.workers = MagicMock()
        mock_mindspeed_rl.workers.scheduler = MagicMock()
        mock_mindspeed_rl.workers.scheduler.launcher = MagicMock()
        mock_mindspeed_rl.workers.scheduler.launcher.RayActorGroup = MagicMock()
        mock_mindspeed_rl.workers.resharding = MagicMock()
        mock_mindspeed_rl.workers.resharding.megatron_sharding_manager = MagicMock()
        mock_mindspeed_rl.workers.resharding.megatron_sharding_manager.MegatronShardingManager = MagicMock()
        mock_mindspeed_rl.utils = MagicMock()
        mock_mindspeed_rl.utils.utils = MagicMock()
        mock_mindspeed_rl.utils.utils.mstx_timer_decorator = MagicMock()

        # Create module patches
        self.mocked_modules = {
            "aura.base.log.loggers": self.mock_loggers_module,
            "aura.trainer.rollout.rollout_main": MagicMock(),
            "aura.trainer.train_adapter.mindspeed_rl": MagicMock(),
            "aura.trainer.train_adapter.mindspeed_rl.hybrid_policy": MagicMock(),
            "aura.trainer.train_adapter.mindspeed_rl.hybrid_policy.train_service": MagicMock(),
            "aura.trainer.train_adapter.mindspeed_rl.one_step_off_policy": MagicMock(),
            "aura.trainer.train_adapter.mindspeed_rl.one_step_off_policy.train": MagicMock(),
            "aura.trainer.train_adapter.mindspeed_rl.one_step_off_policy.train.train_service": MagicMock(),
            "aura.trainer.train_adapter.mindspeed_rl.one_step_off_policy.train_service": MagicMock(),
            "aura.trainer.train_adapter.verl.full_async.train_main": MagicMock(),
            "aura.trainer.train_adapter.verl.hybrid.train_main": MagicMock(),
            "aura.trainer.train_adapter.omni_rl.hybrid.train_main": MagicMock(),
            "ray": self.mock_ray,
            "ray.util": self.mock_ray.util,
            "ray.util.scheduling_strategies": self.mock_ray.util.scheduling_strategies,
            "mindspeed_rl": mock_mindspeed_rl,
            "mindspeed_rl.workers": mock_mindspeed_rl.workers,
            "mindspeed_rl.workers.scheduler": mock_mindspeed_rl.workers.scheduler,
            "mindspeed_rl.workers.scheduler.launcher": mock_mindspeed_rl.workers.scheduler.launcher,
            "mindspeed_rl.workers.resharding": mock_mindspeed_rl.workers.resharding,
            "mindspeed_rl.workers.resharding.megatron_sharding_manager": mock_mindspeed_rl.workers.resharding.megatron_sharding_manager,
            "mindspeed_rl.utils": mock_mindspeed_rl.utils,
            "mindspeed_rl.utils.utils": mock_mindspeed_rl.utils.utils,
        }

        # Save original modules
        self.original_modules = {}
        for mod_name in self.mocked_modules.keys():
            if mod_name in sys.modules:
                self.original_modules[mod_name] = sys.modules[mod_name]

        # Apply patches
        sys.modules.update(self.mocked_modules)

        # Remove cached train_register module if it exists
        if 'aura.trainer.trainer_register.train_register' in sys.modules:
            del sys.modules['aura.trainer.trainer_register.train_register']

        # Import after patching
        import aura.trainer.trainer_register.train_register as train_register
        self.train_register = train_register

    def teardown_method(self):
        """
        Runs after each test: restore original modules
        """
        # Restore original modules
        for mod_name, mod in self.original_modules.items():
            sys.modules[mod_name] = mod

        # Remove mocked modules that weren't originally present
        for mod_name in self.mocked_modules.keys():
            if mod_name not in self.original_modules and mod_name in sys.modules:
                del sys.modules[mod_name]

    def test_initialization(self):
        """
        Test that TrainBackendRegistry initializes correctly.
        """
        TrainBackendRegistry = self.train_register.TrainBackendRegistry
        registry_instance = TrainBackendRegistry()
        assert hasattr(registry_instance, "_registry")
        assert isinstance(registry_instance._registry, dict)
        assert len(registry_instance._registry) == 0

    def test_register_and_get_method(self):
        """
        Test register and get_method functionality.
        """
        TrainBackendRegistry = self.train_register.TrainBackendRegistry
        registry_instance = TrainBackendRegistry()

        train_mock = MagicMock()

        registry_instance.register("engine1", "mode1", train_mock)

        assert ("engine1", "mode1") in registry_instance._registry
        assert registry_instance.get_method("engine1", "mode1") == train_mock
        assert registry_instance.get_method("non_exist", "mode1") is None

    def test_train_register_class(self):
        """
        Test that TrainRegister class initializes correctly.
        """
        TrainRegister = self.train_register.TrainRegister
        register_instance = TrainRegister()
        assert hasattr(register_instance, "registry")
        assert isinstance(register_instance.registry, self.train_register.TrainBackendRegistry)

    def test_register_methods(self):
        """
        Test registry methods work correctly.
        """
        TrainRegister = self.train_register.TrainRegister
        register_instance = TrainRegister()

        # Verify methods exist
        assert hasattr(register_instance, 'registry_msrl_train')
        assert hasattr(register_instance, 'registry_verl_train')
        assert hasattr(register_instance, 'get_method')

        # Call registry methods
        register_instance.registry_msrl_train()
        register_instance.registry_verl_train()

        # Verify registrations
        assert register_instance.get_method("mindspeed_rl", "hybrid") is not None
        assert register_instance.get_method("mindspeed_rl", "one_step_off") is not None
        assert register_instance.get_method("mindspeed_rl", "dummy_train") is not None
        assert register_instance.get_method("verl", "hybrid") is not None
        assert register_instance.get_method("verl", "one_step_off") is not None
        assert register_instance.get_method("verl", "dummy_train") is not None

    def test_get_train_method(self):
        """
        Test get_train_method function.
        """
        with patch('ray.get_actor') as mock_get_actor, \
             patch('ray.get') as mock_ray_get:

            mock_actor = MagicMock()
            mock_get_actor.return_value = mock_actor
            mock_ray_get.return_value = MagicMock()

            # Call the function
            result = self.train_register.get_train_method("test_engine", "test_mode")

            # Verify calls
            mock_get_actor.assert_called_once_with(self.train_register.TRAIN_REGISTER_ACTOR_NAME, namespace=self.train_register.TRAIN_REGISTER_NAMESPACE)
            mock_actor.get_method.remote.assert_called_once_with("test_engine", "test_mode")
            mock_ray_get.assert_called_once()
            assert result == mock_ray_get.return_value

    def test_get_train_actor(self):
        """
        Test get_train_actor function.
        """
        with patch.object(self.train_register, 'create_actor') as mock_create_actor, \
             patch.object(self.train_register, 'DEFAULT_CPUS', 1), \
             patch.object(self.train_register, 'MAX_CONCURRENCY', 100):

            mock_actor = MagicMock()
            mock_create_actor.return_value = mock_actor

            # Call the function
            result = self.train_register.get_train_actor()

            # Verify call
            mock_create_actor.assert_called_once()
            args, kwargs = mock_create_actor.call_args
            assert kwargs['name'] == self.train_register.TRAIN_REGISTER_ACTOR_NAME
            assert kwargs['cls'] == self.train_register.TrainRegister
            assert kwargs['namespace'] == self.train_register.TRAIN_REGISTER_NAMESPACE
            assert kwargs['options'] == {"num_cpus": 1, "max_concurrency": 100}
            assert result == mock_actor
