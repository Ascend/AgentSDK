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
import asyncio
import pytest
from unittest.mock import patch, MagicMock, AsyncMock

from agentic_rl.base.execution.executor_manager import ExecutorItem, ExecutorInstance, ExecutorManager
from agentic_rl.base.execution.executor import Executor
from agentic_rl.base.resources.resources import ResourceSet

# Import ActorHandle type
try:
    from ray.actor import ActorHandle
except ImportError:
    # Fallback if ray is not available
    ActorHandle = type('ActorHandle', (), {})


class TestExecutorItem:
    """
    Tests for ExecutorItem class.
    """

    def test_executor_item_initialization(self):
        """
        Test that ExecutorItem initializes correctly.
        """
        # Create a mock that passes isinstance check for ActorHandle
        mock_ref = MagicMock(spec=ActorHandle)
        mock_resource_set = MagicMock(spec=ResourceSet)
        
        executor_item = ExecutorItem(ref=mock_ref, resource_set=mock_resource_set)
        
        assert executor_item.ref == mock_ref
        assert executor_item.resource_set == mock_resource_set

    def test_executor_item_getattr(self):
        """
        Test that __getattr__ correctly delegates to the ref object.
        """
        # Create a more flexible mock without strict spec
        mock_ref = MagicMock()
        mock_ref.some_method.return_value = "test_result"
        mock_ref.some_attribute = "test_value"
        mock_resource_set = MagicMock(spec=ResourceSet)
        
        # Use __dict__ to bypass type checking when setting ref
        executor_item = ExecutorItem.__new__(ExecutorItem)
        executor_item.__dict__['ref'] = mock_ref
        executor_item.__dict__['resource_set'] = mock_resource_set
        
        # Test method delegation
        result = executor_item.some_method()
        assert result == "test_result"
        mock_ref.some_method.assert_called_once()
        
        # Test attribute delegation
        assert executor_item.some_attribute == "test_value"

    def test_executor_item_getattr_nonexistent(self):
        """
        Test that __getattr__ raises AttributeError for non-existent attributes.
        """
        mock_ref = MagicMock(spec=ActorHandle)
        mock_resource_set = MagicMock(spec=ResourceSet)
        
        executor_item = ExecutorItem(ref=mock_ref, resource_set=mock_resource_set)
        
        with pytest.raises(AttributeError):
            _ = executor_item.nonexistent_attribute


class TestExecutorInstance:
    """
    Tests for ExecutorInstance class.
    """

    def test_executor_instance_initialization(self):
        """
        Test that ExecutorInstance initializes correctly.
        """
        # Create an actual subclass of Executor instead of a mock
        class TestExecutorSubclass(Executor):
            pass
        
        # Create actual ExecutorItem objects with mocked ref and resource_set
        mock_ref1 = MagicMock(spec=ActorHandle)
        mock_resource_set1 = MagicMock(spec=ResourceSet)
        mock_ref2 = MagicMock(spec=ActorHandle)
        mock_resource_set2 = MagicMock(spec=ResourceSet)
        
        # Create real ExecutorItem instances
        executor_item1 = ExecutorItem(ref=mock_ref1, resource_set=mock_resource_set1)
        executor_item2 = ExecutorItem(ref=mock_ref2, resource_set=mock_resource_set2)
        executor_items = [executor_item1, executor_item2]
        
        # Call the constructor with proper ExecutorItem instances
        instance = ExecutorInstance(
            name="test_instance",
            executor_class=TestExecutorSubclass,
            executor_num=2,
            executor_kwargs={"param1": "value1"},
            resource_info=["resource1", "resource2"],
            executor_list=executor_items
        )
        
        # Test that we can access the attributes
        assert instance.name == "test_instance"
        assert instance.executor_class == TestExecutorSubclass
        assert instance.executor_num == 2
        assert instance.executor_kwargs == {"param1": "value1"}
        assert instance.resource_info == ["resource1", "resource2"]
        assert instance.executor_list == executor_items


class TestExecutorManager:
    """
    Tests for ExecutorManager class.
    """

    def setup_method(self):
        # Create a mock ray module
        self.mock_ray = MagicMock()
        # Add required attributes for ray module
        self.mock_ray.__commit__ = "{{RAY_COMMIT_SHA}}"  # Mock commit SHA
        self.mock_ray.__version__ = "2.0.0"  # Mock version
        
        # Create a mock ref that passes ActorHandle type check
        self.mock_actor_handle = MagicMock()
        self.mock_ray.remote.return_value.options.return_value.remote.return_value = self.mock_actor_handle
        
        # Create a mock PlacementGroupSchedulingStrategy that passes type checks
        self.mock_scheduling_strategy = MagicMock()
        # Set the class name to make it look like the real type
        self.mock_scheduling_strategy.__class__.__name__ = "PlacementGroupSchedulingStrategy"
        
        # Create a mock ResourceSet
        self.mock_resource_set = MagicMock(spec=ResourceSet)
        self.mock_resource_set.ref = None
        self.mock_resource_set.info = []
        
        # Create an actual subclass of Executor instead of a mock
        class TestExecutorSubclass(Executor):
            pass
        
        self.mock_executor_class = TestExecutorSubclass
        # Create a mock instance of the executor class
        self.mock_executor_instance = MagicMock(spec=TestExecutorSubclass)
        self.mock_executor_instance.setup = AsyncMock()
        self.mock_executor_instance.finalize = AsyncMock()
        
        # Setup mock for ray.actor module
        mock_actor = MagicMock()
        
        # Create a mock Actor class with _remote method that doesn't trigger auto init
        mock_actor_cls = MagicMock()
        mock_actor_cls._remote = MagicMock(return_value=self.mock_actor_handle)
        
        # Setup ray.remote to return the mock Actor class
        self.mock_ray.remote.return_value = mock_actor_cls
        
        # Setup options and remote methods
        mock_options = MagicMock()
        mock_options.remote.return_value = self.mock_actor_handle
        mock_actor_cls.options.return_value = mock_options
        
        # Patch sys.modules to mock ray and its submodules
        self.patcher = patch.dict('sys.modules', {
            'ray': self.mock_ray,
            'ray.actor': mock_actor,
            'ray.util': MagicMock(),
            'ray.util.scheduling_strategies': MagicMock(),
        })
        self.patcher.start()
        
        # Setup scheduling strategy mock
        from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
        PlacementGroupSchedulingStrategy.return_value = self.mock_scheduling_strategy

    def teardown_method(self):
        # Stop the patcher
        self.patcher.stop()

    @patch('agentic_rl.base.execution.executor_manager.ray')
    @patch('agentic_rl.base.execution.executor_manager.ExecutorItem')
    @patch('agentic_rl.base.execution.executor_manager.create_resource_set')
    @pytest.mark.asyncio
    async def test_create_executor(self, mock_create_resource_set, mock_executor_item, mock_ray):
        """
        Test that _create_executor correctly creates an executor instance.
        """
        # Setup mocks
        mock_create_resource_set.return_value = self.mock_resource_set
        
        # Create a mock ref with setup method and remote call
        mock_actor_handle = MagicMock()
        mock_setup = AsyncMock()
        mock_setup.remote = AsyncMock()
        mock_actor_handle.setup = mock_setup
        
        # Configure the mock ray.remote call chain
        mock_actor_cls = MagicMock()
        mock_options = MagicMock()
        mock_options.remote.return_value = mock_actor_handle
        mock_actor_cls.options.return_value = mock_options
        mock_ray.remote.return_value = mock_actor_cls
        
        # Configure the mock ExecutorItem
        mock_executor_item_instance = MagicMock()
        mock_executor_item_instance.ref = mock_actor_handle
        mock_executor_item_instance.resource_set = self.mock_resource_set

        mock_executor_item_instance.finalize = AsyncMock()  # Make finalize an AsyncMock
        mock_executor_item.return_value = mock_executor_item_instance
        
        # Create executor manager
        manager = ExecutorManager()
        
        # Call _create_executor
        executor_item = await manager._create_executor(
            self.mock_executor_class,
            {"param1": "value1"},
            ["resource1"]
        )
        
        # Verify resource set creation
        mock_create_resource_set.assert_called_once_with(["resource1"])
        
        # Verify executor creation
        mock_ray.remote.assert_called_once_with(self.mock_executor_class)
        mock_actor_cls.options.assert_called_once()
        mock_options.remote.assert_called_once()
        
        # Verify setup was called
        mock_actor_handle.setup.remote.assert_called_once()
        
        # Verify ExecutorItem was created correctly
        mock_executor_item.assert_called_once_with(
            ref=mock_actor_handle,
            resource_set=self.mock_resource_set
        )
        assert executor_item.ref == mock_actor_handle
        assert executor_item.resource_set == self.mock_resource_set

    @patch('agentic_rl.base.execution.executor_manager.ray')
    @patch('agentic_rl.base.execution.executor_manager.ExecutorItem')
    @patch('agentic_rl.base.execution.executor_manager.create_resource_set')
    @patch('ray.util.scheduling_strategies.PlacementGroupSchedulingStrategy')
    @pytest.mark.asyncio
    async def test_create_executor_with_resource_set_ref(self, mock_pgss, mock_create_resource_set, mock_executor_item, mock_ray):
        """
        Test that _create_executor correctly handles resource sets with ref.
        """
        # Setup mocks
        mock_resource_set_with_ref = MagicMock(spec=ResourceSet)
        mock_resource_set_with_ref.ref = "test_placement_group"
        mock_create_resource_set.return_value = mock_resource_set_with_ref
        
        # Create a mock ref with setup method and remote call
        mock_actor_handle = MagicMock()
        mock_setup = AsyncMock()
        mock_setup.remote = AsyncMock()
        mock_actor_handle.setup = mock_setup
        
        # Configure the mock ray.remote call chain
        mock_actor_cls = MagicMock()
        mock_options = MagicMock()
        mock_options.remote.return_value = mock_actor_handle
        mock_actor_cls.options.return_value = mock_options
        mock_ray.remote.return_value = mock_actor_cls
        
        # Configure the mock ExecutorItem
        mock_executor_item_instance = MagicMock()
        mock_executor_item.return_value = mock_executor_item_instance
        
        # Create executor manager
        manager = ExecutorManager()
        
        # Call _create_executor
        await manager._create_executor(
            self.mock_executor_class,
            {"param1": "value1"},
            ["resource1"]
        )
        
        # Verify that scheduling strategy was used
        mock_pgss.assert_called_once_with(
            placement_group="test_placement_group",
            placement_group_bundle_index=0
        )
        
        # Verify ray options were set correctly
        mock_actor_cls.options.assert_called_once()

    @patch('agentic_rl.base.execution.executor_manager.ray')
    @patch('agentic_rl.base.execution.executor_manager.ExecutorItem')
    @patch('agentic_rl.base.execution.executor_manager.create_resource_set')
    @pytest.mark.asyncio
    async def test_remove_executor(self, mock_create_resource_set, mock_executor_item, mock_ray):
        """
        Test that _remove_executor correctly removes an executor instance.
        """
        # Setup mocks
        mock_create_resource_set.return_value = self.mock_resource_set
        
        # Create a mock ref with finalize method
        mock_actor_handle = MagicMock()
        mock_setup = AsyncMock()
        mock_setup.remote = AsyncMock()
        mock_actor_handle.setup = mock_setup  # Needed for _create_executor
        mock_actor_handle.finalize = AsyncMock()
        
        # Configure the mock ray.remote call chain
        mock_actor_cls = MagicMock()
        mock_options = MagicMock()
        mock_options.remote.return_value = mock_actor_handle
        mock_actor_cls.options.return_value = mock_options
        mock_ray.remote.return_value = mock_actor_cls
        
        # Configure the mock ExecutorItem
        mock_executor_item_instance = MagicMock()
        mock_executor_item_instance.ref = mock_actor_handle
        mock_executor_item_instance.resource_set = self.mock_resource_set
        # Set finalize as AsyncMock since MagicMock won't use __getattr__
        mock_executor_item_instance.finalize = AsyncMock()
        mock_executor_item.return_value = mock_executor_item_instance
        
        # Create executor manager
        manager = ExecutorManager()
        
        # Create an executor first
        executor_item = await manager._create_executor(
            self.mock_executor_class,
            {"param1": "value1"},
            ["resource1"]
        )
        
        # Call _remove_executor
        await manager._remove_executor(executor_item)
        
        # Verify finalize was called
        mock_executor_item_instance.finalize.assert_called_once()
        
        # Verify ray.kill was called
        mock_ray.kill.assert_called_once_with(mock_actor_handle)

    @patch('agentic_rl.base.execution.executor_manager.ExecutorManager._create_executor')
    @patch('agentic_rl.base.execution.executor_manager.ExecutorInstance')
    @pytest.mark.asyncio
    async def test_create_instance(self, mock_executor_instance, mock_create_executor):
        """
        Test that create_instance correctly creates an instance group.
        """
        # Setup mocks
        mock_ref = MagicMock()
        mock_resource_set = MagicMock(spec=ResourceSet)
        
        # Create a mock ExecutorItem
        mock_executor_item = MagicMock()
        mock_executor_item.ref = mock_ref
        mock_executor_item.resource_set = mock_resource_set
        mock_create_executor.return_value = mock_executor_item
        
        # Configure the mock ExecutorInstance
        mock_instance = MagicMock()
        mock_instance.name = "test_instance"
        mock_instance.executor_class = self.mock_executor_class
        mock_instance.executor_num = 2
        mock_instance.executor_kwargs = {"param1": "value1"}
        mock_instance.resource_info = ["resource1"]
        mock_instance.executor_list = [mock_executor_item, mock_executor_item]
        mock_executor_instance.return_value = mock_instance
        
        # Create executor manager
        manager = ExecutorManager()
        
        # Call create_instance
        instance = await manager.create_instance(
            name="test_instance",
            executor_class=self.mock_executor_class,
            executor_num=2,
            executor_kwargs={"param1": "value1"},
            resource_info=["resource1"]
        )
        
        # Verify _create_executor was called twice
        assert mock_create_executor.call_count == 2
        
        # Verify ExecutorInstance was created
        mock_executor_instance.assert_called_once()
        call_args = mock_executor_instance.call_args[1]
        assert call_args["name"] == "test_instance"
        assert call_args["executor_class"] == self.mock_executor_class
        assert call_args["executor_num"] == 2
        assert call_args["executor_kwargs"] == {"param1": "value1"}
        # resource_info is not passed to ExecutorInstance in the current implementation
        assert len(call_args["executor_list"]) == 2
        
        # Verify instance was added to instance_dict
        assert "test_instance" in manager.instance_dict
        assert manager.instance_dict["test_instance"] == mock_instance
        
        # Verify instance properties
        assert instance.name == "test_instance"
        assert instance.executor_class == self.mock_executor_class
        assert instance.executor_num == 2
        assert instance.executor_kwargs == {"param1": "value1"}
        assert len(instance.executor_list) == 2

    @patch('agentic_rl.base.execution.executor_manager.ExecutorManager._create_executor')
    @patch('agentic_rl.base.execution.executor_manager.ExecutorInstance')
    @pytest.mark.asyncio
    async def test_create_instance_already_exists(self, mock_executor_instance, mock_create_executor):
        """
        Test that create_instance raises ValueError if instance already exists.
        """
        # Setup mocks
        mock_executor_item = MagicMock()
        mock_create_executor.return_value = mock_executor_item
        
        # Configure the mock ExecutorInstance
        mock_instance = MagicMock()
        mock_instance.name = "test_instance"
        mock_instance.executor_class = self.mock_executor_class
        mock_instance.executor_num = 1
        mock_instance.executor_kwargs = {"param1": "value1"}
        mock_instance.resource_info = ["resource1"]
        mock_instance.executor_list = [mock_executor_item]
        mock_executor_instance.return_value = mock_instance
        
        # Create executor manager
        manager = ExecutorManager()
        
        # Create an instance first
        await manager.create_instance(
            name="test_instance",
            executor_class=self.mock_executor_class,
            executor_num=1,
            executor_kwargs={"param1": "value1"},
            resource_info=["resource1"]
        )
        
        # Try to create the same instance again
        with pytest.raises(ValueError):
            await manager.create_instance(
                name="test_instance",
                executor_class=self.mock_executor_class,
                executor_num=1,
                executor_kwargs={"param1": "value1"},
                resource_info=["resource1"]
            )

    @patch('agentic_rl.base.execution.executor_manager.ExecutorManager._create_executor')
    @patch('agentic_rl.base.execution.executor_manager.ExecutorInstance')
    @pytest.mark.asyncio
    async def test_get_instance(self, mock_executor_instance, mock_create_executor):
        """
        Test that get_instance correctly retrieves an instance.
        """
        # Setup mocks
        mock_executor_item = MagicMock()
        mock_create_executor.return_value = mock_executor_item
        
        # Configure the mock ExecutorInstance
        mock_instance = MagicMock()
        mock_instance.name = "test_instance"
        mock_instance.executor_class = self.mock_executor_class
        mock_instance.executor_num = 1
        mock_instance.executor_kwargs = {"param1": "value1"}
        mock_instance.resource_info = ["resource1"]
        mock_instance.executor_list = [mock_executor_item]
        mock_executor_instance.return_value = mock_instance
        
        # Create executor manager
        manager = ExecutorManager()
        
        # Create an instance
        await manager.create_instance(
            name="test_instance",
            executor_class=self.mock_executor_class,
            executor_num=1,
            executor_kwargs={"param1": "value1"},
            resource_info=["resource1"]
        )
        
        # Call get_instance
        instance = await manager.get_instance("test_instance")
        
        # Verify instance was retrieved correctly
        assert instance == mock_instance
        assert instance.name == "test_instance"

    @pytest.mark.asyncio
    async def test_get_instance_not_found(self):
        """
        Test that get_instance raises ValueError if instance not found.
        """
        # Create executor manager
        manager = ExecutorManager()
        
        # Try to get a non-existent instance
        with pytest.raises(ValueError):
            await manager.get_instance("non_existent_instance")

    @patch('agentic_rl.base.execution.executor_manager.ExecutorManager._remove_executor')
    @patch('agentic_rl.base.execution.executor_manager.ExecutorManager._create_executor')
    @patch('agentic_rl.base.execution.executor_manager.ExecutorInstance')
    @pytest.mark.asyncio
    async def test_remove_instance(self, mock_executor_instance, mock_create_executor, mock_remove_executor):
        # Setup mocks
        mock_executor_item = MagicMock()
        mock_create_executor.return_value = mock_executor_item
        
        # Configure the mock ExecutorInstance
        mock_instance = MagicMock()
        mock_instance.name = "test_instance"
        mock_instance.executor_class = self.mock_executor_class
        mock_instance.executor_num = 2
        mock_instance.executor_kwargs = {"param1": "value1"}
        mock_instance.resource_info = ["resource1"]
        mock_instance.executor_list = [mock_executor_item, mock_executor_item]
        mock_executor_instance.return_value = mock_instance
        
        # Create executor manager
        manager = ExecutorManager()
        
        # Create an instance
        await manager.create_instance(
            name="test_instance",
            executor_class=self.mock_executor_class,
            executor_num=2,
            executor_kwargs={"param1": "value1"},
            resource_info=["resource1"]
        )
        
        # Call remove_instance
        await manager.remove_instance("test_instance")
        
        # Verify instance was removed from instance_dict
        assert "test_instance" not in manager.instance_dict
        
        # Verify _remove_executor was called twice
        assert mock_remove_executor.call_count == 2

    @patch('agentic_rl.base.execution.executor_manager.ExecutorManager._remove_executor')
    @pytest.mark.asyncio
    async def test_remove_instance_not_found(self, mock_remove_executor):
        """
        Test that remove_instance handles non-existent instances gracefully.
        """
        # Create executor manager
        manager = ExecutorManager()
        
        # Call remove_instance for a non-existent instance
        await manager.remove_instance("non_existent_instance")
        
        # Verify _remove_executor was not called
        mock_remove_executor.assert_not_called()

    @patch('agentic_rl.base.execution.executor_manager.ExecutorManager.remove_instance')
    @patch('agentic_rl.base.execution.executor_manager.ExecutorManager._create_executor')
    @patch('agentic_rl.base.execution.executor_manager.ExecutorInstance')
    @pytest.mark.asyncio
    async def test_finalize(self, mock_executor_instance, mock_create_executor, mock_remove_instance):
        """
        Test that finalize correctly cleans up all instances.
        """
        # Setup mocks
        mock_executor_item = MagicMock()
        mock_create_executor.return_value = mock_executor_item
        
        # Configure the mock ExecutorInstance
        mock_instance = MagicMock()
        mock_instance.name = "test_instance"
        mock_instance.executor_class = self.mock_executor_class
        mock_instance.executor_num = 1
        mock_instance.executor_kwargs = {"param1": "value1"}
        mock_instance.resource_info = ["resource1"]
        mock_instance.executor_list = [mock_executor_item]
        mock_executor_instance.return_value = mock_instance
        
        # Create executor manager
        manager = ExecutorManager()
        
        # Create two instances
        await manager.create_instance(
            name="test_instance_1",
            executor_class=self.mock_executor_class,
            executor_num=1,
            executor_kwargs={"param1": "value1"},
            resource_info=["resource1"]
        )
        await manager.create_instance(
            name="test_instance_2",
            executor_class=self.mock_executor_class,
            executor_num=1,
            executor_kwargs={"param1": "value1"},
            resource_info=["resource1"]
        )
        
        # Configure the mock to actually remove instances from the dict
        async def mock_remove_instance_side_effect(name):
            if name in manager.instance_dict:
                del manager.instance_dict[name]
        
        mock_remove_instance.side_effect = mock_remove_instance_side_effect
        
        # Call finalize
        await manager.finalize()
        
        # Verify remove_instance was called twice
        assert mock_remove_instance.call_count == 2
        
        # Verify instance_dict is empty
        assert len(manager.instance_dict) == 0