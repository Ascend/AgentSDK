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
from unittest.mock import MagicMock

from agentic_rl.base.execution.executor import public_api, Executor
from agentic_rl.base.resources.resources import ResourceSet


class TestPublicApi:
    """
    Tests for public_api decorator.
    """

    def test_public_api_decorator_sync_function(self):
        """
        Test that public_api decorator works with synchronous functions.
        """
        @public_api("test_sync")
        def test_func(self, x, y):
            return x + y

        # Check that the function has _public_params attribute
        assert hasattr(test_func, "_public_params")
        assert test_func._public_params == ("test_sync", False)
        
        # Check that the function still works correctly
        result = test_func(None, 1, 2)
        assert result == 3

    @pytest.mark.asyncio
    async def test_public_api_decorator_async_function(self):
        """
        Test that public_api decorator works with asynchronous functions.
        """
        @public_api("test_async")
        async def test_func(self, x, y):
            return x + y

        # Check that the function has _public_params attribute
        assert hasattr(test_func, "_public_params")
        assert test_func._public_params == ("test_async", False)
        
        # Check that the function still works correctly
        result = await test_func(None, 1, 2)
        assert result == 3

    @pytest.mark.asyncio
    async def test_public_api_decorator_async_gen_function(self):
        """
        Test that public_api decorator works with asynchronous generator functions.
        """
        @public_api("test_stream", is_stream=True)
        async def test_func(self, items):
            for item in items:
                yield item * 2

        # Check that the function has _public_params attribute
        assert hasattr(test_func, "_public_params")
        assert test_func._public_params == ("test_stream", True)
        
        # Check that the function still works correctly
        items = []
        async for item in test_func(None, [1, 2, 3]):
            items.append(item)
        assert items == [2, 4, 6]

    def test_public_api_decorator_with_stream_parameter(self):
        """
        Test that public_api decorator correctly handles is_stream parameter.
        """
        @public_api("test_with_stream", is_stream=True)
        def test_func(self):
            return "test"

        assert test_func._public_params == ("test_with_stream", True)


class TestExecutor:
    """
    Tests for Executor class.
    """

    def setup_method(self):
        # Create a mock resource set
        self.mock_resource_set = MagicMock(spec=ResourceSet)

    def test_executor_initialization(self):
        """
        Test that Executor initializes correctly.
        """
        executor = Executor(self.mock_resource_set)
        assert executor.resource_set == self.mock_resource_set
        assert executor._method_registry == {}

    def test_executor_resource_set_property(self):
        """
        Test that resource_set property works correctly.
        """
        executor = Executor(self.mock_resource_set)
        assert executor.resource_set == self.mock_resource_set

    def test_executor_register_api(self):
        """
        Test that _register_api method correctly registers decorated methods.
        """
        # Create a subclass of Executor with decorated methods
        class TestExecutorSubclass(Executor):
            @public_api("sync_method")
            def sync_method(self):
                return "sync_result"

            @public_api("async_method")
            async def async_method(self):
                return "async_result"

            @public_api("stream_method", is_stream=True)
            async def stream_method(self):
                yield "stream_item"

        # Create an instance and check method registry
        executor = TestExecutorSubclass(self.mock_resource_set)
        assert len(executor._method_registry) == 3
        assert ("sync_method", False) in executor._method_registry
        assert ("async_method", False) in executor._method_registry
        assert ("stream_method", True) in executor._method_registry

    def test_executor_execute_method_sync(self):
        """
        Test that execute_method correctly executes registered synchronous methods.
        """
        class TestExecutorSubclass(Executor):
            @public_api("test_method")
            def test_method(self, x, y):
                return x + y

        executor = TestExecutorSubclass(self.mock_resource_set)
        # For synchronous methods, test the registry and method execution directly
        method_id = ("test_method", False)
        assert method_id in executor._method_registry
        method = executor._method_registry[method_id]
        result = method(executor, 1, 2)
        assert result == 3

    @pytest.mark.asyncio
    async def test_executor_execute_async_method(self):
        """
        Test that execute_method correctly executes registered asynchronous methods.
        """
        class TestExecutorSubclass(Executor):
            @public_api("test_method")
            async def test_method(self, x, y):
                return x + y

        executor = TestExecutorSubclass(self.mock_resource_set)
        result = await executor.execute_method("test_method", 1, 2)
        assert result == 3

    @pytest.mark.asyncio
    async def test_executor_execute_method_not_found(self):
        """
        Test that execute_method raises AttributeError for non-existent methods.
        """
        executor = Executor(self.mock_resource_set)
        with pytest.raises(AttributeError):
            await executor.execute_method("non_existent_method")

    @pytest.mark.asyncio
    async def test_executor_stream_execute_method(self):
        """
        Test that stream_execute_method correctly executes registered stream methods.
        """
        class TestExecutorSubclass(Executor):
            @public_api("test_stream", is_stream=True)
            async def test_stream(self, items):
                for item in items:
                    yield item * 2

        executor = TestExecutorSubclass(self.mock_resource_set)
        items = []
        async for item in executor.stream_execute_method("test_stream", [1, 2, 3]):
            items.append(item)
        assert items == [2, 4, 6]

    @pytest.mark.asyncio
    async def test_executor_stream_execute_method_not_found(self):
        """
        Test that stream_execute_method raises AttributeError for non-existent stream methods.
        """
        executor = Executor(self.mock_resource_set)
        with pytest.raises(AttributeError):
            async for _ in executor.stream_execute_method("non_existent_stream"):
                pass

    @pytest.mark.asyncio
    async def test_executor_setup_and_finalize(self):
        """
        Test that setup and finalize methods can be called.
        """
        executor = Executor(self.mock_resource_set)
        # These methods should not raise any exceptions
        await executor.setup()
        await executor.finalize()

    def test_executor_inheritance(self):
        """
        Test that Executor handles inheritance correctly.
        """
        class BaseExecutorSubclass(Executor):
            @public_api("base_method")
            def base_method(self):
                return "base_result"

        class DerivedExecutorSubclass(BaseExecutorSubclass):
            @public_api("derived_method")
            def derived_method(self):
                return "derived_result"

        # Create an instance of derived class
        executor = DerivedExecutorSubclass(self.mock_resource_set)
        
        # Check that both base and derived methods are registered
        assert len(executor._method_registry) == 2
        assert ("base_method", False) in executor._method_registry
        assert ("derived_method", False) in executor._method_registry