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
#          http://license.coscl.org.cn/MulanPSL2
#
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
# -------------------------------------------------------------------------
import functools
import inspect
from typing import AsyncIterator, Callable, Any, Optional

from agentic_rl.base.log.loggers import Loggers
from agentic_rl.base.resources.resources import ResourceSet

logger = Loggers(__name__).get_logger()


def public_api(name: str, is_stream: bool = False):
    def decorator(func: Callable):
        func._public_params = (name, is_stream)

        if inspect.isasyncgenfunction(func):
            # Asynchronous generator function
            @functools.wraps(func)
            async def async_gen_wrapper(self, *args, **kwargs) -> AsyncIterator[Any]:
                async for item in func(self, *args, **kwargs):
                    yield item

            return async_gen_wrapper

        elif inspect.iscoroutinefunction(func):
            # Ordinary asynchronous function
            @functools.wraps(func)
            async def async_wrapper(self, *args, **kwargs):
                result = await func(self, *args, **kwargs)
                return result

            return async_wrapper

        else:
            # Synchronous function
            @functools.wraps(func)
            def sync_wrapper(self, *args, **kwargs):
                result = func(self, *args, **kwargs)
                return result

            return sync_wrapper

    return decorator


class Executor:
    _method_registry = {}

    def __init__(self, resource_set: ResourceSet, *args, **kwargs):
        self._register_api()
        self._resource_set: Optional[ResourceSet] = resource_set

    @property
    def resource_set(self):
        return self._resource_set

    def _register_api(self):
        # Obtain the class (which might be a subclass) to which the current instance belongs
        cls = self.__class__

        # Scan all the methods defined in the subclass (including the inheritance chain)
        self._method_registry = {
            getattr(method, "_public_params"): method
            for name, method in inspect.getmembers(cls, predicate=inspect.isfunction)
            if hasattr(method, "_public_params")
        }

        logger.debug(f"[Instance initialization stage] {cls.__name__} has been registered as: {self._method_registry}")

    async def setup(self, *args, **kwargs):
        pass

    async def execute_method(self, method_name, *args, **kwargs):
        method_id = (method_name, False)
        if method_id not in self._method_registry:
            raise AttributeError(f"method={method_name}, is_stream=False does not exist")
        method = self._method_registry[method_id]
        return await method(self, *args, **kwargs)

    async def stream_execute_method(self, method_name, *args, **kwargs):
        method_id = (method_name, True)
        if method_id not in self._method_registry:
            raise AttributeError(f"method={method_name}, is_stream=True does not exist")
        method = self._method_registry[method_id]
        async for item in method(self, *args, **kwargs):
            yield item

    async def finalize(self, *args, **kwargs):
        pass
