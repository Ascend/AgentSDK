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
from unittest.mock import patch
import pytest

# ---------------------------------------------------------------------------
# Fixture: fake module tree for exceptions
# ---------------------------------------------------------------------------
@pytest.fixture
def fake_env():
    """Build an isolated fake module tree and return the module under test."""

    # ---- fake fastapi ----
    fake_fastapi = types.ModuleType("fastapi")

    class FakeHTTPException(Exception):
        def __init__(self, status_code, detail=None):
            self.status_code = status_code
            self.detail = detail

    fake_fastapi.HTTPException = FakeHTTPException

    # ---- fake ray ----
    fake_ray = types.ModuleType("ray")
    fake_ray_serve = types.ModuleType("ray.serve")
    fake_ray_serve_context = types.ModuleType("ray.serve.context")
    # Provide a placeholder object so _serve_request_context can be referenced
    fake_ray_serve_context._serve_request_context = object()
    fake_ray_serve.context = fake_ray_serve_context
    fake_ray.serve = fake_ray_serve
    fake_ray.serve_context = None  # Will be set inside the decorator

    # ---- aura packages to locate the real exceptions module ----
    import aura as _aura
    base = _aura.__path__[0] if _aura.__path__ else "."
    fake_aura = types.ModuleType("aura")
    fake_aura.__path__ = _aura.__path__
    fake_aura_base = types.ModuleType("aura.base")
    fake_aura_base.__path__ = []
    fake_aura_base_exceptions = types.ModuleType("aura.base.exceptions")
    fake_aura_base_exceptions.__path__ = [os.path.join(base, "base/exceptions")]

    fakes = {
        "fastapi": fake_fastapi,
        "ray": fake_ray,
        "ray.serve": fake_ray_serve,
        "ray.serve.context": fake_ray_serve_context,
        "aura": fake_aura,
        "aura.base": fake_aura_base,
        "aura.base.exceptions": fake_aura_base_exceptions,
    }

    target = "aura.base.exceptions.exceptions"
    if target in sys.modules:
        del sys.modules[target]

    with patch.dict(sys.modules, fakes):
        import aura.base.exceptions.exceptions as mod
        yield {
            "mod": mod,
            "fake_fastapi": fake_fastapi,
            "fake_ray": fake_ray,
        }

    if target in sys.modules:
        del sys.modules[target]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
class TestAsyncRaiseHttpException:

    @pytest.mark.asyncio
    async def test_normal_execution_sets_serve_context_and_returns_result(self, fake_env):
        """Normal execution: should set ray.serve_context and return the original result."""
        mod = fake_env["mod"]
        fake_ray = fake_env["fake_ray"]

        async def sample_func(x):
            return x + 1

        decorated = mod.async_raise_http_exception(sample_func)
        result = await decorated(1)

        assert result == 2
        assert fake_ray.serve_context is fake_ray.serve.context._serve_request_context

    @pytest.mark.asyncio
    async def test_catches_file_not_found_error_and_raises_404(self, fake_env):
        """Original function raises FileNotFoundError -> convert to HTTP 404."""
        mod = fake_env["mod"]
        HTTPException = fake_env["fake_fastapi"].HTTPException

        async def failing_func():
            raise FileNotFoundError("missing")

        decorated = mod.async_raise_http_exception(failing_func)

        with pytest.raises(HTTPException) as exc_info:
            await decorated()
        assert exc_info.value.status_code == 404
        assert exc_info.value.detail == "File not found"

    @pytest.mark.asyncio
    async def test_catches_key_error_and_raises_400(self, fake_env):
        """Original function raises KeyError -> convert to HTTP 400, detail contains the missing key."""
        mod = fake_env["mod"]
        HTTPException = fake_env["fake_fastapi"].HTTPException

        async def failing_func():
            raise KeyError("missing_key")

        decorated = mod.async_raise_http_exception(failing_func)

        with pytest.raises(HTTPException) as exc_info:
            await decorated()
        assert exc_info.value.status_code == 400
        assert exc_info.value.detail == "Key error: missing_key"

    @pytest.mark.asyncio
    async def test_catches_general_exception_and_raises_500(self, fake_env):
        """Original function raises a general Exception -> convert to HTTP 500."""
        mod = fake_env["mod"]
        HTTPException = fake_env["fake_fastapi"].HTTPException

        async def failing_func():
            raise ValueError("something wrong")

        decorated = mod.async_raise_http_exception(failing_func)

        with pytest.raises(HTTPException) as exc_info:
            await decorated()
        assert exc_info.value.status_code == 500
        assert exc_info.value.detail == "Internal server error: something wrong"

    @pytest.mark.asyncio
    async def test_preserves_function_metadata_via_wraps(self, fake_env):
        """Verify that functools.wraps preserves the original function metadata."""
        mod = fake_env["mod"]

        async def my_func(x):
            """docstring of my_func"""
            return x

        decorated = mod.async_raise_http_exception(my_func)
        assert decorated.__name__ == "my_func"
        assert decorated.__doc__ == "docstring of my_func"
