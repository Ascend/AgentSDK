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
import asyncio
import sys
import types
from unittest.mock import MagicMock, patch, AsyncMock

import pytest


@pytest.fixture
def fake_serve_env():
    # Fake fastapi module
    fake_fastapi = types.ModuleType("fastapi")
    fake_fastapi.FastAPI = MagicMock()
    app_instance = fake_fastapi.FastAPI.return_value

    # Make app.get a pass-through decorator so the original async function is preserved
    def fake_get(path):
        def decorator(func):
            return func
        return decorator
    app_instance.get.side_effect = fake_get

    # Fake ray.serve module
    fake_ray_serve = types.ModuleType("ray.serve")

    def deployment_decorator_factory(*args, **kwargs):
        def decorator(cls):
            cls.bind = MagicMock(return_value=MagicMock(name="Application"))
            return cls
        return decorator
    fake_ray_serve.deployment = MagicMock(side_effect=deployment_decorator_factory)

    def ingress_decorator_factory(app):
        def decorator(cls):
            return cls
        return decorator
    fake_ray_serve.ingress = MagicMock(side_effect=ingress_decorator_factory)

    # Fake aura.serve.router module
    fake_router_mod = types.ModuleType("aura.serve.router")
    fake_router = MagicMock(name="router")
    fake_router_mod.router = fake_router

    all_fakes = {
        "fastapi": fake_fastapi,
        "ray.serve": fake_ray_serve,
        "aura.serve.router": fake_router_mod,
    }

    with patch.dict(sys.modules, all_fakes):
        yield {
            "fastapi_mock": fake_fastapi,
            "ray_serve_mock": fake_ray_serve,
            "router_mock": fake_router,
        }


def test_serve_module_import_and_app_creation(fake_serve_env):
    import aura.serve.serve as serve_module

    env = fake_serve_env
    fastapi_mock = env["fastapi_mock"]
    ray_serve_mock = env["ray_serve_mock"]
    router_mock = env["router_mock"]

    # FastAPI app was created
    fastapi_mock.FastAPI.assert_called_once()
    app = fastapi_mock.FastAPI.return_value

    # Router is included
    app.include_router.assert_called_once_with(router_mock)

    # Deployment decorator called with correct args
    ray_serve_mock.deployment.assert_called_once_with(
        autoscaling_config={"min_replicas": 4, "max_replicas": 4, "target_ongoing_requests": 32},
        max_ongoing_requests=128,
    )

    # Ingress decorator called with the app
    ray_serve_mock.ingress.assert_called_once_with(app)

    # AgenticAIDeployment class exists and bind was called
    agentic_class = serve_module.AgenticAIDeployment
    assert agentic_class is not None
    agentic_class.bind.assert_called_once()
    assert serve_module.deployment is agentic_class.bind.return_value


def test_serve_root_route_registration(fake_serve_env):
    import aura.serve.serve as serve_module

    app = serve_module.app
    # Two routes should be registered: "/" and "/delay"
    assert app.get.call_count >= 2

    paths = [call[0][0] for call in app.get.call_args_list]
    assert "/" in paths
    assert "/delay" in paths


def test_deployment_has_correct_methods(fake_serve_env):
    import aura.serve.serve as serve_module

    agentic_class = serve_module.AgenticAIDeployment
    assert hasattr(agentic_class, "root")


def test_deployment_bind_result(fake_serve_env):
    import aura.serve.serve as serve_module

    deployment = serve_module.deployment
    assert deployment is not None

    agentic_class = serve_module.AgenticAIDeployment
    assert deployment is agentic_class.bind.return_value


def test_root_method_execution(fake_serve_env):
    import aura.serve.serve as serve_module

    agentic_class = serve_module.AgenticAIDeployment
    instance = agentic_class()

    # Use AsyncMock so it can be awaited inside the coroutine
    with patch("asyncio.sleep", new_callable=AsyncMock):
        result = asyncio.run(instance.root())
        assert result == "welcome to agentic ai!"
