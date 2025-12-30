#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
-------------------------------------------------------------------------
This file is part of the AgentSDK project.
Copyright (c) 2025 Huawei Technologies Co.,Ltd.

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
import re
import sys
from unittest.mock import MagicMock, patch, Mock

import pytest
from ray.exceptions import RayError
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy


class MockActorHandlerParams:
    def __init__(self, master_addr, master_port, world_size, rank_index, placement_group, bundle_index):
        self.master_addr = master_addr
        self.master_port = master_port
        self.world_size = world_size
        self.rank_index = rank_index
        self.placement_group = placement_group
        self.bundle_index = bundle_index


class MockRayActorGroup:
    def __init__(self, worker):
        self.worker = worker


class MockWorker:
    def __init__(self):
        self.__ray_metadata__ = MagicMock()
        self.__ray_metadata__.modified_class = MagicMock()
        self.__ray_metadata__.modified_class.__name__ = "MockActor"

    def options(self, name, scheduling_strategy, runtime_env):
        self.name = name
        self.scheduling_strategy = scheduling_strategy
        self.runtime_env = runtime_env
        return self

    def remote(self, **kwargs):
        return self


class TestLauncher:
    pytest.fixture()

    @pytest.fixture()
    def patch_modules(self):
        with patch.dict(sys.modules, {"mindspeed_rl.workers.scheduler.launcher": Mock()}):
            yield

    @pytest.fixture
    def test_actor(self, patch_modules):
        actor = MockRayActorGroup(worker=MockWorker())
        actor.megatron_config = MagicMock()
        actor.rl_config = MagicMock()
        actor.generate_config = MagicMock()
        actor.model_provider = MagicMock()
        actor.get_megatron_module = MagicMock()
        actor.initialize_func = MagicMock()
        actor.tokenizer = MagicMock()
        actor.kwargs = {}
        yield actor

    def test_create_actor_handlers_patch(self, test_actor):
        with patch("agentic_rl.trainer.train_adapter.mindspeed_rl.patch.launcher."
                   "ActorHandlerParams", MockActorHandlerParams), \
                patch("uuid.uuid4") as uuid_patch:
            uuid_value = MagicMock()
            uuid_value.hex = "e0a1b2c3d4"
            uuid_patch.return_value = uuid_value

            param = MockActorHandlerParams("localhost", 8000, 4, 0, "placement_group", 0)

            from agentic_rl.trainer.train_adapter.mindspeed_rl.patch.launcher import create_actor_handlers_patch

            test_actor = create_actor_handlers_patch(test_actor, param)

            assert test_actor.name == "MockActor_0_0_e0a1b2c3d4"
            assert isinstance(test_actor.scheduling_strategy, PlacementGroupSchedulingStrategy)
            assert test_actor.runtime_env == {
                "env_vars": {
                    "MASTER_ADDR": "localhost",
                    "MASTER_PORT": "8000",
                    "WORLD_SIZE": "4",
                    "RANK": "0",
                }
            }

    def test_create_actor_handlers_patch_failed_with_invalid_master_addr(self, test_actor):
        with patch("agentic_rl.trainer.train_adapter.mindspeed_rl.patch.launcher."
                   "ActorHandlerParams", MockActorHandlerParams):
            param = MockActorHandlerParams("abcabc", 8000, 4, 0, "placement_group", 0)

            from agentic_rl.trainer.train_adapter.mindspeed_rl.patch.launcher import create_actor_handlers_patch

            with pytest.raises(ValueError, match="master addr must be localhost or 127.0.0.1"):
                create_actor_handlers_patch(test_actor, param)

    def test_create_actor_handlers_patch_failed_with_invalid_master_port(self, test_actor):
        with patch("agentic_rl.trainer.train_adapter.mindspeed_rl.patch.launcher."
                   "ActorHandlerParams", MockActorHandlerParams):
            from agentic_rl.trainer.train_adapter.mindspeed_rl.patch.launcher import create_actor_handlers_patch
            param = MockActorHandlerParams("127.0.0.1", "8000", 4, 0, "placement_group", 0)
            with pytest.raises(ValueError, match="master port for create worker must be an integer"):
                create_actor_handlers_patch(test_actor, param)

            param = MockActorHandlerParams("localhost", 99999, 4, 0, "placement_group", 0)
            with pytest.raises(ValueError, match=re.escape("master port must be in range [1, 65535]")):
                create_actor_handlers_patch(test_actor, param)

    def test_create_actor_handlers_patch_failed_with_invalid_world_size(self, test_actor):
        with patch("agentic_rl.trainer.train_adapter.mindspeed_rl.patch.launcher."
                   "ActorHandlerParams", MockActorHandlerParams):
            from agentic_rl.trainer.train_adapter.mindspeed_rl.patch.launcher import create_actor_handlers_patch

            param = MockActorHandlerParams("127.0.0.1", None, 16, 0, "placement_group", 0)
            with pytest.raises(ValueError, match=re.escape("world size must be in range [1, 8]")):
                create_actor_handlers_patch(test_actor, param)

            param = MockActorHandlerParams("127.0.0.1", None, 8, 18, "placement_group", 0)
            with pytest.raises(ValueError, match=re.escape("rank index must within range [0, world_size)")):
                create_actor_handlers_patch(test_actor, param)

    def test_create_actor_handlers_patch_failed_with_worker(self, test_actor):
        with patch("agentic_rl.trainer.train_adapter.mindspeed_rl.patch.launcher."
                   "ActorHandlerParams", MockActorHandlerParams), \
                patch("uuid.uuid4") as uuid_patch:
            uuid_value = MagicMock()
            uuid_value.hex = "e0a1b2c3d4"
            uuid_patch.return_value = uuid_value

            param = MockActorHandlerParams("localhost", 8000, 4, 0, "placement_group", 0)

            from agentic_rl.trainer.train_adapter.mindspeed_rl.patch.launcher import create_actor_handlers_patch

            test_actor.worker.options = MagicMock()

            test_actor.worker.options.side_effect = RayError("test")
            with pytest.raises(RayError, match="create actor failed"):
                create_actor_handlers_patch(test_actor, param)

            test_actor.worker.options.side_effect = ValueError("test")
            with pytest.raises(RuntimeError, match="Unexpected error occurred when create actor"):
                create_actor_handlers_patch(test_actor, param)
