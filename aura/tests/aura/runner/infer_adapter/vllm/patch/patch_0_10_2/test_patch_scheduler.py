#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# -------------------------------------------------------------------------
#  This file is part of the AgentSDK project.
# Copyright (c) 2026 Huawei Technologies Co.,Ltd.
#
# AgentSDK is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#
#           http://license.coscl.org.cn/MulanPSL2
#
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
# -------------------------------------------------------------------------
import sys
import types
from unittest.mock import MagicMock, patch
import pytest


# ---------------------------------------------------------------------------
# Fixture: fake module tree for patch_scheduler
# ---------------------------------------------------------------------------
@pytest.fixture
def fake_scheduler_env():
    # ---- Fake vllm.config ----
    fake_vllm_config = types.ModuleType("vllm.config")
    fake_vllm_config.VllmConfig = MagicMock

    # ---- Fake vllm.multimodal ----
    fake_vllm_multimodal = types.ModuleType("vllm.multimodal")
    fake_vllm_multimodal.MULTIMODAL_REGISTRY = object()
    fake_vllm_multimodal.MultiModalRegistry = MagicMock

    # ---- Fake vllm.v1.core.sched.output ----
    fake_vllm_v1_core_sched_output = types.ModuleType("vllm.v1.core.sched.output")
    fake_vllm_v1_core_sched_output.SchedulerOutput = MagicMock

    # ---- Fake vllm.v1.core.sched.utils ----
    fake_vllm_v1_core_sched_utils = types.ModuleType("vllm.v1.core.sched.utils")
    fake_vllm_v1_core_sched_utils.check_stop = MagicMock()

    # ---- Fake vllm.v1.engine ----
    fake_vllm_v1_engine = types.ModuleType("vllm.v1.engine")
    class EngineCoreEventType:
        QUEUED = "queued"
    fake_vllm_v1_engine.EngineCoreEventType = EngineCoreEventType

    # ---- Fake vllm.v1.kv_cache_interface ----
    fake_vllm_v1_kv_cache_interface = types.ModuleType("vllm.v1.kv_cache_interface")
    fake_vllm_v1_kv_cache_interface.KVCacheConfig = MagicMock

    # ---- Fake vllm.v1.request ----
    fake_vllm_v1_request = types.ModuleType("vllm.v1.request")
    fake_vllm_v1_request.Request = MagicMock

    # ---- Fake vllm.v1.structured_output ----
    fake_vllm_v1_structured_output = types.ModuleType("vllm.v1.structured_output")
    fake_vllm_v1_structured_output.StructuredOutputManager = MagicMock

    # ---- Fake vllm.v1.core.sched.scheduler ----
    fake_vllm_v1_core_sched_scheduler = types.ModuleType("vllm.v1.core.sched.scheduler")

    class FakeScheduler:
        @staticmethod
        def __init__(self, *args, **kwargs):
            pass

        @staticmethod
        def _update_after_schedule(self, scheduler_output):
            pass

        @staticmethod
        def _update_request_with_output(self, request, new_token_ids):
            pass

        @staticmethod
        def add_request(self, request):
            pass

        @staticmethod
        def reset_prefix_cache(self):
            pass

        @staticmethod
        def _update_from_kv_xfer_finished(self, kv_connector_output):
            pass

    FakeScheduler.__init__ = MagicMock(return_value=None)
    FakeScheduler._update_after_schedule = MagicMock()
    FakeScheduler._update_request_with_output = MagicMock(return_value=(None, False))
    FakeScheduler.add_request = MagicMock()
    FakeScheduler.reset_prefix_cache = MagicMock(return_value=True)
    FakeScheduler._update_from_kv_xfer_finished = MagicMock()

    fake_vllm_v1_core_sched_scheduler.Scheduler = FakeScheduler

    # ---- Fake vllm.v1.worker.kv_connector_model_runner_mixin ----
    fake_vllm_v1_worker_kv = types.ModuleType("vllm.v1.worker.kv_connector_model_runner_mixin")
    fake_vllm_v1_worker_kv.KVConnectorOutput = MagicMock

    # ---- Fake vllm.logger ----
    fake_vllm_logger = types.ModuleType("vllm.logger")
    fake_vllm_logger.logger = MagicMock()

    # ---- Fake comm.scheduler_stat ----
    fake_scheduler_stat = types.ModuleType("aura.runner.infer_adapter.vllm.patch.comm.scheduler_stat")
    class RequestStats:
        def stat_schedule(self, request_id):
            pass

        def stat_finish(self, request_id, prompt_tokens, output_tokens):
            pass

        def stat_prefill_done(self, request_id):
            pass

        def stat_add(self, request_id):
            pass

        def print(self):
            pass
    RequestStats = MagicMock(return_value=MagicMock())
    fake_scheduler_stat.RequestStats = RequestStats

    # ---- Aura packages ----
    import os as _os
    import aura as _aura
    real_aura_path = _aura.__path__
    base_path = real_aura_path[0] if real_aura_path else "."
    fake_aura = types.ModuleType("aura")
    fake_aura.__path__ = real_aura_path
    fake_aura_runner = types.ModuleType("aura.runner")
    fake_aura_runner.__path__ = [_os.path.join(base_path, "runner")]
    fake_aura_runner_infer_adapter = types.ModuleType("aura.runner.infer_adapter")
    fake_aura_runner_infer_adapter.__path__ = [_os.path.join(base_path, "runner/infer_adapter")]
    fake_vllm_pkg = types.ModuleType("aura.runner.infer_adapter.vllm")
    fake_vllm_pkg.__path__ = [_os.path.join(base_path, "runner/infer_adapter/vllm")]
    fake_patch_pkg = types.ModuleType("aura.runner.infer_adapter.vllm.patch")
    fake_patch_pkg.__path__ = [_os.path.join(base_path, "runner/infer_adapter/vllm/patch")]
    fake_0_10_2_pkg = types.ModuleType("aura.runner.infer_adapter.vllm.patch.patch_0_10_2")
    fake_0_10_2_pkg.__path__ = [_os.path.join(base_path, "runner/infer_adapter/vllm/patch/patch_0_10_2")]
    fake_comm_pkg = types.ModuleType("aura.runner.infer_adapter.vllm.patch.comm")
    fake_comm_pkg.__path__ = []

    fakes = {
        "vllm.config": fake_vllm_config,
        "vllm.multimodal": fake_vllm_multimodal,
        "vllm.v1.core.sched.output": fake_vllm_v1_core_sched_output,
        "vllm.v1.core.sched.utils": fake_vllm_v1_core_sched_utils,
        "vllm.v1.engine": fake_vllm_v1_engine,
        "vllm.v1.kv_cache_interface": fake_vllm_v1_kv_cache_interface,
        "vllm.v1.request": fake_vllm_v1_request,
        "vllm.v1.structured_output": fake_vllm_v1_structured_output,
        "vllm.v1.core.sched.scheduler": fake_vllm_v1_core_sched_scheduler,
        "vllm.v1.worker.kv_connector_model_runner_mixin": fake_vllm_v1_worker_kv,
        "vllm.logger": fake_vllm_logger,
        "aura.runner.infer_adapter.vllm.patch.comm.scheduler_stat": fake_scheduler_stat,
        "aura": fake_aura,
        "aura.runner": fake_aura_runner,
        "aura.runner.infer_adapter": fake_aura_runner_infer_adapter,
        "aura.runner.infer_adapter.vllm": fake_vllm_pkg,
        "aura.runner.infer_adapter.vllm.patch": fake_patch_pkg,
        "aura.runner.infer_adapter.vllm.patch.patch_0_10_2": fake_0_10_2_pkg,
        "aura.runner.infer_adapter.vllm.patch.comm": fake_comm_pkg,
    }

    yield {
        "fakes": fakes,
        "Scheduler": FakeScheduler,
        "check_stop": fake_vllm_v1_core_sched_utils.check_stop,
        "RequestStats": fake_scheduler_stat.RequestStats,
        "logger": fake_vllm_logger.logger,
        "EngineCoreEventType": EngineCoreEventType,
    }


# ---------------------------------------------------------------------------
# Helper: import the module under test with fake modules injected
# ---------------------------------------------------------------------------
def import_module(fake_scheduler_env):
    module_name = "aura.runner.infer_adapter.vllm.patch.patch_0_10_2.patch_scheduler"
    if module_name in sys.modules:
        del sys.modules[module_name]

    fakes = fake_scheduler_env["fakes"]
    with patch.dict(sys.modules, fakes):
        import aura.runner.infer_adapter.vllm.patch.patch_0_10_2.patch_scheduler as mod
    return mod


# ---------------------------------------------------------------------------
# Helpers to create mock objects for self and arguments
# ---------------------------------------------------------------------------
def make_self_mock():
    """Create a mock Scheduler instance with common attributes."""
    self_mock = MagicMock()
    self_mock.requests = {}
    self_mock.req_stats = MagicMock()  # will be set by scheduler_init
    self_mock.max_model_len = 100
    self_mock.waiting = MagicMock()
    self_mock.log_stats = False
    self_mock.connector = None
    self_mock.finished_recving_kv_req_ids = set()
    self_mock.kv_cache_manager = MagicMock()
    self_mock._free_blocks = MagicMock()
    self_mock._free_encoder_inputs = MagicMock()
    return self_mock


# ---------------------------------------------------------------------------
# Tests for scheduler_init
# ---------------------------------------------------------------------------
class TestSchedulerInit:
    @pytest.fixture(autouse=True)
    def setup(self, fake_scheduler_env):
        self.mod = import_module(fake_scheduler_env)
        self.env = fake_scheduler_env

    def test_init_calls_original_and_sets_req_stats(self):
        """scheduler_init calls original __init__ and sets req_stats and is_prefill."""
        Scheduler = self.env["Scheduler"]
        original_init_mock = self.mod.original_scheduler_init
        assert isinstance(original_init_mock, MagicMock)

        self_mock = make_self_mock()
        vllm_config = MagicMock()
        vllm_config.kv_transfer_config = None
        kv_cache_config = MagicMock()
        structured_output_manager = MagicMock()

        self.mod.scheduler_init(self_mock, vllm_config, kv_cache_config, structured_output_manager)

        original_init_mock.assert_called_once_with(
            self_mock, vllm_config, kv_cache_config, structured_output_manager,
            self.env["fakes"]["vllm.multimodal"].MULTIMODAL_REGISTRY, False, False
        )
        assert isinstance(self_mock.req_stats, MagicMock)
        assert self_mock.is_prefill == False

    def test_init_with_kv_producer_sets_is_prefill_true(self):
        """is_prefill is True when kv_transfer_config.kv_role == 'kv_producer'."""
        self_mock = make_self_mock()
        vllm_config = MagicMock()
        vllm_config.kv_transfer_config = MagicMock()
        vllm_config.kv_transfer_config.kv_role = "kv_producer"

        self.mod.scheduler_init(self_mock, vllm_config, MagicMock(), MagicMock())
        assert self_mock.is_prefill == True

    def test_init_with_non_producer_role_sets_is_prefill_false(self):
        """is_prefill is False when kv_role is not 'kv_producer'."""
        self_mock = make_self_mock()
        vllm_config = MagicMock()
        vllm_config.kv_transfer_config = MagicMock()
        vllm_config.kv_transfer_config.kv_role = "kv_consumer"

        self.mod.scheduler_init(self_mock, vllm_config, MagicMock(), MagicMock())
        assert self_mock.is_prefill == False


# ---------------------------------------------------------------------------
# Tests for update_after_schedule_patch
# ---------------------------------------------------------------------------
class TestUpdateAfterSchedule:
    @pytest.fixture(autouse=True)
    def setup(self, fake_scheduler_env):
        self.mod = import_module(fake_scheduler_env)

    def test_basic_update_calls_stat_and_increments(self):
        """Update increments num_computed_tokens and calls stat_schedule when appropriate."""
        self_mock = make_self_mock()
        # Setup requests
        req1 = MagicMock()
        req1.request_id = "req1"
        req1.num_computed_tokens = 0
        req1.num_cached_tokens = 5
        req1.has_encoder_inputs = False

        req2 = MagicMock()
        req2.request_id = "req2"
        req2.num_computed_tokens = 5
        req2.num_cached_tokens = 5  # equal => should trigger stat
        req2.has_encoder_inputs = False

        self_mock.requests = {"req1": req1, "req2": req2}

        scheduler_output = MagicMock()
        scheduler_output.num_scheduled_tokens = {"req1": 3, "req2": 2}

        self.mod.update_after_schedule_patch(self_mock, scheduler_output)

        # req1: num_computed_tokens was 0 -> stat_schedule called
        self_mock.req_stats.stat_schedule.assert_any_call("req1")
        # req2: num_computed_tokens == num_cached_tokens -> stat_schedule called
        self_mock.req_stats.stat_schedule.assert_any_call("req2")
        assert req1.num_computed_tokens == 3
        assert req2.num_computed_tokens == 7

    def test_no_stat_when_not_initial_or_cached(self):
        """stat_schedule not called when num_computed_tokens != 0 and != num_cached_tokens."""
        self_mock = make_self_mock()
        req = MagicMock()
        req.request_id = "req"
        req.num_computed_tokens = 3
        req.num_cached_tokens = 5
        req.has_encoder_inputs = False
        self_mock.requests = {"req": req}

        scheduler_output = MagicMock()
        scheduler_output.num_scheduled_tokens = {"req": 1}

        self.mod.update_after_schedule_patch(self_mock, scheduler_output)
        self_mock.req_stats.stat_schedule.assert_not_called()

    def test_calls_free_encoder_inputs_when_needed(self):
        """_free_encoder_inputs is called when request has encoder inputs."""
        self_mock = make_self_mock()
        req = MagicMock()
        req.request_id = "req"
        req.num_computed_tokens = 0
        req.num_cached_tokens = 0
        req.has_encoder_inputs = True
        self_mock.requests = {"req": req}

        scheduler_output = MagicMock()
        scheduler_output.num_scheduled_tokens = {"req": 2}

        self.mod.update_after_schedule_patch(self_mock, scheduler_output)
        self_mock._free_encoder_inputs.assert_called_once_with(req)


# ---------------------------------------------------------------------------
# Tests for update_request_with_output_patch
# ---------------------------------------------------------------------------
class TestUpdateRequestWithOutput:
    @pytest.fixture(autouse=True)
    def setup(self, fake_scheduler_env):
        self.mod = import_module(fake_scheduler_env)
        self.check_stop = fake_scheduler_env["check_stop"]

    def test_normal_append_no_stop(self):
        """Appends tokens, no stop, triggers prefill_done when first token generated."""
        self_mock = make_self_mock()
        request = MagicMock()
        request.request_id = "req"
        request.num_output_tokens = 0
        request.num_prompt_tokens = 10
        request.append_output_token_ids = MagicMock()
        def inc_num(token):
            request.num_output_tokens += 1
        request.append_output_token_ids.side_effect = inc_num

        self.check_stop.return_value = False

        new_tokens, stopped = self.mod.update_request_with_output_patch(
            self_mock, request, [101]
        )

        assert new_tokens == [101]
        assert stopped == False
        request.append_output_token_ids.assert_called_once_with(101)
        self_mock.req_stats.stat_finish.assert_not_called()
        self_mock.req_stats.stat_prefill_done.assert_called_once_with("req")

    def test_stop_in_middle_trims_tokens(self):
        """When stop occurs, remaining tokens are removed and stat_finish called with correct counts."""
        self_mock = make_self_mock()
        request = MagicMock()
        request.request_id = "req"
        request.num_output_tokens = 0
        request.num_prompt_tokens = 10
        request.append_output_token_ids = MagicMock()
        def inc_num(token):
            request.num_output_tokens += 1
        request.append_output_token_ids.side_effect = inc_num

        # Make check_stop return True on the second token
        self.check_stop.side_effect = [False, True, False]

        new_tokens, stopped = self.mod.update_request_with_output_patch(
            self_mock, request, [201, 202, 203]
        )

        assert new_tokens == [201, 202]
        assert stopped == True
        self_mock.req_stats.stat_finish.assert_called_once_with("req", 10, 2)

    def test_prefill_done_not_called_when_output_tokens_not_one(self):
        """stat_prefill_done not called if initial num_output_tokens is not 0."""
        self_mock = make_self_mock()
        request = MagicMock()
        request.request_id = "req"
        request.num_output_tokens = 5
        request.num_prompt_tokens = 10
        request.append_output_token_ids = MagicMock()
        def inc_num(token):
            request.num_output_tokens += 1
        request.append_output_token_ids.side_effect = inc_num

        self.check_stop.return_value = False

        self.mod.update_request_with_output_patch(self_mock, request, [301])
        self_mock.req_stats.stat_prefill_done.assert_not_called()


# ---------------------------------------------------------------------------
# Tests for add_request_patch
# ---------------------------------------------------------------------------
class TestAddRequest:
    @pytest.fixture(autouse=True)
    def setup(self, fake_scheduler_env):
        self.mod = import_module(fake_scheduler_env)

    def test_add_request_log_stats_false(self):
        """add_request with log_stats=False does not record QUEUED event."""
        self_mock = make_self_mock()
        self_mock.log_stats = False
        request = MagicMock()
        request.request_id = "r1"

        self.mod.add_request_patch(self_mock, request)

        self_mock.waiting.add_request.assert_called_once_with(request)
        assert self_mock.requests["r1"] == request
        request.record_event.assert_not_called()
        self_mock.req_stats.stat_add.assert_called_once_with("r1")

    def test_add_request_log_stats_true(self):
        """add_request with log_stats=True records QUEUED event."""
        self_mock = make_self_mock()
        self_mock.log_stats = True
        request = MagicMock()
        request.request_id = "r2"

        self.mod.add_request_patch(self_mock, request)

        request.record_event.assert_called_once_with(
            self.mod.EngineCoreEventType.QUEUED
        )
        self_mock.req_stats.stat_add.assert_called_once_with("r2")


# ---------------------------------------------------------------------------
# Tests for reset_prefix_cache_patch
# ---------------------------------------------------------------------------
class TestResetPrefixCache:
    @pytest.fixture(autouse=True)
    def setup(self, fake_scheduler_env):
        self.mod = import_module(fake_scheduler_env)

    def test_reset_calls_print_and_delegates(self):
        """reset_prefix_cache prints stats and calls original kv_cache_manager method."""
        self_mock = make_self_mock()
        self_mock.req_stats.print = MagicMock()
        self_mock.kv_cache_manager.reset_prefix_cache = MagicMock(return_value="result")

        result = self.mod.reset_prefix_cache_patch(self_mock)

        self_mock.req_stats.print.assert_called_once()
        self_mock.kv_cache_manager.reset_prefix_cache.assert_called_once()
        assert result == "result"


# ---------------------------------------------------------------------------
# Tests for _update_from_kv_xfer_finished_patch
# ---------------------------------------------------------------------------
class TestUpdateFromKVXferFinished:
    @pytest.fixture(autouse=True)
    def setup(self, fake_scheduler_env):
        self.mod = import_module(fake_scheduler_env)
        self.logger = fake_scheduler_env["logger"]

    def test_connector_not_none_updates(self):
        """When connector is not None, update_connector_output is called."""
        self_mock = make_self_mock()
        connector = MagicMock()
        self_mock.connector = connector
        kv_output = MagicMock()
        kv_output.finished_recving = None
        kv_output.finished_sending = None

        self.mod._update_from_kv_xfer_finished_patch(self_mock, kv_output)

        connector.update_connector_output.assert_called_once_with(kv_output)

    def test_finished_recving_adds_to_set(self):
        """Finished recving requests are added to finished_recving_kv_req_ids."""
        self_mock = make_self_mock()
        self_mock.connector = None
        kv_output = MagicMock()
        kv_output.finished_recving = ["req_a", "req_b"]
        kv_output.finished_sending = None

        self.mod._update_from_kv_xfer_finished_patch(self_mock, kv_output)

        assert "req_a" in self_mock.finished_recving_kv_req_ids
        assert "req_b" in self_mock.finished_recving_kv_req_ids
        self.logger.debug.assert_called()

    def test_finished_sending_frees_blocks(self):
        """Finished sending requests trigger _free_blocks if request exists."""
        self_mock = make_self_mock()
        req = MagicMock()
        self_mock.requests = {"req_x": req}
        kv_output = MagicMock()
        kv_output.finished_recving = None
        kv_output.finished_sending = ["req_x"]

        self.mod._update_from_kv_xfer_finished_patch(self_mock, kv_output)

        self_mock._free_blocks.assert_called_once_with(req)

    def test_finished_sending_request_missing_warns(self):
        """When finished sending request is not in self.requests, a warning is logged."""
        self_mock = make_self_mock()
        self_mock.requests = {}
        kv_output = MagicMock()
        kv_output.finished_recving = None
        kv_output.finished_sending = ["req_missing"]

        self.mod._update_from_kv_xfer_finished_patch(self_mock, kv_output)

        self.logger.warning.assert_called_once()
        self_mock._free_blocks.assert_not_called()


# ---------------------------------------------------------------------------
# Tests for class assignment
# ---------------------------------------------------------------------------
class TestSchedulerPatched:
    """Verify that Scheduler methods are replaced by our patches."""
    @pytest.fixture(autouse=True)
    def setup(self, fake_scheduler_env):
        self.mod = import_module(fake_scheduler_env)
        self.Scheduler = fake_scheduler_env["Scheduler"]

    def test_scheduler_methods_assigned(self):
        """Check that Scheduler class methods point to our functions."""
        assert self.Scheduler.__init__ is self.mod.scheduler_init
        assert self.Scheduler._update_after_schedule is self.mod.update_after_schedule_patch
        assert self.Scheduler._update_request_with_output is self.mod.update_request_with_output_patch
        assert self.Scheduler.add_request is self.mod.add_request_patch
        assert self.Scheduler.reset_prefix_cache is self.mod.reset_prefix_cache_patch
        assert self.Scheduler._update_from_kv_xfer_finished is self.mod._update_from_kv_xfer_finished_patch
