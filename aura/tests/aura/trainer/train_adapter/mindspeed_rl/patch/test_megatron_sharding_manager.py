# -*- coding: utf-8 -*-
import pytest
from unittest.mock import MagicMock, patch


class TestMegatronShardingManager:

    def test_enter_infer_mode_patch_exists(self):
        from aura.trainer.train_adapter.mindspeed_rl.patch.megatron_sharding_manager import enter_infer_mode_patch

        assert callable(enter_infer_mode_patch)

    def test_exit_infer_mode_patch_exists(self):
        from aura.trainer.train_adapter.mindspeed_rl.patch.megatron_sharding_manager import exit_infer_mode_patch

        assert callable(exit_infer_mode_patch)

    def test_exit_train_mode_patch_exists(self):
        from aura.trainer.train_adapter.mindspeed_rl.patch.megatron_sharding_manager import exit_train_mode_patch

        assert callable(exit_train_mode_patch)

    def test_logger_exists(self):
        from aura.trainer.train_adapter.mindspeed_rl.patch.megatron_sharding_manager import logger

        assert logger is not None

    def test_module_imports(self):
        import torch
        import gc
        from mindspeed_rl.utils.utils import mstx_timer_decorator
        from aura.base.log.loggers import Loggers

        assert torch is not None
        assert gc is not None
        assert mstx_timer_decorator is not None
        assert Loggers is not None
