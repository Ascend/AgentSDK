# -*- coding: utf-8 -*-
import pytest
from unittest.mock import MagicMock, patch


class TestExtendMegatronConfig:

    def test_default_config(self):
        from aura.trainer.train_adapter.mindspeed_rl.config_cls.extend_megatron_config import ExtendMegatronConfig
        config = ExtendMegatronConfig({}, {})
        assert config.test_data_path is None
        assert config.train_data_path is None
        assert config.swap_optimizer_times == 4
        assert config.fix_router is False

    def test_custom_config(self):
        from aura.trainer.train_adapter.mindspeed_rl.config_cls.extend_megatron_config import ExtendMegatronConfig
        config = ExtendMegatronConfig({"test_data_path": "/path/to/test"}, {})
        assert config.test_data_path == "/path/to/test"
