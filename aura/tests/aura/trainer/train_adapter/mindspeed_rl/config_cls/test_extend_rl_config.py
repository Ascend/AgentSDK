# -*- coding: utf-8 -*-
import pytest
from unittest.mock import MagicMock, patch


class TestExtendedRLConfig:

    def test_default_config(self):
        from aura.trainer.train_adapter.mindspeed_rl.config_cls.extend_rl_config import ExtendedRLConfig
        config = ExtendedRLConfig({})
        assert config.use_stepwise_advantage is False
        assert config.validate_freq == 10
        assert config.validate_num_samples == 100
