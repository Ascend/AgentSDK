# -*- coding: utf-8 -*-
import pytest
from unittest.mock import MagicMock


class TestValidateConfig:

    def test_validate_agent_rl_args(self):
        from aura.trainer.train_adapter.mindspeed_rl.config_cls.validate_config import validate_agent_rl_args

        actor_config = MagicMock()
        ref_config = MagicMock()
        reward_config = MagicMock()
        rl_config = MagicMock()
        generate_config = MagicMock()
        agentic_config = MagicMock()

        result = validate_agent_rl_args(actor_config, ref_config, reward_config, rl_config, generate_config, agentic_config)
        assert result is None

    def test_validate_agent_rl_args_no_api_key(self):
        from aura.trainer.train_adapter.mindspeed_rl.config_cls.validate_config import validate_agent_rl_args

        actor_config = MagicMock()
        ref_config = MagicMock()
        reward_config = MagicMock()
        rl_config = MagicMock()
        generate_config = MagicMock()
        agentic_config = MagicMock()

        result = validate_agent_rl_args(actor_config, ref_config, reward_config, rl_config, generate_config, agentic_config)
        assert result is None
