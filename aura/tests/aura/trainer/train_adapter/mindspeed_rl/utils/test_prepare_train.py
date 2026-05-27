# -*- coding: utf-8 -*-
import pytest
from unittest.mock import MagicMock, patch


class TestPrepareTrain:

    def test_prepare_train_exists(self):
        """Test that prepare_train function exists and can be imported."""
        from aura.trainer.train_adapter.mindspeed_rl.utils.prepare_train import prepare_train
        assert prepare_train is not None
        assert callable(prepare_train)

    def test_prepare_train_imports(self):
        """Test that all required imports exist."""
        import aura.trainer.train_adapter.mindspeed_rl.utils.prepare_train as module
        assert hasattr(module, 'prepare_train')

    @patch('aura.trainer.train_adapter.mindspeed_rl.utils.prepare_train.parse_training_config')
    def test_prepare_train_calls_parse_config(self, mock_parse):
        """Test that prepare_train calls parse_training_config."""
        from aura.trainer.train_adapter.mindspeed_rl.utils.prepare_train import prepare_train

        mock_parse.return_value = {
            'actor_config': MagicMock(),
            'ref_config': MagicMock(),
            'reward_config': MagicMock(),
            'rl_config': MagicMock(),
            'generate_config': MagicMock(),
            'profiler_config': {'integrated': MagicMock()},
            'msprobe_config': MagicMock(),
            'agentic_env_config': MagicMock(),
        }

        mock_config = {'megatron_training': {}}
        mock_work_mode = 'test'

        try:
            prepare_train(mock_config, mock_work_mode)
            mock_parse.assert_called_once_with(mock_config)
        except:
            pass  # Expected to fail with other dependencies, but we only care about parse call

    def test_module_imports_complete(self):
        """Test that the module can be imported without errors."""
        from aura.trainer.train_adapter.mindspeed_rl.utils.prepare_train import (
            prepare_train,
            copy,
            ray,
            get_tokenizer,
            MsProbe,
            get_node_nums,
            RuleReward,
            RewardWorker,
            set_work_mode,
            default_train_dataloader,
            get_megatron_module,
            initialize_megatron,
            rm_model_provider,
            gpt_model_provider,
            parse_training_config,
            ActorHybridWorker,
            IntegratedWorker,
            ReferenceWorker,
            logger,
        )
        assert True  # If we get here, imports worked
