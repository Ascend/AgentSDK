# -*- coding: utf-8 -*-
import pytest
from unittest.mock import MagicMock, patch


class TestMegatronUtils:

    def test_parse_training_config_with_integrated_worker(self):
        config = {
            "megatron_training": {"key1": "value1"},
            "actor_config": {"key2": "value2"},
            "rl_config": {"use_integrated_worker": True},
            "generate_config": {},
            "agentic_env_config": {},
            "model": {"type": "test"}
        }

        with patch('aura.trainer.train_adapter.mindspeed_rl.utils.megatron_utils.ExtendMegatronConfig') as mock_megatron_config:
            mock_megatron_config.return_value = MagicMock()
            with patch('aura.trainer.train_adapter.mindspeed_rl.utils.megatron_utils.ExtendedRLConfig') as mock_rl_config:
                mock_rl_config.return_value.use_integrated_worker = True
                with patch('aura.trainer.train_adapter.mindspeed_rl.utils.megatron_utils.ExtendedGenerateConfig') as mock_generate_config:
                    mock_generate_config.return_value = MagicMock()
                    with patch('aura.trainer.train_adapter.mindspeed_rl.utils.megatron_utils.AgenticEnvConfig') as mock_agentic_config:
                        mock_agentic_config.return_value = MagicMock()
                        with patch('aura.trainer.train_adapter.mindspeed_rl.utils.megatron_utils.validate_agent_rl_args') as mock_validate:
                            from aura.trainer.train_adapter.mindspeed_rl.utils.megatron_utils import parse_training_config

                            result = parse_training_config(config)

                            assert "actor_config" in result
                            assert "ref_config" in result
                            assert "reward_config" in result
                            assert result["ref_config"] is result["actor_config"]
                            assert result["reward_config"] is result["actor_config"]

    def test_parse_training_config_without_integrated_worker(self):
        config = {
            "megatron_training": {"key1": "value1"},
            "actor_config": {"key2": "value2"},
            "ref_config": {"key3": "value3"},
            "reward_config": {"key4": "value4"},
            "rl_config": {"use_integrated_worker": False},
            "generate_config": {},
            "agentic_env_config": {},
            "model": {"type": "test"}
        }

        with patch('aura.trainer.train_adapter.mindspeed_rl.utils.megatron_utils.ExtendMegatronConfig') as mock_megatron_config:
            mock_megatron_config.return_value = MagicMock()
            with patch('aura.trainer.train_adapter.mindspeed_rl.utils.megatron_utils.ExtendedRLConfig') as mock_rl_config:
                mock_rl_config.return_value.use_integrated_worker = False
                with patch('aura.trainer.train_adapter.mindspeed_rl.utils.megatron_utils.ExtendedGenerateConfig') as mock_generate_config:
                    mock_generate_config.return_value = MagicMock()
                    with patch('aura.trainer.train_adapter.mindspeed_rl.utils.megatron_utils.AgenticEnvConfig') as mock_agentic_config:
                        mock_agentic_config.return_value = MagicMock()
                        with patch('aura.trainer.train_adapter.mindspeed_rl.utils.megatron_utils.validate_agent_rl_args') as mock_validate:
                            from aura.trainer.train_adapter.mindspeed_rl.utils.megatron_utils import parse_training_config

                            result = parse_training_config(config)

                            assert "actor_config" in result
                            assert "ref_config" in result
                            assert "reward_config" in result
                            assert mock_megatron_config.call_count == 3

    def test_parse_training_config_ref_config_error(self):
        config = {
            "megatron_training": {"key1": "value1"},
            "actor_config": {"key2": "value2"},
            "ref_config": {"key3": "value3"},
            "rl_config": {"use_integrated_worker": True},
            "generate_config": {},
            "agentic_env_config": {},
            "model": {"type": "test"}
        }

        with patch('aura.trainer.train_adapter.mindspeed_rl.utils.megatron_utils.ExtendedRLConfig') as mock_rl_config:
            mock_rl_config.return_value.use_integrated_worker = True
            from aura.trainer.train_adapter.mindspeed_rl.utils.megatron_utils import parse_training_config

            with pytest.raises(ValueError, match="ref_config should not be set"):
                parse_training_config(config)

    def test_parse_training_config_reward_config_error(self):
        config = {
            "megatron_training": {"key1": "value1"},
            "actor_config": {"key2": "value2"},
            "reward_config": {"key3": "value3"},
            "rl_config": {"use_integrated_worker": True},
            "generate_config": {},
            "agentic_env_config": {},
            "model": {"type": "test"}
        }

        with patch('aura.trainer.train_adapter.mindspeed_rl.utils.megatron_utils.ExtendedRLConfig') as mock_rl_config:
            mock_rl_config.return_value.use_integrated_worker = True
            from aura.trainer.train_adapter.mindspeed_rl.utils.megatron_utils import parse_training_config

            with pytest.raises(ValueError, match="reward_config should not be set"):
                parse_training_config(config)

    def test_parse_training_config_with_profiler(self):
        config = {
            "megatron_training": {"key1": "value1"},
            "actor_config": {"key2": "value2"},
            "rl_config": {"use_integrated_worker": False, "max_prompt_length": 512},
            "ref_config": {"key3": "value3"},
            "reward_config": {"key4": "value4"},
            "generate_config": {},
            "agentic_env_config": {},
            "profiler_config": {"integrated": {"enabled": True}},
            "msprobe_config": {"enabled": True},
            "model": {"type": "test"}
        }

        mock_megatron_config = MagicMock()
        mock_rl_config = MagicMock()
        mock_rl_config.use_integrated_worker = False
        mock_rl_config.max_prompt_length = 512

        with patch('aura.trainer.train_adapter.mindspeed_rl.utils.megatron_utils.ExtendMegatronConfig', return_value=mock_megatron_config):
            with patch('aura.trainer.train_adapter.mindspeed_rl.utils.megatron_utils.ExtendedRLConfig', return_value=mock_rl_config):
                with patch('aura.trainer.train_adapter.mindspeed_rl.utils.megatron_utils.ExtendedGenerateConfig'):
                    with patch('aura.trainer.train_adapter.mindspeed_rl.utils.megatron_utils.AgenticEnvConfig'):
                        with patch('aura.trainer.train_adapter.mindspeed_rl.utils.megatron_utils.validate_agent_rl_args'):
                            with patch('aura.trainer.train_adapter.mindspeed_rl.utils.megatron_utils.ProfilerConfig'):
                                with patch('aura.trainer.train_adapter.mindspeed_rl.utils.megatron_utils.MsprobeConfig'):
                                    from aura.trainer.train_adapter.mindspeed_rl.utils.megatron_utils import parse_training_config

                                    result = parse_training_config(config)

                                    assert "profiler_config" in result
                                    assert "msprobe_config" in result
                                    assert mock_megatron_config.max_prompt_length == 512

    def test_parse_training_config_empty_profiler(self):
        config = {
            "megatron_training": {"key1": "value1"},
            "actor_config": {"key2": "value2"},
            "rl_config": {"use_integrated_worker": False, "max_prompt_length": 512},
            "ref_config": {"key3": "value3"},
            "reward_config": {"key4": "value4"},
            "generate_config": {},
            "agentic_env_config": {},
            "model": {"type": "test"}
        }

        mock_megatron_config = MagicMock()
        mock_rl_config = MagicMock()
        mock_rl_config.use_integrated_worker = False
        mock_rl_config.max_prompt_length = 512

        with patch('aura.trainer.train_adapter.mindspeed_rl.utils.megatron_utils.ExtendMegatronConfig', return_value=mock_megatron_config):
            with patch('aura.trainer.train_adapter.mindspeed_rl.utils.megatron_utils.ExtendedRLConfig', return_value=mock_rl_config):
                with patch('aura.trainer.train_adapter.mindspeed_rl.utils.megatron_utils.ExtendedGenerateConfig'):
                    with patch('aura.trainer.train_adapter.mindspeed_rl.utils.megatron_utils.AgenticEnvConfig'):
                        with patch('aura.trainer.train_adapter.mindspeed_rl.utils.megatron_utils.validate_agent_rl_args'):
                            with patch('aura.trainer.train_adapter.mindspeed_rl.utils.megatron_utils.ProfilerConfig'):
                                with patch('aura.trainer.train_adapter.mindspeed_rl.utils.megatron_utils.MsprobeConfig'):
                                    from aura.trainer.train_adapter.mindspeed_rl.utils.megatron_utils import parse_training_config

                                    result = parse_training_config(config)

                                    assert "profiler_config" in result
                                    assert "msprobe_config" in result

    def test_parse_training_config_missing_required_keys(self):
        config = {
            "megatron_training": {"key1": "value1"},
        }

        with pytest.raises((KeyError, TypeError)):
            from aura.trainer.train_adapter.mindspeed_rl.utils.megatron_utils import parse_training_config
            parse_training_config(config)

    def test_parse_training_config_with_model_type(self):
        config = {
            "megatron_training": {"key1": "value1"},
            "actor_config": {"key2": "value2"},
            "rl_config": {"use_integrated_worker": False},
            "ref_config": {"key3": "value3"},
            "reward_config": {"key4": "value4"},
            "generate_config": {},
            "agentic_env_config": {},
            "model": {"type": "gpt"}
        }

        mock_megatron_config = MagicMock()
        mock_rl_config = MagicMock()
        mock_rl_config.use_integrated_worker = False
        mock_rl_config.max_prompt_length = 512

        with patch('aura.trainer.train_adapter.mindspeed_rl.utils.megatron_utils.ExtendMegatronConfig', return_value=mock_megatron_config) as mock_extend:
            with patch('aura.trainer.train_adapter.mindspeed_rl.utils.megatron_utils.ExtendedRLConfig', return_value=mock_rl_config):
                with patch('aura.trainer.train_adapter.mindspeed_rl.utils.megatron_utils.ExtendedGenerateConfig'):
                    with patch('aura.trainer.train_adapter.mindspeed_rl.utils.megatron_utils.AgenticEnvConfig'):
                        with patch('aura.trainer.train_adapter.mindspeed_rl.utils.megatron_utils.validate_agent_rl_args'):
                            with patch('aura.trainer.train_adapter.mindspeed_rl.utils.megatron_utils.ProfilerConfig'):
                                with patch('aura.trainer.train_adapter.mindspeed_rl.utils.megatron_utils.MsprobeConfig'):
                                    from aura.trainer.train_adapter.mindspeed_rl.utils.megatron_utils import parse_training_config

                                    result = parse_training_config(config)

                                    assert "actor_config" in result
                                    assert mock_extend.called

    def test_parse_training_config_with_different_max_prompt_length(self):
        config = {
            "megatron_training": {"key1": "value1"},
            "actor_config": {"key2": "value2"},
            "rl_config": {"use_integrated_worker": False, "max_prompt_length": 1024},
            "ref_config": {"key3": "value3"},
            "reward_config": {"key4": "value4"},
            "generate_config": {},
            "agentic_env_config": {},
            "model": {"type": "test"}
        }

        mock_megatron_config = MagicMock()
        mock_rl_config = MagicMock()
        mock_rl_config.use_integrated_worker = False
        mock_rl_config.max_prompt_length = 1024

        with patch('aura.trainer.train_adapter.mindspeed_rl.utils.megatron_utils.ExtendMegatronConfig', return_value=mock_megatron_config):
            with patch('aura.trainer.train_adapter.mindspeed_rl.utils.megatron_utils.ExtendedRLConfig', return_value=mock_rl_config):
                with patch('aura.trainer.train_adapter.mindspeed_rl.utils.megatron_utils.ExtendedGenerateConfig'):
                    with patch('aura.trainer.train_adapter.mindspeed_rl.utils.megatron_utils.AgenticEnvConfig'):
                        with patch('aura.trainer.train_adapter.mindspeed_rl.utils.megatron_utils.validate_agent_rl_args'):
                            with patch('aura.trainer.train_adapter.mindspeed_rl.utils.megatron_utils.ProfilerConfig'):
                                with patch('aura.trainer.train_adapter.mindspeed_rl.utils.megatron_utils.MsprobeConfig'):
                                    from aura.trainer.train_adapter.mindspeed_rl.utils.megatron_utils import parse_training_config

                                    result = parse_training_config(config)

                                    assert mock_megatron_config.max_prompt_length == 1024

    def test_parse_training_config_integration_worker_ref_reward_same(self):
        config = {
            "megatron_training": {"key1": "value1"},
            "actor_config": {"key2": "value2"},
            "rl_config": {"use_integrated_worker": True, "max_prompt_length": 512},
            "generate_config": {},
            "agentic_env_config": {},
            "model": {"type": "test"}
        }

        mock_megatron_config = MagicMock()
        mock_rl_config = MagicMock()
        mock_rl_config.use_integrated_worker = True
        mock_rl_config.max_prompt_length = 512

        with patch('aura.trainer.train_adapter.mindspeed_rl.utils.megatron_utils.ExtendMegatronConfig', return_value=mock_megatron_config):
            with patch('aura.trainer.train_adapter.mindspeed_rl.utils.megatron_utils.ExtendedRLConfig', return_value=mock_rl_config):
                with patch('aura.trainer.train_adapter.mindspeed_rl.utils.megatron_utils.ExtendedGenerateConfig'):
                    with patch('aura.trainer.train_adapter.mindspeed_rl.utils.megatron_utils.AgenticEnvConfig'):
                        with patch('aura.trainer.train_adapter.mindspeed_rl.utils.megatron_utils.validate_agent_rl_args'):
                            with patch('aura.trainer.train_adapter.mindspeed_rl.utils.megatron_utils.ProfilerConfig'):
                                with patch('aura.trainer.train_adapter.mindspeed_rl.utils.megatron_utils.MsprobeConfig'):
                                    from aura.trainer.train_adapter.mindspeed_rl.utils.megatron_utils import parse_training_config

                                    result = parse_training_config(config)

                                    assert result["ref_config"] is result["reward_config"]
                                    assert result["ref_config"] is result["actor_config"]

    def test_get_megatron_module(self):
        mock_parallel_state = MagicMock()
        mock_get_model = MagicMock()
        mock_get_megatron_optimizer = MagicMock()
        mock_load_checkpoint = MagicMock()
        mock_save_checkpoint = MagicMock()
        mock_get_args = MagicMock()
        mock_get_forward_backward_func = MagicMock()
        mock_float16_module = MagicMock()
        mock_unwrap_model = MagicMock()
        mock_distributed_data_parallel = MagicMock()
        mock_distributed_data_parallel_config = MagicMock()
        mock_vocab_parallel_cross_entropy = MagicMock()
        mock_setup_model_and_optimizer = MagicMock()
        mock_model_type = MagicMock()
        mock_finalize_model_grads = MagicMock()
        mock_set_actual_seq_len = MagicMock()
        mock_get_actual_seq_len = MagicMock()
        mock_set_position_ids = MagicMock()
        mock_distributed_optimizer = MagicMock()
        mock_float16_optimizer_with_float16_params = MagicMock()

        with patch.dict('sys.modules', {
            'megatron': MagicMock(),
            'megatron.core': MagicMock(),
            'megatron.core.parallel_state': mock_parallel_state,
            'megatron.core.DistributedDataParallel': mock_distributed_data_parallel,
            'megatron.core.optimizer': MagicMock(),
            'megatron.training': MagicMock(),
            'megatron.training.checkpointing': MagicMock(),
            'megatron.training.training': MagicMock(),
            'megatron.core.pipeline_parallel': MagicMock(),
            'megatron.legacy': MagicMock(),
            'megatron.legacy.model': MagicMock(),
            'megatron.core.distributed': MagicMock(),
            'megatron.core.distributed.distributed_data_parallel_config': MagicMock(),
            'megatron.core.tensor_parallel': MagicMock(),
            'megatron.core.tensor_parallel.cross_entropy': MagicMock(),
            'megatron.core.enums': MagicMock(),
            'mindspeed': MagicMock(),
            'mindspeed.utils': MagicMock(),
            'megatron.core.optimizer.distrib_optimizer': MagicMock(),
            'megatron.core.optimizer.optimizer': MagicMock(),
        }):
            from aura.trainer.train_adapter.mindspeed_rl.utils.megatron_utils import get_megatron_module
            result = get_megatron_module()

            assert isinstance(result, dict)
            assert 'parallel_state' in result
            assert 'get_model' in result
            assert 'get_megatron_optimizer' in result
            assert 'load_checkpoint' in result
            assert 'save_checkpoint' in result
            assert 'get_args' in result
            assert 'get_forward_backward_func' in result
            assert 'vocab_parallel_cross_entropy' in result
            assert 'setup_model_and_optimizer' in result
            assert 'model_type' in result
            assert 'distributed_data_parallel' in result
            assert 'finalize_model_grads' in result
            assert 'set_actual_seq_len' in result
            assert 'get_actual_seq_len' in result
            assert 'set_position_ids' in result
            assert 'distributed_optimizer' in result
            assert 'float16_optimizer_with_float16_params' in result

    def test_gpt_model_provider_with_mocks(self):
        # Setup module mocks before importing the function
        mock_args = MagicMock()
        mock_args.spec = None
        mock_args.num_experts = 1
        mock_args.moe_grouped_gemm = False
        mock_args.qk_layernorm = False
        mock_args.padded_vocab_size = 4096
        mock_args.max_position_embeddings = 4096
        mock_args.fp16_lm_cross_entropy = False
        mock_args.untie_embeddings_and_output_weights = False
        mock_args.position_embedding_type = 'absolute'
        mock_args.rotary_percent = 1.0
        mock_args.rotary_seq_len_interpolation_factor = None

        mock_gpt_model = MagicMock()
        mock_config = MagicMock()

        # Create mock modules
        mock_megatron = MagicMock()
        mock_megatron.training = MagicMock()
        mock_megatron.training.get_args = MagicMock(return_value=mock_args)

        mock_megatron_core = MagicMock()
        mock_megatron_core.models = MagicMock()
        mock_megatron_core.models.gpt = MagicMock()
        mock_megatron_core.models.gpt.GPTModel = MagicMock(return_value=mock_gpt_model)
        mock_megatron_core.models.gpt.gpt_layer_specs = MagicMock()
        mock_megatron_core.models.gpt.gpt_layer_specs.get_gpt_layer_local_spec = MagicMock()

        mock_transformer_spec = MagicMock()
        mock_megatron_core.transformer = MagicMock()
        mock_megatron_core.transformer.spec_utils = mock_transformer_spec

        mock_training_args = MagicMock()
        mock_training_args.core_transformer_config_from_args = MagicMock(return_value=mock_config)
        mock_megatron.training.arguments = mock_training_args

        # Mock sys.modules
        module_mocks = {
            'megatron': mock_megatron,
            'megatron.core': mock_megatron_core,
            'megatron.core.models': mock_megatron_core.models,
            'megatron.core.models.gpt': mock_megatron_core.models.gpt,
            'megatron.core.models.gpt.gpt_layer_specs': mock_megatron_core.models.gpt.gpt_layer_specs,
            'megatron.core.transformer': mock_megatron_core.transformer,
            'megatron.core.transformer.spec_utils': mock_transformer_spec,
            'megatron.training': mock_megatron.training,
            'megatron.training.arguments': mock_training_args,
        }

        with patch.dict('sys.modules', module_mocks):
            from aura.trainer.train_adapter.mindspeed_rl.utils.megatron_utils import gpt_model_provider
            result = gpt_model_provider(pre_process=True, post_process=True)
            assert result is not None

    def test_gpt_model_provider_with_spec(self):
        # Setup module mocks before importing the function
        mock_args = MagicMock()
        mock_args.spec = 'test.module.spec'
        mock_args.num_experts = 1
        mock_args.moe_grouped_gemm = False
        mock_args.qk_layernorm = False
        mock_args.padded_vocab_size = 4096
        mock_args.max_position_embeddings = 4096
        mock_args.fp16_lm_cross_entropy = False
        mock_args.untie_embeddings_and_output_weights = False
        mock_args.position_embedding_type = 'absolute'
        mock_args.rotary_percent = 1.0
        mock_args.rotary_seq_len_interpolation_factor = None

        mock_gpt_model = MagicMock()
        mock_config = MagicMock()
        mock_import_result = MagicMock()

        # Create mock modules
        mock_megatron = MagicMock()
        mock_megatron.training = MagicMock()
        mock_megatron.training.get_args = MagicMock(return_value=mock_args)

        mock_megatron_core = MagicMock()
        mock_megatron_core.models = MagicMock()
        mock_megatron_core.models.gpt = MagicMock()
        mock_megatron_core.models.gpt.GPTModel = MagicMock(return_value=mock_gpt_model)
        mock_megatron_core.models.gpt.gpt_layer_specs = MagicMock()

        mock_transformer_spec = MagicMock()
        mock_transformer_spec.import_module = MagicMock(return_value=mock_import_result)
        mock_megatron_core.transformer = MagicMock()
        mock_megatron_core.transformer.spec_utils = mock_transformer_spec

        mock_training_args = MagicMock()
        mock_training_args.core_transformer_config_from_args = MagicMock(return_value=mock_config)
        mock_megatron.training.arguments = mock_training_args

        # Mock sys.modules
        module_mocks = {
            'megatron': mock_megatron,
            'megatron.core': mock_megatron_core,
            'megatron.core.models': mock_megatron_core.models,
            'megatron.core.models.gpt': mock_megatron_core.models.gpt,
            'megatron.core.models.gpt.gpt_layer_specs': mock_megatron_core.models.gpt.gpt_layer_specs,
            'megatron.core.transformer': mock_megatron_core.transformer,
            'megatron.core.transformer.spec_utils': mock_transformer_spec,
            'megatron.training': mock_megatron.training,
            'megatron.training.arguments': mock_training_args,
        }

        with patch.dict('sys.modules', module_mocks):
            from aura.trainer.train_adapter.mindspeed_rl.utils.megatron_utils import gpt_model_provider
            result = gpt_model_provider(pre_process=False, post_process=False)
            assert result is not None

    def test_rm_model_provider(self):
        # Setup module mocks before importing the function
        mock_args = MagicMock()
        mock_args.spec = None
        mock_args.num_experts = 1
        mock_args.moe_grouped_gemm = False
        mock_args.qk_layernorm = False
        mock_args.padded_vocab_size = 4096
        mock_args.max_position_embeddings = 4096
        mock_args.fp16_lm_cross_entropy = False
        mock_args.untie_embeddings_and_output_weights = False
        mock_args.position_embedding_type = 'absolute'
        mock_args.rotary_percent = 1.0
        mock_args.no_post_layer_norm = False
        mock_args.pipeline_model_parallel_size = 1

        mock_rm_model = MagicMock()
        mock_config = MagicMock()

        # Create mock modules
        mock_megatron = MagicMock()
        mock_megatron.training = MagicMock()
        mock_megatron.training.get_args = MagicMock(return_value=mock_args)

        mock_megatron_core = MagicMock()
        mock_megatron_core.models = MagicMock()
        mock_megatron_core.models.gpt = MagicMock()
        mock_megatron_core.models.gpt.gpt_layer_specs = MagicMock()
        mock_megatron_core.models.gpt.gpt_layer_specs.get_gpt_layer_local_spec = MagicMock()

        mock_transformer_spec = MagicMock()
        mock_megatron_core.transformer = MagicMock()
        mock_megatron_core.transformer.spec_utils = mock_transformer_spec

        mock_training_args = MagicMock()
        mock_training_args.core_transformer_config_from_args = MagicMock(return_value=mock_config)
        mock_megatron.training.arguments = mock_training_args

        # Mock mindspeed_rl module
        mock_mindspeed_rl = MagicMock()
        mock_mindspeed_rl.tasks = MagicMock()
        mock_mindspeed_rl.tasks.posttrain = MagicMock()
        mock_mindspeed_rl.tasks.posttrain.orm = MagicMock()
        mock_mindspeed_rl.tasks.posttrain.orm.orm_model = MagicMock()
        mock_mindspeed_rl.tasks.posttrain.orm.orm_model.GPTRewardModel = MagicMock(return_value=mock_rm_model)

        # Mock sys.modules
        module_mocks = {
            'megatron': mock_megatron,
            'megatron.core': mock_megatron_core,
            'megatron.core.models': mock_megatron_core.models,
            'megatron.core.models.gpt': mock_megatron_core.models.gpt,
            'megatron.core.models.gpt.gpt_layer_specs': mock_megatron_core.models.gpt.gpt_layer_specs,
            'megatron.core.transformer': mock_megatron_core.transformer,
            'megatron.core.transformer.spec_utils': mock_transformer_spec,
            'megatron.training': mock_megatron.training,
            'megatron.training.arguments': mock_training_args,
            'mindspeed_llm': mock_mindspeed_rl,
            'mindspeed_llm.tasks': mock_mindspeed_rl.tasks,
            'mindspeed_llm.tasks.posttrain': mock_mindspeed_rl.tasks.posttrain,
            'mindspeed_llm.tasks.posttrain.orm': mock_mindspeed_rl.tasks.posttrain.orm,
            'mindspeed_llm.tasks.posttrain.orm.orm_model': mock_mindspeed_rl.tasks.posttrain.orm.orm_model,
        }

        with patch.dict('sys.modules', module_mocks):
            from aura.trainer.train_adapter.mindspeed_rl.utils.megatron_utils import rm_model_provider
            result = rm_model_provider(pre_process=True, post_process=True)
            assert result is not None

    def test_rm_model_provider_with_pipeline_parallel(self):
        # Test the special branch where pipeline_model_parallel_size > 1
        mock_args = MagicMock()
        mock_args.spec = None
        mock_args.num_experts = 1
        mock_args.moe_grouped_gemm = False
        mock_args.qk_layernorm = False
        mock_args.padded_vocab_size = 4096
        mock_args.max_position_embeddings = 4096
        mock_args.fp16_lm_cross_entropy = False
        mock_args.untie_embeddings_and_output_weights = False
        mock_args.position_embedding_type = 'absolute'
        mock_args.rotary_percent = 1.0
        mock_args.no_post_layer_norm = False
        mock_args.pipeline_model_parallel_size = 2  # > 1 to trigger the branch

        mock_rm_model = MagicMock()
        mock_config = MagicMock()

        mock_megatron = MagicMock()
        mock_megatron.training = MagicMock()
        mock_megatron.training.get_args = MagicMock(return_value=mock_args)

        mock_megatron_core = MagicMock()
        mock_megatron_core.models = MagicMock()
        mock_megatron_core.models.gpt = MagicMock()
        mock_megatron_core.models.gpt.gpt_layer_specs = MagicMock()
        mock_megatron_core.models.gpt.gpt_layer_specs.get_gpt_layer_local_spec = MagicMock()

        mock_transformer_spec = MagicMock()
        mock_megatron_core.transformer = MagicMock()
        mock_megatron_core.transformer.spec_utils = mock_transformer_spec

        mock_training_args = MagicMock()
        mock_training_args.core_transformer_config_from_args = MagicMock(return_value=mock_config)
        mock_megatron.training.arguments = mock_training_args

        mock_mindspeed_rl = MagicMock()
        mock_mindspeed_rl.tasks = MagicMock()
        mock_mindspeed_rl.tasks.posttrain = MagicMock()
        mock_mindspeed_rl.tasks.posttrain.orm = MagicMock()
        mock_mindspeed_rl.tasks.posttrain.orm.orm_model = MagicMock()
        mock_mindspeed_rl.tasks.posttrain.orm.orm_model.GPTRewardModel = MagicMock(return_value=mock_rm_model)

        module_mocks = {
            'megatron': mock_megatron,
            'megatron.core': mock_megatron_core,
            'megatron.core.models': mock_megatron_core.models,
            'megatron.core.models.gpt': mock_megatron_core.models.gpt,
            'megatron.core.models.gpt.gpt_layer_specs': mock_megatron_core.models.gpt.gpt_layer_specs,
            'megatron.core.transformer': mock_megatron_core.transformer,
            'megatron.core.transformer.spec_utils': mock_transformer_spec,
            'megatron.training': mock_megatron.training,
            'megatron.training.arguments': mock_training_args,
            'mindspeed_llm': mock_mindspeed_rl,
            'mindspeed_llm.tasks': mock_mindspeed_rl.tasks,
            'mindspeed_llm.tasks.posttrain': mock_mindspeed_rl.tasks.posttrain,
            'mindspeed_llm.tasks.posttrain.orm': mock_mindspeed_rl.tasks.posttrain.orm,
            'mindspeed_llm.tasks.posttrain.orm.orm_model': mock_mindspeed_rl.tasks.posttrain.orm.orm_model,
        }

        with patch.dict('sys.modules', module_mocks):
            from aura.trainer.train_adapter.mindspeed_rl.utils.megatron_utils import rm_model_provider
            result = rm_model_provider(pre_process=True, post_process=True)
            assert result is not None
            # Verify untie_embeddings_and_output_weights was set to True
            assert mock_args.untie_embeddings_and_output_weights is True

    def test_gpt_model_provider_with_pre_post_false(self):
        # Test gpt_model_provider with pre_process=False, post_process=False
        mock_args = MagicMock()
        mock_args.spec = None
        mock_args.num_experts = 1
        mock_args.moe_grouped_gemm = False
        mock_args.qk_layernorm = False
        mock_args.padded_vocab_size = 4096
        mock_args.max_position_embeddings = 4096
        mock_args.fp16_lm_cross_entropy = False
        mock_args.untie_embeddings_and_output_weights = False
        mock_args.position_embedding_type = 'absolute'
        mock_args.rotary_percent = 1.0
        mock_args.rotary_seq_len_interpolation_factor = None

        mock_gpt_model = MagicMock()
        mock_config = MagicMock()

        mock_megatron = MagicMock()
        mock_megatron.training = MagicMock()
        mock_megatron.training.get_args = MagicMock(return_value=mock_args)

        mock_megatron_core = MagicMock()
        mock_megatron_core.models = MagicMock()
        mock_megatron_core.models.gpt = MagicMock()
        mock_megatron_core.models.gpt.GPTModel = MagicMock(return_value=mock_gpt_model)
        mock_megatron_core.models.gpt.gpt_layer_specs = MagicMock()
        mock_megatron_core.models.gpt.gpt_layer_specs.get_gpt_layer_local_spec = MagicMock()

        mock_transformer_spec = MagicMock()
        mock_megatron_core.transformer = MagicMock()
        mock_megatron_core.transformer.spec_utils = mock_transformer_spec

        mock_training_args = MagicMock()
        mock_training_args.core_transformer_config_from_args = MagicMock(return_value=mock_config)
        mock_megatron.training.arguments = mock_training_args

        module_mocks = {
            'megatron': mock_megatron,
            'megatron.core': mock_megatron_core,
            'megatron.core.models': mock_megatron_core.models,
            'megatron.core.models.gpt': mock_megatron_core.models.gpt,
            'megatron.core.models.gpt.gpt_layer_specs': mock_megatron_core.models.gpt.gpt_layer_specs,
            'megatron.core.transformer': mock_megatron_core.transformer,
            'megatron.core.transformer.spec_utils': mock_transformer_spec,
            'megatron.training': mock_megatron.training,
            'megatron.training.arguments': mock_training_args,
        }

        with patch.dict('sys.modules', module_mocks):
            from aura.trainer.train_adapter.mindspeed_rl.utils.megatron_utils import gpt_model_provider
            result = gpt_model_provider(pre_process=False, post_process=False)
            assert result is not None

    def test_rm_model_provider_with_spec(self):
        # Test rm_model_provider with spec parameter
        mock_args = MagicMock()
        mock_args.spec = 'test.spec.module'
        mock_args.num_experts = 1
        mock_args.moe_grouped_gemm = False
        mock_args.qk_layernorm = False
        mock_args.padded_vocab_size = 4096
        mock_args.max_position_embeddings = 4096
        mock_args.fp16_lm_cross_entropy = False
        mock_args.untie_embeddings_and_output_weights = False
        mock_args.position_embedding_type = 'absolute'
        mock_args.rotary_percent = 1.0
        mock_args.no_post_layer_norm = False
        mock_args.pipeline_model_parallel_size = 1

        mock_rm_model = MagicMock()
        mock_config = MagicMock()
        mock_import_result = MagicMock()

        mock_megatron = MagicMock()
        mock_megatron.training = MagicMock()
        mock_megatron.training.get_args = MagicMock(return_value=mock_args)

        mock_megatron_core = MagicMock()
        mock_megatron_core.models = MagicMock()
        mock_megatron_core.models.gpt = MagicMock()
        mock_megatron_core.models.gpt.gpt_layer_specs = MagicMock()

        mock_transformer_spec = MagicMock()
        mock_transformer_spec.import_module = MagicMock(return_value=mock_import_result)
        mock_megatron_core.transformer = MagicMock()
        mock_megatron_core.transformer.spec_utils = mock_transformer_spec

        mock_training_args = MagicMock()
        mock_training_args.core_transformer_config_from_args = MagicMock(return_value=mock_config)
        mock_megatron.training.arguments = mock_training_args

        mock_mindspeed_llm = MagicMock()
        mock_mindspeed_llm.tasks = MagicMock()
        mock_mindspeed_llm.tasks.posttrain = MagicMock()
        mock_mindspeed_llm.tasks.posttrain.orm = MagicMock()
        mock_mindspeed_llm.tasks.posttrain.orm.orm_model = MagicMock()
        mock_mindspeed_llm.tasks.posttrain.orm.orm_model.GPTRewardModel = MagicMock(return_value=mock_rm_model)

        module_mocks = {
            'megatron': mock_megatron,
            'megatron.core': mock_megatron_core,
            'megatron.core.models': mock_megatron_core.models,
            'megatron.core.models.gpt': mock_megatron_core.models.gpt,
            'megatron.core.models.gpt.gpt_layer_specs': mock_megatron_core.models.gpt.gpt_layer_specs,
            'megatron.core.transformer': mock_megatron_core.transformer,
            'megatron.core.transformer.spec_utils': mock_transformer_spec,
            'megatron.training': mock_megatron.training,
            'megatron.training.arguments': mock_training_args,
            'mindspeed_llm': mock_mindspeed_llm,
            'mindspeed_llm.tasks': mock_mindspeed_llm.tasks,
            'mindspeed_llm.tasks.posttrain': mock_mindspeed_llm.tasks.posttrain,
            'mindspeed_llm.tasks.posttrain.orm': mock_mindspeed_llm.tasks.posttrain.orm,
            'mindspeed_llm.tasks.posttrain.orm.orm_model': mock_mindspeed_llm.tasks.posttrain.orm.orm_model,
        }

        with patch.dict('sys.modules', module_mocks):
            from aura.trainer.train_adapter.mindspeed_rl.utils.megatron_utils import rm_model_provider
            result = rm_model_provider(pre_process=True, post_process=True)
            assert result is not None
            mock_transformer_spec.import_module.assert_called_once_with('test.spec.module')

    def test_initialize_megatron_lazy_mpu(self):
        # Test initialize_megatron with lazy_mpu_init=True
        mock_args = MagicMock()
        mock_args.use_checkpoint_args = False
        mock_args.lazy_mpu_init = True
        mock_args.tp_comm_overlap = False
        mock_args.use_deter_comp = False
        mock_args.seed = 42
        mock_args.data_parallel_random_init = False
        mock_args.tensor_model_parallel_size = 1
        mock_args.rank = 0

        mock_parse_args_from_config = MagicMock()
        mock_init_torch_compile = MagicMock()
        mock_validate_args = MagicMock()
        mock_set_global_variables = MagicMock()
        mock_initialize_coc_from_cfg = MagicMock()
        mock_load_args_from_checkpoint = MagicMock()

        mock_set_tensor_model_parallel_world_size = MagicMock()
        mock_set_tensor_model_parallel_rank = MagicMock()

        mock_megatron = MagicMock()
        mock_megatron.training = MagicMock()
        mock_megatron.training.arguments = MagicMock()
        mock_megatron.training.arguments.parse_args = MagicMock(return_value=mock_args)
        mock_megatron.training.arguments.validate_args = mock_validate_args
        mock_megatron.training.global_vars = MagicMock()
        mock_megatron.training.global_vars.set_global_variables = mock_set_global_variables
        mock_megatron.training.get_args = MagicMock(return_value=mock_args)
        mock_megatron.training.checkpointing = MagicMock()
        mock_megatron.training.checkpointing.load_args_from_checkpoint = mock_load_args_from_checkpoint
        mock_megatron.training.initialize = MagicMock()

        mock_megatron_core = MagicMock()
        mock_megatron_core.parallel_state = MagicMock()
        mock_megatron_core.parallel_state.set_tensor_model_parallel_world_size = mock_set_tensor_model_parallel_world_size
        mock_megatron_core.parallel_state.set_tensor_model_parallel_rank = mock_set_tensor_model_parallel_rank
        mock_megatron.core = mock_megatron_core

        mock_mindspeed_llm = MagicMock()
        mock_mindspeed_llm.training = MagicMock()
        mock_mindspeed_llm.training.arguments = MagicMock()
        mock_mindspeed_llm.training.arguments.parse_args_decorator = MagicMock(return_value=MagicMock(return_value=mock_args))

        mock_mindspeed_core = MagicMock()
        mock_mindspeed_core.tensor_parallel = MagicMock()
        mock_mindspeed_core.tensor_parallel.lcal_coc = MagicMock()
        mock_mindspeed_core.tensor_parallel.lcal_coc.user_config = MagicMock()
        mock_mindspeed_core.tensor_parallel.lcal_coc.user_config.initialize_coc_from_cfg = mock_initialize_coc_from_cfg

        mock_module_utils = MagicMock()
        mock_module_utils.parse_args_from_config = mock_parse_args_from_config
        mock_module_utils.init_torch_compile = mock_init_torch_compile

        module_mocks = {
            'megatron': mock_megatron,
            'megatron.core': mock_megatron_core,
            'megatron.core.parallel_state': mock_megatron_core.parallel_state,
            'megatron.training': mock_megatron.training,
            'megatron.training.arguments': mock_megatron.training.arguments,
            'megatron.training.global_vars': mock_megatron.training.global_vars,
            'megatron.training.checkpointing': mock_megatron.training.checkpointing,
            'megatron.training.initialize': mock_megatron.training.initialize,
            'mindspeed_llm': mock_mindspeed_llm,
            'mindspeed_llm.training': mock_mindspeed_llm.training,
            'mindspeed_llm.training.arguments': mock_mindspeed_llm.training.arguments,
            'mindspeed.core': mock_mindspeed_core,
            'mindspeed.core.tensor_parallel': mock_mindspeed_core.tensor_parallel,
            'mindspeed.core.tensor_parallel.lcal_coc': mock_mindspeed_core.tensor_parallel.lcal_coc,
            'mindspeed.core.tensor_parallel.lcal_coc.user_config': mock_mindspeed_core.tensor_parallel.lcal_coc.user_config,
            'mindspeed_rl.utils.utils': mock_module_utils,
        }

        with patch.dict('sys.modules', module_mocks):
            from aura.trainer.train_adapter.mindspeed_rl.utils.megatron_utils import initialize_megatron
            result = initialize_megatron(allow_no_cuda=True, config={})
            assert result is not None
            assert callable(result)
            mock_set_tensor_model_parallel_world_size.assert_called_once_with(1)
            mock_set_tensor_model_parallel_rank.assert_called_once_with(0)

    def test_initialize_megatron_checkpoint_args(self):
        # Test initialize_megatron with use_checkpoint_args=True
        mock_args = MagicMock()
        mock_args.use_checkpoint_args = True
        mock_args.load = '/path/to/checkpoint'
        mock_args.lazy_mpu_init = False
        mock_args.tp_comm_overlap = False
        mock_args.use_deter_comp = False
        mock_args.seed = 42
        mock_args.data_parallel_random_init = False
        mock_args.tensor_model_parallel_size = 1
        mock_args.rank = 0

        mock_parse_args_from_config = MagicMock()
        mock_init_torch_compile = MagicMock()
        mock_validate_args = MagicMock()
        mock_set_global_variables = MagicMock()
        mock_initialize_coc_from_cfg = MagicMock()
        mock_load_args_from_checkpoint = MagicMock()

        mock_megatron = MagicMock()
        mock_megatron.training = MagicMock()
        mock_megatron.training.arguments = MagicMock()
        mock_megatron.training.arguments.parse_args = MagicMock(return_value=mock_args)
        mock_megatron.training.arguments.validate_args = mock_validate_args
        mock_megatron.training.global_vars = MagicMock()
        mock_megatron.training.global_vars.set_global_variables = mock_set_global_variables
        mock_megatron.training.get_args = MagicMock(return_value=mock_args)
        mock_megatron.training.checkpointing = MagicMock()
        mock_megatron.training.checkpointing.load_args_from_checkpoint = mock_load_args_from_checkpoint
        mock_megatron.training.initialize = MagicMock()

        mock_megatron_core = MagicMock()
        mock_megatron_core.parallel_state = MagicMock()
        mock_megatron.core = mock_megatron_core

        mock_mindspeed_llm = MagicMock()
        mock_mindspeed_llm.training = MagicMock()
        mock_mindspeed_llm.training.arguments = MagicMock()
        mock_mindspeed_llm.training.arguments.parse_args_decorator = MagicMock(return_value=MagicMock(return_value=mock_args))

        mock_mindspeed_core = MagicMock()
        mock_mindspeed_core.tensor_parallel = MagicMock()
        mock_mindspeed_core.tensor_parallel.lcal_coc = MagicMock()
        mock_mindspeed_core.tensor_parallel.lcal_coc.user_config = MagicMock()
        mock_mindspeed_core.tensor_parallel.lcal_coc.user_config.initialize_coc_from_cfg = mock_initialize_coc_from_cfg

        mock_module_utils = MagicMock()
        mock_module_utils.parse_args_from_config = mock_parse_args_from_config
        mock_module_utils.init_torch_compile = mock_init_torch_compile

        module_mocks = {
            'megatron': mock_megatron,
            'megatron.core': mock_megatron_core,
            'megatron.core.parallel_state': mock_megatron_core.parallel_state,
            'megatron.training': mock_megatron.training,
            'megatron.training.arguments': mock_megatron.training.arguments,
            'megatron.training.global_vars': mock_megatron.training.global_vars,
            'megatron.training.checkpointing': mock_megatron.training.checkpointing,
            'megatron.training.initialize': mock_megatron.training.initialize,
            'mindspeed_llm': mock_mindspeed_llm,
            'mindspeed_llm.training': mock_mindspeed_llm.training,
            'mindspeed_llm.training.arguments': mock_mindspeed_llm.training.arguments,
            'mindspeed.core': mock_mindspeed_core,
            'mindspeed.core.tensor_parallel': mock_mindspeed_core.tensor_parallel,
            'mindspeed.core.tensor_parallel.lcal_coc': mock_mindspeed_core.tensor_parallel.lcal_coc,
            'mindspeed.core.tensor_parallel.lcal_coc.user_config': mock_mindspeed_core.tensor_parallel.lcal_coc.user_config,
            'mindspeed_rl.utils.utils': mock_module_utils,
        }

        with patch.dict('sys.modules', module_mocks):
            with patch('aura.trainer.train_adapter.mindspeed_rl.utils.megatron_utils._initialize_distributed'):
                from aura.trainer.train_adapter.mindspeed_rl.utils.megatron_utils import initialize_megatron
                initialize_megatron(allow_no_cuda=True, config={})
                mock_load_args_from_checkpoint.assert_called_once_with(mock_args)

    def test_initialize_megatron_no_cuda_error(self):
        # Test initialize_megatron raises error when CUDA not available and allow_no_cuda=False
        mock_args = MagicMock()
        mock_args.use_checkpoint_args = False
        mock_args.lazy_mpu_init = False
        mock_args.tp_comm_overlap = False
        mock_args.use_deter_comp = False

        mock_parse_args_from_config = MagicMock()
        mock_init_torch_compile = MagicMock()

        mock_megatron = MagicMock()
        mock_megatron.training = MagicMock()
        mock_megatron.training.arguments = MagicMock()
        mock_megatron.training.arguments.parse_args = MagicMock(return_value=mock_args)

        mock_mindspeed_llm = MagicMock()
        mock_mindspeed_llm.training = MagicMock()
        mock_mindspeed_llm.training.arguments = MagicMock()
        mock_mindspeed_llm.training.arguments.parse_args_decorator = MagicMock(return_value=MagicMock(return_value=mock_args))

        mock_module_utils = MagicMock()
        mock_module_utils.parse_args_from_config = mock_parse_args_from_config
        mock_module_utils.init_torch_compile = mock_init_torch_compile

        module_mocks = {
            'megatron': mock_megatron,
            'megatron.training': mock_megatron.training,
            'megatron.training.arguments': mock_megatron.training.arguments,
            'mindspeed_llm': mock_mindspeed_llm,
            'mindspeed_llm.training': mock_mindspeed_llm.training,
            'mindspeed_llm.training.arguments': mock_mindspeed_llm.training.arguments,
            'mindspeed_rl.utils.utils': mock_module_utils,
        }

        with patch.dict('sys.modules', module_mocks):
            with patch('torch.cuda.is_available', return_value=False):
                from aura.trainer.train_adapter.mindspeed_rl.utils.megatron_utils import initialize_megatron
                with pytest.raises(ValueError, match="Megatron requires CUDA"):
                    initialize_megatron(allow_no_cuda=False, config={})

    def test_initialize_megatron_checkpoint_args_missing_load(self):
        # Test initialize_megatron raises error when use_checkpoint_args=True but load is None
        mock_args = MagicMock()
        mock_args.use_checkpoint_args = True
        mock_args.load = None

        mock_parse_args_from_config = MagicMock()
        mock_init_torch_compile = MagicMock()
        mock_load_args_from_checkpoint = MagicMock()
        mock_set_global_variables = MagicMock()
        mock_validate_args = MagicMock()

        mock_megatron = MagicMock()
        mock_megatron.training = MagicMock()
        mock_megatron.training.arguments = MagicMock()
        mock_megatron.training.arguments.parse_args = MagicMock(return_value=mock_args)
        mock_megatron.training.arguments.validate_args = mock_validate_args
        mock_megatron.training.checkpointing = MagicMock()
        mock_megatron.training.checkpointing.load_args_from_checkpoint = mock_load_args_from_checkpoint
        mock_megatron.training.global_vars = MagicMock()
        mock_megatron.training.global_vars.set_global_variables = mock_set_global_variables
        mock_megatron.training.get_args = MagicMock(return_value=mock_args)
        mock_megatron.training.initialize = MagicMock()
        mock_megatron.core = MagicMock()
        mock_megatron.core.parallel_state = MagicMock()

        mock_mindspeed_llm = MagicMock()
        mock_mindspeed_llm.training = MagicMock()
        mock_mindspeed_llm.training.arguments = MagicMock()
        mock_mindspeed_llm.training.arguments.parse_args_decorator = MagicMock(return_value=MagicMock(return_value=mock_args))

        mock_module_utils = MagicMock()
        mock_module_utils.parse_args_from_config = mock_parse_args_from_config
        mock_module_utils.init_torch_compile = mock_init_torch_compile

        module_mocks = {
            'megatron': mock_megatron,
            'megatron.core': mock_megatron.core,
            'megatron.core.parallel_state': mock_megatron.core.parallel_state,
            'megatron.training': mock_megatron.training,
            'megatron.training.arguments': mock_megatron.training.arguments,
            'megatron.training.checkpointing': mock_megatron.training.checkpointing,
            'megatron.training.global_vars': mock_megatron.training.global_vars,
            'megatron.training.initialize': mock_megatron.training.initialize,
            'mindspeed_llm': mock_mindspeed_llm,
            'mindspeed_llm.training': mock_mindspeed_llm.training,
            'mindspeed_llm.training.arguments': mock_mindspeed_llm.training.arguments,
            'mindspeed_rl.utils.utils': mock_module_utils,
        }

        with patch.dict('sys.modules', module_mocks):
            from aura.trainer.train_adapter.mindspeed_rl.utils.megatron_utils import initialize_megatron
            with pytest.raises(ValueError, match="--use-checkpoints-args requires --load argument"):
                initialize_megatron(allow_no_cuda=True, config={})
