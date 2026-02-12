import math
import pytest  # type: ignore[import-not-found]
from pydantic import ValidationError

from agentic_rl.trainer.train_adapter.schema import (
    BaseConfig,
    MindspeedRLConfig,
    VerlConfig,
    GlobalConfig,
)


def _make_valid_mindspeed_rl_config(**overrides):
    """Helper to create a valid MindspeedRLConfig."""
    config = {
        "data_path": "/path/to/data",
        "load_params_path": "/path/to/load",
        "save_params_path": "/path/to/save",
    }
    config.update(overrides)
    return config


def _make_valid_verl_config(**overrides):
    """Helper to create a valid VerlConfig."""
    config = {
        "train_files": "/path/to/train",
        "val_files": "/path/to/val",
    }
    config.update(overrides)
    return config


def _make_valid_global_config_with_verl(**overrides):
    """Helper to create a valid GlobalConfig with verl backend."""
    config = {
        "tokenizer_name_or_path": "/path/to/tokenizer",
        "model_name": "llama",
        "agent_name": "my_agent",
        "agent_engine_wrapper_path": "/path/to/wrapper",
        "train_backend": "verl",
        "verl": {
            "train_files": "/path/to/train",
            "val_files": "/path/to/val",
        },
    }
    config.update(overrides)
    return config


def _make_valid_global_config_with_mindspeed(**overrides):
    """Helper to create a valid GlobalConfig with mindspeed_rl backend."""
    config = {
        "tokenizer_name_or_path": "/path/to/tokenizer",
        "model_name": "llama",
        "agent_name": "my_agent",
        "agent_engine_wrapper_path": "/path/to/wrapper",
        "train_backend": "mindspeed_rl",
        "mindspeed_rl": {
            "data_path": "/path/to/data",
            "load_params_path": "/path/to/load",
            "save_params_path": "/path/to/save",
        },
    }
    config.update(overrides)
    return config


@pytest.fixture(autouse=True)
def _disable_file_checks(monkeypatch):
    """Disable filesystem checks for all tests."""
    monkeypatch.setattr(
        "agentic_rl.trainer.train_adapter.schema.FileCheck.check_data_path_is_valid",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        "os.path.exists",
        lambda *_args, **_kwargs: True,
    )


class TestBaseConfig:
    """Test BaseConfig behavior."""

    def test_extra_fields_forbidden(self):
        """Test that extra fields are rejected."""
        with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
            BaseConfig(unknown_field="value")


class TestMindspeedRLConfig:
    """Test MindspeedRLConfig validation."""
    
    def test_valid_config(self):
        """Test that a valid config is accepted."""
        config = MindspeedRLConfig(**_make_valid_mindspeed_rl_config())
        assert config.data_path == "/path/to/data"
        assert config.epochs == 1
        assert config.adv_estimator == "group_norm"

    def test_positive_validation_epochs(self):
        """Test that epochs must be positive."""
        with pytest.raises(ValidationError, match="must be positive"):
            MindspeedRLConfig(**_make_valid_mindspeed_rl_config(epochs=0))

        with pytest.raises(ValidationError, match="must be positive"):
            MindspeedRLConfig(**_make_valid_mindspeed_rl_config(epochs=-1))

    def test_positive_validation_seq_length(self):
        """Test that seq_length must be positive."""
        with pytest.raises(ValidationError, match="must be positive"):
            MindspeedRLConfig(**_make_valid_mindspeed_rl_config(seq_length=0))

    def test_positive_validation_global_batch_size(self):
        """Test that global_batch_size must be positive."""
        with pytest.raises(ValidationError, match="must be positive"):
            MindspeedRLConfig(**_make_valid_mindspeed_rl_config(global_batch_size=0))

    def test_positive_validation_save_interval(self):
        """Test that save_interval must be positive."""
        with pytest.raises(ValidationError, match="must be positive"):
            MindspeedRLConfig(**_make_valid_mindspeed_rl_config(save_interval=0))

    def test_positive_validation_train_iters(self):
        """Test that train_iters must be positive."""
        with pytest.raises(ValidationError, match="must be positive"):
            MindspeedRLConfig(**_make_valid_mindspeed_rl_config(train_iters=0))

    def test_positive_validation_mini_batch_size(self):
        """Test that mini_batch_size must be positive."""
        with pytest.raises(ValidationError, match="must be positive"):
            MindspeedRLConfig(**_make_valid_mindspeed_rl_config(mini_batch_size=0))

    def test_positive_validation_micro_batch_size(self):
        """Test that micro_batch_size must be positive."""
        with pytest.raises(ValidationError, match="must be positive"):
            MindspeedRLConfig(**_make_valid_mindspeed_rl_config(micro_batch_size=0))

    def test_positive_validation_tensor_model_parallel_size(self):
        """Test that tensor_model_parallel_size must be positive."""
        with pytest.raises(ValidationError, match="must be positive"):
            MindspeedRLConfig(**_make_valid_mindspeed_rl_config(tensor_model_parallel_size=0))

    def test_positive_validation_pipeline_model_parallel_size(self):
        """Test that pipeline_model_parallel_size must be positive."""
        with pytest.raises(ValidationError, match="must be positive"):
            MindspeedRLConfig(**_make_valid_mindspeed_rl_config(pipeline_model_parallel_size=0))

    def test_adv_estimator_literal_validation(self):
        """Test that adv_estimator only accepts valid literals."""
        with pytest.raises(ValidationError):
            MindspeedRLConfig(**_make_valid_mindspeed_rl_config(adv_estimator="invalid"))

        # Valid values should work
        config = MindspeedRLConfig(**_make_valid_mindspeed_rl_config(adv_estimator="gae"))
        assert config.adv_estimator == "gae"

    def test_extra_fields_rejected(self):
        """Test that extra fields are rejected."""
        with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
            MindspeedRLConfig(**_make_valid_mindspeed_rl_config(unknown_field="value"))


class TestVerlConfig:
    """Test VerlConfig validation."""

    def test_valid_config(self):
        """Test that a valid config is accepted."""
        config = VerlConfig(**_make_valid_verl_config())
        assert config.train_files == "/path/to/train"
        assert config.total_epochs == 2
        assert config.adv_estimator == "grpo"

    def test_positive_validation_total_epochs(self):
        """Test that total_epochs must be positive."""
        with pytest.raises(ValidationError, match="must be positive"):
            VerlConfig(**_make_valid_verl_config(total_epochs=0))

        with pytest.raises(ValidationError, match="must be positive"):
            VerlConfig(**_make_valid_verl_config(total_epochs=-1))

    def test_positive_validation_save_freq(self):
        """Test that save_freq must be positive."""
        with pytest.raises(ValidationError, match="must be positive"):
            VerlConfig(**_make_valid_verl_config(save_freq=0))

    def test_positive_validation_ppo_mini_batch_size(self):
        """Test that ppo_mini_batch_size must be positive."""
        with pytest.raises(ValidationError, match="must be positive"):
            VerlConfig(**_make_valid_verl_config(ppo_mini_batch_size=0))

    def test_positive_validation_ppo_max_token_len_per_gpu(self):
        """Test that ppo_max_token_len_per_gpu must be positive."""
        with pytest.raises(ValidationError, match="must be positive"):
            VerlConfig(**_make_valid_verl_config(ppo_max_token_len_per_gpu=0))

    def test_positive_validation_ppo_epochs(self):
        """Test that ppo_epochs must be positive."""
        with pytest.raises(ValidationError, match="must be positive"):
            VerlConfig(**_make_valid_verl_config(ppo_epochs=0))

    def test_positive_validation_max_response_length(self):
        """Test that max_response_length must be positive."""
        with pytest.raises(ValidationError, match="must be positive"):
            VerlConfig(**_make_valid_verl_config(max_response_length=0))

    def test_positive_validation_train_batch_size(self):
        """Test that train_batch_size must be positive."""
        with pytest.raises(ValidationError, match="must be positive"):
            VerlConfig(**_make_valid_verl_config(train_batch_size=0))

    def test_positive_validation_val_batch_size(self):
        """Test that val_batch_size must be positive."""
        with pytest.raises(ValidationError, match="must be positive"):
            VerlConfig(**_make_valid_verl_config(val_batch_size=0))

    def test_positive_validation_dataloader_num_workers(self):
        """Test that dataloader_num_workers must be positive."""
        with pytest.raises(ValidationError, match="must be positive"):
            VerlConfig(**_make_valid_verl_config(dataloader_num_workers=0))

    def test_positive_validation_nnodes(self):
        """Test that nnodes must be positive."""
        with pytest.raises(ValidationError, match="must be positive"):
            VerlConfig(**_make_valid_verl_config(nnodes=0))

    def test_positive_validation_grad_clip(self):
        """Test that grad_clip must be positive."""
        with pytest.raises(ValidationError, match="must be positive"):
            VerlConfig(**_make_valid_verl_config(grad_clip=0))

    def test_test_freq_validation(self):
        """Test that test_freq accepts -1 or positive values."""
        # -1 should be valid (disabled)
        config = VerlConfig(**_make_valid_verl_config(test_freq=-1))
        assert config.test_freq == -1

        # Positive values should be valid
        config = VerlConfig(**_make_valid_verl_config(test_freq=10))
        assert config.test_freq == 10

        # 0 and other negative values should be invalid
        with pytest.raises(ValidationError, match="must be -1 or positive"):
            VerlConfig(**_make_valid_verl_config(test_freq=0))

        with pytest.raises(ValidationError, match="must be -1 or positive"):
            VerlConfig(**_make_valid_verl_config(test_freq=-2))

    def test_fraction_validation_policy_loss_ppo_kl_coef(self):
        """Test that policy_loss_ppo_kl_coef must be between 0 and 1."""
        with pytest.raises(ValidationError, match="must be between 0 and 1"):
            VerlConfig(**_make_valid_verl_config(policy_loss_ppo_kl_coef=1.5))

        with pytest.raises(ValidationError, match="must be between 0 and 1"):
            VerlConfig(**_make_valid_verl_config(policy_loss_ppo_kl_coef=-0.1))

        # Boundary values should work
        config = VerlConfig(**_make_valid_verl_config(policy_loss_ppo_kl_coef=0.0))
        assert math.isclose(config.policy_loss_ppo_kl_coef, 0.0, rel_tol=1e-5) == True

        config = VerlConfig(**_make_valid_verl_config(policy_loss_ppo_kl_coef=1.0))
        assert math.isclose(config.policy_loss_ppo_kl_coef, 1.0, rel_tol=1e-5) == True

    def test_fraction_validation_kl_loss_coef(self):
        """Test that kl_loss_coef must be between 0 and 1."""
        with pytest.raises(ValidationError, match="must be between 0 and 1"):
            VerlConfig(**_make_valid_verl_config(kl_loss_coef=1.5))

        with pytest.raises(ValidationError, match="must be between 0 and 1"):
            VerlConfig(**_make_valid_verl_config(kl_loss_coef=-0.1))

    def test_non_negative_validation_min_lr_ratio(self):
        """Test that min_lr_ratio must be non-negative."""
        with pytest.raises(ValidationError, match="between 0 and 1"):
            VerlConfig(**_make_valid_verl_config(min_lr_ratio=-0.1))

        # 0 should be valid
        config = VerlConfig(**_make_valid_verl_config(min_lr_ratio=0.0))
        assert math.isclose(config.min_lr_ratio, 0.0, rel_tol=1e-5) == True

    def test_ckpt_content_unique_validation(self):
        """Test that ckpt_content list must have unique values."""
        with pytest.raises(ValidationError, match="must be unique"):
            VerlConfig(**_make_valid_verl_config(ckpt_content=["model", "model"]))

        # Unique values should work
        config = VerlConfig(**_make_valid_verl_config(ckpt_content=["model", "optimizer"]))
        assert config.ckpt_content == ["model", "optimizer"]

    def test_adv_estimator_literal_validation(self):
        """Test that adv_estimator only accepts valid literals."""
        with pytest.raises(ValidationError):
            VerlConfig(**_make_valid_verl_config(adv_estimator="invalid"))

        # Valid values should work
        config = VerlConfig(**_make_valid_verl_config(adv_estimator="gae"))
        assert config.adv_estimator == "gae"

    def test_warmup_style_literal_validation(self):
        """Test that warmup_style only accepts valid literals."""
        with pytest.raises(ValidationError):
            VerlConfig(**_make_valid_verl_config(warmup_style="invalid"))

        # Valid values should work
        config = VerlConfig(**_make_valid_verl_config(warmup_style="cosine"))
        assert config.warmup_style == "cosine"

    def test_policy_loss_mode_literal_validation(self):
        """Test that policy_loss_mode only accepts valid literals."""
        with pytest.raises(ValidationError):
            VerlConfig(**_make_valid_verl_config(policy_loss_mode="invalid"))

        # Valid values should work
        config = VerlConfig(**_make_valid_verl_config(policy_loss_mode="clip-cov"))
        assert config.policy_loss_mode == "clip-cov"

    def test_loss_agg_mode_literal_validation(self):
        """Test that loss_agg_mode only accepts valid literals."""
        with pytest.raises(ValidationError):
            VerlConfig(**_make_valid_verl_config(loss_agg_mode="invalid"))

        # Valid values should work
        config = VerlConfig(**_make_valid_verl_config(loss_agg_mode="seq-mean-token-sum"))
        assert config.loss_agg_mode == "seq-mean-token-sum"

    def test_kl_loss_type_literal_validation(self):
        """Test that kl_loss_type only accepts valid literals."""
        with pytest.raises(ValidationError):
            VerlConfig(**_make_valid_verl_config(kl_loss_type="invalid"))

        # Valid values should work
        config = VerlConfig(**_make_valid_verl_config(kl_loss_type="full"))
        assert config.kl_loss_type == "full"

    def test_truncation_literal_validation(self):
        """Test that truncation only accepts valid literals."""
        with pytest.raises(ValidationError):
            VerlConfig(**_make_valid_verl_config(truncation="invalid"))

        # Valid values should work
        config = VerlConfig(**_make_valid_verl_config(truncation="middle"))
        assert config.truncation == "middle"

    def test_extra_fields_rejected(self):
        """Test that extra fields are rejected."""
        with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
            VerlConfig(**_make_valid_verl_config(unknown_field="value"))

    def test_policy_loss_boundary(self):
        """Test that min_lr_ratio must be non-negative."""
        with pytest.raises(ValidationError, match="upper bound should be larger than its lower bound"):
            VerlConfig(**_make_valid_verl_config(policy_loss_clip_cov_lb=5, policy_loss_clip_cov_ub=5))


class TestGlobalConfig:
    """Test GlobalConfig validation."""

    def test_valid_verl_config(self):
        """Test that a valid verl config is accepted."""
        config = GlobalConfig(**_make_valid_global_config_with_verl())
        assert config.train_backend == "verl"
        assert config.verl is not None
        assert config.mindspeed_rl is None

    def test_valid_mindspeed_config(self):
        """Test that a valid mindspeed_rl config is accepted."""
        config = GlobalConfig(**_make_valid_global_config_with_mindspeed())
        assert config.train_backend == "mindspeed_rl"
        assert config.mindspeed_rl is not None
        assert config.verl is None

    def test_missing_verl_config_section(self):
        """Test that missing verl section raises error when backend is verl."""
        config_dict = _make_valid_global_config_with_verl()
        del config_dict["verl"]
        with pytest.raises(ValidationError, match="verl config section is required"):
            GlobalConfig(**config_dict)

    def test_missing_mindspeed_rl_config_section(self):
        """Test that missing mindspeed_rl section raises error when backend is mindspeed_rl."""
        config_dict = _make_valid_global_config_with_mindspeed()
        del config_dict["mindspeed_rl"]
        with pytest.raises(ValidationError, match="mindspeed_rl config section is required"):
            GlobalConfig(**config_dict)

    def test_fraction_validation_gpu_memory_utilization(self):
        """Test that gpu_memory_utilization must be between 0 and 1."""
        with pytest.raises(ValidationError, match="must be between 0 and 1"):
            GlobalConfig(**_make_valid_global_config_with_verl(gpu_memory_utilization=1.5))

        with pytest.raises(ValidationError, match="must be between 0 and 1"):
            GlobalConfig(**_make_valid_global_config_with_verl(gpu_memory_utilization=-0.1))

    def test_fraction_validation_top_p(self):
        """Test that top_p must be between 0 and 1."""
        with pytest.raises(ValidationError, match="must be between 0 and 1"):
            GlobalConfig(**_make_valid_global_config_with_verl(top_p=1.5))

    def test_fraction_validation_min_p(self):
        """Test that min_p must be between 0 and 1."""
        with pytest.raises(ValidationError, match="must be between 0 and 1"):
            GlobalConfig(**_make_valid_global_config_with_verl(min_p=1.5))

    def test_fraction_validation_lr_warmup_fraction(self):
        """Test that lr_warmup_fraction must be between 0 and 1."""
        with pytest.raises(ValidationError, match="must be between 0 and 1"):
            GlobalConfig(**_make_valid_global_config_with_verl(lr_warmup_fraction=1.5))

    def test_fraction_validation_gamma(self):
        """Test that gamma must be between 0 and 1."""
        with pytest.raises(ValidationError, match="must be between 0 and 1"):
            GlobalConfig(**_make_valid_global_config_with_verl(gamma=1.5))

    def test_fraction_validation_lam(self):
        """Test that lam must be between 0 and 1."""
        with pytest.raises(ValidationError, match="must be between 0 and 1"):
            GlobalConfig(**_make_valid_global_config_with_verl(lam=1.5))

    def test_fraction_validation_weight_decay(self):
        """Test that weight_decay must be between 0 and 1."""
        with pytest.raises(ValidationError, match="must be between 0 and 1"):
            GlobalConfig(**_make_valid_global_config_with_verl(weight_decay=1.5))

    def test_fraction_validation_clip_ratio(self):
        """Test that clip_ratio must be between 0 and 1."""
        with pytest.raises(ValidationError, match="must be between 0 and 1"):
            GlobalConfig(**_make_valid_global_config_with_verl(clip_ratio=1.5))

    def test_positive_validation_num_gpus_per_node(self):
        """Test that num_gpus_per_node must be positive."""
        with pytest.raises(ValidationError, match="must be positive"):
            GlobalConfig(**_make_valid_global_config_with_verl(num_gpus_per_node=0))

    def test_positive_validation_max_num_seqs(self):
        """Test that max_num_seqs must be positive."""
        with pytest.raises(ValidationError, match="must be positive"):
            GlobalConfig(**_make_valid_global_config_with_verl(max_num_seqs=0))

    def test_positive_validation_rollout_n(self):
        """Test that rollout_n must be positive."""
        with pytest.raises(ValidationError, match="must be positive"):
            GlobalConfig(**_make_valid_global_config_with_verl(rollout_n=0))

    def test_positive_validation_max_model_len(self):
        """Test that max_model_len must be positive."""
        with pytest.raises(ValidationError, match="must be positive"):
            GlobalConfig(**_make_valid_global_config_with_verl(max_model_len=0))

    def test_positive_validation_kl_horizon(self):
        """Test that kl_horizon must be positive."""
        with pytest.raises(ValidationError, match="must be positive"):
            GlobalConfig(**_make_valid_global_config_with_verl(kl_horizon=0))

    def test_positive_validation_lr(self):
        """Test that lr must be positive."""
        with pytest.raises(ValidationError, match="must be positive"):
            GlobalConfig(**_make_valid_global_config_with_verl(lr=0))

    def test_positive_validation_clip_grad(self):
        """Test that clip_grad must be positive."""
        with pytest.raises(ValidationError, match="must be positive"):
            GlobalConfig(**_make_valid_global_config_with_verl(clip_grad=0))

    def test_positive_validation_kl_coef(self):
        """Test that kl_coef must be positive."""
        with pytest.raises(ValidationError, match="must be positive"):
            GlobalConfig(**_make_valid_global_config_with_verl(kl_coef=0))

    def test_positive_validation_temperature(self):
        """Test that temperature must be positive."""
        with pytest.raises(ValidationError, match="must be positive"):
            GlobalConfig(**_make_valid_global_config_with_verl(temperature=0))

    def test_non_negative_validation_entropy_coeff(self):
        """Test that entropy_coeff must be non-negative."""
        with pytest.raises(ValidationError, match="must be non-negative"):
            GlobalConfig(**_make_valid_global_config_with_verl(entropy_coeff=-0.1))

        # 0 should be valid
        config = GlobalConfig(**_make_valid_global_config_with_verl(entropy_coeff=0.0))
        assert math.isclose(config.entropy_coeff, 0.0, rel_tol=1e-5) == True

    def test_non_negative_validation_kl_target(self):
        """Test that kl_target must be non-negative."""
        with pytest.raises(ValidationError, match="must be non-negative"):
            GlobalConfig(**_make_valid_global_config_with_verl(kl_target=-0.1))

    def test_dtype_literal_validation(self):
        """Test that dtype only accepts valid literals."""
        with pytest.raises(ValidationError):
            GlobalConfig(**_make_valid_global_config_with_verl(dtype="invalid"))

        # Valid values should work
        config = GlobalConfig(**_make_valid_global_config_with_verl(dtype="float16"))
        assert config.dtype == "float16"

    def test_kl_penalty_literal_validation(self):
        """Test that kl_penalty only accepts valid literals."""
        with pytest.raises(ValidationError):
            GlobalConfig(**_make_valid_global_config_with_verl(kl_penalty="invalid"))

        # Valid values should work
        config = GlobalConfig(**_make_valid_global_config_with_verl(kl_penalty="abs"))
        assert config.kl_penalty == "abs"

    def test_kl_ctrl_type_literal_validation(self):
        """Test that kl_ctrl_type only accepts valid literals."""
        with pytest.raises(ValidationError):
            GlobalConfig(**_make_valid_global_config_with_verl(kl_ctrl_type="invalid"))

        # Valid values should work
        config = GlobalConfig(**_make_valid_global_config_with_verl(kl_ctrl_type="adaptive"))
        assert config.kl_ctrl_type == "adaptive"

    def test_extra_fields_rejected(self):
        """Test that extra fields are rejected."""
        with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
            GlobalConfig(**_make_valid_global_config_with_verl(unknown_field="value"))

    def test_missing_required_fields(self):
        """Test that missing required fields raise errors."""
        with pytest.raises(ValidationError):
            GlobalConfig(train_backend="verl")

    def test_dataset_additional_keys_optional(self):
        """Test that dataset_additional_keys is optional."""
        config = GlobalConfig(**_make_valid_global_config_with_verl())
        assert config.dataset_additional_keys is None

        config = GlobalConfig(**_make_valid_global_config_with_verl(dataset_additional_keys=["labels", "extra"]))
        assert config.dataset_additional_keys == ["labels", "extra"]
