# Qwen3-8B Quick Start Guide

## **Prerequisites**

Ensure that you have read the [Quick Start Guide](../03_quick_start.md).

## **Obtaining the Model**

This experiment uses the Qwen3-8B model. You can obtain the model from the [ModelScope model page](https://www.modelscope.cn/models/Qwen/Qwen3-8B).

## **Obtaining the Dataset**

You can obtain the Math dataset used in this experiment from the [Preparing Training Data](../02_installation_guide.md#preparing-training-data) section of the installation guide.

## **Modifying Files**

Before you start the Qwen3-8B Math scenario as described in this quick start guide, modify the following training and inference configuration files according to the mode. For the meaning of the parameters, see the comments at the beginning of the corresponding file. In the training configuration, modify at least the following paths (keep the remaining parameters at their default values):

In co-located mode, modify the following files:

- [Co-located training configuration file](../../../../aura/configs/train/verl_train_hybrid_A3_t16_qwen3_8b_math_fsdp.yaml)

```yaml
hydra:
  searchpath:
    - file:///verl/verl/trainer/config
    - file:///path/to/AgentSDK/aura/configs/train/verl_conf # Change this to the absolute path of the aura/configs/train/verl_conf directory in this code repository

verl_conf:
  data:
    train_files: /path/to/data/train.parquet # Training dataset (parquet) for co-located mode
    val_files: /path/to/data/test.parquet # Test dataset (parquet) for co-located mode
  actor_rollout_ref:
    model:
      path: /path/to/models/Qwen3-8B # Model weight path
```

In one-step asynchronous disaggregation mode, modify the following files:

- [One-step asynchronous disaggregation training configuration file](../../../../aura/configs/train/verl_train_async_A3_t16_qwen3_8b_math_fsdp.yaml)
- [One-step asynchronous disaggregation inference configuration file](../../../../aura/configs/infer/vllm_infer_i16_qwen3_8b.yaml)

```yaml
hydra:
  searchpath:
    - file:///verl/verl/trainer/config
    - file:///path/to/AgentSDK/aura/configs/train/verl_conf # Change this to the absolute path of the aura/configs/train/verl_conf directory in this code repository

verl_conf:
  extras:
    data_loader:
      train_data_path: /path/to/data/math_train_dataset/rl # Dataset path (bin/idx) for disaggregation mode, which must end with /rl
  actor_rollout_ref:
    model:
      path: /path/to/models/Qwen3-8B # Model weight path
```

> [!NOTE]
> By default, `weight_save_dir` is set to `aura/weights` in the code directory. In multi-node disaggregation mode, save both the code and the weights on the shared drive. You do not need to modify this parameter.

For the inference configuration, modify only the model weight path:

```yaml
infer_model_path: /path/to/models/Qwen3-8B # Model weight path on the inference side
```

> [!NOTE]
>
>- Co-located mode uses the verl backend, which uses the parquet dataset by default.
>- In disaggregation mode, the verl backend currently supports only the Megatron bin format dataset.
>- In multi-node disaggregation mode, save both the code and the weights on the shared drive to ensure that all machines can obtain the data at the same time.

In addition, follow the [Quick Start Guide](../03_quick_start.md) to modify [base.conf](../../../../aura/configs/base.conf) and [hosts.conf](../../../../aura/configs/hosts.conf) according to the mode, set `env.local` in the `aura/configs/env/env.conf` path, and then use the corresponding command to start the experiment with one click.
