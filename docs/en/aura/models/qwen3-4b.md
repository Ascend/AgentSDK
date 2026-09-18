# Quick Startup Guide for Qwen3-4B

## Prerequisites

Make sure you have read the [Quick Start](../03_quick_start.md).

## Model Acquisition

This experiment uses the Qwen3-4B model. You can obtain the model from the [ModelScope model page](https://www.modelscope.cn/models/Qwen/Qwen3-4B).

## Dataset Acquisition

For information about obtaining the Math dataset used in this experiment, see [Training Data Preparation](../02_installation_guide.md#training-data-preparation).

## Modifying Files

Before starting Qwen3-4B in the math scenario, modify the following files. Refer to the comments at the beginning of each file for the parameters that need to be modified.

For Hybrid mode:

1. [Hybrid configuration file](../../../../aura/configs/train/verl_train_hybrid_A3_t16_qwen3_4b_math_fsdp.yaml)

For One-Step-Off mode:

1. [One-Step-Off configuration file](../../../../aura/configs/train/verl_train_async_A3_t16_qwen3_4b_math_fsdp.yaml)
2. [One-Step-Off inference configuration file](../../../../aura/configs/infer/vllm_infer_i16_qwen3_4b.yaml)

> [!NOTE]
>
>- Hybrid mode uses Parquet datasets by default with the veRL backend.
>- One-Step-Off mode currently supports only datasets in the Megatron `bin` format with the veRL backend.
>- In One-Step-Off multi-node mode, save both the code and model weights on a shared disk to ensure that they can be accessed by all nodes simultaneously.

In addition, modify [base.conf](../../../../aura/configs/base.conf) and [hosts.conf](../../../../aura/configs/hosts.conf) according to the selected mode by following the steps in [Quick Start](../03_quick_start.md), and then start the experiment with the corresponding command.
