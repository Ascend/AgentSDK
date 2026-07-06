# Qwen3-235B-A22B快速拉起指南

## **前置**

请确保已经阅读过[快速开始指南](../03_quick_start.md)。

## **模型获取**

本实验使用Qwen3-235B-A22B模型，相关模型可以通过[本链接](https://huggingface.co/Qwen/Qwen3-235B-A22B)获取。

## **数据集获取**

本实验使用的Math数据集可通过快速安装中的[准备训练数据](../02_installation_guide.md#准备训练数据)章节查看获取。

## **文件修改**

在快速拉起Qwen3-235B-A22B math场景前，需要您修改以下文件，需要进行修改的参数可以参照文件头的注释。

共卡模式请修改：

1. [共卡配置文件](../../../../aura/configs/train/verl_train_hybrid_A3_t128_qwen3_235b_a22b_math_fsdp.yaml)

分离模式请修改：

1. [分离配置文件](../../../../aura/configs/train/verl_train_async_A3_t128_qwen3_235b_a22b_math_fsdp.yaml)
2. [分离推理配置](../../../../aura/configs/infer/vllm_infer_i128_qwen3_235b_a22b.yaml)

> [!NOTE] 说明
>
>- 共卡模式使用verl后端默认使用parquet数据集
>- 分离模式在verl后端时暂时仅支持megatron的bin格式数据集
>- 分离多机器模式下请将代码，权重均保存在共享盘内，保证数据可以同时被所有机器获取

您也需要按照[快速开始指南](../03_quick_start.md)根据模式修改[base.conf](../../../../aura/configs/base.conf)和[hosts.conf](../../../../aura/configs/hosts.conf), 随后根据相应的命令一键拉起实验。
