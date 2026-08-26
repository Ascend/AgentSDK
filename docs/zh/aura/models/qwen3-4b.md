# Qwen3-4B快速入门指南

## **前置**

请确保已经阅读过[快速入门指南](../03_quick_start.md)。

## **模型获取**

本实验使用Qwen3-4B模型，相关模型可以通过[ModelScope 模型页面](https://www.modelscope.cn/models/Qwen/Qwen3-4B)获取。

## **数据集获取**

本实验使用的Math数据集可通过快速安装中的[准备训练数据](../02_installation_guide.md#准备训练数据)章节查看获取。

## **文件修改**

在快速入门qwen3-4b math场景前，需修改以下文件，需要进行修改的参数可以参照文件头的注释。

单步异步分离模式请修改：

1. [共卡配置文件](../../../../aura/configs/train/verl_train_hybrid_A3_t16_qwen3_4b_math_fsdp.yaml)

单步异步分离模式请修改：

1. [单步异步分离配置文件](../../../../aura/configs/train/verl_train_async_A3_t16_qwen3_4b_math_fsdp.yaml)
2. [单步异步分离推理配置](../../../../aura/configs/infer/vllm_infer_i16_qwen3_4b.yaml)

> [!NOTE]
>
>- 共卡模式使用verl后端默认使用parquet数据集
>- 分离模式在verl后端时暂时仅支持megatron的bin格式数据集
>- 分离多机器模式下请将代码、权重均保存在共享盘内，保证数据可以同时被所有机器获取

另需按照[快速入门指南](../03_quick_start.md)根据模式修改[base.conf](../../../../aura/configs/base.conf)和[hosts.conf](../../../../aura/configs/hosts.conf)，并在 `aura/configs/env/env.conf` 路径下设置 `env.local`，随后根据相应的命令一键拉起实验。
