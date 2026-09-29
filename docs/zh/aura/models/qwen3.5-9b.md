# Qwen3.5-9B快速拉起指南

## **前置**

请确保已经阅读过[快速开始指南](../03_quick_start.md)。

本文档以 Atlas 800I A2推理服务器（8 × 64 GB）、Ubuntu 系统为例进行说明。

> [!NOTE]
>
>- Qwen3.5-9B 使用 Gated DeltaNet 推理路径，昇腾社区发布的 agentsdk:26.1.0 镜像未包含该路径的 initial-state 无效索引修复。镜像必须基于包含 `aura/third_party/patch/vllm-ascend.patch`（对 `ops/triton/fla/sigmoid_gating.py` 的修改）的源码构建，构建方法请参考[安装指南](../02_installation_guide.md#步骤-1获取镜像)

## **模型获取**

本实验使用Qwen3.5-9B模型，相关模型可以通过[ModelScope 模型页面](https://www.modelscope.cn/models/Qwen/Qwen3.5-9B)获取。

```shell
modelscope download --model Qwen/Qwen3.5-9B --local_dir /path/to/Qwen3.5-9B
```

## **数据集获取**

本实验使用的Math数据集可通过快速安装中的[准备训练数据](../02_installation_guide.md#准备训练数据)章节查看获取。

## **文件修改**

在快速拉起Qwen3.5-9B Math场景前，需修改以下文件，需要修改的参数可以参照文件头的注释。

共卡模式请修改：

1. [共卡配置文件](../../../../aura/configs/train/verl_train_hybrid_A2_t8_qwen3_5_9b_math_fsdp.yaml)

> [!NOTE]
>
>- 共卡模式使用verl后端默认使用parquet数据集
>- 由于Atlas 800I A2推理服务器（8 × 64 GB）为8卡配置，故配置NPU设备序号为0～7，若使用Atlas 900 A3 SuperPoD超节点（16 × 64 GB），应配置序号为0～15的NPU设备
>- Qwen3.5 包含 Gated DeltaNet/Mamba 层，当前固定的 vLLM-Ascend 版本不支持 Mamba prefix cache，共卡配置中的 `enable_prefix_caching` 须保持为 `False`

另需按照[快速开始指南](../03_quick_start.md)修改[base.conf](../../../../aura/configs/base.conf)和[hosts.conf](../../../../aura/configs/hosts.conf)，随后根据相应的命令一键拉起实验。
