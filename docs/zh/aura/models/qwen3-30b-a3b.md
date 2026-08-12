# Qwen3-30B-A3B快速拉起指南

## **前置**

请确保已经阅读过[快速开始指南](../03_quick_start.md)。

## **模型获取**

本实验使用Qwen3-30B-A3B模型，相关模型可以通过[ModelScope 模型页面](https://www.modelscope.cn/models/Qwen/Qwen3-30B-A3B)获取。

## **数据集获取**

本实验使用的Math数据集可通过快速安装中的[准备训练数据](../02_installation_guide.md#准备训练数据)章节查看获取。

## **文件修改**

在快速拉起Qwen3-30B-A3B math场景前，需修改以下文件，需要进行修改的参数可以参照文件头的注释。

共卡模式请修改：

1. [共卡配置文件](../../../../aura/configs/train/verl_train_hybrid_A3_t16_qwen3_30b_a3b_math_fsdp.yaml)

分离模式请修改：

1. [分离配置文件](../../../../aura/configs/train/verl_train_async_A3_t16_qwen3_30b_a3b_math_fsdp.yaml)
2. [分离推理配置](../../../../aura/configs/infer/vllm_infer_i16_qwen3_30b_a3b.yaml)

## **环境配置**

如果参照[快速开始指南](../03_quick_start.md)使用预构建镜像或一键拉起脚本构建环境，需通过下面的方式切换到该模型的运行环境：

```shell
cd /home/work/model_env/qwen3_moe
source bin/activate
```

并修改[env.conf](../../../../aura/configs/env/env.conf)中vllm的版本：

```text
# VLLM版本  【会被 infer yaml(vllm_version) 覆盖】
VLLM_VERSION=0.11.0
```

如需运行其他dense模型，可以通过下面的方式退出该模型的虚拟环境：

```shell
deactivate
```

并修改vllm版本为原本值：

```text
# VLLM版本  【会被 infer yaml(vllm_version) 覆盖】
VLLM_VERSION=0.16.0
```

> [!NOTE]
>
>- 在运行moe模型的虚拟环境下要基于其他dense模型启动agent强化学习训练，需要执行deactivate命令退出虚拟环境后再执行一键拉起脚本
>- 在运行moe模型的环境中支持的vllm版本为0.11.0，由于在共卡模式下不会读取infer yaml中的vllm版本，所以需要在env.conf中修改vllm版本为0.11.0，在运行其他dense模型时也需要将vllm版本重新修改为0.16.0
>- 共卡模式使用verl后端默认使用parquet数据集
>- 分离模式在verl后端时暂时仅支持megatron的bin格式数据集
>- 分离多机器模式下请将代码、权重均保存在共享盘内，保证数据可以同时被所有机器获取

另需按照[快速开始指南](../03_quick_start.md)根据模式修改[base.conf](../../../../aura/configs/base.conf)和[hosts.conf](../../../../aura/configs/hosts.conf)，并在 `aura/configs/env/env.conf` 路径下设置 `env.local`，随后根据相应的命令一键拉起实验。
