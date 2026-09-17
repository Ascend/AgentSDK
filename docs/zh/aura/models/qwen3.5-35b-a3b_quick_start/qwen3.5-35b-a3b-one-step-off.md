# Qwen3.5-35B-A3B 分离 One-step off-policy 快速拉起指南

## 前置条件

请先阅读[安装指南](../../02_installation_guide.md)和[快速入门](../../03_quick_start.md)，完成容器环境、模型权重、训练数据及环境变量配置。

分离模式需要独立的训练节点和推理节点。两个节点应使用相同的 AgentSDK 镜像、代码和配置，并通过共享存储访问模型权重、训练数据和 rollout 权重。默认配置以两个 Atlas A3 16 卡节点为例，推理侧使用 TP4、DP4、EP16。

## 模型和数据

模型权重可从 [ModelScope Qwen3.5-35B-A3B](https://www.modelscope.cn/models/Qwen/Qwen3.5-35B-A3B) 获取。Math 场景使用 GSM8K 数据集，下载和分离模式数据处理步骤请参考[安装指南中的准备训练数据](../../02_installation_guide.md#准备训练数据)。

分离模式数据加载器读取 Megatron `bin/idx` 数据。处理数据时，将 `aura/configs/datasets/gsm8k.yaml` 中的 tokenizer 设置为 Qwen3.5-35B-A3B 权重目录，并使输出数据前缀与训练配置中的 `train_data_path` 一致。

## 配置

根据实际路径修改以下配置文件中的占位路径：

1. [分离训练配置](../../../../../aura/configs/train/verl_train_async_A3_t16_qwen3_5_35b_a3b_math_fsdp.yaml)
2. [分离推理配置](../../../../../aura/configs/infer/vllm_infer_i16_qwen3_5_35b_a3b.yaml)

训练配置至少需要修改以下路径：

```yaml
hydra:
  searchpath:
    - file:///verl/verl/trainer/config
    - file:///path/to/AgentSDK/aura/configs/train/verl_conf

verl_conf:
  extras:
    data_loader:
      train_data_path: /path/to/data/math_train_dataset/rl
    weight_save_dir: /path/to/shared/weights
  actor_rollout_ref:
    model:
      path: /path/to/models/Qwen3.5-35B-A3B
```

推理配置中的 `infer_model_path` 也必须指向同一模型权重目录，并保持：

```yaml
enable_prefix_caching: false
```

Qwen3.5 包含 Gated DeltaNet/Mamba 层，AgentSDK 当前固定的 vLLM-Ascend 版本不支持 Mamba prefix cache。镜像必须基于包含 `aura/third_party/patch/vllm-ascend.patch` 的源码构建；通用镜像构建方法请参考[安装指南](../../02_installation_guide.md#步骤-1获取镜像)。

按照[快速入门中的分离模式配置](../../03_quick_start.md)修改 `base.conf` 和 `hosts.conf`，并确保训练、推理节点中的共享路径一致。训练配置中的关键模式参数位于 `train_instances[0].executor_kwargs`：

```yaml
train_instances:
  - executor_kwargs:
      work_mode: one_step_off
      rollout_config:
        use_on_policy: false
```

## 启动

分别在推理节点和训练节点执行以下命令，先启动推理节点，再启动训练节点：

```shell
cd /path/to/AgentSDK/aura
bash scripts/start_rl_with_verl_vllm.sh
```

训练和推理并行度、模型路径、数据路径及共享权重路径必须与 YAML 配置一致。其他卡数需要同步调整 TP、DP、EP 和训练并行参数，并重新验证显存与通信配置。
