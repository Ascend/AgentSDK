# Qwen3.5-9B 分离 One-step off-policy 快速拉起指南

## 前置条件

请先阅读[安装指南](../../02_installation_guide.md)和[快速入门](../../03_quick_start.md)，完成容器环境、模型权重、训练数据和环境变量配置。

本文以两台 Atlas A3 服务器、每台 16 卡为例：一台用于训练，一台用于推理。训练和推理节点需要使用相同的 AgentSDK 镜像、代码和配置，并通过共享存储访问模型、数据和训练权重。

## 软件版本

| 依赖 | 版本 |
|---|---|
| CANN | 9.0.0 |
| vLLM | `0.16.1rc1.dev140+g4034c3d32` |
| vLLM-Ascend | `0.16.0rc2.dev32+gfe4cad24e` |
| Transformers | `cc7ab9be50` |

Qwen3.5-9B 使用 Gated DeltaNet 路径。当前固定版本的 vLLM-Ascend 不支持该路径的 prefix cache，因此推理配置中需要关闭 `enable_prefix_caching`。镜像还需要包含 AgentSDK 构建阶段回移的 GDN 修复补丁。

## 模型和数据

模型：[Qwen3.5-9B](https://www.modelscope.cn/models/Qwen/Qwen3.5-9B)。

Math 数据集使用 GSM8K。下载和数据转换请参考[安装指南中的准备训练数据](../../02_installation_guide.md#准备训练数据)及[分离模式数据处理流程](../../02_installation_guide.md#分离模式数据处理)。分离模式使用 Megatron `bin/idx` 数据格式，数据 tokenizer 使用 Qwen3.5-9B 模型目录。

## 配置文件

修改以下配置中的模型路径、数据路径和共享权重路径：

1. [分离训练配置](../../../../../aura/configs/train/verl_train_async_A3_t16_qwen3_5_9b_math_fsdp.yaml)
2. [分离推理配置](../../../../../aura/configs/infer/vllm_infer_i16_qwen3_5_9b.yaml)

训练配置的关键参数如下：

```yaml
train_instances:
  - executor_kwargs:
      work_mode: one_step_off
      rollout_config:
        use_on_policy: false
verl_conf:
  extras:
    weight_save_dir: /path/to/shared/weights
```

推理配置保持 `tensor_parallel_size: 4`、`data_parallel_size: 4`、`enable_expert_parallel: false`，并设置：

```yaml
enable_prefix_caching: false
```

按照[快速入门指南](../../03_quick_start.md)修改 `base.conf` 和 `hosts.conf`，选择上述训练和推理配置，并填写训练、推理节点的可通信地址。两个节点中的共享路径必须保持一致。

## 启动

在两个节点分别配置实际网卡和可见 NPU：

```shell
export DEFAULT_SOCKET_IFNAME=<NETWORK_INTERFACE_NAME>
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
```

先启动推理节点，再启动训练节点：

```shell
cd <AGENTSDK_ROOT>/aura
bash scripts/start_rl_with_verl_vllm.sh
```

启动后应能看到推理服务、Math rollout、reward、actor update 和参数同步日志。详细的通用启动流程和参数说明请参考[快速入门指南](../../03_quick_start.md)。

## 运行限制

- 训练和推理节点各使用 16 张 NPU；其他卡数需要同步调整训练卡数及 TP/DP 参数。
- 训练和推理必须使用包含 GDN 构建补丁的同一版本镜像。
- `enable_prefix_caching` 为 `false` 是 Qwen3.5-9B 当前版本的必要配置。
