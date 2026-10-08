# Qwen3.5-35B-A3B 共卡模式快速拉起指南

## **前置**

请先阅读[快速入门指南](../03_quick_start.md)。本文档补充
Qwen3.5-35B-A3B 在单台 Atlas A3 16 卡服务器上进行 Math 场景共卡训练时需要调整的内容。

| 依赖 | 版本 |
| -- | -- |
| CANN | 9.0.0 |
| vLLM | `0.16.1rc1.dev140+g4034c3d32`（源码提交 `4034c3d32`） |
| vLLM-Ascend | `0.16.0rc2.dev32+gfe4cad24e`（源码提交 `fe4cad24e`） |
| Transformers | 源码提交 `cc7ab9be50` |

## **Qwen3.5 特殊依赖**

Qwen3.5 包含 Gated DeltaNet 层，需要使用由当前 `tech_26.1.0` 源码构建的镜像。
官方镜像构建流程已通过 `aura/third_party/patch/vllm-ascend.patch` 包含 vLLM-Ascend 上游提交
`3fdd5284` 的修复，解决部分解码场景下的 Triton 算子编译失败，共卡模式复用该修复。

> [!IMPORTANT]
>
> - 本配置使用镜像中的默认 Python 环境和 vLLM 0.16。
> - 请勿激活 `/home/work/model_env/qwen3_moe` 中的 vLLM 0.11 环境。
> - 当前依赖版本不支持 Qwen3.5 的 Mamba prefix cache，配置文件已将 `enable_prefix_caching` 设置为 `false`。

## **文件修改**

共卡模式请修改[共卡训练配置文件](../../../../aura/configs/train/verl_train_hybrid_A3_t16_qwen3_5_35b_a3b_math_fsdp.yaml)，
按照文件头部注释替换以下路径，并确认 `enable_prefix_caching` 保持为 `false`：

1. `hydra.searchpath` 中 AgentSDK 的 verl 配置目录。
2. `verl_conf.data.train_files` 和 `verl_conf.data.val_files` 中 parquet 数据路径。
3. `verl_conf.actor_rollout_ref.model.path` 和 `llm_tokenizer_path` 中模型权重路径。

另需按照[快速入门指南](../03_quick_start.md)修改 `base.conf` 和 `hosts.conf`：

- `base.conf` 使用 `hybrid` 模式，并将 `train_config_name` 设置为
  `verl_train_hybrid_A3_t16_qwen3_5_35b_a3b_math_fsdp`。
- `hosts.conf` 中填写当前服务器可通信的 IP。
- 根据实际环境设置网卡名称和可见 NPU。

完成上述修改后，按照[快速入门指南](../03_quick_start.md)中的共卡模式流程准备模型和数据并启动训练。
