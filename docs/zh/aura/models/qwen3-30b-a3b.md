# Qwen3-30B-A3B快速入门指南

## **前置**

请确保已经阅读过[快速入门指南](../03_quick_start.md)。

## **模型获取**

本实验使用Qwen3-30B-A3B模型，相关模型可以通过[ModelScope 模型页面](https://www.modelscope.cn/models/Qwen/Qwen3-30B-A3B)获取。

## **数据集获取**

本实验使用的Math数据集可通过快速安装中的[准备训练数据](../02_installation_guide.md#准备训练数据)章节查看获取。

## **文件修改**

在快速入门 Qwen3-30B-A3B Math 场景前，需按模式修改以下训练/推理配置文件，参数含义可参见对应文件头的注释。训练配置至少需要修改以下路径（其余参数保持默认即可）：

共卡模式请修改：

- [共卡训练配置文件](../../../../aura/configs/train/verl_train_hybrid_A3_t16_qwen3_30b_a3b_math_fsdp.yaml)

```yaml
hydra:
  searchpath:
    - file:///verl/verl/trainer/config
    - file:///path/to/AgentSDK/aura/configs/train/verl_conf # 改为本代码仓 aura/configs/train/verl_conf 的绝对路径

verl_conf:
  data:
    train_files: /path/to/data/train.parquet # 共卡模式训练数据集（parquet）
    val_files: /path/to/data/test.parquet # 共卡模式测试数据集（parquet）
  actor_rollout_ref:
    model:
      path: /path/to/models/Qwen3-30B-A3B # 模型权重路径
```

单步异步分离模式请修改：

- [单步异步分离训练配置文件](../../../../aura/configs/train/verl_train_async_A3_t16_qwen3_30b_a3b_math_fsdp.yaml)
- [单步异步分离推理配置文件](../../../../aura/configs/infer/vllm_infer_i16_qwen3_30b_a3b.yaml)

```yaml
hydra:
  searchpath:
    - file:///verl/verl/trainer/config
    - file:///path/to/AgentSDK/aura/configs/train/verl_conf # 改为本代码仓 aura/configs/train/verl_conf 的绝对路径

verl_conf:
  extras:
    data_loader:
      train_data_path: /path/to/data/math_train_dataset/rl # 分离模式数据集路径（bin/idx），末尾需要带上 /rl
  actor_rollout_ref:
    model:
      path: /path/to/models/Qwen3-30B-A3B # 模型权重路径
```

> [!NOTE]
> `weight_save_dir` 默认为代码目录下的 `aura/weights`，分离多机模式下请将代码与权重均保存在共享盘内，无需修改该参数。

推理配置只需修改模型权重路径：

```yaml
infer_model_path: /path/to/models/Qwen3-30B-A3B # 推理侧模型权重路径
```

> [!NOTE]
>
>- 共卡模式使用verl后端默认使用parquet数据集
>- 分离模式在verl后端时暂时仅支持megatron的bin格式数据集
>- 分离多机器模式下请将代码、权重均保存在共享盘内，保证数据可以同时被所有机器获取

另需按照[快速入门指南](../03_quick_start.md)根据模式修改[base.conf](../../../../aura/configs/base.conf)和[hosts.conf](../../../../aura/configs/hosts.conf)，并在 `aura/configs/env/env.conf` 路径下设置 `env.local`，随后根据相应的命令一键拉起实验。
