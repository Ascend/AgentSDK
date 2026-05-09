# 命令行接口说明<a name="ZH-CN_TOPIC_0000002469525620"></a>

## agentic\_rl<a name="ZH-CN_TOPIC_0000002469365676"></a>

**命令格式**

```bash
bash run_start_in_local.sh --config-name <配置文件名>
```

**参数说明<a name="section973741317611"></a>**

| 参数名 | 说明 |
|--------|------|
| --config-name | 配置文件名。该配置文件的类型为yaml类型，默认存放在项目configs目录下。 |

**配置文件结构说明**

配置文件采用YAML格式，主要包含以下几个部分：

1. **agentic_ai**: 全局配置，包括运行模式、日志级别等
2. **serve_conf**: 服务化部署配置（SERVE模式）
3. **direct_conf**: 直连模式配置（DIRECT模式）
4. **verl_conf**: 训练配置参数（verl后端）
5. **train_instances**: 训练任务实例配置
6. **agent_instances**: Agent服务实例配置
7. **infer_instances**: 推理服务实例配置

---

## 全局配置参数<a name="section61851418129"></a>

### agentic_ai 配置

| 参数名 | 类型 | 说明 | 约束 |
|--------|------|------|------|
| mode | str | 运行模式选择 | 可选值为"serve"（服务化部署）或"direct"（直连模式） |
| log_level | str | 日志级别 | 可选值为"DEBUG"、"INFO"、"WARNING"、"ERROR"等 |
| log_dir | str | 日志目录路径 | 路径必须存在且具有写入权限 |

### serve_conf 配置（SERVE模式）

| 参数名 | 类型 | 说明 | 约束 |
|--------|------|------|------|
| host | str | 服务监听地址 | 默认值为"0.0.0.0" |
| port | int | 服务监听端口 | 应为有效的端口号，默认值为8030 |

### direct_conf 配置（DIRECT模式）

| 参数名 | 类型 | 说明 | 约束 |
|--------|------|------|------|
| entrypoints | list | 任务入口点列表 | 包含job_type、job_name、job_kwargs等字段 |

---

## 训练配置参数（verl后端）<a name="section61851418129"></a>

### verl_conf.extras 配置

| 参数名 | 类型 | 说明 | 约束 |
|--------|------|------|------|
| agent_service | str | 依赖的Agent服务名称 | 应与agent_instances中的name对应 |
| infer_service | str | 依赖的推理服务名称 | 应与infer_instances中的name对应 |

### verl_conf.algorithm 配置

| 参数名 | 类型 | 说明 | 约束 |
|--------|------|------|------|
| adv_estimator | str | 优势评估器 | 可选值为"grpo"或"gae"，默认值为"grpo" |
| kl_ctrl.kl_coef | float | KL散度系数 | 应为大于0的浮点数，默认值为0.001 |

### verl_conf.data 配置

| 参数名 | 类型 | 说明 | 约束 |
|--------|------|------|------|
| train_files | str | 训练数据文件路径 | 路径必须存在，支持 `.parquet` 或 `.json` 格式 |
| val_files | str | 验证数据文件路径 | 路径必须存在，支持 `.parquet` 或 `.json` 格式 |
| train_batch_size | int | 训练批次大小 | 应为大于0的整数，默认值为16 |
| max_prompt_length | int | 最大提示词长度 | 应为大于0的整数，默认值为2048 |
| max_response_length | int | 最大响应长度 | 应为大于0的整数，默认值为2048 |
| filter_overlong_prompts | bool | 是否过滤超过max_prompt_length的样本 | 默认值为True |
| truncation | str | 截断方式 | 默认值为"error"。verl 默认仅支持 "error" 模式，其他截断方式（left、right、middle）需自定义数据集类实现 |

### verl_conf.actor_rollout_ref.model 配置

| 参数名 | 类型 | 说明 | 约束 |
|--------|------|------|------|
| path | str | 模型权重路径 | 路径必须存在且包含完整的模型文件 |
| use_remove_padding | bool | 是否在训练时“移除 padding token，只对真实 token 做计算” | 默认值为False |
| enable_gradient_checkpointing | bool | 用“以计算换显存”的方式，减少训练时的显存占用 | 默认值为True |

> **enable_gradient_checkpointing 参数说明**：
>
> - **关闭时**：
>
>   ```text
>   forward → 保存所有中间激活值 → backward 直接用
>   ```
>
>   训练速度会提升，但非常吃显存。
>
> - **开启后**：
>
>   ```text
>   forward → 不保存中间激活值
>   backward → 重新计算一遍 forward → 再求梯度
>   ```
>
>   训练变慢，显存大幅下降。

### verl_conf.actor_rollout_ref.actor 配置

| 参数名 | 类型 | 说明 | 约束 |
|--------|------|------|------|
| strategy | str | 训练分布式策略 | 可选值为"megatron"、"fsdp"、"fsdp2"、"veomni"，默认值为"megatron" |
| optim.lr | float | 学习率 | 应为大于0的浮点数，默认值为5e-7 |
| entropy_coeff | float | 熵正则化系数，控制模型“要不要多探索” | 应为非负数，默认值为0.001 |
| ppo_mini_batch_size | int | 每次 PPO 更新时，用多少样本做一次梯度更新 | 应为大于0的整数，默认值为2 |
| ppo_micro_batch_size_per_gpu | int | 每张卡一次 forward/backward 实际处理的样本数 | 应为大于0的整数，默认值为2 |
| use_kl_loss | bool | 是否把 KL loss 添加到 loss 里作为惩罚项 | 默认值为True，与 use_kl_in_reward 互斥使用 |
| kl_loss_coef | float | KL loss的系数 | 应为0与1之间的浮点数，默认值为0.001 |
| kl_loss_type | str | KL loss计算方式 | 可选值为"kl"、"abs"、"mse"、"low_var_kl"、"full"，默认值为"low_var_kl" |

### verl_conf.actor_rollout_ref.actor.megatron 配置

| 参数名 | 类型 | 说明 | 约束 |
|--------|------|------|------|
| seed | int | 随机种子 | 应为非负整数，默认值为0 |
| pipeline_model_parallel_size | int | 流水线并行大小 | 应为大于0的整数，默认值为1 |
| tensor_model_parallel_size | int | 张量并行大小 | 应为大于0的整数，默认值为4 |
| override_transformer_config.use_flash_attn | bool | 是否使用Flash Attention（减少显存访问 + 融合 kernel 的高效attention实现） | 默认值为True |

### verl_conf.actor_rollout_ref.rollout 配置

| 参数名 | 类型 | 说明 | 约束 |
|--------|------|------|------|
| prompt_length | int | 提示词长度 | 应为大于0的整数，默认值为2048 |
| response_length | int | 响应长度 | 应为大于0的整数，默认值为2048 |
| agent.agent_loop_manager_class | str | Agent循环管理器类路径 | 应为有效的Python类路径 |
| log_prob_micro_batch_size_per_gpu | int | 每张卡处理micro_batch_size的个数 | 应为大于0的整数，默认值为2 |
| enable_chunked_prefill | bool | 是否把“prompt 预填充（prefill）阶段”拆成多个 chunk 来执行，开启时，对于长prompt，可有效减少KV cache 的显存占用 | 默认值为False |
| tensor_model_parallel_size | int | 张量并行大小 | 应为大于0的整数，默认值为4 |
| name | str | 推理引擎名称 | 默认值为"vllm" |
| gpu_memory_utilization | float | rollout 推理引擎（比如vllm）允许占用的 GPU 显存比例上限 | 应为0到1之间的浮点数，默认值为0.6 |
| n | int | 每个 prompt 生成多少条候选 response | 应为大于0的整数，默认值为2 |
| do_sample | bool | 是否进行采样 | 默认值为False |

### verl_conf.actor_rollout_ref.ref 配置

| 参数名 | 类型 | 说明 | 约束 |
|--------|------|------|------|
| log_prob_micro_batch_size_per_gpu | int | 每张卡处理micro_batch_size的个数 | 应为大于0的整数，默认值为2 |
| megatron.pipeline_model_parallel_size | int | 流水线并行大小 | 应为大于0的整数，默认值为2 |
| megatron.tensor_model_parallel_size | int | 张量并行大小 | 应为大于0的整数，默认值为2 |

### verl_conf.trainer 配置

| 参数名 | 类型 | 说明 | 约束 |
|--------|------|------|------|
| val_before_train | bool | 是否在训练前进行验证，测量“初始模型水平”，用于判断训练是否真的有效 | 默认值为False |
| device | str | 训练设备 | 可选值为"npu"、"gpu"、“cpu”，默认值为"npu" |
| critic_warmup | int | Actor 更新延迟开关。只有当当前 global step 大于等于 critic_warmup 时，才会执行 Actor 模型的更新 | 应为非负整数，默认值为0 |
| project_name | str | 项目名称 | 用于标识训练项目 |
| experiment_name | str | 实验名称 | 用于标识训练实验 |
| n_gpus_per_node | int | 每个节点的GPU/NPU数量 | 应为大于0的整数，默认值为8 |
| nnodes | int | 节点数量 | 应为大于0的整数，默认值为1 |
| save_freq | int | 每隔多少 step 保存一次 checkpoint | 应为-1或大于0的整数，-1表示不保存，默认值为-1 |
| test_freq | int | 每隔多少 step 做一次评估（evaluation） | 应为-1或大于0的整数，-1表示不进行测试，默认值为-1 |
| total_epochs | int | 整个训练数据集要完整跑多少轮 | 应为大于0的整数，默认值为1 |
| logger | list | 日志记录器 | 可包含"console"、"tensorboard"等 |

---

## 训练实例配置<a name="section61851418129"></a>

### train_instances 配置

| 参数名 | 类型 | 说明 | 约束 |
|--------|------|------|------|
| name | str | 训练服务名称 | 必须唯一，建议使用有意义的命名 |
| executor_num | int | 执行器数量 | 应为大于0的整数，默认值为1 |
| executor_kwargs.cluster_mode | str | 集群模式 | 可选值为"hybrid"（共卡）、"one_step_off"（分离）、"dummy_train"（仅验证） |
| executor_kwargs.train_engine | str | 训练引擎 | 可选值为“mindspeed_rl”、"verl"等，默认为“verl” |
| executor_kwargs.train_config | dict | 训练配置 | 应引用verl_conf配置 |
| executor_kwargs.rollout_config | dict | Rollout配置 | 默认为空字典 |
| executor_kwargs.agent_service | str | Agent服务名称 | 应引用agent_instances中的name |
| executor_kwargs.infer_service | str | 推理服务名称 | 应引用infer_instances中的name |
| resource_info | list | 整个训练服务的资源描述 | 默认为空列表 |

---

## Agent实例配置<a name="section61851418129"></a>

### agent_instances 配置

| 参数名 | 类型 | 说明 | 约束 |
|--------|------|------|------|
| name | str | Agent服务名称 | 必须唯一，建议使用有意义的命名 |
| executor_num | int | 执行器数量 | 应为大于0的整数，默认值为1 |
| executor_kwargs.agent_engine | str | Agent引擎类型 | 可选值为"rllm"等 |
| executor_kwargs.agent_engine_kwargs.agent_name | str | Agent名称 | 应为有效的内置Agent名称，如"math" |
| executor_kwargs.agent_engine_kwargs.simplify_think_content | bool | 是否简化思考内容 | 默认值为false |
| executor_kwargs.agent_engine_kwargs.max_steps | int | 最大步数 | 应为大于0的整数，默认值为5 |
| executor_kwargs.agent_engine_kwargs.max_prompt_length | int | agent 输入 prompt 最大长度 | 应为大于0的整数，默认值为2048 |
| executor_kwargs.agent_engine_kwargs.max_model_len | int | 模型最大上下文长度 | 应为大于0的整数，默认值为4096 |
| executor_kwargs.agent_engine_kwargs.n_parallel_agents | int | 并行Agent数量 | 应为大于0的整数，默认值为1024 |
| executor_kwargs.agent_engine_kwargs.tokenizer | str | 分词器路径 | 应为有效的分词器路径 |
| executor_kwargs.infer_service_params | dict | 推理服务参数 | 包含top_p、temperature、max_tokens、model等 |
| executor_kwargs.trajectory_save_dir | str | 轨迹保存目录 | 应为有效的文件路径 |
| resource_info | list | 整个agent服务的资源描述 | 默认为空列表 |

---

## 推理实例配置<a name="section61851418129"></a>

### infer_instances 配置

| 参数名 | 类型 | 说明 | 约束 |
|--------|------|------|------|
| name | str | 推理服务名称 | 必须唯一，建议使用模型名称 |
| executor_num | int | 执行器数量 | 应为大于0的整数，默认值为1 |
| executor_kwargs.engine | str | 推理引擎类型 | 默认值为"vllm_proxy"，表示通过 HTTP 调 vLLM server |
| executor_kwargs.engine_kwargs.chat_server | str | vLLM HTTP 推理服务地址 | 应为有效的HTTP/HTTPS地址 |
| executor_kwargs.engine_kwargs.prefill_server_list | list | 专门处理 prefill的 server 列表 | 可为空列表 |
| executor_kwargs.engine_kwargs.decode_server_list | list | 专门处理 decode的 server 列表 | 可为空列表 |
| executor_kwargs.engine_kwargs.model_name | str | 推理服务加载的模型名称 | 应为有效的模型名称 |
| resource_info | list | 整个推理服务的资源描述 | 默认为空列表 |

---

## 配置文件示例<a name="section_config_example"></a>

以下是一个完整的配置文件示例（verl 后端，hybrid 模式），其他配置文件参考[configs](../../configs)目录。使用前请根据实际环境修改以下配置项：

| 配置项 | 说明 |
|--------|------|
| `train_files` / `val_files` | 训练集和验证集路径 |
| `model.path` | 模型权重路径 |
| `tokenizer` | 分词器路径 |
| `trajectory_save_dir` | 轨迹数据保存路径 |

```yaml
# 全局配置
agentic_ai:
  mode: direct
  log_level: DEBUG
  log_dir: /var/log/

# DIRECT模式配置
direct_conf:
  entrypoints:
    - job_type: train
      job_name: ${train_instances.0.name}
      job_kwargs: { }

# Hydra配置
hydra:
  searchpath:
    - file:///verl/verl/trainer/config
    - file://AgenticRL/configs/verl_conf

defaults:
  - ppo_megatron_trainer
  - ppo_megatron_trainer@verl_conf
  - _self_

# 训练配置
verl_conf:
  extras:
    agent_service: ${agent_instances.0.name}
    infer_service: ${infer_instances.0.name}
  algorithm:
    adv_estimator: grpo
    kl_ctrl:
      kl_coef: 0.001
  data:
    train_files: /path/to/train.parquet
    val_files: /path/to/test.parquet
    train_batch_size: 16
    max_prompt_length: 2048
    max_response_length: 2048
    filter_overlong_prompts: True
    truncation: 'error'
  actor_rollout_ref:
    model:
      path: /path/to/model
      use_remove_padding: False
      enable_gradient_checkpointing: True
    actor:
      strategy: megatron
      optim:
        lr: 5e-7
      entropy_coeff: 0.001
      ppo_mini_batch_size: 2
      ppo_micro_batch_size_per_gpu: 2
      use_kl_loss: True
      kl_loss_coef: 0.001
      kl_loss_type: low_var_kl
      megatron:
        seed: 0
        pipeline_model_parallel_size: 1
        tensor_model_parallel_size: 4
        override_transformer_config:
          use_flash_attn: True
    rollout:
      prompt_length: 2048
      response_length: 2048
      agent:
        agent_loop_manager_class: agentic_rl.trainer.train_adapter.verl.hybrid.agent_loop_manager.HybridAgentLoopManager
      log_prob_micro_batch_size_per_gpu: 2
      enable_chunked_prefill: False
      tensor_model_parallel_size: 4
      name: vllm
      gpu_memory_utilization: 0.6
      n: 2
      do_sample: False
    ref:
      log_prob_micro_batch_size_per_gpu: 2
      megatron:
        pipeline_model_parallel_size: 2
        tensor_model_parallel_size: 2
  trainer:
    val_before_train: False
    device: npu
    critic_warmup: 0
    project_name: 'verl_grpo_example'
    experiment_name: 'qwen2_7b_experiment'
    n_gpus_per_node: 8
    nnodes: 1
    save_freq: -1
    test_freq: 1000
    total_epochs: 1
    logger: ['console','tensorboard']

# 训练实例
train_instances:
  - name: RL-QWEN-7B-TRAIN
    executor_num: 1
    executor_kwargs:
      cluster_mode: hybrid
      train_engine: verl
      train_config: ${verl_conf}
      rollout_config: { }
      agent_service: ${agent_instances.0.name}
      infer_service: ${infer_instances.0.name}
    resource_info: [ ]

# Agent实例
agent_instances:
  - name: MATH-AGENT
    executor_num: 1
    executor_kwargs:
      agent_engine: rllm
      agent_engine_kwargs:
        agent_name: math
        simplify_think_content: false
        max_steps: 5
        max_prompt_length: 2048
        max_model_len: 4096
        n_parallel_agents: 1024
        tokenizer: /path/to/tokenizer
      infer_service_params:
        top_p: 1
        temperature: 1
        max_tokens: 4096
        model: ${infer_instances.0.name}
      trajectory_save_dir: /path/to/trajectory.jsonl
    resource_info: [ ]

# 推理实例
infer_instances:
  - name: QWEN2.5-7B
    executor_num: 1
    executor_kwargs:
      engine: vllm_proxy
      engine_kwargs:
        chat_server: "http://0.0.0.0:8080"
        prefill_server_list: []
        decode_server_list: []
        model_name: Qwen2.5-7B-Instruct
    resource_info: [ ]
```

---

## 断点续训说明

当前命令行接口默认开启断点续训功能。训练过程中的checkpoint默认保存在当前目录下的`checkpoints/${project_name}/${experiment_name}`路径中。当再次启动训练时，系统会自动检测该路径下是否存在checkpoint文件，若存在则自动加载并恢复训练。

如果用户希望从头开始训练而非恢复之前的训练状态，需要修改`project_name`或`experiment_name`参数。修改任意一个参数都会使用新的checkpoint目录，从而避免加载旧的checkpoint文件。

> [!IMPORTANT]
> checkpoint目录包含模型权重、优化器状态等敏感信息，请参考[安全加固](security_hardening.md)对模型保存路径进行安全配置。
> [!NOTE]
> 本章节中涉及到的所有路径，需满足以下要求：
> 
> - 该路径必须存在。
> - 该路径下的文件夹权限要求750，文件权限要求为640。
> - 该路径不能为软链接。
> - 该路径的字符串长度不能大于1024。
> - 该路径的属主为当前用户。
