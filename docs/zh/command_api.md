# 命令行接口说明<a name="ZH-CN_TOPIC_0000002469525620"></a>

## agentic\_rl<a name="ZH-CN_TOPIC_0000002469365676"></a>

**参数说明<a name="section973741317611"></a>**

|参数名|类型|说明|
|--|--|--|
|--config-path|--config-path|配置文件的路径。该配置文件的类型为yaml类型。|


**配置文件参数说明<a name="section61851418129"></a>**

| 参数名                       | 类型   | 说明  | 约束  |
|---------------------------|------|-------------------------------------|--------------------------------|
| tokenizer_name_or_path    | str  | mindspeed_rl侧表示分词器路径，verl侧表示模型路径（分词器由verl根据模型自动推测）|路径存在且路径中所有文件夹的权限为750，文件权限位640，且所有文件/文件夹属主均为当前属主。|
| model_name                | str  | 当前支持的模型，当前仅在后端为mindspeed_rl时生效|支持字符串为“qwen3-8b”或“qwen2.5-7b”。|
| agent_name                | str  | Agent的名称 | 必须以字母开头，由大小写字母，数字和下划线组成。|
| agent_engine_wrapper_path | str  | 继承BaseEngineWrapper类的文件路径。|路径存在且路径中所有文件夹的权限为750，文件权限位640，且所有文件/文件夹属主均为当前属主。|
| train_backend             | str  | 训练后端 | 应为verl或mindspeed_rl。|
| use_stepwise_advantage    | bool | 是否启用stepwise_advantage | True或False，默认值为False。|
| infer_tensor_parallel_size | int | 推理张量并行数量 | 默认值为4，张量并行数量应为大于0的整数，根据不同模型以及模型参数量配置，应被推理使用卡数整除，且通常取值为1，2，4，8。详细信息请参考[优化与调优-vLLM-vLLM文档](http://docs.vllm.com.cn/en/latest/configuration/optimization/#chunked-prefill)。 |
| infer_pipeline_parallel_size | int | 推理流水并行数量 | 默认值为1，通常情况下应大于0，当前版本后端为verl时设置无效，且固定为1。|
| infer_expert_parallel_size | int | 推理专家并行数量 | 默认值为1，仅在模型为MOE架构设置下有效，应为大于0整数，且专家并行数量应被模型专家数量整除。当前版本不支持设置，固定为1。|
| max_num_seqs              | int | 推理单个批次中的最大并行序列数量 | 默认值为1024，应设置为大于0的整数。过大设置该参数可能会造成显存不足，详细信息请参考[优化与调优-vLLM-vLLM文档](http://docs.vllm.com.cn/en/latest/configuration/optimization/#chunked-prefill)。 |
| max_num_batched_tokens    | int | 推理单个批次中最多允许的token数量 | 应输入为大于0的整数，默认值为8192，详细信息请参考[优化与调优-vLLM-vLLM文档](http://docs.vllm.com.cn/en/latest/configuration/optimization/#chunked-prefill)。 |
| max_model_len             | int | 推理模型能处理的最大上下文长度 | 应为大于0的整数，默认值为16384，该参数与模型有关，不同模型支持的上下文长度不同，设置超过模型特定的上下文长度会导致推理结果不准确。|
| gpu_memory_utilization    | float | 推理引擎使用的显存占比 | 默认值为0.85，应为大于0小于等于1之间的浮点数。|
| max_tokens                | int | 推理最大token数 | 默认值为8192，应为大于0的整数。|
| dtype                     | str | 推理模型使用的数据类型 | 默认值为bfloat16，应为bfloat16，float16中的一种。|
| top_k                     | int | nucleus sampling参数，表示选前 k 个概率最大的token | 默认值为20，详细信息请参考[Sampling Parameters-vLLM](http://docs.vllm.ai/en/v0.5.5/dev/sampling_params.html)。|
| top_p                     | float | nucleus sampling参数，表示挑选累积概率超过top_p的token | 默认值为1.0，应为大于0小于等于1之间的浮点数，详细信息请参考[Sampling Parameters-vLLM](http://docs.vllm.ai/en/v0.5.5/dev/sampling_params.html)。|
| min_p                     | float | nucleus sampling参数，用于控制token出现的最小概率 | 默认值为0.01，应为大于0小于等于1之间的浮点数，详细信息请参考[Sampling Parameters-vLLM](http://docs.vllm.ai/en/v0.5.5/dev/sampling_params.html)。|
| temperature               | float | nucleus sampling参数，用于控制采样的随机率 | 默认值为0.6，应为大于0的浮点数。|
| enforce_eager             | bool | 是否启用eager模式 | 默认值为True。|
| use_kl_in_reward          | bool | 是否启用kl散度在奖励中 | 默认值为False。|
| clip_ratio                | float | 策略更新期间裁剪目标函数中的ε | 默认值为0.2，应为0与1之间的浮点数。|
| entropy_coeff             | float | 熵正则化系数 | 默认值为0.2，应为非负数。|
| kl_penalty                | str | KL散度函数 | 默认值为kl，应为kl，abs，mse，low_var_kl，full中的一个。|
| kl_coef                   | float | KL散度系数 | 默认值为0.05，应为正数。|
| gamma                     | float | 优势估计时的折扣因子 | 默认值为1.0，应为0与1之间的浮点数。|
| lam                       | float | 优势估计时偏置与方差间的交换系数 | 默认值为1.0，应为0与1之间的浮点数。|
| kl_horizon                | int | KL中滑动窗口的距离 | 默认值为10000，应为大于0的整数。|
| kl_target                 | float | 自适应控制器中的目标KL散度值 | 默认值为0.1，应为大于0的浮点数。|
| kl_ctrl_type              | str | KL控制器类型 | 默认值为fixed，应为fixed或adaptive。|
| lr                        | float | 学习率 | 默认值为1.0e-6，应为大于0的浮点数。|
| min_lr                    | float | 训练衰减期间的最低学习率 | 默认值为0.0，应为非负浮点数。|
| lr_warmup_fraction        | float | 训练期间预热训练次数占比 | 默认值为0.0，应为0与1之间的浮点数。|
| clip_grad                 | float | 梯度裁剪 | 应为大于0的浮点数，当前仅在训练后端为mindspeed_rl时生效。|
| weight_decay              | float | 模型权重L2正则化系数 | 默认值为0，应为0与1之间的浮点数。|
| num_gpus_per_node         | int | 单节点可用NPU数量 | 默认值为8，应为正整数。|
| max_prompt_length         | int | 最大提示词长度 | 默认值为2048，应为大于0的正整数，当前最大支持至128K。|
| rollout_n                 | int | 每一轮rollout阶段生成回复的数量 | 默认值为2，应为大于0且不超过64的正整数。|
| use_tensorboard           | bool | 是否启用tensorboard | 默认值为False。|


**mindspeed-rl相关参数**

| 参数名 | 类型 | 含义 | 约束 |
|-------------------|------------|----------------------------------|----------------------------------|
| data_path         | str | 数据路径 | 路径存在且路径中所有文件夹的权限为750，文件权限位640，且所有文件/文件夹属主均为当前属主，且文件名应为以下后缀之一：<br>“_packed_attention_mask_document.bin”，<br>“_packed_attention_mask_document.idx”，<br>“_packed_input_ids_document.bin”，<br>“_packed_input_ids_document.idx”，<br>“_packed_labels_document.bin”，<br>“_packed_labels_document.idx”。|
| load_params_path  | str | 训练模型的文件路径，需要包含完整的Megatron格式模型 | 路径存在且路径中所有文件夹的权限为750，文件权限位640，且所有文件/文件夹属主均为当前属主。|
| save_params_path  | str | 训练模型的文件保存路径 | 路径存在且路径中所有文件夹的权限为750，文件权限位640，且所有文件/文件夹属主均为当前属主。|
| train_iters      | int | 训练迭代次数 | 默认值为1，应为大于0的正整数。|
| epochs            | int | 训练任务每次更新需要迭代的次数 | 默认值为1，应为大于0的正整数。|
| seq_len           | int | 序列长度 | 默认值为8192，应为大于0的正整数。|
| global_batch_size | int | 全局批次大小 | 默认值为16，应为大于0的正整数。|
| save_interval     | int | 保存间隔 | 默认值为1000，应为大于0的正整数。|
| mini_batch_size   | int | 单卡批次权重更新时的批次大小 | 默认值为16，应为大于0的正整数。|
| micro_batch_size  | int | 单卡上一次前向与后向传播所处理的批次大小 | 默认值为1，应为大于0的正整数。|
| tensor_model_parallel_size  | int | 训练张量并行 | 默认值为4，应为大于0的正整数，根据不同模型以及模型参数量配置，应被推理使用卡数整除，通常取值为1，2，4，8。|
| pipeline_model_parallel_size | int | 训练流水并行数量 | 默认值为1，应为大于0的正整数。|
| adv_estimator     | str | 训练优势评估器 | 默认值为group_norm，应为“group_norm”或“gae”中的一种。|

**verl相关参数**

| 参数名 | 类型 | 含义 | 约束 |
|-------------------|------|--------------------------------|--------------------------------|
| train_files       | str | 训练数据路径 | 路径存在且路径中所有文件夹的权限为750，文件权限位640，且所有文件/文件夹属主均为当前属主。同时需要满足verl要求的parquet数据集格式。|
| val_files         | str | 训练验证数据集 | 路径存在且路径中所有文件夹的权限为750，文件权限位640，且所有文件/文件夹属主均为当前属主。同时需要满足verl要求的parquet数据集格式。|
| total_epochs      | int | 训练任务每次更新需要迭代的次数 | 默认值为2，应为大于0的正整数。|
| total_training    | optional[int] | 训练迭代次数 | 默认不提供，verl后端会通过输入数据的长度自动计算；若提供，应为大于0的整数，则会覆写verl默认值。|
| save_freq         | int | 训练保存频率 | 默认值为1000，应为大于0的整数。|
| ppo_mini_batch_size | int | PPO更新全局迷你批次大小 | 默认值为16，应为大于0的整数。|
| ppo_max_token_len_per_gpu | int | 一个NPU PPO轮次中处理的最大token数量 | 默认值为24000，应为大于0的整数，verl建议设置至少n*(prompt + respons)。|
| ppo_epochs        | int | 重复同一批轨迹的PPO更新所需要的轮次 | 默认值为1，应为大于0的整数。|
| project_name      | str | 项目名称 | 默认值为default-agent。|
| experiment_name   | str | 实验名称 | 默认值为default-experiment。|
| max_response_length | int | 最大生成长度 | 默认值为2048，应为大于0的整数。|
| train_batch_size  | int | 训练批次大小 | 默认值为8，应为大于0的整数。|
| val_batch_size    | int | 验证批次大小 | 默认值为512，应为大于0的整数。|
| dataloader_num_workers | int | 数据读取器所使用的worker数量 | 默认值为8，应为大于0的整数。|
| nnodes            | int | 训练集群机器数量 | 默认值为1，应为大于0的整数。|
| adv_estimator     | str | 优势评估器 | 默认值为grpo，应为“grpo”或“gae”中的一种。|
| warmup_style      | str | 预热方式 | 默认值为constant，应为“constant”或“cosine”中的一种。|
| min_lr_ratio      | float | 最小lr占比 | 当预热方式为cosine时设置有效，其他情况无效；默认值为0.0，应为0与1之间的浮点数。|
| num_cycles        | float | cosine周期 | 当预热方式为cosine时设置有效，其他情况无效；默认值为0.5，表示半个周期，应为大于0的浮点数。|
| ckpt_content      | list | 检查点保存内容 | 默认值为['model','optimizer','extra']，应为“model”，“optimizer”，“extra”，“hf_model”中的非重叠项组成的列表。|
| policy_loss_model | str | 策略损失计算模式 | 默认值为vanilla，应为“vanilla”，“clip-cov”，“kl-cov”，“gpg”中的一个。|
| policy_loss_clip_cov_ratio | float | token裁剪比例 | 默认值为0.0002，应为大于0的浮点数。该参数仅在policy_loss_model为clip-cov时有效，其他情况下设置无效。|
| policy_loss_clip_cov_lb | float | clip-cov下界 | 默认值为1.0，应为大于0的浮点数。该参数仅在policy_loss_model为clip-cov时有效，其他情况下设置无效。|
| policy_loss_clip_cov_ub | float | clip-cov上界 | 默认值为5.0，应为大于0的浮点数。该参数仅在policy_loss_model为clip-cov时有效，其他情况下设置无效。注意，clip-cov下界应小于上界。|
| policy_loss_kl_cov_ratio | float | 计算kl_cov时所选取的token比例 | 默认值为0.0002，应为大于0的浮点数。该参数仅在policy_loss_model为kl-cov时有效，其他情况下设置无效。|
| policy_loss_ppo_kl_coef | float | 计算kl_cov时的kl系数 | 默认值为0.1，应为0与1之间的浮点数。该参数仅在policy_loss_model为kl-cov时有效，其他情况下设置无效。|
| fsdp_param_offload | bool | 启动fsdp参数卸载 | 默认值为False。|
| fsdp_optimizer_offload | bool | 启动fsdp优化器卸载 | 默认值为False。|
| loss_agg_mode      | str | PPO损失聚合方式 | 默认值为token-mean，应为“token-mean”，“seq-mean-token-sum”，“seq-mean-token-mean”中的一个。|
| use_kl_loss        | bool | 是否使用kl损失替代kl reward penalty | 默认值为False。|
| kl_loss_coeff      | float | kl损失系数 | 默认值为0.001，应为0与1之间的浮点数，仅在use_kl_loss=True时生效，其他情况下设置无效。|
| kl_loss_type       | str | kl损失格式 | 默认值为low_bar_kl，应为“kl”，“abs”，“mse”，“low_var_kl”，“full”中的一个。仅在use_kl_loss=True时生效，其他情况下设置无效。|
| grad_clip          | float | actor的梯度裁剪值 | 默认值为1.0，应为大于0的浮点数。|
| entropy_from_logits_with_chunking | bool | 是否将熵分块计算 | 默认值为False。|
| balance_batch      | bool | 是否在分布式训练中对各worker的batch size进行均衡 | 默认值为True。|
| val_before_train   | bool | 是否在正式训练前进行一次验证 | 默认值为True。|
| val_only           | bool | 是否直接写验证而不进行训练 | 默认值为False。|
| test_freq          | int | 验证执行频率 | 默认值为-1，应为-1或大于0的正整数。|
| truncation         | str | 截断方式 | 默认值为error，应为“error”，“left”，“right”，“middle”中的一个。|


> [!NOTE] 
> 本章节中涉及到的所有路径，需满足以下要求：
>-   该路径必须存在。
>-   该路径下的文件夹权限要求750，文件权限要求为640。
>-   该路径不能为软链接。
>-   该路径的字符串长度不能大于1024。
>-   该路径的属主为当前用户。

