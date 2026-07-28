# Python接口说明

> 注意：AgentSDK可通过Python接口进行应用开发，从代码调用角度上来说所有Python侧接口都可以被调用。本章节仅列出业务提供的对外接口，其余未进行说明的接口用户请勿直接调用。

AgentSDK 是一个 Agent 训推调框架，支持对接任意 Agent 引擎、训练引擎、推理引擎。本文档介绍框架对外暴露的核心接口。

---

## 一、核心基类

核心基类是用户必须继承并实现的抽象类，用于自定义 Agent、环境、工具等核心组件。

### 1.1 BaseAgent - Agent 抽象基类

**功能描述**

Agent 抽象基类，负责与模型交互、维护对话状态、解析模型响应、记录轨迹。用户需要继承此类实现自定义 Agent。

**类定义**

```python
class BaseAgent(ABC):
    @property
    def chat_completions(self) -> list[dict[str, str]]: ...

    @property
    def trajectory(self) -> "Trajectory": ...

    @abstractmethod
    def update_from_env(self, observation: Any, reward: float, done: bool, info: dict, **kwargs): ...

    @abstractmethod
    def update_from_model(self, response: str, **kwargs) -> "Action": ...

    @abstractmethod
    def reset(self): ...

    def get_current_state(self) -> "Step | None": ...
```

**抽象方法说明**

| 方法名                 | 说明                            |
|---------------------|-------------------------------|
| `update_from_env`   | 从环境接收观测、奖励、终止信号，更新 Agent 内部状态 |
| `update_from_model` | 从模型接收响应，解析并返回动作               |
| `reset`             | 重置 Agent 状态，开始新的轨迹            |

**文件位置**: `aura/runner/agent_engine_wrapper/base/agent/base_agent.py`

---

### 1.2 BaseEnv - 环境抽象基类

**功能描述**

环境抽象基类，负责工具执行、奖励计算、状态管理。用户需要继承此类实现自定义环境。

**类定义**

```python
class BaseEnv(ABC):
    @abstractmethod
    def reset(self) -> tuple[dict, dict]: ...

    @abstractmethod
    def step(self, action: Any) -> tuple[Any, float, bool, dict]: ...

    def close(self): ...

    @staticmethod
    @abstractmethod
    def from_dict(info: dict) -> "BaseEnv": ...

    @staticmethod
    def is_multithread_safe() -> bool: ...
```

**抽象方法说明**

| 方法名         | 说明                           |
|-------------|------------------------------|
| `reset`     | 重置环境，返回初始观测和附加信息             |
| `step`      | 执行动作，返回 (观测, 奖励, 是否终止, 附加信息) |
| `from_dict` | 从配置字典创建环境实例                  |

**文件位置**: `aura/runner/agent_engine_wrapper/base/environment/base_env.py`

---

### 1.3 BaseEngineWrapper - 引擎包装器基类

**功能描述**

引擎包装器抽象基类，提供统一的 Agent 引擎适配接口。用户可继承此类对接不同的 Agent 引擎。

**类定义**

```python
class BaseEngineWrapper(ABC):
    @abstractmethod
    async def generate_trajectory(self, task: AgentTask, stream_queue=None, *args, **kwargs) -> "Trajectory": ...
```

**参数说明（构造函数参数）**

| 参数名                 | 类型     | 说明                  |
|---------------------|--------|---------------------|
| agent_name          | str    | Agent 场景名称          |
| tokenizer           | object | 文本分词器对象             |
| sampling_params     | dict   | 模型推理时的采样参数          |
| max_prompt_length   | int    | 输入提示的最大长度，默认 128k   |
| max_response_length | int    | 输出响应的最大长度，默认 8k     |
| n_parallel_agents   | int    | 并行执行的 Agent 数量，默认 8 |
| max_steps           | int    | Agent 执行的最大步骤数，默认 5 |

**文件位置**: `aura/runner/agent_engine_wrapper/base_engine_wrapper.py`

---

## 二、注册表接口

注册表接口用于注册自定义的Agent。

### 2.4 AGENTS_MAPPING - Agent 配置映射

**功能描述**

Agent 配置映射，存储已注册的 Agent 配置信息。

**数据结构**

```python
AGENTS_MAPPING = [
    {
        "name": "my_agent",
        ...
    }
]


def get_agent_by_name(name: str) -> Optional[dict]:
    for agent_config in AGENTS_MAPPING:
        if name == agent_config.get("name", ""):
            return agent_config

    return None
```

**配置项说明**：

| 字段                             | 类型       | 说明                                 |
|--------------------------------|----------|------------------------------------|
| `name`                         | str      | Agent 名称，配置文件中通过 `agent_name` 引用   |
| `env_class`                    | class    | 环境类，必须继承自 `BaseEnv`                |
| `env_args`                     | dict     | 环境初始化参数，传递给 `env_class` 构造函数       |
| `agent_class`                  | class    | Agent 类，必须继承自 `BaseAgent`          |
| `agent_args`                   | dict     | Agent 初始化参数，传递给 `agent_class` 构造函数 |
| `compute_trajectory_reward_fn` | callable | 轨迹奖励计算函数，用于计算最终奖励                  |

**使用方式**：

在配置文件中通过 `agent_name` 引用已注册的 Agent：

```yaml
agent_instances:
  - name: MY-AGENT
    executor_kwargs:
      agent_engine: rllm
      agent_engine_kwargs:
        agent_name: my_agent    # 引用注册的 Agent 名称
```

**文件位置**: `agents/agents_mapping.py`

---

## 三、数据类

数据类定义了 Agent 运行过程中的核心数据结构。

### 3.1 Step - 单步数据

**功能描述**

记录 Agent 运行的单步信息，包含对话上下文、动作、观测、奖励等。

**类定义**

```python
@dataclass
class Step:
    chat_completions: list[dict[str, str]] = field(default_factory=list)
    thought: str = ""
    action: Any = None
    observation: Any = None
    model_response: str = ""
    info: dict = field(default_factory=dict)
    reward: float = 0.0
    done: bool = False
    mc_return: float = 0.0
    step_id: int = 0
```

**参数说明**

| 参数名              | 类型                   | 说明                                                |
|------------------|----------------------|---------------------------------------------------|
| chat_completions | list[dict[str, str]] | 推理所有的完整对话上下文（含历史轮次），用于构造模型输入                      |
| thought          | str                  | 模型回复中 `<think>` 标签内的内容，表示模型在本步骤的内部推理              |
| action           | Any                  | 模型回复中 `<tool call>` 标签内的内容，表示模型决定执行的动作（如工具调用）     |
| observation      | Any                  | 本步骤接收到的外部观测：第 0 轮为用户原始提问，后续轮次为上一轮动作的执行结果（如工具返回）   |
| model_response   | str                  | 大模型生成的完整回复内容（即 `'role': 'assistant'` 的 `content`） |
| info             | dict                 | 附加信息字典，默认为空，可用于记录工具 ID、耗时等元数据                     |
| reward           | float                | 本步骤获得的即时奖励，默认为 `0.0`，反映当前动作的质量                    |
| done             | bool                 | 是否在本步骤终止轨迹，默认为 `False`，标识任务是否完成                   |
| mc_return        | float                | 从本步骤开始的 Monte Carlo 回报，默认为 `0.0`，用于策略梯度训练         |
| step_id          | int                  | 步骤编号                                              |

**文件位置**: `aura/runner/agent_engine_wrapper/base/agent/base_agent.py`

---

### 3.2 Trajectory - 轨迹数据

**功能描述**

记录 Agent 运行的完整轨迹信息，包含所有步骤和整体奖励。

**类定义**

```python
@dataclass
class Trajectory:
    task: Any = None
    steps: list[Step] = field(default_factory=list)
    reward: float = 0.0
    toolcall_reward: float = 0.0
    res_reward: float = 0.0
    prompt_id: int = 0
    data_id: str = None
    training_id: str = None
    epoch_id: int = 0
    iteration_id: int = 0
    sample_id: int = 0
    trajectory_id: int = 0
    application_id: str = ""
    termination_reason: str = "unknown"
```

**参数说明**

| 参数名                | 类型         | 说明     |
|--------------------|------------|--------|
| task               | Any        | 原始任务输入 |
| steps              | list[Step] | 所有步骤列表 |
| reward             | float      | 轨迹总奖励  |
| toolcall_reward    | float      | 工具调用奖励 |
| res_reward         | float      | 最终结果奖励 |
| termination_reason | str        | 终止原因   |

**文件位置**: `aura/runner/agent_engine_wrapper/base/agent/base_agent.py`

---

### 3.3 Action - 动作数据

**功能描述**

记录 Agent 决定的动作信息。

**类定义**

```python
@dataclass
class Action:
    action: Any = None
```

**文件位置**: `aura/runner/agent_engine_wrapper/base/agent/base_agent.py`

---

### 3.4 AgentTask - 任务数据

**功能描述**

定义 Agent 任务的数据结构。

**类定义**

```python
class AgentTask(BaseModel):
    task_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    sample_id: int
    iteration: int
    agent_name: str
    problem: str
    ground_truth: str = ""
    prompt_id: int = 0
    content: str = ""
    extra_args: dict[str, Any] = None
```

**参数说明**

| 参数名          | 类型   | 说明       |
|--------------|------|----------|
| task_id      | str  | 任务唯一标识   |
| sample_id    | int  | 样本编号     |
| iteration    | int  | 迭代次数     |
| agent_name   | str  | Agent 名称 |
| problem      | str  | 问题描述     |
| ground_truth | str  | 正确答案     |
| content      | str  | 额外内容     |
| extra_args   | dict | 额外参数     |

**文件位置**: `aura/runner/agent_engine_wrapper/base_engine_wrapper.py`

---

## 四、配置文件

### 服务启动说明

**命令格式**

```bash
bash scripts/start_rl_with_verl_vllm.sh
```

程序通过脚本 `start_rl_with_verl_vllm.sh` 启动，真实的 Python 入口为 `aura/start.py`，脚本内部通过以下指令执行：

```bash
python aura/start.py --config-name=${CONFIG_NAME} 2>&1 | tee ${LOG_PATH}/train_unit_${timestamp}.log
```

> 注意：运行 `start.py` 前，需先执行相关脚本以准备前置组件（如环境初始化、依赖服务启动等），确保运行环境就绪后再启动训练任务。

**hosts.conf 文件设置说明**

服务启动需要设置hosts.conf，该文件位于aura/configs目录下，用于设置单机或者双机部署，单机部署共卡模式，双机部署分离模式，具体示例详情可见“[修改hosts.conf](./03_quick_start.md#修改hostsconf)”。

**参数说明**

| 参数名                | 说明                           |
|--------------------|------------------------------|
| host               | 节点IP                         |
| index              | 当前节点的索引值。从0开始计数，用于区分不同节点     |
| train_master_index | 训练任务的主节点索引。为1时该节点启动训练任务      |
| infer_master_index | 推理任务的主节点索引。默认为0，为1时该节点启动推理任务 |

1. 单机部署共卡模式，单机设置单个节点，设置train_master_index和infer_master_index均为1。

2. 双机部署分离模式，双机设置两个节点，只需设置train_master_index分别为0和1，其中设置train_master_index为0是推理节点，设置为1是训练节点。

**base.conf 文件设置说明**

服务启动需要设置base.conf文件，该文件位于aura/configs目录下，用于设置工作模式与启动的配置文件，详情可见“[修改base.conf](./03_quick_start.md#修改baseconf)”。

**参数说明**

| 参数名               | 说明                                                                 |
|-------------------|--------------------------------------------------------------------|
| work_mode         | 工作模式。可以设置为hybrid（共卡模式）或者设置为one_step_off（全异步分离模式），需与hosts.conf的设置一致 |
| train_config_name | 训练yaml配置文件名                                                        |
| infer_config_name | 推理yaml配置文件名。共卡模式该配置不生效                                             |
| monitor_cmd       | 启动脚本名。需要监控的启动脚本，用于区分训练后端（verl和msrl）                                |
| max_retries       | 断点续训重试次数。默认为100次                                                   |
| clean_old_ckpt    | 第一次启动是否需要清空ckpt文件夹。0为不清理，1为需要清理                                    |

---

### 主配置文件参数

训练主配置文件采用YAML格式，主要包含以下几个部分：

1. **agentic_ai**: 全局配置，包括运行模式、日志级别等
2. **serve_conf**: 服务化部署配置（SERVE模式）
3. **direct_conf**: 直连模式配置（DIRECT模式）
4. **verl_conf**: 训练配置参数（verl后端）
5. **train_instances**: 训练任务实例配置
6. **agent_instances**: Agent服务实例配置
7. **infer_instances**: 推理服务实例配置

> 在分离模式（`one_step_off`）下，推理服务由独立的推理配置文件（`vllm_infer_*.yaml`
> ）驱动部署，详见 [推理服务配置参数](#推理服务配置参数)。主配置文件中的 `infer_instances` 仅用于服务发现与参数引用。

#### agentic_ai 配置

| 参数名       | 类型  | 说明     | 约束                                    |
|-----------|-----|--------|---------------------------------------|
| mode      | str | 运行模式选择 | 可选值为"serve"（服务化部署）或"direct"（直连模式）     |
| log_level | str | 日志级别   | 可选值为"DEBUG"、"INFO"、"WARNING"、"ERROR"等 |
| log_dir   | str | 日志目录路径 | 路径必须存在且具有写入权限                         |

#### serve_conf配置（SERVE模式）

| 参数名  | 类型  | 说明     | 约束                |
|------|-----|--------|-------------------|
| host | str | 服务监听地址 | 默认值为"0.0.0.0"     |
| port | int | 服务监听端口 | 应为有效的端口号，默认值为8030 |

#### direct_conf 配置（DIRECT模式）

| 参数名         | 类型   | 说明      | 约束                                |
|-------------|------|---------|-----------------------------------|
| entrypoints | list | 任务入口点列表 | 包含job_type、job_name、job_kwargs等字段 |

---

### 推理服务配置参数

分离模式（`one_step_off`）下，推理服务由独立的推理配置文件（`vllm_infer_*.yaml`
）驱动部署。配置文件示例参考 [configs/infer](../../../aura/configs/infer) 目录。使用前请根据实际环境修改以下配置项：

| 配置项                | 说明       |
|--------------------|----------|
| `infer_model_path` | 推理模型权重路径 |

#### 基础配置

| 参数名                    | 类型   | 说明       | 约束                                                                    |
|------------------------|------|----------|-----------------------------------------------------------------------|
| vllm_version           | str  | vLLM 版本号 | 应为有效的vLLM版本字符串                                                        |
| infer_model_name       | str  | 推理模型名称   | 应与主配置文件 `infer_instances.executor_kwargs.engine_kwargs.model_name` 对应 |
| infer_model_path       | str  | 模型权重路径   | 路径必须存在且包含完整的模型文件                                                      |
| enable_expert_parallel | bool | 是否开启专家并行 | MOE模型默认需要开启，Dense模型需要关闭，默认值为false                                     |

#### 部署模式配置

| 参数名                    | 类型  | 说明                    | 约束                                    |
|------------------------|-----|-----------------------|---------------------------------------|
| pd_mode                | int | 是否开启PD分离              | 1：PD分离（Prefill/Decode分离），0：PD混部，默认值为0 |
| prefill_instance_count | int | prefill实例数量           | 应为大于0的整数                              |
| decode_instance_count  | int | decode实例数量            | 应为非负整数，PD混部时需配置为0                     |
| tensor_parallel_size   | int | prefill/decode 张量并行大小 | 应为大于0的整数                              |
| data_parallel_size     | int | prefill/decode 数据并行大小 | 应为大于0的整数                              |

#### 推理性能配置

| 参数名                     | 类型    | 说明                          | 约束                  |
|-------------------------|-------|-----------------------------|---------------------|
| max_model_len           | int   | 模型最大上下文长度                   | 应为大于0的整数            |
| max_num_batched_tokens  | int   | 单次批量处理的最大token数             | 应为大于0的整数            |
| gpu_memory_utilization  | float | GPU显存利用率上限                  | 应为0到1之间的浮点数，默认值为0.6 |
| max_num_seqs            | int   | 最大并发序列数                     | 应为大于0的整数            |
| cudagraph_capture_sizes | list  | 图模式（Graph模式）的capture size列表 | 应为正整数列表             |

#### 高级配置

| 参数名                            | 类型   | 说明                     | 约束                           |
|--------------------------------|------|------------------------|------------------------------|
| kv_backend                     | str  | KV cache在PD节点之间同步传输的后端 | 可选值为"mooncake"或"llmdatadist" |
| enable_vllm_stat               | bool | 是否开启vLLM统计             | 默认值为false                    |
| enable_tensor_similarity_check | bool | 是否开启权重相似度检查            | 默认值为false                    |
| vllm_ascend_enable_flashcomm   | int  | 是否开启flash comm算法       | 0关闭，1开启                      |
| vllm_ascend_enable_flashcomm1  | int  | 是否开启flash comm1算法      | 0关闭，1开启                      |
| tool_call_enable               | bool | 是否开启tool call支持        | 默认值为false                    |
| use_vllm_opt                   | bool | 是否使用优化版vLLM            | 默认值为false                    |

---

### 训练配置参数（verl后端）

#### verl_conf.extras 配置

| 参数名           | 类型  | 说明           | 约束                        |
|---------------|-----|--------------|---------------------------|
| agent_service | str | 依赖的Agent服务名称 | 应与agent_instances中的name对应 |
| infer_service | str | 依赖的推理服务名称    | 应与infer_instances中的name对应 |

#### verl_conf.algorithm 配置

| 参数名             | 类型    | 说明     | 约束                          |
|-----------------|-------|--------|-----------------------------|
| adv_estimator   | str   | 优势评估器  | 可选值为"grpo"或"gae"，默认值为"grpo" |
| kl_ctrl.kl_coef | float | KL散度系数 | 应为大于0的浮点数，默认值为0.001         |

#### verl_conf.data 配置

| 参数名                     | 类型   | 说明                         | 约束                                                                    |
|-------------------------|------|----------------------------|-----------------------------------------------------------------------|
| train_files             | str  | 训练数据文件路径                   | 路径必须存在，支持 `.parquet`格式                                                |
| val_files               | str  | 验证数据文件路径                   | 路径必须存在，支持 `.parquet`格式                                                |
| train_batch_size        | int  | 训练批次大小                     | 应为大于0的整数，默认值为16                                                       |
| max_prompt_length       | int  | 最大提示词长度                    | 应为大于0的整数，默认值为2048                                                     |
| max_response_length     | int  | 最大响应长度                     | 应为大于0的整数，默认值为2048                                                     |
| filter_overlong_prompts | bool | 是否过滤超过max_prompt_length的样本 | 默认值为True                                                              |
| truncation              | str  | 截断方式                       | 默认值为"error"。verl 默认仅支持 "error" 模式，其他截断方式（left、right、middle）需自定义数据集类实现 |

#### verl_conf.actor_rollout_ref.model 配置

| 参数名                           | 类型   | 说明                                      | 约束               |
|-------------------------------|------|-----------------------------------------|------------------|
| path                          | str  | 模型权重路径                                  | 路径必须存在且包含完整的模型文件 |
| use_remove_padding            | bool | 是否在训练时"移除 padding token，只对真实 token 做计算" | 默认值为False        |
| enable_gradient_checkpointing | bool | 用"以计算换显存"的方式，减少训练时的显存占用                 | 默认值为True         |

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

#### verl_conf.actor_rollout_ref.actor 配置

| 参数名                          | 类型    | 说明                              | 约束                                                        |
|------------------------------|-------|---------------------------------|-----------------------------------------------------------|
| strategy                     | str   | 训练分布式策略                         | 可选值为"megatron"、"fsdp"、"fsdp2"、"veomni"，默认值为"megatron"     |
| optim.lr                     | float | 学习率                             | 应为大于0的浮点数，默认值为5e-7                                        |
| entropy_coeff                | float | 熵正则化系数，控制模型"要不要多探索"             | 应为非负数，默认值为0.001                                           |
| ppo_mini_batch_size          | int   | 每次 PPO 更新时，用多少样本做一次梯度更新         | 应为大于0的整数，默认值为2                                            |
| ppo_micro_batch_size_per_gpu | int   | 每张卡一次 forward/backward 实际处理的样本数 | 应为大于0的整数，默认值为2                                            |
| use_kl_loss                  | bool  | 是否把 KL loss 添加到 loss 里作为惩罚项     | 默认值为True，与 use_kl_in_reward 互斥使用                          |
| kl_loss_coef                 | float | KL loss的系数                      | 应为0与1之间的浮点数，默认值为0.001                                     |
| kl_loss_type                 | str   | KL loss计算方式                     | 可选值为"kl"、"abs"、"mse"、"low_var_kl"、"full"，默认值为"low_var_kl" |

#### verl_conf.actor_rollout_ref.actor.megatron 配置

| 参数名                                        | 类型   | 说明                                                     | 约束             |
|--------------------------------------------|------|--------------------------------------------------------|----------------|
| seed                                       | int  | 随机种子                                                   | 应为非负整数，默认值为0   |
| pipeline_model_parallel_size               | int  | 流水线并行大小                                                | 应为大于0的整数，默认值为1 |
| tensor_model_parallel_size                 | int  | 张量并行大小                                                 | 应为大于0的整数，默认值为4 |
| override_transformer_config.use_flash_attn | bool | 是否使用Flash Attention（减少显存访问 + 融合 kernel 的高效attention实现） | 默认值为True       |

#### verl_conf.actor_rollout_ref.rollout 配置

| 参数名                               | 类型    | 说明                                                                         | 约束                  |
|-----------------------------------|-------|----------------------------------------------------------------------------|---------------------|
| prompt_length                     | int   | 提示词长度                                                                      | 应为大于0的整数，默认值为2048   |
| response_length                   | int   | 响应长度                                                                       | 应为大于0的整数，默认值为2048   |
| agent.agent_loop_manager_class    | str   | Agent循环管理器类路径                                                              | 应为有效的Python类路径      |
| log_prob_micro_batch_size_per_gpu | int   | 每张卡处理micro_batch_size的个数                                                   | 应为大于0的整数，默认值为2      |
| enable_chunked_prefill            | bool  | 是否把"prompt 预填充（prefill）阶段"拆成多个 chunk 来执行，开启时，对于长prompt，可有效减少KV cache 的显存占用 | 默认值为False           |
| tensor_model_parallel_size        | int   | 张量并行大小                                                                     | 应为大于0的整数，默认值为4      |
| name                              | str   | 推理引擎名称                                                                     | 默认值为"vllm"          |
| gpu_memory_utilization            | float | rollout 推理引擎（比如vllm）允许占用的 GPU 显存比例上限                                       | 应为0到1之间的浮点数，默认值为0.6 |
| n                                 | int   | 每个 prompt 生成多少条候选 response                                                 | 应为大于0的整数，默认值为2      |
| do_sample                         | bool  | 是否进行采样                                                                     | 默认值为False           |

#### verl_conf.actor_rollout_ref.ref 配置

| 参数名                                   | 类型  | 说明                       | 约束             |
|---------------------------------------|-----|--------------------------|----------------|
| log_prob_micro_batch_size_per_gpu     | int | 每张卡处理micro_batch_size的个数 | 应为大于0的整数，默认值为2 |
| megatron.pipeline_model_parallel_size | int | 流水线并行大小                  | 应为大于0的整数，默认值为2 |
| megatron.tensor_model_parallel_size   | int | 张量并行大小                   | 应为大于0的整数，默认值为2 |

#### verl_conf.trainer 配置

| 参数名              | 类型   | 说明                                                                   | 约束                              |
|------------------|------|----------------------------------------------------------------------|---------------------------------|
| val_before_train | bool | 是否在训练前进行验证，测量"初始模型水平"，用于判断训练是否真的有效                                   | 默认值为False                       |
| device           | str  | 训练设备                                                                 | 可选值为"npu"、"gpu"、"cpu"，默认值为"npu" |
| critic_warmup    | int  | Actor 更新延迟开关。只有当当前 global step 大于等于 critic_warmup 时，才会执行 Actor 模型的更新 | 应为非负整数，默认值为0                    |
| project_name     | str  | 项目名称                                                                 | 用于标识训练项目                        |
| experiment_name  | str  | 实验名称                                                                 | 用于标识训练实验                        |
| n_gpus_per_node  | int  | 每个节点的GPU/NPU数量                                                       | 应为大于0的整数，默认值为8                  |
| nnodes           | int  | 节点数量                                                                 | 应为大于0的整数，默认值为1                  |
| save_freq        | int  | 每隔多少 step 保存一次 checkpoint                                            | 应为-1或大于0的整数，-1表示不保存，默认值为-1      |
| test_freq        | int  | 每隔多少 step 做一次评估（evaluation）                                          | 应为-1或大于0的整数，-1表示不进行测试，默认值为-1    |
| total_epochs     | int  | 整个训练数据集要完整跑多少轮                                                       | 应为大于0的整数，默认值为1                  |
| logger           | list | 日志记录器                                                                | 可包含"console"、"tensorboard"等     |

---

### 训练实例配置

#### train_instances 配置

| 参数名                            | 类型   | 说明          | 约束                                   |
|--------------------------------|------|-------------|--------------------------------------|
| name                           | str  | 训练服务名称      | 必须唯一，建议使用有意义的命名                      |
| executor_num                   | int  | 执行器数量       | 应为大于0的整数，默认值为1                       |
| executor_kwargs.cluster_mode   | str  | 集群模式        | 可选值为"hybrid"（共卡）、"one_step_off"（分离）  |
| executor_kwargs.train_engine   | str  | 训练引擎        | 可选值为"mindspeed_rl"、"verl"等，默认为"verl" |
| executor_kwargs.train_config   | dict | 训练配置        | 应引用verl_conf配置                       |
| executor_kwargs.rollout_config | dict | Rollout配置   | 默认为空字典                               |
| executor_kwargs.agent_service  | str  | Agent服务名称   | 应引用agent_instances中的name             |
| executor_kwargs.infer_service  | str  | 推理服务名称      | 应引用infer_instances中的name             |
| resource_info                  | list | 整个训练服务的资源描述 | 默认为空列表                               |

---

### Agent实例配置

#### agent_instances 配置

| 参数名                                                        | 类型   | 说明                   | 约束                                    |
|------------------------------------------------------------|------|----------------------|---------------------------------------|
| name                                                       | str  | Agent服务名称            | 必须唯一，建议使用有意义的命名                       |
| executor_num                                               | int  | 执行器数量                | 应为大于0的整数，默认值为1                        |
| executor_kwargs.agent_engine                               | str  | Agent引擎类型            | 可选值为"rllm"等                           |
| executor_kwargs.agent_engine_kwargs.agent_name             | str  | Agent名称              | 应为有效的内置Agent名称，如"math"                |
| executor_kwargs.agent_engine_kwargs.simplify_think_content | bool | 是否简化思考内容             | 默认值为false                             |
| executor_kwargs.agent_engine_kwargs.max_steps              | int  | 最大步数                 | 应为大于0的整数，默认值为5                        |
| executor_kwargs.agent_engine_kwargs.max_prompt_length      | int  | agent 输入 prompt 最大长度 | 应为大于0的整数，默认值为2048                     |
| executor_kwargs.agent_engine_kwargs.max_model_len          | int  | 模型最大上下文长度            | 应为大于0的整数，默认值为4096                     |
| executor_kwargs.agent_engine_kwargs.n_parallel_agents      | int  | 并行Agent数量            | 应为大于0的整数，默认值为1024                     |
| executor_kwargs.agent_engine_kwargs.tokenizer              | str  | 分词器路径                | 应为有效的分词器路径                            |
| executor_kwargs.infer_service_params                       | dict | 推理服务参数               | 包含top_p、temperature、max_tokens、model等 |
| executor_kwargs.trajectory_save_dir                        | str  | 轨迹保存目录               | 应为有效的文件路径                             |
| resource_info                                              | list | 整个agent服务的资源描述       | 默认为空列表                                |

---

### 推理实例配置

#### infer_instances 配置

| 参数名                                               | 类型   | 说明                      | 约束                                       |
|---------------------------------------------------|------|-------------------------|------------------------------------------|
| name                                              | str  | 推理服务名称                  | 必须唯一，建议使用模型名称                            |
| executor_num                                      | int  | 执行器数量                   | 应为大于0的整数，默认值为1                           |
| executor_kwargs.engine                            | str  | 推理引擎类型                  | 默认值为"vllm_proxy"，表示通过 HTTP 调 vLLM server |
| executor_kwargs.engine_kwargs.chat_server         | str  | vLLM HTTP 推理服务地址        | 应为有效的HTTP/HTTPS地址                        |
| executor_kwargs.engine_kwargs.prefill_server_list | list | 专门处理 prefill的 server 列表 | 可为空列表                                    |
| executor_kwargs.engine_kwargs.decode_server_list  | list | 专门处理 decode的 server 列表  | 可为空列表                                    |
| executor_kwargs.engine_kwargs.model_name          | str  | 推理服务加载的模型名称             | 应为有效的模型名称                                |
| resource_info                                     | list | 整个推理服务的资源描述             | 默认为空列表                                   |

---

### 配置文件示例

以下是一个完整的配置文件示例（verl 后端，hybrid 模式），其他配置文件参考[configs](../../../aura/configs)
目录。使用前请根据实际环境修改以下配置项：

| 配置项                         | 说明           |
|-----------------------------|--------------|
| `hydra.searchpath`          | Hydra 配置搜索路径 |
| `train_files` / `val_files` | 训练集和验证集路径    |
| `model.path`                | 模型权重路径       |
| `tokenizer`                 | 分词器路径        |
| `trajectory_save_dir`       | 轨迹数据保存路径     |

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
    - file:///path/to/AgentSDK/aura/configs/train/verl_conf

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
        agent_loop_manager_class: aura.trainer.train_adapter.verl.hybrid.agent_loop_manager.HybridAgentLoopManager
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
    test_freq: -1
    total_epochs: 1
    logger: [ 'console','tensorboard' ]

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
        prefill_server_list: [ ]
        decode_server_list: [ ]
        model_name: Qwen2.5-7B-Instruct
    resource_info: [ ]
```
