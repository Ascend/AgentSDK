# Python接口说明<a name="ZH-CN_TOPIC_0000002459514668"></a>

> 注意：AgentSDK可通过Python接口进行应用开发，从代码调用角度上来说所有Python侧接口都可以被调用。本章节仅列出业务提供的对外接口，其余未进行说明的接口用户请勿直接调用。

AgentSDK 是一个 Agent 训推调框架，支持对接任意 Agent 引擎、训练引擎、推理引擎。本文档介绍框架对外暴露的核心接口。

---

## 一、核心基类<a name="section_base_classes"></a>

核心基类是用户必须继承并实现的抽象类，用于自定义 Agent、环境、工具等核心组件。

### 1.1 BaseAgent - Agent 抽象基类<a name="section_base_agent"></a>

**功能描述**

Agent 抽象基类，负责与模型交互、维护对话状态、解析模型响应、记录轨迹。用户需要继承此类实现自定义 Agent。

**类定义**

```python
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

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

| 方法名 | 说明 |
|--------|------|
| `update_from_env` | 从环境接收观测、奖励、终止信号，更新 Agent 内部状态 |
| `update_from_model` | 从模型接收响应，解析并返回动作 |
| `reset` | 重置 Agent 状态，开始新的轨迹 |

**文件位置**: `aura/runner/agent_engine_wrapper/base/agent/base_agent.py`

---

### 1.2 BaseEnv - 环境抽象基类<a name="section_base_env"></a>

**功能描述**

环境抽象基类，负责工具执行、奖励计算、状态管理。用户需要继承此类实现自定义环境。

**类定义**

```python
from abc import ABC, abstractmethod
from typing import Any, tuple

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

| 方法名 | 说明 |
|--------|------|
| `reset` | 重置环境，返回初始观测和附加信息 |
| `step` | 执行动作，返回 (观测, 奖励, 是否终止, 附加信息) |
| `from_dict` | 从配置字典创建环境实例 |

**文件位置**: `aura/runner/agent_engine_wrapper/base/environment/base_env.py`

---

### 1.3 BaseEngineWrapper - 引擎包装器基类<a name="section_base_engine_wrapper"></a>

**功能描述**

引擎包装器抽象基类，提供统一的 Agent 引擎适配接口。用户可继承此类对接不同的 Agent 引擎。

**类定义**

```python
from abc import ABC, abstractmethod
from typing import List
from pydantic import BaseModel

class AgentTask(BaseModel):
    task_id: str
    sample_id: int
    iteration: int
    agent_name: str
    problem: str
    ground_truth: str = ""
    prompt_id: int = 0
    content: str = ""
    extra_args: dict = None

class BaseEngineWrapper(ABC):
    @abstractmethod
    async def generate_trajectory(self, task: AgentTask, stream_queue = None, *args, **kwargs) -> "Trajectory": ...
```

**参数说明**

| 参数名 | 类型 | 说明 |
|--------|------|------|
| agent_name | str | Agent 场景名称 |
| tokenizer | object | 文本分词器对象 |
| sampling_params | dict | 模型推理时的采样参数 |
| max_prompt_length | int | 输入提示的最大长度，默认 128K |
| max_response_length | int | 输出响应的最大长度，默认 8K |
| n_parallel_agents | int | 并行执行的 Agent 数量，默认 8 |
| max_steps | int | Agent 执行的最大步骤数，默认 5 |

**文件位置**: `aura/runner/agent_engine_wrapper/base_engine_wrapper.py`

---

## 二、注册表接口<a name="section_registry"></a>

注册表接口用于注册自定义的训练引擎、推理引擎、数据管理器等组件。

### 2.1 TrainRegistry - 训练引擎注册表<a name="section_train_registry"></a>

**功能描述**

训练引擎注册表，用于注册自定义的训练方法和 Rollout 方法。

**类定义**

```python
class TrainRegistry:
    def register(self, train_engine: str, cluster_mode: str,
                 rollout_method: Callable | None, train_method: Callable) -> None: ...

    def get_method(self, train_engine: str, cluster_mode: str) -> tuple | None: ...

# 全局实例
registry = TrainRegistry()
```

**已注册引擎**

| train_engine   | cluster_mode   | 说明                |
|----------------|----------------|-------------------|
| `verl`         | `hybrid`       | verl 共卡模式         |
| `verl`         | `one_step_off` | verl 全异步模式        |

**文件位置**: `aura/trainer/train_register.py`

---

### 2.2 InferBackendRegistry - 推理引擎注册表<a name="section_infer_registry"></a>

**功能描述**

推理引擎注册表，用于注册自定义的推理服务后端。

**类定义**

```python
class InferBackendRegistry:
    def register(self, name: str, cls: type) -> None: ...

    def get_class(self, name: str) -> type | None: ...

# 全局实例
registry = InferBackendRegistry()
```

**已注册后端**

| 名称 | 说明 |
|------|------|
| `vllm` | vLLM 推理服务 |
| `vllm_pd` | vLLM PD 分离推理服务 |

**文件位置**: `aura/runner/infer_adapter/infer_registry.py`

---

### 2.3 DataManagerRegistry - 数据管理器注册表<a name="section_data_registry"></a>

**功能描述**

数据管理器注册表，用于注册自定义的数据管理器。

**类定义**

```python
class DataManagerRegistry:
    def register(self, train_backend: str, service_mode: str, cls: type) -> None: ...

    def get_class(self, train_backend: str, service_mode: str) -> type | None: ...

# 全局实例
registry = DataManagerRegistry()
```

**已注册管理器**

| train_backend | service_mode | 数据管理器类 | 说明 |
|---------------|--------------|--------------|------|
| `verl` | `train` | `VerlDataManager` | verl 训练数据管理器 |
| `verl` | `infer` | `InferDataManager` | 统一推理数据管理器 |

**文件位置**: `aura/data_manager/data_registry.py`

---

### 2.4 AGENTS_MAPPING - Agent 配置映射<a name="section_agents_mapping"></a>

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

| 字段 | 类型 | 说明 |
|------|------|------|
| `name` | str | Agent 名称，配置文件中通过 `agent_name` 引用 |
| `env_class` | class | 环境类，必须继承自 `BaseEnv` |
| `env_args` | dict | 环境初始化参数，传递给 `env_class` 构造函数 |
| `agent_class` | class | Agent 类，必须继承自 `BaseAgent` |
| `agent_args` | dict | Agent 初始化参数，传递给 `agent_class` 构造函数 |
| `compute_trajectory_reward_fn` | callable | 轨迹奖励计算函数，用于计算最终奖励 |

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

## 三、数据类<a name="section_data_classes"></a>

数据类定义了 Agent 运行过程中的核心数据结构。

### 3.1 Step - 单步数据<a name="section_step"></a>

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

| 参数名 | 类型 | 说明 |
|--------|------|------|
| chat_completions | list[dict[str, str]] | 推理所有的完整对话上下文（含历史轮次），用于构造模型输入 |
| thought | str | 模型回复中 `<think>` 标签内的内容，表示模型在本步骤的内部推理 |
| action | Any | 模型回复中 `<tool call>` 标签内的内容，表示模型决定执行的动作（如工具调用） |
| observation | Any | 本步骤接收到的外部观测：第 0 轮为用户原始提问，后续轮次为上一轮动作的执行结果（如工具返回） |
| model_response | str | 大模型生成的完整回复内容（即 `'role': 'assistant'` 的 `content`） |
| info | dict | 附加信息字典，默认为空，可用于记录工具 ID、耗时等元数据 |
| reward | float | 本步骤获得的即时奖励，默认为 `0.0`，反映当前动作的质量 |
| done | bool | 是否在本步骤终止轨迹，默认为 `False`，标识任务是否完成 |
| mc_return | float | 从本步骤开始的 Monte Carlo 回报，默认为 `0.0`，用于策略梯度训练 |
| step_id | int | 步骤编号 |

---

### 3.2 Trajectory - 轨迹数据<a name="section_trajectory"></a>

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

| 参数名 | 类型 | 说明 |
|--------|------|------|
| task | Any | 原始任务输入 |
| steps | list[Step] | 所有步骤列表 |
| reward | float | 轨迹总奖励 |
| toolcall_reward | float | 工具调用奖励 |
| res_reward | float | 最终结果奖励 |
| termination_reason | str | 终止原因 |

---

### 3.3 Action - 动作数据<a name="section_action"></a>

**功能描述**

记录 Agent 决定的动作信息。

**类定义**

```python
@dataclass
class Action:
    action: Any = None
```

---

### 3.4 AgentTask - 任务数据<a name="section_agent_task"></a>

**功能描述**

定义 Agent 任务的数据结构。

**类定义**

```python
from pydantic import BaseModel, Field
import uuid

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

| 参数名 | 类型 | 说明 |
|--------|------|------|
| task_id | str | 任务唯一标识 |
| sample_id | int | 样本编号 |
| iteration | int | 迭代次数 |
| agent_name | str | Agent 名称 |
| problem | str | 问题描述 |
| ground_truth | str | 正确答案 |
| content | str | 额外内容 |
| extra_args | dict | 额外参数 |

---
