# 使用指南<a name="ZH-CN_TOPIC_0000002492554189"></a>

本文档介绍 AgentSDK 的核心使用方法，包括运行模式、自定义 Agent 开发、配置文件使用等内容。

## 运行模式<a name="section_run_mode"></a>

AgentSDK 支持两种运行模式：

| 模式 | 说明 |
|------|------|
| **Serve 模式** | 服务化部署，提供 HTTP API 接口，仅支持推理 |
| **Direct 模式** | 本地部署，本地执行训练和推理任务 |

### Serve 模式

Serve 模式启动后，系统将长期运行并提供 HTTP API 服务。该模式仅支持推理功能，不支持训练。

**启动方式**

```bash
bash run_start_in_local.sh --config-name your_serve_config.yaml
```

**配置示例**

```yaml
agentic_ai:
  mode: serve

serve_conf:
  host: "0.0.0.0"
  port: 8030
```

**API 接口**

Serve 模式提供以下 HTTP API 接口：

| 接口路径 | HTTP 方法 | 说明 |
|---------|----------|------|
| `/agent/invoke` | POST | Agent 轨迹生成接口，以 SSE（Server-Sent Events）流式模式返回结果 |
| `/v1/chat/completions` | POST | 标准 Chat Completions 接口，支持流式（`stream=true`）和非流式（`stream=false`）两种模式 |
| `/` | GET | 健康检查接口，返回欢迎信息 |
| `/delay` | GET | 延迟测试接口，用于测试服务响应 |

**接口调用示例**

```bash
# Agent 轨迹生成
curl -X POST http://localhost:8030/agent/invoke \
  -H "Content-Type: application/json" \
  -d '{"sample_id": 1, "iteration": 0, "agent_name": "math", "problem": "What is 2+2?"}'

# Chat Completions（非流式）
curl -X POST http://localhost:8030/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "qwen", "messages": [{"role": "user", "content": "Hello"}], "stream": false}'
```

### Direct 模式

Direct 模式用于执行训练和推理任务，任务完成后自动退出。该模式同时支持训练和推理功能。

**启动方式**

```bash
bash run_start_in_local.sh --config-name your_direct_config.yaml
```

**配置示例**

```yaml
agentic_ai:
  mode: direct

direct_conf:
  entrypoints:
    - job_type: train
      job_name: ${train_instances.0.name}
      job_kwargs: {}
```

---

## 自定义 Agent 开发<a name="section_custom_agent"></a>

使用 AgentSDK 框架时，开发者需要实现以下核心组件：

### 核心组件

| 组件 | 基类 | 说明 | 参考实现 |
|------|------|------|----------|
| **Agent** | `BaseAgent` | 定义 Agent 的决策逻辑和状态管理 | `agents/math_agent/tool_agent.py` |
| **Environment** | `BaseEnv` | 定义 Agent 与外部环境的交互接口 | `agents/math_agent/environment/tool_env.py` |
| **Engine Wrapper** | `BaseEngineWrapper` | 封装 Agent 执行引擎 | `aura/runner/agent_engine_wrapper/rllm/rllm_engine_wrapper.py` |
| **Tool** | `Tool` | 定义 Agent 可调用的工具 | `agents/math_agent/environment/tools/tool_base.py` |
| **Reward Function** | `RewardFunction` | 定义奖励计算逻辑 | `agents/math_agent/reward/reward_fn.py` |

### 开发步骤

> **参考示例**：完整的 Agent 实现示例可在 `agents/math_agent` 目录中找到，包括 Agent、Environment、Tool、Reward 等组件的完整实现。
> **math_agent 目录结构**：
>
> ```text
> agents/math_agent/
> ├── tool_agent.py              # Agent 核心实现
> ├── environment/
> │   ├── tool_env.py           # 环境实现
> │   └── tools/                # 工具实现
> │       ├── tool_base.py      # 工具基类
> │       ├── multi_tool.py     # 多工具实现
> │       └── mcp_tool.py       # MCP工具实现
> ├── reward/
> │   ├── reward_fn.py          # 奖励函数
> │   ├── math_reward.py        # 数学奖励
> │   └── code_reward.py        # 代码奖励
> ├── parser/
> │   └── tool_parser/          # 工具解析器
> │       ├── qwen_tool_parser.py
> │       └── r1_tool_parser.py
> └── prompt/
>     └── system_prompts.py     # 系统提示词
> ```

#### 步骤 1：实现自定义 Agent

继承 `BaseAgent` 类，实现 Agent 的核心逻辑。参考实现：`agents/math_agent/tool_agent.py`

```python
from aura.runner.agent_engine_wrapper.base.agent.base_agent import BaseAgent, Action, Step, Trajectory

class MyAgent(BaseAgent):
    def __init__(self, system_prompt: str, tools: list = None):
        self.system_prompt = system_prompt
        self.tools = tools or []
        self._chat_completions = []
        self._trajectory = Trajectory()

    @property
    def chat_completions(self) -> list[dict[str, str]]:
        return self._chat_completions

    @property
    def trajectory(self) -> Trajectory:
        return self._trajectory

    def update_from_env(self, observation, reward, done, info, **kwargs):
        """从环境接收观测和奖励"""
        self._trajectory.reward = reward
        step = Step(
            observation=observation,
            reward=reward,
            done=done,
            info=info
        )
        self._trajectory.steps.append(step)

    def update_from_model(self, response: str, **kwargs) -> Action:
        """从模型响应解析动作"""
        action = self._parse_response(response)
        return Action(action=action)

    def reset(self):
        """重置 Agent 状态"""
        self._chat_completions = [
            {"role": "system", "content": self.system_prompt}
        ]
        self._trajectory = Trajectory()

    def _parse_response(self, response: str):
        """解析模型响应，提取动作"""
        # 实现响应解析逻辑
        pass
```

#### 步骤 2：实现自定义环境

继承 `BaseEnv` 类，定义环境交互逻辑。参考实现：`agents/math_agent/environment/tool_env.py`

```python
from aura.runner.agent_engine_wrapper.base.environment.base_env import BaseEnv
from typing import Any, tuple

class MyEnv(BaseEnv):
    def __init__(self, task: dict, max_steps: int = 10):
        self.task = task
        self.max_steps = max_steps
        self.current_step = 0
        self.done = False

    def reset(self) -> tuple[dict, dict]:
        """重置环境"""
        self.current_step = 0
        self.done = False
        observation = {"question": self.task.get("question", "")}
        return observation, {}

    def step(self, action: Any) -> tuple[Any, float, bool, dict]:
        """执行动作"""
        self.current_step += 1

        # 执行动作并获取观测
        observation = self._execute_action(action)

        # 计算奖励
        reward = self._compute_reward(action)

        # 判断是否终止
        self.done = self.current_step >= self.max_steps or self._check_success(action)

        return observation, reward, self.done, {}

    def close(self):
        """清理资源"""
        pass

    @staticmethod
    def from_dict(info: dict) -> "MyEnv":
        """从配置创建环境实例"""
        return MyEnv(task=info, max_steps=info.get("max_steps", 10))

    @staticmethod
    def is_multithread_safe() -> bool:
        """是否线程安全"""
        return True
```

#### 步骤 3：实现奖励函数

实现 `RewardFunction` 协议，定义奖励计算逻辑。参考实现：`agents/math_agent/reward/reward_fn.py`

```python
from agents.math_agent.reward.reward_types import RewardInput, RewardOutput

def my_reward_fn(task_info: dict, action: str) -> RewardOutput:
    """自定义奖励函数"""
    ground_truth = task_info.get("ground_truth", "")

    # 判断答案是否正确
    is_correct = action.strip() == ground_truth.strip()

    # 计算奖励
    reward = 1.0 if is_correct else 0.0

    return RewardOutput(
        reward=reward,
        is_correct=is_correct,
        metadata={"ground_truth": ground_truth}
    )
```

#### 步骤 4：注册 Agent 配置

将自定义组件注册到 Agent 映射表：

```python
from agents.agents_mapping import AGENTS_MAPPING

AGENTS_MAPPING.append({
    "name": "my_agent",
    "agent_class": MyAgent,
    "agent_args": {"system_prompt": "You are a helpful assistant."},
    "env_class": MyEnv,
    "env_args": {"max_steps": 10},
    "compute_trajectory_reward_fn": my_reward_fn,
})
```

#### 步骤 5：实现自定义工具（可选）

如果 Agent 需要调用工具，可以实现自定义 Tool 类。参考实现：

- 工具基类：`agents/math_agent/environment/tools/tool_base.py`

- 多工具实现：`agents/math_agent/environment/tools/multi_tool.py`

- MCP工具实现：`agents/math_agent/environment/tools/mcp_tool.py`

```python
from agents.math_agent.environment.tools.tool_base import Tool, ToolOutput

class MyTool(Tool):
    def __init__(self, name: str = "my_tool", description: str = "A custom tool"):
        super().__init__(name=name, description=description)

    def forward(self, *args, **kwargs) -> ToolOutput:
        # 实现工具执行逻辑
        result = self._execute_tool_logic(*args, **kwargs)
        return ToolOutput(name=self.name, output=result)

    def _execute_tool_logic(self, *args, **kwargs):
        # 具体的工具逻辑实现
        pass
```

---

## Engine Wrapper 对接示例<a name="section_engine_wrapper"></a>

开发者可通过继承 `BaseEngineWrapper` 实现自定义引擎包装器。

> **参考实现**：
>
> - 基类：`aura/runner/agent_engine_wrapper/base_engine_wrapper.py`
>
> - RLLM引擎包装器：`aura/runner/agent_engine_wrapper/rllm/rllm_engine_wrapper.py`
>
> - SmolAgent引擎包装器：`aura/runner/agent_engine_wrapper/smolagent/smolagent_wrapper.py`

以下是一个 Mock 示例，用于调试和验证。

### 实现 MockEngineWrapper

```python
import random
import torch
from typing import List
from aura.runner.agent_engine_wrapper.base_engine_wrapper import BaseEngineWrapper, AgentTask, Trajectory

class MockEngineWrapper(BaseEngineWrapper):
    def __init__(
            self,
            agent_name: str,
            tokenizer,
            sampling_params: dict,
            max_prompt_length: int = 128 * 1024,
            max_response_length: int = 8 * 1024,
            n_parallel_agents: int = 8,
            max_steps: int = 5
    ):
        super().__init__()
        self.agent_name = agent_name
        self.tokenizer = tokenizer
        self.sampling_params = sampling_params
        self.max_prompt_length = max_prompt_length
        self.max_response_length = max_response_length
        self.n_parallel_agents = n_parallel_agents
        self.max_steps = max_steps

    async def generate_trajectory(
            self,
            task: AgentTask,
            stream_queue=None,
            *args, **kwargs
    ) -> Trajectory:
        """生成模拟轨迹"""
        mock_data = {
            "prompt_tokens": torch.tensor([101, 200, 300, 400], dtype=torch.long),
            "response_tokens": torch.tensor([500, 600, 700], dtype=torch.long),
            "response_masks": torch.tensor([1, 1, 1], dtype=torch.long),
            "trajectory_reward": random.uniform(-1, 1),
            "idx": random.randint(0, 9999),
            "chat_completions": [
                {"role": "assistant", "content": "This is a mock response."}
            ],
            "metrics": {
                "steps": 3,
                "reward_time": 0.01,
                "env_time": 0.05,
                "llm_time": 0.02,
                "total_time": 0.08,
            },
        }
        return Trajectory(**mock_data)
```

### 配置使用

修改配置文件，指定自定义引擎路径：

```yaml
agent_instances:
  - name: MY-AGENT
    executor_num: 1
    executor_kwargs:
      agent_engine: mock
      agent_engine_kwargs:
        agent_name: my_agent
        tokenizer: /path/to/tokenizer
      infer_service_params:
        top_p: 1
        temperature: 1
        max_tokens: 4096
      trajectory_save_dir: /path/to/trajectory.jsonl
```

---

## Hydra配置说明

配置文件使用Hydra框架进行配置管理，采用分层组合、按key合并和覆盖的机制，支持灵活的配置管理。

### 配置加载机制

Hydra配置系统采用分层组合的方式，通过`hydra.searchpath`和`defaults`两个关键配置实现配置的加载和合并。

#### hydra.searchpath

指定配置文件的搜索路径，用于加载预定义的配置模板：

```yaml
hydra:
  searchpath:
    - file:///verl/verl/trainer/config    # verl原始默认配置路径
    - file://AgenticRL/configs/verl_conf  # 项目自定义配置路径
```

**说明**：

> - `file:///verl/verl/trainer/config`：指向verl框架的默认配置文件目录
>
> - `file://AgenticRL/configs/verl_conf`：指向项目自定义的配置文件目录
>
> - Hydra会按照列表顺序依次搜索配置文件

#### defaults

指定配置文件的加载顺序和合并方式，决定最终配置的组成：

```yaml
defaults:
  - ppo_megatron_trainer           # 加载verl默认的PPO Megatron训练器配置
  - ppo_megatron_trainer@verl_conf # 加载项目自定义的verl_conf配置
  - _self_                         # 最后加载当前yaml文件的配置
```

### 配置合并顺序（从上到下依次覆盖）

```text
verl默认配置 (ppo_megatron_trainer)
    ↓ 按key合并
项目自定义配置 (verl_conf)
    ↓ 按key合并
当前yaml配置 (_self_)
```

**合并规则**：

1. **分层组合**：按照defaults列表的顺序依次加载配置文件

2. **按key合并**：相同层级的配置项按key进行合并，后加载的配置会覆盖先加载的同名配置

3. **覆盖机制**：`_self_`表示当前yaml文件的配置具有最高优先级，会覆盖之前加载的所有同名配置

**示例说明**：

假设有以下配置结构：

```yaml
# verl默认配置 (ppo_megatron_trainer.yaml)
trainer:
  n_gpus_per_node: 8
  device: gpu

# 项目自定义配置 (verl_conf/ppo_megatron_trainer.yaml)
trainer:
  device: npu
  save_freq: 1000

# 当前yaml配置
trainer:
  save_freq: -1
  project_name: 'my_project'
```

**最终合并结果**：

```yaml
trainer:
  n_gpus_per_node: 8      # 来自verl默认配置
  device: npu             # 被项目配置覆盖
  save_freq: -1           # 被当前yaml覆盖
  project_name: 'my_project'  # 来自当前yaml
```

完整配置文件示例可参考：【[配置文件示例](../command_api.md#section_config_example)】

---

## 注册自定义引擎<a name="section_registry"></a>

AgentSDK 提供注册表机制，支持注册自定义的训练引擎、推理引擎和数据管理器。

### 注册训练引擎

```python
from aura.trainer.train_register import registry as train_registry

# 注册自定义训练引擎
train_registry.register(
    train_engine="my_train_engine",
    cluster_mode="hybrid",
    rollout_method=my_rollout_fn,
    train_method=my_train_fn
)
```

### 注册推理引擎

```python
from aura.runner.infer_adapter.infer_registry import registry as infer_registry

# 注册自定义推理后端
infer_registry.register("my_infer_backend", MyInferServer)
```

### 注册 Agent 引擎

Agent 引擎通过 `AGENTS_MAPPING` 列表进行注册，在 `agents/agents_mapping.py` 中添加配置：

```python
from agents.agents_mapping import AGENTS_MAPPING
from agents.math_agent.environment.tool_env import ToolEnvironment
from agents.math_agent.tool_agent import ToolAgent
from agents.math_agent.reward.reward_fn import math_reward_fn
from aura.runner.agent_engine_wrapper.base.environment.env_utils import compute_trajectory_reward

# 注册自定义 Agent
AGENTS_MAPPING.append({
    "name": "my_agent",
    "env_class": ToolEnvironment,
    "env_args": {
        "tools": ["python", "search"],
        "reward_fn": my_reward_fn,
        "tool_timeout": 120,
        "max_steps": 10,
    },
    "agent_class": ToolAgent,
    "agent_args": {
        "tools": ["python", "search"],
        "parser_name": "qwen",
        "system_prompt": "You are a helpful assistant...",
    },
    "compute_trajectory_reward_fn": compute_trajectory_reward,
})
```

### 注册数据管理器

```python
from aura.data_manager.data_registry import registry as data_registry

# 注册自定义数据管理器
data_registry.register(
    train_backend="my_backend",
    service_mode="train",
    cls=MyDataManager
)
```

---

## 更多 Agent 场景实践<a name="section_more_examples"></a>

- 数学 Math Agent 请参考：[Math Agent](math_agent.md)

---

## 相关文档<a name="section_related_docs"></a>

| 文档 | 说明 |
|------|------|
| [Python 接口说明](../api_python.md) | 框架对外暴露的 Python 接口 |
| [命令行接口说明](../command_api.md) | 命令行参数和配置文件说明 |
| [快速入门](../quick_start.md) | 快速上手指南 |
| [附录](../appendix.md) | 支持的后端和模型列表 |
