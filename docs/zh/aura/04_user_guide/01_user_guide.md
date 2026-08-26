# 使用指南

本文档介绍 Agent SDK 的核心使用方法，包括运行模式、自定义 Agent 开发、配置文件使用等内容。

## 运行模式

Agent SDK 支持两种运行模式：

| 模式 | 说明 |
|------|------|
| **Direct 模式** | 本地部署，本地执行训练和推理任务 |
| **Serve 模式** | 服务化部署，提供 HTTP API 接口，仅支持推理 |

### Direct 模式

Direct 模式用于执行训练和推理任务，任务完成后自动退出。该模式同时支持训练和推理功能。

**启动方式**

```bash
cd /home/work/AgentSDK/aura
bash scripts/start_rl_with_verl_vllm.sh
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

### Serve 模式

Serve 模式启动后，系统将长期运行并提供 HTTP API 服务。该模式仅支持推理功能，不支持训练。

**启动方式**

```bash
cd /home/work/AgentSDK/aura
bash scripts/start_rl_with_verl_vllm.sh
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

---

## 自定义 Agent 开发

详见 [自定义 Agent 接入指南](05_custom_agent.md)。

---

## Engine Wrapper 对接示例

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
      agent_engine: MockEngineWrapper
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
    - file://path/to/AgentSDK/aura/configs/train/verl_conf  # 项目自定义配置路径
```

**说明**：

> - `file:///verl/verl/trainer/config`：指向verl框架的默认配置文件目录
>
> - `file://path/to/AgentSDK/aura/configs/train/verl_conf`：指向项目自定义的配置文件目录
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

完整配置文件示例请参见“[配置文件示例](../05_api_python.md#配置文件示例)”。

---

## 注册自定义引擎

Agent SDK 提供注册表机制，支持注册自定义的Agent。

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

---

## 更多实践指南

- 训推共卡模式（On-Policy 策略）：[02_hybrid.md](02_hybrid.md)
- 训推单步异步分离模式（One Step Off 策略）：[03_one_step_off.md](03_one_step_off.md)
- 训推全异步分离模式（Fully Async 策略）：[04_fully_async.md](04_fully_async.md)
- 自定义 Agent 接入指南：[05_custom_agent.md](05_custom_agent.md)

---

## 相关文档

| 文档 | 说明 |
|------|------|
| [Python 接口说明](../05_api_python.md) | 框架对外暴露的 Python 接口 |
| [快速入门](../03_quick_start.md) | 快速上手指南 |
| [附录](../10_appendix.md) | 支持的后端和模型列表 |
