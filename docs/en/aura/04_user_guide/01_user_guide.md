# User Guide

This document describes the core usage of Agent SDK, including running modes, custom agent development, and configuration file usage.

## Running Modes

Agent SDK supports two running modes.

| Mode | Description |
| ---- | ----------- |
| **Direct mode** | Runs locally and executes training and inference tasks. |
| **Serve mode** | Runs as a service and provides HTTP APIs. Only inference is supported. |

### Direct Mode

Direct mode is used to execute training and inference tasks. The process exits automatically after the tasks are completed. This mode supports both training and inference.

**Starting Agent SDK**

```bash
cd /home/work/AgentSDK/aura
bash scripts/start_rl_with_verl_vllm.sh
```

**Configuration Example**

```yaml
agentic_ai:
  mode: direct

direct_conf:
  entrypoints:
    - job_type: train
      job_name: ${train_instances.0.name}
      job_kwargs: {}
```

### Serve Mode

After Serve mode is started, the system continues running and provides HTTP APIs. This mode supports inference only and does not support training.

**Starting Agent SDK**

```bash
cd /home/work/AgentSDK/aura
bash scripts/start_rl_with_verl_vllm.sh
```

**Configuration Example**

```yaml
agentic_ai:
  mode: serve

serve_conf:
  host: "0.0.0.0"
  port: 8030
```

**APIs**

Serve mode provides the following HTTP APIs.

| API Path | HTTP Method | Description |
| --- | --- | --- |
| `/agent/invoke` | POST | Generates agent trajectories and returns results in SSE (Server-Sent Events) streaming mode. |
| `/v1/chat/completions` | POST | Standard chat completions API that supports both streaming (`stream=true`) and non-streaming (`stream=false`) modes. |
| `/` | GET | Health check API that returns a welcome message. |
| `/delay` | GET | Latency test API for measuring service response times. |

**API Call Examples**

```bash
# Generate an agent trajectory
curl -X POST http://localhost:8030/agent/invoke \
  -H "Content-Type: application/json" \
  -d '{"sample_id": 1, "iteration": 0, "agent_name": "math", "problem": "What is 2+2?"}'

# Chat completions (non-streaming)
curl -X POST http://localhost:8030/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "qwen", "messages": [{"role": "user", "content": "Hello"}], "stream": false}'
```

---

## Custom Agent Development

For details, see [Custom Agent Integration](04_custom_agent.md).

---

## Engine Wrapper Integration Example

You can implement a custom engine wrapper by inheriting from `BaseEngineWrapper`.

> **Reference Implementations**
>
> - Base class: `aura/runner/agent_engine_wrapper/base_engine_wrapper.py`
>
> - RLLM engine wrapper: `aura/runner/agent_engine_wrapper/rllm/rllm_engine_wrapper.py`
>
> - SmolAgent engine wrapper: `aura/runner/agent_engine_wrapper/smolagent/smolagent_wrapper.py`

The following is a mock example for debugging and verification.

### Implementing `MockEngineWrapper`

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
        """Generate a mock trajectory."""
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

### Using the Configuration

Modify the configuration file to specify the path to the custom engine.

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

## Hydra Configuration

The configuration files use the Hydra framework for configuration management. It supports flexible configuration management through hierarchical composition, key-based merging, and overriding.

### Configuration Loading Mechanism

The Hydra configuration system uses hierarchical composition to load and merge configurations through two key configuration options: `hydra.searchpath` and `defaults`.

#### `hydra.searchpath`

Specifies the search paths for loading predefined configuration templates.

```yaml
hydra:
  searchpath:
    - file:///verl/verl/trainer/config    # Default configuration path of verl
    - file://path/to/AgentSDK/aura/configs/train/verl_conf  # Custom configuration path of the project
```

**Notes**

> - `file:///verl/verl/trainer/config`: points to the default configuration directory of the verl framework.
>
> - `file://path/to/AgentSDK/aura/configs/train/verl_conf`: points to the custom configuration directory of the project.
>
> - Hydra searches for configuration files in the order specified in the list.

#### `defaults`

Specifies the loading order and merging method of configuration files, determining the final configuration.

```yaml
defaults:
  - ppo_megatron_trainer           # Load the default PPO Megatron trainer configuration of verl
  - ppo_megatron_trainer@verl_conf # Load the custom verl_conf configuration of the project
  - _self_                         # Load the configuration of the current YAML file last
```

### Configuration Merge Order (from Top to Bottom)

```text
Default verl configuration (ppo_megatron_trainer)
    ↓ Merge by key
Custom project configuration (verl_conf)
    ↓ Merge by key
Current YAML configuration (_self_)
```

**Merge Rules**

1. **Hierarchical composition**: Configuration files are loaded in the order specified in the `defaults` list.

2. **Key-based merging**: Configuration items at the same level are merged by key. A later configuration overrides an earlier configuration with the same key.

3. **Overriding**: `_self_` indicates that the configuration of the current YAML file has the highest priority and overrides all previously loaded configurations with the same keys.

**Example**

Assume that the configuration is structured as follows.

```yaml
# Default verl configuration (ppo_megatron_trainer.yaml)
trainer:
  n_gpus_per_node: 8
  device: gpu

# Custom project configuration (verl_conf/ppo_megatron_trainer.yaml)
trainer:
  device: npu
  save_freq: 1000

# Current YAML configuration
trainer:
  save_freq: -1
  project_name: 'my_project'
```

**Final Merged Configuration**

```yaml
trainer:
  n_gpus_per_node: 8      # From the default verl configuration
  device: npu             # Overridden by the project configuration
  save_freq: -1           # Overridden by the current YAML configuration
  project_name: 'my_project'  # From the current YAML configuration
```

For a complete configuration file example, see [Configuration File Example](../05_api_python.md#configuration-file-example).

---

## Registering a Custom Engine

Agent SDK provides a registry mechanism for registering custom Agents.

### Registering an Agent Engine

Agent engines are registered through the `AGENTS_MAPPING` list. Add the configuration to `agents/agents_mapping.py`.

```python
from agents.agents_mapping import AGENTS_MAPPING
from agents.math_agent.environment.tool_env import ToolEnvironment
from agents.math_agent.tool_agent import ToolAgent
from agents.math_agent.reward.reward_fn import math_reward_fn
from aura.runner.agent_engine_wrapper.base.environment.env_utils import compute_trajectory_reward

# Register a custom Agent
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

## More Practice Guides

- Hybrid Mode (On-Policy): [02_hybrid.md](02_hybrid.md)
- One-Step-Off Mode (Off-Policy): [03_one_step_off.md](03_one_step_off.md)
- Custom Agent Integration: [04_custom_agent.md](04_custom_agent.md)

---

## Related Documentation

| Document | Description |
| -------- | ----------- |
| [Python API Reference](../05_api_python.md) | Python APIs exposed by the framework |
| [Quick Start](../03_quick_start.md) | Quick start guide |
| [Appendix](../10_appendix.md) | List of supported backends and models |
