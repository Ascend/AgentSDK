# Usage Guide

This document describes the core usage of Agent SDK, including run modes, custom agent development, and configuration file usage.

## Run Modes

Agent SDK supports two run modes:

| Mode | Description |
|------|------|
| **Direct mode** | Local deployment, which runs training and inference tasks locally |
| **Serve mode** | Service-based deployment, which provides HTTP API endpoints and supports inference only |

### Direct Mode

Direct mode runs training and inference tasks and exits automatically after the tasks complete. This mode supports both training and inference.

**Startup**

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

After Serve mode starts, the system runs continuously and provides HTTP API services. This mode supports inference only and does not support training.

**Startup**

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

**API Endpoints**

Serve mode provides the following HTTP API endpoints:

| Endpoint Path | HTTP Method | Description |
|---------|----------|------|
| /agent/invoke | POST | Agent trajectory generation endpoint, which returns results in the Server-Sent Events (SSE) streaming mode |
| /v1/chat/completions | POST | Standard Chat Completions endpoint, which supports the streaming (`stream=true`) and non-streaming (`stream=false`) modes |
| / | GET | Health check endpoint, which returns a welcome message |
| /delay | GET | Latency test endpoint, which is used to test the service response |

**API Call Examples**

```bash
# Agent trajectory generation
curl -X POST http://localhost:8030/agent/invoke \
  -H "Content-Type: application/json" \
  -d '{"sample_id": 1, "iteration": 0, "agent_name": "math", "problem": "What is 2+2?"}'

# Chat Completions (non-streaming)
curl -X POST http://localhost:8030/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "qwen", "messages": [{"role": "user", "content": "Hello"}], "stream": false}'
```

---

## Custom Agent Development

For details, see the [Custom Agent Integration Guide](06_custom_agent.md).

---

## Engine Wrapper Integration Example

You can implement a custom engine wrapper by inheriting from `BaseEngineWrapper`.

> **Reference Implementations**:
>
> - Base class: `aura/runner/agent_engine_wrapper/base_engine_wrapper.py`
>
> - RLLM engine wrapper: `aura/runner/agent_engine_wrapper/rllm/rllm_engine_wrapper.py`
>
> - SmolAgent engine wrapper: `aura/runner/agent_engine_wrapper/smolagent/smolagent_wrapper.py`

The following is a mock example used for debugging and verification.

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
        """Generate a mock trajectory"""
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

### Configuration Usage

Modify the configuration file to specify the custom engine path:

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

Configuration files use the Hydra framework for configuration management, which adopts hierarchical composition along with key-based merging and overriding to support flexible configuration management.

### Configuration Loading Mechanism

The Hydra configuration system uses hierarchical composition and loads and merges configurations through the two key settings `hydra.searchpath` and `defaults`.

#### `hydra.searchpath`

Specify the search paths of configuration files to load predefined configuration templates:

```yaml
hydra:
  searchpath:
    - file:///verl/verl/trainer/config    # Path of the original verl default configuration
    - file://path/to/AgentSDK/aura/configs/train/verl_conf  # Path of the project custom configuration
```

**Notes**:

> - `file:///verl/verl/trainer/config`: Points to the default configuration file directory of the verl framework
>
> - `file://path/to/AgentSDK/aura/configs/train/verl_conf`: Points to the project custom configuration file directory
>
> - Hydra searches for configuration files in the order of the list

#### `defaults`

Specify the loading order and merge method of configuration files, which determine the composition of the final configuration:

```yaml
defaults:
  - ppo_megatron_trainer           # Load the verl default PPO Megatron trainer configuration
  - ppo_megatron_trainer@verl_conf # Load the project custom verl_conf configuration
  - _self_                         # Load the configuration of the current YAML file last
```

### Configuration Merge Order (Overridden from Top to Bottom)

```text
verl default configuration (ppo_megatron_trainer)
    ↓ Key-based merge
project custom configuration (verl_conf)
    ↓ Key-based merge
configuration of the current YAML file (_self_)
```

**Merge Rules**:

1. **Hierarchical composition**: Load configuration files in the order of the `defaults` list.

2. **Key-based merge**: Configuration items at the same level are merged by key, and a configuration loaded later overrides a configuration with the same name that was loaded earlier.

3. **Override mechanism**: `_self_` indicates that the configuration of the current YAML file has the highest priority and overrides all configurations with the same name that were loaded earlier.

**Example**:

Assume the following configuration structure:

```yaml
# verl default configuration (ppo_megatron_trainer.yaml)
trainer:
  n_gpus_per_node: 8
  device: gpu

# Project custom configuration (verl_conf/ppo_megatron_trainer.yaml)
trainer:
  device: npu
  save_freq: 1000

# Configuration of the current YAML file
trainer:
  save_freq: -1
  project_name: 'my_project'
```

**Final Merge Result**:

```yaml
trainer:
  n_gpus_per_node: 8      # From the verl default configuration
  device: npu             # Overridden by the project configuration
  save_freq: -1           # Overridden by the current YAML file
  project_name: 'my_project'  # From the current YAML file
```

For a complete configuration file example, see [Configuration File Example](../05_api_python.md#configuration-file-example).

---

## Registering Custom Engines

Agent SDK provides a registry mechanism that supports registering custom agents.

### Registering an Agent Engine

Register an agent engine through the `AGENTS_MAPPING` list by adding a configuration in `agents/agents_mapping.py`:

```python
from agents.agents_mapping import AGENTS_MAPPING
from agents.math_agent.environment.tool_env import ToolEnvironment
from agents.math_agent.tool_agent import ToolAgent
from agents.math_agent.reward.reward_fn import math_reward_fn
from aura.runner.agent_engine_wrapper.base.environment.env_utils import compute_trajectory_reward

# Register a custom agent
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

## More Practical Guides

- Co-located Training and Inference (On-Policy Strategy): [02_hybrid.md](02_hybrid.md)
- One-Step Asynchronous Disaggregation of Training and Inference (One Step Off Strategy): [03_one_step_off.md](03_one_step_off.md)
- Fully Asynchronous Disaggregation of Training and Inference (Fully Async Strategy): [04_fully_async.md](04_fully_async.md)
- Mixed Batch Scheduling: [05_mixed_batch.md](05_mixed_batch.md)
- Custom Agent Integration Guide: [06_custom_agent.md](06_custom_agent.md)

---

## Related Documents

| Document | Description |
|------|------|
| [Python API Reference](../05_api_python.md) | Python APIs exposed by the framework |
| [Quick Start](../03_quick_start.md) | Guide for getting started quickly |
| [Appendix](../10_appendix.md) | Supported backends and models |
