# Python API Reference

> Note: You can develop applications with Agent SDK through the Python API. From the perspective of code invocation, every Python-side API can be called. This chapter lists only the external APIs provided by the service. Do not call other undocumented APIs directly.

Agent SDK is a training, inference, and tuning framework for agents. It supports integration with any agent engine, training engine, or inference engine. This document describes the core APIs that the framework exposes.

---

## 1. Core Base Classes

The core base classes are abstract classes that you must inherit from and implement to customize core components such as agents, environments, and tools.

### 1.1 `BaseAgent` - Agent Abstract Base Class

**Function Description**

The agent abstract base class. It interacts with the model, maintains the conversation state, parses model responses, and records trajectories. You must inherit from this class to implement a custom agent.

**Class Definition**

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

**Abstract Method Descriptions**

| Method Name         | Description                                                        |
|---------------------|--------------------------------------------------------------------|
| `update_from_env`   | Receives observations, rewards, and termination signals from the environment and updates the internal state of the agent |
| `update_from_model` | Receives a response from the model, parses it, and returns an action |
| `reset`             | Resets the agent state and starts a new trajectory                 |

**File location**: `aura/runner/agent_engine_wrapper/base/agent/base_agent.py`

---

### 1.2 `BaseEnv` - Environment Abstract Base Class

**Function Description**

The environment abstract base class. It executes tools, computes rewards, and manages states. You must inherit from this class to implement a custom environment.

**Class Definition**

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

**Abstract Method Descriptions**

| Method Name | Description                                                  |
|-------------|--------------------------------------------------------------|
| `reset`     | Resets the environment and returns the initial observation and additional information |
| `step`      | Executes an action and returns (observation, reward, whether the episode terminates, additional information) |
| `from_dict` | Creates an environment instance from a configuration dictionary |

**File location**: `aura/runner/agent_engine_wrapper/base/environment/base_env.py`

---

### 1.3 `BaseEngineWrapper` - Engine Wrapper Base Class

**Function Description**

The engine wrapper abstract base class, which provides a unified adaptation API for agent engines. You can inherit from this class to integrate with different agent engines.

**Class Definition**

```python
class BaseEngineWrapper(ABC):
    @abstractmethod
    async def generate_trajectory(self, task: AgentTask, stream_queue=None, *args, **kwargs) -> "Trajectory": ...
```

**Parameters (Constructor Parameters)**

| Parameter Name      | Type   | Description                                          |
|---------------------|--------|------------------------------------------------------|
| agent_name          | str    | Agent scenario name                                  |
| tokenizer           | object | Text tokenizer object                                |
| sampling_params     | dict   | Sampling parameters for model inference              |
| max_prompt_length   | int    | Maximum length of the input prompt, default: 128k    |
| max_response_length | int    | Maximum length of the output response, default: 8k   |
| n_parallel_agents   | int    | Number of agents executed in parallel, default: 8    |
| max_steps           | int    | Maximum number of steps that the agent executes, default: 5 |

**File location**: `aura/runner/agent_engine_wrapper/base_engine_wrapper.py`

---

## 2. Registry APIs

The registry APIs are used to register custom agents.

### 2.1 `AGENTS_MAPPING` - Agent Configuration Mapping

**Function Description**

The agent configuration mapping, which stores the configuration information of registered agents.

**Data Structure**

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

**Configuration Item Descriptions**:

| Field                          | Type     | Description                                                         |
|--------------------------------|----------|---------------------------------------------------------------------|
| name                         | str      | Agent name. It is referenced by `agent_name` in the configuration file |
| env_class                    | class    | Environment class, which must inherit from `BaseEnv`                |
| env_args                     | dict     | Environment initialization parameters, which are passed to the `env_class` constructor |
| agent_class                  | class    | Agent class, which must inherit from `BaseAgent`                    |
| agent_args                   | dict     | Agent initialization parameters, which are passed to the `agent_class` constructor |
| compute_trajectory_reward_fn | callable | Trajectory reward computation function, which computes the final reward |

**Usage**:

Reference a registered agent by `agent_name` in the configuration file:

```yaml
agent_instances:
  - name: MY-AGENT
    executor_kwargs:
      agent_engine: rllm
      agent_engine_kwargs:
        agent_name: my_agent    # Name of the registered Agent to reference
```

**File location**: `agents/agents_mapping.py`

---

## 3. Data Classes

The data classes define the core data structures used during an agent run.

### 3.1 `Step` - Single-Step Data

**Function Description**

Records the single-step information of an agent run, including the conversation context, the action, the observation, and the reward.

**Class Definition**

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

**Parameters**

| Parameter Name   | Type                 | Description                                                                                                         |
|------------------|----------------------|---------------------------------------------------------------------------------------------------------------------|
| chat_completions | list[dict[str, str]] | The complete conversation context of inference, including historical turns, which is used to construct the model input |
| thought          | str                  | The content within the `<think>` tag in the model response, which represents the internal reasoning of the model at this step |
| action           | Any                  | The content within the `<tool call>` tag in the model response, which represents the action that the model decides to execute, such as a tool call |
| observation      | Any                  | The external observation received at this step. For round 0, it is the original question of the user, and for subsequent rounds, it is the execution result of the action from the previous round, such as a tool return value |
| model_response   | str                  | The complete response content generated by the LLM, that is, the `content` of `'role': 'assistant'`                |
| info             | dict                 | Additional information dictionary. It is empty by default and can record metadata such as tool IDs and elapsed time |
| reward           | float                | The immediate reward obtained at this step, default 0.0, which reflects the quality of the current action        |
| done             | bool                 | Whether the trajectory terminates at this step, default `False`, which indicates whether the task is complete      |
| mc_return        | float                | The Monte Carlo return starting from this step, default 0.0, which is used for policy gradient training          |
| step_id          | int                  | Step number                                                                                                         |

**File location**: `aura/runner/agent_engine_wrapper/base/agent/base_agent.py`

---

### 3.2 `Trajectory` - Trajectory Data

**Function Description**

Records the complete trajectory information of an agent run, including all steps and the overall reward.

**Class Definition**

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

**Parameters**

| Parameter Name     | Type       | Description               |
|--------------------|------------|---------------------------|
| task               | Any        | Original task input       |
| steps              | list[Step] | List of all steps         |
| reward             | float      | Total trajectory reward   |
| toolcall_reward    | float      | Tool call reward          |
| res_reward         | float      | Final result reward       |
| termination_reason | str        | Termination reason        |

**File location**: `aura/runner/agent_engine_wrapper/base/agent/base_agent.py`

---

### 3.3 `Action` - Action Data

**Function Description**

Records the action information that the agent decides.

**Class Definition**

```python
@dataclass
class Action:
    action: Any = None
```

**File location**: `aura/runner/agent_engine_wrapper/base/agent/base_agent.py`

---

### 3.4 `AgentTask` - Task Data

**Function Description**

Defines the data structure of an agent task.

**Class Definition**

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

**Parameters**

| Parameter Name | Type | Description             |
|----------------|------|-------------------------|
| task_id        | str  | Unique task identifier  |
| sample_id      | int  | Sample number           |
| iteration      | int  | Number of iterations    |
| agent_name     | str  | Agent name              |
| problem        | str  | Problem description     |
| ground_truth   | str  | Correct answer          |
| content        | str  | Additional content      |
| extra_args     | dict | Additional parameters   |

**File location**: `aura/runner/agent_engine_wrapper/base_engine_wrapper.py`

---

## 4. Configuration Files

### Service Startup Instructions

**Command Format**

```bash
bash scripts/start_rl_with_verl_vllm.sh
```

You start the program with the `start_rl_with_verl_vllm.sh` script. The actual Python entry point is `aura/start.py`, which the script runs with the following command:

```bash
python aura/start.py --config-name=${CONFIG_NAME} 2>&1 | tee ${LOG_PATH}/train_unit_${timestamp}.log
```

> Note: Before you run `start.py`, run the related scripts to prepare the prerequisite components, such as environment initialization and dependency service startup. Start the training task only after the runtime environment is ready.

**`hosts.conf` File Settings**

You must set `hosts.conf` to start the service. This file is located in the `aura/configs` directory and is used to configure single-node or two-node deployment. Single-node deployment uses co-located mode, and two-node deployment uses disaggregation mode. For a detailed example, see [Modifying hosts.conf](./03_quick_start.md#modifying-hostsconf).

**Parameters**

| Parameter Name     | Description                                                                                |
|--------------------|--------------------------------------------------------------------------------------------|
| host               | Node IP address                                                                            |
| index              | Index value of the current node. It starts from 0 and distinguishes different nodes        |
| train_master_index | Index of the master node for the training task. When it is 1, the node starts the training task |
| infer_master_index | Index of the master node for the inference task. The default value is 0. When it is 1, the node starts the inference task |

1. For single-node deployment in co-located mode, configure a single node for the single machine, and set both `train_master_index` and `infer_master_index` to 1.

2. For two-node deployment in disaggregation mode, configure two nodes for the two machines, and set `train_master_index` to 0 and 1 respectively. The node whose `train_master_index` is set to 0 is the inference node, and the node whose `train_master_index` is set to 1 is the training node.

**`base.conf` File Settings**

You must set the `base.conf` file to start the service. This file is located in the `aura/configs` directory and is used to set the working mode and the configuration files to start. For details, see [Modifying base.conf](./03_quick_start.md#modifying-baseconf).

**Parameters**

| Parameter Name    | Description                                                                                          |
|-------------------|------------------------------------------------------------------------------------------------------|
| work_mode         | Working mode. It can be set to hybrid (co-located mode) or one_step_off (fully asynchronous disaggregation mode), and it must be consistent with the `hosts.conf` settings |
| train_config_name | Training YAML configuration file name                                                                 |
| infer_config_name | Inference YAML configuration file name. This configuration does not take effect in co-located mode    |
| monitor_cmd       | Startup script name. The startup script to monitor, which distinguishes the training backend (verl and msrl) |
| max_retries       | Number of retries for training resumption. The default value is 100                                    |
| clean_old_ckpt    | Whether to clear the ckpt folder on the first startup. 0 means no cleanup, and 1 means cleanup is required |

---

### Main Configuration File Parameters

The main training configuration file uses the YAML format and mainly contains the following parts:

1. `agentic_ai`: Global configuration, including the running mode and the log level
2. `serve_conf`: Service-based deployment configuration (SERVE mode)
3. `direct_conf`: Direct mode configuration (DIRECT mode)
4. `verl_conf`: Training configuration parameters (verl backend)
5. `train_instances`: Training task instance configuration
6. `agent_instances`: agent service instance configuration
7. `infer_instances`: Inference service instance configuration

> In disaggregation mode (`one_step_off`), a separate inference configuration file (`vllm_infer_*.yaml`)
) drives inference service deployment. For details, see [Inference Service Configuration Parameters](#inference-service-configuration-parameters). The `infer_instances` section in the main configuration file is used only for service discovery and parameter references.

#### `agentic_ai` Configuration

| Parameter Name | Type | Description          | Constraints                                                        |
|-----------|-----|--------|---------------------------------------|
| mode      | str | Running mode selection | Valid values are "serve" (service-based deployment) and "direct" (direct mode) |
| log_level | str | Log level   | Valid values are "DEBUG", "INFO", "WARNING", "ERROR", and so on    |
| log_dir   | str | Log directory path | The path must exist and be writable                                |

#### `serve_conf` Configuration (SERVE Mode)

| Parameter Name  | Type  | Description          | Constraints                |
|------|-----|--------|-------------------|
| host | str | Service listening address | The default value is "0.0.0.0"     |
| port | int | Service listening port | It must be a valid port number. The default value is 8030 |

#### `direct_conf` Configuration (DIRECT Mode)

| Parameter Name         | Type   | Description              | Constraints                                |
|-------------|------|---------|-----------------------------------|
| entrypoints | list | List of task entry points | It contains fields such as `job_type`, `job_name`, and `job_kwargs` |

---

### Inference Service Configuration Parameters

In disaggregation mode (`one_step_off`), a separate inference configuration file (`vllm_infer_*.yaml`)
) drives inference service deployment. For a configuration file example, see the [configs/infer](../../../aura/configs/infer) directory. Before use, modify the following configuration items based on your actual environment:

| Configuration Item | Description       |
|--------------------|----------|
| `infer_model_path` | Path to the inference model weights |

#### Basic Configuration

| Parameter Name                    | Type   | Description       | Constraints                                                                    |
|------------------------|------|----------|-----------------------------------------------------------------------|
| vllm_version           | str  | vLLM version number | It must be a valid vLLM version string                                                        |
| infer_model_name       | str  | Inference model name   | It must correspond to `infer_instances.executor_kwargs.engine_kwargs.model_name` in the main configuration file |
| infer_model_path       | str  | Path to the model weights   | The path must exist and contain complete model files                                                      |
| enable_expert_parallel | bool | Whether to enable expert parallelism | It must be enabled by default for MoE models and disabled for dense models. The default value is false                                     |

#### Deployment Mode Configuration

| Parameter Name                    | Type  | Description                    | Constraints                                    |
|------------------------|-----|-----------------------|---------------------------------------|
| pd_mode                | int | Whether to enable PD disaggregation              | 1: PD disaggregation (Prefill/Decode disaggregation), 0: PD co-location. The default value is 0 |
| prefill_instance_count | int | Number of prefill instances           | It must be an integer greater than 0                              |
| decode_instance_count  | int | Number of decode instances            | It must be a non-negative integer. Set it to 0 for PD co-location                     |
| tensor_parallel_size   | int | Prefill/decode tensor parallelism size | It must be an integer greater than 0                              |
| data_parallel_size     | int | Prefill/decode data parallelism size | It must be an integer greater than 0                              |

#### Inference Performance Configuration

| Parameter Name                     | Type    | Description                          | Constraints                  |
|-------------------------|-------|-----------------------------|---------------------|
| max_model_len           | int   | Maximum model context length                   | It must be an integer greater than 0            |
| max_num_batched_tokens  | int   | Maximum number of tokens processed in a single batch             | It must be an integer greater than 0            |
| gpu_memory_utilization  | float | Upper limit of the GPU memory utilization ratio                  | It must be a floating-point number between 0 and 1. The default value is 0.6 |
| max_num_seqs            | int   | Maximum number of concurrent sequences                     | It must be an integer greater than 0            |
| cudagraph_capture_sizes | list  | List of capture sizes for graph mode (Graph mode) | It must be a list of positive integers             |

#### Advanced Configuration

| Parameter Name                            | Type   | Description                     | Constraints                           |
|--------------------------------|------|------------------------|------------------------------|
| kv_backend                     | str  | Backend for synchronizing KV cache between PD nodes | Valid values are "mooncake" and "llmdatadist" |
| enable_vllm_stat               | bool | Whether to enable vLLM statistics             | The default value is false                    |
| enable_tensor_similarity_check | bool | Whether to enable the weight similarity check            | The default value is false                    |
| vllm_ascend_enable_flashcomm   | int  | Whether to enable the flash comm algorithm       | 0 disables it, and 1 enables it                      |
| vllm_ascend_enable_flashcomm1  | int  | Whether to enable the flash comm1 algorithm      | 0 disables it, and 1 enables it                      |
| tool_call_enable               | bool | Whether to enable tool call support        | The default value is false                    |
| use_vllm_opt                   | bool | Whether to use the optimized version of vLLM            | The default value is false                    |

---

### Training Configuration Parameters (verl Backend)

#### `verl_conf.extras` Configuration

| Parameter Name           | Type  | Description           | Constraints                        |
|---------------|-----|--------------|---------------------------|
| agent_service | str | Name of the agent service that this service depends on | It must correspond to name in agent_instances |
| infer_service | str | Name of the inference service that this service depends on    | It must correspond to name in infer_instances |

#### `verl_conf.algorithm` Configuration

| Parameter Name             | Type    | Description     | Constraints                          |
|-----------------|-------|--------|-----------------------------|
| adv_estimator   | str   | Advantage estimator  | Valid values are "grpo" and "gae". The default value is "grpo" |
| kl_ctrl.kl_coef | float | KL divergence coefficient | It must be a floating-point number greater than 0. The default value is 0.001         |

#### `verl_conf.data` Configuration

| Parameter Name                     | Type   | Description                         | Constraints                                                                    |
|-------------------------|------|----------------------------|-----------------------------------------------------------------------|
| train_files             | str  | Path to the training data file                   | The path must exist. The `.parquet` format is supported                                                |
| val_files               | str  | Path to the validation data file                   | The path must exist. The `.parquet` format is supported                                                |
| train_batch_size        | int  | Training batch size                     | It must be an integer greater than 0. The default value is 16                                                       |
| max_prompt_length       | int  | Maximum prompt length                    | It must be an integer greater than 0. The default value is 2048                                                     |
| max_response_length     | int  | Maximum response length                     | It must be an integer greater than 0. The default value is 2048                                                     |
| filter_overlong_prompts | bool | Whether to filter out samples that exceed `max_prompt_length` | The default value is True                                                              |
| truncation              | str  | Truncation method                       | The default value is "error". By default, verl supports only the "error" mode. Other truncation methods (left, right, and middle) require a custom dataset class |

#### `verl_conf.actor_rollout_ref.model` Configuration

| Parameter Name                           | Type   | Description                                      | Constraints               |
|-------------------------------|------|-----------------------------------------|------------------|
| path                          | str  | Path to the model weights                                  | The path must exist and contain complete model files |
| use_remove_padding            | bool | Whether to "remove padding tokens during training and compute only on real tokens" | The default value is False        |
| enable_gradient_checkpointing | bool | Reduces device memory usage during training by trading computation for device memory                 | The default value is True         |

> **About the `enable_gradient_checkpointing` parameter**:
>
> - **When disabled**:
>
>   ```text
>   forward → save all intermediate activation values → backward uses them directly
>   ```
>
>   Training speed increases, but a large amount of device memory is consumed.
>
> - **When enabled**:
>
>   ```text
>   forward → do not save intermediate activation values
>   backward → recompute forward once → then compute gradients
>   ```
>
>   Training becomes slower, but device memory usage drops significantly.

#### `verl_conf.actor_rollout_ref.actor` Configuration

| Parameter Name                          | Type    | Description                              | Constraints                                                        |
|------------------------------|-------|---------------------------------|-----------------------------------------------------------|
| strategy                     | str   | Training distributed strategy                         | Valid values are "megatron", "fsdp", "fsdp2", and "veomni". The default value is "megatron"     |
| optim.lr                     | float | Learning rate                             | It must be a floating-point number greater than 0. The default value is 5e-7                                        |
| entropy_coeff                | float | Entropy regularization coefficient, which controls whether the model explores more             | It must be a non-negative number. The default value is 0.001                                           |
| ppo_mini_batch_size          | int   | Number of samples used for one gradient update in each PPO update         | It must be an integer greater than 0. The default value is 2                                            |
| ppo_micro_batch_size_per_gpu | int   | Number of samples that each device actually processes in one forward/backward pass | It must be an integer greater than 0. The default value is 2                                            |
| use_kl_loss                  | bool  | Whether to add the KL loss to the loss as a penalty term     | The default value is True. It is mutually exclusive with `use_kl_in_reward`                          |
| kl_loss_coef                 | float | Coefficient of the KL loss                      | It must be a floating-point number between 0 and 1. The default value is 0.001                                     |
| kl_loss_type                 | str   | KL loss computation method                     | Valid values are "kl", "abs", "mse", "low_var_kl", and "full". The default value is "low_var_kl" |

#### `verl_conf.actor_rollout_ref.actor.megatron` Configuration

| Parameter Name                                        | Type   | Description                                                     | Constraints             |
|--------------------------------------------|------|--------------------------------------------------------|----------------|
| seed                                       | int  | Random seed                                                   | It must be a non-negative integer. The default value is 0   |
| pipeline_model_parallel_size               | int  | Pipeline parallelism size                                                | It must be an integer greater than 0. The default value is 1 |
| tensor_model_parallel_size                 | int  | Tensor parallelism size                                                 | It must be an integer greater than 0. The default value is 4 |
| override_transformer_config.use_flash_attn | bool | Whether to use Flash Attention (an efficient attention implementation that reduces device memory access and fuses kernels) | The default value is True       |

#### `verl_conf.actor_rollout_ref.rollout` Configuration

| Parameter Name                               | Type    | Description                                                                         | Constraints                  |
|-----------------------------------|-------|----------------------------------------------------------------------------|---------------------|
| prompt_length                     | int   | Prompt length                                                                      | It must be an integer greater than 0. The default value is 2048   |
| response_length                   | int   | Response length                                                                       | It must be an integer greater than 0. The default value is 2048   |
| agent.agent_loop_manager_class    | str   | Path to the agent loop manager class                                                              | It must be a valid Python class path      |
| log_prob_micro_batch_size_per_gpu | int   | Number of micro batches that each device processes                                                   | It must be an integer greater than 0. The default value is 2      |
| enable_chunked_prefill            | bool  | Whether to split the prompt prefill phase into multiple chunks. When enabled, it effectively reduces the device memory usage of the KV cache for long prompts | The default value is False           |
| tensor_model_parallel_size        | int   | Tensor parallelism size                                                                     | It must be an integer greater than 0. The default value is 4      |
| name                              | str   | Inference engine name                                                                     | The default value is "vllm"          |
| gpu_memory_utilization            | float | Upper limit of the GPU memory ratio that the rollout inference engine (for example, vLLM) can occupy                                       | It must be a floating-point number between 0 and 1. The default value is 0.6 |
| n                                 | int   | Number of candidate responses generated for each prompt                                                 | It must be an integer greater than 0. The default value is 2      |
| do_sample                         | bool  | Whether to sample                                                                     | The default value is False           |

#### `verl_conf.actor_rollout_ref.ref` Configuration

| Parameter Name                                   | Type  | Description                       | Constraints             |
|---------------------------------------|-----|--------------------------|----------------|
| log_prob_micro_batch_size_per_gpu     | int | Number of micro batches that each device processes | It must be an integer greater than 0. The default value is 2 |
| megatron.pipeline_model_parallel_size | int | Pipeline parallelism size                  | It must be an integer greater than 0. The default value is 2 |
| megatron.tensor_model_parallel_size   | int | Tensor parallelism size                   | It must be an integer greater than 0. The default value is 2 |

#### `verl_conf.trainer` Configuration

| Parameter Name              | Type   | Description                                                                   | Constraints                              |
|------------------|------|----------------------------------------------------------------------|---------------------------------|
| val_before_train | bool | Whether to validate before training, which measures the "initial model level" and is used to determine whether training is actually effective                                   | The default value is False                       |
| device           | str  | Training device                                                                 | Valid values are "npu", "gpu", and "cpu". The default value is "npu" |
| critic_warmup    | int  | Actor update delay switch. The Actor model is updated only when the current global step is greater than or equal to critic_warmup | It must be a non-negative integer. The default value is 0                    |
| project_name     | str  | Project name                                                                 | Used to identify the training project                        |
| experiment_name  | str  | Experiment name                                                                 | Used to identify the training experiment                        |
| n_gpus_per_node  | int  | Number of GPUs/NPUs per node                                                       | It must be an integer greater than 0. The default value is 8                  |
| nnodes           | int  | Number of nodes                                                                 | It must be an integer greater than 0. The default value is 1                  |
| save_freq        | int  | How often to save a checkpoint, in steps                                            | It must be -1 or an integer greater than 0. -1 means no saving. The default value is -1      |
| test_freq        | int  | How often to run an evaluation, in steps                                          | It must be -1 or an integer greater than 0. -1 means no evaluation. The default value is -1    |
| total_epochs     | int  | Number of complete passes over the entire training dataset                                                       | It must be an integer greater than 0. The default value is 1                  |
| logger           | list | Logger                                                                | It can include "console", "tensorboard", and so on     |

---

### Training Instance Configuration

#### `train_instances` Configuration

| Parameter Name                            | Type   | Description          | Constraints                                   |
|--------------------------------|------|-------------|--------------------------------------|
| name                           | str  | Training service name      | It must be unique. You are advised to use a meaningful name                      |
| executor_num                   | int  | Number of executors       | It must be an integer greater than 0. The default value is 1                       |
| executor_kwargs.cluster_mode   | str  | Cluster mode        | Valid values are "hybrid" (co-located) and "one_step_off" (disaggregated)  |
| executor_kwargs.train_engine   | str  | Training engine        | Valid values are "mindspeed_rl", "verl", and so on. The default value is "verl" |
| executor_kwargs.train_config   | dict | Training configuration        | It must reference the verl_conf configuration                       |
| executor_kwargs.rollout_config | dict | Rollout configuration   | The default value is an empty dictionary                               |
| executor_kwargs.agent_service  | str  | Agent service name   | It must reference name in agent_instances             |
| executor_kwargs.infer_service  | str  | Inference service name      | It must reference name in infer_instances             |
| resource_info                  | list | Resource description of the entire training service | The default value is an empty list                               |

---

### Agent Instance Configuration

#### `agent_instances` Configuration

| Parameter Name                                                        | Type   | Description                   | Constraints                                    |
|------------------------------------------------------------|------|----------------------|---------------------------------------|
| name                                                       | str  | Agent service name            | It must be unique. You are advised to use a meaningful name                       |
| executor_num                                               | int  | Number of executors                | It must be an integer greater than 0. The default value is 1                        |
| executor_kwargs.agent_engine                               | str  | Agent engine type            | Valid values are "rllm" and so on                           |
| executor_kwargs.agent_engine_kwargs.agent_name             | str  | Agent name              | It must be a valid built-in agent name, such as "math"                |
| executor_kwargs.agent_engine_kwargs.simplify_think_content | bool | Whether to simplify the thinking content             | The default value is false                             |
| executor_kwargs.agent_engine_kwargs.max_steps              | int  | Maximum number of steps                 | It must be an integer greater than 0. The default value is 5                        |
| executor_kwargs.agent_engine_kwargs.max_prompt_length      | int  | Maximum length of the prompt that the agent receives | It must be an integer greater than 0. The default value is 2048                     |
| executor_kwargs.agent_engine_kwargs.max_model_len          | int  | Maximum model context length            | It must be an integer greater than 0. The default value is 4096                     |
| executor_kwargs.agent_engine_kwargs.n_parallel_agents      | int  | Number of parallel agents            | It must be an integer greater than 0. The default value is 1024                     |
| executor_kwargs.agent_engine_kwargs.tokenizer              | str  | Tokenizer path                | It must be a valid tokenizer path                            |
| executor_kwargs.infer_service_params                       | dict | Inference service parameters               | It contains top_p, temperature, max_tokens, model, and so on |
| executor_kwargs.trajectory_save_dir                        | str  | Trajectory save directory               | It must be a valid file path                             |
| resource_info                                              | list | Resource description of the entire agent service       | The default value is an empty list                                |

#### Black-Box Agent Mode (vaee) Configuration

When `agent_engine` is set to `vaee`, black-box virtual engine mode is used. The agent runs in an external standalone service and communicates over HTTP. In addition to the preceding general parameters, configure the following parameters:

| Parameter Name | Type | Description | Constraints |
|--------|------|------|------|
| executor_kwargs.agent_engine | str | Agent engine | It must be vaee |
| executor_kwargs.agent_engine_kwargs.agent_name | str | Agent scenario name   | It corresponds to the proxy configuration in agents_mapping.py |
| executor_kwargs.agent_engine_kwargs.agent_proxy_args.agent_addr | str | HTTP address of the external agent service | It must be a valid communication address |
| executor_kwargs.agent_engine_kwargs.agent_proxy_args.traj_addr | str | TrajProxy service address | It must be consistent with traj_proxy_args.infer_url |
| executor_kwargs.agent_engine_kwargs.agent_proxy_args.run_id | str | Run identifier. TrajProxy groups and manages trajectories by this value | It must be a unique identifier |
| executor_kwargs.agent_engine_kwargs.traj_proxy_args.infer_url | str | TrajProxy service address, which is used to pull trajectories | It must be a valid communication address |
| executor_kwargs.agent_engine_kwargs.res_reward_func | str | Path to the step-level reward function (optional) | It must be a valid function path |
| executor_kwargs.agent_engine_kwargs.traj_refine_func | str | Path to the trajectory aggregation function (optional). If it is not configured, the default implementation is used | It must be a valid function path |

**Black-box Agent configuration example**:

```yaml
agent_instances:
  - name: proxy_agent
    executor_num: 1
    executor_kwargs:
      agent_engine: vaee
      agent_engine_kwargs:
        agent_name: proxy
        env_args:
          max_steps: 5
        traj_proxy_args:
          infer_url: "http://0.0.0.0:12300"
        agent_proxy_args:
          agent_addr: "http://0.0.0.0:28124"
          traj_addr: "http://0.0.0.0:12300"
          run_id: app_math_blackbox
        res_reward_func: "agents.proxy_agent.math.ozy_math_reward.math_res_reward_fn"
        traj_refine_func: "agents.proxy_agent.math.ozy_traj_refine_reward.ozy_token_traj_refine_func"
        tokenizer: ${verl_conf.actor_rollout_ref.model.path}
      infer_service_params:
        temperature: 1.0
        top_p: 1.0
        max_tokens: 8192
        model: ${infer_instances.0.name}
      trajectory_save_dir: ${hydra:runtime.cwd}/outputs
    resource_info: []
```

---

### Inference Instance Configuration

#### `infer_instances` Configuration

| Parameter Name                                               | Type   | Description                      | Constraints                                       |
|---------------------------------------------------|------|-------------------------|------------------------------------------|
| name                                              | str  | Inference service name                  | It must be unique. You are advised to use the model name                            |
| executor_num                                      | int  | Number of executors                   | It must be an integer greater than 0. The default value is 1                           |
| executor_kwargs.engine                            | str  | Inference engine type                  | The default value is "vllm_proxy", which means that vLLM server is called over HTTP |
| executor_kwargs.engine_kwargs.chat_server         | str  | vLLM HTTP inference service address        | It must be a valid HTTP/HTTPS address                        |
| executor_kwargs.engine_kwargs.prefill_server_list | list | List of servers dedicated to prefill | It can be an empty list                                    |
| executor_kwargs.engine_kwargs.decode_server_list  | list | List of servers dedicated to decode  | It can be an empty list                                    |
| executor_kwargs.engine_kwargs.model_name          | str  | Name of the model loaded by the inference service             | It must be a valid model name                                |
| resource_info                                     | list | Resource description of the entire inference service             | The default value is an empty list                                   |

---

### Configuration File Example

The following is a complete configuration file example (verl backend, hybrid mode). For other configuration files, see the [configs](../../../aura/configs)
directory. Before use, modify the following configuration items based on your actual environment:

| Configuration Item                         | Description           |
|-----------------------------|--------------|
| `hydra.searchpath`          | Hydra configuration search path |
| `train_files` / `val_files` | Paths to the training set and the validation set    |
| `model.path`                | Path to the model weights       |
| `tokenizer`                 | Tokenizer path        |
| `trajectory_save_dir`       | Path to save the trajectory data     |

```yaml
# Global configuration
agentic_ai:
  mode: direct
  log_level: DEBUG
  log_dir: /var/log/

# DIRECT mode configuration
direct_conf:
  entrypoints:
    - job_type: train
      job_name: ${train_instances.0.name}
      job_kwargs: { }

# Hydra configuration
hydra:
  searchpath:
    - file:///verl/verl/trainer/config
    - file:///path/to/AgentSDK/aura/configs/train/verl_conf

defaults:
  - ppo_megatron_trainer
  - ppo_megatron_trainer@verl_conf
  - _self_

# Training configuration
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

# Training instances
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

# Agent instances
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

# Inference instances
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

---

## 5. Black-Box Agent Training API Reference

### Overview

Virtual Agent Engine Execution Wrapper (VAEE) is the virtual engine of black-box agent mode. It communicates with external agent services over HTTP and performs trajectory aggregation and reward computation. VAEE mode treats the agent as a "black box". It does not care about the internal implementation of the agent and interacts with it only through standardized APIs.

### Exposed APIs

#### 5.1 `AgentProxyClient` — Sending Prompts to an External Agent

**Function Description**

Sends a prompt to the external agent service and triggers the agent loop to run.

**HTTP Request**

| Item | Description |
|------|------|
| Method | `POST` |
| Path | Specified by the `agent_proxy_args.agent_addr` configuration. The external agent service must expose `POST /v1/chat/completions` |
| Content-Type | `application/json` |

**Request Body (Sent to the External Agent Service)**

```json
{
  "model": "Qwen3-8B",
  "messages": [
    {"role": "system", "content": "You are a math assistant..."},
    {"role": "user", "content": "Solve: 1+2+...+100 = ?"}
  ],
  "infer_params": {
    "temperature": 1.0,
    "top_p": 1.0,
    "max_tokens": 8192,
    "model": "/path/to/model"
  },
  "extra_params": {},
  "infer_url": "http://trajproxy:12300/s/run_id/session_id/v1/"
}
```

**Parameters**

| Field | Type | Description |
|------|------|------|
| `model` | str | Model name, which the agent service uses to identify which model to use |
| `messages` | list | Conversation history, including system and user messages |
| `infer_params` | dict | Inference sampling parameters (temperature, top_p, max_tokens, and so on) |
| `extra_params` | dict | Extension parameters, a reserved field |
| `infer_url` | str | **TrajProxy address**. The agent **must** send requests to this address: `http://{traj_addr}/s/{run_id}/{session_id}/v1/` |

**File location**: `aura/runner/agent_engine_wrapper/proxy_client/agent_proxy_client.py`

#### 5.2 `TrajProxyClient` — Pulling Trajectory Records

**Function Description**

Pulls all LLM call records of a specified session from the TrajProxy service for subsequent trajectory aggregation and reward computation.

**HTTP Request**

| Item | Description |
|------|------|
| Method | `GET` |
| Path | `/trajectory?session_id={session_id}` |
| Service address | Specified by the `traj_proxy_args.infer_url` configuration |

**Response Body**

```json
{
  "records": [
    {
      "unique_id": "7f3d8a2e...",
      "session_id": "session_xxx",
      "messages": [
        {"role": "system", "content": "..."},
        {"role": "user", "content": "..."}
      ],
      "raw_request": {
        "model": "Qwen3-8B",
        "messages": [...],
        "stream": false
      },
      "raw_response": {
        "choices": [{"message": {"role": "assistant", "content": "..."}, "logprobs": {...}}]
      },
      "start_time": 1700000000.123,
      "end_time": 1700000002.456,
      "error_traceback": null
    }
  ]
}
```

**Parameters**

| Field | Type | Description |
|------|------|------|
| `records` | list | List of LLM call records, sorted by `start_time` |
| `unique_id` | str | Unique record identifier |
| `session_id` | str | Session ID, which associates all LLM calls of the same agent loop |
| `messages` | list | Complete conversation history sent in this LLM call |
| `raw_request` | dict | Raw LLM request body |
| `raw_response` | dict | Raw LLM response body (including logprobs) |
| `start_time` | float | Timestamp when the request is initiated |
| `end_time` | float | Timestamp when the response is returned |
| `error_traceback` | str/null | Error traceback. A value of null means that the call is normal |

**File location**: `aura/runner/agent_engine_wrapper/proxy_client/traj_proxy_client.py`

### 5.3 Black-Box Agent Integration Constraints

#### Constraint 1: Implementing the `POST /v1/chat/completions` HTTP API

The external agent service must expose the `POST /v1/chat/completions` API to receive the prompts that the training framework sends. The `AgentProxyClient` constructs the request body of this API, which contains fields such as `messages`, `infer_params`, and `infer_url`.

**Every LLM call in the agent loop must go through `infer_url`.** For every LLM call, the agent service **must send the request to `infer_url`** (that is, TrajProxy) instead of calling vLLM directly, which ensures that every LLM call passes through TrajProxy.

```python
# Correct: go through infer_url so that TrajProxy records the trajectory
url = f"{self.infer_url}/chat/completions"
response = await httpx.post(url, json=body)

# Incorrect: connect to vLLM directly, so TrajProxy receives no records and the trajectory is empty
response = await httpx.post("http://vllm:8000/v1/chat/completions", json=body)
```

**Minimal reference implementation**:

```python
from fastapi import FastAPI
from pydantic import BaseModel
import httpx

app = FastAPI()

class ChatCompletionRequest(BaseModel):
    model: str
    messages: list
    infer_params: dict = {}
    extra_params: dict = {}
    infer_url: str = ""

class LLMClient:
    """LLM call client, which must send requests to infer_url"""
    def __init__(self, infer_url: str, model: str, infer_params: dict):
        self.infer_url = infer_url.rstrip("/")
        self.model = model
        self.infer_params = infer_params
        self.client = httpx.AsyncClient(timeout=300.0)

    async def chat_completion(self, messages: list) -> dict:
        body = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            **{k: v for k, v in self.infer_params.items() if k != "model"},
        }
        # Key point: send the request to infer_url instead of connecting to vLLM directly
        url = f"{self.infer_url}/chat/completions"
        resp = await self.client.post(url, json=body)
        resp.raise_for_status()
        return resp.json()

class MyAgentSession:
    """Custom agent loop logic"""
    def __init__(self, infer_url, model, infer_params, max_steps=5):
        self.llm_client = LLMClient(infer_url, model, infer_params)
        self.messages = []
        self.max_steps = max_steps

    def initialize(self, system_prompt: str, problem: str):
        self.messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": problem},
        ]

    async def run(self):
        for step in range(self.max_steps):
            # 1. Call the LLM (through infer_url to TrajProxy)
            resp = await self.llm_client.chat_completion(self.messages)
            content = resp["choices"][0]["message"]["content"]
            self.messages.append({"role": "assistant", "content": content})

            # 2. Parse and execute tool calls (custom logic)
            # tool_calls = parse_tool_calls(content)
            # obs = execute_tools(tool_calls)
            # self.messages.extend(obs)
            pass

@app.post("/v1/chat/completions")
async def handle(body: ChatCompletionRequest):
    session = MyAgentSession(
        infer_url=body.infer_url,
        model=body.model,
        infer_params=body.infer_params,
    )
    # Extract the system prompt and the user problem from body.messages
    system_prompt = "You are a helpful assistant."
    problem = ""
    for msg in body.messages:
        if msg["role"] == "user":
            problem = msg.get("content", "")
            break
    session.initialize(system_prompt, problem)
    await session.run()
    return {"code": 0}
```

#### Constraint 2: The Trajectory Aggregation and Reward Functions Must Match the Signatures

If you customize `traj_refine_func` or `res_reward_func`, the function signatures must match the following API definitions. Otherwise, the training process is interrupted by a type error. If you do not configure them, the framework uses the default implementation.

##### `traj_refine_func` — Trajectory Aggregation Function

**Function description**: Aggregates the raw request and response records (`List[RequestRecord]`) pulled from TrajProxy into a trajectory-level trajectory (`Episode`).

**Function Example**

```python
def traj_refine_func(
    task_id: str,
    task: dict,
    records: List[RequestRecord],
    tokenizer=None,
    *args, **kwargs
) -> Episode:
```

**Core Types**

| Type | Description |
|------|------|
| `RequestRecord` | Raw LLM call record, which contains messages, raw_response, start_time, error_traceback, and so on |
| `Step` | Single-step data, which contains prompt_ids, response_ids, logprobs, reward, action, tool_outputs, and so on |
| `Trajectory` | Trajectory data, which contains multiple Steps and the overall reward |
| `Episode` | Contains one or more `Trajectory` objects, and each `Trajectory` contains multiple `Step` objects |

**Reference implementation**: `aura/agents/proxy_agent/math/ozy_traj_refine_reward.py`

##### `traj_reward_func` — Trajectory-Level Reward Function

**Function description**: Computes the final reward for the entire trajectory.

**Function Example**

```python
def traj_reward_func(
    episode: Episode,
    answer: Optional[str] = None,
    *args, **kwargs
) -> Episode:
```

**Reference implementation**: `aura/agents/proxy_agent/math/ozy_math_reward.py`
