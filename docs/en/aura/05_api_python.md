# Python API Reference

> **Note:** Agent SDK supports application development through Python APIs. Although all Python APIs are technically callable from code, this document describes only the externally exposed APIs. Do not directly call APIs that are not documented here.

Agent SDK is an agent training, inference, and tuning framework that supports integration with arbitrary agent engines, training engines, and inference engines. This document describes the core APIs exposed by the framework.

---

## 1. Core Base Classes

Core base classes are abstract classes that users must inherit from and implement to customize core components such as agents, environments, and tools.

### 1.1 `BaseAgent` - Agent Abstract Base Class

**Description**

Handles model interaction, conversation state maintenance, model response parsing, and trajectory recording. Users must inherit from this class to implement custom agents.

**Class definition**

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

**Abstract methods**

| Method | Description |
| -------- | ------------- |
| `update_from_env` | Receives observations, rewards, and termination signals from the environment and updates the internal agent state. |
| `update_from_model` | Receives a response from the model, parses it, and returns an action. |
| `reset` | Resets the agent state and starts a new trajectory. |

**File location**: `aura/runner/agent_engine_wrapper/base/agent/base_agent.py`

---

### 1.2 `BaseEnv` - Environment Abstract Base Class

**Description**

Handles tool execution, reward calculation, and state management. Users must inherit from this class to implement custom environments.

**Class definition**

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

**Abstract methods**

| Method | Description |
| -------- | ------------- |
| `reset` | Resets the environment and returns the initial observation and additional information. |
| `step` | Executes an action and returns the observation, reward, termination status, and additional information. |
| `from_dict` | Creates an environment instance from a configuration dictionary. |

**File location**: `aura/runner/agent_engine_wrapper/base/environment/base_env.py`

---

### 1.3 `BaseEngineWrapper` - Engine Wrapper Base Class

**Description**

Provides a unified interface for adapting agent engines. Users can inherit from this class to integrate different agent engines.

**Class definition**

```python
class BaseEngineWrapper(ABC):
    @abstractmethod
    async def generate_trajectory(self, task: AgentTask, stream_queue=None, *args, **kwargs) -> "Trajectory": ...
```

**Parameters**

| Parameter | Type | Description |
| ----------- | ------ | ------------- |
| agent_name | str | Name of the agent. |
| tokenizer | object | Text tokenizer object. |
| sampling_params | dict | Sampling parameters used during model inference. |
| max_prompt_length | int | Maximum length of the input prompt. Default value: `128K`. |
| max_response_length | int | Maximum length of the output response. Default value: `8K`. |
| n_parallel_agents | int | Number of agents executed in parallel. Default value: `8`. |
| max_steps | int | Maximum number of steps an agent can take. Default value: `5`. |

**File location**: `aura/runner/agent_engine_wrapper/base_engine_wrapper.py`

---

## 2. Registry APIs

Registry APIs are used to register custom agents.

### 2.1 `AGENTS_MAPPING` - Agent Configuration Mapping

**Description**

Stores the configuration information of registered agents.

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

**Configuration items**

| Field | Type | Description |
| ------- | ------ | ------------- |
| `name` | str | Agent name. The name is referenced through `agent_name` in the configuration file. |
| `env_class` | class | Environment class. The class must inherit from `BaseEnv`. |
| `env_args` | dict | Environment initialization parameters passed to the constructor of `env_class`. |
| `agent_class` | class | Agent class. The class must inherit from `BaseAgent`. |
| `agent_args` | dict | Agent initialization parameters passed to the constructor of `agent_class`. |
| `compute_trajectory_reward_fn` | callable | Trajectory reward calculation function used to calculate the final reward. |

**Usage**

Reference a registered agent through `agent_name` in the configuration file.

```yaml
agent_instances:
  - name: MY-AGENT
    executor_kwargs:
      agent_engine: rllm
      agent_engine_kwargs:
        agent_name: my_agent    # Name of the registered agent
```

**File location**: `agents/agents_mapping.py`

---

## 3. Data Classes

Data classes define the core data structures used during agent execution.

### 3.1 `Step` - Step Data

**Description**

Records information for a single agent execution step, including the conversation context, action, observation, and reward.

**Class definition**

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

| Parameter | Type | Description |
| ----------- | ------ | ------------- |
| chat_completions | list[dict[str, str]] | Complete conversation context for the entire inference process, including historical turns. Used to construct the model input. |
| thought | str | Content inside the `<think>` tag in the model response, representing the model's internal reasoning in the current step. |
| action | Any | Content inside the `<tool call>` tag in the model response, representing the action chosen by the model, such as a tool call. |
| observation | Any | External observation received in the current step. For step `0`, this is the user's original question. For subsequent steps, this is the result of the previous action, such as a tool response. |
| model_response | str | Complete response generated by the LLM, that is, the `content` of the message with `'role': 'assistant'`. |
| info | dict | Additional information dictionary. Default value: empty. It can be used to record metadata such as tool IDs and time spent. |
| reward | float | Immediate reward obtained in the current step. Default value: `0.0`, reflecting the quality of the current action. |
| done | bool | Indicates whether the trajectory terminates in the current step. Default value: `False`, indicating that the task is incomplete. |
| mc_return | float | Monte Carlo return from the current step onward. Default value: `0.0`. It is used for policy gradient training. |
| step_id | int | Step ID. |

**File location**: `aura/runner/agent_engine_wrapper/base/agent/base_agent.py`

---

### 3.2 `Trajectory` - Trajectory Data

**Description**

Records the complete trajectory of an agent run, including all steps and the overall reward.

**Class definition**

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

| Parameter | Type | Description |
| ----------- | ------ | ------------- |
| task | Any | Original task input. |
| steps | list[Step] | List of all steps. |
| reward | float | Total trajectory reward. |
| toolcall_reward | float | Tool call reward. |
| res_reward | float | Final result reward. |
| termination_reason | str | Reason for trajectory termination. |

**File location**: `aura/runner/agent_engine_wrapper/base/agent/base_agent.py`

---

### 3.3 `Action` - Action Data

**Description**

Records the action chosen by the agent.

**Class definition**

```python
@dataclass
class Action:
    action: Any = None
```

**File location**: `aura/runner/agent_engine_wrapper/base/agent/base_agent.py`

---

### 3.4 `AgentTask` - Task Data

**Description**

Defines the data structure for agent tasks.

**Class definition**

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

| Parameter | Type | Description |
| ----------- | ------ | ------------- |
| task_id | str | Unique task ID. |
| sample_id | int | Sample ID. |
| iteration | int | Number of iterations. |
| agent_name | str | Agent name. |
| problem | str | Problem description. |
| ground_truth | str | Correct answer. |
| content | str | Additional content. |
| extra_args | dict | Additional arguments. |

**File location**: `aura/runner/agent_engine_wrapper/base_engine_wrapper.py`

---

## 4. Configuration Files

### Service Startup

**Syntax**

```bash
bash scripts/start_rl_with_verl_vllm.sh
```

The program is started by the `start_rl_with_verl_vllm.sh` script. The actual Python entry point is `aura/start.py`. The script executes the following command internally:

```bash
python aura/start.py --config-name=${CONFIG_NAME} 2>&1 | tee ${LOG_PATH}/train_unit_${timestamp}.log
```

> **Note:** Before running `start.py`, run the relevant scripts to prepare prerequisite components, such as initializing the environment and starting dependent services. Start the training task only after the runtime environment is ready.

### `hosts.conf` Settings

The service startup requires the `hosts.conf` file, which is located in the `aura/configs` directory. The file is used to configure single-node or two-node deployment. Single-node deployment uses the hybrid mode, while two-node deployment uses the One-Step-Off mode. For detailed examples, see [Modifying hosts.conf](./03_quick_start.md#modifying-hostsconf).

**Parameters**

| Parameter | Description |
| ----------- | ------------- |
| host | Node IP address. |
| index | Index of the current node. The index starts from `0` and is used to distinguish different nodes. |
| train_master_index | Index of the master node for the training task. When the value is `1`, the node starts the training task. |
| infer_master_index | Index of the master node for the inference task. Default value: `0`. When the value is `1`, the node starts the inference task. |

1. For single-node deployment in Hybrid mode, configure a single node and set both `train_master_index` and `infer_master_index` to `1`.

2. For two-node deployment in One-Step-Off mode, configure two nodes and set `train_master_index` to `0` and `1`, respectively. The node with `train_master_index` set to `0` is the inference node, and the node with `train_master_index` set to `1` is the training node.

### `base.conf` Settings

The service startup requires the `base.conf` file, which is located in the `aura/configs` directory. The file is used to configure the working mode and startup configuration files. For details, see [Modifying `base.conf`](./03_quick_start.md#modifying-baseconf).

**Parameters**

| Parameter | Description |
| ----------- | ------------- |
| work_mode | Working mode. Set this parameter to `hybrid` (hybrid mode) or `one_step_off` (One-Step-Off mode). The setting must be consistent with that in `hosts.conf`. |
| train_config_name | Name of the training YAML configuration file. |
| infer_config_name | Name of the inference YAML configuration file. This parameter does not take effect in hybrid mode. |
| monitor_cmd | Startup script name. Specifies the startup script to monitor and is used to distinguish between training backends (`verl` and `msrl`). |
| max_retries | Number of retries for resumed training. Default value: `100`. |
| clean_old_ckpt | Specifies whether to clear the `ckpt` directory on the first startup. `0` means do not clear it, and `1` means clear it. |

---

### Main Configuration File Parameters

The main training configuration file uses the YAML format and mainly contains the following sections:

1. **`agentic_ai`**: Global configuration, including the running mode and log level.
2. **`serve_conf`**: Service deployment configuration (`serve` mode).
3. **`direct_conf`**: Direct connection mode configuration (`direct` mode).
4. **`verl_conf`**: Training configuration parameters (`verl` backend).
5. **`train_instances`**: Training task instance configuration.
6. **`agent_instances`**: Agent service instance configuration.
7. **`infer_instances`**: Inference service instance configuration.

> In One-Step-Off mode, the inference service is deployed based on a separate inference configuration file (`vllm_infer_*.yaml`).
For details, see [Inference Service Configuration Parameters](#inference-service-configuration-parameters). The `infer_instances` section in the main configuration file is used only for service discovery and parameter references.

#### `agentic_ai` Configuration

| Parameter | Type | Description | Constraint |
| ----------- | ------ | ------------- | ------------ |
| mode | str | Running mode. | Valid values: `"serve"` (service deployment) or `"direct"` (direct connection mode). |
| log_level | str | Log level. | Valid values include `"DEBUG"`, `"INFO"`, `"WARNING"` and `"ERROR"`. |
| log_dir | str | Log directory path. | The path must exist and be writable. |

#### `serve_conf` Configuration (SERVE Mode)

| Parameter | Type | Description | Constraint |
|-----------|------|-------------|------------|
| host | str | Service listening address. | Default value: `"0.0.0.0"`. |
| port | int | Service listening port. | The value must be a valid port number. Default value: `8030`. |

#### `direct_conf` Configuration (DIRECT Mode)

| Parameter | Type | Description | Constraint |
|-----------|------|-------------|------------|
| entrypoints | list | List of task entry points. | Each entry contains fields such as `job_type`, `job_name`, and `job_kwargs`. |

---

### Inference Service Configuration Parameters

In One-Step-Off mode, the inference service is deployed based on a separate inference configuration file (`vllm_infer_*.yaml`).
For an example configuration file, see the [configs/infer](../../../aura/configs/infer) directory. Modify the following parameters according to the actual environment before use.

| Parameter | Description |
| -------------------- | ------------- |
| `infer_model_path` | Path to the inference model weights. |

#### Basic Configuration

| Parameter | Type | Description | Constraint |
| ----------- | ------ | ------------- | ------------ |
| vllm_version | str | vLLM version. | The value must be a valid vLLM version string. |
| infer_model_name | str | Inference model name. | The value must correspond to `infer_instances.executor_kwargs.engine_kwargs.model_name` in the main configuration file. |
| infer_model_path | str | Path to the inference model weights. | The path must exist and contain all required model files. |
| enable_expert_parallel | bool | Specifies whether to enable expert parallelism. | Set this parameter to `true` for MoE models and `false` for dense models. Default value: `false`. |

#### Deployment Mode Configuration

| Parameter | Type | Description | Constraint |
| ----------- | ------ | ------------- | ------------ |
| pd_mode | int | Specifies whether to enable prefill/decode (PD) disaggregation. | Valid values: `1` (enables PD disaggregation) or `0` (disables PD disaggregation). Default value: `0`. |
| prefill_instance_count | int | Number of prefill instances. | The value must be an integer greater than `0`. |
| decode_instance_count | int | Number of decode instances. | The value must be a non-negative integer. Set this parameter to `0` when PD disaggregation is disabled. |
| tensor_parallel_size | int | Tensor parallel size for prefill/decode. | The value must be an integer greater than `0`. |
| data_parallel_size | int | Data parallel size for prefill/decode. | The value must be an integer greater than `0`. |

#### Inference Performance Configuration

| Parameter | Type | Description | Constraint |
| ----------- | ------ | ------------- | ------------ |
| max_model_len | int | Maximum context length of the model. | The value must be an integer greater than `0`. |
| max_num_batched_tokens | int | Maximum number of tokens processed in a single batch. | The value must be an integer greater than `0`. |
| gpu_memory_utilization | float | Maximum proportion of GPU memory that can be used. | The value must be a floating-point number ranging from `0` to `1`. Default value: `0.6`. |
| max_num_seqs | int | Maximum number of concurrent sequences. | The value must be an integer greater than `0`. |
| cudagraph_capture_sizes | list | List of capture sizes for graph mode. | The value must be a list of positive integers. |

#### Advanced Configuration

| Parameter | Type | Description | Constraint |
| ----------- | ------ | ------------- | ------------ |
| kv_backend | str | Backend for synchronizing and transferring KV cache between PD nodes. | Valid values: `"mooncake"` or `"llmdatadist"`. |
| enable_vllm_stat | bool | Specifies whether to enable vLLM statistics. | Default value: `false`. |
| enable_tensor_similarity_check | bool | Specifies whether to enable weight similarity checking. | Default value: `false`. |
| vllm_ascend_enable_flashcomm | int | Specifies whether to enable the FlashComm algorithm. | Valid values: `0` (disabled) or `1` (enabled). |
| vllm_ascend_enable_flashcomm1 | int | Specifies whether to enable the FlashComm1 algorithm. | Valid values: `0` (disabled) or `1` (enabled). |
| tool_call_enable | bool | Specifies whether to enable tool calls. | Default value: `false`. |
| use_vllm_opt | bool | Specifies whether to use the optimized vLLM. | Default value: `false`. |

---

### Training Configuration Parameters (`verl` Backend)

#### `verl_conf.extras` Configuration

| Parameter | Type | Description | Constraint |
|-----------|------|-------------|------------|
| agent_service | str | Name of the dependent agent service. | The value must correspond to the `name` in `agent_instances`. |
| infer_service | str | Name of the dependent inference service. | The value must correspond to the `name` in `infer_instances`. |

#### `verl_conf.algorithm` Configuration

| Parameter | Type | Description | Constraint |
|-----------|------|-------------|------------|
| adv_estimator | str | Advantage estimator. | Valid values: `"grpo"` or `"gae"`. Default value: `"grpo"`. |
| kl_ctrl.kl_coef | float | KL divergence coefficient. | The value must be a floating-point number greater than `0`. Default value: `0.001`. |

#### `verl_conf.data` Configuration

| Parameter | Type | Description | Constraint |
| ----------- | ------ | ------------- | ------------ |
| train_files | str | Path to the training data file. | The path must exist. The `.parquet` format is supported. |
| val_files | str | Path to the validation data file. | The path must exist. The `.parquet` format is supported. |
| train_batch_size | int | Training batch size. | The value must be an integer greater than `0`. Default value: `16`. |
| max_prompt_length | int | Maximum prompt length. | The value must be an integer greater than `0`. Default value: `2048`. |
| max_response_length | int | Maximum response length. | The value must be an integer greater than `0`. Default value: `2048`. |
| filter_overlong_prompts | bool | Specifies whether to filter samples that exceed `max_prompt_length`. | Default value: `True`. |
| truncation | str | Truncation method. | Default value: `"error"`. By default, `verl` supports only `"error"`. Other truncation methods (`left`, `right`, and `middle`) require a custom dataset class. |

#### `verl_conf.actor_rollout_ref.model` Configuration

| Parameter | Type | Description | Constraint |
| ----------- | ------ | ------------- | ------------ |
| path | str | Path to the model weights. | The path must exist and contain all required model files. |
| use_remove_padding | bool | Specifies whether to remove padding tokens during training and perform computation only on actual tokens. | Default value: `False`. |
| enable_gradient_checkpointing | bool | Specifies whether to enable gradient checkpointing to reduce training memory usage by recomputing intermediate activations during backpropagation. | Default value: `True`. |

> **Description of the `enable_gradient_checkpointing` Parameter**
>
> * **When disabled:**
>
>   ```text
>   forward → save all intermediate activations → use them directly during backward
>   ```
>
>   Training is faster but requires significantly more GPU memory.
>
> * **When enabled:**
>
>   ```text
>   forward → do not save intermediate activations
>   backward → recompute forward → calculate gradients
>   ```
>
>   Training is slower, but GPU memory usage is significantly reduced.

#### `verl_conf.actor_rollout_ref.actor` Configuration

| Parameter | Type | Description | Constraint |
| ----------- | ------ | ------------- | ------------ |
| strategy | str | Distributed training strategy. | Valid values: `"megatron"`, `"fsdp"`, `"fsdp2"`, or `"veomni"`. Default value: `"megatron"`. |
| optim.lr | float | Learning rate. | The value must be a floating-point number greater than `0`. Default value: `5e-7`. |
| entropy_coeff | float | Entropy regularization coefficient that controls the degree of exploration by the model. | The value must be a non-negative number. Default value: `0.001`. |
| ppo_mini_batch_size | int | Number of samples used for one gradient update during each PPO update. | The value must be an integer greater than `0`. Default value: `2`. |
| ppo_micro_batch_size_per_gpu | int | Number of samples actually processed by each GPU during one forward/backward pass. | The value must be an integer greater than `0`. Default value: `2`. |
| use_kl_loss | bool | Specifies whether to add KL loss to the loss function as a penalty term. | Default value: `True`. This parameter is mutually exclusive with `use_kl_in_reward`. |
| kl_loss_coef | float | KL loss coefficient. | The value must be a floating-point number ranging from `0` to `1`. Default value: `0.001`. |
| kl_loss_type | str | KL loss calculation method. | Valid values: `"kl"`, `"abs"`, `"mse"`, `"low_var_kl"`, or `"full"`. Default value: `"low_var_kl"`. |

#### `verl_conf.actor_rollout_ref.actor.megatron` Configuration

| Parameter | Type | Description | Constraint |
| ----------- | ------ | ------------- | ------------ |
| seed | int | Random seed. | The value must be a non-negative integer. Default value: `0`. |
| pipeline_model_parallel_size | int | Pipeline parallel size. | The value must be an integer greater than `0`. Default value: `1`. |
| tensor_model_parallel_size | int | Tensor parallel size. | The value must be an integer greater than `0`. Default value: `4`. |
| override_transformer_config.use_flash_attn | bool | Specifies whether to use Flash Attention, an efficient attention implementation that reduces memory access and fuses kernels. | Default value: `True`. |

#### `verl_conf.actor_rollout_ref.rollout` Configuration

| Parameter | Type | Description | Constraint |
| ----------- | ------ | ------------- | ------------ |
| prompt_length | int | Prompt length. | The value must be an integer greater than `0`. Default value: `2048`. |
| response_length | int | Response length. | The value must be an integer greater than `0`. Default value: `2048`. |
| agent.agent_loop_manager_class | str | Agent loop manager class path. | The value must be a valid Python class path. |
| log_prob_micro_batch_size_per_gpu | int | Number of micro-batches processed by each GPU. | The value must be an integer greater than `0`. Default value: `2`. |
| enable_chunked_prefill | bool | Specifies whether to split the prompt prefill stage into multiple chunks. When enabled, this can effectively reduce KV cache memory usage for long prompts. | Default value: `False`. |
| tensor_model_parallel_size | int | Tensor parallel size. | The value must be an integer greater than `0`. Default value: `4`. |
| name | str | Inference engine name. | Default value: `"vllm"`. |
| gpu_memory_utilization | float | Maximum proportion of GPU memory that the rollout inference engine, such as vLLM, can use. | The value must be a floating-point number ranging from `0` to `1`. Default value: `0.6`. |
| n | int | Number of candidate responses generated for each prompt. | The value must be an integer greater than `0`. Default value: `2`. |
| do_sample | bool | Specifies whether to perform sampling. | Default value: `False`. |

#### `verl_conf.actor_rollout_ref.ref` Configuration

| Parameter | Type | Description | Constraint |
| ----------- | ------ | ------------- | ------------ |
| log_prob_micro_batch_size_per_gpu | int | Number of micro batches processed by each GPU. | The value must be an integer greater than `0`. Default value: `2`. |
| megatron.pipeline_model_parallel_size | int | Pipeline parallel size. | The value must be an integer greater than `0`. Default value: `2`. |
| megatron.tensor_model_parallel_size | int | Tensor parallel size. | The value must be an integer greater than `0`. Default value: `2`. |

#### `verl_conf.trainer` Configuration

| Parameter | Type | Description | Constraint |
| ----------- | ------ | ------------- | ------------ |
| val_before_train | bool | Specifies whether to perform validation before training to measure the initial model performance and determine whether training is effective. | Default value: `False`. |
| device | str | Training device. | Valid values: `"npu"`, `"gpu"`, or `"cpu"`. Default value: `"npu"`. |
| critic_warmup | int | Switch that delays Actor updates. Actor model updates are performed only when the current global step is greater than or equal to `critic_warmup`. | The value must be a non-negative integer. Default value: `0`. |
| project_name | str | Project name. | Used to identify the training project. |
| experiment_name | str | Experiment name. | Used to identify the training experiment. |
| n_gpus_per_node | int | Number of GPUs/NPUs per node. | The value must be an integer greater than `0`. Default value: `8`. |
| nnodes | int | Number of nodes. | The value must be an integer greater than `0`. Default value: `1`. |
| save_freq | int | Number of steps between checkpoint saves. | The value must be `-1` or an integer greater than `0`. `-1` means that checkpoints are not saved. Default value: `-1`. |
| test_freq | int | Number of steps between evaluations. | The value must be `-1` or an integer greater than `0`. `-1` means that evaluation is not performed. Default value: `-1`. |
| total_epochs | int | Number of complete passes through the entire training dataset. | The value must be an integer greater than `0`. Default value: `1`. |
| logger | list | Loggers. | The list can contain `"console"`, `"tensorboard"`, and other supported loggers. |

---

### Training Instance Configuration

#### `train_instances` Configuration

| Parameter | Type | Description | Constraint |
| ----------- | ------ | ------------- | ------------ |
| name | str | Training service name. | The value must be unique. Meaningful names are recommended. |
| executor_num | int | Number of executors. | The value must be an integer greater than `0`. Default value: `1`. |
| executor_kwargs.cluster_mode | str | Cluster mode. | Valid values: `"hybrid"` (hybrid mode) or `"one_step_off"` (One-Step-Off mode). |
| executor_kwargs.train_engine | str | Training engine. | The value can be `"mindspeed_rl"`, `"verl"`, or another supported engine. Default value: `"verl"`. |
| executor_kwargs.train_config | dict | Training configuration. | The value should reference the `verl_conf` configuration. |
| executor_kwargs.rollout_config | dict | Rollout configuration. | Default value: an empty dictionary. |
| executor_kwargs.agent_service | str | Agent service name. | The value should reference the `name` in `agent_instances`. |
| executor_kwargs.infer_service | str | Inference service name. | The value should reference the `name` in `infer_instances`. |
| resource_info | list | Resource description for the entire training service. | Default value: an empty list. |

---

### Agent Instance Configuration

#### `agent_instances` Configuration

| Parameter | Type | Description | Constraint |
| ----------- | ------ | ------------- | ------------ |
| name | str | Agent service name. | The value must be unique. Meaningful names are recommended. |
| executor_num | int | Number of executors. | The value must be an integer greater than `0`. Default value: `1`. |
| executor_kwargs.agent_engine | str | Agent engine type. | The value can be `"rllm"` or another supported agent engine. |
| executor_kwargs.agent_engine_kwargs.agent_name | str | Agent name. | The value must be a valid built-in agent name, such as `"math"`. |
| executor_kwargs.agent_engine_kwargs.simplify_think_content | bool | Specifies whether to simplify thinking content. | Default value: `false`. |
| executor_kwargs.agent_engine_kwargs.max_steps | int | Maximum number of steps. | The value must be an integer greater than `0`. Default value: `5`. |
| executor_kwargs.agent_engine_kwargs.max_prompt_length | int | Maximum length of the agent input prompt. | The value must be an integer greater than `0`. Default value: `2048`. |
| executor_kwargs.agent_engine_kwargs.max_model_len | int | Maximum model context length. | The value must be an integer greater than `0`. Default value: `4096`. |
| executor_kwargs.agent_engine_kwargs.n_parallel_agents | int | Number of agents executed in parallel. | The value must be an integer greater than `0`. Default value: `1024`. |
| executor_kwargs.agent_engine_kwargs.tokenizer | str | Tokenizer path. | The value must be a valid tokenizer path. |
| executor_kwargs.infer_service_params | dict | Inference service parameters. | The dictionary contains `top_p`, `temperature`, `max_tokens`, `model`, and other parameters. |
| executor_kwargs.trajectory_save_dir | str | Trajectory save directory. | The value must be a valid file path. |
| resource_info | list | Resource description for the entire agent service. | Default value: an empty list. |

---

### Inference Instance Configuration

#### `infer_instances` Configuration

| Parameter | Type | Description | Constraint |
| ----------- | ------ | ------------- | ------------ |
| name | str | Inference service name. | The value must be unique. Using the model name is recommended. |
| executor_num | int | Number of executors. | The value must be an integer greater than `0`. Default value: `1`. |
| executor_kwargs.engine | str | Inference engine type. | Default value: `"vllm_proxy"`, which indicates that the vLLM server is accessed through HTTP. |
| executor_kwargs.engine_kwargs.chat_server | str | vLLM HTTP inference service address. | The value must be a valid HTTP/HTTPS address. |
| executor_kwargs.engine_kwargs.prefill_server_list | list | List of servers dedicated to prefill processing. | The value can be an empty list. |
| executor_kwargs.engine_kwargs.decode_server_list | list | List of servers dedicated to decode processing. | The value can be an empty list. |
| executor_kwargs.engine_kwargs.model_name | str | Model name loaded by the inference service. | The value must be a valid model name. |
| resource_info | list | Resource description for the entire inference service. | Default value: an empty list. |

---

### Configuration File Example

The following is a complete configuration file example using the `verl` backend in hybrid mode. For other configuration files, see the [configs](../../../aura/configs) directory.
Modify the following parameters according to the actual environment before use.

| Parameter | Description |
| --------- | ----------- |
| `hydra.searchpath` | Hydra configuration search path. |
| `train_files` / `val_files` | Paths to the training and validation datasets. |
| `model.path` | Path to the model weights. |
| `tokenizer` | Tokenizer path. |
| `trajectory_save_dir` | Path for saving trajectory data. |

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

# Training instance
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

# Agent instance
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

# Inference instance
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
