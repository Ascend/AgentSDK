# CLI API<a name="ZH-CN_TOPIC_0000002469525620"></a>

## agentic\_rl<a name="ZH-CN_TOPIC_0000002469365676"></a>

**Parameters<a name="section973741317611"></a>**

|Parameter|Description|
|--|--|
|--config-path|Path of the configuration file. The file must be in the YAML format.|

**Configuration File Parameters<a name="section61851418129"></a>**

| Parameter                     | Type  | Description | Constraints |
|---------------------------|------|-------------------------------------|--------------------------------|
| tokenizer_name_or_path    | str  | In `mindspeed_rl`, this is the tokenizer path. In `verl`, this is the model path, and `verl` automatically infers the tokenizer from the model.|The path must exist. All folders in the path must have permissions set to `750`. Files must have permissions set to `640`. The owner of every file and folder must be the current user.|
| model_name                | str  | Supported model. This takes effect only when the backend is `mindspeed_rl`.|Supported values are `qwen3-8b` and `qwen2.5-7b`.|
| agent_name                | str  | Name of the agent.| The value must start with a letter and can contain letters, digits, and underscores (_).|
| agent_engine_wrapper_path | str  | Path to the file that inherits from the `BaseEngineWrapper` class.|The path must exist. All folders in the path must have permissions set to `750`. Files must have permissions set to `640`. The owner of every file and folder must be the current user.|
| train_backend             | str  | Training backend.| The value must be `verl` or `mindspeed_rl`.|
| use_stepwise_advantage    | bool | Specifies whether to enable `stepwise_advantage`.| The value can be `True` or `False` (default).|
| infer_tensor_parallel_size | int | Tensor parallel size for inference.| The default value is 4. The tensor parallel size must be a positive integer. Depending on the model and parameter count, it must be divisible by the number of devices used for inference. Typical values are 1, 2, 4, and 8. For details, see [Optimization and Tuning - vLLM Documentation](https://docs.vllm.ai/en/latest/configuration/optimization/#chunked-prefill).|
| infer_pipeline_parallel_size | int | Pipeline parallel size for inference.| The default value is 1. In general, this value must be greater than 0. In the current version, this setting is invalid when the backend is `verl`, and it is fixed to 1.|
| infer_expert_parallel_size | int | Expert parallel size for inference.| The default value is 1. This takes effect only when the model uses an MoE architecture. It must be a positive integer, and the expert parallel size must be divisible by the number of model experts. This value cannot be set in the current version and is fixed to 1.|
| max_num_seqs              | int | Maximum number of parallel sequences in a single inference batch.| The default value is 1024. It must be a positive integer. A value that is too large may cause insufficient GPU memory. For details, see [Optimization and Tuning - vLLM Documentation](https://docs.vllm.ai/en/latest/configuration/optimization/#chunked-prefill).|
| max_num_batched_tokens    | int | Maximum number of tokens allowed in a single inference batch.| The default value is 8192. It must be a positive integer. For details, see [Optimization and Tuning - vLLM Documentation](https://docs.vllm.ai/en/latest/configuration/optimization/#chunked-prefill).|
| max_model_len             | int | Maximum context length that the inference model can process.| The default value is 16384. It must be a positive integer. This parameter depends on the model, and different models support different context lengths. Setting a value that exceeds the model-specific context length may make inference results inaccurate.|
| gpu_memory_utilization    | float | Proportion of GPU memory used by the inference engine.| The default value is 0.85. It must be a float greater than 0 and less than or equal to 1.|
| max_tokens                | int | Maximum number of inference tokens.| The default value is 8192. It must be a positive integer.|
| dtype                     | str | Data type used by the inference model.| The default value is `bfloat16`. It must be either `bfloat16` or `float16`.|
| top_k                     | int | Top-k parameter for nucleus sampling, which selects the `k` tokens with the highest probability.| The default value is 20. For details, see [Sampling Parameters - vLLM](http://docs.vllm.ai/en/v0.5.5/dev/sampling_params.html).|
| top_p                     | float | Top-p parameter for nucleus sampling, which keeps sampling until cumulative probability exceeds `top_p`.| The default value is 1.0. It must be a float greater than 0 and less than or equal to 1. For details, see [Sampling Parameters - vLLM](http://docs.vllm.ai/en/v0.5.5/dev/sampling_params.html).|
| min_p                     | float | Min-p parameter for nucleus sampling, which is used to control the minimum probability for a token to appear.| The default value is 0.01. It must be a float greater than 0 and less than or equal to 1. For details, see [Sampling Parameters - vLLM](http://docs.vllm.ai/en/v0.5.5/dev/sampling_params.html).|
| temperature               | float | Nucleus sampling parameter used to control sampling randomness.| The default value is 0.6. It must be a float greater than 0.|
| enforce_eager             | bool | Indicates whether to enable eager mode.| The default value is `True`.|
| use_kl_in_reward          | bool | Indicates whether to enable KL divergence in the reward.| The default value is `False`.|
| clip_ratio                | float | ε in the clipped objective during policy updates.| The default value is 0.2. It must be a floating-point number between 0 and 1.|
| entropy_coeff             | float | Entropy regularization coefficient.| The default value is 0.2. It must be non-negative.|
| kl_penalty                | str | KL divergence function.| The default value is `kl`. It must be one of `kl`, `abs`, `mse`, `low_var_kl`, and `full`.|
| kl_coef                   | float | KL divergence coefficient.| The default value is 0.05. It must be a positive number.|
| gamma                     | float | Discount factor for advantage estimation.| The default value is 1.0. It must be a floating-point number between 0 and 1.|
| lam                       | float | Tradeoff coefficient between bias and variance for advantage estimation.| The default value is 1.0. It must be a floating-point number between 0 and 1.|
| kl_horizon                | int | Window size for the sliding window in KL.| The default value is 10000. It must be a positive integer.|
| kl_target                 | float | Target KL divergence value in the adaptive controller.| The default value is 0.1. It must be a positive floating-point number.|
| kl_ctrl_type              | str | KL controller type.| The default value is `fixed`. It must be `fixed` or `adaptive`.|
| lr                        | float | Learning rate.| The default value is 1.0e-6. It must be a positive floating-point number.|
| min_lr                    | float | Minimum learning rate during decay.| The default value is 0.0. It must be a non-negative floating-point number.|
| lr_warmup_fraction        | float | Proportion of training steps used for warmup.| The default value is 0.0. It must be a floating-point number between 0 and 1.|
| clip_grad                 | float | Gradient clipping.| It must be a positive floating-point number. Currently this takes effect only when the training backend is `mindspeed_rl`.|
| weight_decay              | float | L2 regularization coefficient for model weights.| The default value is 0. It must be a floating-point number between 0 and 1.|
| num_gpus_per_node         | int | Number of available NPUs per node.| The default value is 8. It must be a positive integer.|
| max_prompt_length         | int | Maximum prompt length.| The default value is 2048. It must be a positive integer. Currently, the maximum value is 128K.|
| rollout_n                 | int | Number of responses generated in each rollout stage.| The default value is 2. It must be a positive integer no greater than 64.|
| use_tensorboard           | bool | Indicates whether to enable TensorBoard.| The default value is `False`.|
| dataset_additional_keys   | List[str] | Fields used by the dataset.| This takes effect only when the backend is `mindspeed_rl`. Pass a list of strings.|
| max_steps                 | int | Maximum number of responses in one trajectory.| The default value is 5. It must be a positive integer.|
| test_only                 | bool | Indicates whether to run testing only, without training.| The default value is `False`.|
| test_before_train         | bool | Indicates whether to test the original model before training.| The default value is `False`.|
| use_stepwise_advantage    | bool | Indicates whether to enable stepwise advantage mode.| The default value is `False`.|

**Parameters related to MindSpeed-RL**

| Parameter| Type| Description| Constraints|
|-------------------|------------|----------------------------------|----------------------------------|
| data_path         | str | Data path.| The path must exist. All folders in the path must have permissions set to `750`. Files must have permissions set to `640`. The owner of every file and folder must be the current user. You must also specify the fields to use through `dataset_additional_keys`.|
| load_params_path  | str | Training model path. It must contain a complete Megatron-format model.| The path must exist. All folders in the path must have permissions set to `750`. Files must have permissions set to `640`. The owner of every file and folder must be the current user.|
| save_params_path  | str | Path for storing the training model file.| The path must exist. All folders in the path must have permissions set to `750`. Files must have permissions set to `640`. The owner of every file and folder must be the current user.|
| train_iters      | int | Number of training iterations.| The default value is 1. It must be a positive integer.|
| epochs            | int | Number of iterations required for each training task update.| The default value is 1. It must be a positive integer.|
| seq_len           | int | Sequence length.| The default value is 8192. It must be a positive integer.|
| global_batch_size | int | Global batch size.| The default value is 16. It must be a positive integer.|
| save_interval     | int | Saving interval.| The default value is 1000. It must be a positive integer.|
| mini_batch_size   | int | Batch size used for weight updates on a single device.| The default value is 16. It must be a positive integer.|
| micro_batch_size  | int | Batch size processed by one forward and backward pass on a single device.| The default value is 1. It must be a positive integer.|
| tensor_model_parallel_size  | int | Tensor parallel size for training.| The default value is 4. It must be a positive integer greater than 0. Depending on the model and parameter count, it must be divisible by the number of devices used for inference. Typical values are 1, 2, 4, and 8.|
| pipeline_model_parallel_size | int | Pipeline parallel size for training.| The default value is 1. It must be a positive integer.|
| adv_estimator     | str | Advantage estimator for training.| The default value is `group_norm`. It must be either `group_norm` or `gae`.|

**Parameters relaed to verl**

| Parameter| Type| Description| Constraints|
|-------------------|------|--------------------------------|--------------------------------|
| train_files       | str | Training data path.| The path must exist. All folders in the path must have permissions set to `750`. Files must have permissions set to `640`. The owner of every file and folder must be the current user. The path must also match the parquet dataset format required by `verl`.|
| val_files         | str | Validation dataset path.| The path must exist. All folders in the path must have permissions set to `750`. Files must have permissions set to `640`. The owner of every file and folder must be the current user. The path must also match the parquet dataset format required by `verl`.|
| total_epochs      | int | Number of iterations required for each training task update.| The default value is 2. It must be a positive integer.|
| total_training_steps    | optional[int] | Number of training iterations.| No default value is provided. The `verl` backend computes this value automatically based on the length of the input data. If provided, it must be a positive integer and overrides the `verl` default value.|
| save_freq         | int | Training saving frequency.| The default value is 1000. It must be a positive integer.|
| ppo_mini_batch_size | int | Global mini-batch size for PPO updates.| The default value is 16. It must be a positive integer.|
| ppo_max_token_len_per_gpu | int | Maximum number of tokens processed in one PPO round on an NPU.| The default value is 24000. It must be a positive integer greater than 0. `verl` recommends setting this to at least `n * (prompt + response)`.|
| ppo_epochs        | int | Number of PPO update rounds required to repeat the same batch of trajectories.| The default value is 1. It must be a positive integer.|
| project_name      | str | Project name.| The default value is `default-agent`.|
| experiment_name   | str | Experiment name.| The default value is `default-experiment`.|
| max_response_length | int | Maximum generation length.| The default value is 2048. It must be a positive integer.|
| train_batch_size  | int | Training batch size.| The default value is 8. It must be a positive integer.|
| val_batch_size    | int | Validation batch size.| The default value is 512. It must be a positive integer.|
| dataloader_num_workers | int | Number of workers used by the data loader.| The default value is 8. It must be a positive integer.|
| nnodes            | int | Number of machines in the training cluster.| The default value is 1. It must be a positive integer.|
| adv_estimator     | str | Advantage estimator.| The default value is `grpo`. It must be either `grpo` or `gae`.|
| warmup_style      | str | Warmup style.| The default value is `constant`. It must be either `constant` or `cosine`.|
| min_lr_ratio      | float | Minimum learning-rate ratio.| This parameter is valid only when the warmup style is `cosine`. The default value is 0.0. It must be a floating-point number between 0 and 1.|
| num_cycles        | float | Cosine cycle count.| This parameter is valid only when the warmup style is `cosine`. The default value is 0.5, which represents half a cycle. It must be a positive floating-point number.|
| ckpt_content      | list | Contents saved in the checkpoint.| The default value is `['model', 'optimizer', 'extra']`. It must be a list of non-overlapping items chosen from `model`, `optimizer`, `extra`, and `hf_model`.|
| policy_loss_model | str | Policy loss calculation mode.| The default value is `vanilla`. It must be one of `vanilla`, `clip-cov`, `kl-cov`, and `gpg`.|
| policy_loss_clip_cov_ratio | float | Token clipping ratio.| The default value is 0.0002. It must be a positive floating-point number. This parameter takes effect only when `policy_loss_model` is `clip-cov`.|
| policy_loss_clip_cov_lb | float | Lower bound for `clip-cov`.| The default value is 1.0. It must be a positive floating-point number. This parameter takes effect only when `policy_loss_model` is `clip-cov`.|
| policy_loss_clip_cov_ub | float | Upper bound for `clip-cov`.| The default value is 5.0. It must be a positive floating-point number. This parameter takes effect only when `policy_loss_model` is `clip-cov`. Note that the lower bound for `clip-cov` must be less than the upper bound.|
| policy_loss_kl_cov_ratio | float | Token ratio used for computing `kl_cov`.| The default value is 0.0002. It must be a positive floating-point number. This parameter takes effect only when `policy_loss_model` is `kl-cov`.|
| policy_loss_ppo_kl_coef | float | KL coefficient used for computing `kl_cov`.| The default value is 0.1. It must be a floating-point number between 0 and 1. This parameter takes effect only when `policy_loss_model` is `kl-cov`.|
| fsdp_param_offload | bool | Indicates whether to enable FSDP parameter offload.| The default value is `False`.|
| fsdp_optimizer_offload | bool | Indicates whether to enable FSDP optimizer offload.| The default value is `False`.|
| loss_agg_mode      | str | PPO loss aggregation mode.| The default value is `token-mean`. It must be one of `token-mean`, `seq-mean-token-sum`, and `seq-mean-token-mean`.|
| use_kl_loss        | bool | Indicates whether to use KL loss instead of a KL reward penalty.| The default value is `False`.|
| kl_loss_coeff      | float | KL loss coefficient.| The default value is 0.001. It must be a floating-point number between 0 and 1. It takes effect only when `use_kl_loss=True`.|
| kl_loss_type       | str | KL loss format.| The default value is `low_bar_kl`. It must be one of `kl`, `abs`, `mse`, `low_var_kl`, and `full`. It takes effect only when `use_kl_loss=True`.|
| grad_clip          | float | Actor gradient clipping value.| The default value is 1.0. It must be a positive floating-point number.|
| entropy_from_logits_with_chunking | bool | Indicates whether to compute entropy in chunks.| The default value is `False`.|
| balance_batch      | bool | Indicates whether to balance batch sizes across workers in distributed training.| The default value is `True`.|
| val_before_train   | bool | Indicates whether to run one validation pass before formal training.| The default value is `True`.|
| val_only           | bool | Indicates whether to run validation only without training.| The default value is `False`.|
| test_freq          | int | Validation frequency.| The default value is -1. It must be -1 or a positive integer.|
| truncation         | str | Truncation mode.| The default value is `error`. It must be one of `error`, `left`, `right`, and `middle`.|

**Resumable Training Description**

The CLI supports resumable training by default. During training, checkpoints are saved by default in `checkpoints/${project_name}/${experiment_name}` in the current directory. When training starts again, the system automatically checks whether a checkpoint file exists in that path. If one exists, it loads the checkpoint and resumes training.

To start training from scratch instead of resuming the previous training state, change the `project_name` or `experiment_name` parameter. Changing either parameter uses a new checkpoint directory, which prevents loading old checkpoint files.

> [!IMPORTANT]
> The checkpoint directory contains sensitive information such as model weights and optimizer states. For security configuration of the model save path, see [Security Hardening for the Model Saving Path](security_hardening.md#security-hardening-for-the-model-saving-path).

And

> [!NOTE]
> All paths mentioned in this section must meet the following requirements:
>
>- The path must exist.
>- Folders in the path must have permissions set to 750, and files must have permissions set to 640.
>- The path must not be a symbolic link.
>- The length of the path string must not exceed 1024.
>- The owner of the path must be the current user.

**MindSpeed-RL Configuration Reference**

Parameters in the configuration file parameter category must be placed at the first level of the configuration file.

Parameters in the MindSpeed-RL-related parameter category must be placed at the second level under `mindspeed_rl`.

The following is a sample configuration structure. This configuration file is a reference template and cannot be used directly. Add, remove, or adjust parameters as needed for your actual scenario:

```text
tokenizer_name_or_path: /path/to/model
model_name: qwen2.5-7b
agent_name: math
agent_engine_wrapper_path: /path/to/rllm_engine_wrapper.py
use_stepwise_advantage: false
train_backend: mindspeed_rl
max_model_len: 10240
gpu_memory_utilization: 0.7
infer_tensor_parallel_size: 4
max_tokens: 1024
top_k: 10
num_gpus_per_node: 4
max_prompt_length: 1024
max_num_seqs: 8
rollout_n: 1
lr: 0.000005
entropy_coeff: 0.01
use_tensorboard: true
dataset_additional_keys: ["problem", "answer"]
mindspeed_rl:
  data_path: /path/to/data
  load_params_path: /path/to/params
  save_params_path: /path/to/save_params
```

**Verl Configuration Reference**

Parameters in the configuration file parameter category must be placed at the first level of the configuration file.

Parameters in the Verl-related parameter category must be placed at the second level under `verl`.

The following is a sample configuration structure. This configuration file is a reference template and cannot be used directly. Add, remove, or adjust parameters as needed for your actual scenario:

```text
tokenizer_name_or_path: /path/to/model
model_name: qwen2.5-7b
agent_name: math
agent_engine_wrapper_path: /path/to/rllm_engine_wrapper.py
use_stepwise_advantage: false
train_backend: verl
max_model_len: 10240
gpu_memory_utilization: 0.7
infer_tensor_parallel_size: 4
max_tokens: 1024
top_k: 10
num_gpus_per_node: 4
max_prompt_length: 1024
max_num_seqs: 8
rollout_n: 1
lr: 0.000005
entropy_coeff: 0.01
use_tensorboard: true
verl:
  total_epochs: 1
  total_training_steps: 100
  save_freq: 100
  train_batch_size: 4
  val_batch_size: 4
  project_name: default-agent
  experiment_name: default-experiment
  train_files: /data/gsm8k/train.parquet
  val_files: /data/gsm8k/test.parquet
```
