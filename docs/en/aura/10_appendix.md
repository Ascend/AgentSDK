# Appendix

## Environment Variable Usage

Aura may use the following environment variables during runtime.

| Environment Variable | Description |
| -------------------- | ----------- |
| `LOCAL_RANK` | A `torch` distributed training setting that describes the rank of the current thread on the current node. The value range is [0, 8). |
| `RANK` | A `torch` distributed training setting that describes the rank of the current thread across all nodes. The value range is [0, 8). |

When Aura starts, it validates environment variables against a whitelist. Only the following environment variables are retained.

| Environment Variable | Description |
| -------------------- | ----------- |
| `ASCEND_WORK_PATH` | Location for files generated during the CANN runtime. |
| `ASCEND_AICPU_PATH` | Installation path of the AI CPU in `ascend-toolkit`. |
| `ASCEND_HOME_PATH` | Same as `ASCEND_TOOLKIT_HOME`. Represents the path for storing files after the CANN Toolkit software is installed. |
| `ASCEND_OPP_PATH` | Root directory of the operator library. |
| `ASCEND_TOOLKIT_HOME` | Path for storing files after the CANN Toolkit software package is installed. |
| `ASDOPS_LOG_LEVEL` | Operator library log level. |
| `ASDOPS_LOG_PATH` | Path for storing operator library logs. |
| `ASDOPS_LOG_TO_BOOST_TYPE` | Name of the acceleration library log directory. |
| `ASDOPS_LOG_TO_FILE` | Specifies whether to output operator library logs to a file. |
| `ASDOPS_LOG_TO_FILE_FLUSH` | Specifies whether to flush the log file after writing logs. |
| `ASDOPS_LOG_TO_STDOUT` | Specifies whether to output operator library logs to the console. |
| `ATB_COMPARE_TILING_EVERY_KERNEL` | After each kernel runs, compares the tiling content on the NPU before and after execution to check for tiling memory corruption. |
| `ATB_DEVICE_TILING_BUFFER_BLOCK_NUM` | Number of `DeviceTilingBuffer` blocks in the context. The number is related to the maximum number of parallel operators. The default value is generally used and is not recommended for modification. |
| `ATB_HOME_PATH` | Path for storing files after the nnal package is installed. |
| `ATB_HOST_TILING_BUFFER_BLOCK_NUM` | Number of `HostTilingBuffer` blocks in the context. The number is related to the maximum number of parallel operators. The default value is generally used and is not recommended for modification. |
| `ATB_MATMUL_SHUFFLE_K_ENABLE` | Specifies whether to enable Shuffle-K. This variable controls whether the accumulation order is consistent when computing the matrix multiplication result at different positions. It affects the internal accumulation order of the matmul operator. |
| `ATB_OPSRUNNER_KERNEL_CACHE_GLOBAL_COUNT` | Number of slots in the global kernel cache. <br>- Increasing the number of slots improves the cache hit rate but reduces retrieval efficiency.<br>- Decreasing the number of slots improves retrieval efficiency but reduces the cache hit rate. |
| `ATB_OPSRUNNER_KERNEL_CACHE_LOCAL_COUNT` | Number of slots in the local kernel cache. <br>- Increasing the number of slots improves the cache hit rate but reduces retrieval efficiency.<br>- Decreasing the number of slots improves retrieval efficiency but reduces the cache hit rate. |
| `ATB_OPSRUNNER_SETUP_CACHE_ENABLE` | Specifies whether to enable the SetupCache function of ATB. When the input and output tensors of an operation are detected to be unchanged, this function skips most of the setup process to improve scheduling-side performance. It is enabled by default for performance acceleration. |
| `ATB_STREAM_SYNC_EVERY_KERNEL_ENABLE` | Used to locate the kernel where an error is reported. When the variable is set to `1`, stream synchronization is performed when the execution of each kernel ends. |
| `ATB_STREAM_SYNC_EVERY_OPERATION_ENABLE` | Used to locate the operation where an error is reported. When the variable is set to `1`, synchronization is performed during the execution of each operation. |
| `ATB_STREAM_SYNC_EVERY_RUNNER_ENABLE` | Used to locate the runner where an error is reported. When the variable is set to `1`, stream synchronization is performed during the execution of each runner. |
| `ATB_SHARE_MEMORY_NAME_SUFFIX` | Suffix of the shared memory name. When a communication operator is used by multiple users, this variable must be set to distinguish the shared memory. |
| `ATB_WORKSPACE_MEM_ALLOC_ALG_TYPE` | Selects the workspace memory allocation algorithm. ATB selects different algorithms to calculate the workspace size and perform workspace allocation based on the environment variable configuration. You can select different algorithms to test workspace allocation. |
| `ATB_WORKSPACE_MEM_ALLOC_GLOBAL` | Specifies whether to use the global intermediate tensor memory allocation algorithm. When enabled, the size of intermediate tensor memory is calculated and allocated. |
| `HOME` | Home directory path of the current user. |
| `LCCL_DETERMINISTIC` | Specifies whether to enable deterministic LCCL AllReduce (with order-preserving addition). Note that this function takes effect when `rankSize` ≤ 8. When enabled, it affects the following:<br>- Performance of some communication operators.<br>- Accumulation order of `lccl` communication operators. |
| `LD_LIBRARY_PATH` | Dynamic library search path (Linux only). |
| `PATH` | Executable file search path. |
| `PYTHONPATH` | Python module search path. |
| `TOOLCHAIN_HOME` | Installation path of the Toolkit toolchain. |

> [!NOTE]
>
>- Aura uses open-source software at runtime. For the environment variables used by the relevant open-source software, see the corresponding software documentation.
>- Aura depends on CANN. When CANN is running, folders such as `kernel_meta` are generated. Aura does not provide functionality for dumping or deleting these files. You can use an environment variable to manage these files in a unified manner. For details, see [ASCEND_WORK_PATH](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/latest/maintenref/envvar/envref_07_0007.html).

## Supported Inference Backends

Aura supports the following inference backends.

| Inference Backend | Description |
| ----------------- | ----------- |
| vllm-ascend | An Ascend NPU adaptation of vLLM that provides high-performance foundation model inference. |

## Supported Training Backends

Aura supports the following training backends.

| Training Backend | Description |
| ---------------- | ----------- |
| verl | A reinforcement learning training framework that supports reinforcement learning algorithms such as PPO and GRPO. |

## Supported Agent Backends

Aura supports the following Agent backends.

| Agent Backend | Description |
| ------------- | ----------- |
| rLLM | A reinforcement learning-based large language model Agent framework that supports tool calling and multi-turn conversations. |

## Supported Models

Aura supports the following models.

| Model | Parameter Size | Description |
| ----- | -------------- | ----------- |
| Qwen3-4B | 4B | A lightweight model in the Qwen3 series, suitable for fast training and inference in resource-constrained scenarios. |
| Qwen3-8B | 8B | A medium-sized model in the Qwen3 series that balances performance and resource consumption. |
| Qwen3-14B | 14B | A medium-to-large model in the Qwen3 series with strong complex reasoning and mathematical reasoning capabilities. |
| Qwen3-32B | 32B | A general-purpose large language model in the Qwen3 series with strong language understanding and generation capabilities. |
| Qwen3-30B-A3B | 30B | A MoE model in the Qwen3 series that balances large language model performance and inference efficiency. |
