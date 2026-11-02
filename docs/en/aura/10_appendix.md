# Appendix

## Environment Variable Usage

Aura may use the following environment variables during runtime.

|Environment Variable|Description|
|--|--|
|LOCAL_RANK|A `torch` distributed training setting that describes the rank of the current thread on the current node. The value range is [0, 8).|
|RANK|A `torch` distributed training setting that describes the rank of the current thread across all nodes. The value range is [0, 8).|

When Aura starts, it validates environment variables against an allowlist. Only the following environment variables are retained.

|Environment Variable|Description|
|--|--|
|ASCEND_WORK_PATH|Location for files generated during unified CANN runtime.|
|ASCEND_AICPU_PATH|Installation path of AICPU in `ascend-toolkit`.|
|ASCEND_HOME_PATH|Same as `ASCEND_TOOLKIT_HOME`. This is the file storage path after you install the CANN Toolkit.|
|ASCEND_OPP_PATH|Root directory of the operator library.|
|ASCEND_TOOLKIT_HOME|Path for storing files after the CANN Toolkit software package is installed.|
|ASDOPS_LOG_LEVEL|Operator library log level.|
|ASDOPS_LOG_PATH|Path for storing operator library logs.|
|ASDOPS_LOG_TO_BOOST_TYPE|Name of the acceleration library log directory.|
|ASDOPS_LOG_TO_FILE|Specifies whether to output operator library logs to a file.|
|ASDOPS_LOG_TO_FILE_FLUSH|Specifies whether to flush the log file after writes.|
|ASDOPS_LOG_TO_STDOUT|Specifies whether to output operator library logs to the console.|
|ATB_COMPARE_TILING_EVERY_KERNEL|After each kernel runs, the system compares the tiling content on the NPU before and after the running to check whether tiling memory corruption occurs.|
|ATB_DEVICE_TILING_BUFFER_BLOCK_NUM|Number of `DeviceTilingBuffer` blocks in the context. The number is related to the maximum number of parallel operators. Generally, the default value is used and should not be changed unless necessary.|
|ATB_HOME_PATH|Path for storing files after the nnal package is installed.|
|ATB_HOST_TILING_BUFFER_BLOCK_NUM|Number of `HostTilingBuffer` blocks in the context. The number is related to the maximum number of parallel operators. Generally, the default value is used and should not be changed unless necessary.|
|ATB_MATMUL_SHUFFLE_K_ENABLE|Specifies whether to enable Shuffle-K to control the consistency of the accumulation order of the matrix multiplication result when different positions of the matrix are computed. This variable affects the internal accumulation order of the matmul operator.|
|ATB_OPSRUNNER_KERNEL_CACHE_GLOBAL_COUNT|Number of slots for the global kernel cache. <li>When the number of slots increases, the cache hit rate is improved but the retrieval efficiency is reduced.</li><li>When the number of slots decreases, the retrieval efficiency is improved but the cache hit rate is reduced.</li>|
|ATB_OPSRUNNER_KERNEL_CACHE_LOCAL_COUNT|Number of slots for the local kernel cache. <li>When the number of slots increases, the cache hit rate is improved but the retrieval efficiency is reduced.</li><li>When the number of slots decreases, the retrieval efficiency is improved but the cache hit rate is reduced.</li>|
|ATB_OPSRUNNER_SETUP_CACHE_ENABLE|Specifies whether to enable the SetupCache function of the ATB. When detecting that the input and output tensors of the operation do not change, this function skips most of the processes of the setup to improve the performance on the scheduling side. This function is enabled by default for performance acceleration.|
|ATB_STREAM_SYNC_EVERY_KERNEL_ENABLE|Used to locate the kernel where the error is reported. When the variable is set to 1, stream synchronization is performed when the execution of each kernel ends.|
|ATB_STREAM_SYNC_EVERY_OPERATION_ENABLE|Used to locate the operation where the error is reported. When the variable is set to 1, synchronization is performed during the execution of each operation.|
|ATB_STREAM_SYNC_EVERY_RUNNER_ENABLE|Used to locate the runner where the error is reported. When the variable is set to 1, stream synchronization is performed during the execution of each runner.|
|ATB_SHARE_MEMORY_NAME_SUFFIX|Name suffix of the shared memory. When the communication operator is used by multiple users, this variable needs to be set to distinguish the shared memory.|
|ATB_WORKSPACE_MEM_ALLOC_ALG_TYPE|Workspace memory allocation algorithms. The ATB selects different algorithms to compute the workspace size and allocation based on the environment variable configuration. You can select different algorithms to test the workspace allocation.|
|ATB_WORKSPACE_MEM_ALLOC_GLOBAL|Specifies whether to use the global intermediate tensor memory allocation algorithm. After this algorithm is used, the size of the intermediate tensor memory is computed and allocated.|
|HOME|Home directory path of the current user|
|LCCL_DETERMINISTIC|Specifies whether to enable deterministic LCCL AllReduce (with order-preserving addition). Note that this function takes effect when `rankSize` ≤ 8. When enabled, it affects <li>Performance of some communication operators</li><li>Accumulation order of the `lccl` communication operators</li>|
|LD_LIBRARY_PATH|Dynamic library search path (for Linux only).|
|PATH|Executable file search path.|
|PYTHONPATH|Python module search path.|
|TOOLCHAIN_HOME|Installation path of the Toolkit toolchain.|

> [!NOTE]
>
>- Aura runtime uses open-source software. For the environment variables used by the relevant open-source software, refer to the corresponding software documentation.
>- Aura depends on CANN. During the operation of CANN, folders such as `kernel_meta` are generated. Aura does not have the function of dumping or deleting these files. You can use environment variables to manage files in a unified manner. For details, see **Installation and Configuration** > `ASCEND_WORK_PATH` in the [CANN Environment Variable Reference](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/900beta2/maintenref/envvar/envref_07_0007.html).

## Supported Inference Backends

Aura supports the following inference backends:

|Inference Backend|Description|
|--|--|
|vllm-ascend|An Ascend NPU-adapted version based on the vLLM framework that provides high-performance LLM inference capabilities.|

## Supported Training Backends

Aura supports the following training backends:

|Training Backend|Description|
|--|--|
|verl|A verifiable reinforcement learning training framework that supports reinforcement learning algorithms such as PPO and GRPO.|

## Supported Agent Backends

Aura supports the following Agent backends:

|Agent Backend|Description|
|--|--|
|rLLM|An LLM Agent framework based on reinforcement learning that supports tool calling and multi-turn conversations.|

## Supported Models

Aura supports the following models:

|Model Name|Parameter Scale|Description|
|--|--|--|
|Qwen3-4B|4B|A Qwen3-series dense model that uses the Transformer architecture and supports inference, coding, and multilingual capabilities.|
|Qwen3-8B|8B|A Qwen3-series dense model that uses the Transformer architecture and has strong inference, coding, and multilingual capabilities.|
|Qwen3-14B|14B|A Qwen3-series dense model that uses the Transformer architecture and has strong math, coding, and logical reasoning capabilities.|
|Qwen3-32B|32B|A general-purpose Qwen3-series LLM that uses the Transformer architecture and has strong complex reasoning, coding, and Agent capabilities.|
|Qwen3-30B-A3B|30B|A Qwen3-series MoE model that uses a sparse mixture-of-experts architecture. It activates about 3B parameters per token, balancing complex reasoning capability and computing efficiency.|
|Qwen3.6-27B|27B|A Qwen3.6-series dense multimodal model that uses a hybrid architecture of Gated DeltaNet and Gated Attention and has strong complex reasoning and long-context understanding capabilities.|
|Qwen3.6-35B-A3B|35B|A Qwen3.6-series MoE model that uses sparse mixture-of-experts and hybrid attention architectures. It activates about 3B parameters per token, balancing complex reasoning capability and computing efficiency.|
