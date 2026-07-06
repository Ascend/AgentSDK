# Appendix

## Public Network Addresses Included in the Software

The websites in the Agent SDK installation package will be cleared and will not be accessed after installation, posing no security risks.

The SDK itself does not access the public URLs and email addresses in this manual. Therefore, they do not pose any risk.

For more public network addresses, see [AgentSDK Public Network Addresses.xlsx](resource/AgentSDK_public_network_addresses_0000002516443057.xlsx).

## Environment Variable Usage

The Agent SDK may use the following environment variables during runtime.

|Environment Variable|Description|
|--|--|
|LOCAL_RANK|A `torch` distributed training setting that describes the rank of the current thread on the current node. The value range is [0, 8).|
|RANK|A `torch` distributed training setting that describes the rank of the current thread across all nodes. The value range is [0, 8).|

When Agent SDK starts, it validates environment variables against a whitelist. Only the following environment variables are retained.

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
|ATB_DEVICE_TILING_BUFFER_BLOCK_NUM|Number of `DeviceTilingBuffer` blocks in the context. The number is related to the maximum number of parallel operators. Generally, the default value is used should not be changed unless necessary.|
|ATB_HOME_PATH|Path for storing files after the nnal package is installed.|
|ATB_HOST_TILING_BUFFER_BLOCK_NUM|Number of `HostTilingBuffer` blocks in the context. The number is related to the maximum number of parallel operators. Generally, the default value is used should not be changed unless necessary.|
|ATB_MATMUL_SHUFFLE_K_ENABLE|Specifies whether to enable Shuffle-K to control the consistency of the accumulation order of the matrix multiplication result when different positions of the matrix are computed. This variable affects the internal accumulation order of the matmul operator.|
|ATB_OPSRUNNER_KERNEL_CACHE_GLOABL_COUNT|Number of slots for the global kernel cache. <li>When the number of slots increases, the cache hit rate is improved but the retrieval efficiency is reduced. </li><li>When the number of slots decreases, the retrieval efficiency is improved but the cache hit rate is reduced.</li>|
|ATB_OPSRUNNER_KERNEL_CACHE_LOCAL_COUNT|Number of slots for the local kernel cache. <li>When the number of slots increases, the cache hit rate is improved but the retrieval efficiency is reduced. </li><li>When the number of slots decreases, the retrieval efficiency is improved but the cache hit rate is reduced.</li>|
|ATB_OPSRUNNER_SETUP_CACHE_ENABLE|Specifies whether to enable the SetupCache function of the ATB. When detecting that the input and output tensors of the operation do not change, this function skips most of the processes of the setup to improve the performance on the scheduling side. This function is enabled by default for performance acceleration.|
|ATB_STREAM_SYNC_EVERY_KERNEL_ENABLE|Used to locate the kernel where the error is reported. When the variable is set to `1`, stream synchronization is performed when the execution of each kernel ends.|
|ATB_STREAM_SYNC_EVERY_OPERATION_ENABLE|Used to locate the operation where the error is reported. When the variable is set to `1`, synchronization is performed during the execution of each operation.|
|ATB_STREAM_SYNC_EVERY_RUNNER_ENABLE|Used to locate the runner where the error is reported. When the variable is set to `1`, stream synchronization is performed during the execution of each runner.|
|ATB_SHARE_MEMORY_NAME_SUFFIX|Name suffix of the shared memory. When the communication operator is used by multiple users, this variable needs to be set to distinguish the shared memory.|
|ATB_WORKSPACE_MEM_ALLOC_ALG_TYPE|Workspace memory allocation algorithms. The ATB selects different algorithms to compute the workspace size and allocation based on the environment variable configuration. You can select different algorithms to test the workspace allocation.|
|ATB_WORKSPACE_MEM_ALLOC_GLOBAL|Specifies whether to use the global intermediate tensor memory allocation algorithm. After this algorithm is used, the size of the intermediate tensor memory is computed and allocated.|
|HOME|Home directory path of the current user|
|LCCL_DETERMINISTIC|Specifies whether to enable deterministic LCCL AllReduce (with order-preserving addition). Note that this function takes effect when `rankSize` ≤ 8. When enabled, it affects <li>Performance of some communication operators </li><li>Accumulation order of the `lccl` communication operators|
|LD_LIBRARY_PATH|Dynamic library search path (for Linux only).</li>|
|PATH|Executable file search path.|
|PYTHONPATH|Python module search path.|
|TOOLCHAIN_HOME|Installation path of the Toolkit toolchain.|

> [!NOTE]NOTE
>
>- Agent SDK runtime uses open-source software. For the environment variables used by the relevant open-source software, refer to the corresponding software documentation.
>- Agent SDK depends on CANN. During the operation of CANN, folders such as `kernel_meta` will be generated. Agent SDK does not have the function of dumping or deleting these files. You can use environment variables to manage files in a unified manner. For details, see **Installation and Configuration** > **Configuring Flushing Files** > **ASCEND\_WORK\_PATH** in the *CANN Environment Variable Reference*.
